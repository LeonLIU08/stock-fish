"""股票 × run：不复权成交、复权信号、一个账户一份报告。

各账户独立。summary 把同一只股票的 run 并列，不把金额加在一起。
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

from analysis.backtest.analytics import (
    daily_frame,
    data_gaps,
    extreme_days,
    factor_exposures,
    monthly_return_table,
    nav_from_capital,
    nav_from_price,
    rolling_stats,
    split_cost_table,
    to_hk_index,
)
from analysis.backtest.bars import coverage_note, slice_eval_window
from analysis.backtest.charts import (
    plot_daily_kline_with_trades,
    plot_factor_exposure,
    plot_monthly_heatmap,
    plot_nav_and_underwater,
    plot_position_turnover_leverage,
    plot_rolling,
)
from analysis.backtest.config import BacktestConfig, CostModel
from analysis.backtest.metrics import compute_metrics
from analysis.backtest.report import trade_to_dict, write_case_html, write_reports
from analysis.backtest.trend.audit import audit_summary, decision_records, to_engine_result, write_jsonl
from analysis.backtest.trend.baselines import simulate_buy_hold, simulate_even_dca
from analysis.backtest.trend.config import BreakEntry, D0Entry, HK, TrendEntry, TrendStrategyConfig
from analysis.backtest.trend.entries import create_entry
from analysis.backtest.trend.entries.base import scale_label
from analysis.backtest.trend.feed import enclose_ohlc, load_raw_bars
from analysis.backtest.trend.markets import US
from analysis.backtest.trend.prices import CorporateAction, adjust_prices
from analysis.backtest.trend.report import comparison_html, comparison_markdown, write_summary_html
from analysis.backtest.trend.session import ReplayBar, run_account
from analysis.backtest.trend.tape import build_tape, canonical_tape, window_bounds
from analysis.structure.config import default_params, partial_history_min_bars
from analysis.structure.series import StandardSeries, build_series
from analysis.structure.snapshot import build_snapshot

_RAW = "raw"


class _SummaryView:
    """现有报告只读这三项。金额和区间在 payload['config'] 里。"""

    def __init__(self, config: TrendStrategyConfig) -> None:
        self.symbol_names = {item.code: item.name for item in config.symbols}
        self.benchmarks = {
            key: {"name": spec.name, "yahoo": spec.yahoo, "fallbacks": []}
            for key, spec in config.benchmarks.items()
        }
        self.strategy_name = "trend_state"


@dataclass(frozen=True)
class StrategyArm:
    """相对当前 YAML 只改一项。空字段表示沿用配置。"""

    key: str
    trail_buffer: Optional[float] = None
    trail_states: Optional[Tuple[str, ...]] = None
    breakout_lookback: Optional[int] = None
    breakout_buffer: Optional[float] = None
    pullback_break_lookback: Optional[int] = None


def run_trend_backtest(
    config: TrendStrategyConfig,
    *,
    frames: Optional[Mapping[str, pd.DataFrame]] = None,
    sources: Optional[Mapping[str, str]] = None,
    load_benchmarks: Optional[bool] = None,
    now: Optional[datetime] = None,
    arms: Optional[Sequence[StrategyArm]] = None,
    persist_cases: bool = True,
) -> Dict:
    if not config.symbols:
        raise ValueError("symbols 不能为空")
    if not config.runs:
        raise ValueError("runs 不能为空")

    params = default_params(config.structure.price_axis, version=config.structure.param_version)
    fetch_benchmarks = frames is None if load_benchmarks is None else load_benchmarks
    probe = _case_config(config, config.symbols[0], config.runs[0])
    benchmarks, bench_daily, bench_meta, bench_accounts = _load_benchmarks(config, probe, fetch_benchmarks)

    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    out_root = Path(config.output_dir) / f"{stamp}_{config.window.interval}_trend_state"
    results = []
    failures = []

    for symbol in config.symbols:
        symbol_rows = []
        try:
            _run_symbol(
                config,
                symbol,
                params=params,
                frames=frames,
                sources=sources,
                benchmarks=benchmarks,
                bench_daily=bench_daily,
                bench_accounts=bench_accounts,
                out_root=out_root,
                symbol_rows=symbol_rows,
                arms=arms,
                persist_cases=persist_cases,
            )
        except Exception as exc:
            _drop_case_dirs(out_root, symbol.code, config.runs)
            failures.append({
                "symbol": symbol.code,
                "name": symbol.name,
                "error": str(exc),
            })
            if isinstance(exc, (ValueError, RuntimeError, OSError)):
                logger.error(f"{symbol.code} 跳过：{exc}")
            else:
                logger.exception(f"{symbol.code} 跳过：{exc}")
            continue
        results.extend(symbol_rows)

    if failures:
        logger.error(
            f"完成 {len(config.symbols) - len(failures)} 只，跳过 {len(failures)} 只"
        )

    run_meta = [{"key": run.key, "label": run.label} for run in config.runs]
    payload = {
        "generated_at": (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S"),
        "config": config.to_dict(),
        "benchmarks": bench_meta,
        "results": results,
        "comparison_markdown": comparison_markdown(results, run_meta),
        "comparison_html": comparison_html(results, run_meta, out_root),
        "failures": failures,
    }
    payload["output_dir"] = str(out_root)
    if not persist_cases:
        payload["report_md"] = ""
        payload["index_html"] = ""
        logger.info(f"对照完成，成功 {len(results)} 个账户，跳过 {len(failures)} 只")
        return payload
    md_path = write_reports(out_root, payload, _SummaryView(config))
    write_summary_html(
        out_root,
        results,
        run_meta,
        generated_at=payload["generated_at"],
        config=payload["config"],
        failures=failures,
    )
    payload["report_md"] = str(md_path)
    payload["index_html"] = str(out_root / "index.html")
    logger.info(f"报告已写入 {md_path}")
    return payload


def _run_symbol(
    config,
    symbol,
    *,
    params,
    frames,
    sources,
    benchmarks,
    bench_daily,
    bench_accounts,
    out_root,
    symbol_rows,
    arms: Optional[Sequence[StrategyArm]] = None,
    persist_cases: bool = True,
) -> None:
    frame, actions, source, basis = _load_frame(
        symbol.code,
        config,
        params,
        frames,
        sources,
    )
    frame = _through(frame, config.window.end)
    frame, repaired = enclose_ohlc(frame)
    actions = tuple(item for item in actions if item.index < len(frame))
    adjusted = adjust_prices(
        frame["open"].astype(float).tolist(),
        frame["high"].astype(float).tolist(),
        frame["low"].astype(float).tolist(),
        frame["close"].astype(float).tolist(),
        frame["volume"].astype(float).tolist(),
        actions,
    )
    series = bars_to_series(_adjusted_frame(frame, adjusted), params)
    snapshot = build_snapshot(
        series,
        symbol=symbol.code,
        requested_start=config.window.start,
        requested_end=config.window.end,
        visible_scales=[config.structure.scale],
        data_source=source,
    )
    eval_bars = slice_eval_window(frame, config.window.start, config.window.end)
    if eval_bars.empty:
        raise RuntimeError(
            f"{symbol.code} 在 {config.window.start} ~ {config.window.end} 无可用 K 线"
        )
    start_index, end_index = window_bounds(series, config.window.start, config.window.end)
    tape = build_tape(
        series,
        snapshot,
        scale=config.structure.scale,
        span=config.unit.span,
        floor=config.unit.floor,
        start_index=0,
        end_index=end_index,
        extra_scale=_sell_scale(config),
        break_scales=_break_scales(config),
    )
    if len(tape) != len(series):
        raise RuntimeError(f"{symbol.code} 磁带 {len(tape)} 根，序列 {len(series)} 根")
    replay = _replay_bars(frame, adjusted, actions, series, tape)
    if len(replay) != end_index + 1:
        raise RuntimeError(f"{symbol.code} 回放 {len(replay)} 根，评估终点 {end_index}")
    identity = snapshot.get("identity") or {}
    structure = {
        "param_version": identity.get("param_version"),
        "data_hash": identity.get("data_hash"),
        "engine_version": identity.get("engine_version"),
        "price_axis": identity.get("price_axis"),
        "scale": config.structure.scale,
        "bar_count": identity.get("bar_count"),
        "methods": sorted((snapshot.get("methods") or {}).keys()),
        "price_basis": basis,
        "tape_bars": end_index - start_index + 1,
        "tape_hash": hashlib.sha256(canonical_tape(tape[start_index:]).encode("utf-8")).hexdigest(),
        "git": _git_short(),
    }
    state_days = _state_days(replay, start_index)
    logger.info(
        f"{symbol.code} 快照 {structure['param_version']} "
        f"{structure['bar_count']} 根，哈希 {structure['data_hash']}"
        + (f"，修补 {repaired} 根" if repaired else "")
    )

    chosen = list(arms) if arms else [None]
    for arm in chosen:
        for run in config.runs:
            _run_one(
                config,
                symbol,
                run,
                arm,
                replay=replay,
                tape=tape,
                frame=frame,
                adjusted=adjusted,
                actions=actions,
                start_index=start_index,
                end_index=end_index,
                eval_bars=eval_bars,
                benchmarks=benchmarks,
                bench_daily=bench_daily,
                bench_accounts=bench_accounts,
                source=source,
                basis=basis,
                structure=structure,
                repaired=repaired,
                state_days=state_days,
                out_root=out_root,
                symbol_rows=symbol_rows,
                persist_cases=persist_cases,
            )


def _run_one(
    config,
    symbol,
    run,
    arm: Optional[StrategyArm],
    *,
    replay,
    tape,
    frame,
    adjusted,
    actions,
    start_index,
    end_index,
    eval_bars,
    benchmarks,
    bench_daily,
    bench_accounts,
    source,
    basis,
    structure,
    repaired,
    state_days,
    out_root,
    symbol_rows,
    persist_cases: bool,
) -> None:
    case_config = _case_config(config, symbol, run)
    strategy = _strategy_for_arm(config, run, arm)
    tactic = create_entry(run.strategy, strategy, price_axis=config.structure.price_axis)
    account = run_account(
        replay,
        tactic,
        strategy,
        config.markets[symbol.market],
        lot_size=symbol.lot_size,
        capital=float(config.markets[symbol.market].capital),
        price_axis=config.structure.price_axis,
        withholding=symbol.dividend_withholding,
        trade_from=start_index,
    )
    engine_result = to_engine_result(account, case_config.capital, tactic=run.strategy)
    audit = audit_summary(account, exhaustion_enabled=run.exhaustion_action == "reduce")
    records = []
    if persist_cases:
        records = decision_records(
            account,
            strategy=run.strategy,
            run=run.key,
            scale=_scale_text(config, strategy),
            price_axis=config.structure.price_axis,
            currency=config.markets[symbol.market].currency,
            lot_size=symbol.lot_size,
            audit=config.evaluation.audit,
            tape_by_index={row.index: row for row in tape},
        )
    metrics = compute_metrics(
        engine_result,
        case_config.capital,
        frame,
        eval_bars,
        {key: value for key, value in benchmarks.items() if value is not None},
        config.window.start,
        config.window.end,
        list(config.benchmarks.keys()),
    )
    metrics["cumulative_fees"] = round(engine_result.cumulative_fees, 2)
    hold, dca = _account_baselines(
        frame, adjusted, actions, start_index, end_index,
        lot_size=symbol.lot_size,
        capital=case_config.capital,
        market=config.markets[symbol.market],
        withholding=symbol.dividend_withholding,
    )
    metrics["buy_hold_return"] = round(hold.total_return, 6)
    metrics["buy_hold_dividend_income"] = round(hold.dividend_income, 2)
    metrics["dca_return"] = round(dca.total_return, 6)
    metrics["dca_dividend_income"] = round(dca.dividend_income, 2)
    metrics["excess_vs_dca"] = round(metrics["total_return"] - dca.total_return, 6)
    for key, baseline in bench_accounts.items():
        if baseline is None:
            continue
        metrics[f"{key}_return"] = round(baseline.total_return, 6)
        metrics[f"excess_vs_{key}"] = round(metrics["total_return"] - baseline.total_return, 6)
    metrics["dca_n_lots"] = dca.n_lots
    metrics["dca_interval"] = round(dca.interval, 4) if dca.interval is not None else None
    metrics["dca_n_lots_bought"] = dca.n_lots_bought
    metrics["dca_n_buy_days"] = dca.n_buy_days
    metrics["dca_n_trading_days"] = dca.n_trading_days
    metrics["dca_start_price"] = round(dca.start_price, 4)
    metrics["dca_lot_size"] = dca.lot_size
    metrics["max_consecutive_losses"] = audit["max_consecutive_losses"]
    chart_bars = _with_stop(eval_bars, engine_result.equity)
    daily = daily_frame(engine_result.equity)
    split = split_cost_table(daily, engine_result.trades, case_config)
    extremes = extreme_days(daily, engine_result.trades, config.evaluation.extreme_days)
    gaps = data_gaps(case_config, daily)
    gaps.append("策略信号和对照账户的买卖用复权价。对照账户的股息另计入收入，不重复折进价格。策略成交仍用不复权价。")
    if basis != _RAW and basis != "provided":
        gaps.append(f"价格口径 {basis}。")
    if repaired:
        gaps.append(f"修补了 {repaired} 根 K 线的最高价或最低价，使它们包住开盘价和收盘价。")

    charts = {}
    if persist_cases:
        folder = run.key if arm is None else f"{arm.key}_{run.key}"
        case_dir = out_root / f"{symbol.code}_{folder}"
        case_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(case_dir / "decisions.jsonl", records)
        charts = _write_charts(
            case_dir,
            chart_bars,
            engine_result.trades,
            daily,
            dca,
            bench_daily,
            case_config,
        )
    coverage = coverage_note(
        eval_bars,
        config.window.start,
        config.window.end,
        config.window.interval,
    )
    coverage += "；" + _timing_note(strategy)
    row = {
        "symbol": symbol.code,
        "strategy": "trend_state",
        "strategy_display": "趋势状态",
        "variant": run.key,
        "variant_label": run.label,
        "bar_source": source,
        "coverage": coverage,
        "metrics": metrics,
        "split": split,
        "round_trips": audit["round_trips"],
        "extremes": extremes,
        "gaps": gaps,
        "trades": [trade_to_dict(item) for item in engine_result.trades],
        "skipped": [trade_to_dict(item) for item in engine_result.skipped],
        "charts": {key: str(path) if path else None for key, path in charts.items()},
        "structure": structure,
        "run": run.key,
        "exhaustion_action": run.exhaustion_action,
        "currency": "HKD" if symbol.market == HK else "USD",
        "capital": case_config.capital,
        "name": symbol.name,
        "audit": audit,
        "summary_href": "../index.html",
        "arm": None if arm is None else arm.key,
        "repaired_bars": repaired,
        "state_days": state_days,
        "preamble": _preamble(
            config, symbol, run, structure, hold.dividend_income, dca.dividend_income,
        ),
    }
    if arm is not None:
        row["variant"] = f"{arm.key}_{run.key}"
        row["variant_label"] = f"{arm.key} · {run.label}"
    if persist_cases:
        html_path = write_case_html(row, case_config, case_dir, charts)
        row["report_html"] = str(html_path)
    else:
        row["report_html"] = ""
    symbol_rows.append(row)
    arm_name = "" if arm is None else f" {arm.key}"
    logger.info(
        f"{symbol.code}{arm_name} {run.key}: equity={metrics['final_equity']} "
        f"trades={metrics['n_trades']}"
    )


def _strategy_for_arm(config: TrendStrategyConfig, run, arm: Optional[StrategyArm]):
    strategy = replace(config.strategies[run.strategy], exit=config.exit_for(run))
    if arm is None:
        return strategy
    if arm.trail_buffer is not None:
        strategy = replace(strategy, exit=replace(strategy.exit, trail_buffer=arm.trail_buffer))
    if arm.trail_states is not None:
        strategy = replace(strategy, exit=replace(strategy.exit, trail_states=arm.trail_states))
    if arm.breakout_lookback is not None and run.strategy == "breakout":
        strategy = replace(strategy, entry=replace(strategy.entry, lookback=arm.breakout_lookback))
    if arm.breakout_buffer is not None and run.strategy == "breakout":
        strategy = replace(strategy, entry=replace(strategy.entry, buffer=arm.breakout_buffer))
    if arm.pullback_break_lookback is not None and run.strategy == "pullback":
        strategy = replace(strategy, entry=replace(strategy.entry, break_lookback=arm.pullback_break_lookback))
    return strategy


def _state_days(replay, start_index: int) -> dict:
    structures: Counter = Counter()
    categories: Counter = Counter()
    for bar in replay[start_index:]:
        structures[bar.structure or "不可用"] += 1
        categories[bar.category or "不可用"] += 1
    return {
        "structure": dict(structures),
        "category": dict(categories),
        "bars": len(replay) - start_index,
    }


def _drop_case_dirs(out_root: Path, code: str, runs) -> None:
    for run in runs:
        path = out_root / f"{code}_{run.key}"
        if path.is_dir():
            shutil.rmtree(path)


def _adjusted_frame(frame: pd.DataFrame, adjusted) -> pd.DataFrame:
    out = frame.copy()
    out["open"] = adjusted.open
    out["high"] = adjusted.high
    out["low"] = adjusted.low
    out["close"] = adjusted.close
    out["volume"] = adjusted.volume
    return out


def _through(frame: pd.DataFrame, end) -> pd.DataFrame:
    times = pd.to_datetime(frame["datetime"])
    if getattr(times.dt, "tz", None) is not None:
        days = times.dt.tz_convert("Asia/Hong_Kong").dt.date
    else:
        days = times.dt.date
    kept = frame.loc[days <= end].reset_index(drop=True)
    if kept.empty:
        raise RuntimeError(f"评估终点 {end} 之前没有 K 线")
    return kept


def _replay_bars(frame, adjusted, actions: Sequence[CorporateAction], series, tape) -> Tuple[ReplayBar, ...]:
    dividends = {item.index: item.value for item in actions if item.kind == "dividend"}
    splits = {item.index: item.value for item in actions if item.kind == "split"}
    by_index = {row.index: row for row in tape}
    bars = []
    for index in range(len(frame)):
        row = by_index.get(index)
        if row is None:
            raise RuntimeError(f"第 {index} 根没有磁带")
        stamp = pd.Timestamp(frame["datetime"].iloc[index])
        bars.append(ReplayBar(
            time=stamp.isoformat(),
            open=float(frame["open"].iloc[index]),
            high=float(frame["high"].iloc[index]),
            low=float(frame["low"].iloc[index]),
            close=float(frame["close"].iloc[index]),
            structure=row.structure,
            category=row.category,
            a_t=row.a_t,
            factor=float(adjusted.factor[index]),
            ema=row.ema,
            dividend=float(dividends.get(index, 0.0)),
            split=float(splits.get(index, 1.0)),
            gap_before=bool(series.bars[index].gap_before),
            phase_start_index=row.phase_start_index,
            direction_start_index=row.direction_start_index,
            exit_scale=row.exit_scale,
            exit_structure=row.exit_structure,
            exit_direction_start_index=row.exit_direction_start_index,
            up_breaks=tuple(row.up_breaks),
            down_breaks=tuple(row.down_breaks),
            support_valid=bool(row.support_valid),
            phase_trigger=row.phase_trigger,
            range_floor=row.range_floor,
        ))
    return tuple(bars)


def _with_stop(bars: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    if equity is None or equity.empty or "stop" not in equity.columns or len(equity) != len(bars):
        return bars
    out = bars.copy()
    out["stop"] = equity["stop"].to_numpy()
    return out


def _preamble(config: TrendStrategyConfig, symbol, run, structure: dict, hold_income: float, dca_income: float) -> Tuple[str, ...]:
    market = config.markets[symbol.market]
    return (
        f"市场 {symbol.market}，币种 {market.currency}，每手 {symbol.lot_size} 股（{symbol.lot_size_source}）。",
        "价格口径：信号和对照账户的买卖用复权价。买入持有、均匀定投和指数基准的股息另计入收入。策略成交用不复权价。",
        (
            f"买入持有股息收入 {hold_income:.2f} {market.currency}；"
            f"均匀定投股息收入 {dca_income:.2f} {market.currency}。"
        ),
        (
            f"结构参数 {structure.get('param_version')}，价格轴 {config.structure.price_axis}，"
            f"{_scale_sentence(config, run)}"
        ),
        f"数据哈希 {structure.get('data_hash')}，引擎 git {structure.get('git')}。",
        _timing_note(config.strategies[run.strategy]) + "。",
    )


def _break_scales(config: TrendStrategyConfig) -> Tuple[str, ...]:
    found = []
    for strategy in config.strategies.values():
        entry = strategy.entry
        if not isinstance(entry, BreakEntry):
            continue
        for scale in (entry.scale, entry.context_scale):
            if scale not in found:
                found.append(scale)
    return tuple(found)


def _sell_scale(config: TrendStrategyConfig) -> Optional[str]:
    found = []
    for run in config.runs:
        entry = config.strategies[run.strategy].entry
        if isinstance(entry, TrendEntry):
            found.append(entry.sell_scale)
    if not found:
        return None
    if len(set(found)) != 1:
        raise RuntimeError("所选 run 里策略 C 的卖出尺度不一致")
    return found[0]


def _scale_text(config: TrendStrategyConfig, strategy) -> str:
    entry = strategy.entry
    if isinstance(entry, D0Entry):
        return entry.scale
    if isinstance(entry, TrendEntry):
        return f"{entry.buy_scale}/{entry.sell_scale}"
    if isinstance(entry, BreakEntry):
        return f"{entry.scale}/{entry.context_scale}"
    return config.structure.scale


def _scale_sentence(config: TrendStrategyConfig, run) -> str:
    entry = config.strategies[run.strategy].entry
    if isinstance(entry, D0Entry):
        drawdown = config.strategies[run.strategy].exit.drawdown
        stop = ""
        if drawdown is not None:
            percent = f"{float(drawdown) * 100:.4f}".rstrip("0").rstrip(".")
            stop = f"前高回撤 {percent}%，"
        return (
            f"买卖尺度 {scale_label(entry.scale)}，{stop}"
            f"策略 {config.version}，run {run.key}。"
        )
    if isinstance(entry, TrendEntry):
        return (
            f"买入尺度 {scale_label(entry.buy_scale)}，卖出尺度 {scale_label(entry.sell_scale)}，"
            f"策略 {config.version}，run {run.key}。"
        )
    if isinstance(entry, BreakEntry):
        return (
            f"买卖尺度 {scale_label(entry.scale)}，参照尺度 {scale_label(entry.context_scale)}，"
            f"策略 {config.version}，run {run.key}。"
        )
    return f"尺度 {config.structure.scale}，策略 {config.version}，run {run.key}。"


def _timing_note(strategy) -> str:
    entry = strategy.entry
    if isinstance(entry, D0Entry):
        drawdown = strategy.exit.drawdown
        name = scale_label(entry.scale)
        if drawdown is None:
            return (
                "信号使用当根 as-of，成交在下一根开盘。"
                f"策略 D0 不设价格止损，{name}上升趋势确认后的下一根开盘买入，"
                f"{name}下降趋势确认后的下一根开盘退出"
            )
        percent = f"{float(drawdown) * 100:.4f}".rstrip("0").rstrip(".")
        return (
            "信号使用当根 as-of，成交在下一根开盘。"
            f"策略 D1 按持有以来最高价回撤 {percent}% 止损，前高在收盘后更新，新止损下一根生效。"
            f"{name}下降趋势确认后的下一根开盘也退出"
        )
    if isinstance(entry, TrendEntry):
        sell = scale_label(entry.sell_scale)
        return (
            "信号使用当根 as-of，成交在下一根开盘。"
            f"策略 C 不设价格止损，{sell}下降趋势确认后的下一根开盘退出"
        )
    return "信号使用当根 as-of，成交在下一根开盘；盘中止损使用上一根收盘已经生效的止损"


def _git_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or "unknown"
    except Exception:
        return "unknown"


def bars_to_series(frame: pd.DataFrame, params) -> StandardSeries:
    if frame is None or frame.empty:
        raise ValueError("K 线为空")
    missing = [
        column
        for column in ("datetime", "open", "high", "low", "close", "volume")
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"K 线缺少列: {', '.join(missing)}")
    return build_series(
        [_as_datetime(value) for value in frame["datetime"]],
        [float(value) for value in frame["close"]],
        interval="1d",
        params=params,
        highs=[float(value) for value in frame["high"]],
        lows=[float(value) for value in frame["low"]],
        volumes=[float(value) for value in frame["volume"]],
        opens=[float(value) for value in frame["open"]],
    )


def _load_frame(symbol, config, params, frames, sources):
    if frames is not None:
        if symbol not in frames:
            raise ValueError(f"没有 {symbol} 的 K 线")
        source = "injected" if not sources else str(sources.get(symbol) or "injected")
        frame = frames[symbol]
        actions = ()
        if isinstance(frame, tuple):
            frame, actions = frame
        return frame, tuple(actions), source, "provided"
    symbol_spec = _symbol_named(config, symbol)
    case = _case_config(config, symbol_spec, config.runs[0])
    padded = config.window.start - timedelta(days=case.warmup_calendar_days(config.window.warmup_bars))
    frame, actions, source = load_raw_bars(symbol, padded, config.window.end)
    if len(frame) < partial_history_min_bars(params):
        raise RuntimeError(
            f"{symbol} 不复权 K 线只有 {len(frame)} 根，少于结构分析所需的 {partial_history_min_bars(params)} 根"
        )
    return frame, actions, source, _RAW


def _symbol_named(config: TrendStrategyConfig, code: str):
    for item in config.symbols:
        if item.code == code:
            return item
    raise ValueError(f"配置里没有 {code}")


def _case_config(config: TrendStrategyConfig, symbol, run) -> BacktestConfig:
    market = config.markets[symbol.market]
    strategy = config.strategies[run.strategy]
    if symbol.market == HK:
        cost = CostModel(
            commission_rate=market.commission_rate,
            stamp_duty_rate=market.stamp_duty_rate,
            lot_size=symbol.lot_size,
        )
    else:
        cost = CostModel(commission_rate=0.0, stamp_duty_rate=0.0, lot_size=symbol.lot_size)
    return BacktestConfig(
        symbols=[symbol.code],
        symbol_names={symbol.code: symbol.name},
        symbol_lot_sizes={symbol.code: symbol.lot_size},
        strategy_name="trend_state",
        interval=config.window.interval,
        start=config.window.start,
        end=config.window.end,
        lookback_years=config.window.lookback_years,
        capital=float(market.capital),
        max_trades_per_day=strategy.order.max_trades_per_day,
        cost=cost,
        timezone=market.timezone,
        benchmarks={
            key: {"name": spec.name, "yahoo": spec.yahoo, "fallbacks": []}
            for key, spec in config.benchmarks.items()
        },
        oos_ratio=config.evaluation.oos_ratio,
        rolling_window_days=config.evaluation.rolling_window_days,
        extreme_days=config.evaluation.extreme_days,
        output_dir=config.output_dir,
        config_path=config.config_path,
    )


_BENCHMARK_LOTS = {"2800.HK": 500, "SPY": 1}


def _load_benchmarks(config: TrendStrategyConfig, probe: BacktestConfig, enabled: bool):
    benchmarks: Dict[str, object] = {}
    bench_daily: Dict[str, object] = {}
    bench_meta: Dict[str, Dict] = {}
    accounts: Dict[str, object] = {}
    for key, spec in config.benchmarks.items():
        if not enabled:
            benchmarks[key] = None
            bench_daily[key] = None
            accounts[key] = None
            bench_meta[key] = {
                "name": spec.name,
                "source": None,
                "coverage": "未加载基准",
                "buy_hold_return": None,
                "dividend_income": None,
            }
            continue
        try:
            hold, source = _benchmark_hold(config, spec.yahoo)
            shown = hold.equity.rename(columns={"price": "close"})
            benchmarks[key] = shown
            bench_daily[key] = _equity_series(hold.equity)
            accounts[key] = hold
            bench_meta[key] = {
                "name": spec.name,
                "source": source,
                "coverage": coverage_note(shown, config.window.start, config.window.end, "1d"),
                "buy_hold_return": hold.total_return,
                "dividend_income": hold.dividend_income,
            }
            logger.info(f"基准 {key}: {bench_meta[key]['coverage']} ({source})")
        except Exception as exc:
            logger.warning(f"基准 {key} 加载失败: {exc}")
            benchmarks[key] = None
            bench_daily[key] = None
            accounts[key] = None
            bench_meta[key] = {
                "name": spec.name,
                "source": None,
                "coverage": f"加载失败: {exc}",
                "buy_hold_return": None,
                "dividend_income": None,
            }
    return benchmarks, bench_daily, bench_meta, accounts


def _benchmark_hold(config: TrendStrategyConfig, yahoo: str):
    market_key = HK if yahoo.upper().endswith(".HK") else US
    market = config.markets[market_key]
    lot_size = _BENCHMARK_LOTS.get(yahoo.upper(), market.lot_size if market_key == US else 100)
    padded = config.window.start - timedelta(days=_case_config(
        config, config.symbols[0], config.runs[0],
    ).warmup_calendar_days(config.window.warmup_bars))
    frame, actions, source = load_raw_bars(yahoo, padded, config.window.end)
    frame = _through(frame, config.window.end)
    actions = tuple(item for item in actions if item.index < len(frame))
    adjusted = adjust_prices(
        frame["open"].astype(float).tolist(),
        frame["high"].astype(float).tolist(),
        frame["low"].astype(float).tolist(),
        frame["close"].astype(float).tolist(),
        frame["volume"].astype(float).tolist(),
        actions,
    )
    start, end = _date_bounds(frame, config.window.start, config.window.end)
    dates, opens, closes, factors, splits, dividends = _window_columns(
        frame, adjusted, actions, start, end,
    )
    hold = simulate_buy_hold(
        dates, opens, closes, factors, splits, dividends,
        lot_size=lot_size,
        capital=float(market.capital),
        market=market,
        withholding=float(market.dividend_withholding),
    )
    return hold, source


def _account_baselines(frame, adjusted, actions, start, end, *, lot_size, capital, market, withholding):
    dates, opens, closes, factors, splits, dividends = _window_columns(
        frame, adjusted, actions, start, end,
    )
    shared = dict(
        lot_size=lot_size, capital=capital, market=market, withholding=withholding,
    )
    return (
        simulate_buy_hold(dates, opens, closes, factors, splits, dividends, **shared),
        simulate_even_dca(dates, opens, closes, factors, splits, dividends, **shared),
    )


def _window_columns(frame, adjusted, actions, start, end):
    dates = [frame["datetime"].iloc[index] for index in range(start, end + 1)]
    dividends = [0.0] * (end - start + 1)
    for action in actions:
        if action.kind == "dividend" and start <= action.index <= end:
            dividends[action.index - start] = float(action.value)
    return (
        dates,
        adjusted.open[start:end + 1],
        adjusted.close[start:end + 1],
        adjusted.factor[start:end + 1],
        adjusted.split_ratio[start:end + 1],
        dividends,
    )


def _date_bounds(frame: pd.DataFrame, start, end) -> Tuple[int, int]:
    times = pd.to_datetime(frame["datetime"])
    if getattr(times.dt, "tz", None) is not None:
        days = times.dt.tz_convert("Asia/Hong_Kong").dt.date
    else:
        days = times.dt.date
    chosen = [index for index, day in enumerate(days) if start <= day <= end]
    if not chosen:
        raise ValueError(f"评估窗口 {start} ~ {end} 内没有 K 线")
    return chosen[0], chosen[-1]


def _equity_series(equity: pd.DataFrame) -> pd.Series:
    if equity is None or equity.empty:
        return pd.Series(dtype=float)
    index = to_hk_index(equity["datetime"])
    return pd.Series(equity["equity"].to_numpy(dtype=float), index=index)


def _write_charts(case_dir, bars, trades, daily, dca, bench_daily, config):
    charts: Dict[str, Optional[Path]] = {}
    charts["kline"] = _safe_chart(plot_daily_kline_with_trades, bars, trades, case_dir)
    if daily.empty or "equity" not in daily.columns:
        return charts
    strat_nav = nav_from_price(daily["equity"])
    bench_navs = {}
    bench_rets = {}
    for key, prices in bench_daily.items():
        if prices is None or getattr(prices, "empty", True):
            continue
        aligned = prices.reindex(daily.index).ffill()
        name = (config.benchmarks.get(key) or {}).get("name", key)
        bench_navs[name] = nav_from_price(aligned.dropna())
        bench_rets[key] = aligned.pct_change()
    dca_daily = daily_frame(dca.equity) if dca.equity is not None and not dca.equity.empty else None
    if dca_daily is not None and not dca_daily.empty and "equity" in dca_daily.columns:
        dca_aligned = dca_daily["equity"].reindex(daily.index).ffill()
        bench_navs["均匀定投"] = nav_from_capital(dca_aligned, config.capital)
        bench_rets["dca"] = dca_aligned.pct_change()
    charts["nav"] = _safe_chart(plot_nav_and_underwater, strat_nav, bench_navs, case_dir)
    heatmap = monthly_return_table(daily["ret_net"]) if "ret_net" in daily.columns else None
    charts["heatmap"] = _safe_chart(plot_monthly_heatmap, heatmap, case_dir)
    roll = rolling_stats(
        daily["ret_net"] if "ret_net" in daily.columns else daily["equity"].pct_change(),
        bench_rets,
        config.rolling_window_days,
    )
    charts["rolling"] = _safe_chart(plot_rolling, roll, config.rolling_window_days, case_dir)
    charts["position"] = _safe_chart(plot_position_turnover_leverage, daily, case_dir)
    charts["factor"] = _safe_chart(plot_factor_exposure, factor_exposures(daily), case_dir)
    return charts


def _safe_chart(fn, *args, **kwargs) -> Optional[Path]:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.warning(f"图表生成失败 {getattr(fn, '__name__', fn)}: {exc}")
        return None


def _as_datetime(value):
    if isinstance(value, datetime):
        return value
    to_py = getattr(value, "to_pydatetime", None)
    if callable(to_py):
        parsed = to_py()
        if isinstance(parsed, datetime):
            return parsed
    raise ValueError(f"无法解析 K 线时间 {value!r}")
