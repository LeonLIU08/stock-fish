"""股票 × run 的空跑：取数、一次结构快照、因果磁带、空账户，再写现有格式的报告。

这一阶段不成交。真实取数仍走现有 K 线加载器，价格是前复权；
不复权价和除权事件要等价格口径接上之后才能当作交易回测。
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional

import pandas as pd
from loguru import logger

from analysis.backtest.analytics import (
    align_benchmark_daily,
    daily_frame,
    data_gaps,
    extreme_days,
    factor_exposures,
    monthly_return_table,
    nav_from_capital,
    nav_from_price,
    rolling_stats,
    round_trips,
    simulate_even_dca,
    split_cost_table,
)
from analysis.backtest.bars import BarLoader, coverage_note, slice_eval_window
from analysis.backtest.charts import (
    plot_daily_kline_with_trades,
    plot_factor_exposure,
    plot_monthly_heatmap,
    plot_nav_and_underwater,
    plot_position_turnover_leverage,
    plot_rolling,
)
from analysis.backtest.config import BacktestConfig, CostModel
from analysis.backtest.metrics import benchmark_buy_hold_return, compute_metrics
from analysis.backtest.report import trade_to_dict, write_case_html, write_reports
from analysis.backtest.trend.config import HK, TrendStrategyConfig
from analysis.backtest.trend.session import run_empty_session
from analysis.backtest.trend.tape import build_tape, canonical_tape, window_bounds
from analysis.structure.config import default_params, partial_history_min_bars
from analysis.structure.series import StandardSeries, build_series
from analysis.structure.snapshot import build_snapshot

_FORWARD_ADJUSTED = "forward_adjusted_loader"


class _SummaryView:
    """现有报告只读这三项。金额和区间在 payload['config'] 里。"""

    def __init__(self, config: TrendStrategyConfig) -> None:
        self.symbol_names = {item.code: item.name for item in config.symbols}
        self.benchmarks = {
            key: {"name": spec.name, "yahoo": spec.yahoo, "fallbacks": []}
            for key, spec in config.benchmarks.items()
        }
        self.strategy_name = "trend_state"


def run_trend_backtest(
    config: TrendStrategyConfig,
    *,
    frames: Optional[Mapping[str, pd.DataFrame]] = None,
    sources: Optional[Mapping[str, str]] = None,
    load_benchmarks: Optional[bool] = None,
    now: Optional[datetime] = None,
) -> Dict:
    if not config.symbols:
        raise ValueError("symbols 不能为空")
    if not config.runs:
        raise ValueError("runs 不能为空")

    params = default_params(config.structure.price_axis, version=config.structure.param_version)
    fetch_benchmarks = frames is None if load_benchmarks is None else load_benchmarks
    probe = _case_config(config, config.symbols[0], config.runs[0])
    benchmarks, bench_daily, bench_meta = _load_benchmarks(config, probe, fetch_benchmarks)

    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    out_root = Path(config.output_dir) / f"{stamp}_{config.window.interval}_trend_state"
    results = []

    for symbol in config.symbols:
        frame, source, basis = _load_frame(
            symbol.code,
            config,
            params,
            frames,
            sources,
        )
        series = bars_to_series(frame, params)
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
            start_index=start_index,
            end_index=end_index,
        )
        if len(tape) != len(eval_bars):
            raise RuntimeError(
                f"{symbol.code} 磁带 {len(tape)} 根，评估窗口 {len(eval_bars)} 根"
            )
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
            "tape_bars": len(tape),
            "tape_hash": hashlib.sha256(canonical_tape(tape).encode("utf-8")).hexdigest(),
        }
        logger.info(
            f"{symbol.code} 快照 {structure['param_version']} "
            f"{structure['bar_count']} 根，哈希 {structure['data_hash']}"
        )

        for run in config.runs:
            case_config = _case_config(config, symbol, run)
            engine_result = run_empty_session(eval_bars, case_config.capital)
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
            dca = simulate_even_dca(eval_bars, case_config, symbol.code)
            metrics["dca_return"] = round(dca.total_return, 6)
            metrics["excess_vs_dca"] = round(metrics["total_return"] - dca.total_return, 6)
            metrics["dca_n_lots"] = dca.n_lots
            metrics["dca_interval"] = round(dca.interval, 4) if dca.interval is not None else None
            metrics["dca_n_lots_bought"] = dca.n_lots_bought
            metrics["dca_n_buy_days"] = dca.n_buy_days
            metrics["dca_n_trading_days"] = dca.n_trading_days
            metrics["dca_start_price"] = round(dca.start_price, 4)
            metrics["dca_lot_size"] = dca.lot_size
            daily = daily_frame(engine_result.equity)
            split = split_cost_table(daily, engine_result.trades, case_config)
            trips = round_trips(engine_result.trades)
            extremes = extreme_days(daily, engine_result.trades, config.evaluation.extreme_days)
            gaps = data_gaps(case_config, daily)
            gaps.append("空会话：这一阶段只核对取数、结构快照和报告，不产生买卖。")
            if basis == _FORWARD_ADJUSTED:
                gaps.append("K 线仍是前复权。成交口径尚未启用，不要把这份结果当成交易回测。")

            case_dir = out_root / f"{symbol.code}_{run.key}"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "decisions.jsonl").write_text("", encoding="utf-8")
            charts = _write_charts(
                case_dir,
                eval_bars,
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
            coverage += "；空会话，这一阶段不产生买卖"
            if basis == _FORWARD_ADJUSTED:
                coverage += "；K 线仍是前复权，成交口径尚未启用"
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
                "round_trips": trips,
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
            }
            html_path = write_case_html(row, case_config, case_dir, charts)
            row["report_html"] = str(html_path)
            results.append(row)
            logger.info(
                f"{symbol.code} {run.key}: 空会话 equity={metrics['final_equity']} "
                f"trades={metrics['n_trades']}"
            )

    payload = {
        "generated_at": (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S"),
        "config": config.to_dict(),
        "benchmarks": bench_meta,
        "results": results,
    }
    md_path = write_reports(out_root, payload, _SummaryView(config))
    payload["output_dir"] = str(out_root)
    payload["report_md"] = str(md_path)
    payload["index_html"] = str(out_root / "index.html")
    logger.info(f"报告已写入 {md_path}")
    return payload


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
        return frames[symbol], source, "provided"
    case = _case_config(config, _symbol_named(config, symbol), config.runs[0])
    frame, source = BarLoader().load_symbol(
        symbol,
        case,
        warmup_bars=config.window.warmup_bars,
        allow_short_history=True,
        min_bars=partial_history_min_bars(params),
    )
    return frame, source, _FORWARD_ADJUSTED


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


def _load_benchmarks(config: TrendStrategyConfig, probe: BacktestConfig, enabled: bool):
    benchmarks: Dict[str, object] = {}
    bench_daily: Dict[str, object] = {}
    bench_meta: Dict[str, Dict] = {}
    loader = BarLoader() if enabled else None
    for key, spec in config.benchmarks.items():
        if not enabled or loader is None:
            benchmarks[key] = None
            bench_daily[key] = None
            bench_meta[key] = {
                "name": spec.name,
                "source": None,
                "coverage": "未加载基准",
                "buy_hold_return": None,
            }
            continue
        try:
            frame, source = loader.load_benchmark(key, probe, warmup_bars=config.window.warmup_bars)
            benchmarks[key] = frame
            bench_daily[key] = align_benchmark_daily(frame)
            bench_meta[key] = {
                "name": spec.name,
                "source": source,
                "coverage": coverage_note(frame, config.window.start, config.window.end, "1d"),
                "buy_hold_return": benchmark_buy_hold_return(frame, config.window.start, config.window.end),
            }
            logger.info(f"基准 {key}: {bench_meta[key]['coverage']} ({source})")
        except Exception as exc:
            logger.warning(f"基准 {key} 加载失败: {exc}")
            benchmarks[key] = None
            bench_daily[key] = None
            bench_meta[key] = {
                "name": spec.name,
                "source": None,
                "coverage": f"加载失败: {exc}",
                "buy_hold_return": None,
            }
    return benchmarks, bench_daily, bench_meta


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
