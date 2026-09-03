"""Run strategy backtests and write reports."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from analysis.backtest import strategies as _builtin_strategies  # noqa: F401
from analysis.backtest.analytics import (
    align_benchmark_daily,
    daily_frame,
    data_gaps,
    extreme_days,
    factor_exposures,
    monthly_return_table,
    nav_from_price,
    rolling_stats,
    round_trips,
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
from analysis.backtest.config import BacktestConfig
from analysis.backtest.engine import simulate
from analysis.backtest.metrics import benchmark_buy_hold_return, compute_metrics
from analysis.backtest.report import trade_to_dict, write_case_html, write_reports
from analysis.backtest.strategy import create_strategy


def _safe_chart(fn, *args, **kwargs) -> Optional[Path]:
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.warning(f"图表生成失败 {getattr(fn, '__name__', fn)}: {e}")
        return None


def run_backtest(config: BacktestConfig) -> Dict:
    strategy = create_strategy(config.strategy_name)
    variants = strategy.variants(config)
    if not variants:
        raise ValueError(f"策略 {config.strategy_name} 未返回任何可运行 variant")

    loader = BarLoader()
    benchmarks: Dict[str, object] = {}
    bench_meta: Dict[str, Dict] = {}
    bench_daily: Dict[str, object] = {}

    warmup_bars = max(strategy.warmup_bars(config, variant) for variant in variants)

    for key, spec in (config.benchmarks or {}).items():
        try:
            df, source = loader.load_benchmark(key, config, warmup_bars=warmup_bars)
            benchmarks[key] = df
            bench_daily[key] = align_benchmark_daily(df)
            bench_meta[key] = {
                "name": spec.get("name", key),
                "source": source,
                "coverage": coverage_note(df, config.start, config.end, "1d"),
                "buy_hold_return": benchmark_buy_hold_return(df, config.start, config.end),
            }
            logger.info(f"基准 {key}: {bench_meta[key]['coverage']} ({source})")
        except Exception as e:
            logger.warning(f"基准 {key} 加载失败: {e}")
            benchmarks[key] = None
            bench_daily[key] = None
            bench_meta[key] = {
                "name": spec.get("name", key),
                "source": None,
                "coverage": f"加载失败: {e}",
                "buy_hold_return": None,
            }

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path(config.output_dir) / f"{stamp}_{config.interval}_{config.strategy_name}"
    results: List[Dict] = []

    for symbol in config.symbols:
        logger.info(f"加载 {symbol} {config.interval} ...")
        bars, source = loader.load_symbol(symbol, config, warmup_bars=warmup_bars)
        eval_bars = slice_eval_window(bars, config.start, config.end)
        if eval_bars.empty:
            raise RuntimeError(f"{symbol} 在 {config.start} ~ {config.end} 无可用 K 线")

        for variant in variants:
            prepared = strategy.prepare(bars, config, variant)
            signaled = slice_eval_window(prepared.bars, config.start, config.end)
            engine_result = simulate(
                signaled,
                config,
                symbol=symbol,
                overlay_columns=prepared.overlay_columns,
            )
            metrics = compute_metrics(
                engine_result,
                config.capital,
                bars,
                eval_bars,
                {k: v for k, v in benchmarks.items() if v is not None},
                config.start,
                config.end,
                list(config.benchmarks.keys()),
            )
            metrics["cumulative_fees"] = round(engine_result.cumulative_fees, 2)
            daily = daily_frame(engine_result.equity)
            split = split_cost_table(daily, engine_result.trades, config)
            trips = round_trips(engine_result.trades)
            extremes = extreme_days(daily, engine_result.trades, config.extreme_days)
            gaps = data_gaps(config, daily)

            case_dir = out_root / f"{symbol}_{variant.key}"
            case_dir.mkdir(parents=True, exist_ok=True)
            charts: Dict[str, Optional[Path]] = {}
            overlays = strategy.chart_overlays(variant)
            charts["kline"] = _safe_chart(
                plot_daily_kline_with_trades,
                prepared.bars,
                engine_result.trades,
                case_dir,
                overlays=overlays,
            )
            if not daily.empty and "equity" in daily.columns:
                strat_nav = nav_from_price(daily["equity"])
                bench_navs = {}
                bench_rets = {}
                for key, px in bench_daily.items():
                    if px is None or getattr(px, "empty", True):
                        continue
                    aligned = px.reindex(daily.index).ffill()
                    name = (config.benchmarks.get(key) or {}).get("name", key)
                    bench_navs[name] = nav_from_price(aligned.dropna())
                    bench_rets[key] = aligned.pct_change()
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

            row = {
                "symbol": symbol,
                "strategy": config.strategy_name,
                "strategy_display": getattr(strategy, "display_name", config.strategy_name),
                "variant": variant.key,
                "variant_label": variant.label,
                "bar_source": source,
                "coverage": coverage_note(signaled, config.start, config.end, config.interval),
                "metrics": metrics,
                "split": split,
                "round_trips": trips,
                "extremes": extremes,
                "gaps": gaps,
                "trades": [trade_to_dict(t) for t in engine_result.trades],
                "skipped": [trade_to_dict(t) for t in engine_result.skipped],
                "charts": {k: str(v) if v else None for k, v in charts.items()},
            }
            row.update(strategy.result_metadata(variant))
            html_path = write_case_html(row, config, case_dir, charts)
            row["report_html"] = str(html_path)
            results.append(row)
            logger.info(
                f"{symbol} {variant.key}: return={metrics['total_return']:.2%} "
                f"dd={metrics['max_drawdown']:.2%} trades={metrics['n_trades']}"
            )

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config.to_dict(),
        "benchmarks": bench_meta,
        "results": results,
    }
    md_path = write_reports(out_root, payload, config)
    payload["output_dir"] = str(out_root)
    payload["report_md"] = str(md_path)
    payload["index_html"] = str(out_root / "index.html")
    logger.info(f"报告已写入 {md_path}")
    return payload
