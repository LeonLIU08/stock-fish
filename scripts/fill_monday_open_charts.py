#!/usr/bin/env python3
"""Fill chart images into an existing monday_open HTML report."""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/data2/stock-fish/.mplconfig")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.backtest.analytics import (  # noqa: E402
    align_benchmark_daily,
    daily_frame,
    factor_exposures,
    monthly_return_table,
    nav_from_capital,
    nav_from_price,
    rolling_stats,
    simulate_even_dca,
)
from analysis.backtest.bars import BarLoader, slice_eval_window  # noqa: E402
from analysis.backtest.charts import (  # noqa: E402
    plot_daily_kline_with_trades,
    plot_factor_exposure,
    plot_monthly_heatmap,
    plot_nav_and_underwater,
    plot_position_turnover_leverage,
    plot_rolling,
)
from analysis.backtest.config import BacktestConfig, CostModel, load_backtest_config  # noqa: E402
from analysis.backtest.strategies.monday_open import MondayOpenStrategy  # noqa: E402

OUT = PROJECT_ROOT / "backtest_results" / "20261027-171406_60m_monday_open"
SECTIONS = (
    ("日K线 + 均线 + 买卖点", "daily_kline_trades.png"),
    ("净值曲线（策略/基准/超额，对数坐标）+ 回撤水下图", "nav_underwater.png"),
    ("逐年逐月收益热力图", "monthly_heatmap.png"),
    ("滚动 12 个月收益、滚动 Sharpe、滚动 Beta", "rolling.png"),
    ("持仓数量 / 换手率 / 杠杆", "position_turnover_leverage.png"),
    ("因子暴露时序（风格漂移代理）", "factor_exposure.png"),
)


def _config(symbol: str, lot: int) -> BacktestConfig:
    return BacktestConfig(
        symbols=[symbol],
        symbol_names={},
        symbol_lot_sizes={symbol: lot},
        interval="60m",
        start=date(2026, 4, 15),
        end=date(2026, 10, 27),
        capital=1_000_000,
        cost=CostModel(commission_rate=0.0003, stamp_duty_rate=0.001, lot_size=lot),
        strategy_name="monday_open",
        strategy_params={"take_profit": 0.15, "stop_loss": 0.05},
        benchmarks={},
        allow_short_history=True,
        write_charts=True,
    )


def _patch(html_path: Path, made: dict) -> None:
    text = html_path.read_text(encoding="utf-8")
    for title, name in SECTIONS:
        if name not in made:
            continue
        old = f"<h2>{title}</h2><p class='muted'>无数据</p>"
        new = (
            f'<h2>{title}</h2><div class="chart">'
            f'<img src="{name}" alt="{title}" /></div>'
        )
        text = text.replace(old, new, 1)
    html_path.write_text(text, encoding="utf-8")


def render_symbol(symbol: str, lot: int, bench_daily: dict) -> str:
    import logging
    import warnings

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*font weight.*")
    case_dir = OUT / f"{symbol}_band"
    html_path = case_dir / "report.html"
    if not html_path.is_file():
        return f"{symbol} missing html"
    cfg = _config(symbol, lot)
    strategy = MondayOpenStrategy()
    variant = strategy.variants(cfg)[0]
    bars, _source = BarLoader().load_symbol(
        symbol, cfg, warmup_bars=0, allow_short_history=True,
    )
    prepared = strategy.prepare(bars, cfg, variant)
    signaled = slice_eval_window(prepared.bars, cfg.start, cfg.end)
    engine = strategy.simulate(signaled, cfg, variant, symbol)
    daily = daily_frame(engine.equity)
    dca = simulate_even_dca(signaled, cfg, symbol)
    made = {}

    path = plot_daily_kline_with_trades(prepared.bars, engine.trades, case_dir)
    if path:
        made[path.name] = path

    if not daily.empty and "equity" in daily.columns:
        strat_nav = nav_from_price(daily["equity"])
        bench_navs = {}
        bench_rets = {}
        for key, px in bench_daily.items():
            if px is None or getattr(px, "empty", True):
                continue
            aligned = px.reindex(daily.index).ffill()
            name = {"hsi": "恒生指数", "hstech": "恒生科技指数"}.get(key, key)
            bench_navs[name] = nav_from_price(aligned.dropna())
            bench_rets[key] = aligned.pct_change()
        dca_daily = daily_frame(dca.equity) if dca.equity is not None and not dca.equity.empty else None
        if dca_daily is not None and not dca_daily.empty and "equity" in dca_daily.columns:
            dca_aligned = dca_daily["equity"].reindex(daily.index).ffill()
            bench_navs["均匀定投"] = nav_from_capital(dca_aligned, cfg.capital)
        nav = plot_nav_and_underwater(strat_nav, bench_navs, case_dir)
        if nav:
            made[nav.name] = nav
        heatmap = monthly_return_table(daily["ret_net"]) if "ret_net" in daily.columns else None
        heat = plot_monthly_heatmap(heatmap, case_dir)
        if heat:
            made[heat.name] = heat
        roll = rolling_stats(
            daily["ret_net"] if "ret_net" in daily.columns else daily["equity"].pct_change(),
            bench_rets,
            cfg.rolling_window_days,
        )
        rolled = plot_rolling(roll, cfg.rolling_window_days, case_dir)
        if rolled:
            made[rolled.name] = rolled
        pos = plot_position_turnover_leverage(daily, case_dir)
        if pos:
            made[pos.name] = pos
        fac = plot_factor_exposure(factor_exposures(daily), case_dir)
        if fac:
            made[fac.name] = fac

    _patch(html_path, made)
    return f"{symbol} charts={len(made)}"


def main() -> int:
    import json

    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    lots = summary["config"]["symbol_lot_sizes"]
    symbols = [row["symbol"] for row in summary["results"]]
    base = load_backtest_config(PROJECT_ROOT / "analysis/backtest/monday_open.yaml")
    base.start = date(2026, 4, 15)
    base.end = date(2026, 10, 27)
    loader = BarLoader()
    bench_daily = {}
    for key in ("hsi", "hstech"):
        df, source = loader.load_benchmark(key, base)
        bench_daily[key] = align_benchmark_daily(df)
        print(f"bench {key} {source} {len(bench_daily[key])}", flush=True)

    workers = 4
    done = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(render_symbol, symbol, int(lots.get(symbol, 100)), bench_daily): symbol
            for symbol in symbols
        }
        for fut in as_completed(futures):
            symbol = futures[fut]
            done += 1
            try:
                line = fut.result()
            except Exception as exc:
                failed += 1
                print(f"FAIL {symbol} {exc}", flush=True)
                continue
            if done % 25 == 0 or done == len(symbols):
                print(f"{done}/{len(symbols)} {line}", flush=True)
    print(f"finished {done} failed {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
