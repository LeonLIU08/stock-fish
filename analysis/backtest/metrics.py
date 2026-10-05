"""Performance metrics versus Hang Seng benchmarks and buy-and-hold."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.backtest.engine import EngineResult, Trade

_HK_TZ = "Asia/Hong_Kong"


def _to_float(v) -> Optional[float]:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


def _daily_equity(equity_df: pd.DataFrame) -> pd.Series:
    if equity_df is None or equity_df.empty:
        return pd.Series(dtype=float)
    eq = equity_df.copy()
    eq["datetime"] = pd.to_datetime(eq["datetime"])
    if eq["datetime"].dt.tz is None:
        eq["datetime"] = eq["datetime"].dt.tz_localize(_HK_TZ)
    else:
        eq["datetime"] = eq["datetime"].dt.tz_convert(_HK_TZ)
    daily = eq.set_index("datetime")["equity"].resample("1D").last().dropna()
    return daily


def _total_return(start: float, end: float) -> float:
    if start is None or start == 0:
        return 0.0
    return float(end / start - 1.0)


def _ann_return(total: float, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> float:
    days = max((pd.Timestamp(end_ts) - pd.Timestamp(start_ts)).days, 1)
    years = days / 365.25
    if years <= 0 or total <= -1:
        return total
    return float((1.0 + total) ** (1.0 / years) - 1.0)


def _sharpe(daily: pd.Series) -> Optional[float]:
    rets = daily.pct_change().dropna()
    if len(rets) < 5 or rets.std(ddof=1) == 0:
        return None
    return float(rets.mean() / rets.std(ddof=1) * np.sqrt(252))


def _buy_hold_return(bars: pd.DataFrame, eval_bars: pd.DataFrame) -> Optional[float]:
    src = eval_bars if eval_bars is not None and not eval_bars.empty else bars
    if src is None or src.empty:
        return None
    first = float(src["close"].iloc[0])
    last = float(src["close"].iloc[-1])
    if first <= 0:
        return None
    return last / first - 1.0


def benchmark_buy_hold_return(bench: pd.DataFrame, start, end) -> Optional[float]:
    if bench is None or bench.empty:
        return None
    ts = bench["datetime"]
    if getattr(ts.dt, "tz", None) is None:
        day = ts.dt.date
    else:
        day = ts.dt.tz_convert(_HK_TZ).dt.date
    window = bench[(day >= start) & (day <= end)]
    if window.empty:
        window = bench
    first = float(window["close"].iloc[0])
    last = float(window["close"].iloc[-1])
    if first <= 0:
        return None
    return last / first - 1.0


def _round_trips(trades: List[Trade]) -> List[Dict]:
    trips = []
    open_trade = None
    for t in trades:
        if t.skipped:
            continue
        if t.side == "buy":
            open_trade = t
        elif t.side == "sell" and open_trade is not None:
            pnl = (t.price - open_trade.price) * t.shares - t.fee - open_trade.fee
            trips.append({
                "entry": str(open_trade.timestamp),
                "exit": str(t.timestamp),
                "pnl": pnl,
                "return": (t.price / open_trade.price - 1.0) if open_trade.price else 0.0,
            })
            open_trade = None
    return trips


def compute_metrics(
    result: EngineResult,
    capital: float,
    bars: pd.DataFrame,
    eval_bars: pd.DataFrame,
    benchmarks: Dict[str, pd.DataFrame],
    start,
    end,
    benchmark_keys: Optional[List[str]] = None,
) -> Dict:
    equity_df = result.equity
    daily = _daily_equity(equity_df)
    final_equity = float(equity_df["equity"].iloc[-1]) if not equity_df.empty else capital
    total = _total_return(capital, final_equity)
    start_ts = equity_df["datetime"].iloc[0] if not equity_df.empty else start
    end_ts = equity_df["datetime"].iloc[-1] if not equity_df.empty else end
    trips = _round_trips(result.trades)
    wins = [t for t in trips if t["pnl"] > 0]
    losses = [t for t in trips if t["pnl"] <= 0]
    gross_win = sum(t["pnl"] for t in wins) if wins else 0.0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0.0

    metrics = {
        "initial_capital": capital,
        "final_equity": round(final_equity, 2),
        "final_cash": round(result.final_cash, 2),
        "final_shares": result.final_shares,
        "total_return": round(total, 6),
        "annualized_return": round(_ann_return(total, start_ts, end_ts), 6),
        "max_drawdown": round(_max_drawdown(daily if not daily.empty else equity_df.get("equity", pd.Series(dtype=float))), 6),
        "sharpe": _to_float(_sharpe(daily)),
        "n_trades": len(result.trades),
        "n_skipped": len(result.skipped),
        "n_round_trips": len(trips),
        "win_rate": round(len(wins) / len(trips), 4) if trips else None,
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 0 else None,
        "buy_hold_return": _to_float(_buy_hold_return(bars, eval_bars)),
        "open_position": result.final_shares > 0,
    }
    keys = benchmark_keys or list(benchmarks.keys()) or ["hsi", "hstech"]
    for key in keys:
        bench_ret = benchmark_buy_hold_return(benchmarks.get(key), start, end)
        metrics[f"{key}_return"] = _to_float(bench_ret)
        if bench_ret is not None:
            metrics[f"excess_vs_{key}"] = round(total - bench_ret, 6)
            metrics[f"{key}_return"] = round(metrics[f"{key}_return"], 6)
        else:
            metrics[f"excess_vs_{key}"] = None
    if metrics["sharpe"] is not None:
        metrics["sharpe"] = round(metrics["sharpe"], 4)
    if metrics["buy_hold_return"] is not None:
        metrics["buy_hold_return"] = round(metrics["buy_hold_return"], 6)
    return metrics
