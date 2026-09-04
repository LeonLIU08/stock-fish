"""Derived series for backtest reports: NAV, drawdown, heatmap, rolling, extremes."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.backtest.config import BacktestConfig
from analysis.backtest.engine import Trade
from analysis.backtest.metrics import _ann_return, _max_drawdown, _sharpe, _to_float, _total_return

_HK_TZ = "Asia/Hong_Kong"


def to_hk_index(series_or_df_times: pd.Series) -> pd.DatetimeIndex:
    ts = pd.to_datetime(series_or_df_times)
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(_HK_TZ)
    else:
        ts = ts.dt.tz_convert(_HK_TZ)
    return pd.DatetimeIndex(ts)


def daily_frame(equity: pd.DataFrame) -> pd.DataFrame:
    if equity is None or equity.empty:
        return pd.DataFrame()
    df = equity.copy()
    df["datetime"] = to_hk_index(df["datetime"])
    df = df.set_index("datetime").sort_index()
    numeric = [c for c in df.columns if c != "datetime"]
    daily = df[numeric].resample("1D").last().dropna(how="all")
    if daily.empty:
        return daily
    daily["ret_net"] = daily["equity"].pct_change()
    if "equity_gross" in daily.columns:
        daily["ret_gross"] = daily["equity_gross"].pct_change()
    if "turnover" in df.columns:
        daily["turnover_d"] = df["turnover"].resample("1D").sum()
    else:
        daily["turnover_d"] = 0.0
    return daily


def align_benchmark_daily(bench: pd.DataFrame) -> pd.Series:
    if bench is None or bench.empty:
        return pd.Series(dtype=float)
    df = bench.copy()
    df["datetime"] = to_hk_index(df["datetime"])
    return df.set_index("datetime")["close"].astype(float).resample("1D").last().dropna()


def nav_from_price(price: pd.Series, start_value: float = 1.0) -> pd.Series:
    price = price.dropna()
    if price.empty or float(price.iloc[0]) == 0:
        return pd.Series(dtype=float)
    return start_value * price / float(price.iloc[0])


def nav_from_capital(equity: pd.Series, capital: float, start_value: float = 1.0) -> pd.Series:
    """Scale account equity by initial capital so leftover cash is visible."""
    if equity is None or equity.empty or not capital:
        return pd.Series(dtype=float)
    return start_value * equity.astype(float) / float(capital)


@dataclass
class DcaResult:
    """Evenly spaced 1-lot buys over the evaluation window."""

    equity: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_lots: int = 0
    n_trading_days: int = 0
    interval: Optional[float] = None
    n_lots_bought: int = 0
    n_buy_days: int = 0
    start_price: float = 0.0
    lot_size: int = 1
    total_return: float = 0.0
    final_equity: float = 0.0
    final_cash: float = 0.0
    final_shares: int = 0


def _session_daily_ohlc(bars: pd.DataFrame, tz: str = _HK_TZ) -> pd.DataFrame:
    """One row per exchange session: first open, last close, last timestamp."""
    if bars is None or bars.empty:
        return pd.DataFrame(columns=["open", "close", "datetime"])
    work = bars.loc[bars["open"].notna() & bars["close"].notna(), ["datetime", "open", "close"]].copy()
    if work.empty:
        return pd.DataFrame(columns=["open", "close", "datetime"])
    work["datetime"] = to_hk_index(work["datetime"])
    work["_day"] = work["datetime"].dt.normalize()
    return work.groupby("_day", sort=True).agg(
        open=("open", "first"),
        close=("close", "last"),
        datetime=("datetime", "last"),
    )


def _dca_buy_lots_by_day(n_lots: int, n_days: int) -> List[int]:
    """Spread N 1-lot buys from day 0 at step = n_days / N."""
    lots = [0] * n_days
    if n_lots <= 0 or n_days <= 0:
        return lots
    interval = n_days / n_lots
    for i in range(n_lots):
        idx = int(i * interval)
        if idx >= n_days:
            idx = n_days - 1
        lots[idx] += 1
    return lots


def simulate_even_dca(
    bars: pd.DataFrame,
    config: BacktestConfig,
    symbol: str,
) -> DcaResult:
    """Dollar-cost average 1 lot every (trading_days / N) sessions.

    N is how many lots ``capital`` can buy at the first session's close
    (lot size + the same commission / stamp model as the strategy).
    Buys fill at that session's open; leftover cash stays in the account.
    """
    capital = float(config.capital)
    lot_size = int(config.lot_size_for(symbol))
    empty = DcaResult(lot_size=lot_size, final_equity=capital, final_cash=capital)
    daily = _session_daily_ohlc(bars, config.timezone or _HK_TZ)
    if daily.empty:
        return empty

    start_price = float(daily["close"].iloc[0])
    n_days = len(daily)
    empty = DcaResult(
        n_trading_days=n_days,
        start_price=start_price,
        lot_size=lot_size,
        final_equity=capital,
        final_cash=capital,
    )
    if start_price <= 0:
        return empty

    cost = config.cost
    max_shares = cost.max_shares(capital, start_price, lot_size=lot_size)
    n_lots = (max_shares // lot_size) if lot_size > 0 else 0
    interval = (n_days / n_lots) if n_lots > 0 else None
    buy_lots = _dca_buy_lots_by_day(n_lots, n_days)

    cash = capital
    cash_gross = capital
    shares = 0
    n_lots_bought = 0
    n_buy_days = 0
    rows: List[Dict[str, Any]] = []

    for i, (_, row) in enumerate(daily.iterrows()):
        o = float(row["open"])
        c = float(row["close"])
        fee_bar = 0.0
        turnover = 0.0
        lots = buy_lots[i] if i < len(buy_lots) else 0
        if lots > 0 and o > 0 and cash > 0:
            qty = min(lots * lot_size, cost.max_shares(cash, o, lot_size=lot_size))
            if lot_size > 1:
                qty = (qty // lot_size) * lot_size
            if qty > 0:
                notional = qty * o
                fee = cost.fee(notional)
                cash -= notional + fee
                cash_gross -= notional
                shares += qty
                fee_bar = fee
                turnover = notional / capital if capital else 0.0
                n_lots_bought += qty // lot_size
                n_buy_days += 1
        mv = shares * c
        rows.append({
            "datetime": row["datetime"],
            "equity": cash + mv,
            "equity_gross": cash_gross + mv,
            "cash": cash,
            "shares": shares,
            "price": c,
            "position": 1 if shares > 0 else 0,
            "turnover": turnover,
            "leverage": (mv / capital) if capital else 0.0,
            "fee": fee_bar,
        })

    equity = pd.DataFrame(rows)
    final_equity = float(equity["equity"].iloc[-1]) if not equity.empty else capital
    return DcaResult(
        equity=equity,
        n_lots=int(n_lots),
        n_trading_days=n_days,
        interval=float(interval) if interval is not None else None,
        n_lots_bought=int(n_lots_bought),
        n_buy_days=int(n_buy_days),
        start_price=start_price,
        lot_size=lot_size,
        total_return=_total_return(capital, final_equity),
        final_equity=final_equity,
        final_cash=float(cash),
        final_shares=int(shares),
    )


def underwater(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return equity
    peak = equity.cummax()
    return equity / peak - 1.0


def monthly_return_table(daily_ret: pd.Series) -> pd.DataFrame:
    if daily_ret is None or daily_ret.dropna().empty:
        return pd.DataFrame()
    r = daily_ret.dropna()
    monthly = (1.0 + r).resample("ME").prod() - 1.0
    if monthly.empty:
        return pd.DataFrame()
    frame = monthly.to_frame("ret")
    frame["year"] = frame.index.year
    frame["month"] = frame.index.month
    pivot = frame.pivot(index="year", columns="month", values="ret")
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = np.nan
    pivot = pivot.reindex(columns=list(range(1, 13)))
    pivot["全年"] = (1.0 + pivot).prod(axis=1, min_count=1) - 1.0
    return pivot


def rolling_stats(
    strat_ret: pd.Series,
    bench_returns: Dict[str, pd.Series],
    window: int,
) -> pd.DataFrame:
    s = strat_ret.dropna().rename("strat")
    out = pd.DataFrame(index=s.index)
    min_p = max(window // 2, 20)
    out["roll_return"] = s.rolling(window, min_periods=min_p).apply(
        lambda x: float((1.0 + x).prod() - 1.0), raw=True
    )
    roll_mean = s.rolling(window, min_periods=min_p).mean()
    roll_std = s.rolling(window, min_periods=min_p).std(ddof=1)
    out["roll_sharpe"] = (roll_mean / roll_std.replace(0, np.nan)) * np.sqrt(252)
    for key, bench_ret in (bench_returns or {}).items():
        if bench_ret is None or bench_ret.empty:
            continue
        aligned = pd.concat([s, bench_ret.rename("bench")], axis=1).dropna()
        if aligned.empty:
            continue
        cov = aligned["strat"].rolling(window, min_periods=min_p).cov(aligned["bench"])
        var = aligned["bench"].rolling(window, min_periods=min_p).var()
        out[f"roll_beta_{key}"] = (cov / var.replace(0, np.nan)).reindex(out.index)
    return out


def factor_exposures(daily: pd.DataFrame) -> pd.DataFrame:
    """Proxy exposures available without a style-factor library."""
    if daily.empty:
        return daily
    out = pd.DataFrame(index=daily.index)
    out["position"] = daily["position"] if "position" in daily.columns else 0
    if "ma_fast" in daily.columns and "ma_slow" in daily.columns:
        slow = daily["ma_slow"].replace(0, np.nan)
        out["trend_gap"] = (daily["ma_fast"] - daily["ma_slow"]) / slow
    if "ret_net" in daily.columns:
        out["realized_vol_20"] = daily["ret_net"].rolling(20, min_periods=10).std(ddof=1) * np.sqrt(252)
    out["leverage"] = daily["leverage"] if "leverage" in daily.columns else 0
    return out


def _metrics_from_equity(
    equity: pd.Series,
    trades: List[Trade],
    start: date,
    end: date,
) -> Dict[str, Any]:
    empty = {
        "total_return": None,
        "annualized_return": None,
        "max_drawdown": None,
        "sharpe": None,
        "n_trades": 0,
        "final_equity": None,
        "start_equity": None,
    }
    if equity is None or equity.empty:
        return empty
    idx_dates = equity.index.tz_convert(_HK_TZ).date if equity.index.tz is not None else equity.index.date
    window = equity.loc[(idx_dates >= start) & (idx_dates <= end)]
    if window.empty:
        return empty
    first = float(window.iloc[0])
    last = float(window.iloc[-1])
    total = _total_return(first, last)
    n_trades = 0
    for t in trades:
        if t.skipped:
            continue
        ts = pd.Timestamp(t.timestamp)
        d = ts.tz_convert(_HK_TZ).date() if ts.tzinfo is not None else ts.date()
        if start <= d <= end:
            n_trades += 1
    sharpe = _to_float(_sharpe(window))
    return {
        "total_return": round(total, 6),
        "annualized_return": round(_ann_return(total, window.index[0], window.index[-1]), 6),
        "max_drawdown": round(_max_drawdown(window), 6),
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
        "n_trades": n_trades,
        "final_equity": round(last, 2),
        "start_equity": round(first, 2),
    }


def split_cost_table(
    daily: pd.DataFrame,
    trades: List[Trade],
    config: BacktestConfig,
) -> Dict[str, Any]:
    oos = config.resolved_oos_start()
    is_end_date = oos - timedelta(days=1)
    net = daily["equity"] if "equity" in daily.columns else pd.Series(dtype=float)
    gross = daily["equity_gross"] if "equity_gross" in daily.columns else net
    return {
        "full_post_cost": _metrics_from_equity(net, trades, config.start, config.end),
        "full_pre_cost": _metrics_from_equity(gross, trades, config.start, config.end),
        "is_post_cost": _metrics_from_equity(net, trades, config.start, is_end_date),
        "is_pre_cost": _metrics_from_equity(gross, trades, config.start, is_end_date),
        "oos_post_cost": _metrics_from_equity(net, trades, oos, config.end),
        "oos_pre_cost": _metrics_from_equity(gross, trades, oos, config.end),
        "oos_start": oos.isoformat(),
        "is_end": is_end_date.isoformat(),
    }


def extreme_days(
    daily: pd.DataFrame,
    trades: List[Trade],
    n: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    if daily.empty or "ret_net" not in daily.columns:
        return {"best": [], "worst": []}
    rets = daily["ret_net"].dropna()
    if rets.empty:
        return {"best": [], "worst": []}

    trade_days: Dict[str, List[Dict[str, Any]]] = {}
    for t in trades:
        if t.skipped:
            continue
        ts = pd.Timestamp(t.timestamp)
        key = ts.tz_convert(_HK_TZ).strftime("%Y-%m-%d") if ts.tzinfo else ts.strftime("%Y-%m-%d")
        trade_days.setdefault(key, []).append({
            "side": t.side,
            "price": t.price,
            "shares": t.shares,
            "fee": t.fee,
            "reason": t.reason,
        })

    def pack(series: pd.Series) -> List[Dict[str, Any]]:
        rows = []
        for ts, val in series.items():
            pts = pd.Timestamp(ts)
            day = pts.tz_convert(_HK_TZ).strftime("%Y-%m-%d") if pts.tzinfo else pts.strftime("%Y-%m-%d")
            rows.append({
                "date": day,
                "return": round(float(val), 6),
                "equity": round(float(daily.loc[ts, "equity"]), 2) if ts in daily.index else None,
                "position": int(daily.loc[ts, "position"]) if ts in daily.index and "position" in daily.columns else None,
                "trades": trade_days.get(day, []),
            })
        return rows

    return {
        "best": pack(rets.nlargest(min(n, len(rets)))),
        "worst": pack(rets.nsmallest(min(n, len(rets)))),
    }


def round_trips(trades: List[Trade]) -> List[Dict[str, Any]]:
    trips = []
    open_trade = None
    for t in trades:
        if t.skipped:
            continue
        if t.side == "buy":
            open_trade = t
        elif t.side == "sell" and open_trade is not None:
            pnl_net = (t.price - open_trade.price) * t.shares - t.fee - open_trade.fee
            pnl_gross = (t.price - open_trade.price) * t.shares
            denom = open_trade.price * t.shares
            trips.append({
                "entry": str(open_trade.timestamp),
                "exit": str(t.timestamp),
                "shares": t.shares,
                "entry_price": open_trade.price,
                "exit_price": t.price,
                "pnl_net": round(pnl_net, 2),
                "pnl_gross": round(pnl_gross, 2),
                "return_net": round(pnl_net / denom, 6) if denom else None,
            })
            open_trade = None
    return trips


def data_gaps(config: BacktestConfig, daily: pd.DataFrame) -> List[str]:
    gaps = [
        "无港股 Barra/风格因子库（市值、价值、质量、流动性），风格暴露用趋势缺口、仓位、实现波动和基准 Beta 代替。",
        "分股独立核算，持仓数量仅为 0/1，不是组合层面的持股只数。",
        "只做多全仓、无融资，杠杆时间序列约为 0 或 1。",
        f"信号仅支持 {config.signal_on}，成交仅支持 {config.fill_on}。",
    ]
    if daily.empty or len(daily.dropna(subset=["equity"])) < config.rolling_window_days:
        gaps.append(
            f"样本短于 {config.rolling_window_days} 个交易日，滚动 12 个月收益/Sharpe/Beta 有效点很少。"
        )
    return gaps
