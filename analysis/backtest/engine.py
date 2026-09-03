"""Long-only bar simulator driven by standard signal columns."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from analysis.backtest.config import BacktestConfig, CostModel
from analysis.backtest.signals import (
    SIGNAL_ACTION_COL,
    SIGNAL_REASON_COL,
    SIGNAL_SIZE_COL,
    SignalAction,
    ensure_signal_columns,
    validate_signal_columns,
)

_HK_TZ = "Asia/Hong_Kong"


def _opt_float(val):
    try:
        if val is None:
            return None
        num = float(val)
        if num != num:
            return None
        return num
    except (TypeError, ValueError):
        return None


def _normalize_fraction(size: Optional[float]) -> Optional[float]:
    if size is None:
        return None
    try:
        value = float(size)
    except (TypeError, ValueError):
        return None
    if value != value:
        return None
    return min(max(value, 0.0), 1.0)


@dataclass
class Trade:
    timestamp: pd.Timestamp
    side: str
    price: float
    shares: int
    notional: float
    fee: float
    cash_after: float
    equity_after: float
    equity_gross_after: float
    reason: str
    skipped: bool = False


@dataclass
class EngineResult:
    trades: List[Trade] = field(default_factory=list)
    skipped: List[Trade] = field(default_factory=list)
    equity: pd.DataFrame = field(default_factory=pd.DataFrame)
    final_cash: float = 0.0
    final_shares: int = 0
    final_price: float = 0.0
    cumulative_fees: float = 0.0


def _session_date(ts: pd.Timestamp, tz: str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize(tz)
    else:
        t = t.tz_convert(tz)
    return t.normalize()


def _resolve_buy_qty(
    cash: float,
    price: float,
    *,
    cost: CostModel,
    lot_size: int,
    position_mode: str,
    signal_size: Optional[float],
) -> int:
    if cash <= 0 or price <= 0:
        return 0
    budget = cash
    if position_mode == "fraction":
        fraction = _normalize_fraction(signal_size)
        if fraction is not None and fraction > 0:
            budget = cash * fraction
    return cost.max_shares(budget, price, lot_size=lot_size)


def _resolve_sell_qty(
    shares: int,
    *,
    position_mode: str,
    signal_size: Optional[float],
) -> int:
    if shares <= 0:
        return 0
    if position_mode == "fraction":
        fraction = _normalize_fraction(signal_size)
        if fraction is not None and 0 < fraction < 1:
            qty = int(shares * fraction)
            return max(qty, 0)
    return shares


def simulate(
    df: pd.DataFrame,
    config: BacktestConfig,
    symbol: Optional[str] = None,
    overlay_columns: Optional[Sequence[str]] = None,
) -> EngineResult:
    """Replay bars. Signal on bar i close → fill at bar i+1 open.

    Consumes ``signal_action`` / optional ``signal_size`` / ``signal_reason``.
    """
    if not config.long_only:
        raise ValueError("当前回测引擎只支持做多")
    empty_eq = pd.DataFrame(columns=[
        "datetime", "equity", "equity_gross", "cash", "shares", "price",
        "position", "turnover", "leverage", "fee",
    ])
    if df is None or df.empty or len(df) < 2:
        return EngineResult(equity=empty_eq, final_cash=config.capital)

    work = ensure_signal_columns(df)
    validate_signal_columns(work)

    cost: CostModel = config.cost
    lot_size = config.lot_size_for(symbol) if symbol else cost.lot_size
    cash = float(config.capital)
    cash_gross = float(config.capital)
    shares = 0
    trades: List[Trade] = []
    skipped: List[Trade] = []
    equity_rows = []
    fills_on_day: dict = {}
    tz = config.timezone or _HK_TZ
    capital = float(config.capital)
    cumulative_fees = 0.0
    overlays = [col for col in (overlay_columns or []) if col in work.columns]

    opens = work["open"].astype(float).tolist()
    closes = work["close"].astype(float).tolist()
    times = work["datetime"].tolist()
    actions = [SignalAction.normalize(a) for a in work[SIGNAL_ACTION_COL].tolist()]
    reasons = work[SIGNAL_REASON_COL].fillna("").astype(str).tolist()
    sizes = work[SIGNAL_SIZE_COL].tolist() if SIGNAL_SIZE_COL in work.columns else [np.nan] * len(work)
    n = len(work)

    pending_side: Optional[str] = None
    pending_reason = ""
    pending_size: Optional[float] = None

    def day_key(ts) -> str:
        return _session_date(ts, tz).strftime("%Y-%m-%d")

    def can_fill(ts) -> bool:
        return fills_on_day.get(day_key(ts), 0) < config.max_trades_per_day

    def mark_fill(ts) -> None:
        k = day_key(ts)
        fills_on_day[k] = fills_on_day.get(k, 0) + 1

    def snapshot(ts, side, price, qty, fee, reason, skipped_trade=False) -> Trade:
        notional = price * qty
        return Trade(
            timestamp=pd.Timestamp(ts),
            side=side,
            price=float(price),
            shares=int(qty),
            notional=float(notional),
            fee=float(fee),
            cash_after=float(cash),
            equity_after=float(cash + shares * price),
            equity_gross_after=float(cash_gross + shares * price),
            reason=reason,
            skipped=skipped_trade,
        )

    for i in range(n):
        ts = times[i]
        o = opens[i]
        c = closes[i]
        fee_bar = 0.0
        turnover = 0.0

        if pending_side == "buy":
            if can_fill(ts) and shares == 0 and o > 0:
                qty = _resolve_buy_qty(
                    cash,
                    o,
                    cost=cost,
                    lot_size=lot_size,
                    position_mode=config.position_mode,
                    signal_size=pending_size,
                )
                if qty > 0:
                    notional = qty * o
                    fee = cost.fee(notional)
                    cash -= notional + fee
                    cash_gross -= notional
                    shares = qty
                    fee_bar = fee
                    turnover = notional / capital
                    cumulative_fees += fee
                    mark_fill(ts)
                    trades.append(snapshot(ts, "buy", o, qty, fee, pending_reason))
                else:
                    skipped.append(snapshot(ts, "buy", o, 0, 0.0, "资金不足", True))
            else:
                if shares > 0:
                    why = "已持仓"
                elif not can_fill(ts):
                    why = "当日交易次数已达上限"
                else:
                    why = "无效开盘价"
                skipped.append(snapshot(ts, "buy", o, 0, 0.0, why, True))
            pending_side = None
            pending_reason = ""
            pending_size = None
        elif pending_side == "sell":
            if can_fill(ts) and shares > 0 and o > 0:
                qty = _resolve_sell_qty(
                    shares,
                    position_mode=config.position_mode,
                    signal_size=pending_size,
                )
                if qty <= 0:
                    skipped.append(snapshot(ts, "sell", o, 0, 0.0, "可卖数量为 0", True))
                else:
                    notional = qty * o
                    fee = cost.fee(notional)
                    cash += notional - fee
                    cash_gross += notional
                    shares -= qty
                    fee_bar = fee
                    turnover = notional / capital
                    cumulative_fees += fee
                    mark_fill(ts)
                    trades.append(snapshot(ts, "sell", o, qty, fee, pending_reason))
            else:
                if shares <= 0:
                    why = "空仓"
                elif not can_fill(ts):
                    why = "当日交易次数已达上限"
                else:
                    why = "无效开盘价"
                skipped.append(snapshot(ts, "sell", o, 0, 0.0, why, True))
            pending_side = None
            pending_reason = ""
            pending_size = None

        if i < n - 1 and pending_side is None:
            action = actions[i]
            reason = reasons[i] if i < len(reasons) else ""
            size = sizes[i] if i < len(sizes) else None
            if action == SignalAction.BUY and shares == 0:
                pending_side = "buy"
                pending_reason = reason or "buy"
                pending_size = size
            elif action == SignalAction.SELL and shares > 0:
                pending_side = "sell"
                pending_reason = reason or "sell"
                pending_size = size

        mv = shares * c
        row = {
            "datetime": ts,
            "equity": cash + mv,
            "equity_gross": cash_gross + mv,
            "cash": cash,
            "shares": shares,
            "price": c,
            "position": 1 if shares > 0 else 0,
            "turnover": turnover,
            "leverage": (mv / capital) if capital else 0.0,
            "fee": fee_bar,
        }
        for col in overlays:
            row[col] = _opt_float(work[col].iloc[i])
        equity_rows.append(row)

    return EngineResult(
        trades=trades,
        skipped=skipped,
        equity=pd.DataFrame(equity_rows),
        final_cash=cash,
        final_shares=shares,
        final_price=float(closes[-1]) if closes else 0.0,
        cumulative_fees=cumulative_fees,
    )
