"""把账户决策写成记录、成交和报告用的统计。"""
from __future__ import annotations

import json
from typing import Optional, Sequence

import pandas as pd

from analysis.backtest.engine import EngineResult, Trade
from analysis.backtest.trend.session import AccountResult, Decision

_FILLS = {"入场", "减仓", "止损", "趋势退出"}
_CANCEL_KEYS = ("跳空", "开盘价不高于止损", "止损距离", "风险预算", "现金", "占用上限")


def audit_summary(result: AccountResult, *, exhaustion_enabled: bool) -> dict:
    cancels = {key: 0 for key in _CANCEL_KEYS}
    for row in result.decisions:
        if row.action == "取消" and row.reason in cancels:
            cancels[row.reason] += 1
    reduces = sum(1 for row in result.decisions if row.action == "减仓" or row.reason == "取整全平")
    one_lot = sum(1 for row in result.decisions if row.action == "入场" and row.one_lot)
    trips = round_trips(result.decisions)
    multiples = [item["r_multiple"] for item in trips if item["r_multiple"] is not None]
    return {
        "exhaustion_enabled": bool(exhaustion_enabled),
        "exhaustion_reduces": reduces,
        "one_lot_fills": one_lot,
        "cancels": cancels,
        "blocked": {key: value for key, value in result.blocked},
        "max_consecutive_losses": max_consecutive_losses(trips),
        "mean_r_multiple": None if not multiples else sum(multiples) / len(multiples),
        "round_trips": trips,
    }


def round_trips(decisions: Sequence[Decision]) -> list:
    trips = []
    entry: Optional[Decision] = None
    for row in decisions:
        if row.action == "入场":
            entry = row
            continue
        if entry is None or row.action not in ("减仓", "止损", "趋势退出"):
            continue
        if row.shares_after > 0:
            continue
        pnl = 0.0 if row.actual_pnl is None else float(row.actual_pnl)
        planned = entry.planned_risk
        reference = entry.reference_equity
        risk_cash = None if planned is None or reference is None else float(planned) * float(reference)
        multiple = None if not risk_cash else pnl / risk_cash
        loss_fraction = row.actual_loss_fraction or 0.0
        trips.append({
            "entry": entry.time,
            "exit": row.time,
            "shares": entry.shares_after,
            "entry_price": entry.price,
            "exit_price": row.price,
            "pnl_net": round(pnl, 2),
            "pnl_gross": round(pnl + (entry.fee or 0.0) + (row.fee or 0.0), 2),
            "return_net": None if not entry.price or not row.price else round(row.price / entry.price - 1.0, 6),
            "exit_reason": row.exit_rule or row.reason,
            "planned_risk": planned,
            "one_lot": entry.one_lot,
            "loss_exceeds_plan": bool(planned is not None and loss_fraction > float(planned)),
            "r_multiple": None if multiple is None else round(multiple, 4),
        })
        entry = None
    return trips


def max_consecutive_losses(trips: Sequence[dict]) -> int:
    streak = 0
    worst = 0
    for trip in trips:
        if float(trip.get("pnl_net") or 0.0) < 0:
            streak += 1
            worst = max(worst, streak)
        else:
            streak = 0
    return worst


def decision_records(
    result: AccountResult,
    *,
    strategy: str,
    run: str,
    scale: str,
    price_axis: str,
    currency: str,
    lot_size: int,
    audit: str,
    tape_by_index: Optional[dict] = None,
) -> list:
    rows = [_record(row, strategy, run, scale, price_axis, currency, lot_size, tape_by_index) for row in result.decisions]
    if audit != "all_bars":
        return rows
    seen = {row.index for row in result.decisions}
    for equity in result.equity:
        if equity.index in seen:
            continue
        tape = (tape_by_index or {}).get(equity.index)
        rows.append({
            "index": equity.index,
            "time": equity.time,
            "action": "未交易",
            "strategy": strategy,
            "run": run,
            "scale": scale,
            "price_axis": price_axis,
            "currency": currency,
            "structure": equity.structure,
            "category": equity.category,
            "a_t": None if tape is None else tape.a_t,
            "ema": None if tape is None else tape.ema,
            "s_t": None if tape is None else tape.s_t,
            "shares_before": equity.shares,
            "shares_after": equity.shares,
            "lot_size": lot_size,
            "text": "这根没有成交或取消。",
        })
    rows.sort(key=lambda item: (item["index"], item["action"] != "未交易"))
    return rows


def write_jsonl(path, records: Sequence[dict]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records)
    path.write_text(text, encoding="utf-8")


def to_engine_result(result: AccountResult, capital: float, *, tactic: str) -> EngineResult:
    trades = []
    skipped = []
    fees = 0.0
    for row in result.decisions:
        if row.action == "取消":
            skipped.append(_trade(row, "buy", tactic, skipped_trade=True))
            continue
        if row.action not in _FILLS or row.price is None:
            continue
        fees += row.fee or 0.0
        side = "buy" if row.action == "入场" else "sell"
        trades.append(_trade(row, side, tactic))
    frame = pd.DataFrame([
        {
            "datetime": _stamp(row.time),
            "equity": row.equity,
            "equity_gross": row.equity if row.equity_gross is None else row.equity_gross,
            "cash": row.cash,
            "shares": row.shares,
            "price": row.close,
            "position": 1 if row.shares else 0,
            "turnover": 0.0,
            "leverage": 0.0 if capital <= 0 else row.shares * row.close / capital,
            "fee": row.fee,
            "stop": row.raw_stop,
        }
        for row in result.equity
    ])
    final_price = 0.0 if result.equity == () else result.equity[-1].close
    return EngineResult(
        trades=trades,
        skipped=skipped,
        equity=frame,
        final_cash=result.final_cash,
        final_shares=result.final_shares,
        final_price=final_price,
        cumulative_fees=fees,
    )


def _stamp(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        return stamp.tz_convert("UTC")
    return stamp


def _trade(row: Decision, side: str, tactic: str, skipped_trade: bool = False) -> Trade:
    price = 0.0 if row.price is None else float(row.price)
    shares = 0 if skipped_trade else row.shares_before if side == "sell" else row.shares_after
    if side == "sell" and not skipped_trade:
        shares = row.shares_before - row.shares_after
    notional = price * shares
    equity = row.equity_after if row.equity_after is not None else row.cash_after or 0.0
    gross = equity + (row.fee or 0.0)
    return Trade(
        timestamp=_stamp(row.time),
        side=side,
        price=price,
        shares=int(shares),
        notional=notional,
        fee=0.0 if row.fee is None else float(row.fee),
        cash_after=0.0 if row.cash_after is None else float(row.cash_after),
        equity_after=float(equity),
        equity_gross_after=float(gross),
        reason=row.reason or row.action,
        skipped=skipped_trade,
        action=row.action,
        tactic=tactic,
        stop_price=row.raw_stop,
        planned_risk=row.planned_risk,
        position_fraction=row.position_fraction,
        basis=row.text,
        one_lot=row.one_lot,
        shares_after=row.shares_after,
    )


def _record(row, strategy, run, scale, price_axis, currency, lot_size, tape_by_index) -> dict:
    tape = (tape_by_index or {}).get(row.index)
    return {
        "index": row.index,
        "time": row.time,
        "execute_time": row.time if row.action in _FILLS else None,
        "structure": row.structure,
        "category": row.category,
        "scale": scale,
        "price_axis": price_axis,
        "a_t": row.a_t,
        "ema": row.ema,
        "s_t": None if tape is None else tape.s_t,
        "exit_scale": None if tape is None else getattr(tape, "exit_scale", None),
        "exit_structure": None if tape is None else getattr(tape, "exit_structure", None),
        "strategy": strategy,
        "run": run,
        "exit_rule": row.exit_rule,
        "action": row.action,
        "reason": row.reason,
        "price": row.price,
        "adjusted_stop": row.adjusted_stop,
        "raw_stop": row.raw_stop,
        "shares_before": row.shares_before,
        "shares_after": row.shares_after,
        "lot_size": lot_size,
        "position_fraction": row.position_fraction,
        "cash": row.cash_after,
        "equity": row.equity_after,
        "currency": currency,
        "planned_risk": row.planned_risk,
        "one_lot": row.one_lot,
        "actual_pnl": row.actual_pnl,
        "actual_loss_fraction": row.actual_loss_fraction,
        "fee": row.fee,
        "factor": row.factor,
        "text": row.text,
    }
