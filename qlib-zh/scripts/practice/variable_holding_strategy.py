"""把组合目标变成可执行的买卖意图。

失败买单的权重留在现金。卖不掉的旧仓继续占槽位，不提前买替代股票。
Qlib 适配器只在真正回测时导入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass
class ExecutionPlan:
    decision_status: str
    target_weights: dict[str, float]
    sell_instruments: list[str]
    buy_instruments: list[str]
    blocked_sells: list[str]
    blocked_buys: list[str]
    dropped_buy_weight: float
    orders: list[dict[str, Any]] = field(default_factory=list)

    @property
    def target_gross_exposure(self) -> float:
        return float(sum(weight for weight in self.target_weights.values() if weight > 0))

    @property
    def target_cash_weight(self) -> float:
        return 1.0 - self.target_gross_exposure


def lot_size_for_instrument(instrument: str, lot_sizes: Mapping[str, int] | None = None) -> int:
    code = str(instrument)
    for prefix in ("SH", "SZ", "BJ", "HK"):
        if code.startswith(prefix):
            code = code[len(prefix):]
    if lot_sizes and instrument in lot_sizes:
        return int(lot_sizes[instrument])
    if lot_sizes and code in lot_sizes:
        return int(lot_sizes[code])
    if code.startswith("688"):
        return 200
    return 100


def lot_affordable(price: float, lot: int, budget: float, fee_rate: float) -> bool:
    if price <= 0 or lot <= 0 or budget <= 0:
        return False
    return float(price) * int(lot) * (1.0 + float(fee_rate)) <= float(budget) + 1e-8


def affordable_lots(price: float, lot: int, budget: float, fee_rate: float) -> int:
    if not lot_affordable(price, lot, budget, fee_rate):
        return 0
    unit = float(price) * int(lot) * (1.0 + float(fee_rate))
    return int(float(budget) // unit)


def finalize_signal_weights(weights: Mapping[str, float], *, preserve_cash: bool) -> dict[str, float]:
    """保留调用方给出的权重和。preserve_cash 时不做 0.99 归一化。"""
    clean = {
        str(instrument): float(weight)
        for instrument, weight in weights.items()
        if weight is not None and float(weight) > 0
    }
    total = float(sum(clean.values()))
    if total <= 0:
        return {}
    if preserve_cash:
        return clean
    if total < 0.99:
        return clean
    return {instrument: weight / total for instrument, weight in clean.items()}


def resolve_day_targets(
    signal_weights: Mapping[str, float] | None,
    decision_status: str,
    current_weights: Mapping[str, float] | None,
) -> dict[str, float]:
    current = {
        str(instrument): float(weight)
        for instrument, weight in (current_weights or {}).items()
        if float(weight) > 0
    }
    if decision_status in {"DATA_ERROR", "NO_REBALANCE"}:
        return current
    if decision_status == "VALID_EMPTY":
        return {}
    return finalize_signal_weights(signal_weights or {}, preserve_cash=True)


def plan_execution(
    *,
    decision_status: str,
    desired_weights: Mapping[str, float],
    current_weights: Mapping[str, float],
    tradable_buy: Mapping[str, bool] | None = None,
    tradable_sell: Mapping[str, bool] | None = None,
    max_holdings: int | None = None,
    prices: Mapping[str, float] | None = None,
    equity: float | None = None,
    fee_rate: float = 0.0,
    lot_sizes: Mapping[str, int] | None = None,
) -> ExecutionPlan:
    tradable_buy = tradable_buy or {}
    tradable_sell = tradable_sell or {}
    prices = prices or {}
    desired = resolve_day_targets(desired_weights, decision_status, current_weights)
    current = {
        str(instrument): float(weight)
        for instrument, weight in current_weights.items()
        if float(weight) > 0
    }
    if decision_status in {"DATA_ERROR", "NO_REBALANCE"}:
        return ExecutionPlan(
            decision_status=decision_status,
            target_weights=dict(current),
            sell_instruments=[],
            buy_instruments=[],
            blocked_sells=[],
            blocked_buys=[],
            dropped_buy_weight=0.0,
            orders=[{"instrument": code, "side": "hold", "target_weight": weight} for code, weight in current.items()],
        )

    held = list(current)
    desired_positive = {code: weight for code, weight in desired.items() if weight > 0}
    planned_sell = [code for code in held if code not in desired_positive]
    blocked_sells = [code for code in planned_sell if not tradable_sell.get(code, True)]
    successful_sells = [code for code in planned_sell if tradable_sell.get(code, True)]
    occupied = set(held) - set(successful_sells)
    if max_holdings is None:
        free_slots = len(desired_positive)
    else:
        free_slots = max(int(max_holdings) - len(occupied), 0)
        if len(occupied) > int(max_holdings):
            free_slots = 0
    new_buys = [code for code, _weight in desired_positive.items() if code not in current]
    allowed: list[str] = []
    blocked_buys: list[str] = []
    dropped = 0.0
    for code in new_buys:
        weight = float(desired_positive[code])
        can_trade = tradable_buy.get(code, True)
        can_afford = True
        if equity is not None and code in prices:
            budget = float(equity) * weight
            can_afford = lot_affordable(
                float(prices[code]),
                lot_size_for_instrument(code, lot_sizes),
                budget,
                fee_rate,
            )
        if not can_trade or not can_afford or len(allowed) >= free_slots:
            blocked_buys.append(code)
            dropped += weight
            continue
        allowed.append(code)
    kept = {
        code: float(weight)
        for code, weight in desired_positive.items()
        if code in current or code in allowed
    }
    orders: list[dict[str, Any]] = []
    for code in successful_sells:
        orders.append({"instrument": code, "side": "sell", "target_weight": 0.0, "status": "submitted"})
    for code in blocked_sells:
        orders.append({"instrument": code, "side": "sell", "target_weight": 0.0, "status": "BLOCKED_SELL"})
    for code in allowed:
        orders.append({"instrument": code, "side": "buy", "target_weight": kept[code], "status": "submitted"})
    for code in blocked_buys:
        orders.append(
            {
                "instrument": code,
                "side": "buy",
                "target_weight": 0.0,
                "status": "BLOCKED_BUY",
                "released_weight": float(desired_positive.get(code, 0.0)),
            }
        )
    for code in kept:
        if code in current and code not in allowed:
            orders.append({"instrument": code, "side": "hold", "target_weight": kept[code], "status": "NORMAL"})
    return ExecutionPlan(
        decision_status=decision_status,
        target_weights=kept,
        sell_instruments=successful_sells,
        buy_instruments=allowed,
        blocked_sells=blocked_sells,
        blocked_buys=blocked_buys,
        dropped_buy_weight=dropped,
        orders=orders,
    )


def create_qlib_weight_strategy(
    signal,
    *,
    liquidate_dates: set | None = None,
    freeze_dates: set | None = None,
    max_holdings: int | None = None,
    decision_status_by_date: Mapping | None = None,
):
    """构造不重分配失败买单、风险系数固定为 1 的 Qlib 策略。"""
    import pandas as pd
    from qlib.contrib.strategy.order_generator import OrderGenWOInteract
    from qlib.contrib.strategy.signal_strategy import WeightStrategyBase

    liquidate_dates = {pd.Timestamp(value) for value in (liquidate_dates or set())}
    freeze_dates = {pd.Timestamp(value) for value in (freeze_dates or set())}
    status_by_date = {
        pd.Timestamp(key): value for key, value in (decision_status_by_date or {}).items()
    }

    class _CashPreservingOrderGen(OrderGenWOInteract):
        def generate_order_list_from_target_weight_position(
            self,
            current,
            trade_exchange,
            target_weight_position,
            risk_degree,
            pred_start_time,
            pred_end_time,
            trade_start_time,
            trade_end_time,
        ):
            current_weights = _position_weights(current)
            status = status_by_date.get(pd.Timestamp(trade_start_time), "VALID")
            if pd.Timestamp(trade_start_time) in liquidate_dates:
                status = "VALID_EMPTY"
            if pd.Timestamp(trade_start_time) in freeze_dates:
                status = "DATA_ERROR"
            names = set(current_weights) | set(target_weight_position or {})
            tradable = {}
            for stock_id in names:
                tradable[str(stock_id)] = bool(
                    trade_exchange.is_stock_tradable(
                        stock_id=stock_id,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                    )
                )
            execution = plan_execution(
                decision_status=status,
                desired_weights=target_weight_position or {},
                current_weights=current_weights,
                tradable_buy=tradable,
                tradable_sell=tradable,
                max_holdings=max_holdings,
            )
            if not execution.target_weights and not current_weights:
                return []
            return super().generate_order_list_from_target_weight_position(
                current=current,
                trade_exchange=trade_exchange,
                target_weight_position=execution.target_weights,
                risk_degree=1.0,
                pred_start_time=pred_start_time,
                pred_end_time=pred_end_time,
                trade_start_time=trade_start_time,
                trade_end_time=trade_end_time,
            )

    class _CashPreservingStrategy(WeightStrategyBase):
        def __init__(self, **kwargs):
            super().__init__(order_generator_cls_or_obj=_CashPreservingOrderGen(), **kwargs)

        def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
            trade_day = pd.Timestamp(trade_start_time)
            current_weights = _position_weights(current)
            status = status_by_date.get(trade_day, "VALID")
            if trade_day in freeze_dates:
                status = "DATA_ERROR"
            if trade_day in liquidate_dates:
                status = "VALID_EMPTY"
            signal_weights = _score_weights(score)
            if not signal_weights and status == "VALID":
                status = "NO_REBALANCE"
            return resolve_day_targets(signal_weights, status, current_weights)

    return _CashPreservingStrategy(signal=signal, risk_degree=1.0)


def _position_weights(current) -> dict[str, float]:
    if current is None:
        return {}
    stocks = list(current.get_stock_list()) if hasattr(current, "get_stock_list") else []
    weights: dict[str, float] = {}
    for stock in stocks:
        weight = None
        if hasattr(current, "get_stock_weight"):
            try:
                weight = current.get_stock_weight(stock)
            except Exception:
                weight = None
        if weight is None and hasattr(current, "get_stock_amount") and hasattr(current, "calculate_value"):
            try:
                price = current.get_stock_price(stock) if hasattr(current, "get_stock_price") else None
                amount = current.get_stock_amount(code=stock)
                equity = current.calculate_value()
                if price and equity:
                    weight = float(price) * float(amount) / float(equity)
            except Exception:
                weight = None
        if weight is not None and float(weight) > 0:
            weights[str(stock)] = float(weight)
        elif weight is None:
            weights[str(stock)] = 0.0
    positive = {key: value for key, value in weights.items() if value > 0}
    if positive:
        return positive
    if weights:
        share = 1.0 / len(weights)
        return {key: share for key in weights}
    return {}


def _score_weights(score) -> dict[str, float]:
    import pandas as pd

    if score is None:
        return {}
    if isinstance(score, pd.Series):
        series = pd.to_numeric(score, errors="coerce")
    else:
        frame = score.copy()
        column = "weight" if "weight" in frame.columns else frame.columns[0]
        series = pd.to_numeric(frame[column], errors="coerce")
    series = series.replace([float("inf"), float("-inf")], pd.NA).dropna()
    series = series[series > 0]
    return {str(key): float(value) for key, value in series.items()}
