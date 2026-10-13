"""按风险预算、现金和占用上限计算股数，再按真实每手向下取整。"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR

from analysis.backtest.trend.markets import trade_fees


@dataclass(frozen=True)
class SizeResult:
    shares: int
    planned_risk: float
    one_lot: bool
    reason: str
    text: str
    q_risk: int
    q_cash: int
    q_cap: int
    open_price: float
    raw_stop: float
    cost_per_share: float


def estimate_cost_per_share(market, open_price: float, raw_stop: float, slippage_rate: float) -> float:
    """每股预估成本 c。只含按比例和按股的费率，不含最低佣金、印花税进位和平台费。"""
    return float(_cost(market, open_price, raw_stop, slippage_rate))


def size_position(
    *,
    equity: float,
    cash: float,
    open_price: float,
    raw_stop: float,
    lot_size: int,
    risk_per_trade: float,
    one_lot_max_risk: float,
    max_gross_exposure: float,
    slippage_rate: float,
    market,
) -> SizeResult:
    """三者取最小，再向下取整到整手。风险预算不够一手时，才询问一手规则。"""
    _require_lot(lot_size)
    if open_price <= 0:
        raise ValueError("开盘成交价必须为正")
    if raw_stop <= 0:
        raise ValueError("止损价必须为正")
    if equity < 0 or cash < 0:
        raise ValueError("净值和现金不能为负")
    if min(risk_per_trade, one_lot_max_risk, max_gross_exposure, slippage_rate) < 0:
        raise ValueError("风险参数不能为负")

    cost = _cost(market, open_price, raw_stop, slippage_rate)
    distance = abs(_decimal(open_price) - _decimal(raw_stop))
    per_share = distance + cost
    q_cash = _affordable_shares(cash, open_price, market)
    q_cap = _floor_ratio(_decimal(equity) * _decimal(max_gross_exposure), _decimal(open_price))
    if per_share <= 0 or equity <= 0:
        q_risk = 0
    else:
        q_risk = _floor_ratio(_decimal(equity) * _decimal(risk_per_trade), per_share)
    shares = _floor_lot(min(q_risk, q_cash, q_cap), lot_size)
    one_lot = False
    reason = ""
    if shares <= 0:
        shares, one_lot, reason = _rescue_one_lot(
            equity=equity,
            per_share=per_share,
            lot_size=lot_size,
            one_lot_max_risk=one_lot_max_risk,
            q_risk=q_risk,
            q_cash=q_cash,
            q_cap=q_cap,
        )
    planned = 0.0 if shares <= 0 or equity <= 0 else float(_decimal(shares) * per_share / _decimal(equity))
    text = _size_text(
        shares=shares,
        q_risk=q_risk,
        q_cash=q_cash,
        q_cap=q_cap,
        lot_size=lot_size,
        planned=planned,
        one_lot=one_lot,
        reason=reason,
        one_lot_max_risk=one_lot_max_risk,
    )
    return SizeResult(
        shares=shares,
        planned_risk=planned,
        one_lot=one_lot,
        reason=reason,
        text=text,
        q_risk=q_risk,
        q_cash=q_cash,
        q_cap=q_cap,
        open_price=float(open_price),
        raw_stop=float(raw_stop),
        cost_per_share=float(cost),
    )


def format_percent(fraction: float) -> str:
    text = f"{float(fraction) * 100:.4f}".rstrip("0").rstrip(".")
    return f"{text or '0'}%"


def _rescue_one_lot(
    *,
    equity: float,
    per_share: Decimal,
    lot_size: int,
    one_lot_max_risk: float,
    q_risk: int,
    q_cash: int,
    q_cap: int,
) -> tuple[int, bool, str]:
    cash_ok = q_cash >= lot_size
    cap_ok = q_cap >= lot_size
    risk_short = q_risk < lot_size
    if risk_short and cash_ok and cap_ok:
        if one_lot_max_risk == 0:
            return 0, False, "风险预算"
        limit = _decimal(equity) * _decimal(one_lot_max_risk)
        if _decimal(lot_size) * per_share <= limit:
            return lot_size, True, ""
        return 0, False, "风险预算"
    if not cash_ok:
        return 0, False, "现金"
    if not cap_ok:
        return 0, False, "占用上限"
    return 0, False, "风险预算"


def _size_text(
    *,
    shares: int,
    q_risk: int,
    q_cash: int,
    q_cap: int,
    lot_size: int,
    planned: float,
    one_lot: bool,
    reason: str,
    one_lot_max_risk: float,
) -> str:
    counts = f"风险预算 {q_risk} 股，现金 {q_cash} 股，占用上限 {q_cap} 股"
    if one_lot:
        return (
            f"风险预算不够一手，按一手规则买入，计划风险 {format_percent(planned)}。"
            f"{counts}，每手 {lot_size} 股。"
        )
    if shares > 0:
        return (
            f"{counts}，按每手 {lot_size} 股取整后 {shares} 股，"
            f"计划风险 {format_percent(planned)}。"
        )
    if reason == "风险预算":
        gate = "一手规则已关闭" if one_lot_max_risk == 0 else "一手规则也不够"
        return f"风险预算不够一手，{gate}。{counts}。"
    if reason == "现金":
        return f"现金不足，股数取整为 0。{counts}。"
    if reason == "占用上限":
        return f"占用上限把股数取成了 0。{counts}。"
    return f"股数取整为 0。{counts}。"


def _cost(market, open_price: float, raw_stop: float, slippage_rate: float) -> Decimal:
    buy_rate, buy_each, sell_rate, sell_each = _variable_rates(market)
    return (
        _decimal(open_price) * buy_rate
        + buy_each
        + _decimal(raw_stop) * (sell_rate + _decimal(slippage_rate))
        + sell_each
    )


def _variable_rates(market) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    currency = getattr(market, "currency", None)
    if currency == "HKD":
        rate = (
            _decimal(market.commission_rate)
            + _decimal(market.stamp_duty_rate)
            + _decimal(market.exchange_fee_rate)
            + _decimal(market.settlement_rate)
        )
        return rate, Decimal(0), rate, Decimal(0)
    if currency == "USD":
        each = _decimal(market.commission_per_share)
        return (
            Decimal(0),
            each,
            _decimal(market.sec_fee_rate),
            each + _decimal(market.taf_per_share),
        )
    raise ValueError(f"未知市场币种: {currency}")


def _affordable_shares(cash: float, price: float, market) -> int:
    if cash <= 0 or price <= 0:
        return 0
    lo = 0
    hi = int(_decimal(cash) // _decimal(price))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        fee = trade_fees(market, side="buy", notional=price * mid, shares=mid).total
        if price * mid + fee <= cash + 1e-6:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _floor_ratio(numerator: Decimal, denominator: Decimal) -> int:
    if denominator <= 0:
        return 0
    if numerator <= 0:
        return 0
    return int((numerator / denominator).to_integral_value(rounding=ROUND_FLOOR))


def _floor_lot(shares: int, lot_size: int) -> int:
    if shares < lot_size:
        return 0
    return (shares // lot_size) * lot_size


def _require_lot(lot_size: int) -> None:
    if isinstance(lot_size, bool) or not isinstance(lot_size, int) or lot_size < 1:
        raise ValueError("每手股数必须是正整数")


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))
