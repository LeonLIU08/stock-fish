"""开盘成交、跳空和盘中止损。滑点往对持仓不利的方向加。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

from analysis.backtest.trend.markets import trade_fees
from analysis.backtest.trend.prices import dividend_credit, raw_from_adjusted, upward_gap_cancel


@dataclass(frozen=True)
class CashFill:
    price: float
    shares: int
    notional: float
    fee: float
    cash_after: float


def slipped_buy(raw_open: float, slippage: float) -> float:
    price = _slip(raw_open, slippage, buy=True)
    return price


def slipped_sell(raw_price: float, slippage: float) -> float:
    return _slip(raw_price, slippage, buy=False)


def entry_cancel_reason(
    *,
    raw_open: float,
    factor: float,
    adjusted_stop: float,
    adjusted_signal_close: float,
    a_t: float,
    gap_cancel: float,
    max_stop_distance: float,
    price_axis: str,
) -> Optional[str]:
    """跳空、止损距离、开盘价不高于止损。比较在复权价上做，门槛冻结在信号那根。"""
    if factor <= 0:
        raise ValueError("复权因子必须为正")
    if adjusted_stop <= 0:
        return "止损无效"
    adjusted_open = float(raw_open) * float(factor)
    if adjusted_open <= float(adjusted_stop):
        return "开盘价不高于止损"
    if upward_gap_cancel(
        raw_open,
        factor,
        adjusted_signal_close,
        a_t,
        gap_cancel,
        price_axis=price_axis,
    ):
        return "跳空"
    if price_axis == "log":
        distance = math.log(adjusted_open / float(adjusted_stop))
    elif price_axis == "uniform":
        distance = adjusted_open - float(adjusted_stop)
    else:
        raise ValueError(f"价格轴必须是 log 或 uniform，收到 {price_axis}")
    if distance > float(max_stop_distance) * float(a_t):
        return "止损距离"
    return None


def apply_buy(cash: float, price: float, shares: int, market) -> CashFill:
    _require_fill(price, shares)
    notional = float(price) * int(shares)
    fee = trade_fees(market, side="buy", notional=notional, shares=shares).total
    if notional + fee > float(cash) + 1e-6:
        raise ValueError("现金不足以买入")
    return CashFill(float(price), int(shares), notional, fee, float(cash) - notional - fee)


def apply_sell(cash: float, price: float, shares: int, market) -> CashFill:
    _require_fill(price, shares)
    notional = float(price) * int(shares)
    fee = trade_fees(market, side="sell", notional=notional, shares=shares).total
    return CashFill(float(price), int(shares), notional, fee, float(cash) + notional - fee)


def raw_stop(adjusted_stop: float, factor: float) -> float:
    return raw_from_adjusted(adjusted_stop, factor)


def credit_dividend(cash: float, shares: float, dividend: float, withholding: float):
    """除息日入账。返回新现金和入账明细。"""
    credit = dividend_credit(shares, dividend, withholding)
    return float(cash) + credit.cash, credit


def apply_split(shares: int, ratio: float) -> int:
    """股数乘以拆股比例。现金和复权止损不动。"""
    if isinstance(shares, bool) or not isinstance(shares, int) or shares < 0:
        raise ValueError("股数必须是非负整数")
    if ratio <= 0:
        raise ValueError("拆股比例必须为正")
    return whole_shares(Decimal(shares) * Decimal(str(ratio)))


def whole_shares(value: Decimal) -> int:
    """浮点误差贴着整数时收成整数。差得更多仍视为拆股不成整股。"""
    if not isinstance(value, Decimal):
        value = Decimal(str(value))
    nearest = value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    if abs(value - nearest) <= Decimal("0.000001"):
        return int(nearest)
    raise ValueError(f"拆股后的股数不是整数: {value}")


def _slip(raw_price: float, slippage: float, *, buy: bool) -> float:
    if raw_price <= 0:
        raise ValueError("成交参考价必须为正")
    if slippage < 0:
        raise ValueError("滑点不能为负")
    price = float(raw_price) * (1.0 + float(slippage) if buy else 1.0 - float(slippage))
    if price <= 0:
        raise ValueError("滑点后的成交价必须为正")
    return price


def _require_fill(price: float, shares: int) -> None:
    if price <= 0:
        raise ValueError("成交价必须为正")
    if isinstance(shares, bool) or not isinstance(shares, int) or shares < 1:
        raise ValueError("成交股数必须是正整数")
