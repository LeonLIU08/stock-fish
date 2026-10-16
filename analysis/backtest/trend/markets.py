"""港股、美股代码识别，以及成交时按市场收取的费用。

按比例、按股的费率在这里算成一笔委托的实际费用。最低佣金、印花税进位、
只在卖出收取的监管费，都在成交时计算，不进仓位公式里的每股成本估算。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING

HK = "hk"
US = "us"

_US_BODY = re.compile(r"[A-Z]{1,5}(?:\.[A-Z])?")


@dataclass(frozen=True)
class ParsedSymbol:
    code: str
    market: str

    @property
    def currency(self) -> str:
        if self.market == HK:
            return "HKD"
        return "USD"


def parse_symbol(raw: str) -> ParsedSymbol:
    """港股收成 5 位数字，美股收成大写代码。6 位数字视为 A 股并拒绝。"""
    text = str(raw or "").strip().upper()
    if not text:
        raise ValueError("股票代码为空")
    if text.endswith((".SH", ".SZ", ".SS", ".BJ")):
        raise ValueError(f"{raw} 看起来是 A 股代码，第一期只做港股和美股")

    market = None
    body = text
    if text.endswith(".HK"):
        market = HK
        body = text[:-3]
    elif text.endswith(".US"):
        market = US
        body = text[:-3]

    if market == HK or (market is None and body.isdigit()):
        if not body.isdigit():
            raise ValueError(f"无法识别的港股代码: {raw}")
        if len(body) == 6:
            raise ValueError(f"{body} 是 6 位数字，第一期不做 A 股")
        if not 1 <= len(body) <= 5:
            raise ValueError(f"无法识别的港股代码: {raw}")
        return ParsedSymbol(body.zfill(5), HK)

    if not _US_BODY.fullmatch(body):
        raise ValueError(f"无法识别的代码: {raw}")
    return ParsedSymbol(body, US)


@dataclass(frozen=True)
class FeeBreakdown:
    """一笔成交的实际费用。某一市场不收的项目为 0。"""

    commission: float
    platform_fee: float
    stamp_duty: float
    exchange_fee: float
    settlement_fee: float
    sec_fee: float
    taf: float

    @property
    def total(self) -> float:
        return (
            self.commission
            + self.platform_fee
            + self.stamp_duty
            + self.exchange_fee
            + self.settlement_fee
            + self.sec_fee
            + self.taf
        )


def trade_fees(market, *, side: str, notional: float, shares: float) -> FeeBreakdown:
    """按市场计算一笔买入或卖出的费用。

    港股买卖都收佣金、最低佣金、平台费、进位后的印花税、交易所费用和交收费。
    美股佣金按股，有最低和成交额百分比上限；百分比上限更低时按上限收。
    SEC 费和 FINRA 交易活动费只在卖出时收。美股不收印花税。
    """
    if side not in ("buy", "sell"):
        raise ValueError(f"side 必须是 buy 或 sell，收到 {side}")
    _non_negative(notional, "成交额")
    _non_negative(shares, "股数")
    currency = getattr(market, "currency", None)
    if currency == "HKD":
        return _hk_fees(market, notional)
    if currency == "USD":
        return _us_fees(market, side=side, notional=notional, shares=shares)
    raise ValueError(f"未知市场币种: {currency}")


def _hk_fees(market, notional: float) -> FeeBreakdown:
    amount = _decimal(notional)
    commission = max(amount * _decimal(market.commission_rate), _decimal(market.min_commission))
    return FeeBreakdown(
        commission=_money(commission),
        platform_fee=_money(_decimal(market.platform_fee)),
        stamp_duty=_money(_ceil_dollar(amount * _decimal(market.stamp_duty_rate))),
        exchange_fee=_money(amount * _decimal(market.exchange_fee_rate)),
        settlement_fee=_money(amount * _decimal(market.settlement_rate)),
        sec_fee=0.0,
        taf=0.0,
    )


def _us_fees(market, *, side: str, notional: float, shares: float) -> FeeBreakdown:
    amount = _decimal(notional)
    per_share = _decimal(market.commission_per_share) * _decimal(shares)
    commission = max(per_share, _decimal(market.min_commission))
    commission = min(commission, amount * _decimal(market.max_commission_rate))
    if side == "buy":
        sec = Decimal(0)
        taf = Decimal(0)
    else:
        sec = amount * _decimal(market.sec_fee_rate)
        taf = min(_decimal(market.taf_per_share) * _decimal(shares), _decimal(market.taf_max))
    return FeeBreakdown(
        commission=_money(commission),
        platform_fee=0.0,
        stamp_duty=0.0,
        exchange_fee=0.0,
        settlement_fee=0.0,
        sec_fee=_money(sec),
        taf=_money(taf),
    )


def _non_negative(value: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} 必须是数字")
    if value < 0:
        raise ValueError(f"{label} 不能为负")


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))


def _ceil_dollar(amount: Decimal) -> Decimal:
    return amount.to_integral_value(rounding=ROUND_CEILING)


def _money(amount: Decimal) -> float:
    return float(amount)
