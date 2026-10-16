"""比例复权，以及持仓跨过除息、拆股时的现金、股数和止损。

因子锚在第一根。之后的派息和拆股只乘到后面的 K 线，不改写已经算过的复权价。
止损记在复权价上；和当根不复权的开盘价、最低价比较之前，先除以当根因子。
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class CorporateAction:
    """`kind` 是 dividend 或 split。dividend 的 value 是每股现金，split 的 value 是 1 股拆成的股数。"""

    index: int
    kind: str
    value: float


@dataclass(frozen=True)
class AdjustedPrices:
    factor: Tuple[float, ...]
    split_ratio: Tuple[float, ...]
    open: Tuple[float, ...]
    high: Tuple[float, ...]
    low: Tuple[float, ...]
    close: Tuple[float, ...]
    volume: Tuple[float, ...]


def adjust_prices(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
    actions: Sequence[CorporateAction] = (),
) -> AdjustedPrices:
    """不复权 OHLC 乘上累计因子。成交量只按累计拆股比例缩小。"""
    count = len(closes)
    if not (len(opens) == len(highs) == len(lows) == len(volumes) == count):
        raise ValueError("开盘、最高、最低、收盘和成交量的长度必须一致")
    if count == 0:
        raise ValueError("K 线为空")
    grouped = _group_actions(actions, count)
    factors: list[float] = []
    split_ratios: list[float] = []
    factor = 1.0
    split_ratio = 1.0
    for index in range(count):
        for action in grouped.get(index, ()):
            factor *= _multiplier(action, closes, index)
            if action.kind == "split":
                split_ratio *= action.value
        factors.append(factor)
        split_ratios.append(split_ratio)
    return AdjustedPrices(
        factor=tuple(factors),
        split_ratio=tuple(split_ratios),
        open=tuple(price * scale for price, scale in zip(opens, factors)),
        high=tuple(price * scale for price, scale in zip(highs, factors)),
        low=tuple(price * scale for price, scale in zip(lows, factors)),
        close=tuple(price * scale for price, scale in zip(closes, factors)),
        volume=tuple(volume / scale for volume, scale in zip(volumes, split_ratios)),
    )


def _group_actions(actions: Sequence[CorporateAction], count: int) -> dict:
    grouped: dict = {}
    for action in actions:
        if action.kind not in ("dividend", "split"):
            raise ValueError(f"未知除权类型 {action.kind}")
        if not 0 <= int(action.index) < count:
            raise ValueError(f"除权序号 {action.index} 超出 K 线范围")
        grouped.setdefault(int(action.index), []).append(action)
    return grouped


def _multiplier(action: CorporateAction, closes: Sequence[float], index: int) -> float:
    if action.kind == "split":
        if action.value <= 0:
            raise ValueError("拆股比例必须为正")
        return float(action.value)
    if index < 1:
        raise ValueError("除息需要前一根不复权收盘价")
    cash = float(action.value)
    previous = float(closes[index - 1])
    if cash <= 0 or cash >= previous:
        raise ValueError(f"每股股息必须为正，并且小于前收 {previous}")
    return previous / (previous - cash)


@dataclass(frozen=True)
class Holding:
    """复权止损在除权日保持不动。换成不复权价时再除以当根因子。"""

    shares: float
    cash: float
    adjusted_stop: Optional[float] = None


@dataclass(frozen=True)
class DividendCredit:
    cash: float
    tax: float


@dataclass(frozen=True)
class CorporateDecision:
    """一笔派息或拆股。canonical 用于核对未来事件不会改写过去的记录。"""

    index: int
    time: Optional[str]
    action: str
    shares_before: float
    shares_after: float
    cash_before: float
    cash_after: float
    dividend_per_share: Optional[float]
    withholding: Optional[float]
    tax: float
    cash_credited: float
    split_ratio: Optional[float]
    factor: float
    adjusted_stop: Optional[float]
    raw_stop: Optional[float]
    currency: str
    text: str

    def canonical(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def dividend_credit(shares: float, dividend_per_share: float, withholding: float) -> DividendCredit:
    """入账现金 = 股数 × 每股股息 × (1 − 预扣税率)。"""
    if shares < 0:
        raise ValueError("股数不能为负")
    if dividend_per_share <= 0:
        raise ValueError("每股股息必须为正")
    if not 0 <= withholding <= 1:
        raise ValueError("预扣税率必须在 [0, 1]")
    gross = float(shares) * float(dividend_per_share)
    tax = gross * float(withholding)
    return DividendCredit(gross - tax, tax)


def raw_from_adjusted(adjusted: float, factor: float) -> float:
    """复权价除以当根因子，得到可以和不复权开盘价、最低价比较的门槛。"""
    if factor <= 0:
        raise ValueError("复权因子必须为正")
    return float(adjusted) / float(factor)


def long_stop_hit(raw_price: float, adjusted_stop: float, factor: float) -> bool:
    """多头开盘价或最低价触及不复权止损。等于止损也算触及。"""
    return float(raw_price) <= raw_from_adjusted(adjusted_stop, factor)


def upward_gap_cancel(
    raw_open: float,
    factor: float,
    adjusted_signal_close: float,
    a_t: float,
    gap_cancel: float,
    *,
    price_axis: str = "log",
) -> bool:
    """开盘相对信号收盘向上跳空超过 gap_cancel 倍 a_t 时取消。

    门槛先在复权价上算，再除以当根因子。除息或拆股造成的不复权缺口不会触发。
    """
    if factor <= 0:
        raise ValueError("复权因子必须为正")
    if a_t < 0 or gap_cancel < 0:
        raise ValueError("波动单位和跳空倍数不能为负")
    if price_axis == "log":
        threshold = float(adjusted_signal_close) * math.exp(float(gap_cancel) * float(a_t))
    elif price_axis == "uniform":
        threshold = float(adjusted_signal_close) + float(gap_cancel) * float(a_t)
    else:
        raise ValueError(f"价格轴必须是 log 或 uniform，收到 {price_axis}")
    return float(raw_open) > raw_from_adjusted(threshold, factor)


def replay_holding(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
    actions: Sequence[CorporateAction],
    holding: Holding,
    *,
    withholding: float,
    currency: str = "",
    times: Sequence[Optional[str]] | None = None,
) -> tuple[AdjustedPrices, tuple[CorporateDecision, ...], Holding]:
    """按时间顺序结算除权。同一根上的事件按传入顺序，股息用当时的股数。

    返回复权价、决策记录，以及结算后的持仓。复权止损不因派息或拆股改写。
    """
    if holding.shares < 0:
        raise ValueError("股数不能为负")
    if not 0 <= withholding <= 1:
        raise ValueError("预扣税率必须在 [0, 1]")
    adjusted = adjust_prices(opens, highs, lows, closes, volumes, actions)
    if times is not None and len(times) != len(closes):
        raise ValueError("时间戳数量必须和 K 线一致")
    grouped = _group_actions(actions, len(closes))
    shares = float(holding.shares)
    cash = float(holding.cash)
    factor = 1.0
    decisions = []
    for index in range(len(closes)):
        for action in grouped.get(index, ()):
            factor *= _multiplier(action, closes, index)
            before_shares = shares
            before_cash = cash
            when = None if times is None else times[index]
            if action.kind == "dividend":
                credit = dividend_credit(shares, action.value, withholding)
                cash += credit.cash
                decisions.append(
                    _decision(
                        index=index,
                        time=when,
                        action="派息入账",
                        shares_before=before_shares,
                        shares_after=shares,
                        cash_before=before_cash,
                        cash_after=cash,
                        dividend_per_share=float(action.value),
                        withholding=float(withholding),
                        tax=credit.tax,
                        cash_credited=credit.cash,
                        split_ratio=None,
                        factor=factor,
                        adjusted_stop=holding.adjusted_stop,
                        currency=currency,
                    )
                )
                continue
            shares *= float(action.value)
            decisions.append(
                _decision(
                    index=index,
                    time=when,
                    action="拆股",
                    shares_before=before_shares,
                    shares_after=shares,
                    cash_before=before_cash,
                    cash_after=cash,
                    dividend_per_share=None,
                    withholding=None,
                    tax=0.0,
                    cash_credited=0.0,
                    split_ratio=float(action.value),
                    factor=factor,
                    adjusted_stop=holding.adjusted_stop,
                    currency=currency,
                )
            )
    return adjusted, tuple(decisions), Holding(shares, cash, holding.adjusted_stop)


def _decision(
    *,
    index: int,
    time: Optional[str],
    action: str,
    shares_before: float,
    shares_after: float,
    cash_before: float,
    cash_after: float,
    dividend_per_share: Optional[float],
    withholding: Optional[float],
    tax: float,
    cash_credited: float,
    split_ratio: Optional[float],
    factor: float,
    adjusted_stop: Optional[float],
    currency: str,
) -> CorporateDecision:
    raw_stop = None if adjusted_stop is None else raw_from_adjusted(adjusted_stop, factor)
    text = _decision_text(
        action=action,
        shares_before=shares_before,
        shares_after=shares_after,
        cash_before=cash_before,
        cash_after=cash_after,
        dividend_per_share=dividend_per_share,
        withholding=withholding,
        tax=tax,
        cash_credited=cash_credited,
        split_ratio=split_ratio,
        factor=factor,
        adjusted_stop=adjusted_stop,
        raw_stop=raw_stop,
        currency=currency,
    )
    return CorporateDecision(
        index=index,
        time=time,
        action=action,
        shares_before=shares_before,
        shares_after=shares_after,
        cash_before=cash_before,
        cash_after=cash_after,
        dividend_per_share=dividend_per_share,
        withholding=withholding,
        tax=tax,
        cash_credited=cash_credited,
        split_ratio=split_ratio,
        factor=factor,
        adjusted_stop=adjusted_stop,
        raw_stop=raw_stop,
        currency=currency,
        text=text,
    )


def _decision_text(
    *,
    action: str,
    shares_before: float,
    shares_after: float,
    cash_before: float,
    cash_after: float,
    dividend_per_share: Optional[float],
    withholding: Optional[float],
    tax: float,
    cash_credited: float,
    split_ratio: Optional[float],
    factor: float,
    adjusted_stop: Optional[float],
    raw_stop: Optional[float],
    currency: str,
) -> str:
    unit = f" {currency}" if currency else ""
    stop = _stop_text(adjusted_stop, raw_stop, unit)
    if action == "派息入账":
        return (
            f"派息入账：每股股息 {_num(dividend_per_share)}{unit}，"
            f"预扣税率 {_num(withholding)}，预扣税 {_num(tax)}{unit}，"
            f"入账现金 {_num(cash_credited)}{unit}，"
            f"股数 {_num(shares_after)}，"
            f"现金由 {_num(cash_before)}{unit} 变为 {_num(cash_after)}{unit}，"
            f"复权因子 {_num(factor)}，{stop}"
        )
    return (
        f"拆股：1 股变为 {_num(split_ratio)} 股，"
        f"股数由 {_num(shares_before)} 变为 {_num(shares_after)}，"
        f"现金不变仍为 {_num(cash_after)}{unit}，"
        f"复权因子 {_num(factor)}，{stop}"
    )


def _stop_text(adjusted_stop: Optional[float], raw_stop: Optional[float], unit: str) -> str:
    if adjusted_stop is None or raw_stop is None:
        return "没有止损"
    return f"复权止损 {_num(adjusted_stop)}，不复权止损 {_num(raw_stop)}{unit}"


def _num(value: Optional[float]) -> str:
    if value is None:
        return "无"
    text = f"{float(value):.8f}".rstrip("0").rstrip(".")
    return text or "0"
