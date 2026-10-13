"""比例复权。因子锚在第一根，之后的派息和拆股只乘到后面的 K 线。

现金入账和持仓股数调整不在这里做。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple


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
