"""只读上下文。判断方法从这里取基础层已经算好的结果，不再重算拐点或边界。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from analysis.structure.series import StandardSeries


@dataclass(frozen=True)
class StructureContext:
    """一次运行的基础层结果。方法只读，不修改里面的对象。"""

    series: StandardSeries
    volatility: Tuple[Optional[float], ...]
    pivots: object
    segments: object
    boundaries: Tuple[object, ...]
    zones: Tuple[object, ...]
    events: Tuple[object, ...]


def build_context(
    series: StandardSeries,
    *,
    volatility: Sequence[Optional[float]],
    pivots: object,
    segments: object,
    boundaries: Sequence[object],
    zones: Sequence[object],
    events: Sequence[object],
) -> StructureContext:
    if len(volatility) != len(series):
        raise ValueError(
            f"波动序列长度 {len(volatility)} 与 K 线 {len(series)} 不一致"
        )
    return StructureContext(
        series=series,
        volatility=tuple(volatility),
        pivots=pivots,
        segments=segments,
        boundaries=tuple(boundaries),
        zones=tuple(zones),
        events=tuple(events),
    )
