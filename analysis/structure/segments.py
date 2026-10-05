"""基础线段与临时末端。

相邻两个已确认拐点连成一条基础线段。只连接同一尺度、角色交替、
极值序号严格递增、且两端极值之间没有间断的拐点。基础线段全部保留，
不把多条线段合并成一条。

临时末端从「确认序号等于本段起点」的最后那个已确认拐点，连到当前
尚未确认的反向极值。它单独存放，不进入已确认线段，也不能当作以后
边界候选的端点。

线段生成之后再记录包含关系：长尺度记下序号范围内的中尺度线段，
中尺度记下短尺度线段。这里只追加标注，不改拐点。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

from analysis.structure.config import LONG, MID, SCALE_NAMES, SHORT
from analysis.structure.pivots import (
    ROLE_HIGH,
    ROLE_LOW,
    Pivot,
    PivotResult,
    TemporaryExtreme,
    detect_pivots,
)
from analysis.structure.series import StandardSeries

DIRECTION_UP = "up"
DIRECTION_DOWN = "down"
DIRECTION_FLAT = "flat"
DIRECTIONS = (DIRECTION_UP, DIRECTION_DOWN, DIRECTION_FLAT)


@dataclass(frozen=True)
class Segment:
    """两个已确认拐点之间的基础线段。确认后不再改写。"""

    id: str
    scale: str
    direction: str
    start: Pivot
    end: Pivot
    bar_span: int
    change: float
    change_multiple: float
    slope: float
    max_deviation: float
    adverse_excursion: float
    volatility: float
    confirm_time: datetime
    available_time: datetime
    contains: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.scale not in SCALE_NAMES:
            raise ValueError(f"未知尺度 {self.scale!r}")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"未知方向 {self.direction!r}")
        if self.start.scale != self.scale or self.end.scale != self.scale:
            raise ValueError("线段两端必须属于这条线段的尺度")
        span = self.end.extreme_index - self.start.extreme_index
        if self.bar_span != span or span <= 0:
            raise ValueError(
                "跨越根数必须是终点极值序号减起点极值序号，且为正: "
                f"span={self.bar_span} start={self.start.extreme_index} "
                f"end={self.end.extreme_index}"
            )
        if self.confirm_time != self.end.confirm_time:
            raise ValueError("线段确认时间必须等于终点拐点的确认时间")
        if self.available_time != self.end.available_time:
            raise ValueError("线段可用时间必须等于终点拐点的可用时间")
        if self.volatility != self.end.volatility:
            raise ValueError("段初波动尺度必须使用终点拐点上冻结的值")
        if len(set(self.contains)) != len(self.contains):
            raise ValueError(f"包含关系有重复线段 {self.contains}")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "scale": self.scale,
            "direction": self.direction,
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "bar_span": self.bar_span,
            "change": self.change,
            "change_multiple": self.change_multiple,
            "slope": self.slope,
            "max_deviation": self.max_deviation,
            "adverse_excursion": self.adverse_excursion,
            "volatility": self.volatility,
            "confirm_time": self.confirm_time.isoformat(),
            "available_time": self.available_time.isoformat(),
            "contains": list(self.contains),
        }


@dataclass(frozen=True)
class TemporarySegment:
    """末端尚未确认的一段。不参与边界候选。"""

    id: str
    scale: str
    direction: str
    start: Pivot
    end: TemporaryExtreme
    bar_span: int
    change: float
    change_multiple: Optional[float]
    slope: float
    max_deviation: float
    adverse_excursion: float
    volatility: Optional[float]

    def __post_init__(self) -> None:
        if self.scale not in SCALE_NAMES:
            raise ValueError(f"未知尺度 {self.scale!r}")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"未知方向 {self.direction!r}")
        if self.start.scale != self.scale or self.end.scale != self.scale:
            raise ValueError("临时线段两端必须属于这条线段的尺度")
        if self.end.role not in (ROLE_HIGH, ROLE_LOW):
            raise ValueError("临时线段的终点必须是高点或低点候选")
        span = self.end.extreme_index - self.start.extreme_index
        if self.bar_span != span or span <= 0:
            raise ValueError("临时线段的跨越根数必须为正")
        if (self.volatility is None) != (self.change_multiple is None):
            raise ValueError("临时线段的波动尺度和倍数必须同时存在或同时缺失")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "scale": self.scale,
            "direction": self.direction,
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "bar_span": self.bar_span,
            "change": self.change,
            "change_multiple": self.change_multiple,
            "slope": self.slope,
            "max_deviation": self.max_deviation,
            "adverse_excursion": self.adverse_excursion,
            "volatility": self.volatility,
            "confirmed": False,
        }


@dataclass(frozen=True)
class SegmentResult:
    segments: Tuple[Segment, ...]
    temporary: Tuple[TemporarySegment, ...]

    def segments_for(self, scale: str) -> Tuple[Segment, ...]:
        return tuple(segment for segment in self.segments if segment.scale == scale)

    def temporary_for(self, scale: str) -> Tuple[TemporarySegment, ...]:
        return tuple(segment for segment in self.temporary if segment.scale == scale)


def build_segments(
    series: StandardSeries,
    pivots: Optional[PivotResult] = None,
    bar_count: Optional[int] = None,
) -> SegmentResult:
    """由已确认拐点生成基础线段。

    `bar_count` 只使用前这么多根，并重新检测拐点，保证和把更短的序列
    单独喂进来一致。已经传入的 `pivots` 在指定 `bar_count` 时不会沿用。
    """
    if bar_count is not None:
        series = series.prefix(bar_count)
        pivots = None
    if pivots is None:
        pivots = detect_pivots(series)
    _require_pivots_in_series(series, pivots)

    segments: List[Segment] = []
    temporary: List[TemporarySegment] = []
    for scale in SCALE_NAMES:
        confirmed = pivots.confirmed_for(scale)
        segments.extend(_confirmed_segments(series, confirmed))
        pending = _temporary_segment(series, confirmed, pivots.temporary_for(scale))
        if pending is not None:
            temporary.append(pending)
    return SegmentResult(
        segments=_attach_contains(segments),
        temporary=tuple(temporary),
    )


def format_segments(result: SegmentResult) -> str:
    """给人核对线段端点、涨跌幅和包含关系的文本。不参与拟合。"""
    lines = []
    for segment in result.segments:
        contained = ",".join(segment.contains) if segment.contains else "-"
        lines.append(
            "SEGMENT "
            f"{segment.scale:5} {segment.direction:4} "
            f"{segment.start.extreme_index}->{segment.end.extreme_index} "
            f"span={segment.bar_span} change={segment.change:.6f} "
            f"multiple={segment.change_multiple:.3f} slope={segment.slope:.6f} "
            f"dev={segment.max_deviation:.6f} adverse={segment.adverse_excursion:.6f} "
            f"confirm={segment.end.confirm_index} contains={contained}"
        )
    for segment in result.temporary:
        multiple = (
            "None"
            if segment.change_multiple is None
            else f"{segment.change_multiple:.3f}"
        )
        lines.append(
            "TEMPORARY "
            f"{segment.scale:5} {segment.direction:4} "
            f"{segment.start.extreme_index}->{segment.end.extreme_index} "
            f"span={segment.bar_span} change={segment.change:.6f} "
            f"multiple={multiple} dev={segment.max_deviation:.6f} "
            f"adverse={segment.adverse_excursion:.6f}"
        )
    return "\n".join(lines)


def _confirmed_segments(
    series: StandardSeries,
    confirmed: Sequence[Pivot],
) -> List[Segment]:
    segments: List[Segment] = []
    for left, right in zip(confirmed, confirmed[1:]):
        if not _can_connect(series, left.role, left.extreme_index, right.role, right.extreme_index):
            continue
        segments.append(_make_segment(series, left, right))
    return segments


def _temporary_segment(
    series: StandardSeries,
    confirmed: Sequence[Pivot],
    temps: Sequence[TemporaryExtreme],
) -> Optional[TemporarySegment]:
    active = [item for item in temps if item.role in (ROLE_HIGH, ROLE_LOW)]
    if len(active) != 1:
        return None
    temp = active[0]
    anchor = next(
        (pivot for pivot in confirmed if pivot.confirm_index == temp.segment_start_index),
        None,
    )
    if anchor is None:
        return None
    if not _can_connect(
        series,
        anchor.role,
        anchor.extreme_index,
        temp.role,
        temp.extreme_index,
    ):
        return None
    return _make_temporary(series, anchor, temp)


def _can_connect(
    series: StandardSeries,
    start_role: str,
    start_index: int,
    end_role: str,
    end_index: int,
) -> bool:
    if start_role == end_role:
        return False
    if end_index <= start_index:
        return False
    return not _crosses_gap(series, start_index, end_index)


def _make_segment(series: StandardSeries, start: Pivot, end: Pivot) -> Segment:
    span = end.extreme_index - start.extreme_index
    change = end.geometric_price - start.geometric_price
    direction = _direction(start.role, end.role, change)
    return Segment(
        id=_segment_id(end.scale, start.extreme_index, end.extreme_index),
        scale=end.scale,
        direction=direction,
        start=start,
        end=end,
        bar_span=span,
        change=change,
        change_multiple=change / end.volatility,
        slope=change / span,
        max_deviation=_max_deviation(
            series,
            start.extreme_index,
            start.geometric_price,
            end.extreme_index,
            end.geometric_price,
        ),
        adverse_excursion=_adverse_excursion(
            series,
            start.extreme_index,
            end.extreme_index,
            direction,
        ),
        volatility=end.volatility,
        confirm_time=end.confirm_time,
        available_time=end.available_time,
    )


def _make_temporary(
    series: StandardSeries,
    start: Pivot,
    end: TemporaryExtreme,
) -> TemporarySegment:
    span = end.extreme_index - start.extreme_index
    change = end.geometric_price - start.geometric_price
    direction = _direction(start.role, end.role, change)
    volatility = end.volatility
    multiple = None if volatility is None or volatility <= 0 else change / volatility
    return TemporarySegment(
        id="tmp:" + _segment_id(end.scale, start.extreme_index, end.extreme_index),
        scale=end.scale,
        direction=direction,
        start=start,
        end=end,
        bar_span=span,
        change=change,
        change_multiple=multiple,
        slope=change / span,
        max_deviation=_max_deviation(
            series,
            start.extreme_index,
            start.geometric_price,
            end.extreme_index,
            end.geometric_price,
        ),
        adverse_excursion=_adverse_excursion(
            series,
            start.extreme_index,
            end.extreme_index,
            direction,
        ),
        volatility=volatility if multiple is not None else None,
    )


def _attach_contains(segments: Sequence[Segment]) -> Tuple[Segment, ...]:
    by_scale = {
        scale: [segment for segment in segments if segment.scale == scale]
        for scale in SCALE_NAMES
    }
    attached: List[Segment] = []
    for segment in segments:
        if segment.scale == LONG:
            inner = by_scale[MID]
        elif segment.scale == MID:
            inner = by_scale[SHORT]
        else:
            inner = ()
        contains = tuple(
            candidate.id
            for candidate in inner
            if _contains_range(segment, candidate)
        )
        attached.append(replace(segment, contains=contains))
    return tuple(attached)


def _contains_range(outer: Segment, inner: Segment) -> bool:
    return (
        outer.start.extreme_index <= inner.start.extreme_index
        and inner.end.extreme_index <= outer.end.extreme_index
    )


def _direction(start_role: str, end_role: str, change: float) -> str:
    if start_role == ROLE_LOW and end_role == ROLE_HIGH:
        return DIRECTION_UP
    if start_role == ROLE_HIGH and end_role == ROLE_LOW:
        return DIRECTION_DOWN
    if change > 0:
        return DIRECTION_UP
    if change < 0:
        return DIRECTION_DOWN
    return DIRECTION_FLAT


def _max_deviation(
    series: StandardSeries,
    start_index: int,
    start_geom: float,
    end_index: int,
    end_geom: float,
) -> float:
    """内部 K 线到两端极值连线的最大绝对几何距离。没有内部 K 线时为 0。"""
    span = end_index - start_index
    worst = 0.0
    for index in range(start_index + 1, end_index):
        weight = (index - start_index) / span
        line = start_geom + weight * (end_geom - start_geom)
        deviation = abs(series.bars[index].geometric - line)
        if deviation > worst:
            worst = deviation
    return worst


def _adverse_excursion(
    series: StandardSeries,
    start_index: int,
    end_index: int,
    direction: str,
) -> float:
    """从起点极值走到终点极值，几何价格上的最大逆向幅度。"""
    worst = 0.0
    if direction == DIRECTION_UP:
        peak = series.bars[start_index].geometric
        for index in range(start_index + 1, end_index + 1):
            price = series.bars[index].geometric
            drop = peak - price
            if drop > worst:
                worst = drop
            if price > peak:
                peak = price
        return worst
    if direction == DIRECTION_DOWN:
        trough = series.bars[start_index].geometric
        for index in range(start_index + 1, end_index + 1):
            price = series.bars[index].geometric
            rise = price - trough
            if rise > worst:
                worst = rise
            if price < trough:
                trough = price
        return worst
    return worst


def _crosses_gap(series: StandardSeries, start_index: int, end_index: int) -> bool:
    for index in range(start_index + 1, end_index + 1):
        if series.bars[index].gap_before:
            return True
    return False


def _segment_id(scale: str, start_index: int, end_index: int) -> str:
    return f"{scale}:{start_index}-{end_index}"


def _require_pivots_in_series(series: StandardSeries, pivots: PivotResult) -> None:
    limit = len(series)
    for pivot in pivots.confirmed:
        if pivot.confirm_index >= limit or pivot.extreme_index >= limit:
            raise ValueError(
                f"{pivot.scale} 拐点序号超出序列长度 {limit}: "
                f"extreme={pivot.extreme_index} confirm={pivot.confirm_index}"
            )
    for item in pivots.temporary:
        if item.extreme_index >= limit:
            raise ValueError(
                f"{item.scale} 临时极值序号超出序列长度 {limit}: {item.extreme_index}"
            )
