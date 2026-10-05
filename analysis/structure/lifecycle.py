"""趋势线和趋势区间的生命周期。

候选和被拒绝的对象保持原状。已经验证的线和区间都会记录事件；
主结果标签以后可以重选，但不删除已经写下来的事件。

已验证时冻结斜率、截距、接触统计和评分。之后只追加事件：
收盘价在错误一侧连续停留，且深度超过缓冲，记为突破；
有效区间结束后走过规定根数，仍没有突破，也没有新的验证，记为过期。
价格可以更早越过边界，那一根本身只记成价格事实。可引用的事件不早于
结构本身的可用时间。

本模块不生成买卖单，也不假设能以信号当根的收盘价成交。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

from analysis.structure.boundaries import (
    ROLE_SUPPORT,
    STATUS_BROKEN,
    STATUS_EXPIRED,
    STATUS_VALIDATED,
    Boundary,
    build_boundaries,
)
from analysis.structure.pivots import PivotResult
from analysis.structure.series import StandardSeries
from analysis.structure.zones import Zone, ZoneOverlap, build_zones

EVENT_VALIDATED = "validated"
EVENT_BREAKOUT = "breakout"
EVENT_EXPIRED = "expired"
_EVENT_ORDER = {EVENT_VALIDATED: 0, EVENT_BREAKOUT: 1, EVENT_EXPIRED: 2}

KIND_BOUNDARY = "boundary"
KIND_ZONE = "zone"


@dataclass(frozen=True)
class StructureEvent:
    """一次状态变化。历史回放里可用时间等于事件所在的那根已完成 K 线。"""

    event_type: str
    event_index: int
    event_time: datetime
    available_index: int
    available_time: datetime
    reason: str
    reference_close: float
    object_id: str
    object_kind: str
    revision: int
    price_cross_index: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "event_index": self.event_index,
            "event_time": self.event_time.isoformat(),
            "available_index": self.available_index,
            "available_time": self.available_time.isoformat(),
            "reason": self.reason,
            "reference_close": self.reference_close,
            "object_id": self.object_id,
            "object_kind": self.object_kind,
            "revision": self.revision,
            "price_cross_index": self.price_cross_index,
        }


@dataclass(frozen=True)
class LifecycleResult:
    boundaries: Tuple[Boundary, ...]
    zones: Tuple[Zone, ...]
    overlaps: Tuple[ZoneOverlap, ...]
    events: Tuple[StructureEvent, ...]

    def events_for(self, object_id: str) -> Tuple[StructureEvent, ...]:
        return tuple(event for event in self.events if event.object_id == object_id)

    def boundary(self, object_id: str) -> Boundary:
        for item in self.boundaries:
            if item.id == object_id:
                return item
        raise KeyError(object_id)

    def zone(self, object_id: str) -> Zone:
        for item in self.zones:
            if item.id == object_id:
                return item
        raise KeyError(object_id)


def build_lifecycle(
    series: StandardSeries,
    pivots: Optional[PivotResult] = None,
    bar_count: Optional[int] = None,
) -> LifecycleResult:
    """按当前序列给出生命周期状态和事件。

    `bar_count` 只使用前这么多根。更早事件的可用时间和边界参数保持不变，
    之后新出现的结构使用更大的修订号。
    """
    if bar_count is not None:
        series = series.prefix(bar_count)
        pivots = None
    boundaries = build_boundaries(series, pivots)
    zones = build_zones(series, boundaries=boundaries)
    revisions = _revisions(boundaries.boundaries, zones.zones)
    events: List[StructureEvent] = []
    updated_boundaries = [
        _track_boundary(series, item, revisions[item.id], events) for item in boundaries.boundaries
    ]
    updated_zones = [_track_zone(series, item, revisions[item.id], events) for item in zones.zones]
    events.sort(key=lambda event: (event.available_index, _EVENT_ORDER[event.event_type], event.object_id))
    return LifecycleResult(
        boundaries=tuple(updated_boundaries),
        zones=tuple(updated_zones),
        overlaps=zones.overlaps,
        events=tuple(events),
    )


def format_events(result: LifecycleResult) -> str:
    """给人核对事件时间和突破位置的文本。不参与拟合。"""
    lines = []
    for event in result.events:
        cross = "-" if event.price_cross_index is None else str(event.price_cross_index)
        lines.append(
            f"{event.event_type:10} {event.object_kind:8} "
            f"event={event.event_index} cross={cross} rev={event.revision} "
            f"close={event.reference_close:.4f} {event.reason} id={event.object_id}"
        )
    return "\n".join(lines)


def _track_boundary(
    series: StandardSeries,
    boundary: Boundary,
    revision: int,
    events: List[StructureEvent],
) -> Boundary:
    status = _advance(series, boundary, revision, events, KIND_BOUNDARY, boundary.end_index, _beyond_boundary)
    return replace(boundary, revision=revision, status=status)


def _track_zone(
    series: StandardSeries,
    zone: Zone,
    revision: int,
    events: List[StructureEvent],
) -> Zone:
    status = _advance(series, zone, revision, events, KIND_ZONE, zone.effective_end, _beyond_zone)
    draw = zone.draw_on_main and status == STATUS_VALIDATED
    return replace(zone, revision=revision, status=status, draw_on_main=draw)


def _advance(series, structure, revision, events, kind, effective_end, beyond) -> str:
    # 主结果标签会随后到的线重选。事件按几何对象本身记录，避免后到的数据把已经发生的事件删掉。
    if structure.status != STATUS_VALIDATED:
        return structure.status
    _append_event(
        series,
        events,
        event_type=EVENT_VALIDATED,
        event_index=structure.confirm_index,
        reason="硬约束已满足，边界参数就此冻结",
        object_id=structure.id,
        object_kind=kind,
        revision=revision,
        price_cross_index=None,
    )
    breakout = _breakout(series, structure, beyond)
    expiry_index = _expiry_index(series, structure.confirm_index, effective_end, series.params.expire_bars)
    if breakout is not None and (expiry_index is None or breakout[0] <= expiry_index):
        event_index, cross_index = breakout
        _append_event(
            series,
            events,
            event_type=EVENT_BREAKOUT,
            event_index=event_index,
            reason=_breakout_reason(series, kind, structure),
            object_id=structure.id,
            object_kind=kind,
            revision=revision,
            price_cross_index=cross_index,
        )
        return STATUS_BROKEN
    if expiry_index is not None:
        _append_event(
            series,
            events,
            event_type=EVENT_EXPIRED,
            event_index=expiry_index,
            reason=f"有效区间结束后 {series.params.expire_bars} 根没有突破，也没有新的验证",
            object_id=structure.id,
            object_kind=kind,
            revision=revision,
            price_cross_index=None,
        )
        return STATUS_EXPIRED
    return STATUS_VALIDATED


def _breakout(series, structure, beyond) -> Optional[Tuple[int, int]]:
    """返回 (事件序号, 价格越过序号)。事件不早于结构确认。"""
    needed = series.params.breakout_bars
    run_start: Optional[int] = None
    run_length = 0
    for index in range(len(series)):
        if beyond(series, structure, index):
            if run_start is None:
                run_start = index
            run_length += 1
            if run_length >= needed and index >= structure.confirm_index:
                return index, run_start
        else:
            run_start = None
            run_length = 0
    return None


def _expiry_index(series: StandardSeries, confirm_index: int, effective_end: int, expire_bars: int) -> Optional[int]:
    raw = effective_end + expire_bars
    event_index = raw if raw >= confirm_index else confirm_index
    if event_index >= len(series):
        return None
    return event_index


def _beyond_boundary(series: StandardSeries, boundary: Boundary, index: int) -> bool:
    geometric = series.bars[index].geometric
    buffer = series.params.breakout_buffer_ratio * boundary.volatility
    line = boundary.line_at(index)
    if boundary.role == ROLE_SUPPORT:
        return line - geometric > buffer
    return geometric - line > buffer


def _beyond_zone(series: StandardSeries, zone: Zone, index: int) -> bool:
    geometric = series.bars[index].geometric
    buffer = series.params.breakout_buffer_ratio * zone.volatility
    if zone.lower_at(index) - geometric > buffer:
        return True
    return geometric - zone.upper_at(index) > buffer


def _breakout_reason(series: StandardSeries, kind: str, structure) -> str:
    bars = series.params.breakout_bars
    ratio = series.params.breakout_buffer_ratio
    if kind == KIND_ZONE:
        return f"收盘价连续 {bars} 根落在区间外，深度超过 {ratio:g} 倍波动"
    if structure.role == ROLE_SUPPORT:
        return f"收盘价连续 {bars} 根落在支撑下方，深度超过 {ratio:g} 倍波动"
    return f"收盘价连续 {bars} 根落在压力上方，深度超过 {ratio:g} 倍波动"


def _append_event(
    series: StandardSeries,
    events: List[StructureEvent],
    *,
    event_type: str,
    event_index: int,
    reason: str,
    object_id: str,
    object_kind: str,
    revision: int,
    price_cross_index: Optional[int],
) -> None:
    available_index = event_index
    if price_cross_index is not None and price_cross_index > event_index:
        price_cross_index = event_index
    bar = series.bars[available_index]
    events.append(
        StructureEvent(
            event_type=event_type,
            event_index=event_index,
            event_time=bar.timestamp,
            available_index=available_index,
            available_time=bar.timestamp,
            reason=reason,
            reference_close=bar.close,
            object_id=object_id,
            object_kind=object_kind,
            revision=revision,
            price_cross_index=price_cross_index,
        )
    )


def _revisions(boundaries: Sequence[Boundary], zones: Sequence[Zone]) -> dict:
    """同一确认序号共用一个修订号。更晚确认的结构使用更大的修订号。"""
    moments = sorted({item.confirm_index for item in boundaries} | {item.confirm_index for item in zones})
    rank = {moment: position + 1 for position, moment in enumerate(moments)}
    revisions = {item.id: rank[item.confirm_index] for item in boundaries}
    revisions.update({item.id: rank[item.confirm_index] for item in zones})
    return revisions
