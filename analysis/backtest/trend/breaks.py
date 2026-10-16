"""策略 F 在一根 K 线上读到的突破和支撑。

只使用 as-of 视图里已经可用的主结果事件。备选线不产生突破事件。
一条事件记在它的可用序号上，新鲜窗口和「用过一次」由账户会话处理。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from analysis.structure.series import StandardSeries


@dataclass(frozen=True)
class BreakFrame:
    up: tuple
    down: tuple
    support_valid: bool
    phase_trigger: Optional[str]
    range_floor: Optional[float]


def read_breaks(
    view: dict,
    series: StandardSeries,
    index: int,
    scales: Sequence[str],
    *,
    entry_scale: str,
) -> BreakFrame:
    """`view` 必须已经按当根时间做过 as-of。`scales` 含交易尺度和参照尺度。"""
    wanted = tuple(dict.fromkeys(scales))
    if not wanted:
        return BreakFrame((), (), False, None, None)
    params = series.params
    buffer_ratio = float(params.breakout_buffer_ratio)
    touch_ratio = float(params.touch_distance_ratio)
    boundaries = {item.get("id"): item for item in view.get("boundaries") or [] if item.get("id")}
    zones = {item.get("id"): item for item in view.get("zones") or [] if item.get("id")}
    up = []
    down = []
    for event in view.get("events") or []:
        if event.get("event_type") != "breakout" or _as_int(event.get("available_index"), -1) != index:
            continue
        kind = event.get("object_kind")
        object_id = event.get("object_id")
        if kind == "boundary":
            side, scale = _boundary_side(boundaries.get(object_id), wanted)
        elif kind == "zone":
            side, scale = _zone_side(zones.get(object_id), series, index, buffer_ratio, wanted)
        else:
            continue
        if side is None or scale is None:
            continue
        _append(up if side == "up" else down, _token(scale, kind, object_id, side))
    phase_objects = ((view.get("methods") or {}).get("phase") or {}).get("objects") or []
    direction_objects = ((view.get("methods") or {}).get("direction") or {}).get("objects") or []
    trigger = None
    floor = None
    for scale in wanted:
        phase = _covering(phase_objects, scale, index)
        if phase is None or _as_int(phase.get("start_index"), -1) != index:
            continue
        phase_trigger = phase.get("trigger")
        if phase_trigger == "range_break_up":
            _append(up, _token(scale, "phase", phase.get("id"), "up"))
        elif phase_trigger == "range_break_down":
            _append(down, _token(scale, "phase", phase.get("id"), "down"))
        if scale == entry_scale:
            trigger = None if phase_trigger is None else str(phase_trigger)
            floor = _range_floor(phase.get("range_bounds"), series.price_axis)
    entry_phase = _covering(phase_objects, entry_scale, index)
    if trigger is None and entry_phase is not None:
        trigger = None if entry_phase.get("trigger") is None else str(entry_phase.get("trigger"))
        floor = _range_floor(entry_phase.get("range_bounds"), series.price_axis)
    support = _support_valid(
        view,
        series,
        index,
        wanted,
        entry_scale,
        direction_objects,
        buffer_ratio,
        touch_ratio,
    )
    return BreakFrame(tuple(up), tuple(down), support, trigger, floor)


def _as_int(value, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _token(scale: str, kind: str, object_id, side: str) -> str:
    return f"{scale}|{kind}|{object_id}|{side}"


def _append(bucket: list, token: str) -> None:
    if token not in bucket:
        bucket.append(token)


def _boundary_side(boundary: Optional[dict], scales: Sequence[str]):
    if not boundary or boundary.get("scale") not in scales:
        return None, None
    if boundary.get("role") == "resistance":
        return "up", boundary.get("scale")
    if boundary.get("role") == "support":
        return "down", boundary.get("scale")
    return None, None


def _zone_side(zone: Optional[dict], series: StandardSeries, index: int, buffer_ratio: float, scales: Sequence[str]):
    if not zone or zone.get("scale") not in scales:
        return None, None
    geometric = series.bars[index].geometric
    volatility = float(zone.get("volatility") or 0.0)
    if volatility <= 0 or not math.isfinite(geometric):
        return None, None
    upper = float(zone["upper_slope"]) * index + float(zone["upper_intercept"])
    lower = float(zone["lower_slope"]) * index + float(zone["lower_intercept"])
    buffer = buffer_ratio * volatility
    if geometric - upper > buffer:
        return "up", zone.get("scale")
    if lower - geometric > buffer:
        return "down", zone.get("scale")
    return None, None


def _support_valid(
    view: dict,
    series: StandardSeries,
    index: int,
    scales: Sequence[str],
    entry_scale: str,
    direction_objects: Sequence[dict],
    buffer_ratio: float,
    touch_ratio: float,
) -> bool:
    direction = _covering(direction_objects, entry_scale, index)
    if direction is None or direction.get("structure") != "uptrend":
        return False
    start = int(direction.get("start_index") or 0)
    geometric = series.bars[index].geometric
    if not math.isfinite(geometric):
        return False
    pivots = [item for item in view.get("pivots") or [] if isinstance(item, dict)]
    for boundary in view.get("boundaries") or []:
        if not boundary.get("primary") or boundary.get("role") != "support":
            continue
        if boundary.get("status") != "validated" or boundary.get("scale") not in scales:
            continue
        volatility = float(boundary.get("volatility") or 0.0)
        if volatility <= 0:
            continue
        line = float(boundary["slope"]) * index + float(boundary["intercept"])
        if line - geometric > buffer_ratio * volatility:
            continue
        low = _latest_low(pivots, boundary.get("scale"), start, index)
        if low is None:
            continue
        extreme = int(low["extreme_index"])
        low_line = float(boundary["slope"]) * extreme + float(boundary["intercept"])
        if abs(float(low["geometric_price"]) - low_line) <= touch_ratio * volatility:
            return True
    return False


def _latest_low(pivots: Sequence[dict], scale: str, start: int, index: int) -> Optional[dict]:
    chosen = None
    for pivot in pivots:
        if pivot.get("scale") != scale or pivot.get("role") != "low":
            continue
        confirm = _as_int(pivot.get("confirm_index"), -1)
        extreme = _as_int(pivot.get("extreme_index"), -1)
        if confirm > index or extreme < start:
            continue
        if chosen is None or (confirm, extreme) > (
            _as_int(chosen.get("confirm_index"), -1),
            _as_int(chosen.get("extreme_index"), -1),
        ):
            chosen = pivot
    return chosen


def _covering(objects: Sequence[dict], scale: str, index: int) -> Optional[dict]:
    chosen = None
    for item in objects:
        if item.get("scale") != scale:
            continue
        start = int(item.get("start_index") or 0)
        end = int(item["end_index"])
        if start <= index <= end and (chosen is None or start >= int(chosen["start_index"])):
            chosen = item
    return chosen


def _range_floor(bounds, price_axis: str) -> Optional[float]:
    if not isinstance(bounds, dict) or bounds.get("lower") is None:
        return None
    lower = float(bounds["lower"])
    if not math.isfinite(lower):
        return None
    if price_axis == "log":
        return math.exp(lower)
    return lower
