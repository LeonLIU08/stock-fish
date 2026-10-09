"""区间形状：通道、楔形、三角形和扩张形态。

几何全部来自第一版已经验证过的趋势区间。这里只按两条轨的走向和宽度变化分类，
不另拟合边界，也不改区间的状态和事件。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from analysis.structure.boundaries import LIVE_STATUSES
from analysis.structure.config import (
    LONG,
    MID,
    SCALE_NAMES,
    SHORT,
    zone_shape_params_from_dict,
)
from analysis.structure.lifecycle import EVENT_BREAKOUT, EVENT_EXPIRED, EVENT_VALIDATED, KIND_ZONE
from analysis.structure.methods.base import (
    KIND_INSTANCE,
    CategorySpec,
    ColumnSpec,
    MethodResult,
    MethodSample,
    TableSpec,
    make_mark,
)
from analysis.structure.view_model import price_on_line

RISING = "rising"
FALLING = "falling"
LEVEL = "level"
STABLE = "stable"
CONVERGING = "converging"
DIVERGING = "diverging"
UNCLASSIFIED = "unclassified"

RISING_CHANNEL = "rising_channel"
FALLING_CHANNEL = "falling_channel"
FLAT_CHANNEL = "flat_channel"
RISING_WEDGE = "rising_wedge"
FALLING_WEDGE = "falling_wedge"
SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
ASCENDING_TRIANGLE = "ascending_triangle"
DESCENDING_TRIANGLE = "descending_triangle"
BROADENING = "broadening"

COURSE_NAME = {RISING: "上升", FALLING: "下降", LEVEL: "水平"}
WIDTH_NAME = {STABLE: "稳定", CONVERGING: "收敛", DIVERGING: "张开"}
LEGACY_NAME = {
    "channel": "通道",
    "convergence": "收敛区间",
    "sideways": "横向区间",
    "other": "其他边界对",
}
STATUS_NAME = {
    "validated": "已验证",
    "candidate": "候选",
    "rejected": "拒绝",
    "broken": "已被收盘价突破",
    "expired": "过期",
}
SCALE_LEGEND = {SHORT: "短尺度", MID: "中尺度", LONG: "长尺度"}
_EVENT_TYPES = (EVENT_VALIDATED, EVENT_BREAKOUT, EVENT_EXPIRED)
_EVENT_RANK = {EVENT_VALIDATED: 0, EVENT_BREAKOUT: 1, EVENT_EXPIRED: 2}
_TEXT_COLOR = "#F3EFE4"
_NOTE = "只按两条轨在有效区间里的位移和宽度变化分类。未归类不画色带。"

# (宽度走向, 上轨走向, 下轨走向) → 类别。收敛时能出现的组合都在这里。
_BY_COURSES = {
    (STABLE, RISING, RISING): RISING_CHANNEL,
    (STABLE, FALLING, FALLING): FALLING_CHANNEL,
    (STABLE, LEVEL, LEVEL): FLAT_CHANNEL,
    (CONVERGING, RISING, RISING): RISING_WEDGE,
    (CONVERGING, FALLING, FALLING): FALLING_WEDGE,
    (CONVERGING, FALLING, RISING): SYMMETRICAL_TRIANGLE,
    (CONVERGING, LEVEL, LEVEL): SYMMETRICAL_TRIANGLE,
    (CONVERGING, LEVEL, RISING): ASCENDING_TRIANGLE,
    (CONVERGING, FALLING, LEVEL): DESCENDING_TRIANGLE,
}


class ZoneShapeMethod:
    key = "zone_shape"
    name = "区间形状"
    section = "2.2"
    version = "zone_shape-v1"
    kind = KIND_INSTANCE
    requires = ("zones", "events")
    categories = (
        CategorySpec(RISING_CHANNEL, "上升通道", "rgba(232, 93, 76, 0.28)"),
        CategorySpec(FALLING_CHANNEL, "下降通道", "rgba(61, 220, 151, 0.28)"),
        CategorySpec(FLAT_CHANNEL, "横向通道", "rgba(240, 176, 96, 0.28)"),
        CategorySpec(RISING_WEDGE, "上升楔形", "rgba(255, 122, 92, 0.30)"),
        CategorySpec(FALLING_WEDGE, "下降楔形", "rgba(72, 196, 168, 0.30)"),
        CategorySpec(SYMMETRICAL_TRIANGLE, "对称三角形", "rgba(196, 148, 255, 0.30)"),
        CategorySpec(ASCENDING_TRIANGLE, "上升三角形", "rgba(240, 112, 140, 0.30)"),
        CategorySpec(DESCENDING_TRIANGLE, "下降三角形", "rgba(96, 176, 214, 0.30)"),
        CategorySpec(BROADENING, "扩张形态", "rgba(230, 196, 92, 0.30)"),
    )
    replaces = ("zones",)

    def default_params(self) -> dict:
        return zone_shape_params_from_dict(None).to_dict()

    def check_params(self, params: dict) -> None:
        zone_shape_params_from_dict(params)

    def judge(self, context, params: dict, upstream: dict) -> MethodResult:
        del upstream
        pack = zone_shape_params_from_dict(params)
        events_by_zone: Dict[str, List[object]] = {}
        for event in context.events:
            if getattr(event, "object_kind", None) != KIND_ZONE:
                continue
            if event.event_type not in _EVENT_TYPES:
                continue
            events_by_zone.setdefault(event.object_id, []).append(event)
        objects: List[dict] = []
        events: List[dict] = []
        rows: List[dict] = []
        names = {item.key: item.name for item in self.categories}
        for zone in context.zones:
            if zone.status not in LIVE_STATUSES:
                continue
            judged = classify_rails(
                upper_slope=zone.upper_slope,
                lower_slope=zone.lower_slope,
                start_index=zone.effective_start,
                end_index=zone.effective_end,
                start_width=zone.start_width,
                width_ratio=zone.width_ratio,
                level_tolerance=pack.level_tolerance,
                converging_width_ratio=pack.converging_width_ratio,
                diverging_width_ratio=pack.diverging_width_ratio,
            )
            category = judged["category"]
            rows.append(
                {
                    "zone_id": zone.id,
                    "scale": zone.scale,
                    "legacy_label": zone.label,
                    "upper_displacement": judged["upper_displacement"],
                    "lower_displacement": judged["lower_displacement"],
                    "width_ratio": zone.width_ratio,
                    "upper_course": judged["upper_course"],
                    "lower_course": judged["lower_course"],
                    "width_course": judged["width_course"],
                    "category": category,
                    "available_index": int(zone.confirm_index),
                }
            )
            if category == UNCLASSIFIED:
                continue
            shape_id = f"zone_shape:{zone.id}"
            objects.append(_shape(zone, shape_id, judged))
            events.extend(_shape_events(events_by_zone.get(zone.id, ()), shape_id, names[category]))
        order = {name: index for index, name in enumerate(SCALE_NAMES)}
        objects.sort(key=lambda item: (item["start_index"], order[item["scale"]], item["id"]))
        rows.sort(key=lambda item: (item["available_index"], order[item["scale"]], item["zone_id"]))
        events.sort(
            key=lambda item: (
                item["available_index"],
                item["object_id"],
                _EVENT_RANK.get(item["event_type"], 9),
            )
        )
        return MethodResult(objects=objects, events=events, extra={"classifications": rows}, notes=[_NOTE])

    def present(self, record: dict, view: dict) -> List[dict]:
        category = record.get("category") or ""
        names = {item.key: item.name for item in self.categories}
        colors = {item.key: item.color for item in self.categories}
        if category not in names or not record.get("primary"):
            return []
        evidence = record.get("evidence") or {}
        zone_id = evidence.get("zone_id") if isinstance(evidence, dict) else None
        zone = (view.get("references") or {}).get(zone_id) if zone_id else None
        if not isinstance(zone, dict):
            return []
        if zone.get("lower_slope") is None or zone.get("upper_slope") is None:
            return []
        if zone.get("effective_start") is None or zone.get("effective_end") is None:
            return []
        start = int(zone["effective_start"])
        end = int(zone["effective_end"])
        status = record.get("status") or ""
        if status == "broken":
            breakout = _breakout_index(view.get("events") or [], record.get("id"))
            if breakout is not None:
                end = min(end, breakout)
        stop = view.get("stop_index")
        if stop is not None and status == "validated":
            end = min(end, int(stop)) if end > start else end
        if end < start:
            return []
        axis = view.get("price_axis") or "log"
        lower_start = price_on_line(zone["lower_slope"], zone["lower_intercept"], start, axis)
        lower_end = price_on_line(zone["lower_slope"], zone["lower_intercept"], end, axis)
        upper_end = price_on_line(zone["upper_slope"], zone["upper_intercept"], end, axis)
        upper_start = price_on_line(zone["upper_slope"], zone["upper_intercept"], start, axis)
        points = [
            [start, lower_start],
            [end, lower_end],
            [end, upper_end],
            [start, upper_start],
        ]
        if not all(math.isfinite(point[1]) and point[1] > 0 for point in points):
            return []
        color = colors[category]
        legend = f"区间形状 · {SCALE_LEGEND.get(record.get('scale'), record.get('scale'))}"
        hover = _hover_lines(record, names[category])
        line_type = "dashed" if status in ("broken", "expired") else "solid"
        shared = dict(
            object_id=record["id"],
            method=self.key,
            scale=record.get("scale") or "",
            category=category,
            legend=legend,
            start_index=start,
            end_index=end,
            available_index=int(record.get("available_index") or start),
            hover=hover,
        )
        return [
            make_mark(kind="band", points=points, line_type=line_type, color=color, **shared),
            make_mark(
                kind="text",
                text=names[category],
                color=_TEXT_COLOR,
                points=[[end, upper_end]],
                anchor="price",
                index=end,
                price=upper_end,
                **shared,
            ),
        ]

    def tables(self) -> Tuple[TableSpec, ...]:
        shape_columns = (
            ColumnSpec("标识", "id", "text"),
            ColumnSpec("尺度", "scale", "token", hover=True),
            ColumnSpec("类别", "category", "token", hover=True),
            ColumnSpec("状态", "status", "status", hover=True),
            ColumnSpec("起点", "start_index", "index"),
            ColumnSpec("终点", "end_index", "index"),
            ColumnSpec("上轨走向", "upper_course", "token", hover=True),
            ColumnSpec("下轨走向", "lower_course", "token", hover=True),
            ColumnSpec("宽度走向", "width_course", "token", hover=True),
            ColumnSpec("上轨位移", "upper_displacement", "multiple", hover=True),
            ColumnSpec("下轨位移", "lower_displacement", "multiple", hover=True),
            ColumnSpec("宽度比", "width_ratio", "multiple", hover=True),
            ColumnSpec("第一版标签", "legacy_label", "token", hover=True),
        )
        row_columns = (
            ColumnSpec("区间", "zone_id", "text"),
            ColumnSpec("尺度", "scale", "token"),
            ColumnSpec("第一版标签", "legacy_label", "token", hover=True),
            ColumnSpec("上轨位移", "upper_displacement", "multiple", hover=True),
            ColumnSpec("下轨位移", "lower_displacement", "multiple", hover=True),
            ColumnSpec("宽度比", "width_ratio", "multiple", hover=True),
            ColumnSpec("上轨走向", "upper_course", "token", hover=True),
            ColumnSpec("下轨走向", "lower_course", "token", hover=True),
            ColumnSpec("宽度走向", "width_course", "token", hover=True),
            ColumnSpec("类别", "category", "token", hover=True),
            ColumnSpec("可用序号", "available_index", "index"),
        )
        return (
            TableSpec("shapes", "形状对象", shape_columns, appendix=False, source="objects"),
            TableSpec(
                "classifications",
                "分类记录",
                row_columns,
                appendix=True,
                source="extra.classifications",
            ),
        )

    def samples(self) -> Tuple[MethodSample, ...]:
        return (
            MethodSample(
                symbol="SAMPLE-WEDGE",
                title="区间形状：上升楔形",
                blurb="上下轨都上升，宽度收窄，标成上升楔形。同一段里还有第一版标成通道、这里未归类的区间。",
                levels=rising_wedge_levels(),
            ),
            MethodSample(
                symbol="SAMPLE-ASC-TRI",
                title="区间形状：上升三角形",
                blurb="上轨水平、下轨上升，宽度收窄，标成上升三角形。",
                levels=ascending_triangle_levels(),
            ),
            MethodSample(
                symbol="SAMPLE-BROAD",
                title="区间形状：扩张形态",
                blurb="上轨上升、下轨下降，宽度张开，标成扩张形态。",
                levels=broadening_levels(),
            ),
            MethodSample(
                symbol="SAMPLE-SLOW-RISE",
                title="区间形状：缓升通道",
                blurb="两条轨每根都很缓，整段升幅超过水平容差。第一版有的区间标成横向，这里是上升通道。",
                levels=slow_rising_levels(),
            ),
        )


def classify_rails(
    *,
    upper_slope: float,
    lower_slope: float,
    start_index: int,
    end_index: int,
    start_width: float,
    width_ratio: float,
    level_tolerance: float = 0.2,
    converging_width_ratio: float = 0.8,
    diverging_width_ratio: float = 1.25,
) -> dict:
    """按冻结的斜率和起点宽度给一个区间归类。

    位移是这条轨在有效区间内升了多少，以起点宽度为单位。
    宽度走向用第一版存下的宽度比，不另用单根斜率。
    """
    if start_width <= 0:
        raise ValueError("起点宽度必须为正")
    if end_index < start_index:
        raise ValueError("有效区间终点不能早于起点")
    span = int(end_index) - int(start_index)
    upper = float(upper_slope) * span / float(start_width)
    lower = float(lower_slope) * span / float(start_width)
    upper_course = _course(upper, level_tolerance)
    lower_course = _course(lower, level_tolerance)
    width_course = _width_course(width_ratio, converging_width_ratio, diverging_width_ratio)
    return {
        "upper_displacement": upper,
        "lower_displacement": lower,
        "upper_course": upper_course,
        "lower_course": lower_course,
        "width_course": width_course,
        "category": _category(width_course, upper_course, lower_course),
    }


def rising_wedge_levels() -> Tuple[float, ...]:
    return _rail_levels(
        upper=lambda index: 0.0008 * index + 0.10,
        lower=lambda index: 0.0016 * index,
        pairs=6,
        step=8,
    )


def ascending_triangle_levels() -> Tuple[float, ...]:
    return _rail_levels(upper=lambda index: 0.16, lower=lambda index: 0.002 * index, pairs=5, step=8)


def broadening_levels() -> Tuple[float, ...]:
    return _rail_levels(
        upper=lambda index: 0.05 + 0.002 * index,
        lower=lambda index: 0.03 - 0.001 * index,
        pairs=4,
        step=8,
    )


def slow_rising_levels() -> Tuple[float, ...]:
    return _rail_levels(a=0.00025, width=0.02, pairs=8, step=6)


def rising_channel_levels() -> Tuple[float, ...]:
    return _rail_levels()


def _category(width_course: str, upper_course: str, lower_course: str) -> str:
    if width_course == DIVERGING and upper_course in (RISING, LEVEL) and lower_course in (FALLING, LEVEL):
        return BROADENING
    return _BY_COURSES.get((width_course, upper_course, lower_course), UNCLASSIFIED)


def _course(displacement: float, tolerance: float) -> str:
    if displacement > tolerance:
        return RISING
    if displacement < -tolerance:
        return FALLING
    return LEVEL


def _width_course(width_ratio: float, converging: float, diverging: float) -> str:
    if width_ratio < converging:
        return CONVERGING
    if width_ratio > diverging:
        return DIVERGING
    return STABLE


def _shape(zone, shape_id: str, judged: dict) -> dict:
    return {
        "id": shape_id,
        "method": "zone_shape",
        "scale": zone.scale,
        "category": judged["category"],
        "status": zone.status,
        "start_index": int(zone.effective_start),
        "end_index": int(zone.effective_end),
        "start_time": _iso(zone.start_time),
        "end_time": _iso(zone.end_time),
        "available_index": int(zone.confirm_index),
        "available_time": _iso(zone.available_time),
        "revision": int(zone.revision),
        "primary": bool(zone.primary),
        "alternate_of": zone.alternate_of,
        "evidence": {
            "zone_id": zone.id,
            "lower_id": zone.lower_id,
            "upper_id": zone.upper_id,
        },
        "constraints": [item.to_dict() for item in zone.constraints],
        "score": None if zone.score is None else zone.score.to_dict(),
        "stats": None,
        "upper_course": judged["upper_course"],
        "lower_course": judged["lower_course"],
        "width_course": judged["width_course"],
        "upper_displacement": judged["upper_displacement"],
        "lower_displacement": judged["lower_displacement"],
        "width_ratio": zone.width_ratio,
        "legacy_label": zone.label,
    }


def _shape_events(events: Sequence[object], shape_id: str, category_name: str) -> List[dict]:
    copied = []
    for event in events:
        payload = event.to_dict()
        payload["object_id"] = shape_id
        payload["object_kind"] = "zone_shape"
        payload["reason"] = f"{category_name}：{event.reason}"
        copied.append(payload)
    return copied


def _hover_lines(record: dict, category_name: str) -> List[str]:
    status = STATUS_NAME.get(record.get("status"), record.get("status") or "")
    scale = SCALE_LEGEND.get(record.get("scale"), record.get("scale") or "")
    upper = COURSE_NAME.get(record.get("upper_course"), record.get("upper_course") or "")
    lower = COURSE_NAME.get(record.get("lower_course"), record.get("lower_course") or "")
    width = WIDTH_NAME.get(record.get("width_course"), record.get("width_course") or "")
    legacy = LEGACY_NAME.get(record.get("legacy_label"), record.get("legacy_label") or "")
    return [
        f"区间形状 · {category_name}",
        f"{scale} · {status}",
        f"{record.get('start_index')}–{record.get('end_index')}",
        f"上轨{upper} · 下轨{lower} · 宽度{width}",
        (
            f"ru {_num(record.get('upper_displacement'))} · "
            f"rl {_num(record.get('lower_displacement'))} · "
            f"宽度比 {_num(record.get('width_ratio'))}"
        ),
        f"第一版标签 {legacy}",
    ]


def _breakout_index(events: Sequence[dict], object_id: Optional[str]) -> Optional[int]:
    found = None
    for event in events:
        if event.get("object_id") != object_id or event.get("event_type") != EVENT_BREAKOUT:
            continue
        found = event
    if not found or found.get("event_index") is None:
        return None
    return int(found["event_index"])


def _iso(value) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _num(value) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "—"


def _rail_levels(a=0.001, width=0.08, pairs=4, step=6, upper=None, lower=None) -> Tuple[float, ...]:
    """热身之后交替放高点和低点，中间用两条线的中线填充。"""
    warm = [0.0, 0.001]
    warm.extend([0.001] * 28)
    events = []
    cursor = len(warm)
    for _ in range(pairs):
        events.append(("H", cursor))
        cursor += step
        events.append(("L", cursor))
        cursor += step
    confirm = events[-1][1] + step
    levels: List[Optional[float]] = [None] * (confirm + 1)
    for index, value in enumerate(warm):
        levels[index] = value

    def low_at(index: int) -> float:
        return a * index if lower is None else lower(index)

    def high_at(index: int) -> float:
        return a * index + width if upper is None else upper(index)

    for kind, index in events:
        levels[index] = high_at(index) if kind == "H" else low_at(index)
    levels[confirm] = high_at(confirm)
    for index in range(len(warm), len(levels)):
        if levels[index] is None:
            levels[index] = (low_at(index) + high_at(index)) / 2.0
    return tuple(float(item) for item in levels)
