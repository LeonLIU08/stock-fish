"""次级折返：中尺度上逆着主趋势的基础线段。

不重新解释结构方向，只读主趋势已经写出的类别。回撤比例和时长只记录，
不参与筛选。道氏理论的 1/3 到 2/3 写成对照标记。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from analysis.structure.config import MID
from analysis.structure.methods.base import (
    KIND_INSTANCE,
    CategorySpec,
    ColumnSpec,
    MethodResult,
    TableSpec,
    make_mark,
)
from analysis.structure.segments import DIRECTION_DOWN, DIRECTION_UP

RETRACEMENT = "retracement"
_AGAINST = {"uptrend": DIRECTION_DOWN, "downtrend": DIRECTION_UP}
_WITH = {"uptrend": DIRECTION_UP, "downtrend": DIRECTION_DOWN}
_DIRECTION_NAME = {DIRECTION_UP: "上升", DIRECTION_DOWN: "下降", "flat": "走平"}
_MAJOR_NAME = {"uptrend": "上升趋势", "downtrend": "下降趋势", "range": "震荡趋势", "": "未定"}
_DOW_INSIDE = "dow_inside"
_DOW_OUTSIDE = "dow_outside"
_NOTE = "只保留逆着主趋势的中尺度基础线段。回撤比例和 1/3 到 2/3 不参与筛选。"


class RetracementMethod:
    key = "retracement"
    name = "次级折返"
    section = "2.4"
    version = "retracement-v1"
    kind = KIND_INSTANCE
    requires = ("segments", "major_trend")
    categories = (CategorySpec(RETRACEMENT, "次级折返", "#E0A045"),)
    replaces = ()

    def default_params(self) -> dict:
        return {}

    def check_params(self, params: dict) -> None:
        if not isinstance(params, dict):
            raise ValueError("次级折返参数必须是映射")
        if params:
            raise ValueError("次级折返没有可调参数")

    def judge(self, context, params: dict, upstream: dict) -> MethodResult:
        self.check_params(params or {})
        record = (upstream or {}).get("major_trend") or {}
        if not record or record.get("status") not in (None, "completed"):
            return MethodResult(notes=[_NOTE, "没有主趋势结果，不产生次级折返。"])
        majors = [item for item in record.get("objects") or [] if item.get("scale") == "long"]
        objects: List[dict] = []
        events: List[dict] = []
        segments = _segments(context, MID)
        bars = list(context.series.bars)
        for index, segment in enumerate(segments):
            major = _covering(majors, int(segment.start.extreme_index))
            category = (major or {}).get("category") or ""
            if segment.direction != _AGAINST.get(category):
                continue
            previous = segments[index - 1] if index else None
            ratio, prior_change = _ratio(context.series, segment, previous, category)
            obj = _object(segment, major, category, ratio, prior_change)
            objects.append(obj)
            events.append(_event(bars, obj))
        objects.sort(key=lambda item: (item["start_index"], item["id"]))
        events.sort(key=lambda item: (item["available_index"], item["object_id"]))
        return MethodResult(objects=objects, events=events, notes=[_NOTE])

    def present(self, record: dict, view: dict) -> List[dict]:
        del view
        if record.get("category") != RETRACEMENT or not record.get("primary"):
            return []
        start_price = record.get("start_price")
        end_price = record.get("end_price")
        if not _price_ok(start_price) or not _price_ok(end_price):
            return []
        start = int(record["start_index"])
        end = int(record["end_index"])
        hover = _hover(record)
        shared = dict(
            object_id=record["id"],
            method=self.key,
            scale=MID,
            category=RETRACEMENT,
            color="#E0A045",
            legend="次级折返",
            start_index=start,
            end_index=end,
            available_index=int(record.get("available_index") or end),
            hover=hover,
        )
        return [
            make_mark(kind="segment", points=[[start, start_price], [end, end_price]], **shared),
            make_mark(kind="marker", anchor="price", index=end, price=end_price, **shared),
            make_mark(kind="text", text="折返", points=[[end, end_price]], **shared),
        ]

    def tables(self) -> Tuple[TableSpec, ...]:
        columns = (
            ColumnSpec("标识", "id", "text"),
            ColumnSpec("类别", "category", "token", hover=True),
            ColumnSpec("状态", "status", "status", hover=True),
            ColumnSpec("起点", "start_index", "index"),
            ColumnSpec("终点", "end_index", "index"),
            ColumnSpec("线段方向", "segment_direction_name", "text", hover=True),
            ColumnSpec("主趋势", "major_category", "token", hover=True),
            ColumnSpec("回撤比例", "retracement_ratio", "multiple", hover=True),
            ColumnSpec("跨越根数", "bars", "index", hover=True),
            ColumnSpec("自然日", "calendar_days", "index", hover=True),
            ColumnSpec("道氏对照", "dow_band", "token", hover=True),
            ColumnSpec("几何变动", "change", "multiple"),
        )
        return (TableSpec("retracements", "次级折返", columns),)

    def samples(self) -> Tuple:
        return ()


def _segments(context, scale: str) -> List:
    result = context.segments
    items = getattr(result, "segments", result) or ()
    chosen = [item for item in items if getattr(item, "scale", None) == scale]
    chosen.sort(key=lambda item: (item.start.extreme_index, item.end.extreme_index, item.id))
    return chosen


def _covering(majors: Sequence[dict], index: int) -> Optional[dict]:
    found = [
        item
        for item in majors
        if int(item.get("start_index") or 0) <= index <= int(item.get("end_index") or 0)
    ]
    if not found:
        return None
    found.sort(key=lambda item: int(item["start_index"]))
    return found[-1]


def _ratio(series, segment, previous, major_category: str):
    expected = _WITH.get(major_category)
    if previous is None or expected is None or previous.direction != expected:
        return None, None
    if _separated(series, previous, segment):
        return None, None
    prior = abs(float(previous.change))
    if prior == 0.0:
        return None, None
    return abs(float(segment.change)) / prior, float(previous.change)


def _separated(series, previous, current) -> bool:
    left = int(previous.end.extreme_index)
    right = int(current.start.extreme_index)
    if right < left:
        return True
    bars = series.bars
    for index in range(left + 1, right + 1):
        if 0 <= index < len(bars) and bars[index].gap_before:
            return True
    return False


def _object(segment, major, major_category: str, ratio, prior_change) -> dict:
    start = segment.start
    end = segment.end
    detail = (
        f"主趋势{_MAJOR_NAME.get(major_category, major_category)}，"
        f"中尺度线段{_DIRECTION_NAME.get(segment.direction, segment.direction)}"
    )
    return {
        "id": f"retracement:{segment.id}",
        "method": "retracement",
        "scale": MID,
        "category": RETRACEMENT,
        "status": "validated",
        "start_index": int(start.extreme_index),
        "end_index": int(end.extreme_index),
        "start_time": start.extreme_time.isoformat(),
        "end_time": end.extreme_time.isoformat(),
        "available_index": int(end.confirm_index),
        "available_time": end.available_time.isoformat(),
        "revision": 0,
        "primary": True,
        "alternate_of": None,
        "evidence": [
            {"object_kind": "segment", "id": segment.id},
            {"object_kind": "major_trend", "id": None if major is None else major.get("id")},
        ],
        "constraints": [{"name": "against_major", "passed": True, "detail": detail}],
        "score": None,
        "stats": {},
        "segment_direction": segment.direction,
        "segment_direction_name": _DIRECTION_NAME.get(segment.direction, segment.direction),
        "major_category": major_category,
        "major_id": None if major is None else major.get("id"),
        "change": float(segment.change),
        "prior_change": prior_change,
        "retracement_ratio": ratio,
        "bars": int(segment.bar_span),
        "calendar_days": (end.extreme_time.date() - start.extreme_time.date()).days,
        "dow_band": _dow_band(ratio),
        "start_price": float(start.price),
        "end_price": float(end.price),
    }


def _dow_band(ratio) -> Optional[str]:
    if ratio is None:
        return None
    if (1.0 / 3.0) <= float(ratio) <= (2.0 / 3.0):
        return _DOW_INSIDE
    return _DOW_OUTSIDE


def _event(bars, obj: dict) -> dict:
    index = int(obj["available_index"])
    bar = bars[index]
    return {
        "event_type": "validated",
        "event_index": index,
        "event_time": bar.timestamp.isoformat(),
        "available_index": index,
        "available_time": bar.timestamp.isoformat(),
        "reason": "次级折返",
        "reference_close": bar.close,
        "object_id": obj["id"],
        "object_kind": "retracement",
        "revision": 0,
        "price_cross_index": None,
    }


def _hover(record: dict) -> List[str]:
    ratio = record.get("retracement_ratio")
    ratio_text = "—" if ratio is None else f"{float(ratio):.3f}"
    band = {"dow_inside": "1/3～2/3", "dow_outside": "不在 1/3～2/3"}.get(record.get("dow_band") or "", "—")
    return [
        "次级折返",
        f"主趋势 {_MAJOR_NAME.get(record.get('major_category'), record.get('major_category'))}",
        f"线段{record.get('segment_direction_name') or ''}",
        f"回撤比例 {ratio_text} · {band}",
        f"{record.get('bars')} 根 · {record.get('calendar_days')} 个自然日",
    ]


def _price_ok(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number > 0.0
