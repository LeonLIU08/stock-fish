"""短期波动：每条已确认的短尺度基础线段一个对象。

不按主趋势筛选，也不另判突破或过期。方向、幅度和跨越根数从线段原样记下。
"""
from __future__ import annotations

from typing import List, Tuple

from analysis.structure.config import SHORT
from analysis.structure.methods.base import (
    KIND_INSTANCE,
    CategorySpec,
    ColumnSpec,
    MethodResult,
    TableSpec,
    make_mark,
)
from analysis.structure.segments import DIRECTION_DOWN, DIRECTION_FLAT, DIRECTION_UP

MINOR_SWING = "minor_swing"
_DIRECTION_NAME = {DIRECTION_UP: "上升", DIRECTION_DOWN: "下降", DIRECTION_FLAT: "走平"}
_DIRECTION_COLOR = {DIRECTION_UP: "#E85D4C", DIRECTION_DOWN: "#3DDC97", DIRECTION_FLAT: "#8A8478"}
_NOTE = "每条已确认的短尺度基础线段一个对象，不按主趋势筛选。"


class MinorSwingMethod:
    key = "minor_swing"
    name = "短期波动"
    section = "2.4"
    version = "minor_swing-v1"
    kind = KIND_INSTANCE
    requires = ("segments",)
    categories = (CategorySpec(MINOR_SWING, "短期波动", "#C4B47A"),)
    replaces = ()

    def default_params(self) -> dict:
        return {}

    def check_params(self, params: dict) -> None:
        if not isinstance(params, dict):
            raise ValueError("短期波动参数必须是映射")
        if params:
            raise ValueError("短期波动没有可调参数")

    def judge(self, context, params: dict, upstream: dict) -> MethodResult:
        del upstream
        self.check_params(params or {})
        objects: List[dict] = []
        events: List[dict] = []
        bars = list(context.series.bars)
        for segment in _segments(context):
            obj = _object(segment)
            objects.append(obj)
            if 0 <= int(obj["available_index"]) < len(bars):
                events.append(_event(bars, obj))
        objects.sort(key=lambda item: (item["start_index"], item["id"]))
        events.sort(key=lambda item: (item["available_index"], item["object_id"]))
        return MethodResult(objects=objects, events=events, notes=[_NOTE])

    def present(self, record: dict, view: dict) -> List[dict]:
        del view
        if record.get("category") != MINOR_SWING or not record.get("primary"):
            return []
        start_price = record.get("start_price")
        end_price = record.get("end_price")
        if not _price_ok(start_price) or not _price_ok(end_price):
            return []
        start = int(record["start_index"])
        end = int(record["end_index"])
        direction = record.get("segment_direction") or DIRECTION_FLAT
        color = _DIRECTION_COLOR.get(direction, "#C4B47A")
        hover = [
            "短期波动",
            f"线段{record.get('segment_direction_name') or ''}",
            f"序号 {start}–{end}",
            f"{record.get('bars')} 根",
        ]
        shared = dict(
            object_id=record["id"],
            method=self.key,
            scale=SHORT,
            category=MINOR_SWING,
            color=color,
            legend="短期波动",
            start_index=start,
            end_index=end,
            available_index=int(record.get("available_index") or end),
            hover=hover,
        )
        return [
            make_mark(kind="segment", points=[[start, start_price], [end, end_price]], **shared),
            make_mark(kind="marker", anchor="price", index=end, price=end_price, **shared),
        ]

    def tables(self) -> Tuple[TableSpec, ...]:
        columns = (
            ColumnSpec("标识", "id", "text"),
            ColumnSpec("类别", "category", "token", hover=True),
            ColumnSpec("线段方向", "segment_direction_name", "text", hover=True),
            ColumnSpec("状态", "status", "status", hover=True),
            ColumnSpec("起点", "start_index", "index"),
            ColumnSpec("终点", "end_index", "index"),
            ColumnSpec("跨越根数", "bars", "index", hover=True),
            ColumnSpec("几何变动", "change", "multiple"),
            ColumnSpec("倍数", "change_multiple", "multiple"),
        )
        return (TableSpec("swings", "短期波动", columns),)

    def samples(self) -> Tuple:
        return ()


def _segments(context) -> List:
    result = context.segments
    items = getattr(result, "segments", result) or ()
    chosen = [item for item in items if getattr(item, "scale", None) == SHORT]
    chosen.sort(key=lambda item: (item.start.extreme_index, item.end.extreme_index, item.id))
    return chosen


def _object(segment) -> dict:
    start = segment.start
    end = segment.end
    return {
        "id": f"minor_swing:{segment.id}",
        "method": "minor_swing",
        "scale": SHORT,
        "category": MINOR_SWING,
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
        "evidence": [{"object_kind": "segment", "id": segment.id}],
        "constraints": [
            {
                "name": "confirmed_short_segment",
                "passed": True,
                "detail": "已确认的短尺度基础线段",
            }
        ],
        "score": None,
        "stats": {},
        "segment_direction": segment.direction,
        "segment_direction_name": _DIRECTION_NAME.get(segment.direction, segment.direction),
        "change": float(segment.change),
        "change_multiple": None if segment.change_multiple is None else float(segment.change_multiple),
        "bars": int(segment.bar_span),
        "start_price": float(start.price),
        "end_price": float(end.price),
    }


def _event(bars, obj: dict) -> dict:
    index = int(obj["available_index"])
    bar = bars[index]
    return {
        "event_type": "validated",
        "event_index": index,
        "event_time": bar.timestamp.isoformat(),
        "available_index": index,
        "available_time": bar.timestamp.isoformat(),
        "reason": "短期波动",
        "reference_close": bar.close,
        "object_id": obj["id"],
        "object_kind": "minor_swing",
        "revision": 0,
        "price_cross_index": None,
    }


def _price_ok(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number > 0.0
