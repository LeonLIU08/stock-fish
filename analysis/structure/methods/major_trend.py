"""主趋势：长尺度每一根的类别等于该根的结构方向。

停滞只改方向标签，不改结构方向，所以不切开主趋势。结构方向变化，
或间断后的第一根，才新开一段。不调用方向的内部函数，只读它写出的区段。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from analysis.structure.config import LONG
from analysis.structure.methods.base import (
    KIND_RUN,
    CategorySpec,
    ColumnSpec,
    MethodResult,
    MethodSample,
    TableSpec,
    make_mark,
)

UPTREND = "uptrend"
DOWNTREND = "downtrend"
RANGE = "range"
UNDECIDED = "undecided"
_STRUCTURES = (UPTREND, DOWNTREND, RANGE, UNDECIDED)
_NAMES = {
    UPTREND: "上升趋势",
    DOWNTREND: "下降趋势",
    RANGE: "震荡趋势",
    UNDECIDED: "未定",
    "": "未定",
}
_STATUS = {"open": "进行中", "closed": "已结束"}
_TRIGGER = {"structure": "结构方向变化", "gap": "间断"}
_NOTE = "只读长尺度结构方向。停滞把方向标签改成震荡时，主趋势不切开。"


class MajorTrendMethod:
    key = "major_trend"
    name = "主趋势"
    section = "2.4"
    version = "major_trend-v1"
    kind = KIND_RUN
    requires = ("direction",)
    categories = (
        CategorySpec(UPTREND, "上升趋势", "#E85D4C"),
        CategorySpec(DOWNTREND, "下降趋势", "#3DDC97"),
        CategorySpec(RANGE, "震荡趋势", "#8A8478"),
    )
    replaces = ()

    def default_params(self) -> dict:
        return {}

    def check_params(self, params: dict) -> None:
        _reject_tunables(params, "主趋势")

    def judge(self, context, params: dict, upstream: dict) -> MethodResult:
        self.check_params(params or {})
        series = context.series
        record = (upstream or {}).get("direction") or {}
        direction_objects = list(record.get("objects") or [])
        objects, events, uncovered = _run(series, direction_objects)
        notes = [_NOTE]
        long_ones = [item for item in direction_objects if item.get("scale") == LONG]
        if not long_ones:
            notes.append("没有长尺度方向区段，主趋势按未定覆盖全部 K 线。")
        elif uncovered:
            notes.append("有的 K 线没有落在长尺度方向区段里，这些根按未定计。")
        elif objects and all(not item.get("category") for item in objects):
            notes.append("长尺度样本不足，主趋势整段未定。")
        return MethodResult(objects=objects, events=events, notes=notes)

    def present(self, record: dict, view: dict) -> List[dict]:
        del view
        category = record.get("category") or ""
        colors = {item.key: item.color for item in self.categories}
        if category not in colors:
            return []
        color = colors[category]
        legend = "主趋势 · 长尺度"
        hover = _hover(record)
        start = int(record["start_index"])
        end = int(record["end_index"])
        available = int(record.get("available_index") or start)
        shared = dict(
            object_id=record["id"],
            method=self.key,
            scale=LONG,
            category=category,
            color=color,
            legend=legend,
            start_index=start,
            end_index=end,
            available_index=available,
            hover=hover,
        )
        return [
            make_mark(kind="label_bar", anchor="lane", **shared),
            make_mark(kind="marker", anchor="lane", index=start, **shared),
        ]

    def tables(self) -> Tuple[TableSpec, ...]:
        columns = (
            ColumnSpec("标识", "id", "text"),
            ColumnSpec("尺度", "scale", "token", hover=True),
            ColumnSpec("类别", "category", "token", hover=True),
            ColumnSpec("结构方向", "structure", "token", hover=True),
            ColumnSpec("状态", "status", "status", hover=True),
            ColumnSpec("起点", "start_index", "index"),
            ColumnSpec("终点", "end_index", "index"),
            ColumnSpec("触发", "trigger", "token", hover=True),
            ColumnSpec("方向区段", "direction_id", "text", hover=True),
            ColumnSpec("持续根数", "stats.bar_count", "index", hover=True),
        )
        return (TableSpec("segments", "主趋势区段", columns),)

    def samples(self) -> Tuple[MethodSample, ...]:
        return (
            MethodSample(
                symbol="SAMPLE-LEVEL",
                title="级别：主趋势、折返和短期波动",
                blurb=(
                    "长尺度结构方向画成主趋势。逆着它的中尺度线段是次级折返，"
                    "短尺度线段是短期波动。同一页里方向、区间形状和停顿仍在。"
                ),
                levels=tuple(_sample_levels()),
            ),
        )


def _reject_tunables(params: dict, name: str) -> None:
    if params is None:
        return
    if not isinstance(params, dict):
        raise ValueError(f"{name}参数必须是映射")
    if params:
        raise ValueError(f"{name}没有可调参数")


def _run(series, direction_objects: Sequence[dict]) -> Tuple[List[dict], List[dict], bool]:
    bars = list(series.bars)
    count = len(bars)
    structure, owner, uncovered = _coverage(direction_objects, count)
    objects: List[dict] = []
    events: List[dict] = []
    current: Optional[dict] = None
    for index, bar in enumerate(bars):
        gap = bool(bar.gap_before) and index > 0
        value = structure[index]
        if current is not None and not gap and value == current["structure"]:
            continue
        if current is not None:
            _close(current, index - 1, bars)
            objects.append(current)
        trigger = None if current is None else ("gap" if gap else "structure")
        before = "" if current is None else current["structure"]
        current = _open(index, value, owner[index], trigger, bars)
        if trigger:
            events.append(_event(current, before, bars))
    if current is not None:
        current["end_index"] = count - 1 if count else 0
        current["end_time"] = bars[-1].timestamp.isoformat() if bars else ""
        current["status"] = "open"
        current["stats"] = {"bar_count": count - int(current["start_index"]) if count else 0}
        objects.append(current)
    return objects, events, uncovered


def _coverage(direction_objects: Sequence[dict], count: int) -> Tuple[List[str], List[Optional[str]], bool]:
    structure = [UNDECIDED] * count
    owner: List[Optional[str]] = [None] * count
    seen = [False] * count
    ordered = [item for item in direction_objects if item.get("scale") == LONG]
    ordered.sort(key=lambda item: (int(item.get("start_index") or 0), str(item.get("id") or "")))
    for item in ordered:
        start = int(item.get("start_index") or 0)
        end = int(item.get("end_index") if item.get("end_index") is not None else start)
        value = item.get("structure") or UNDECIDED
        if value not in _STRUCTURES:
            value = UNDECIDED
        for index in range(max(0, start), min(count - 1, end) + 1):
            structure[index] = value
            owner[index] = item.get("id")
            seen[index] = True
    uncovered = count > 0 and not all(seen)
    return structure, owner, uncovered


def _open(index: int, structure: str, direction_id: Optional[str], trigger: Optional[str], bars) -> dict:
    bar = bars[index]
    category = "" if structure == UNDECIDED else structure
    return {
        "id": f"major_trend:{LONG}:{index}",
        "method": "major_trend",
        "scale": LONG,
        "category": category,
        "status": "open",
        "start_index": index,
        "end_index": index,
        "start_time": bar.timestamp.isoformat(),
        "end_time": bar.timestamp.isoformat(),
        "available_index": index,
        "available_time": bar.timestamp.isoformat(),
        "revision": None,
        "primary": None,
        "alternate_of": None,
        "evidence": [{"object_kind": "direction", "id": direction_id}] if direction_id else [],
        "constraints": [],
        "score": None,
        "stats": {"bar_count": 1},
        "structure": structure,
        "trigger": trigger,
        "direction_id": direction_id,
    }


def _close(segment: dict, end_index: int, bars) -> None:
    segment["end_index"] = end_index
    segment["end_time"] = bars[end_index].timestamp.isoformat()
    segment["status"] = "closed"
    segment["stats"] = {"bar_count": end_index - int(segment["start_index"]) + 1}


def _event(segment: dict, before: str, bars) -> dict:
    index = int(segment["start_index"])
    after = segment["structure"]
    return {
        "event_type": "major_trend_change",
        "event_index": index,
        "event_time": bars[index].timestamp.isoformat(),
        "available_index": index,
        "available_time": bars[index].timestamp.isoformat(),
        "reason": f"主趋势：{_NAMES.get(before, before)} → {_NAMES.get(after, after)}",
        "reference_close": bars[index].close,
        "object_id": segment["id"],
        "object_kind": "major_trend",
        "revision": 0,
        "price_cross_index": None,
        "structure_before": before or UNDECIDED,
        "structure_after": after,
        "trigger": segment.get("trigger"),
    }


def _hover(record: dict) -> List[str]:
    category = record.get("category") or ""
    structure = record.get("structure") or UNDECIDED
    trigger = _TRIGGER.get(record.get("trigger") or "", "序列开头")
    status = _STATUS.get(record.get("status"), record.get("status") or "")
    return [
        f"主趋势 · {_NAMES.get(category or structure, structure)}",
        f"长尺度 · {status}",
        f"序号 {record.get('start_index')}–{record.get('end_index')}",
        f"结构方向 {_NAMES.get(structure, structure)} · {trigger}",
    ]


def _sample_levels() -> List[float]:
    levels = [0.0, 0.001]
    levels.extend([0.001] * 28)
    for value in (0.08, 0.03, 0.16, 0.08, 0.24, 0.14, 0.32):
        levels.extend([value] * 6)
    return levels
