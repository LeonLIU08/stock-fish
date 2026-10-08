"""方向：上升、下降、震荡。

每个尺度、每个无间断区段从第一根逐根推进。只用已确认拐点，
不读临时拐点、基础线段、支撑、压力和趋势区间。
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from analysis.structure.config import (
    LONG,
    MID,
    SCALE_NAMES,
    SHORT,
    direction_params_from_dict,
)
from analysis.structure.methods.base import (
    KIND_RUN,
    CategorySpec,
    ColumnSpec,
    MethodResult,
    MethodSample,
    TableSpec,
    make_mark,
    pivot_ref,
)

HIGHER = "higher"
LOWER = "lower"
FLAT = "flat"
UNKNOWN = "unknown"
UPTREND = "uptrend"
DOWNTREND = "downtrend"
RANGE = "range"
UNDECIDED = "undecided"

RELATION_NAME = {
    HIGHER: "抬高",
    LOWER: "降低",
    FLAT: "持平",
    UNKNOWN: "未知",
}
CATEGORY_NAME = {
    "": "未定",
    UPTREND: "上升趋势",
    DOWNTREND: "下降趋势",
    RANGE: "震荡趋势",
}
STRUCTURE_NAME = {
    UNDECIDED: "未定",
    UPTREND: "上升",
    DOWNTREND: "下降",
    RANGE: "震荡",
}
TRIGGER_NAME = {
    "pivot": "拐点比较",
    "cross": "有效穿越",
    "stall": "停滞",
    "progress": "推进",
    "gap": "间断",
}
CAUSE_NAME = {
    "mixed": "结构不一致",
    "widening": "高点抬高、低点降低",
    "narrowing": "高点降低、低点抬高",
    "trend_break": "趋势被破",
    "stall": "停滞",
}
SCALE_LEGEND = {SHORT: "短尺度", MID: "中尺度", LONG: "长尺度"}
STATUS_NAME = {"open": "进行中", "closed": "已结束"}
STALL_COLOR = "#C8C2B4"
_CAUSE_OF = {
    (HIGHER, LOWER): "widening",
    (LOWER, HIGHER): "narrowing",
}


class DirectionMethod:
    key = "direction"
    name = "方向"
    section = "2.1"
    version = "direction-v1"
    kind = KIND_RUN
    requires = ("closes", "pivots", "volatility")
    categories = (
        CategorySpec(UPTREND, "上升趋势", "#E85D4C"),
        CategorySpec(DOWNTREND, "下降趋势", "#3DDC97"),
        CategorySpec(RANGE, "震荡趋势", "#8A8478"),
    )
    replaces = ()

    def default_params(self) -> dict:
        return direction_params_from_dict(None).to_dict()

    def check_params(self, params: dict) -> None:
        direction_params_from_dict(params)

    def judge(self, context, params: dict, upstream: dict) -> MethodResult:
        del upstream
        pack = direction_params_from_dict(params)
        series = context.series
        volatility = context.volatility
        if volatility is None or len(volatility) != len(series):
            raise ValueError("方向判断需要与 K 线对齐的波动序列")
        confirmed = tuple(context.pivots.confirmed)
        objects: List[dict] = []
        events: List[dict] = []
        comparisons: List[dict] = []
        notes: List[str] = []
        for scale in SCALE_NAMES:
            scale_objects, scale_events, scale_rows = _run_scale(
                series,
                volatility,
                [pivot for pivot in confirmed if pivot.scale == scale],
                scale,
                pack,
            )
            if scale_objects and all(not item.get("category") for item in scale_objects):
                notes.append(f"{SCALE_LEGEND[scale]}样本不足，整段未定。")
            objects.extend(scale_objects)
            events.extend(scale_events)
            comparisons.extend(scale_rows)
        order = {name: index for index, name in enumerate(SCALE_NAMES)}
        objects.sort(key=lambda item: (item["start_index"], order[item["scale"]], item["id"]))
        events.sort(key=lambda item: (item["available_index"], item["object_id"]))
        return MethodResult(
            objects=objects,
            events=events,
            extra={"swing_comparisons": comparisons},
            notes=notes,
        )

    def present(self, record: dict, view: dict) -> List[dict]:
        category = record.get("category") or ""
        if not category:
            return []
        colors = {item.key: item.color for item in self.categories}
        color = STALL_COLOR if record.get("cause") == "stall" else colors.get(category, "#8A8478")
        legend = f"方向 · {SCALE_LEGEND.get(record.get('scale'), record.get('scale'))}"
        hover = _hover_lines(record)
        start = int(record["start_index"])
        end = int(record["end_index"])
        marks = [
            make_mark(
                kind="label_bar",
                object_id=record["id"],
                method=self.key,
                scale=record.get("scale") or "",
                category=category,
                color=color,
                legend=legend,
                start_index=start,
                end_index=end,
                available_index=int(record.get("available_index") or start),
                hover=hover,
                anchor="lane",
            ),
            make_mark(
                kind="marker",
                object_id=record["id"],
                method=self.key,
                scale=record.get("scale") or "",
                category=category,
                color=color,
                legend=legend,
                start_index=start,
                end_index=end,
                available_index=int(record.get("available_index") or start),
                hover=hover,
                anchor="lane",
                index=start,
            ),
        ]
        if category in (UPTREND, DOWNTREND) and record.get("cause") != "stall":
            points = _step_points(record, view.get("price_axis") or "log")
            if points:
                marks.append(
                    make_mark(
                        kind="step_line",
                        object_id=record["id"],
                        method=self.key,
                        scale=record.get("scale") or "",
                        category=category,
                        color=color,
                        legend=legend,
                        start_index=start,
                        end_index=end,
                        available_index=start,
                        points=points,
                        hover=hover,
                        line_type="dashed",
                        show="focus",
                    )
                )
            origin = record.get("structure_origin") or None
            if origin and origin.get("extreme_index") is not None:
                marks.append(
                    make_mark(
                        kind="marker",
                        object_id=record["id"],
                        method=self.key,
                        scale=record.get("scale") or "",
                        category=category,
                        color=color,
                        legend=legend,
                        start_index=start,
                        end_index=end,
                        available_index=start,
                        hover=["结构起点 " + _pivot_text(origin)],
                        anchor="price",
                        show="focus",
                        index=int(origin["extreme_index"]),
                        price=_price(origin.get("geometric_price"), view.get("price_axis") or "log"),
                    )
                )
        return marks

    def tables(self) -> Tuple[TableSpec, ...]:
        segment_columns = (
            ColumnSpec("标识", "id", "text"),
            ColumnSpec("尺度", "scale", "token", hover=True),
            ColumnSpec("类别", "category", "token", hover=True),
            ColumnSpec("结构方向", "structure", "token", hover=True),
            ColumnSpec("状态", "status", "status", hover=True),
            ColumnSpec("起点", "start_index", "index"),
            ColumnSpec("终点", "end_index", "index"),
            ColumnSpec("触发", "trigger", "token", hover=True),
            ColumnSpec("来由", "cause", "token", hover=True),
            ColumnSpec("开始时高点", "swing_high", "token", hover=True),
            ColumnSpec("开始时低点", "swing_low", "token", hover=True),
            ColumnSpec("持续根数", "stats.bar_count", "index", hover=True),
            ColumnSpec("推进次数", "stats.progress_count", "index", hover=True),
            ColumnSpec("未创新高或新低", "stats.failed_swing_count", "index"),
            ColumnSpec("极值收盘价", "stats.extreme_close", "price"),
            ColumnSpec("涨跌幅", "stats.change", "multiple", hover=True),
            ColumnSpec("涨跌幅倍数", "stats.change_multiple", "multiple"),
            ColumnSpec("效率比", "stats.efficiency", "multiple"),
            ColumnSpec("震荡宽度", "stats.range_width", "multiple"),
            ColumnSpec("结束时高点", "stats.swing_high_end", "token"),
            ColumnSpec("结束时低点", "stats.swing_low_end", "token"),
            ColumnSpec("保护价", "protection.-1.geometric_price", "geometric", hover=True),
        )
        comparison_columns = (
            ColumnSpec("尺度", "scale", "token"),
            ColumnSpec("一侧", "side", "token"),
            ColumnSpec("方式", "mode", "token"),
            ColumnSpec("参照极值", "reference.extreme_index", "index"),
            ColumnSpec("新拐点或完成序号", "subject_index", "index"),
            ColumnSpec("d", "d", "multiple", hover=True),
            ColumnSpec("结果", "result", "token", hover=True),
            ColumnSpec("可用序号", "available_index", "index"),
            ColumnSpec("价格越过序号", "price_cross_index", "index"),
        )
        return (
            TableSpec("segments", "方向区段", segment_columns, appendix=False, source="objects"),
            TableSpec(
                "swing_comparisons",
                "摆动比较",
                comparison_columns,
                appendix=True,
                source="extra.swing_comparisons",
            ),
        )

    def samples(self) -> Tuple[MethodSample, ...]:
        return (
            MethodSample(
                symbol="SAMPLE-DIRECTION",
                title="方向：依次抬高",
                blurb="高点和低点依次抬高后，短尺度标成上升趋势。方向条画在收盘价下方。",
                levels=tuple(_sample_rising_levels()),
            ),
        )


def format_direction(result: MethodResult) -> str:
    """给人核对区段的文本。不参与判断。"""
    lines = []
    for item in result.objects:
        lines.append(
            f"{item['id']} {item.get('category') or 'undecided'} "
            f"structure={item.get('structure')} {item['start_index']}-{item['end_index']} "
            f"trigger={item.get('trigger')} cause={item.get('cause')}"
        )
    return "\n".join(lines)


class _Side:
    def __init__(self) -> None:
        self.relation = UNKNOWN
        self.reference = None
        self.pair_left = None
        self.pair_right = None
        self.crossed = False
        self.cross_run = 0
        self.cross_start = None
        self.cross_index = None
        self.updated = False
        self.update_mode = None
        self.last_row = None

    def clear_bar(self) -> None:
        self.updated = False
        self.update_mode = None


class _State:
    def __init__(self) -> None:
        self.region_start = 0
        self.structure = UNDECIDED
        self.high = _Side()
        self.low = _Side()
        self.origin = None
        self.protection = None
        self.level = None
        self.anchor = None
        self.new_pivots: List[Any] = []

    def reset(self, index: int) -> None:
        self.region_start = index
        self.structure = UNDECIDED
        self.high = _Side()
        self.low = _Side()
        self.origin = None
        self.protection = None
        self.level = None
        self.anchor = None
        self.new_pivots = []

    def begin_bar(self) -> None:
        self.high.clear_bar()
        self.low.clear_bar()
        self.new_pivots = []


def _run_scale(series, volatility, pivots: Sequence[Any], scale: str, pack) -> Tuple[List[dict], List[dict], List[dict]]:
    bars = series.bars
    count = len(bars)
    geoms = [bar.geometric for bar in bars]
    closes = [bar.close for bar in bars]
    times = [bar.timestamp.isoformat() for bar in bars]
    by_confirm: Dict[int, List[Any]] = {}
    for pivot in pivots:
        by_confirm.setdefault(int(pivot.confirm_index), []).append(pivot)
    state = _State()
    sealed: List[dict] = []
    events: List[dict] = []
    rows: List[dict] = []
    open_seg: Optional[dict] = None
    stall_n = pack.stall_for(scale)
    floor = float(series.params.volatility_floor)

    def build(index: int, *, trigger, category, structure, cause, label_before, structure_before) -> dict:
        origin = _origin_dict(state.origin) if structure in (UPTREND, DOWNTREND) else None
        protection = []
        if structure in (UPTREND, DOWNTREND) and state.protection is not None:
            protection.append(_protection_row(state.protection, index, times[index], "inherit"))
        is_first = not sealed and not any(item.get("_event_sent") for item in sealed)
        # 还没有任何已经留下的区段时，这一段是序列第一段，不写事件。
        is_first = len(sealed) == 0 and (open_seg is None or not open_seg.get("_event_sent"))
        if open_seg is not None and open_seg.get("start_index") == index and not open_seg.get("_event_sent"):
            is_first = bool(open_seg.get("_is_first"))
        comparisons = []
        price_cross_index = None
        if trigger in ("pivot", "cross"):
            comparisons = _trigger_rows(state)
            price_cross_index = _cross_index(state, trigger)
        return {
            "id": f"direction:{scale}:{index}",
            "method": "direction",
            "scale": scale,
            "category": category,
            "status": "open",
            "start_index": index,
            "end_index": index,
            "start_time": times[index],
            "end_time": times[index],
            "available_index": index,
            "available_time": times[index],
            "revision": None,
            "primary": None,
            "alternate_of": None,
            "evidence": _evidence(state) if trigger in ("pivot", "cross") else [],
            "constraints": [],
            "score": None,
            "structure": structure,
            "trigger": trigger,
            "label_before": label_before or "",
            "structure_before": structure_before or UNDECIDED,
            "structure_origin": origin,
            "protection": protection,
            "cause": cause,
            "swing_high": state.high.relation if category == RANGE else None,
            "swing_low": state.low.relation if category == RANGE else None,
            "comparisons": comparisons,
            "price_cross_index": price_cross_index,
            "_progress": 0,
            "_failed": 0,
            "_swing_high": state.high.relation,
            "_swing_low": state.low.relation,
            "_is_first": is_first,
            "_event_sent": False,
        }

    def seal(segment: dict, end_index: int) -> None:
        if end_index < int(segment["start_index"]):
            return
        segment["end_index"] = end_index
        segment["end_time"] = times[end_index]
        segment["status"] = "closed"
        segment["stats"] = _stats(segment, end_index, geoms, closes, volatility, floor)
        sealed.append(_public(segment))

    def open_new(index: int, **kwargs) -> None:
        nonlocal open_seg
        if open_seg is not None and open_seg["start_index"] == index and not open_seg.get("_event_sent"):
            kwargs["label_before"] = open_seg.get("label_before") or ""
            kwargs["structure_before"] = open_seg.get("structure_before") or UNDECIDED
            inherited_first = bool(open_seg.get("_is_first"))
            open_seg = build(index, **kwargs)
            open_seg["_is_first"] = inherited_first
            return
        label_before = kwargs.get("label_before")
        structure_before = kwargs.get("structure_before")
        if open_seg is not None:
            if label_before is None:
                kwargs["label_before"] = open_seg.get("category") or ""
            if structure_before is None:
                kwargs["structure_before"] = open_seg.get("structure") or UNDECIDED
            seal(open_seg, index - 1)
        open_seg = build(index, **kwargs)

    def mark_kept(segment: dict) -> None:
        if segment.get("_event_sent"):
            return
        segment["_event_sent"] = True
        if segment.get("_is_first") or not segment.get("trigger"):
            return
        index = int(segment["start_index"])
        events.append(
            {
                "event_type": "direction_change",
                "event_index": index,
                "event_time": times[index],
                "available_index": index,
                "available_time": times[index],
                "reason": _reason(
                    segment.get("label_before") or "",
                    segment.get("category") or "",
                    segment.get("trigger"),
                    segment.get("cause"),
                ),
                "reference_close": closes[index],
                "object_id": segment["id"],
                "object_kind": "direction",
                "revision": 0,
                "price_cross_index": segment.get("price_cross_index"),
                "label_before": segment.get("label_before") or "",
                "label_after": segment.get("category") or "",
                "structure_before": segment.get("structure_before") or UNDECIDED,
                "structure_after": segment.get("structure") or UNDECIDED,
                "trigger": segment.get("trigger"),
                "evidence": copy.deepcopy(segment.get("evidence") or []),
            }
        )

    open_seg = build(
        0,
        trigger=None,
        category="",
        structure=UNDECIDED,
        cause=None,
        label_before="",
        structure_before=UNDECIDED,
    )
    open_seg["_is_first"] = True

    for index in range(count):
        if bars[index].gap_before and index > 0:
            state.reset(index)
            open_new(
                index,
                trigger="gap",
                category="",
                structure=UNDECIDED,
                cause=None,
                label_before=None,
                structure_before=None,
            )
        assert open_seg is not None
        state.begin_bar()
        for pivot in by_confirm.get(index, []):
            if int(pivot.extreme_index) < state.region_start or int(pivot.confirm_index) < state.region_start:
                continue
            _take_pivot(state, pivot, index, times[index], pack.epsilon, rows)
        _take_crosses(state, index, geoms, times[index], pack, rows)
        _note_failed(state, open_seg)
        transition = _transfer(state)
        transitioned = transition is not None
        if transition is not None:
            _apply_transition(state, transition, index, geoms)
            open_new(
                index,
                trigger=transition["trigger"],
                category=_category(state.structure),
                structure=state.structure,
                cause=transition.get("cause"),
                label_before=None,
                structure_before=None,
            )
        else:
            _replace_protection(state, open_seg, index, times[index])
        assert open_seg is not None
        progressed = _try_progress(state, index, geoms, volatility, pack.epsilon, floor)
        if progressed and open_seg.get("category") == RANGE and state.structure in (UPTREND, DOWNTREND):
            open_new(
                index,
                trigger="progress",
                category=state.structure,
                structure=state.structure,
                cause=None,
                label_before=None,
                structure_before=None,
            )
            open_seg["_progress"] = 1
        elif (not transitioned) and _should_stall(state, index, progressed, open_seg.get("category") or "", stall_n):
            open_new(
                index,
                trigger="stall",
                category=RANGE,
                structure=state.structure,
                cause="stall",
                label_before=None,
                structure_before=None,
            )
        elif progressed:
            open_seg["_progress"] = int(open_seg.get("_progress") or 0) + 1
        open_seg["end_index"] = index
        open_seg["end_time"] = times[index]
        open_seg["_swing_high"] = state.high.relation
        open_seg["_swing_low"] = state.low.relation
        mark_kept(open_seg)

    if open_seg is not None:
        open_seg["status"] = "open"
        open_seg["stats"] = _stats(open_seg, int(open_seg["end_index"]), geoms, closes, volatility, floor)
        mark_kept(open_seg)
        sealed.append(_public(open_seg))
    return sealed, events, rows


def _take_pivot(state: _State, pivot, index: int, time_text: str, epsilon: float, rows: List[dict]) -> None:
    side = state.high if pivot.role == "high" else state.low
    side.cross_run = 0
    side.cross_start = None
    side.crossed = False
    side.cross_index = None
    if side.reference is not None:
        denominator = float(side.reference.volatility)
        delta = (float(pivot.geometric_price) - float(side.reference.geometric_price)) / denominator
        result = _classify(delta, epsilon)
        row = _comparison_row(
            pivot.scale,
            pivot.role,
            "pivot",
            side.reference,
            int(pivot.extreme_index),
            pivot,
            delta,
            result,
            index,
            time_text,
            None,
        )
        rows.append(row)
        side.pair_left = side.reference
        side.pair_right = pivot
        side.last_row = row
        if result != side.relation:
            side.relation = result
            side.updated = True
            side.update_mode = "pivot"
    side.reference = pivot
    state.new_pivots.append(pivot)


def _take_crosses(state: _State, index: int, geoms: Sequence[float], time_text: str, pack, rows: List[dict]) -> None:
    for side, role, want in ((state.high, "high", HIGHER), (state.low, "low", LOWER)):
        if any(pivot.role == role for pivot in state.new_pivots):
            continue
        reference = side.reference
        if reference is None or side.crossed or index <= int(reference.confirm_index):
            continue
        gap = geoms[index] - float(reference.geometric_price)
        span = pack.cross_buffer * float(reference.volatility)
        ok = gap > span if role == "high" else -gap > span
        if not ok:
            side.cross_run = 0
            side.cross_start = None
            continue
        if side.cross_run == 0:
            side.cross_start = index
        side.cross_run += 1
        if side.cross_run < pack.cross_bars:
            continue
        delta = gap / float(reference.volatility)
        row = _comparison_row(
            reference.scale,
            role,
            "cross",
            reference,
            index,
            None,
            delta,
            want,
            index,
            time_text,
            side.cross_start,
        )
        rows.append(row)
        side.crossed = True
        side.cross_index = side.cross_start
        side.last_row = row
        if side.relation != want:
            side.relation = want
            side.updated = True
            side.update_mode = "cross"


def _note_failed(state: _State, segment: dict) -> None:
    if state.structure == UPTREND:
        for pivot in state.new_pivots:
            if pivot.role == "high" and state.high.pair_right is pivot and state.high.relation in (LOWER, FLAT):
                segment["_failed"] = int(segment.get("_failed") or 0) + 1
    elif state.structure == DOWNTREND:
        for pivot in state.new_pivots:
            if pivot.role == "low" and state.low.pair_right is pivot and state.low.relation in (HIGHER, FLAT):
                segment["_failed"] = int(segment.get("_failed") or 0) + 1


def _transfer(state: _State) -> Optional[dict]:
    if not state.high.updated and not state.low.updated:
        return None
    high = state.high.relation
    low = state.low.relation
    structure = state.structure
    if structure in (UNDECIDED, RANGE):
        if high == UNKNOWN or low == UNKNOWN:
            return None
        if high == HIGHER and low == HIGHER:
            if state.low.pair_left is None:
                return None
            return {"structure": UPTREND, "cause": None, "trigger": _trigger_mode(state)}
        if high == LOWER and low == LOWER:
            if state.high.pair_left is None:
                return None
            return {"structure": DOWNTREND, "cause": None, "trigger": _trigger_mode(state)}
        if structure == RANGE:
            return None
        return {"structure": RANGE, "cause": _cause(high, low), "trigger": _trigger_mode(state)}
    if structure == UPTREND:
        if state.low.updated and low == LOWER:
            if high == LOWER:
                if state.high.pair_left is None:
                    return None
                return {"structure": DOWNTREND, "cause": None, "trigger": _trigger_mode(state)}
            return {"structure": RANGE, "cause": "trend_break", "trigger": _trigger_mode(state)}
        return None
    if structure == DOWNTREND:
        if state.high.updated and high == HIGHER:
            if low == HIGHER:
                if state.low.pair_left is None:
                    return None
                return {"structure": UPTREND, "cause": None, "trigger": _trigger_mode(state)}
            return {"structure": RANGE, "cause": "trend_break", "trigger": _trigger_mode(state)}
        return None
    return None


def _apply_transition(state: _State, transition: dict, index: int, geoms: Sequence[float]) -> None:
    state.structure = transition["structure"]
    if state.structure == UPTREND and state.low.pair_left is not None:
        state.origin = state.low.pair_left
        state.protection = state.low.pair_right
    elif state.structure == DOWNTREND and state.high.pair_left is not None:
        state.origin = state.high.pair_left
        state.protection = state.high.pair_right
    else:
        state.origin = None
        state.protection = None
    _arm_progress(state, index, geoms)


def _replace_protection(state: _State, segment: dict, index: int, time_text: str) -> None:
    if state.structure == UPTREND:
        candidates = [pivot for pivot in state.new_pivots if pivot.role == "low"]
        relation = state.low.relation
        current_ok = relation in (HIGHER, FLAT)
    elif state.structure == DOWNTREND:
        candidates = [pivot for pivot in state.new_pivots if pivot.role == "high"]
        relation = state.high.relation
        current_ok = relation in (LOWER, FLAT)
    else:
        return
    if not current_ok:
        return
    for pivot in candidates:
        if _same_pivot(state.protection, pivot):
            continue
        state.protection = pivot
        segment.setdefault("protection", []).append(_protection_row(pivot, index, time_text, "replace"))


def _arm_progress(state: _State, index: int, geoms: Sequence[float]) -> None:
    if state.structure == UPTREND:
        pivot = state.high.reference
        want_max = True
    elif state.structure == DOWNTREND:
        pivot = state.low.reference
        want_max = False
    else:
        state.level = None
        state.anchor = None
        return
    if pivot is None:
        state.level = None
        state.anchor = index
        return
    start = min(int(pivot.extreme_index), index)
    best_index = start
    best = geoms[start]
    for cursor in range(start, index + 1):
        value = geoms[cursor]
        if want_max and value >= best:
            best = value
            best_index = cursor
        elif not want_max and value <= best:
            best = value
            best_index = cursor
    state.level = (best, best_index)
    state.anchor = index


def _try_progress(state: _State, index: int, geoms, volatility, epsilon: float, floor: float) -> bool:
    if state.structure not in (UPTREND, DOWNTREND) or state.level is None:
        return False
    level, place = state.level
    scale = _volatility_at(volatility, place, floor)
    value = geoms[index]
    moved = value > level + epsilon * scale if state.structure == UPTREND else value < level - epsilon * scale
    if not moved:
        return False
    state.level = (value, index)
    state.anchor = index
    return True


def _should_stall(state: _State, index: int, progressed: bool, category: str, stall_n: int) -> bool:
    if progressed or state.structure not in (UPTREND, DOWNTREND):
        return False
    if category != state.structure or state.anchor is None:
        return False
    return (index - int(state.anchor) + 1) >= int(stall_n)


def _volatility_at(volatility: Sequence[Optional[float]], index: int, floor: float) -> float:
    if index < 0 or index >= len(volatility) or volatility[index] is None:
        return floor
    return float(volatility[index])


def _stats(segment: dict, end_index: int, geoms, closes, volatility, floor: float) -> dict:
    start = int(segment["start_index"])
    stats: Dict[str, Any] = {"bar_count": end_index - start + 1}
    structure = segment.get("structure")
    origin = segment.get("structure_origin")
    if structure in (UPTREND, DOWNTREND) and origin:
        extreme, extreme_close = _segment_extreme(geoms, closes, start, end_index, structure)
        change = extreme - float(origin["geometric_price"])
        denominator = float(origin.get("volatility") or 0.0)
        stats.update(
            {
                "progress_count": int(segment.get("_progress") or 0),
                "extreme_close": extreme_close,
                "failed_swing_count": int(segment.get("_failed") or 0),
                "change": change,
                "change_multiple": (change / denominator) if denominator else None,
                "efficiency": _efficiency(geoms, int(origin["extreme_index"]), end_index),
            }
        )
    if segment.get("category") == RANGE:
        window = geoms[start : end_index + 1]
        scale = _volatility_at(volatility, start, floor)
        stats["range_width"] = (max(window) - min(window)) / scale if scale else None
        stats["swing_high_end"] = segment.get("_swing_high")
        stats["swing_low_end"] = segment.get("_swing_low")
    return stats


def _segment_extreme(geoms, closes, start: int, end: int, structure: str) -> Tuple[float, float]:
    best_index = start
    best = geoms[start]
    for cursor in range(start, end + 1):
        value = geoms[cursor]
        if structure == UPTREND and value >= best:
            best = value
            best_index = cursor
        elif structure == DOWNTREND and value <= best:
            best = value
            best_index = cursor
    return best, closes[best_index]


def _efficiency(geoms, origin_index: int, end_index: int) -> Optional[float]:
    if origin_index < 0 or end_index <= origin_index or end_index >= len(geoms):
        return None
    total = 0.0
    for cursor in range(origin_index + 1, end_index + 1):
        total += abs(geoms[cursor] - geoms[cursor - 1])
    if total == 0.0:
        return None
    return abs(geoms[end_index] - geoms[origin_index]) / total


def _public(segment: dict) -> dict:
    return copy.deepcopy({key: value for key, value in segment.items() if not key.startswith("_")})


def _classify(delta: float, epsilon: float) -> str:
    if delta > epsilon:
        return HIGHER
    if delta < -epsilon:
        return LOWER
    return FLAT


def _cause(high: str, low: str) -> str:
    return _CAUSE_OF.get((high, low), "mixed")


def _category(structure: str) -> str:
    if structure == UNDECIDED:
        return ""
    return structure


def _trigger_mode(state: _State) -> str:
    modes = []
    if state.high.updated and state.high.update_mode:
        modes.append(state.high.update_mode)
    if state.low.updated and state.low.update_mode:
        modes.append(state.low.update_mode)
    if "pivot" in modes:
        return "pivot"
    if "cross" in modes:
        return "cross"
    return "pivot"


def _cross_index(state: _State, trigger: str) -> Optional[int]:
    if trigger != "cross":
        return None
    for side in (state.high, state.low):
        if side.updated and side.update_mode == "cross" and side.cross_index is not None:
            return int(side.cross_index)
    return None


def _trigger_rows(state: _State) -> List[dict]:
    rows = []
    for side in (state.high, state.low):
        if side.last_row is not None:
            rows.append(copy.deepcopy(side.last_row))
    return rows


def _evidence(state: _State) -> List[dict]:
    found = []
    seen = set()
    for side in (state.high, state.low):
        pivots = [side.pair_left, side.pair_right]
        if side.update_mode == "cross":
            pivots.append(side.reference)
        for pivot in pivots:
            if pivot is None:
                continue
            key = (pivot.scale, pivot.role, int(pivot.extreme_index))
            if key in seen:
                continue
            seen.add(key)
            found.append(pivot_ref(pivot))
    return found


def _comparison_row(scale, side, mode, reference, subject_index, subject, delta, result, index, time_text, cross_index) -> dict:
    return {
        "scale": scale,
        "side": side,
        "mode": mode,
        "reference": pivot_ref(reference),
        "subject_index": int(subject_index),
        "subject_pivot": None if subject is None else pivot_ref(subject),
        "d": float(delta),
        "result": result,
        "available_index": int(index),
        "available_time": time_text,
        "price_cross_index": None if cross_index is None else int(cross_index),
    }


def _origin_dict(pivot) -> Optional[dict]:
    if pivot is None:
        return None
    return {
        "scale": pivot.scale,
        "role": pivot.role,
        "extreme_index": int(pivot.extreme_index),
        "geometric_price": float(pivot.geometric_price),
        "volatility": float(pivot.volatility),
    }


def _protection_row(pivot, index: int, time_text: str, source: str) -> dict:
    return {
        "available_index": int(index),
        "available_time": time_text,
        "pivot": pivot_ref(pivot),
        "geometric_price": float(pivot.geometric_price),
        "source": source,
    }


def _same_pivot(left, right) -> bool:
    if left is None or right is None:
        return False
    return (
        left.scale == right.scale
        and left.role == right.role
        and int(left.extreme_index) == int(right.extreme_index)
    )


def _reason(label_before: str, label_after: str, trigger: str, cause: Optional[str]) -> str:
    text = (
        f"{CATEGORY_NAME.get(label_before or '', '未定')}改为"
        f"{CATEGORY_NAME.get(label_after or '', '未定')}，"
        f"触发是{TRIGGER_NAME.get(trigger, trigger)}"
    )
    if cause:
        text += f"，来由是{CAUSE_NAME.get(cause, cause)}"
    return text


def _hover_lines(record: dict) -> List[str]:
    category = CATEGORY_NAME.get(record.get("category") or "", "未定")
    scale = SCALE_LEGEND.get(record.get("scale"), record.get("scale"))
    status = STATUS_NAME.get(record.get("status"), record.get("status") or "")
    lines = [f"方向 · {category}", f"{scale} · {status}", f"序号 {record.get('start_index')}–{record.get('end_index')}"]
    if record.get("trigger"):
        before = CATEGORY_NAME.get(record.get("label_before") or "", "未定")
        lines.append(f"{before} → {category}")
        lines.append("触发：" + TRIGGER_NAME.get(record.get("trigger"), record.get("trigger")))
    if record.get("cause"):
        lines.append("来由：" + CAUSE_NAME.get(record.get("cause"), record.get("cause")))
    structure = STRUCTURE_NAME.get(record.get("structure"), record.get("structure"))
    if structure:
        lines.append("结构方向：" + structure)
    for row in record.get("comparisons") or []:
        side = "高点" if row.get("side") == "high" else "低点"
        mode = "拐点比较" if row.get("mode") == "pivot" else "有效穿越"
        lines.append(f"{side}{mode} d={float(row.get('d')):.4f} {RELATION_NAME.get(row.get('result'), row.get('result'))}")
        ref = row.get("reference") or {}
        lines.append(f"参照极值 {ref.get('extreme_index')}")
    if record.get("price_cross_index") is not None:
        lines.append(f"价格越过序号 {record.get('price_cross_index')}")
    return lines


def _step_points(record: dict, price_axis: str) -> List[List[float]]:
    rows = list(record.get("protection") or [])
    if not rows:
        return []
    end = int(record["end_index"])
    points: List[List[float]] = []
    for index, row in enumerate(rows):
        price = _price(row.get("geometric_price"), price_axis)
        if price is None:
            continue
        start = int(row["available_index"])
        if points:
            points.append([start, points[-1][1]])
            points.append([start, price])
        else:
            points.append([start, price])
        if index == len(rows) - 1:
            points.append([end, price])
    return points


def _price(value, price_axis: str) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    if price_axis == "log":
        return math.exp(number)
    return number


def _pivot_text(origin: dict) -> str:
    role = "高点" if origin.get("role") == "high" else "低点"
    return f"{role}@{origin.get('extreme_index')}"


def _sample_rising_levels() -> List[float]:
    levels = [0.0, 0.001]
    levels.extend([0.001] * 28)
    for value in (0.05, 0.01, 0.09, 0.04, 0.13, 0.08, 0.17):
        levels.extend([value] * 4)
    levels.extend([0.2] * 4)
    return levels
