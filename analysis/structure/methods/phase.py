"""阶段：同一尺度上，已经确认的结构方向现在走到哪一步。

方向仍由 2.1 确认。这里不重做拐点、推进和穿越，距离用已经发布的 s_t。
方向没写进 requires：方向失败时阶段仍覆盖全部 K 线。
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
    phase_params_from_dict,
)
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

UP_BIRTH = "up_birth"
UP_CONTINUE = "up_continue"
UP_PULLBACK = "up_pullback"
UP_EXHAUSTION = "up_exhaustion"
DOWN_BIRTH = "down_birth"
DOWN_CONTINUE = "down_continue"
DOWN_REBOUND = "down_rebound"
DOWN_EXHAUSTION = "down_exhaustion"
ACCUMULATION = "suspected_accumulation"
DISTRIBUTION = "suspected_distribution"
NEUTRAL = "neutral_range"
PENDING = "transition_pending"
INSUFFICIENT = "insufficient"

_NAMES = {
    UP_BIRTH: "上涨初生",
    UP_CONTINUE: "上涨延续",
    UP_PULLBACK: "上涨回调",
    UP_EXHAUSTION: "上涨衰竭预警",
    DOWN_BIRTH: "下跌初生",
    DOWN_CONTINUE: "下跌延续",
    DOWN_REBOUND: "下跌反弹",
    DOWN_EXHAUSTION: "下跌衰竭预警",
    ACCUMULATION: "疑似吸筹",
    DISTRIBUTION: "疑似派发",
    NEUTRAL: "中性震荡",
    PENDING: "趋势转换待确认",
    INSUFFICIENT: "数据不足或不可判定",
}
_STRUCTURE_NAME = {
    UPTREND: "上升",
    DOWNTREND: "下降",
    RANGE: "震荡",
    UNDECIDED: "未定",
}
_TRIGGER_NAME = {
    "birth": "初生",
    "continue": "延续",
    "pullback": "回调",
    "rebound": "反弹",
    "exhaustion": "衰竭",
    "accumulation": "吸筹",
    "distribution": "派发",
    "neutral": "中性",
    "range_break_up": "向上离开区间",
    "range_break_down": "向下离开区间",
    "unconfirmed": "尚未确认",
    "insufficient": "数据不足",
}
_EXHAUST_NAME = {
    "progress_fade": "推进减弱",
    "swing_retrace": "创高后回落",
    "wick_reject": "多次冲高收回",
    "volume_stall": "放量但不推进",
}
_RANGE_NAME = {
    "prior_trend": "前趋势背景",
    "range_structure": "区间结构",
    "volume_bias": "量价偏向",
    "fake_break": "假突破",
}
_SCALE = {SHORT: "短尺度", MID: "中尺度", LONG: "长尺度"}
_STATUS = {"open": "进行中", "closed": "已结束"}
_IMMEDIATE = {INSUFFICIENT, PENDING, UP_BIRTH, DOWN_BIRTH, UP_EXHAUSTION, DOWN_EXHAUSTION}
_BIRTHS = {UP_BIRTH, DOWN_BIRTH}
_EXHAUSTIONS = {UP_EXHAUSTION, DOWN_EXHAUSTION}
_SUBSTATES = {UP_PULLBACK, DOWN_REBOUND, UP_CONTINUE, DOWN_CONTINUE, ACCUMULATION, DISTRIBUTION, NEUTRAL}

PARAM_LABELS = {
    "fast_span": "快均线",
    "impulse_span": "短推进",
    "record_span": "记录用推进",
    "birth_bars": "初生长度",
    "pullback_multiple": "回调幅度",
    "pullback_impulse": "回调推进",
    "fade_progress": "衰竭旧推进",
    "fade_ratio": "衰竭衰减",
    "swing_window": "创高近窗",
    "swing_retrace": "创高回撤",
    "wick_window": "影线窗口",
    "wick_count": "影线次数",
    "wick_ratio": "影线比例",
    "wick_amplitude": "影线振幅",
    "volume_window": "放量近窗",
    "volume_baseline": "放量基线",
    "volume_multiple": "放量倍数",
    "volume_impulse": "无推进",
    "exhaustion_score": "衰竭分数",
    "exhaustion_bars": "衰竭连续",
    "structure_shift": "结构位移",
    "bias_window": "量价窗口",
    "bias_minimum": "量价阈值",
    "false_break": "假突破深度",
    "false_break_window": "假突破窗口",
    "range_score": "整理分数",
    "score_gap": "分差",
    "break_bars": "突破连续",
    "break_buffer": "突破缓冲",
    "state_bars": "子状态连续",
}


class PhaseMethod:
    key = "phase"
    name = "阶段"
    section = "2.5"
    version = "phase-v1"
    kind = KIND_RUN
    requires = ("closes", "pivots", "volatility")
    categories = (
        CategorySpec(UP_BIRTH, "上涨初生", "#F0A29A"),
        CategorySpec(UP_CONTINUE, "上涨延续", "#C73E3A"),
        CategorySpec(UP_PULLBACK, "上涨回调", "#E0A045"),
        CategorySpec(UP_EXHAUSTION, "上涨衰竭预警", "#8E3B2F"),
        CategorySpec(DOWN_BIRTH, "下跌初生", "#8FE0B8"),
        CategorySpec(DOWN_CONTINUE, "下跌延续", "#1F9D62"),
        CategorySpec(DOWN_REBOUND, "下跌反弹", "#3DDC97"),
        CategorySpec(DOWN_EXHAUSTION, "下跌衰竭预警", "#1D6A4A"),
        CategorySpec(ACCUMULATION, "疑似吸筹", "#4C78A8"),
        CategorySpec(DISTRIBUTION, "疑似派发", "#B279A2"),
        CategorySpec(NEUTRAL, "中性震荡", "#8A8478"),
        CategorySpec(PENDING, "趋势转换待确认", "#C4B47A"),
        CategorySpec(INSUFFICIENT, "数据不足或不可判定", "#D9D3C7"),
    )
    replaces = ()

    def default_params(self) -> dict:
        return phase_params_from_dict(None).to_dict()

    def check_params(self, params: dict) -> None:
        phase_params_from_dict(params)

    def judge(self, context, params: dict, upstream: dict) -> MethodResult:
        pack = phase_params_from_dict(params)
        series = context.series
        volatility = context.volatility
        if volatility is None or len(volatility) != len(series):
            raise ValueError("阶段判断需要与 K 线对齐的波动序列")
        direction = (upstream or {}).get("direction") or {}
        if direction.get("status") not in (None, "completed"):
            direction = {}
        direction_objects = list(direction.get("objects") or [])
        comparisons = list((direction.get("extra") or {}).get("swing_comparisons") or [])
        pivots = _pivot_list(context.pivots)
        notes = _missing_notes(series)
        if not direction_objects:
            notes.append("没有结构方向，各尺度按趋势转换待确认或数据不足覆盖。")
        objects: List[dict] = []
        events: List[dict] = []
        readings: List[dict] = []
        for scale in SCALE_NAMES:
            scale_objects, scale_events, scale_readings = _run_scale(
                series,
                volatility,
                [item for item in direction_objects if item.get("scale") == scale],
                [item for item in comparisons if item.get("scale") == scale],
                [pivot for pivot in pivots if getattr(pivot, "scale", None) == scale],
                scale,
                pack,
            )
            objects.extend(scale_objects)
            events.extend(scale_events)
            readings.extend(scale_readings)
        order = {name: index for index, name in enumerate(SCALE_NAMES)}
        objects.sort(key=lambda item: (item["start_index"], order[item["scale"]], item["id"]))
        events.sort(key=lambda item: (item["available_index"], item["object_id"]))
        readings.sort(key=lambda item: (item["available_index"], order[item["scale"]], item["object_id"]))
        return MethodResult(objects=objects, events=events, extra={"readings": readings}, notes=notes)

    def present(self, record: dict, view: dict) -> List[dict]:
        category = record.get("category") or ""
        colors = {item.key: item.color for item in self.categories}
        if category not in colors:
            return []
        color = colors[category]
        legend = f"阶段 · {_SCALE.get(record.get('scale'), record.get('scale'))}"
        hover = _hover(record)
        start = int(record["start_index"])
        end = int(record["end_index"])
        available = int(record.get("available_index") or start)
        shared = dict(
            object_id=record["id"],
            method=self.key,
            scale=record.get("scale") or "",
            category=category,
            color=color,
            legend=legend,
            start_index=start,
            end_index=end,
            available_index=available,
            hover=hover,
        )
        marks = [
            make_mark(kind="label_bar", anchor="lane", **shared),
            make_mark(kind="marker", anchor="lane", index=start, **shared),
        ]
        bounds = record.get("range_bounds") or None
        if bounds:
            axis = view.get("price_axis") or "log"
            t0 = int(bounds["t0"])
            for price, label in ((bounds.get("upper"), "U"), (bounds.get("lower"), "D")):
                actual = _price(price, axis)
                if actual is None:
                    continue
                marks.append(
                    make_mark(
                        kind="segment",
                        object_id=record["id"],
                        method=self.key,
                        scale=record.get("scale") or "",
                        category=category,
                        color=color,
                        legend=legend,
                        start_index=min(t0, start),
                        end_index=end,
                        available_index=available,
                        points=[[t0, actual], [end, actual]],
                        hover=hover + [f"{label} {actual:.4f}"],
                        line_type="dashed",
                        show="focus",
                    )
                )
            for key, text in (("upper_pivot", "上沿拐点"), ("lower_pivot", "下沿拐点")):
                pivot = bounds.get(key) or {}
                if pivot.get("extreme_index") is None:
                    continue
                geometric = bounds.get("upper") if key == "upper_pivot" else bounds.get("lower")
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
                        available_index=available,
                        hover=[text + " " + _pivot_text(pivot)],
                        anchor="price",
                        show="focus",
                        index=int(pivot["extreme_index"]),
                        price=_price(geometric, axis),
                    )
                )
        return marks

    def tables(self) -> Tuple[TableSpec, ...]:
        columns = (
            ColumnSpec("标识", "id", "text"),
            ColumnSpec("尺度", "scale", "token", hover=True),
            ColumnSpec("主状态", "category", "token", hover=True),
            ColumnSpec("结构方向", "structure", "token", hover=True),
            ColumnSpec("状态", "status", "status", hover=True),
            ColumnSpec("起点", "start_index", "index"),
            ColumnSpec("终点", "end_index", "index"),
            ColumnSpec("触发", "trigger", "token", hover=True),
            ColumnSpec("确认序号", "confirm_index", "index", hover=True),
            ColumnSpec("年龄", "stats.age", "index", hover=True),
            ColumnSpec("方向区段", "direction_id", "text"),
            ColumnSpec("s_t", "stats.s_t", "multiple", hover=True),
            ColumnSpec("推进参照", "stats.reference", "geometric", hover=True),
            ColumnSpec("Z3", "stats.z3", "multiple", hover=True),
            ColumnSpec("衰竭分数", "stats.exhaustion_score", "index"),
            ColumnSpec("吸筹分", "stats.accumulation_score", "index"),
            ColumnSpec("派发分", "stats.distribution_score", "index"),
        )
        reading_columns = (
            ColumnSpec("可用序号", "available_index", "index"),
            ColumnSpec("尺度", "scale", "token"),
            ColumnSpec("主状态", "category", "token", hover=True),
            ColumnSpec("之前", "state_before", "token"),
            ColumnSpec("触发", "trigger", "token", hover=True),
            ColumnSpec("确认序号", "confirm_index", "index"),
            ColumnSpec("年龄", "age", "index", hover=True),
            ColumnSpec("s_t", "s_t", "multiple", hover=True),
            ColumnSpec("推进参照", "reference", "geometric", hover=True),
            ColumnSpec("Z3", "z3", "multiple", hover=True),
            ColumnSpec("衰竭分数", "exhaustion_score", "index"),
            ColumnSpec("吸筹分", "accumulation_score", "index"),
            ColumnSpec("派发分", "distribution_score", "index"),
            ColumnSpec("量价偏向", "volume_bias", "multiple"),
        )
        return (
            TableSpec("segments", "阶段区段", columns),
            TableSpec("readings", "阶段切换", reading_columns, appendix=True, source="extra.readings"),
        )

    def samples(self) -> Tuple[MethodSample, ...]:
        return (
            MethodSample(
                symbol="SAMPLE-PHASE",
                title="阶段：锯齿上涨",
                blurb=(
                    "同一页里方向、区间形状、停顿、级别和阶段都在。"
                    "阶段条在方向条下方，标出上涨初生、延续、回调或衰竭，不另定方向。"
                ),
                levels=tuple(_sample_levels()),
            ),
        )


class _Run:
    def __init__(self) -> None:
        self.confirmed: Optional[str] = None
        self.streak_key: Optional[str] = None
        self.streak = 0
        self.exhaustion = False
        self.high_run = 0
        self.low_run = 0
        self.bounds: Optional[dict] = None
        self.latch: Optional[str] = None
        self.t0: Optional[int] = None
        self.range_tried = False
        self.ema_sum = 0.0
        self.ema_count = 0
        self.ema: Optional[float] = None
        self.ema_ready = False

    def begin_region(self) -> None:
        self._clear_episode()
        self.ema_sum = 0.0
        self.ema_count = 0
        self.ema = None
        self.ema_ready = False

    def begin_structure(self) -> None:
        self._clear_episode()

    def _clear_episode(self) -> None:
        self.exhaustion = False
        self.high_run = 0
        self.low_run = 0
        self.streak_key = None
        self.streak = 0
        self.bounds = None
        self.latch = None
        self.t0 = None
        self.range_tried = False

    def update_ema(self, price: float, span: int) -> Optional[float]:
        if not math.isfinite(price):
            return self.ema if self.ema_ready else None
        if not self.ema_ready:
            self.ema_sum += price
            self.ema_count += 1
            if self.ema_count >= span:
                self.ema = self.ema_sum / float(self.ema_count)
                self.ema_ready = True
            return self.ema if self.ema_ready else None
        alpha = 2.0 / (span + 1.0)
        self.ema = alpha * price + (1.0 - alpha) * float(self.ema)
        return self.ema


def _run_scale(series, volatility, direction_objects, comparisons, pivots, scale: str, pack) -> Tuple[List[dict], List[dict], List[dict]]:
    bars = list(series.bars)
    count = len(bars)
    floor = float(series.params.volatility_floor)
    structures, owners = _coverage(direction_objects, count)
    geoms = [bar.geometric for bar in bars]
    state = _Run()
    objects: List[dict] = []
    events: List[dict] = []
    readings: List[dict] = []
    current: Optional[dict] = None
    prev_structure: Optional[str] = None
    region_start = 0
    run_start = 0
    columns = _columns(bars)
    for index, bar in enumerate(bars):
        new_region = index == 0 or bool(bar.gap_before)
        structure = structures[index]
        if new_region:
            state.begin_region()
            region_start = index
        elif structure != prev_structure:
            state.begin_structure()
        if new_region or prev_structure is None or structure != prev_structure:
            run_start = index
        if structure == RANGE and not state.range_tried:
            state.range_tried = True
            state.t0 = run_start
            state.latch = None
            state.bounds = _freeze(pivots, volatility, run_start, region_start, floor)
        fast = state.update_ema(bar.geometric, pack.fast_span)
        scale_value = _usable_scale(volatility, index, floor)
        if scale_value is None or not math.isfinite(bar.geometric):
            state.high_run = 0
            state.low_run = 0
            detail = _insufficient_detail()
        else:
            detail = _assess(
                state,
                structure,
                index,
                run_start,
                region_start,
                geoms,
                bars,
                volatility,
                floor,
                scale_value,
                fast,
                direction_objects,
                comparisons,
                pivots,
                pack,
                columns,
            )
        structure_changed = prev_structure is not None and structure != prev_structure and not new_region
        chosen = _switch(state, detail["candidate"], structure_changed, not new_region, pack.state_bars)
        if current is None or chosen != current["category"] or new_region or structure_changed:
            if current is not None:
                _close(current, index - 1, bars)
                objects.append(current)
            current = _open(
                scale,
                index,
                bars,
                chosen,
                structure,
                run_start,
                owners,
                direction_objects,
                detail,
                "" if current is None else current["category"],
                state.bounds,
            )
            if objects:
                events.append(_event(current, bars))
                readings.append(_reading(current))
        else:
            _extend(current, index, bars, detail)
        state.confirmed = chosen
        prev_structure = structure
    if current is not None:
        current["end_index"] = count - 1 if count else 0
        current["end_time"] = bars[-1].timestamp.isoformat() if bars else ""
        current["status"] = "open"
        objects.append(current)
    return objects, events, readings


def _assess(
    state: _Run,
    structure: str,
    index: int,
    run_start: int,
    region_start: int,
    geoms: Sequence[float],
    bars,
    volatility,
    floor: float,
    scale_value: float,
    fast: Optional[float],
    direction_objects,
    comparisons,
    pivots,
    pack,
    columns: dict,
) -> dict:
    age = index - run_start + 1 if structure in (UPTREND, DOWNTREND) else None
    z3 = _impulse(geoms, index, pack.impulse_span, region_start, scale_value)
    z5 = _impulse(geoms, index, pack.record_span, region_start, scale_value)
    refs = _progress_refs(direction_objects, structure, run_start, index, region_start)
    reference = None if not refs else float(refs[-1]["geometric_price"])
    previous = None if len(refs) < 2 else float(refs[-2]["geometric_price"])
    close = geoms[index]
    drawdown = None
    if reference is not None and structure == UPTREND:
        drawdown = (reference - close) / scale_value
    elif reference is not None and structure == DOWNTREND:
        drawdown = (close - reference) / scale_value
    bias = _volume_bias(bars, index, region_start, pack.bias_window, columns)
    exhaustion_flags = {
        "progress_fade": False,
        "swing_retrace": False,
        "wick_reject": False,
        "volume_stall": False,
    }
    exhaustion_score = None
    range_flags = {
        "prior_trend": False,
        "range_structure": False,
        "volume_bias": False,
        "fake_break": False,
    }
    accumulation = None
    distribution = None
    if structure in (UPTREND, DOWNTREND):
        exhaustion_flags = _exhaustion_flags(
            structure,
            index,
            region_start,
            close,
            scale_value,
            reference,
            previous,
            drawdown,
            z3,
            bars,
            volatility,
            floor,
            comparisons,
            pack,
            columns,
        )
        exhaustion_score = sum(1 for item in exhaustion_flags.values() if item)
        _update_exhaustion(state, exhaustion_score, pack.exhaustion_score, pack.exhaustion_bars)
        candidate, trigger = _trend_candidate(
            structure,
            state.exhaustion,
            age or 1,
            drawdown,
            z3,
            close,
            fast,
            reference is not None,
            pack,
        )
    elif structure == RANGE:
        if state.bounds is not None and state.latch is None:
            state.latch = _range_break(geoms, index, int(state.bounds["t0"]), state.bounds, pack)
        accumulation, distribution, range_flags = _range_scores(
            state,
            index,
            region_start,
            _coverage(direction_objects, len(geoms))[0],
            bars,
            pivots,
            bias,
            pack,
            columns,
        )
        if state.latch == "up":
            candidate, trigger = PENDING, "range_break_up"
        elif state.latch == "down":
            candidate, trigger = PENDING, "range_break_down"
        else:
            candidate, trigger = _range_candidate(accumulation, distribution, pack)
    else:
        candidate, trigger = PENDING, "unconfirmed"
    summary = {
        "exhaustion_score": exhaustion_score,
        "exhaustion": exhaustion_flags,
        "accumulation_score": accumulation,
        "distribution_score": distribution,
        "range": range_flags,
        "volume_bias": bias,
        "age": age,
        "s_t": scale_value,
        "reference": reference,
        "z3": z3,
    }
    stats = _stats(scale_value, reference, z3, z5, fast, age, drawdown, exhaustion_score, bias, accumulation, distribution)
    return {"candidate": candidate, "trigger": trigger, "summary": summary, "stats": stats}


def _trend_candidate(structure, exhaustion, age, drawdown, z3, close, fast, has_ref, pack) -> Tuple[str, str]:
    if structure == UPTREND:
        if exhaustion:
            return UP_EXHAUSTION, "exhaustion"
        if age <= pack.birth_bars:
            return UP_BIRTH, "birth"
        if _pullback(drawdown, z3, close, fast, has_ref, pack, up=True):
            return UP_PULLBACK, "pullback"
        return UP_CONTINUE, "continue"
    if exhaustion:
        return DOWN_EXHAUSTION, "exhaustion"
    if age <= pack.birth_bars:
        return DOWN_BIRTH, "birth"
    if _pullback(drawdown, z3, close, fast, has_ref, pack, up=False):
        return DOWN_REBOUND, "rebound"
    return DOWN_CONTINUE, "continue"


def _pullback(drawdown, z3, close, fast, has_ref, pack, up: bool) -> bool:
    if not has_ref or drawdown is None or drawdown < pack.pullback_multiple:
        return False
    if up:
        impulse = z3 is not None and z3 <= -pack.pullback_impulse
        against_average = fast is not None and close < fast
    else:
        impulse = z3 is not None and z3 >= pack.pullback_impulse
        against_average = fast is not None and close > fast
    return impulse or against_average


def _range_candidate(accumulation: int, distribution: int, pack) -> Tuple[str, str]:
    if accumulation >= pack.range_score and accumulation - distribution >= pack.score_gap:
        return ACCUMULATION, "accumulation"
    if distribution >= pack.range_score and distribution - accumulation >= pack.score_gap:
        return DISTRIBUTION, "distribution"
    return NEUTRAL, "neutral"


def _update_exhaustion(state: _Run, score: int, threshold: int, need: int) -> None:
    if score >= threshold:
        state.high_run += 1
        state.low_run = 0
    else:
        state.low_run += 1
        state.high_run = 0
    if state.high_run >= need:
        state.exhaustion = True
    elif state.low_run >= need:
        state.exhaustion = False


def _switch(state: _Run, candidate: str, structure_changed: bool, has_prev: bool, need: int) -> str:
    if (not has_prev) or structure_changed or state.confirmed is None:
        state.streak = 0
        state.streak_key = None
        return candidate
    if candidate in _IMMEDIATE:
        state.streak = 0
        state.streak_key = None
        return candidate
    if state.confirmed in _BIRTHS and candidate != state.confirmed:
        state.streak = 0
        state.streak_key = None
        return candidate
    if state.confirmed in _EXHAUSTIONS and candidate != state.confirmed:
        state.streak = 0
        state.streak_key = None
        return candidate
    if state.confirmed in (INSUFFICIENT, PENDING) and candidate != state.confirmed:
        state.streak = 0
        state.streak_key = None
        return candidate
    if candidate == state.confirmed:
        state.streak = 0
        state.streak_key = None
        return state.confirmed
    if candidate not in _SUBSTATES:
        state.streak = 0
        state.streak_key = None
        return candidate
    if state.streak_key == candidate:
        state.streak += 1
    else:
        state.streak_key = candidate
        state.streak = 1
    if state.streak >= need:
        state.streak = 0
        state.streak_key = None
        return candidate
    return str(state.confirmed)


def _exhaustion_flags(
    structure,
    index,
    region_start,
    close,
    scale_value,
    reference,
    previous,
    drawdown,
    z3,
    bars,
    volatility,
    floor,
    comparisons,
    pack,
    columns,
) -> Dict[str, bool]:
    up = structure == UPTREND
    fade = False
    if reference is not None and previous is not None:
        old = (reference - previous) / scale_value if up else (previous - reference) / scale_value
        new = (close - reference) / scale_value if up else (reference - close) / scale_value
        fade = old >= pack.fade_progress and new <= pack.fade_ratio * old
    side = "high" if up else "low"
    result = "higher" if up else "lower"
    swung = _swing_hit(
        comparisons,
        side,
        result,
        index - pack.swing_window + 1,
        index,
        region_start,
    )
    retrace = bool(swung and drawdown is not None and drawdown >= pack.swing_retrace)
    wick = _wick_hit(bars, volatility, floor, index, region_start, pack, columns, up=up)
    volume = _volume_hit(bars, index, region_start, z3, pack, columns)
    if up:
        return {
            "progress_fade": fade,
            "swing_retrace": retrace,
            "wick_reject": wick,
            "volume_stall": volume,
        }
    return {
        "progress_fade": fade,
        "swing_retrace": retrace,
        "wick_reject": wick,
        "volume_stall": volume,
    }


def _range_scores(state, index, region_start, structures, bars, pivots, bias, pack, columns) -> Tuple[int, int, dict]:
    prior = _prior_structure(structures, int(state.t0 or index), region_start)
    bounds = state.bounds
    higher_low = False
    lower_high = False
    fake_low = False
    fake_high = False
    if bounds is not None:
        higher_low = _adjacent_shift(pivots, "low", int(bounds["t0"]), index, region_start, bounds["s0"], pack.structure_shift, higher=True)
        lower_high = _adjacent_shift(pivots, "high", int(bounds["t0"]), index, region_start, bounds["s0"], pack.structure_shift, higher=False)
        fake_low = _fake_break(bars, index, int(bounds["t0"]), bounds, pack, low=True)
        fake_high = _fake_break(bars, index, int(bounds["t0"]), bounds, pack, low=False)
    acc_bias = bool(columns["volume"] and columns["high"] and columns["low"] and bias is not None and bias >= pack.bias_minimum)
    dist_bias = bool(columns["volume"] and columns["high"] and columns["low"] and bias is not None and bias <= -pack.bias_minimum)
    acc = {
        "prior_trend": prior == DOWNTREND,
        "range_structure": higher_low,
        "volume_bias": acc_bias,
        "fake_break": fake_low,
    }
    dist = {
        "prior_trend": prior == UPTREND,
        "range_structure": lower_high,
        "volume_bias": dist_bias,
        "fake_break": fake_high,
    }
    # 摘要里的四项对照用吸筹这一侧；派发分单独记。两边都写进 readings 时由分数表达。
    flags = {
        "prior_trend": acc["prior_trend"] or dist["prior_trend"],
        "range_structure": acc["range_structure"] or dist["range_structure"],
        "volume_bias": acc["volume_bias"] or dist["volume_bias"],
        "fake_break": acc["fake_break"] or dist["fake_break"],
    }
    return (
        sum(1 for item in acc.values() if item),
        sum(1 for item in dist.values() if item),
        {"accumulation": acc, "distribution": dist, "flags": flags},
    )


def _insufficient_detail() -> dict:
    return {
        "candidate": INSUFFICIENT,
        "trigger": "insufficient",
        "summary": {
            "exhaustion_score": None,
            "exhaustion": {},
            "accumulation_score": None,
            "distribution_score": None,
            "range": {},
            "volume_bias": None,
            "age": None,
            "s_t": None,
            "reference": None,
            "z3": None,
        },
        "stats": _stats(None, None, None, None, None, None, None, None, None, None, None),
    }


def _stats(scale_value, reference, z3, z5, fast, age, drawdown, exhaustion_score, bias, accumulation, distribution) -> dict:
    return {
        "s_t": scale_value,
        "reference": reference,
        "z3": z3,
        "z5": z5,
        "fast_average": fast,
        "age": age,
        "retracement": drawdown,
        "exhaustion_score": exhaustion_score,
        "volume_bias": bias,
        "accumulation_score": accumulation,
        "distribution_score": distribution,
    }


def _open(scale, index, bars, category, structure, run_start, owners, direction_objects, detail, before, bounds) -> dict:
    bar = bars[index]
    origin = _origin_at(direction_objects, run_start)
    protection = _protection_at(direction_objects, index, run_start)
    direction_id = owners[run_start] if 0 <= run_start < len(owners) else owners[index]
    frozen = copy.deepcopy(bounds) if bounds is not None and structure == RANGE else None
    return {
        "id": f"phase:{scale}:{index}",
        "method": "phase",
        "scale": scale,
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
        "stats": detail["stats"],
        "structure": structure,
        "trigger": detail["trigger"],
        "state_before": before,
        "confirm_index": run_start,
        "direction_id": direction_id,
        "structure_origin": origin,
        "protection_pivot": protection,
        "summary": copy.deepcopy(detail["summary"]),
        "range_bounds": frozen,
    }


def _extend(segment: dict, index: int, bars, detail: dict) -> None:
    segment["end_index"] = index
    segment["end_time"] = bars[index].timestamp.isoformat()
    segment["stats"] = detail["stats"]


def _close(segment: dict, end_index: int, bars) -> None:
    if end_index < int(segment["start_index"]):
        return
    segment["end_index"] = end_index
    segment["end_time"] = bars[end_index].timestamp.isoformat()
    segment["status"] = "closed"


def _event(segment: dict, bars) -> dict:
    index = int(segment["start_index"])
    name = _NAMES.get(segment.get("category"), segment.get("category"))
    trigger = segment.get("trigger") or ""
    return {
        "event_type": "phase_change",
        "event_index": index,
        "event_time": bars[index].timestamp.isoformat(),
        "available_index": index,
        "available_time": bars[index].timestamp.isoformat(),
        "reason": f"{name}，触发 {trigger}",
        "reference_close": bars[index].close,
        "object_id": segment["id"],
        "object_kind": "phase",
        "revision": 0,
        "price_cross_index": None,
        "state_before": segment.get("state_before") or "",
        "state_after": segment.get("category"),
        "trigger": trigger,
    }


def _reading(segment: dict) -> dict:
    summary = segment.get("summary") or {}
    return {
        "available_index": int(segment["start_index"]),
        "available_time": segment.get("start_time"),
        "scale": segment.get("scale"),
        "object_id": segment["id"],
        "category": segment.get("category"),
        "state_before": segment.get("state_before") or "",
        "trigger": segment.get("trigger"),
        "confirm_index": segment.get("confirm_index"),
        "age": summary.get("age"),
        "s_t": summary.get("s_t"),
        "reference": summary.get("reference"),
        "z3": summary.get("z3"),
        "exhaustion_score": summary.get("exhaustion_score"),
        "exhaustion": copy.deepcopy(summary.get("exhaustion") or {}),
        "accumulation_score": summary.get("accumulation_score"),
        "distribution_score": summary.get("distribution_score"),
        "range": copy.deepcopy(summary.get("range") or {}),
        "volume_bias": summary.get("volume_bias"),
    }


def _coverage(direction_objects: Sequence[dict], count: int) -> Tuple[List[str], List[Optional[str]]]:
    structure = [UNDECIDED] * count
    owner: List[Optional[str]] = [None] * count
    ordered = sorted(direction_objects, key=lambda item: (int(item.get("start_index") or 0), str(item.get("id") or "")))
    allowed = {UPTREND, DOWNTREND, RANGE, UNDECIDED}
    for item in ordered:
        start = int(item.get("start_index") or 0)
        end = int(item.get("end_index") if item.get("end_index") is not None else start)
        value = item.get("structure") or UNDECIDED
        if value not in allowed:
            value = UNDECIDED
        for index in range(max(0, start), min(count - 1, end) + 1):
            structure[index] = value
            owner[index] = item.get("id")
    return structure, owner


def _usable_scale(volatility: Sequence[Optional[float]], index: int, floor: float) -> Optional[float]:
    if index < 0 or index >= len(volatility) or volatility[index] is None:
        if not math.isfinite(floor) or floor <= 0:
            return None
        return float(floor)
    value = float(volatility[index])
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _impulse(geoms, index: int, span: int, region_start: int, scale_value: float) -> Optional[float]:
    earlier = index - span
    if earlier < region_start or scale_value == 0:
        return None
    return (geoms[index] - geoms[earlier]) / scale_value


def _progress_refs(direction_objects, structure: str, run_start: int, index: int, region_start: int) -> List[dict]:
    if structure not in (UPTREND, DOWNTREND):
        return []
    rows = []
    for item in direction_objects:
        if item.get("structure") != structure:
            continue
        for row in item.get("progress") or []:
            available = int(row.get("available_index") or 0)
            place = int(row.get("reference_index") or 0)
            if available < run_start or available > index or available < region_start or place < region_start:
                continue
            rows.append(row)
    rows.sort(key=lambda row: (int(row["available_index"]), int(row["reference_index"])))
    return rows


def _swing_hit(comparisons, side: str, result: str, start: int, end: int, region_start: int) -> bool:
    for row in comparisons:
        if row.get("side") != side or row.get("result") != result:
            continue
        if row.get("mode") not in ("pivot", "cross"):
            continue
        available = int(row.get("available_index") or 0)
        if available < start or available > end or available < region_start:
            continue
        reference = (row.get("reference") or {}).get("extreme_index")
        if reference is not None and int(reference) < region_start:
            continue
        subject = (row.get("subject_pivot") or {}).get("extreme_index")
        if subject is not None and int(subject) < region_start:
            continue
        if row.get("subject_index") is not None and int(row["subject_index"]) < region_start:
            continue
        return True
    return False


def _wick_hit(bars, volatility, floor, index, region_start, pack, columns, up: bool) -> bool:
    if not columns["high"] or not columns["low"] or not columns["open"]:
        return False
    start = max(region_start, index - pack.wick_window + 1)
    count = 0
    for cursor in range(start, index + 1):
        if _wick_bar(bars[cursor], _usable_scale(volatility, cursor, floor), pack, up):
            count += 1
    return count >= pack.wick_count


def _wick_bar(bar, scale_value, pack, up: bool) -> bool:
    high = bar.high_geometric
    low = bar.low_geometric
    opened = bar.open_geometric
    close = bar.geometric
    if None in (high, low, opened) or not all(math.isfinite(value) for value in (high, low, opened, close)):
        return False
    if opened < low or opened > high:
        return False
    amplitude = high - low
    if amplitude <= 0 or scale_value is None:
        return False
    wick = (high - max(opened, close)) if up else (min(opened, close) - low)
    return wick / amplitude >= pack.wick_ratio and amplitude >= pack.wick_amplitude * scale_value


def _volume_hit(bars, index, region_start, z3, pack, columns) -> bool:
    if not columns["volume"] or z3 is None or abs(z3) > pack.volume_impulse:
        return False
    ratio = _volume_ratio(bars, index, region_start, pack.volume_window, pack.volume_baseline)
    return ratio is not None and ratio >= pack.volume_multiple


def _volume_ratio(bars, index, region_start, recent: int, baseline: int) -> Optional[float]:
    recent_start = index - recent + 1
    base_start = index - recent - baseline + 1
    if base_start < region_start:
        return None
    recent_values = [_volume_value(bars[cursor]) for cursor in range(recent_start, index + 1)]
    base_values = [_volume_value(bars[cursor]) for cursor in range(base_start, recent_start)]
    if any(value is None for value in recent_values + base_values):
        return None
    base_sum = float(sum(base_values))
    if base_sum == 0:
        return None
    return (float(sum(recent_values)) / recent) / (base_sum / baseline)


def _volume_value(bar) -> Optional[float]:
    value = bar.volume
    if value is None or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        return None
    return float(value)


def _volume_bias(bars, index, region_start, window: int, columns) -> Optional[float]:
    if not columns["high"] or not columns["low"] or not columns["volume"]:
        return None
    start = index - window + 1
    if start < region_start:
        return None
    weighted = 0.0
    total = 0.0
    for cursor in range(start, index + 1):
        bar = bars[cursor]
        volume = _volume_value(bar)
        high = bar.high_geometric
        low = bar.low_geometric
        if volume is None or high is None or low is None or not math.isfinite(high) or not math.isfinite(low):
            return None
        if high == low:
            location = 0.0
        else:
            location = (2.0 * bar.geometric - high - low) / (high - low)
        weighted += volume * location
        total += volume
    if total == 0:
        return None
    return weighted / total


def _freeze(pivots, volatility, t0: int, region_start: int, floor: float) -> Optional[dict]:
    scale_value = _usable_scale(volatility, t0, floor)
    if scale_value is None:
        return None
    high = _latest_pivot(pivots, "high", region_start, t0)
    low = _latest_pivot(pivots, "low", region_start, t0)
    if high is None or low is None:
        return None
    return {
        "t0": int(t0),
        "upper_pivot": _pivot_id(high),
        "lower_pivot": _pivot_id(low),
        "upper": float(high.geometric_price),
        "lower": float(low.geometric_price),
        "s0": float(scale_value),
    }


def _latest_pivot(pivots, role: str, region_start: int, index: int):
    chosen = None
    for pivot in _pivots_until(pivots, role, region_start, index):
        chosen = pivot
    return chosen


def _pivots_until(pivots, role: str, region_start: int, index: int) -> List[Any]:
    found = []
    for pivot in pivots:
        if getattr(pivot, "role", None) != role:
            continue
        confirm = int(pivot.confirm_index)
        extreme = int(pivot.extreme_index)
        if confirm < region_start or extreme < region_start or confirm > index:
            continue
        found.append(pivot)
    found.sort(key=lambda pivot: (int(pivot.confirm_index), int(pivot.extreme_index)))
    return found


def _adjacent_shift(pivots, role, t0, index, region_start, s0, shift, higher: bool) -> bool:
    ordered = _pivots_until(pivots, role, region_start, index)
    latest = None
    for pivot in ordered:
        if int(pivot.confirm_index) >= t0:
            latest = pivot
    if latest is None:
        return False
    position = ordered.index(latest)
    if position == 0:
        return False
    previous = ordered[position - 1]
    gap = float(latest.geometric_price) - float(previous.geometric_price)
    if higher:
        return gap >= shift * s0
    return -gap >= shift * s0


def _fake_break(bars, index, t0, bounds, pack, low: bool) -> bool:
    start = max(t0, index - pack.false_break_window + 1)
    upper = float(bounds["upper"])
    lower = float(bounds["lower"])
    scale_value = float(bounds["s0"])
    for cursor in range(start, index + 1):
        bar = bars[cursor]
        if low:
            price = bar.low_geometric
            if price is not None and price < lower - pack.false_break * scale_value and bar.geometric > lower:
                return True
        else:
            price = bar.high_geometric
            if price is not None and price > upper + pack.false_break * scale_value and bar.geometric < upper:
                return True
    return False


def _range_break(geoms, index, t0, bounds, pack) -> Optional[str]:
    start = index - pack.break_bars + 1
    if start < t0:
        return None
    upper = float(bounds["upper"]) + pack.break_buffer * float(bounds["s0"])
    lower = float(bounds["lower"]) - pack.break_buffer * float(bounds["s0"])
    window = geoms[start : index + 1]
    if all(value > upper for value in window):
        return "up"
    if all(value < lower for value in window):
        return "down"
    return None


def _prior_structure(structures: Sequence[str], t0: int, region_start: int) -> Optional[str]:
    cursor = t0 - 1
    while cursor >= region_start:
        value = structures[cursor]
        if value in (UPTREND, DOWNTREND):
            return value
        cursor -= 1
    return None


def _origin_at(direction_objects, index: int) -> Optional[dict]:
    for item in direction_objects:
        start = int(item.get("start_index") or 0)
        end = int(item.get("end_index") if item.get("end_index") is not None else start)
        if start <= index <= end and item.get("structure_origin"):
            return copy.deepcopy(item["structure_origin"])
    return None


def _protection_at(direction_objects, index: int, region_start: int):
    rows = []
    for item in direction_objects:
        for row in item.get("protection") or []:
            available = int(row.get("available_index") or 0)
            if region_start <= available <= index and row.get("pivot"):
                rows.append(row)
    if not rows:
        return None
    rows.sort(key=lambda row: int(row["available_index"]))
    return copy.deepcopy(rows[-1]["pivot"])


def _pivot_id(pivot) -> dict:
    return {
        "scale": pivot.scale,
        "role": pivot.role,
        "extreme_index": int(pivot.extreme_index),
    }


def _pivot_list(pivots) -> list:
    confirmed = getattr(pivots, "confirmed", None)
    if confirmed is None:
        return list(pivots or [])
    return list(confirmed)


def _columns(bars) -> Dict[str, bool]:
    return {
        "high": any(bar.high is not None for bar in bars),
        "low": any(bar.low is not None for bar in bars),
        "open": any(bar.open is not None for bar in bars),
        "volume": any(bar.volume is not None for bar in bars),
    }


def _missing_notes(series) -> List[str]:
    bars = series.bars
    missing = []
    if all(bar.high is None for bar in bars):
        missing.append("最高价")
    if all(bar.low is None for bar in bars):
        missing.append("最低价")
    if all(bar.open is None for bar in bars):
        missing.append("开盘价")
    if all(bar.volume is None for bar in bars):
        missing.append("成交量")
    if not missing:
        return []
    return ["缺少" + "、".join(missing) + "，对应的影线、假突破、放量或量价偏向记 0 分。"]


def _hover(record: dict) -> List[str]:
    category = _NAMES.get(record.get("category"), record.get("category") or "")
    before = _NAMES.get(record.get("state_before") or "", record.get("state_before") or "无")
    scale = _SCALE.get(record.get("scale"), record.get("scale"))
    status = _STATUS.get(record.get("status"), record.get("status") or "")
    summary = record.get("summary") or {}
    trigger = record.get("trigger") or ""
    lines = [
        f"阶段 · {category}",
        f"{scale} · {status}",
        f"序号 {record.get('start_index')}–{record.get('end_index')}",
        f"{before} → {category}",
        "触发：" + _TRIGGER_NAME.get(trigger, trigger),
        f"确认序号 {record.get('confirm_index')}",
    ]
    if summary.get("age") is not None:
        lines.append(f"年龄 {summary.get('age')}")
    if summary.get("s_t") is not None:
        lines.append(f"s_t {float(summary['s_t']):.6g}")
    if summary.get("reference") is not None:
        lines.append(f"X_t {float(summary['reference']):.6g}")
    if summary.get("z3") is not None:
        lines.append(f"Z3 {float(summary['z3']):.4f}")
    exhaustion = summary.get("exhaustion") or {}
    if summary.get("exhaustion_score") is not None:
        parts = []
        for key, value in exhaustion.items():
            label = _EXHAUST_NAME.get(key, key)
            if key == "wick_reject" and record.get("structure") == DOWNTREND:
                label = "多次下探收回"
            parts.append(label + ("是" if value else "否"))
        lines.append(f"衰竭 {summary.get('exhaustion_score')}：" + "，".join(parts))
    if summary.get("accumulation_score") is not None or summary.get("distribution_score") is not None:
        lines.append(
            f"吸筹 {summary.get('accumulation_score')} / 派发 {summary.get('distribution_score')}"
        )
        if summary.get("volume_bias") is not None:
            lines.append(f"B {float(summary['volume_bias']):.4f}")
    return lines


def _price(value, price_axis: str) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    if price_axis == "log":
        return math.exp(number)
    return number


def _pivot_text(pivot: dict) -> str:
    role = "高点" if pivot.get("role") == "high" else "低点"
    return f"{role}@{pivot.get('extreme_index')}"


def _sample_levels() -> List[float]:
    levels = [0.0, 0.001]
    levels.extend([0.001] * 28)
    for value in (0.08, 0.03, 0.16, 0.08, 0.24, 0.14, 0.32):
        levels.extend([value] * 6)
    return levels
