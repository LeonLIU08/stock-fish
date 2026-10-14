"""因果磁带：一次快照，再按每根 K 线的时间戳做 as-of。

方向和主状态只读当时已经可见的区段身份。截断区段上的 stats 不读。
a_t 和几何 EMA10 由回测自己递推，s_t 只从因果波动序列抄来作参考。
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

from analysis.structure.methods.phase import _NAMES as PHASE_NAMES
from analysis.structure.methods.phase import _STRUCTURE_NAME as STRUCTURE_NAMES
from analysis.structure.methods.phase import INSUFFICIENT
from analysis.structure.series import StandardSeries
from analysis.structure.view_model import display_snapshot
from analysis.structure.volatility import compute_volatility

_HK = ZoneInfo("Asia/Hong_Kong")


@dataclass(frozen=True)
class TapeRow:
    index: int
    time: str
    scale: str
    price_axis: str
    structure: Optional[str]
    structure_name: Optional[str]
    category: Optional[str]
    category_name: Optional[str]
    phase_start_index: Optional[int]
    phase_start_time: Optional[str]
    direction_start_index: Optional[int]
    direction_status: Optional[str]
    gap_before: bool
    a_t: Optional[float]
    ema: Optional[float]
    s_t: Optional[float]
    exit_scale: Optional[str] = None
    exit_structure: Optional[str] = None
    exit_direction_start_index: Optional[int] = None


def canonical_row(row: TapeRow) -> str:
    return json.dumps(asdict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_tape(rows: Sequence[TapeRow]) -> str:
    return "\n".join(canonical_row(row) for row in rows)


def true_range_units(series: StandardSeries, span: int, floor: float) -> list[Optional[float]]:
    """Wilder 真实波幅。对数轴用对数，均匀轴用价格差。

    第一根没有前收，真实波幅只用最高价和最低价，和结构间断的那一根一样。
    满 span 根之后，在第 span 根上用这些真实波幅的平均值初始化。
    """
    if span < 1:
        raise ValueError("unit.span 必须 >= 1")
    if floor < 0:
        raise ValueError("unit.floor 不能为负")
    log_axis = series.price_axis == "log"
    ranges: list[float] = []
    value: Optional[float] = None
    out: list[Optional[float]] = []
    for index, bar in enumerate(series.bars):
        if bar.high is None or bar.low is None:
            raise ValueError("计算 a_t 需要最高价和最低价")
        if index == 0 or bar.gap_before:
            width = _width(bar.high, bar.low, log_axis)
        else:
            width = _true_range(bar.high, bar.low, series.bars[index - 1].close, log_axis)
        ranges.append(width)
        if len(ranges) < span:
            out.append(None)
            continue
        if len(ranges) == span:
            value = sum(ranges) / float(span)
        else:
            value = float(value) + (width - float(value)) / float(span)
        value = _floor_unit(value, floor, bar.close, log_axis)
        out.append(value)
    return out


def geometric_emas(series: StandardSeries, span: int) -> list[Optional[float]]:
    """与阶段方法的快均线同一条递推。结构间断会把均线清掉重来。"""
    if span < 1:
        raise ValueError("快均线跨度必须 >= 1")
    ema: Optional[float] = None
    ready = False
    total = 0.0
    count = 0
    out: list[Optional[float]] = []
    for bar in series.bars:
        if bar.index == 0 or bar.gap_before:
            ema = None
            ready = False
            total = 0.0
            count = 0
        price = bar.geometric
        if not math.isfinite(price):
            out.append(ema if ready else None)
            continue
        if not ready:
            total += price
            count += 1
            if count >= span:
                ema = total / float(count)
                ready = True
            out.append(ema if ready else None)
            continue
        alpha = 2.0 / (span + 1.0)
        ema = alpha * price + (1.0 - alpha) * float(ema)
        out.append(ema)
    return out


def window_bounds(series: StandardSeries, start: date, end: date) -> tuple[int, int]:
    """评估窗口在序列里的首尾序号，两端都包含。"""
    chosen = [
        bar.index
        for bar in series.bars
        if start <= _session_day(bar.timestamp) <= end
    ]
    if not chosen:
        raise ValueError(f"评估窗口 {start} ~ {end} 内没有 K 线")
    return chosen[0], chosen[-1]


def build_tape(
    series: StandardSeries,
    snapshot: dict,
    *,
    scale: str,
    span: int,
    floor: float,
    start_index: int = 0,
    end_index: Optional[int] = None,
    extra_scale: Optional[str] = None,
) -> tuple[TapeRow, ...]:
    """评估窗口内每一根一行。暖机只参与递推，不写出。

    `extra_scale` 在同一次 as-of 里再读一个尺度的方向，给策略 C 的卖出用。
    它不改变本尺度的结构方向和主状态。
    """
    if series.params.phase is None:
        raise ValueError("结构参数没有 phase，不能计算快均线")
    if scale not in ("short", "mid", "long"):
        raise ValueError(f"尺度必须是 short、mid、long，收到 {scale}")
    if extra_scale is not None:
        if extra_scale not in ("short", "mid", "long"):
            raise ValueError(f"卖出尺度必须是 short、mid、long，收到 {extra_scale}")
        if extra_scale == scale:
            raise ValueError("卖出尺度不能和买入尺度相同")
    closes = snapshot.get("closes") or []
    if len(closes) != len(series):
        raise ValueError(f"快照有 {len(closes)} 根，序列有 {len(series)} 根")
    last = len(series) - 1 if end_index is None else int(end_index)
    if not 0 <= start_index <= last < len(series):
        raise ValueError(f"磁带区间 {start_index} ~ {last} 超出序列长度 {len(series)}")

    units = true_range_units(series, span, floor)
    emas = geometric_emas(series, int(series.params.phase.fast_span))
    volatility = compute_volatility(series)
    rows = []
    for index in range(start_index, last + 1):
        bar = series.bars[index]
        when = str(closes[index]["time"])
        view = _as_of_view(snapshot, when)
        phase_objects = _method_objects(view, "phase")
        direction_objects = _method_objects(view, "direction")
        phase = _covering(phase_objects, scale, index)
        direction = _covering(direction_objects, scale, index)
        structure = None if phase is None else phase.get("structure")
        category = INSUFFICIENT if phase is None else phase.get("category")
        exit_structure = None
        exit_start = None
        if extra_scale is not None:
            exit_phase = _covering(phase_objects, extra_scale, index)
            exit_direction = _covering(direction_objects, extra_scale, index)
            if exit_phase is not None and exit_phase.get("structure") is not None:
                exit_structure = str(exit_phase.get("structure"))
            if exit_direction is not None and exit_direction.get("start_index") is not None:
                exit_start = int(exit_direction["start_index"])
        rows.append(
            TapeRow(
                index=index,
                time=when,
                scale=scale,
                price_axis=series.price_axis,
                structure=None if structure is None else str(structure),
                structure_name=STRUCTURE_NAMES.get(structure) if structure else None,
                category=None if category is None else str(category),
                category_name=PHASE_NAMES.get(category, str(category) if category else None),
                phase_start_index=None if phase is None else int(phase["start_index"]),
                phase_start_time=None if phase is None else phase.get("start_time"),
                direction_start_index=None if direction is None else int(direction["start_index"]),
                direction_status=None if direction is None else direction.get("status"),
                gap_before=bool(bar.gap_before),
                a_t=units[index],
                ema=emas[index],
                s_t=volatility[index],
                exit_scale=extra_scale,
                exit_structure=exit_structure,
                exit_direction_start_index=exit_start,
            )
        )
    return tuple(rows)


def _as_of_view(snapshot: dict, when: str) -> dict:
    copied = dict(snapshot)
    identity = dict(snapshot.get("identity") or {})
    identity["as_of"] = when
    copied["identity"] = identity
    return display_snapshot(copied)


def _method_objects(snapshot: dict, key: str) -> list:
    record = (snapshot.get("methods") or {}).get(key) or {}
    objects = record.get("objects") or []
    return list(objects)


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


def _width(high: float, low: float, log_axis: bool) -> float:
    if log_axis:
        return math.log(high / low)
    return high - low


def _true_range(high: float, low: float, previous_close: float, log_axis: bool) -> float:
    if log_axis:
        return math.log(max(high, previous_close) / min(low, previous_close))
    return max(high, previous_close) - min(low, previous_close)


def _floor_unit(value: float, floor: float, close: float, log_axis: bool) -> float:
    lower = floor if log_axis else floor * close
    return max(float(value), lower)


def _session_day(stamp: datetime) -> date:
    if stamp.tzinfo is None:
        return stamp.date()
    return stamp.astimezone(_HK).date()
