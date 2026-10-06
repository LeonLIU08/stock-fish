"""趋势区间。

同一尺度上，一条下边界和一条上边界配成一个区间。两条线的序号要有重叠，
并且重叠段不跨间断。有效区间是其中最长的一段：宽度为正、上下都有接触簇、
收盘价没有在任一侧超过硬破坏深度。交叉之后的序号留在有效区间外面，只能
当作投影。硬破坏的线不参与配对；上升通道的上轨即使单独当压力线时斜率
符号不通过，仍然可以作上边界。

平行和横向用同一个波动尺度比较归一化斜率，取两条边界里较大的那个。
这样不会因为两条线各自冻结的波动不同，把几何上平行的轨道判成不平行。

几何标签只描述形状。横向比通道更具体，先判断横向。上轨接近水平、下轨
上升时落在收敛区间或其他边界对，不另做三角形。

去重规则和趋势线一样：硬约束通过且软评分最高的一条是主结果。不同尺度
的已验证区间都保留，并记录序号重叠，分数不相加。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

from analysis.structure.boundaries import (
    CONSTRAINT_MIN_TOUCH_CLUSTERS,
    CONSTRAINT_SLOPE_SIGN,
    CONTEXT_FIT_UNUSED,
    ROLE_RESISTANCE,
    ROLE_SUPPORT,
    LIVE_STATUSES,
    STATUS_BROKEN,
    STATUS_CANDIDATE,
    STATUS_EXPIRED,
    STATUS_REJECTED,
    STATUS_VALIDATED,
    ConstraintCheck,
    SoftScore,
    Boundary,
    BoundaryResult,
    build_boundaries,
)
from analysis.structure.config import SCALE_NAMES, StructureParams
from analysis.structure.pivots import PivotResult
from analysis.structure.series import StandardSeries

LABEL_CHANNEL = "channel"
LABEL_CONVERGENCE = "convergence"
LABEL_SIDEWAYS = "sideways"
LABEL_OTHER = "other"
LABELS = (LABEL_CHANNEL, LABEL_CONVERGENCE, LABEL_SIDEWAYS, LABEL_OTHER)

CONSTRAINT_POSITIVE_WIDTH = "positive_width"
CONSTRAINT_LOWER_TOUCHES = "lower_touches"
CONSTRAINT_UPPER_TOUCHES = "upper_touches"
CONSTRAINT_LABEL = "label_geometry"

OVERLAP_NOTE = "不合并为一条证据"
_MAIN_LABELS = (LABEL_CHANNEL, LABEL_CONVERGENCE, LABEL_SIDEWAYS)


@dataclass(frozen=True)
class Zone:
    """一对上下边界包住的区间。标签不是交易方向。"""

    id: str
    revision: int
    scale: str
    label: str
    status: str
    primary: bool
    alternate_of: Optional[str]
    emits_events: bool
    draw_on_main: bool
    lower_id: str
    upper_id: str
    lower_slope: float
    lower_intercept: float
    upper_slope: float
    upper_intercept: float
    volatility: float
    effective_start: int
    effective_end: int
    projected_start: int
    projected_end: int
    start_time: datetime
    end_time: datetime
    confirm_index: int
    confirm_time: datetime
    available_time: datetime
    start_width: float
    end_width: float
    width_ratio: float
    slope_gap: float
    normalized_slope_gap: float
    lower_touch_clusters: int
    upper_touch_clusters: int
    max_break_depth: float
    constraints: Tuple[ConstraintCheck, ...]
    failed_constraints: Tuple[str, ...]
    simplicity_penalty: float
    context_fit: str
    score: Optional[SoftScore]

    def __post_init__(self) -> None:
        if self.scale not in SCALE_NAMES:
            raise ValueError(f"未知尺度 {self.scale!r}")
        if self.label not in LABELS:
            raise ValueError(f"未知区间标签 {self.label!r}")
        if self.status not in (
            STATUS_VALIDATED,
            STATUS_CANDIDATE,
            STATUS_REJECTED,
            STATUS_BROKEN,
            STATUS_EXPIRED,
        ):
            raise ValueError(f"未知区间状态 {self.status!r}")
        if self.effective_end < self.effective_start:
            raise ValueError("有效区间终点不能早于起点")
        if self.start_width <= 0 or self.end_width <= 0:
            raise ValueError("有效区间两端宽度必须为正")
        if self.emits_events and not (self.primary and self.status in LIVE_STATUSES):
            raise ValueError("只有已验证的主结果才能产生以后的突破事件")
        if self.draw_on_main and not (self.primary and self.status == STATUS_VALIDATED and self.label in _MAIN_LABELS):
            raise ValueError("主图只画已验证的主结果，并且不画其他边界对")
        if (self.status in LIVE_STATUSES) != (self.score is not None):
            raise ValueError("软评分只给曾经已验证的区间")
        if self.simplicity_penalty != 0.0:
            raise ValueError("第一版简化惩罚必须为 0")
        if self.context_fit != CONTEXT_FIT_UNUSED:
            raise ValueError("第一版上下文吻合必须标记为未使用")

    def lower_at(self, index: int) -> float:
        return self.lower_slope * index + self.lower_intercept

    def upper_at(self, index: int) -> float:
        return self.upper_slope * index + self.upper_intercept

    def width_at(self, index: int) -> float:
        return self.upper_at(index) - self.lower_at(index)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "revision": self.revision,
            "scale": self.scale,
            "label": self.label,
            "status": self.status,
            "primary": self.primary,
            "alternate_of": self.alternate_of,
            "emits_events": self.emits_events,
            "draw_on_main": self.draw_on_main,
            "lower_id": self.lower_id,
            "upper_id": self.upper_id,
            "lower_slope": self.lower_slope,
            "lower_intercept": self.lower_intercept,
            "upper_slope": self.upper_slope,
            "upper_intercept": self.upper_intercept,
            "volatility": self.volatility,
            "effective_start": self.effective_start,
            "effective_end": self.effective_end,
            "projected_start": self.projected_start,
            "projected_end": self.projected_end,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "confirm_index": self.confirm_index,
            "confirm_time": self.confirm_time.isoformat(),
            "available_time": self.available_time.isoformat(),
            "start_width": self.start_width,
            "end_width": self.end_width,
            "width_ratio": self.width_ratio,
            "slope_gap": self.slope_gap,
            "normalized_slope_gap": self.normalized_slope_gap,
            "lower_touch_clusters": self.lower_touch_clusters,
            "upper_touch_clusters": self.upper_touch_clusters,
            "max_break_depth": self.max_break_depth,
            "constraints": [item.to_dict() for item in self.constraints],
            "failed_constraints": list(self.failed_constraints),
            "simplicity_penalty": self.simplicity_penalty,
            "context_fit": self.context_fit,
            "score": None if self.score is None else self.score.to_dict(),
        }


@dataclass(frozen=True)
class ZoneOverlap:
    """两个不同尺度的区间序号有重叠。分数不相加，也不合并成一条证据。"""

    zone_ids: Tuple[str, str]
    scales: Tuple[str, str]
    note: str = OVERLAP_NOTE

    def to_dict(self) -> dict:
        return {
            "zone_ids": list(self.zone_ids),
            "scales": list(self.scales),
            "note": self.note,
        }


@dataclass(frozen=True)
class ZoneResult:
    zones: Tuple[Zone, ...]
    overlaps: Tuple[ZoneOverlap, ...]

    def for_scale(self, scale: str) -> Tuple[Zone, ...]:
        return tuple(zone for zone in self.zones if zone.scale == scale)

    def validated(self, scale: Optional[str] = None, label: Optional[str] = None) -> Tuple[Zone, ...]:
        return tuple(
            zone
            for zone in self.zones
            if zone.status == STATUS_VALIDATED
            and (scale is None or zone.scale == scale)
            and (label is None or zone.label == label)
        )


def build_zones(
    series: StandardSeries,
    pivots: Optional[PivotResult] = None,
    boundaries: Optional[BoundaryResult] = None,
    bar_count: Optional[int] = None,
) -> ZoneResult:
    """把同一尺度的下边界和上边界配成趋势区间。

    `bar_count` 只使用前这么多根，并重新生成边界。临时末端不参与。
    """
    if bar_count is not None:
        series = series.prefix(bar_count)
        pivots = None
        boundaries = None
    if boundaries is None:
        boundaries = build_boundaries(series, pivots)
    drafted: List[Zone] = []
    for scale in SCALE_NAMES:
        lines = [item for item in boundaries.for_scale(scale) if _pairable(item)]
        lowers = [item for item in lines if item.role == ROLE_SUPPORT]
        uppers = [item for item in lines if item.role == ROLE_RESISTANCE]
        for lower in lowers:
            for upper in uppers:
                zone = _try_pair(series, lower, upper)
                if zone is not None:
                    drafted.append(zone)
    zoned = _apply_chart_limit(_assign_primaries(drafted, series.params.sideways_normalized_slope), series.params)
    return ZoneResult(zones=zoned, overlaps=_overlaps(zoned))


def format_zones(result: ZoneResult) -> str:
    """给人核对标签、有效区间和失败约束的文本。不参与拟合。"""
    lines = []
    for zone in result.zones:
        failed = ",".join(zone.failed_constraints) if zone.failed_constraints else "-"
        kind = "PRIMARY" if zone.primary else "ALTERNATE"
        chart = "MAIN" if zone.draw_on_main else "ASIDE"
        lines.append(
            f"{kind:9} {chart:5} {zone.status:9} {zone.scale:5} {zone.label:12} "
            f"{zone.effective_start}->{zone.effective_end} "
            f"width={zone.start_width:.4f}->{zone.end_width:.4f} ratio={zone.width_ratio:.3f} "
            f"gap={zone.normalized_slope_gap:.3f} failed={failed} id={zone.id}"
        )
    return "\n".join(lines)


def _pairable(boundary: Boundary) -> bool:
    """硬破坏的线不能当区间边界。斜率符号不符的线仍可参与配对。

    上升通道的上轨是抬高的高点，单独作为压力线时斜率符号不通过，
    但几何上仍然是这条通道的上边界。
    """
    if boundary.status != STATUS_REJECTED:
        return True
    blocking = set(boundary.failed_constraints) - {
        CONSTRAINT_SLOPE_SIGN,
        CONSTRAINT_MIN_TOUCH_CLUSTERS,
    }
    return not blocking


def _try_pair(series: StandardSeries, lower: Boundary, upper: Boundary) -> Optional[Zone]:
    params = series.params
    projected_start = max(lower.start_index, upper.start_index)
    projected_end = min(lower.end_index, upper.end_index)
    if projected_end <= projected_start:
        return None
    if _crosses_gap(series, projected_start, projected_end):
        return None
    effective = _effective_interval(series, lower, upper, projected_start, projected_end, params.hard_break_ratio)
    if effective is None:
        return None
    start, end = effective
    volatility = max(lower.volatility, upper.volatility)
    start_width = upper.line_at(start) - lower.line_at(start)
    end_width = upper.line_at(end) - lower.line_at(end)
    if start_width <= 0 or end_width <= 0:
        return None
    width_ratio = end_width / start_width
    slope_gap = abs(upper.slope - lower.slope)
    normalized_gap = slope_gap / volatility
    lower_norm = abs(lower.slope) / volatility
    upper_norm = abs(upper.slope) / volatility
    label = _label(lower_norm, upper_norm, normalized_gap, width_ratio, params)
    lower_touches = _clusters_inside(lower, start, end)
    upper_touches = _clusters_inside(upper, start, end)
    max_break = _max_break_depth(series, lower, upper, start, end)
    checks = _constraints(
        params,
        label,
        lower_norm,
        upper_norm,
        normalized_gap,
        width_ratio,
        lower_touches,
        upper_touches,
        len(lower.touch_clusters),
        len(upper.touch_clusters),
    )
    failed = tuple(check.name for check in checks if not check.passed)
    blocking = tuple(
        name for name in failed if name in (CONSTRAINT_POSITIVE_WIDTH, CONSTRAINT_LABEL)
    )
    if not failed:
        status = STATUS_VALIDATED
    elif not blocking:
        status = STATUS_CANDIDATE
    else:
        status = STATUS_REJECTED
    score = None
    if status == STATUS_VALIDATED:
        score = _soft_score(
            params,
            lower.scale,
            normalized_gap,
            lower_touches,
            upper_touches,
            start_width,
            volatility,
            max_break,
            end - start,
        )
    confirm_index = max(lower.confirm_index, upper.confirm_index)
    confirm_time = lower.confirm_time if lower.confirm_index >= upper.confirm_index else upper.confirm_time
    available_time = lower.available_time if lower.confirm_index >= upper.confirm_index else upper.available_time
    return Zone(
        id=_zone_id(lower, upper),
        revision=1,
        scale=lower.scale,
        label=label,
        status=status,
        primary=True,
        alternate_of=None,
        emits_events=False,
        draw_on_main=False,
        lower_id=lower.id,
        upper_id=upper.id,
        lower_slope=lower.slope,
        lower_intercept=lower.intercept,
        upper_slope=upper.slope,
        upper_intercept=upper.intercept,
        volatility=volatility,
        effective_start=start,
        effective_end=end,
        projected_start=projected_start,
        projected_end=projected_end,
        start_time=series.bars[start].timestamp,
        end_time=series.bars[end].timestamp,
        confirm_index=confirm_index,
        confirm_time=confirm_time,
        available_time=available_time,
        start_width=start_width,
        end_width=end_width,
        width_ratio=width_ratio,
        slope_gap=slope_gap,
        normalized_slope_gap=normalized_gap,
        lower_touch_clusters=lower_touches,
        upper_touch_clusters=upper_touches,
        max_break_depth=max_break,
        constraints=checks,
        failed_constraints=failed,
        simplicity_penalty=0.0,
        context_fit=CONTEXT_FIT_UNUSED,
        score=score,
    )


def _effective_interval(
    series: StandardSeries,
    lower: Boundary,
    upper: Boundary,
    projected_start: int,
    projected_end: int,
    hard_break_ratio: float,
) -> Optional[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    run_start: Optional[int] = None
    for index in range(projected_start, projected_end + 1):
        kept = _bar_in_zone(series, lower, upper, index, hard_break_ratio)
        if kept and run_start is None:
            run_start = index
        elif not kept and run_start is not None:
            runs.append((run_start, index - 1))
            run_start = None
    if run_start is not None:
        runs.append((run_start, projected_end))
    best: Optional[Tuple[int, int]] = None
    for start, end in runs:
        if start > end:
            continue
        if _clusters_inside(lower, start, end) < 1 or _clusters_inside(upper, start, end) < 1:
            continue
        if best is None or (end - start) > (best[1] - best[0]):
            best = (start, end)
    return best


def _bar_in_zone(
    series: StandardSeries,
    lower: Boundary,
    upper: Boundary,
    index: int,
    hard_break_ratio: float,
) -> bool:
    if upper.line_at(index) - lower.line_at(index) <= 0.0:
        return False
    geometric = series.bars[index].geometric
    lower_depth = max(0.0, lower.line_at(index) - geometric)
    upper_depth = max(0.0, geometric - upper.line_at(index))
    if lower_depth > hard_break_ratio * lower.volatility:
        return False
    if upper_depth > hard_break_ratio * upper.volatility:
        return False
    return True


def _label(
    lower_norm: float,
    upper_norm: float,
    normalized_gap: float,
    width_ratio: float,
    params: StructureParams,
) -> str:
    parallel = normalized_gap < params.parallel_slope_gap
    stable = params.channel_width_ratio_min <= width_ratio <= params.channel_width_ratio_max
    flat = (
        lower_norm < params.sideways_normalized_slope
        and upper_norm < params.sideways_normalized_slope
    )
    if flat and stable and parallel:
        return LABEL_SIDEWAYS
    if stable and parallel:
        return LABEL_CHANNEL
    if width_ratio < params.convergence_width_ratio:
        return LABEL_CONVERGENCE
    return LABEL_OTHER


def _constraints(
    params: StructureParams,
    label: str,
    lower_norm: float,
    upper_norm: float,
    normalized_gap: float,
    width_ratio: float,
    lower_touches: int,
    upper_touches: int,
    lower_total: int,
    upper_total: int,
) -> Tuple[ConstraintCheck, ...]:
    expected = _label(lower_norm, upper_norm, normalized_gap, width_ratio, params)
    lower_ok, lower_detail = _side_touch(params, lower_touches, lower_total, "下侧")
    upper_ok, upper_detail = _side_touch(params, upper_touches, upper_total, "上侧")
    return (
        ConstraintCheck(CONSTRAINT_POSITIVE_WIDTH, width_ratio > 0.0, f"终点宽度 / 起点宽度 = {width_ratio:.3f}"),
        ConstraintCheck(CONSTRAINT_LOWER_TOUCHES, lower_ok, lower_detail),
        ConstraintCheck(CONSTRAINT_UPPER_TOUCHES, upper_ok, upper_detail),
        ConstraintCheck(
            CONSTRAINT_LABEL,
            label == expected,
            f"标签 {label}，宽度比 {width_ratio:.3f}，归一化斜率差 {normalized_gap:.3f}",
        ),
    )


def _side_touch(params: StructureParams, inside: int, total: int, side: str) -> Tuple[bool, str]:
    """有效区间内的簇够数，或 v1 承认这条边界全长已经达到趋势线的接触要求。"""
    minimum = params.zone_min_touch_clusters
    if inside >= minimum:
        return True, f"{side}接触簇 {inside} 个，至少 {minimum} 个"
    if params.zone_credit_full_span and inside >= 1 and total >= params.min_touch_clusters:
        return True, (
            f"{side}有效区间内 {inside} 个；边界全长 {total} 个，"
            f"已达到趋势线要求的 {params.min_touch_clusters} 个"
        )
    return False, f"{side}接触簇 {inside} 个，至少 {minimum} 个"


def _soft_score(
    params: StructureParams,
    scale: str,
    normalized_gap: float,
    lower_touches: int,
    upper_touches: int,
    start_width: float,
    volatility: float,
    max_break: float,
    span: int,
) -> SoftScore:
    geometry = _clamp(1.0 - normalized_gap / params.parallel_slope_gap)
    touch_quality = _clamp(min(lower_touches, upper_touches) / params.zone_min_touch_clusters)
    significance = _clamp(start_width / volatility)
    path_integrity = _clamp(1.0 - max_break / params.hard_break_ratio)
    span_score = _clamp(span / params.scale(scale).max_span)
    parts = (geometry, touch_quality, significance, path_integrity, span_score)
    return SoftScore(
        geometry=geometry,
        touch_quality=touch_quality,
        significance=significance,
        path_integrity=path_integrity,
        span=span_score,
        simplicity_penalty=0.0,
        context_fit=CONTEXT_FIT_UNUSED,
        total=sum(parts) / len(parts),
    )


def _assign_primaries(drafted: Sequence[Zone], sideways: float) -> Tuple[Zone, ...]:
    order = sorted(range(len(drafted)), key=lambda index: _quality_key(drafted[index]))
    assigned: List[Optional[Zone]] = [None] * len(drafted)
    primaries: List[Zone] = []
    for index in order:
        current = drafted[index]
        primary = next((item for item in primaries if _similar(item, current, sideways)), None)
        if primary is None:
            current = replace(
                current,
                primary=True,
                alternate_of=None,
                emits_events=current.status == STATUS_VALIDATED,
            )
            primaries.append(current)
        else:
            current = replace(current, primary=False, alternate_of=primary.id, emits_events=False)
        assigned[index] = current
    return tuple(item for item in assigned if item is not None)


def _apply_chart_limit(zones: Sequence[Zone], params: StructureParams) -> Tuple[Zone, ...]:
    ranked = []
    for zone in zones:
        if zone.primary and zone.status == STATUS_VALIDATED and zone.label in _MAIN_LABELS:
            ranked.append(zone)
    chosen = set()
    by_scale = {scale: [] for scale in SCALE_NAMES}
    for zone in ranked:
        by_scale[zone.scale].append(zone)
    for group in by_scale.values():
        group.sort(key=lambda zone: (-(zone.score.total if zone.score else 0.0), -(zone.effective_end - zone.effective_start), zone.effective_start, zone.id))
        for zone in group[: params.max_zones_per_scale]:
            chosen.add(zone.id)
    return tuple(replace(zone, draw_on_main=zone.id in chosen) for zone in zones)


def _overlaps(zones: Sequence[Zone]) -> Tuple[ZoneOverlap, ...]:
    found: List[ZoneOverlap] = []
    for index, left in enumerate(zones):
        for right in zones[index + 1 :]:
            if left.scale == right.scale:
                continue
            if left.status != STATUS_VALIDATED or right.status != STATUS_VALIDATED:
                continue
            if _ranges_overlap(left.effective_start, left.effective_end, right.effective_start, right.effective_end):
                ordered = tuple(sorted((left, right), key=lambda zone: (_scale_rank(zone.scale), zone.id)))
                found.append(
                    ZoneOverlap(
                        zone_ids=(ordered[0].id, ordered[1].id),
                        scales=(ordered[0].scale, ordered[1].scale),
                    )
                )
    found.sort(key=lambda item: (item.scales, item.zone_ids))
    return tuple(found)


def _similar(primary: Zone, other: Zone, sideways: float) -> bool:
    if primary.scale != other.scale or primary.label != other.label or primary.id == other.id:
        return False
    if abs(primary.lower_slope - other.lower_slope) / primary.volatility > sideways:
        return False
    if abs(primary.upper_slope - other.upper_slope) / primary.volatility > sideways:
        return False
    left = max(primary.effective_start, other.effective_start)
    right = min(primary.effective_end, other.effective_end)
    if right < left:
        return False
    shorter = min(
        primary.effective_end - primary.effective_start,
        other.effective_end - other.effective_start,
    )
    if shorter > 0 and (right - left) / shorter <= 0.5:
        return False
    midpoint = (left + right) // 2
    tolerance = 0.5 * (primary.volatility + other.volatility) / 2.0
    return abs(primary.width_at(midpoint) - other.width_at(midpoint)) <= tolerance


def _quality_key(zone: Zone) -> Tuple[int, float, int, int, str]:
    status_rank = {STATUS_VALIDATED: 0, STATUS_CANDIDATE: 1, STATUS_REJECTED: 2}[zone.status]
    score = 0.0 if zone.score is None else -zone.score.total
    return (status_rank, score, -(zone.effective_end - zone.effective_start), zone.confirm_index, zone.id)


def _clusters_inside(boundary: Boundary, start: int, end: int) -> int:
    count = 0
    for cluster in boundary.touch_clusters:
        if any(start <= index <= end for index in cluster):
            count += 1
    return count


def _max_break_depth(
    series: StandardSeries,
    lower: Boundary,
    upper: Boundary,
    start: int,
    end: int,
) -> float:
    worst = 0.0
    for index in range(start, end + 1):
        geometric = series.bars[index].geometric
        lower_depth = max(0.0, lower.line_at(index) - geometric) / lower.volatility
        upper_depth = max(0.0, geometric - upper.line_at(index)) / upper.volatility
        worst = max(worst, lower_depth, upper_depth)
    return worst


def _crosses_gap(series: StandardSeries, start_index: int, end_index: int) -> bool:
    for index in range(start_index + 1, end_index + 1):
        if series.bars[index].gap_before:
            return True
    return False


def _ranges_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start <= right_end and right_start <= left_end


def _scale_rank(scale: str) -> int:
    return SCALE_NAMES.index(scale)


def _zone_id(lower: Boundary, upper: Boundary) -> str:
    return f"{lower.scale}:zone:{lower.start_index}-{lower.end_index}:{upper.start_index}-{upper.end_index}"


def _clamp(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
