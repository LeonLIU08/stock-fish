"""支撑线与压力线。

一条候选来自同一尺度、同一角色、同一无间断区段里的一段连续已确认拐点。
直线先经过这段的两端。中间拐点全部参加接触和破坏检验，不能跳过。
两端定线之后，只在没有破坏硬约束的接触拐点上做一次小幅最小二乘调整；
调整后仍用原始几何价格重算误差。

波动尺度用起点拐点上已经冻结的值，整条线不再改。接触距离和硬破坏
深度都是这个尺度的倍数。收盘价落到线的错误一侧先记成统计；连续停留
是否构成突破留给生命周期，不在这里把线从已验证改回。

去重时保留硬约束通过且软评分最高的一条做主结果，其余相近线做备选。
后到的线不会改写已经算好的斜率和状态。主结果标签按当前可见的线重选，
所以单独跑到更早的前缀时，只在当时已经确认的线里面选主结果。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

from analysis.structure.config import SCALE_NAMES, StructureParams
from analysis.structure.pivots import ROLE_HIGH, ROLE_LOW, Pivot, PivotResult, detect_pivots
from analysis.structure.series import StandardSeries

ROLE_SUPPORT = "support"
ROLE_RESISTANCE = "resistance"
BOUNDARY_ROLES = (ROLE_SUPPORT, ROLE_RESISTANCE)

STATUS_VALIDATED = "validated"
STATUS_CANDIDATE = "candidate"
STATUS_REJECTED = "rejected"
STATUS_BROKEN = "broken"
STATUS_EXPIRED = "expired"
STATUSES = (
    STATUS_VALIDATED,
    STATUS_CANDIDATE,
    STATUS_REJECTED,
    STATUS_BROKEN,
    STATUS_EXPIRED,
)
LIVE_STATUSES = (STATUS_VALIDATED, STATUS_BROKEN, STATUS_EXPIRED)

CONSTRAINT_TWO_PIVOTS = "two_pivots"
CONSTRAINT_SLOPE_SIGN = "slope_sign"
CONSTRAINT_HARD_BREAK = "hard_break"
CONSTRAINT_NO_SKIPPED_BREAK = "no_skipped_break"
CONSTRAINT_MIN_TOUCH_CLUSTERS = "min_touch_clusters"
HARD_CONSTRAINTS = (
    CONSTRAINT_TWO_PIVOTS,
    CONSTRAINT_SLOPE_SIGN,
    CONSTRAINT_HARD_BREAK,
    CONSTRAINT_NO_SKIPPED_BREAK,
    CONSTRAINT_MIN_TOUCH_CLUSTERS,
)

CONTEXT_FIT_UNUSED = "未使用"
_PIVOT_ROLE = {ROLE_SUPPORT: ROLE_LOW, ROLE_RESISTANCE: ROLE_HIGH}


@dataclass(frozen=True)
class ConstraintCheck:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class SoftScore:
    """只对已验证边界计算。简化惩罚第一版固定为 0，上下文吻合未使用。"""

    geometry: float
    touch_quality: float
    significance: float
    path_integrity: float
    span: float
    simplicity_penalty: float
    context_fit: str
    total: float

    def to_dict(self) -> dict:
        return {
            "geometry": self.geometry,
            "touch_quality": self.touch_quality,
            "significance": self.significance,
            "path_integrity": self.path_integrity,
            "span": self.span,
            "simplicity_penalty": self.simplicity_penalty,
            "context_fit": self.context_fit,
            "total": self.total,
        }


@dataclass(frozen=True)
class Boundary:
    """一条支撑或压力。备选解释不单独产生以后的突破事件。"""

    id: str
    revision: int
    scale: str
    role: str
    status: str
    primary: bool
    alternate_of: Optional[str]
    emits_events: bool
    adjusted: bool
    start_index: int
    end_index: int
    start_time: datetime
    end_time: datetime
    confirm_index: int
    confirm_time: datetime
    available_time: datetime
    slope: float
    intercept: float
    normalized_slope: float
    volatility: float
    span: int
    pivot_indices: Tuple[int, ...]
    touch_indices: Tuple[int, ...]
    touch_clusters: Tuple[Tuple[int, ...], ...]
    touch_error_median: Optional[float]
    touch_error_p90: Optional[float]
    max_break_depth: float
    max_cross_depth: float
    cross_count: int
    longest_violation_bars: int
    constraints: Tuple[ConstraintCheck, ...]
    failed_constraints: Tuple[str, ...]
    simplicity_penalty: float
    context_fit: str
    score: Optional[SoftScore]
    pivots: Tuple[Pivot, ...]

    def __post_init__(self) -> None:
        if self.scale not in SCALE_NAMES:
            raise ValueError(f"未知尺度 {self.scale!r}")
        if self.role not in BOUNDARY_ROLES:
            raise ValueError(f"未知边界角色 {self.role!r}")
        if self.status not in STATUSES:
            raise ValueError(f"未知边界状态 {self.status!r}")
        if self.span != self.end_index - self.start_index or self.span <= 0:
            raise ValueError("边界跨度必须是两端极值序号之差")
        if self.confirm_time != self.pivots[-1].confirm_time:
            raise ValueError("边界确认时间必须等于最后一个证据拐点的确认时间")
        if self.available_time != self.pivots[-1].available_time:
            raise ValueError("边界可用时间必须等于最后一个证据拐点的可用时间")
        if self.emits_events and not (self.primary and self.status in LIVE_STATUSES):
            raise ValueError("只有已验证的主结果才能产生以后的突破事件")
        if (self.status in LIVE_STATUSES) != (self.score is not None):
            raise ValueError("软评分只给曾经已验证的边界")
        if self.simplicity_penalty != 0.0:
            raise ValueError("第一版简化惩罚必须为 0")
        if self.context_fit != CONTEXT_FIT_UNUSED:
            raise ValueError("第一版上下文吻合必须标记为未使用")

    def line_at(self, index: int) -> float:
        return self.slope * index + self.intercept

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "revision": self.revision,
            "scale": self.scale,
            "role": self.role,
            "status": self.status,
            "primary": self.primary,
            "alternate_of": self.alternate_of,
            "emits_events": self.emits_events,
            "adjusted": self.adjusted,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "confirm_index": self.confirm_index,
            "confirm_time": self.confirm_time.isoformat(),
            "available_time": self.available_time.isoformat(),
            "slope": self.slope,
            "intercept": self.intercept,
            "normalized_slope": self.normalized_slope,
            "volatility": self.volatility,
            "span": self.span,
            "pivot_indices": list(self.pivot_indices),
            "touch_indices": list(self.touch_indices),
            "touch_clusters": [list(cluster) for cluster in self.touch_clusters],
            "touch_error_median": self.touch_error_median,
            "touch_error_p90": self.touch_error_p90,
            "max_break_depth": self.max_break_depth,
            "max_cross_depth": self.max_cross_depth,
            "cross_count": self.cross_count,
            "longest_violation_bars": self.longest_violation_bars,
            "constraints": [item.to_dict() for item in self.constraints],
            "failed_constraints": list(self.failed_constraints),
            "simplicity_penalty": self.simplicity_penalty,
            "context_fit": self.context_fit,
            "score": None if self.score is None else self.score.to_dict(),
            "pivots": [pivot.to_dict() for pivot in self.pivots],
        }


@dataclass(frozen=True)
class BoundaryResult:
    boundaries: Tuple[Boundary, ...]

    def for_scale(self, scale: str) -> Tuple[Boundary, ...]:
        return tuple(item for item in self.boundaries if item.scale == scale)

    def validated(self, scale: Optional[str] = None, role: Optional[str] = None) -> Tuple[Boundary, ...]:
        return tuple(
            item
            for item in self.boundaries
            if item.status == STATUS_VALIDATED
            and (scale is None or item.scale == scale)
            and (role is None or item.role == role)
        )


def build_boundaries(
    series: StandardSeries,
    pivots: Optional[PivotResult] = None,
    bar_count: Optional[int] = None,
) -> BoundaryResult:
    """从已确认拐点生成支撑和压力。

    `bar_count` 只使用前这么多根，并重新检测拐点。临时末端不参与。
    """
    if bar_count is not None:
        series = series.prefix(bar_count)
        pivots = None
    if pivots is None:
        pivots = detect_pivots(series)
    _require_pivots_in_series(series, pivots)

    drafted: List[Boundary] = []
    for scale in SCALE_NAMES:
        confirmed = pivots.confirmed_for(scale)
        for role in BOUNDARY_ROLES:
            matching = [pivot for pivot in confirmed if pivot.role == _PIVOT_ROLE[role]]
            for run in _gap_runs(series, matching):
                drafted.extend(_candidates_in_run(series, role, run))
    return BoundaryResult(_assign_primaries(drafted, series.params.sideways_normalized_slope))


def format_boundaries(result: BoundaryResult) -> str:
    """给人核对状态、失败约束和斜率的文本。不参与拟合。"""
    lines = []
    for item in result.boundaries:
        failed = ",".join(item.failed_constraints) if item.failed_constraints else "-"
        score = "-" if item.score is None else f"{item.score.total:.3f}"
        kind = "PRIMARY" if item.primary else "ALTERNATE"
        lines.append(
            f"{kind:9} {item.status:9} {item.scale:5} {item.role:10} "
            f"{item.start_index}->{item.end_index} "
            f"slope={item.normalized_slope:.3f} clusters={len(item.touch_clusters)} "
            f"break={item.max_break_depth:.3f} failed={failed} score={score} "
            f"id={item.id}"
        )
    return "\n".join(lines)


def _candidates_in_run(
    series: StandardSeries,
    role: str,
    run: Sequence[Pivot],
) -> List[Boundary]:
    params = series.params
    max_span = params.scale(run[0].scale).max_span
    max_pivots = params.max_pivots_per_line
    drafted: List[Boundary] = []
    for start in range(len(run)):
        for stop in range(start + 1, len(run)):
            chosen = run[start : stop + 1]
            if len(chosen) > max_pivots:
                break
            span = chosen[-1].extreme_index - chosen[0].extreme_index
            if span > max_span:
                break
            if span <= 0 or _crosses_gap(series, chosen[0].extreme_index, chosen[-1].extreme_index):
                continue
            drafted.append(_make_boundary(series, role, chosen))
    return drafted


def _make_boundary(series: StandardSeries, role: str, pivots: Sequence[Pivot]) -> Boundary:
    params = series.params
    start = pivots[0]
    end = pivots[-1]
    volatility = start.volatility
    raw_slope = (end.geometric_price - start.geometric_price) / (end.extreme_index - start.extreme_index)
    raw_intercept = start.geometric_price - raw_slope * start.extreme_index
    slope, intercept, adjusted = _maybe_adjust(
        role,
        pivots,
        raw_slope,
        raw_intercept,
        volatility,
        params,
    )
    normalized = slope / volatility
    touch_distance = params.touch_distance_ratio * volatility
    hard_limit = params.hard_break_ratio * volatility
    distances = [_distance(pivot, slope, intercept) for pivot in pivots]
    break_depths = [_break_depth(role, pivot.geometric_price, slope * pivot.extreme_index + intercept) for pivot in pivots]
    touch_indices = tuple(
        pivot.extreme_index
        for pivot, distance in zip(pivots, distances)
        if distance <= touch_distance
    )
    clusters = _touch_clusters(touch_indices, params.touch_cluster_window)
    touch_errors = tuple(distance / volatility for distance, pivot in zip(distances, pivots) if pivot.extreme_index in touch_indices)
    max_break = max(break_depths) / volatility
    interior_break = _max_interior_break(pivots, break_depths) / volatility
    cross_count, max_cross, longest_violation = _close_violations(
        series,
        role,
        slope,
        intercept,
        start.extreme_index,
        end.extreme_index,
        volatility,
    )
    checks = _constraints(
        pivots,
        role,
        normalized,
        max_break,
        interior_break,
        len(clusters),
        params,
    )
    failed = tuple(check.name for check in checks if not check.passed)
    blocking = tuple(name for name in failed if name != CONSTRAINT_MIN_TOUCH_CLUSTERS)
    if not failed:
        status = STATUS_VALIDATED
    elif not blocking:
        status = STATUS_CANDIDATE
    else:
        status = STATUS_REJECTED
    score = _soft_score(params, start.scale, normalized, clusters, touch_errors, max_break, end.extreme_index - start.extreme_index) if status == STATUS_VALIDATED else None
    return Boundary(
        id=_boundary_id(start.scale, role, start.extreme_index, end.extreme_index),
        revision=1,
        scale=start.scale,
        role=role,
        status=status,
        primary=True,
        alternate_of=None,
        emits_events=False,
        adjusted=adjusted,
        start_index=start.extreme_index,
        end_index=end.extreme_index,
        start_time=start.extreme_time,
        end_time=end.extreme_time,
        confirm_index=end.confirm_index,
        confirm_time=end.confirm_time,
        available_time=end.available_time,
        slope=slope,
        intercept=intercept,
        normalized_slope=normalized,
        volatility=volatility,
        span=end.extreme_index - start.extreme_index,
        pivot_indices=tuple(pivot.extreme_index for pivot in pivots),
        touch_indices=touch_indices,
        touch_clusters=clusters,
        touch_error_median=_median(touch_errors),
        touch_error_p90=_percentile(touch_errors, 90),
        max_break_depth=max_break,
        max_cross_depth=max_cross,
        cross_count=cross_count,
        longest_violation_bars=longest_violation,
        constraints=checks,
        failed_constraints=failed,
        simplicity_penalty=0.0,
        context_fit=CONTEXT_FIT_UNUSED,
        score=score,
        pivots=tuple(pivots),
    )


def _maybe_adjust(
    role: str,
    pivots: Sequence[Pivot],
    slope: float,
    intercept: float,
    volatility: float,
    params: StructureParams,
) -> Tuple[float, float, bool]:
    """在原线的接触拐点上做最小二乘。移动超过一个接触距离，或因此新产生硬破坏，就保留两点线。"""
    touch_distance = params.touch_distance_ratio * volatility
    hard_limit = params.hard_break_ratio * volatility
    touches = [
        pivot
        for pivot in pivots
        if _distance(pivot, slope, intercept) <= touch_distance
        and _break_depth(role, pivot.geometric_price, slope * pivot.extreme_index + intercept) <= hard_limit
    ]
    if len(touches) < 3:
        return slope, intercept, False
    fitted = _ols(
        [pivot.extreme_index for pivot in touches],
        [pivot.geometric_price for pivot in touches],
    )
    if fitted is None:
        return slope, intercept, False
    new_slope, new_intercept = fitted
    start_index = pivots[0].extreme_index
    end_index = pivots[-1].extreme_index
    if _max_shift(slope, intercept, new_slope, new_intercept, start_index, end_index) > touch_distance:
        return slope, intercept, False
    old_worst = max(_break_depth(role, pivot.geometric_price, slope * pivot.extreme_index + intercept) for pivot in pivots)
    new_worst = max(
        _break_depth(role, pivot.geometric_price, new_slope * pivot.extreme_index + new_intercept)
        for pivot in pivots
    )
    if old_worst <= hard_limit < new_worst:
        return slope, intercept, False
    shift = _max_shift(slope, intercept, new_slope, new_intercept, start_index, end_index)
    return new_slope, new_intercept, shift > 1e-12


def _constraints(
    pivots: Sequence[Pivot],
    role: str,
    normalized_slope: float,
    max_break: float,
    interior_break: float,
    cluster_count: int,
    params: StructureParams,
) -> Tuple[ConstraintCheck, ...]:
    sideways = params.sideways_normalized_slope
    if role == ROLE_SUPPORT:
        slope_ok = normalized_slope >= -sideways
        slope_detail = f"支撑的归一化斜率是 {normalized_slope:.3f}，下限 {-sideways:.3f}"
    else:
        slope_ok = normalized_slope <= sideways
        slope_detail = f"压力的归一化斜率是 {normalized_slope:.3f}，上限 {sideways:.3f}"
    hard_ok = max_break <= params.hard_break_ratio
    skipped_ok = interior_break <= params.hard_break_ratio
    clusters_ok = cluster_count >= params.min_touch_clusters
    return (
        ConstraintCheck(CONSTRAINT_TWO_PIVOTS, len(pivots) >= 2, f"同类拐点 {len(pivots)} 个"),
        ConstraintCheck(CONSTRAINT_SLOPE_SIGN, slope_ok, slope_detail),
        ConstraintCheck(
            CONSTRAINT_HARD_BREAK,
            hard_ok,
            f"最大破坏深度 {max_break:.3f} 倍波动，门槛 {params.hard_break_ratio:.3f}",
        ),
        ConstraintCheck(
            CONSTRAINT_NO_SKIPPED_BREAK,
            skipped_ok,
            f"两端之间的最大破坏深度 {interior_break:.3f} 倍波动，门槛 {params.hard_break_ratio:.3f}",
        ),
        ConstraintCheck(
            CONSTRAINT_MIN_TOUCH_CLUSTERS,
            clusters_ok,
            f"接触簇 {cluster_count} 个，至少 {params.min_touch_clusters} 个",
        ),
    )


def _soft_score(
    params: StructureParams,
    scale: str,
    normalized_slope: float,
    clusters: Sequence[Sequence[int]],
    touch_errors: Sequence[float],
    max_break: float,
    span: int,
) -> SoftScore:
    median_error = _median(touch_errors) or 0.0
    geometry = _clamp(1.0 - median_error / params.touch_distance_ratio)
    touch_quality = _clamp(len(clusters) / params.min_touch_clusters)
    significance = _clamp(abs(normalized_slope) * span)
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


def _assign_primaries(drafted: Sequence[Boundary], sideways: float) -> Tuple[Boundary, ...]:
    """分数高并且已经通过硬约束的线优先成为主结果。

    这里只改主结果标签。斜率和状态在生成时已经写定，后到的线不会改它们。
    单独跑到更早的前缀时，主结果只在当时看得到的线里面重选。
    """
    order = sorted(range(len(drafted)), key=lambda index: _quality_key(drafted[index]))
    assigned: List[Optional[Boundary]] = [None] * len(drafted)
    primaries: List[Boundary] = []
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


def _quality_key(boundary: Boundary) -> Tuple[int, float, int, int, str]:
    status_rank = {STATUS_VALIDATED: 0, STATUS_CANDIDATE: 1, STATUS_REJECTED: 2}[boundary.status]
    score = 0.0 if boundary.score is None else -boundary.score.total
    return (status_rank, score, -boundary.span, boundary.confirm_index, boundary.id)


def _similar(primary: Boundary, other: Boundary, sideways: float) -> bool:
    if primary.scale != other.scale or primary.role != other.role or primary.id == other.id:
        return False
    if abs(primary.normalized_slope - other.normalized_slope) > sideways:
        return False
    left = max(primary.start_index, other.start_index)
    right = min(primary.end_index, other.end_index)
    if right < left:
        return False
    scale = (primary.volatility + other.volatility) / 2.0
    tolerance = 0.5 * scale
    for index in (left, right):
        if abs(primary.line_at(index) - other.line_at(index)) > tolerance:
            return False
    return _clusters_mostly_shared(primary.touch_clusters, other.touch_clusters)


def _clusters_mostly_shared(
    left: Sequence[Sequence[int]],
    right: Sequence[Sequence[int]],
) -> bool:
    if not left or not right:
        return False
    shared_left = sum(1 for cluster in left if _cluster_overlaps(cluster, right))
    shared_right = sum(1 for cluster in right if _cluster_overlaps(cluster, left))
    return shared_left / len(left) > 0.5 and shared_right / len(right) > 0.5


def _cluster_overlaps(cluster: Sequence[int], others: Sequence[Sequence[int]]) -> bool:
    wanted = set(cluster)
    return any(wanted.intersection(other) for other in others)


def _close_violations(
    series: StandardSeries,
    role: str,
    slope: float,
    intercept: float,
    start_index: int,
    end_index: int,
    volatility: float,
) -> Tuple[int, float, int]:
    cross_count = 0
    max_depth = 0.0
    longest = 0
    current = 0
    was_beyond = False
    for index in range(start_index, end_index + 1):
        geometric = series.bars[index].geometric
        depth = _break_depth(role, geometric, slope * index + intercept)
        beyond = depth > 0.0
        if beyond and not was_beyond:
            cross_count += 1
        if beyond:
            current += 1
            longest = max(longest, current)
            max_depth = max(max_depth, depth / volatility)
        else:
            current = 0
        was_beyond = beyond
    return cross_count, max_depth, longest


def _touch_clusters(indices: Sequence[int], window: int) -> Tuple[Tuple[int, ...], ...]:
    ordered = sorted(indices)
    if not ordered:
        return ()
    clusters: List[List[int]] = [[ordered[0]]]
    for index in ordered[1:]:
        if index - clusters[-1][-1] <= window:
            clusters[-1].append(index)
        else:
            clusters.append([index])
    return tuple(tuple(cluster) for cluster in clusters)


def _ols(indices: Sequence[int], values: Sequence[float]) -> Optional[Tuple[float, float]]:
    count = len(indices)
    if count < 2:
        return None
    mean_x = sum(indices) / count
    mean_y = sum(values) / count
    variance = sum((index - mean_x) ** 2 for index in indices)
    if variance == 0.0:
        return None
    covariance = sum((index - mean_x) * (value - mean_y) for index, value in zip(indices, values))
    slope = covariance / variance
    return slope, mean_y - slope * mean_x


def _max_shift(
    old_slope: float,
    old_intercept: float,
    new_slope: float,
    new_intercept: float,
    start_index: int,
    end_index: int,
) -> float:
    shift = 0.0
    for index in (start_index, end_index):
        delta = abs((new_slope - old_slope) * index + (new_intercept - old_intercept))
        shift = max(shift, delta)
    return shift


def _max_interior_break(pivots: Sequence[Pivot], depths: Sequence[float]) -> float:
    if len(pivots) <= 2:
        return 0.0
    return max(depths[1:-1])


def _distance(pivot: Pivot, slope: float, intercept: float) -> float:
    return abs(pivot.geometric_price - (slope * pivot.extreme_index + intercept))


def _break_depth(role: str, geometric: float, line_value: float) -> float:
    if role == ROLE_SUPPORT:
        return max(0.0, line_value - geometric)
    return max(0.0, geometric - line_value)


def _gap_runs(series: StandardSeries, pivots: Sequence[Pivot]) -> List[List[Pivot]]:
    runs: List[List[Pivot]] = []
    current: List[Pivot] = []
    for pivot in pivots:
        if current and _crosses_gap(series, current[-1].extreme_index, pivot.extreme_index):
            runs.append(current)
            current = []
        current.append(pivot)
    if current:
        runs.append(current)
    return runs


def _crosses_gap(series: StandardSeries, start_index: int, end_index: int) -> bool:
    for index in range(start_index + 1, end_index + 1):
        if series.bars[index].gap_before:
            return True
    return False


def _boundary_id(scale: str, role: str, start_index: int, end_index: int) -> str:
    return f"{scale}:{role}:{start_index}-{end_index}"


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _percentile(values: Sequence[float], percent: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = percent / 100.0 * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _clamp(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _require_pivots_in_series(series: StandardSeries, pivots: PivotResult) -> None:
    limit = len(series)
    for pivot in pivots.confirmed:
        if pivot.extreme_index >= limit or pivot.confirm_index >= limit:
            raise ValueError(
                f"{pivot.scale} 拐点序号超出序列长度 {limit}: "
                f"extreme={pivot.extreme_index} confirm={pivot.confirm_index}"
            )
