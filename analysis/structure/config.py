"""参数版本 structure-params-v0。

第 9 节的默认值一旦发出就固定在这个版本里。以后调阈值要换版本号，
不覆盖旧运行目录。`volatility_floor` 是「波动尺度有下限」的实现值，
反转门槛的下限仍然是各尺度的 delta_min。
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Dict, Tuple

PARAM_VERSION = "structure-params-v0"
PARAM_VERSION_V1 = "structure-params-v1"
PARAM_VERSIONS = (PARAM_VERSION, PARAM_VERSION_V1)
# (有效区间内每侧最少接触簇, 是否承认边界全长已经凑满的接触簇)
_ZONE_TOUCH_BY_VERSION = {
    PARAM_VERSION: (3, False),
    PARAM_VERSION_V1: (2, True),
}

SHORT = "short"
MID = "mid"
LONG = "long"
SCALE_NAMES = (SHORT, MID, LONG)

PRICE_AXIS_LOG = "log"
PRICE_AXIS_UNIFORM = "uniform"
PRICE_AXES = (PRICE_AXIS_LOG, PRICE_AXIS_UNIFORM)
PRICE_BASIS_CLOSE = "close"

BAR_INTERVALS = ("1m", "5m", "15m", "60m", "1d")
DAILY_INTERVAL = "1d"

# Nominal bar length. Intraday gap checks use this until a prior interval exists.
INTERVAL_SECONDS = {
    "1m": 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "60m": 60 * 60,
    "1d": 24 * 60 * 60,
}


def normalize_price_axis(price_axis: str) -> str:
    axis = str(price_axis or "").strip().lower()
    if axis not in PRICE_AXES:
        raise ValueError(f"price_axis 必须是 {PRICE_AXES} 之一，收到 {price_axis!r}")
    return axis


def normalize_interval(interval: str) -> str:
    text = str(interval or "").strip().lower()
    if text not in BAR_INTERVALS:
        raise ValueError(f"interval 必须是 {BAR_INTERVALS} 之一，收到 {interval!r}")
    return text


@dataclass(frozen=True)
class ScaleParams:
    """一个尺度的反转倍数、门槛下限和以后趋势线用的最大跨度。"""

    name: str
    k: float
    delta_min: float
    max_span: int

    def __post_init__(self) -> None:
        if self.name not in SCALE_NAMES:
            raise ValueError(f"未知尺度 {self.name!r}")
        if self.k <= 0:
            raise ValueError(f"{self.name} 的 k 必须为正")
        if self.delta_min <= 0:
            raise ValueError(f"{self.name} 的 delta_min 必须为正")
        if self.max_span < 2:
            raise ValueError(f"{self.name} 的 max_span 必须 >= 2")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "k": self.k,
            "delta_min": self.delta_min,
            "max_span": self.max_span,
        }


@dataclass(frozen=True)
class StructureParams:
    """一次运行读入的整包参数。几何、拐点和以后的边界都引用这一份。"""

    version: str = PARAM_VERSION
    price_axis: str = PRICE_AXIS_LOG
    ewma_span: int = 20
    vol_clip_multiple: float = 4.0
    volatility_floor: float = 1e-8
    scales: Tuple[ScaleParams, ...] = (
        ScaleParams(SHORT, 1.5, 0.005, 60),
        ScaleParams(MID, 3.0, 0.01, 120),
        ScaleParams(LONG, 6.0, 0.02, 250),
    )
    touch_distance_ratio: float = 0.5
    touch_cluster_window: int = 3
    min_touch_clusters: int = 3
    zone_min_touch_clusters: int = 3
    zone_credit_full_span: bool = False
    hard_break_ratio: float = 1.0
    breakout_buffer_ratio: float = 0.5
    breakout_bars: int = 2
    max_pivots_per_line: int = 8
    channel_width_ratio_min: float = 0.8
    channel_width_ratio_max: float = 1.25
    convergence_width_ratio: float = 0.8
    sideways_normalized_slope: float = 0.25
    parallel_slope_gap: float = 0.35
    expire_bars: int = 20
    # 相邻日线之间的交易日数（周一至周五）严格大于该值时记间断。
    daily_gap_days: int = 5
    intraday_gap_median_multiple: float = 3.0
    max_zones_per_scale: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "price_axis", normalize_price_axis(self.price_axis))
        if self.version not in _ZONE_TOUCH_BY_VERSION:
            raise ValueError(f"参数版本必须是 {PARAM_VERSIONS} 之一，收到 {self.version!r}")
        expected_min, expected_credit = _ZONE_TOUCH_BY_VERSION[self.version]
        if (
            self.zone_min_touch_clusters != expected_min
            or self.zone_credit_full_span != expected_credit
        ):
            raise ValueError(
                f"{self.version} 的区间接触规则固定为每侧至少 {expected_min} 个，"
                f"全长折算{'开启' if expected_credit else '关闭'}"
            )
        if self.ewma_span < 1:
            raise ValueError("ewma_span 必须 >= 1")
        if self.vol_clip_multiple <= 0:
            raise ValueError("vol_clip_multiple 必须为正")
        if self.volatility_floor <= 0:
            raise ValueError("volatility_floor 必须为正")
        names = tuple(scale.name for scale in self.scales)
        if names != SCALE_NAMES:
            raise ValueError(f"scales 必须按 {SCALE_NAMES} 排列，收到 {names}")
        ks = tuple(scale.k for scale in self.scales)
        if not (ks[0] < ks[1] < ks[2]):
            raise ValueError(f"k 必须满足短 < 中 < 长，收到 {ks}")
        floors = tuple(scale.delta_min for scale in self.scales)
        if not (floors[0] <= floors[1] <= floors[2]):
            raise ValueError(f"delta_min 必须满足短 <= 中 <= 长，收到 {floors}")
        if self.touch_distance_ratio <= 0:
            raise ValueError("touch_distance_ratio 必须为正")
        if self.touch_cluster_window < 1:
            raise ValueError("touch_cluster_window 必须 >= 1")
        if self.min_touch_clusters < 2:
            raise ValueError("min_touch_clusters 必须 >= 2")
        if self.zone_min_touch_clusters < 1:
            raise ValueError("zone_min_touch_clusters 必须 >= 1")
        if self.hard_break_ratio <= 0 or self.breakout_buffer_ratio <= 0:
            raise ValueError("破坏深度和突破缓冲必须为正")
        if self.breakout_bars < 1 or self.expire_bars < 1:
            raise ValueError("突破根数和过期根数必须 >= 1")
        if self.max_pivots_per_line < 2:
            raise ValueError("max_pivots_per_line 必须 >= 2")
        if not (0 < self.channel_width_ratio_min < self.channel_width_ratio_max):
            raise ValueError("通道宽度比必须满足 0 < 下限 < 上限")
        if self.convergence_width_ratio <= 0:
            raise ValueError("convergence_width_ratio 必须为正")
        if self.sideways_normalized_slope <= 0 or self.parallel_slope_gap <= 0:
            raise ValueError("横向斜率和平行斜率差必须为正")
        if self.daily_gap_days < 1:
            raise ValueError("daily_gap_days 必须 >= 1")
        if self.intraday_gap_median_multiple <= 0:
            raise ValueError("intraday_gap_median_multiple 必须为正")
        if self.max_zones_per_scale < 1:
            raise ValueError("max_zones_per_scale 必须 >= 1")

    def scale(self, name: str) -> ScaleParams:
        for item in self.scales:
            if item.name == name:
                return item
        raise KeyError(name)

    def with_price_axis(self, price_axis: str) -> "StructureParams":
        return replace(self, price_axis=normalize_price_axis(price_axis))

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "scales":
                payload[item.name] = [scale.to_dict() for scale in value]
            else:
                payload[item.name] = value
        return payload


def normalize_param_version(version: str) -> str:
    text = str(version or "").strip()
    if text not in PARAM_VERSIONS:
        raise ValueError(f"参数版本必须是 {PARAM_VERSIONS} 之一，收到 {version!r}")
    return text


def default_params(
    price_axis: str = PRICE_AXIS_LOG,
    *,
    version: str = PARAM_VERSION,
) -> StructureParams:
    """返回一份冻结参数包。价格轴只改几何坐标，不改该版本里的阈值。"""
    resolved = normalize_param_version(version)
    zone_min, credit = _ZONE_TOUCH_BY_VERSION[resolved]
    return StructureParams(
        version=resolved,
        price_axis=normalize_price_axis(price_axis),
        zone_min_touch_clusters=zone_min,
        zone_credit_full_span=credit,
    )


def partial_history_min_bars(params: StructureParams | None = None) -> int:
    """上市历史短于请求区间时，至少要有这么多根才继续拟合。

    取波动 EWMA 跨度与短尺度最大跨度的一半里较大的那个。
    中尺度和长尺度的 k、门槛、最大跨度保持不变；线的跨度不会超过已有 K 线。
    """
    resolved = default_params() if params is None else params
    return max(resolved.ewma_span, resolved.scale(SHORT).max_span // 2)
