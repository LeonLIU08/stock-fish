"""参数版本登记。

每一版写明基础参数和启用的判断方法。v0 的默认值一旦发出就固定，
以后调阈值要换版本号，不覆盖旧运行目录。`volatility_floor` 是
「波动尺度有下限」的实现值，反转门槛的下限仍然是各尺度的 delta_min。

v2 的基础参数与 v1 相同，并启用方向。v2 已经有运行结果，不改这一版。
v3 的基础参数与 v2 相同，一次启用方向、区间形状和停顿。三个方法分开计算。
停顿才读取最高价、最低价和成交量。
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Dict, Optional, Tuple

PARAM_VERSION = "structure-params-v0"
PARAM_VERSION_V1 = "structure-params-v1"
PARAM_VERSION_V2 = "structure-params-v2"
PARAM_VERSION_V3 = "structure-params-v3"
PARAM_VERSIONS = (
    PARAM_VERSION,
    PARAM_VERSION_V1,
    PARAM_VERSION_V2,
    PARAM_VERSION_V3,
)
# (有效区间内每侧最少接触簇, 是否承认边界全长已经凑满的接触簇)
_ZONE_TOUCH_BY_VERSION = {
    PARAM_VERSION: (3, False),
    PARAM_VERSION_V1: (2, True),
    PARAM_VERSION_V2: (2, True),
    PARAM_VERSION_V3: (2, True),
}
# 该版本启用的判断方法。v0、v1 不写方法参数，也不写 methods 段。
_METHODS_BY_VERSION = {
    PARAM_VERSION: (),
    PARAM_VERSION_V1: (),
    PARAM_VERSION_V2: ("direction",),
    PARAM_VERSION_V3: ("direction", "zone_shape", "pause"),
}


def series_quote_fields(version: str) -> Tuple[str, ...]:
    """该参数版本的数据哈希要覆盖的行情字段。没有则仍只覆盖时间戳和收盘价。"""
    if "pause" in _METHODS_BY_VERSION[normalize_param_version(version)]:
        return ("high", "low", "volume")
    return ()

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
class DirectionParams:
    """方向判断的参数。只出现在启用了 direction 的参数版本里。"""

    epsilon: float = 0.5
    cross_buffer: float = 0.5
    cross_bars: int = 2
    stall_bars: Tuple[int, int, int] = (30, 60, 125)

    def __post_init__(self) -> None:
        self.check()

    def check(self) -> None:
        if self.epsilon <= 0 or self.cross_buffer <= 0:
            raise ValueError("持平容差和穿越缓冲必须为正")
        if self.cross_buffer < self.epsilon:
            raise ValueError("穿越缓冲不得小于持平容差")
        if self.cross_bars < 1:
            raise ValueError("穿越连续根数必须 >= 1")
        if len(self.stall_bars) != 3 or any(int(item) < 1 for item in self.stall_bars):
            raise ValueError("停滞根数必须按短、中、长给出三个不小于 1 的整数")

    def stall_for(self, scale: str) -> int:
        return int(dict(zip(SCALE_NAMES, self.stall_bars))[scale])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "cross_buffer": self.cross_buffer,
            "cross_bars": self.cross_bars,
            "stall_bars": {
                SHORT: int(self.stall_bars[0]),
                MID: int(self.stall_bars[1]),
                LONG: int(self.stall_bars[2]),
            },
        }


@dataclass(frozen=True)
class ZoneShapeParams:
    """区间形状的参数。只出现在启用了 zone_shape 的参数版本里。"""

    level_tolerance: float = 0.2
    converging_width_ratio: float = 0.8
    diverging_width_ratio: float = 1.25

    def __post_init__(self) -> None:
        self.check()

    def check(self) -> None:
        if self.level_tolerance <= 0:
            raise ValueError("水平容差必须为正")
        if self.converging_width_ratio <= 0 or self.diverging_width_ratio <= 0:
            raise ValueError("收敛宽度比和张开宽度比必须为正")
        if self.converging_width_ratio >= self.diverging_width_ratio:
            raise ValueError("收敛宽度比必须小于张开宽度比")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level_tolerance": self.level_tolerance,
            "converging_width_ratio": self.converging_width_ratio,
            "diverging_width_ratio": self.diverging_width_ratio,
        }


@dataclass(frozen=True)
class PauseParams:
    """停顿判断的参数。只出现在启用了 pause 的参数版本里。"""

    pole_min: int = 5
    pole_max: int = 20
    consolidation_min: int = 8
    consolidation_max: int = 25
    atr_lookback: int = 14
    pole_rise: float = 4.0
    efficiency: float = 0.65
    pole_high: float = 1.0
    drawdown_max: float = 0.5
    amplitude_max: float = 0.6
    pivot_half_width: int = 2
    min_highs: int = 2
    min_lows: int = 2
    pivot_span: float = 0.5
    fit_error: float = 0.35
    coverage: float = 0.85
    boundary_tolerance: float = 0.25
    min_width: float = 0.5
    flag_slope: float = 0.25
    flag_width_min: float = 0.85
    flag_width_max: float = 1.15
    midline_min: float = -0.4
    midline_max: float = 0.05
    pennant_slope: float = 0.2
    pennant_width_min: float = 0.3
    pennant_width_max: float = 0.75
    exit_buffer: float = 0.2
    overlap: float = 0.5
    volume_filter: bool = False
    volume_ratio: float = 0.8
    breakout_volume_filter: bool = False
    breakout_volume: float = 1.3
    pullback_bars: int = 2

    def __post_init__(self) -> None:
        self.check()

    def check(self) -> None:
        pole_min = _require_int("旗杆最短", self.pole_min, 2)
        pole_max = _require_int("旗杆最长", self.pole_max, 1)
        cons_min = _require_int("整理最短", self.consolidation_min, 1)
        cons_max = _require_int("整理最长", self.consolidation_max, 1)
        _require_int("A 的回看", self.atr_lookback, 1)
        half = _require_int("拐点半宽", self.pivot_half_width, 1)
        _require_int("最少高点", self.min_highs, 1)
        _require_int("最少低点", self.min_lows, 1)
        _require_int("快速回落根数", self.pullback_bars, 1)
        if pole_min > pole_max:
            raise ValueError("旗杆最短不能超过最长")
        if cons_min > cons_max:
            raise ValueError("整理最短不能超过最长")
        if cons_min - half < 2:
            raise ValueError("整理最短在拐点半宽之后至少还要留下 2 根")
        if (cons_min - 1 - half) < self.pivot_span * (cons_min - 1):
            raise ValueError("整理最短在确认延迟之后放不下两个有跨度的高低点")
        _require_unit("方向效率", self.efficiency)
        _require_unit("回撤上限", self.drawdown_max)
        _require_unit("整理振幅", self.amplitude_max)
        _require_unit("拐点跨度", self.pivot_span)
        _require_unit("覆盖率", self.coverage)
        _require_unit("重叠阈值", self.overlap)
        if not (0 < self.flag_width_min < self.flag_width_max):
            raise ValueError("旗形宽度比必须满足 0 < 下限 < 上限")
        if not (0 < self.pennant_width_min < self.pennant_width_max):
            raise ValueError("三角旗宽度比必须满足 0 < 下限 < 上限")
        if not self.pennant_width_max < self.flag_width_min:
            raise ValueError("三角旗宽度比上限必须小于旗形宽度比下限")
        for name, value in (
            ("旗杆涨幅", self.pole_rise),
            ("旗杆收在高位", self.pole_high),
            ("最小宽度", self.min_width),
            ("拟合误差", self.fit_error),
            ("边界容差", self.boundary_tolerance),
            ("离开缓冲", self.exit_buffer),
            ("三角旗斜率", self.pennant_slope),
            ("突破量能倍数", self.breakout_volume),
            ("旗形斜率差", self.flag_slope),
            ("整理量能比", self.volume_ratio),
        ):
            if not value > 0:
                raise ValueError(f"{name} 必须为正")
        if self.midline_min > self.midline_max:
            raise ValueError("旗形中线下限不能大于上限")
        if not isinstance(self.volume_filter, bool) or not isinstance(self.breakout_volume_filter, bool):
            raise ValueError("量能过滤开关必须是布尔值")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pole_min": self.pole_min,
            "pole_max": self.pole_max,
            "consolidation_min": self.consolidation_min,
            "consolidation_max": self.consolidation_max,
            "atr_lookback": self.atr_lookback,
            "pole_rise": self.pole_rise,
            "efficiency": self.efficiency,
            "pole_high": self.pole_high,
            "drawdown_max": self.drawdown_max,
            "amplitude_max": self.amplitude_max,
            "pivot_half_width": self.pivot_half_width,
            "min_highs": self.min_highs,
            "min_lows": self.min_lows,
            "pivot_span": self.pivot_span,
            "fit_error": self.fit_error,
            "coverage": self.coverage,
            "boundary_tolerance": self.boundary_tolerance,
            "min_width": self.min_width,
            "flag_slope": self.flag_slope,
            "flag_width_min": self.flag_width_min,
            "flag_width_max": self.flag_width_max,
            "midline_min": self.midline_min,
            "midline_max": self.midline_max,
            "pennant_slope": self.pennant_slope,
            "pennant_width_min": self.pennant_width_min,
            "pennant_width_max": self.pennant_width_max,
            "exit_buffer": self.exit_buffer,
            "overlap": self.overlap,
            "volume_filter": self.volume_filter,
            "volume_ratio": self.volume_ratio,
            "breakout_volume_filter": self.breakout_volume_filter,
            "breakout_volume": self.breakout_volume,
            "pullback_bars": self.pullback_bars,
        }


def _require_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} 必须是不小于 {minimum} 的整数")
    return value


def _require_unit(name: str, value: float) -> None:
    if not (0 < value <= 1):
        raise ValueError(f"{name} 必须在 (0, 1] 内")


def pause_params_from_dict(payload: Dict[str, Any] | None) -> PauseParams:
    """把快照里的 pause 字典收成参数。缺省用 2.3.6 的默认值。"""
    if not payload:
        return PauseParams()
    defaults = PauseParams()
    values = defaults.to_dict()
    values.update(payload)
    integers = {
        "pole_min",
        "pole_max",
        "consolidation_min",
        "consolidation_max",
        "atr_lookback",
        "pivot_half_width",
        "min_highs",
        "min_lows",
        "pullback_bars",
    }
    kwargs = {}
    for key, value in values.items():
        if key not in defaults.to_dict():
            continue
        if key in integers:
            if isinstance(value, bool) or (
                isinstance(value, float) and not float(value).is_integer()
            ):
                raise ValueError(f"{key} 必须是整数")
            kwargs[key] = int(value)
        elif key in ("volume_filter", "breakout_volume_filter"):
            if not isinstance(value, bool):
                raise ValueError(f"{key} 必须是布尔值")
            kwargs[key] = value
        else:
            kwargs[key] = float(value)
    return PauseParams(**kwargs)


def zone_shape_params_from_dict(payload: Dict[str, Any] | None) -> ZoneShapeParams:
    """把快照里的 zone_shape 字典收成参数。缺省用 2.2.6 的默认值。"""
    if not payload:
        return ZoneShapeParams()
    return ZoneShapeParams(
        level_tolerance=float(payload.get("level_tolerance", 0.2)),
        converging_width_ratio=float(payload.get("converging_width_ratio", 0.8)),
        diverging_width_ratio=float(payload.get("diverging_width_ratio", 1.25)),
    )


def direction_params_from_dict(payload: Dict[str, Any] | None) -> DirectionParams:
    """把快照里的 direction 字典收成参数。缺省用 2.1.6 的默认值。"""
    if not payload:
        return DirectionParams()
    stalls = payload.get("stall_bars") or {}
    if isinstance(stalls, dict):
        stall_bars = (
            int(stalls.get(SHORT, 30)),
            int(stalls.get(MID, 60)),
            int(stalls.get(LONG, 125)),
        )
    else:
        stall_bars = tuple(int(item) for item in stalls)
    return DirectionParams(
        epsilon=float(payload.get("epsilon", 0.5)),
        cross_buffer=float(payload.get("cross_buffer", 0.5)),
        cross_bars=int(payload.get("cross_bars", 2)),
        stall_bars=stall_bars,  # type: ignore[arg-type]
    )


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
    direction: Optional[DirectionParams] = None
    zone_shape: Optional[ZoneShapeParams] = None
    pause: Optional[PauseParams] = None

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
        enabled = _METHODS_BY_VERSION[self.version]
        if ("direction" in enabled) != (self.direction is not None):
            raise ValueError("direction 参数只写在启用了方向方法的版本里")
        if ("zone_shape" in enabled) != (self.zone_shape is not None):
            raise ValueError("zone_shape 参数只写在启用了区间形状的版本里")
        if ("pause" in enabled) != (self.pause is not None):
            raise ValueError("pause 参数只写在启用了停顿的版本里")
        if self.direction is not None:
            self.direction.check()
        if self.zone_shape is not None:
            self.zone_shape.check()
        if self.pause is not None:
            self.pause.check()

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
            if item.name in ("direction", "zone_shape", "pause"):
                if value is not None:
                    payload[item.name] = value.to_dict()
                continue
            if item.name == "scales":
                payload[item.name] = [scale.to_dict() for scale in value]
            else:
                payload[item.name] = value
        return payload


def enabled_methods(version: str) -> Tuple[str, ...]:
    """该参数版本要跑的判断方法。顺序与登记表一致。"""
    return _METHODS_BY_VERSION[normalize_param_version(version)]


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
    enabled = _METHODS_BY_VERSION[resolved]
    direction = DirectionParams() if "direction" in enabled else None
    zone_shape = ZoneShapeParams() if "zone_shape" in enabled else None
    pause = PauseParams() if "pause" in enabled else None
    return StructureParams(
        version=resolved,
        price_axis=normalize_price_axis(price_axis),
        zone_min_touch_clusters=zone_min,
        zone_credit_full_span=credit,
        direction=direction,
        zone_shape=zone_shape,
        pause=pause,
    )


def partial_history_min_bars(params: StructureParams | None = None) -> int:
    """上市历史短于请求区间时，至少要有这么多根才继续拟合。

    取波动 EWMA 跨度与短尺度最大跨度的一半里较大的那个。
    中尺度和长尺度的 k、门槛、最大跨度保持不变；线的跨度不会超过已有 K 线。
    """
    resolved = default_params() if params is None else params
    return max(resolved.ewma_span, resolved.scale(SHORT).max_span // 2)
