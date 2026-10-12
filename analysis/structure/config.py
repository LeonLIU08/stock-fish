"""参数版本登记。

每一版的数值写在 `analysis/structure/params/structure-params-vN.yaml`。
v0 的文件一旦发出就固定，以后调阈值要换版本号，不覆盖旧运行目录。
`volatility_floor` 是「波动尺度有下限」的实现值，反转门槛的下限仍然是各尺度的 delta_min。

v2 的基础参数与 v1 相同，并启用方向。v3 在 v2 上启用区间形状和停顿。
v4 与 v3 相同，只把短、中、长的反转倍数改为 1.4、2.8、5.6。
v5 在 v4 上再启用主趋势、次级折返和短期波动。六个方法一次运行全部写出。
v6 保留 v5 的基础层和方法，再启用阶段。方向在这一版记为 direction-v2，只追加推进参照。
停顿才读取最高价、最低价和成交量。阶段不把这些列写成依赖；开盘价只在 v6 声明了并且调用方传入时进入序列和哈希。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

PARAM_VERSION = "structure-params-v0"
PARAM_VERSION_V1 = "structure-params-v1"
PARAM_VERSION_V2 = "structure-params-v2"
PARAM_VERSION_V3 = "structure-params-v3"
PARAM_VERSION_V4 = "structure-params-v4"
PARAM_VERSION_V5 = "structure-params-v5"
PARAM_VERSION_V6 = "structure-params-v6"
# 命令行不写 --param-version 时用这一版。库函数不传版本时仍返回 v0。
CLI_PARAM_VERSION = PARAM_VERSION_V6

_PARAMS_DIR = Path(__file__).resolve().parent / "params"
_VERSION_FILE = re.compile(r"structure-params-v(\d+)$")
_METHOD_NAMES = (
    "direction",
    "zone_shape",
    "pause",
    "major_trend",
    "retracement",
    "minor_swing",
    "phase",
)
_LEVEL_METHODS = ("major_trend", "retracement", "minor_swing")


def series_quote_fields(version: str) -> Tuple[str, ...]:
    """该参数版本的数据哈希要覆盖的行情字段。没有则仍只覆盖时间戳和收盘价。"""
    if "pause" in enabled_methods(version):
        return ("high", "low", "volume")
    return ()


def series_uses_open(version: str) -> bool:
    """启用阶段的版本才接收开盘价。没传入时不改哈希。"""
    return "phase" in enabled_methods(version)

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


@dataclass(frozen=True)
class EmptyMethodParams:
    """没有可调阈值的方法。参数块写成空映射，只用来标明这一版启用了它。"""

    def check(self) -> None:
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {}


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


_PHASE_INTS = (
    "fast_span",
    "impulse_span",
    "record_span",
    "birth_bars",
    "swing_window",
    "wick_window",
    "wick_count",
    "volume_window",
    "volume_baseline",
    "exhaustion_score",
    "exhaustion_bars",
    "bias_window",
    "false_break_window",
    "range_score",
    "score_gap",
    "break_bars",
    "state_bars",
)
_PHASE_UNIT = ("fade_ratio", "wick_ratio", "bias_minimum")
_PHASE_SCORE = ("exhaustion_score", "range_score", "score_gap")


@dataclass(frozen=True)
class PhaseParams:
    """阶段判断的参数。只出现在启用了 phase 的参数版本里。"""

    fast_span: int = 10
    impulse_span: int = 3
    record_span: int = 5
    birth_bars: int = 5
    pullback_multiple: float = 1.0
    pullback_impulse: float = 0.5
    fade_progress: float = 1.0
    fade_ratio: float = 0.3
    swing_window: int = 3
    swing_retrace: float = 1.5
    wick_window: int = 5
    wick_count: int = 2
    wick_ratio: float = 0.5
    wick_amplitude: float = 1.0
    volume_window: int = 3
    volume_baseline: int = 20
    volume_multiple: float = 1.5
    volume_impulse: float = 0.5
    exhaustion_score: int = 2
    exhaustion_bars: int = 2
    structure_shift: float = 0.3
    bias_window: int = 10
    bias_minimum: float = 0.15
    false_break: float = 0.2
    false_break_window: int = 10
    range_score: int = 3
    score_gap: int = 2
    break_bars: int = 2
    break_buffer: float = 0.3
    state_bars: int = 2

    def __post_init__(self) -> None:
        self.check()

    def check(self) -> None:
        for name in _PHASE_INTS:
            _require_int(_PHASE_FIELD_LABEL[name], getattr(self, name), 1)
        for name in _PHASE_UNIT:
            _require_unit(_PHASE_FIELD_LABEL[name], float(getattr(self, name)))
        for name in _PHASE_SCORE:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4:
                raise ValueError(f"{_PHASE_FIELD_LABEL[name]}必须是 1 到 4 的整数")
        for name in (
            "pullback_multiple",
            "pullback_impulse",
            "fade_progress",
            "swing_retrace",
            "wick_amplitude",
            "volume_multiple",
            "volume_impulse",
            "structure_shift",
            "false_break",
            "break_buffer",
        ):
            value = float(getattr(self, name))
            if not value > 0:
                raise ValueError(f"{_PHASE_FIELD_LABEL[name]}必须为正")

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for name in _PHASE_FIELDS:
            value = getattr(self, name)
            payload[name] = int(value) if name in _PHASE_INTS else float(value)
        return payload


_PHASE_FIELDS = (
    "fast_span",
    "impulse_span",
    "record_span",
    "birth_bars",
    "pullback_multiple",
    "pullback_impulse",
    "fade_progress",
    "fade_ratio",
    "swing_window",
    "swing_retrace",
    "wick_window",
    "wick_count",
    "wick_ratio",
    "wick_amplitude",
    "volume_window",
    "volume_baseline",
    "volume_multiple",
    "volume_impulse",
    "exhaustion_score",
    "exhaustion_bars",
    "structure_shift",
    "bias_window",
    "bias_minimum",
    "false_break",
    "false_break_window",
    "range_score",
    "score_gap",
    "break_bars",
    "break_buffer",
    "state_bars",
)
_PHASE_FIELD_LABEL = {
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


def phase_params_from_dict(payload: Dict[str, Any] | None) -> PhaseParams:
    """把快照里的 phase 字典收成参数。缺省用 2.5.6 的默认值。"""
    if not payload:
        return PhaseParams()
    unknown = sorted(set(payload) - set(_PHASE_FIELDS))
    missing = [name for name in _PHASE_FIELDS if name not in payload]
    if unknown:
        raise ValueError("阶段参数有未登记的字段: " + ", ".join(unknown))
    if missing:
        raise ValueError("阶段参数缺少字段: " + ", ".join(missing))
    kwargs: Dict[str, Any] = {}
    for name in _PHASE_FIELDS:
        value = payload[name]
        label = _PHASE_FIELD_LABEL[name]
        if name in _PHASE_INTS:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{label}必须是整数")
            kwargs[name] = value
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{label}必须是数字")
            kwargs[name] = float(value)
    return PhaseParams(**kwargs)


_BASE_KEYS = (
    "ewma_span",
    "vol_clip_multiple",
    "volatility_floor",
    "scales",
    "touch_distance_ratio",
    "touch_cluster_window",
    "min_touch_clusters",
    "zone_min_touch_clusters",
    "zone_credit_full_span",
    "hard_break_ratio",
    "breakout_buffer_ratio",
    "breakout_bars",
    "max_pivots_per_line",
    "channel_width_ratio_min",
    "channel_width_ratio_max",
    "convergence_width_ratio",
    "sideways_normalized_slope",
    "parallel_slope_gap",
    "expire_bars",
    "daily_gap_days",
    "intraday_gap_median_multiple",
    "max_zones_per_scale",
)
_INT_BASE = {
    "ewma_span",
    "touch_cluster_window",
    "min_touch_clusters",
    "zone_min_touch_clusters",
    "breakout_bars",
    "max_pivots_per_line",
    "expire_bars",
    "daily_gap_days",
    "max_zones_per_scale",
}
_PAUSE_INTS = {
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
_PAUSE_BOOLS = {"volume_filter", "breakout_volume_filter"}
_RAW_SPECS: Dict[str, Dict[str, Any]] = {}


def _discover_versions() -> Tuple[str, ...]:
    if not _PARAMS_DIR.is_dir():
        raise RuntimeError(f"找不到参数目录 {_PARAMS_DIR}")
    found = [
        path.stem
        for path in _PARAMS_DIR.glob("structure-params-v*.yaml")
        if _VERSION_FILE.fullmatch(path.stem)
    ]
    versions = tuple(sorted(set(found), key=lambda name: int(_VERSION_FILE.fullmatch(name).group(1))))
    required = (
        PARAM_VERSION,
        PARAM_VERSION_V1,
        PARAM_VERSION_V2,
        PARAM_VERSION_V3,
        PARAM_VERSION_V4,
        PARAM_VERSION_V5,
        PARAM_VERSION_V6,
    )
    missing = [item for item in required if item not in versions]
    if missing:
        raise RuntimeError("缺少参数文件: " + ", ".join(missing))
    return versions


PARAM_VERSIONS = _discover_versions()


def normalize_param_version(version: str) -> str:
    text = str(version or "").strip()
    if text not in PARAM_VERSIONS:
        raise ValueError(f"参数版本必须是 {PARAM_VERSIONS} 之一，收到 {version!r}")
    return text


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} 必须是整数")
    return value


def _as_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} 必须是数字")
    return float(value)


def _as_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} 必须是 true 或 false")
    return value


def _require_mapping(payload: Any, label: str) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} 必须是映射")
    return payload


def _exact_keys(
    payload: Dict[str, Any],
    required: Tuple[str, ...] | set,
    label: str,
    optional: Tuple[str, ...] = (),
) -> None:
    allowed = set(required) | set(optional)
    extra = sorted(set(payload) - allowed)
    missing = [name for name in required if name not in payload]
    if extra:
        raise ValueError(f"{label} 有未登记的字段: {', '.join(extra)}")
    if missing:
        raise ValueError(f"{label} 缺少字段: {', '.join(missing)}")


def _read_spec(version: str) -> Dict[str, Any]:
    cached = _RAW_SPECS.get(version)
    if cached is not None:
        return cached
    path = _PARAMS_DIR / f"{version}.yaml"
    if not path.is_file():
        raise ValueError(f"找不到参数文件 {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    source = path.name
    payload = _require_mapping(payload, source)
    if payload.get("version") != version:
        raise ValueError(f"{source} 的 version 必须是 {version}")
    _exact_keys(payload, ("version", "methods", *_BASE_KEYS), source, optional=_METHOD_NAMES)
    methods = payload.get("methods")
    if not isinstance(methods, list) or any(not isinstance(item, str) for item in methods):
        raise ValueError(f"{source} 的 methods 必须是方法键列表")
    unknown = [item for item in methods if item not in _METHOD_NAMES]
    if unknown:
        raise ValueError(f"{source} 有未知方法: {', '.join(unknown)}")
    if len(methods) != len(set(methods)):
        raise ValueError(f"{source} 的方法键重复")
    ordered = tuple(name for name in _METHOD_NAMES if name in methods)
    if tuple(methods) != ordered:
        raise ValueError(f"{source} 的 methods 请按 {'、'.join(_METHOD_NAMES)} 的顺序写")
    for name in _METHOD_NAMES:
        present = name in methods
        has_block = name in payload and payload[name] is not None
        if present != has_block:
            raise ValueError(f"{source} 里方法 {name} 与它的参数块不一致")
    payload = dict(payload)
    payload["methods"] = ordered
    _RAW_SPECS[version] = payload
    return payload


def enabled_methods(version: str) -> Tuple[str, ...]:
    """该参数版本要跑的判断方法。顺序与登记表一致。"""
    return tuple(_read_spec(normalize_param_version(version))["methods"])


def _scales_from_spec(items: Any, source: str) -> Tuple["ScaleParams", ...]:
    if not isinstance(items, list) or len(items) != len(SCALE_NAMES):
        raise ValueError(f"{source} 的 scales 必须按 short、mid、long 写三项")
    scales = []
    for expected, item in zip(SCALE_NAMES, items):
        row = _require_mapping(item, f"{source} 的 scales")
        _exact_keys(row, ("name", "k", "delta_min", "max_span"), f"{source} 的 {expected}")
        if row.get("name") != expected:
            raise ValueError(f"{source} 的 scales 第 {len(scales) + 1} 项必须是 {expected}")
        scales.append(
            ScaleParams(
                expected,
                _as_float(row["k"], f"{source} 的 {expected}.k"),
                _as_float(row["delta_min"], f"{source} 的 {expected}.delta_min"),
                _as_int(row["max_span"], f"{source} 的 {expected}.max_span"),
            )
        )
    return tuple(scales)


def _direction_from_spec(payload: Any, source: str) -> DirectionParams:
    label = f"{source} 的 direction"
    row = _require_mapping(payload, label)
    _exact_keys(row, ("epsilon", "cross_buffer", "cross_bars", "stall_bars"), label)
    stalls = _require_mapping(row["stall_bars"], f"{label}.stall_bars")
    _exact_keys(stalls, set(SCALE_NAMES), f"{label}.stall_bars")
    return DirectionParams(
        epsilon=_as_float(row["epsilon"], f"{label}.epsilon"),
        cross_buffer=_as_float(row["cross_buffer"], f"{label}.cross_buffer"),
        cross_bars=_as_int(row["cross_bars"], f"{label}.cross_bars"),
        stall_bars=tuple(_as_int(stalls[name], f"{label}.stall_bars.{name}") for name in SCALE_NAMES),
    )


def _zone_shape_from_spec(payload: Any, source: str) -> ZoneShapeParams:
    label = f"{source} 的 zone_shape"
    row = _require_mapping(payload, label)
    names = tuple(item.name for item in fields(ZoneShapeParams))
    _exact_keys(row, names, label)
    return ZoneShapeParams(**{name: _as_float(row[name], f"{label}.{name}") for name in names})


def _empty_from_spec(payload: Any, source: str, name: str) -> EmptyMethodParams:
    label = f"{source} 的 {name}"
    row = _require_mapping(payload, label)
    _exact_keys(row, (), label)
    return EmptyMethodParams()


def _phase_from_spec(payload: Any, source: str) -> PhaseParams:
    label = f"{source} 的 phase"
    row = _require_mapping(payload, label)
    _exact_keys(row, _PHASE_FIELDS, label)
    return phase_params_from_dict(row)


def _pause_from_spec(payload: Any, source: str) -> PauseParams:
    label = f"{source} 的 pause"
    row = _require_mapping(payload, label)
    names = tuple(item.name for item in fields(PauseParams))
    _exact_keys(row, names, label)
    kwargs: Dict[str, Any] = {}
    for name in names:
        value = row[name]
        field_label = f"{label}.{name}"
        if name in _PAUSE_INTS:
            kwargs[name] = _as_int(value, field_label)
        elif name in _PAUSE_BOOLS:
            kwargs[name] = _as_bool(value, field_label)
        else:
            kwargs[name] = _as_float(value, field_label)
    return PauseParams(**kwargs)


def _base_value(key: str, value: Any, source: str) -> Any:
    label = f"{source} 的 {key}"
    if key == "scales":
        return _scales_from_spec(value, source)
    if key in _INT_BASE:
        return _as_int(value, label)
    if key == "zone_credit_full_span":
        return _as_bool(value, label)
    return _as_float(value, label)


def params_kwargs_from_payload(payload: Dict[str, Any], source: str, price_axis: str) -> Dict[str, Any]:
    """把一份已经核对过字段的版本映射收成 StructureParams 的构造参数。"""
    row = _require_mapping(payload, source)
    _exact_keys(row, ("version", "methods", *_BASE_KEYS), source, optional=_METHOD_NAMES)
    methods = row.get("methods")
    if isinstance(methods, tuple):
        method_names = methods
    elif isinstance(methods, list):
        method_names = tuple(methods)
    else:
        raise ValueError(f"{source} 的 methods 必须是方法键列表")
    kwargs: Dict[str, Any] = {
        "version": str(row["version"]),
        "price_axis": normalize_price_axis(price_axis),
    }
    for key in _BASE_KEYS:
        kwargs[key] = _base_value(key, row[key], source)
    builders = {
        "direction": _direction_from_spec,
        "zone_shape": _zone_shape_from_spec,
        "pause": _pause_from_spec,
        "phase": _phase_from_spec,
    }
    for name, builder in builders.items():
        kwargs[name] = builder(row[name], source) if name in method_names else None
    for name in _LEVEL_METHODS:
        kwargs[name] = _empty_from_spec(row[name], source, name) if name in method_names else None
    return kwargs


def _params_kwargs(version: str, price_axis: str = PRICE_AXIS_LOG) -> Dict[str, Any]:
    resolved = normalize_param_version(version)
    return params_kwargs_from_payload(_read_spec(resolved), f"{resolved}.yaml", price_axis)


_V0_KWARGS = _params_kwargs(PARAM_VERSION)


@dataclass(frozen=True)
class StructureParams:
    """一次运行读入的整包参数。数值来自 `analysis/structure/params/<version>.yaml`。"""

    version: str = _V0_KWARGS["version"]
    price_axis: str = _V0_KWARGS["price_axis"]
    ewma_span: int = _V0_KWARGS["ewma_span"]
    vol_clip_multiple: float = _V0_KWARGS["vol_clip_multiple"]
    volatility_floor: float = _V0_KWARGS["volatility_floor"]
    scales: Tuple[ScaleParams, ...] = _V0_KWARGS["scales"]
    touch_distance_ratio: float = _V0_KWARGS["touch_distance_ratio"]
    touch_cluster_window: int = _V0_KWARGS["touch_cluster_window"]
    min_touch_clusters: int = _V0_KWARGS["min_touch_clusters"]
    zone_min_touch_clusters: int = _V0_KWARGS["zone_min_touch_clusters"]
    zone_credit_full_span: bool = _V0_KWARGS["zone_credit_full_span"]
    hard_break_ratio: float = _V0_KWARGS["hard_break_ratio"]
    breakout_buffer_ratio: float = _V0_KWARGS["breakout_buffer_ratio"]
    breakout_bars: int = _V0_KWARGS["breakout_bars"]
    max_pivots_per_line: int = _V0_KWARGS["max_pivots_per_line"]
    channel_width_ratio_min: float = _V0_KWARGS["channel_width_ratio_min"]
    channel_width_ratio_max: float = _V0_KWARGS["channel_width_ratio_max"]
    convergence_width_ratio: float = _V0_KWARGS["convergence_width_ratio"]
    sideways_normalized_slope: float = _V0_KWARGS["sideways_normalized_slope"]
    parallel_slope_gap: float = _V0_KWARGS["parallel_slope_gap"]
    expire_bars: int = _V0_KWARGS["expire_bars"]
    # 相邻日线之间的交易日数（周一至周五）严格大于该值时记间断。
    daily_gap_days: int = _V0_KWARGS["daily_gap_days"]
    intraday_gap_median_multiple: float = _V0_KWARGS["intraday_gap_median_multiple"]
    max_zones_per_scale: int = _V0_KWARGS["max_zones_per_scale"]
    direction: Optional[DirectionParams] = _V0_KWARGS["direction"]
    zone_shape: Optional[ZoneShapeParams] = _V0_KWARGS["zone_shape"]
    pause: Optional[PauseParams] = _V0_KWARGS["pause"]
    major_trend: Optional[EmptyMethodParams] = None
    retracement: Optional[EmptyMethodParams] = None
    minor_swing: Optional[EmptyMethodParams] = None
    phase: Optional[PhaseParams] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "price_axis", normalize_price_axis(self.price_axis))
        if self.version not in PARAM_VERSIONS:
            raise ValueError(f"参数版本必须是 {PARAM_VERSIONS} 之一，收到 {self.version!r}")
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
        enabled = enabled_methods(self.version)
        if ("direction" in enabled) != (self.direction is not None):
            raise ValueError("direction 参数只写在启用了方向方法的版本里")
        if ("zone_shape" in enabled) != (self.zone_shape is not None):
            raise ValueError("zone_shape 参数只写在启用了区间形状的版本里")
        if ("pause" in enabled) != (self.pause is not None):
            raise ValueError("pause 参数只写在启用了停顿的版本里")
        for name in _LEVEL_METHODS:
            present = getattr(self, name) is not None
            if (name in enabled) != present:
                raise ValueError(f"{name} 参数只写在启用了该方法的版本里")
        if ("phase" in enabled) != (self.phase is not None):
            raise ValueError("phase 参数只写在启用了阶段的版本里")
        if self.direction is not None:
            self.direction.check()
        if self.zone_shape is not None:
            self.zone_shape.check()
        if self.pause is not None:
            self.pause.check()
        for name in _LEVEL_METHODS:
            value = getattr(self, name)
            if value is not None:
                value.check()
        if self.phase is not None:
            self.phase.check()

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
            if item.name in _METHOD_NAMES:
                if value is not None:
                    payload[item.name] = value.to_dict()
                continue
            if item.name == "scales":
                payload[item.name] = [scale.to_dict() for scale in value]
            else:
                payload[item.name] = value
        return payload


def default_params(
    price_axis: str = PRICE_AXIS_LOG,
    *,
    version: str = PARAM_VERSION,
) -> StructureParams:
    """返回一份冻结参数包。价格轴只改几何坐标，不改该版本 YAML 里的阈值。"""
    return StructureParams(**_params_kwargs(version, price_axis))


def partial_history_min_bars(params: StructureParams | None = None) -> int:
    """上市历史短于请求区间时，至少要有这么多根才继续拟合。

    取波动 EWMA 跨度与短尺度最大跨度的一半里较大的那个。
    中尺度和长尺度的 k、门槛、最大跨度保持不变；线的跨度不会超过已有 K 线。
    """
    resolved = default_params() if params is None else params
    return max(resolved.ewma_span, resolved.scale(SHORT).max_span // 2)
