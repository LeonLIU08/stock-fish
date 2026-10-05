"""趋势结构引擎的阶段 0–1：标准序列与因果拐点。

边界、趋势区间和 HTML 报告不在当前范围内。
"""

from analysis.structure.config import (
    BAR_INTERVALS,
    LONG,
    MID,
    PARAM_VERSION,
    PRICE_AXIS_LOG,
    PRICE_AXIS_UNIFORM,
    PRICE_BASIS_CLOSE,
    SCALE_NAMES,
    SHORT,
    StructureParams,
    default_params,
)
from analysis.structure.pivots import (
    Pivot,
    PivotDetector,
    PivotResult,
    TemporaryExtreme,
    detect_pivots,
    format_trace,
)
from analysis.structure.series import (
    IDENTITY_FILENAME,
    NonPositivePriceError,
    StandardSeries,
    build_series,
    result_directory,
    write_series_identity,
)
from analysis.structure.volatility import compute_volatility, reversal_threshold

__all__ = [
    "BAR_INTERVALS",
    "IDENTITY_FILENAME",
    "LONG",
    "MID",
    "PARAM_VERSION",
    "PRICE_AXIS_LOG",
    "PRICE_AXIS_UNIFORM",
    "PRICE_BASIS_CLOSE",
    "SCALE_NAMES",
    "SHORT",
    "NonPositivePriceError",
    "Pivot",
    "PivotDetector",
    "PivotResult",
    "StandardSeries",
    "StructureParams",
    "TemporaryExtreme",
    "build_series",
    "compute_volatility",
    "default_params",
    "detect_pivots",
    "format_trace",
    "result_directory",
    "reversal_threshold",
    "write_series_identity",
]
