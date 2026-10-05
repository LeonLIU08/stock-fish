"""截至当前的收益率波动尺度。

收益率是几何价格的一阶差分。`s_t` 是截至当前的收益率平方的指数加权
平均再开方，只用当前和过去。单根极端收益在进入估计前截断，截断不改
原始收盘价。连续相同收盘价时，发布出来的 `s_t` 不低于 `volatility_floor`，
反转门槛再由 `delta_min` 托底。
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from analysis.structure.config import PRICE_AXIS_LOG, ScaleParams, StructureParams
from analysis.structure.series import StandardSeries


def delta_min_geometric(scale: ScaleParams, price_axis: str, anchor_close: float) -> float:
    """门槛下限。对数轴就是参数本身；均匀轴按段初收盘价换成价格差。"""
    if price_axis == PRICE_AXIS_LOG:
        return scale.delta_min
    if anchor_close <= 0:
        raise ValueError("均匀轴的门槛换算需要正的段初收盘价")
    return anchor_close * (math.exp(scale.delta_min) - 1.0)


def reversal_threshold(
    scale: ScaleParams,
    volatility: float,
    price_axis: str,
    anchor_close: float,
) -> float:
    """`delta = max(k * s, delta_min)`。`s` 和 `delta_min` 都在当次价格轴的单位里。"""
    floor = delta_min_geometric(scale, price_axis, anchor_close)
    return max(scale.k * volatility, floor)


def compute_volatility(series: StandardSeries) -> Tuple[Optional[float], ...]:
    """与 K 线对齐的 `s_t`。第 0 根没有收益，值为 None。

    间断上的那一根不把跨间断的价格跳变写进波动估计，只沿用上一根已经算好的值。
    第一个非零收益不截断，因为在此之前没有「当时的 s」。
    """
    params = series.params
    count = len(series)
    published: List[Optional[float]] = [None] * count
    if count == 0:
        return tuple()

    alpha = 2.0 / (params.ewma_span + 1.0)
    variance = 0.0
    initialized = False
    for index in range(1, count):
        if series.bars[index].gap_before:
            published[index] = published[index - 1]
            continue
        change = series.bars[index].geometric - series.bars[index - 1].geometric
        if not initialized:
            if change == 0.0:
                published[index] = params.volatility_floor
                continue
            variance = change * change
            initialized = True
        else:
            variance = _update_variance(variance, change, alpha, params)
        published[index] = max(math.sqrt(variance), params.volatility_floor)
    return tuple(published)


def _update_variance(
    variance: float,
    change: float,
    alpha: float,
    params: StructureParams,
) -> float:
    raw = math.sqrt(variance)
    if raw > 0.0:
        cap = params.vol_clip_multiple * raw
        if change > cap:
            change = cap
        elif change < -cap:
            change = -cap
    return (1.0 - alpha) * variance + alpha * change * change
