"""多头止损只升不降。跟踪在收盘后写下，下一根才生效。"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from analysis.structure.methods.phase import UP_EXHAUSTION, UPTREND


def long_trend_failed(structure: Optional[str]) -> bool:
    """上升以外的方向，下一根全部退出。没有方向对象也算失效。"""
    return structure != UPTREND


def exhaustion_due(category: Optional[str], already_reduced: bool, action: str) -> bool:
    if action != "reduce" or already_reduced:
        return False
    return category == UP_EXHAUSTION


def next_long_stop(
    current: float,
    highest: float,
    a_t: Optional[float],
    *,
    buffer: float,
    price_axis: str,
    category: Optional[str],
    trail_states: Sequence[str],
) -> float:
    """初生和其他不在跟踪列表里的状态保持原止损。新值更低时丢掉。"""
    if category not in trail_states or a_t is None:
        return current
    if a_t < 0 or buffer < 0:
        raise ValueError("跟踪止损的波动和缓冲不能为负")
    if price_axis == "log":
        if highest <= 0:
            raise ValueError("跟踪止损的最高价必须为正")
        candidate = highest * math.exp(-float(buffer) * float(a_t))
    elif price_axis == "uniform":
        candidate = highest - float(buffer) * float(a_t)
    else:
        raise ValueError(f"价格轴必须是 log 或 uniform，收到 {price_axis}")
    if candidate > current:
        return candidate
    return current


def reduce_shares(shares: int, fraction: float, lot_size: int) -> tuple[int, bool]:
    """向下取整到整手。取整为 0 或剩余不足一手时，改为全部退出。"""
    if isinstance(shares, bool) or not isinstance(shares, int) or shares < 1:
        raise ValueError("减仓前的股数必须是正整数")
    if isinstance(lot_size, bool) or not isinstance(lot_size, int) or lot_size < 1:
        raise ValueError("每手股数必须是正整数")
    if not 0 < float(fraction) < 1:
        raise ValueError("减仓比例必须在 (0, 1)")
    raw = int(math.floor(float(shares) * float(fraction) + 1e-9))
    qty = (raw // lot_size) * lot_size
    if qty <= 0 or shares - qty < lot_size:
        return shares, True
    return qty, False
