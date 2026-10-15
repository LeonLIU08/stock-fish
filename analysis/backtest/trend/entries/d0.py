"""策略 D0：短尺度趋势进出。

短尺度上升趋势在当根收盘确认后，下一根开盘买入。同一尺度的下降趋势在当根
收盘确认后，下一根开盘卖出。不看主状态，不设初始止损、跟踪止损和衰竭减仓。
同一段上升只买一次。转成震荡或未定时不卖。

买卖尺度必须和本次运行的 `structure.scale` 相同。卖出看这根自己的结构方向，
不另读一个尺度。
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from analysis.backtest.trend.config import D0Entry, StrategySpec
from analysis.backtest.trend.entries.base import (
    EntrySignal,
    EntryTactic,
    format_number,
    require_price_axis,
    scale_label,
)
from analysis.backtest.trend.entries.registry import register_entry
from analysis.structure.methods.phase import UPTREND
from analysis.structure.methods.phase import _STRUCTURE_NAME as STRUCTURE_NAMES


@register_entry
class D0Tactic(EntryTactic):
    name = "d0"
    provides_stop = False
    allows_add = False
    exit_kind = "scale_downtrend"
    sell_on_entry_scale = True

    def __init__(self, strategy: StrategySpec, *, price_axis: str) -> None:
        if not isinstance(strategy.entry, D0Entry):
            raise TypeError("策略 D0 需要 d0 的入场参数")
        self.entry = strategy.entry
        self.scale = strategy.entry.scale
        self.price_axis = require_price_axis(price_axis)

    def allows(self, bar) -> bool:
        return True

    def skip_reason(self, bars: Sequence) -> Optional[str]:
        """上升趋势已经出现、但这根仍不开仓时，返回原因。其他日子返回 None。"""
        if not bars or not _uptrend_appeared(bars[-1], len(bars) - 1):
            return None
        bar = bars[-1]
        if bar.a_t is None or not math.isfinite(bar.a_t) or bar.a_t < 0:
            return "波动单位尚未就绪"
        return None

    def trigger(self, bars: Sequence) -> Optional[EntrySignal]:
        if self.skip_reason(bars) is not None or not bars:
            return None
        bar = bars[-1]
        if not _uptrend_appeared(bar, len(bars) - 1):
            return None
        return EntrySignal(None, _text(bar, self.scale))


def _uptrend_appeared(bar, index: int) -> bool:
    return bar.structure == UPTREND and bar.direction_start_index == index


def _text(bar, scale: str) -> str:
    name = scale_label(scale)
    current = STRUCTURE_NAMES.get(
        bar.structure,
        "不可用" if bar.structure is None else str(bar.structure),
    )
    width = "尚未就绪" if bar.a_t is None else format_number(bar.a_t)
    return (
        f"策略 D0：{name}上升趋势在当根确认，下一根开盘买入。"
        f"不看主状态，不设价格止损。a_t 为 {width}。"
        f"卖出等待{name}下降趋势出现，当时{name}方向为{current}。"
    )
