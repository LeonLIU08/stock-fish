"""策略 C：只按趋势进出。

短尺度上升趋势在当根收盘确认后，下一根开盘买入。中尺度下降趋势在当根收盘
确认后，下一根开盘卖出。不看主状态，不设初始止损、跟踪止损和衰竭减仓。
同一段上升只买一次；这段还没结束时，不会因为中期下降退出之后再买回来。
中期当时已经是下降，则这次上升出现不开仓。

买入尺度必须和本次运行的 `structure.scale` 相同，磁带上的 `structure` 就是它。
卖出尺度写在这个策略自己的参数块里，磁带把它放在 `exit_structure`。
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from analysis.backtest.trend.config import StrategySpec, TrendEntry
from analysis.backtest.trend.entries.base import (
    EntrySignal,
    EntryTactic,
    format_number,
    require_price_axis,
    scale_label,
)
from analysis.backtest.trend.entries.registry import register_entry
from analysis.structure.methods.phase import DOWNTREND, UPTREND
from analysis.structure.methods.phase import _STRUCTURE_NAME as STRUCTURE_NAMES


@register_entry
class TrendTactic(EntryTactic):
    name = "trend"
    provides_stop = False
    allows_add = False
    exit_kind = "scale_downtrend"

    def __init__(self, strategy: StrategySpec, *, price_axis: str) -> None:
        if not isinstance(strategy.entry, TrendEntry):
            raise TypeError("策略 C 需要 trend 的入场参数")
        self.entry = strategy.entry
        self.price_axis = require_price_axis(price_axis)

    def allows(self, bar) -> bool:
        return True

    def skip_reason(self, bars: Sequence) -> Optional[str]:
        """上升趋势已经出现、但这根仍不开仓时，返回原因。其他日子返回 None。"""
        if not bars or not _uptrend_appeared(bars[-1], len(bars) - 1):
            return None
        bar = bars[-1]
        if bar.exit_scale != self.entry.sell_scale:
            return "缺少卖出尺度"
        if bar.a_t is None or not math.isfinite(bar.a_t) or bar.a_t < 0:
            return "波动单位尚未就绪"
        if bar.exit_structure == DOWNTREND:
            return f"{scale_label(self.entry.sell_scale)}下降，不开仓"
        return None

    def trigger(self, bars: Sequence) -> Optional[EntrySignal]:
        if self.skip_reason(bars) is not None or not bars:
            return None
        bar = bars[-1]
        if not _uptrend_appeared(bar, len(bars) - 1):
            return None
        return EntrySignal(None, _text(bar, self.entry.buy_scale, self.entry.sell_scale))


def _uptrend_appeared(bar, index: int) -> bool:
    return bar.structure == UPTREND and bar.direction_start_index == index


def _text(bar, buy_scale: str, sell_scale: str) -> str:
    buy = scale_label(buy_scale)
    sell = scale_label(sell_scale)
    current = STRUCTURE_NAMES.get(
        bar.exit_structure,
        "不可用" if bar.exit_structure is None else str(bar.exit_structure),
    )
    width = "尚未就绪" if bar.a_t is None else format_number(bar.a_t)
    return (
        f"策略 C：{buy}上升趋势在当根确认，下一根开盘买入。"
        f"不看主状态，不设价格止损。a_t 为 {width}。"
        f"卖出等待{sell}下降趋势出现，当时{sell}方向为{current}。"
    )
