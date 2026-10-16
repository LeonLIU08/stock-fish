"""策略 F：突破与支撑。

试仓条件与策略 E 相同。整理向上突破的小仓、卖出权重和风险放大由会话读取磁带。
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from analysis.backtest.trend.config import BreakEntry, StrategySpec
from analysis.backtest.trend.entries.base import (
    EntrySignal,
    EntryTactic,
    adjusted_close,
    format_number,
    offset_price,
    require_price_axis,
    scale_label,
    state_text,
)
from analysis.backtest.trend.entries.registry import register_entry
from analysis.structure.methods.phase import UP_BIRTH, UP_CONTINUE, UPTREND


@register_entry
class BreakTactic(EntryTactic):
    name = "break"
    provides_stop = True
    allows_add = True
    exit_kind = "break_book"

    def __init__(self, strategy: StrategySpec, *, price_axis: str) -> None:
        if not isinstance(strategy.entry, BreakEntry):
            raise TypeError("策略 F 需要 break 的入场参数")
        self.entry = strategy.entry
        self.exit = strategy.exit
        self.price_axis = require_price_axis(price_axis)
        self.scale = strategy.entry.scale

    def allows(self, bar) -> bool:
        return bar.structure == UPTREND and bar.category in (UP_BIRTH, UP_CONTINUE)

    def skip_reason(self, bars: Sequence) -> Optional[str]:
        if not bars or not _uptrend_appeared(bars[-1], len(bars) - 1):
            return None
        bar = bars[-1]
        if bar.category not in (UP_BIRTH, UP_CONTINUE):
            return "主状态不允许试仓"
        if bar.a_t is None or not math.isfinite(bar.a_t) or bar.a_t < 0:
            return "波动单位尚未就绪"
        return None

    def trigger(self, bars: Sequence) -> Optional[EntrySignal]:
        if self.skip_reason(bars) is not None or not bars:
            return None
        bar = bars[-1]
        if not _uptrend_appeared(bar, len(bars) - 1):
            return None
        if bar.category not in (UP_BIRTH, UP_CONTINUE):
            return None
        close = adjusted_close(bar)
        stop = offset_price(close, bar.a_t, -self.exit.trail_buffer, self.price_axis)
        if not math.isfinite(stop) or stop <= 0:
            return None
        return EntrySignal(float(stop), _text(bar, self.scale, self.price_axis, close, stop))


def _uptrend_appeared(bar, index: int) -> bool:
    return bar.structure == UPTREND and bar.direction_start_index == index


def _text(bar, scale: str, price_axis: str, close: float, stop: float) -> str:
    name = scale_label(scale)
    move = "乘以 e^(-倍数×a_t)" if price_axis == "log" else "减去 倍数×a_t"
    return (
        f"策略 F：{name}上升趋势在当根确认，主状态是{state_text(bar).split('、')[-1]}，"
        f"下一根开盘买入试仓。止损取收盘价 {format_number(close)} {move}，"
        f"复权止损 {format_number(stop)}，a_t 为 {format_number(bar.a_t)}。"
    )
