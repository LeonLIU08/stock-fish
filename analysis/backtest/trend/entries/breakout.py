"""策略 A：顺势突破。

多头只在上升的上涨初生或上涨延续里触发。对数轴把缓冲乘成价格倍数，
均匀轴把同一个系数加在价格上。止损用信号当根及此前若干根的最低价，
再留出 stop_buffer 倍 a_t。系数都来自这个策略自己的参数块。
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from analysis.backtest.trend.config import BreakoutEntry, StrategySpec
from analysis.backtest.trend.entries.base import (
    EntrySignal,
    EntryTactic,
    adjusted_close,
    adjusted_high,
    adjusted_low,
    format_number,
    offset_price,
    require_price_axis,
    state_text,
)
from analysis.backtest.trend.entries.registry import register_entry
from analysis.structure.methods.phase import UP_BIRTH, UP_CONTINUE, UPTREND


@register_entry
class BreakoutTactic(EntryTactic):
    name = "breakout"
    provides_stop = True
    allows_add = True

    def __init__(self, strategy: StrategySpec, *, price_axis: str) -> None:
        if not isinstance(strategy.entry, BreakoutEntry):
            raise TypeError("策略 A 需要 breakout 的入场参数")
        self.entry = strategy.entry
        self.price_axis = require_price_axis(price_axis)

    def allows(self, bar) -> bool:
        return bar.structure == UPTREND and bar.category in (UP_BIRTH, UP_CONTINUE)

    def trigger(self, bars: Sequence) -> Optional[EntrySignal]:
        bar = bars[-1]
        if bar.a_t is None or not math.isfinite(bar.a_t) or bar.a_t < 0:
            return None
        entry = self.entry
        if len(bars) < max(entry.lookback, entry.stop_lookback) + 1:
            return None
        prior = bars[-(entry.lookback + 1):-1]
        prior_high = max(adjusted_high(item) for item in prior)
        close = adjusted_close(bar)
        threshold = offset_price(prior_high, bar.a_t, entry.buffer, self.price_axis)
        if not close > threshold:
            return None
        lows = bars[-(entry.stop_lookback + 1):]
        low = min(adjusted_low(item) for item in lows)
        stop = offset_price(low, bar.a_t, -entry.stop_buffer, self.price_axis)
        if not math.isfinite(stop):
            return None
        return EntrySignal(float(stop), _text(
            bar=bar,
            price_axis=self.price_axis,
            close=close,
            prior_high=prior_high,
            threshold=threshold,
            low=low,
            stop=stop,
            lookback=entry.lookback,
            buffer=entry.buffer,
            stop_lookback=entry.stop_lookback,
            stop_buffer=entry.stop_buffer,
            a_t=bar.a_t,
        ))


def _text(*, bar, price_axis, close, prior_high, threshold, low, stop, lookback, buffer, stop_lookback, stop_buffer, a_t) -> str:
    gate_width = f"{format_number(buffer)}×{format_number(a_t)}"
    stop_width = f"{format_number(stop_buffer)}×{format_number(a_t)}"
    if price_axis == "log":
        gate = f"此前 {lookback} 根最高价 {format_number(prior_high)} 乘以 e^({gate_width})"
        stop_gate = f"最低价 {format_number(low)} 乘以 e^(-{stop_width})"
    else:
        gate = f"此前 {lookback} 根最高价 {format_number(prior_high)} 加上 {gate_width}"
        stop_gate = f"最低价 {format_number(low)} 减去 {stop_width}"
    return (
        f"策略 A：{state_text(bar)}。"
        f"收盘价 {format_number(close)} 高于{gate}，门槛 {format_number(threshold)}，a_t 为 {format_number(a_t)}。"
        f"止损取当根及此前 {stop_lookback} 根的{stop_gate}，复权止损 {format_number(stop)}。"
    )
