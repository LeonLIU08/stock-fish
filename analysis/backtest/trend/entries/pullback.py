"""策略 B：顺势回调结束。

多头在上涨回调或上涨延续里触发。此前若干根（不含当根）里要有一根已确认的
上涨回调，并且那一根和当根还在同一段上升里。结构方向离开上升，或中间出现
结构间断，这段回调记忆就清空。当根仍是上涨回调时也可以触发，不必先变成延续。

空头镜像留到打开做空的那一期，这里只定义、不实现：反弹后几何收盘跌回 EMA10
之下，且跌破此前 3 根最低价，止损放在本轮反弹最高价之上。`allow_short` 为假时
会话不调用空头插件。
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from analysis.backtest.trend.config import PullbackEntry, StrategySpec
from analysis.backtest.trend.entries.base import (
    EntrySignal,
    EntryTactic,
    adjusted_close,
    adjusted_high,
    adjusted_low,
    format_number,
    geometric_close,
    offset_price,
    require_price_axis,
    state_text,
)
from analysis.backtest.trend.entries.registry import register_entry
from analysis.structure.methods.phase import UP_CONTINUE, UP_PULLBACK, UPTREND


@register_entry
class PullbackTactic(EntryTactic):
    """回调结束只用于空仓开仓。`allows_add` 为假。"""

    name = "pullback"
    provides_stop = True
    allows_add = False

    def __init__(self, strategy: StrategySpec, *, price_axis: str) -> None:
        if not isinstance(strategy.entry, PullbackEntry):
            raise TypeError("策略 B 需要 pullback 的入场参数")
        self.entry = strategy.entry
        self.price_axis = require_price_axis(price_axis)

    def allows(self, bar) -> bool:
        return bar.structure == UPTREND and bar.category in (UP_PULLBACK, UP_CONTINUE)

    def trigger(self, bars: Sequence) -> Optional[EntrySignal]:
        bar = bars[-1]
        if bar.a_t is None or bar.ema is None:
            return None
        if not math.isfinite(bar.a_t) or bar.a_t < 0 or not math.isfinite(bar.ema):
            return None
        entry = self.entry
        current = len(bars) - 1
        if current < entry.break_lookback:
            return None
        recent = _latest_pullback(bars, current, entry.lookback)
        if recent is None:
            return None
        close = adjusted_close(bar)
        geometric = geometric_close(close, self.price_axis)
        if not geometric > float(bar.ema):
            return None
        prior = bars[current - entry.break_lookback:current]
        prior_high = max(adjusted_high(item) for item in prior)
        if not close > prior_high:
            return None
        origin = _pullback_origin(bars, recent, current)
        low = min(adjusted_low(item) for item in bars[origin:current + 1])
        stop = offset_price(low, bar.a_t, -entry.stop_buffer, self.price_axis)
        if not math.isfinite(stop):
            return None
        return EntrySignal(float(stop), _text(
            bar=bar,
            price_axis=self.price_axis,
            close=close,
            geometric=geometric,
            prior_high=prior_high,
            low=low,
            stop=stop,
            origin=origin,
            lookback=entry.lookback,
            break_lookback=entry.break_lookback,
            stop_buffer=entry.stop_buffer,
            a_t=bar.a_t,
        ))


def _latest_pullback(bars: Sequence, current: int, lookback: int) -> Optional[int]:
    """此前 lookback 根（不含当根）里，同一段上升中最近的一根上涨回调。"""
    start = max(0, current - lookback)
    found = None
    for index in range(start, current):
        if bars[index].category != UP_PULLBACK:
            continue
        if _same_uptrend(bars, index, current):
            found = index
    return found


def _pullback_origin(bars: Sequence, recent: int, current: int) -> int:
    """本轮上涨回调的起点。窗口里有多段时，用最近一段的起点，它可以早于这 5 根。"""
    declared = bars[recent].phase_start_index
    if isinstance(declared, bool):
        raise ValueError("上涨回调的起点必须是整数序号")
    if declared is None:
        origin = recent
        while origin > 0 and bars[origin - 1].category == UP_PULLBACK and _same_uptrend(bars, origin - 1, current):
            origin -= 1
        return origin
    if not isinstance(declared, int) or not 0 <= declared <= recent:
        raise ValueError("上涨回调的起点不在可见 K 线里")
    if not _same_uptrend(bars, declared, current):
        raise ValueError("上涨回调的起点和信号不在同一段上升里")
    for index in range(declared, recent + 1):
        if bars[index].category != UP_PULLBACK:
            raise ValueError("上涨回调的起点到该段最后一根应该都是上涨回调")
    return declared


def _same_uptrend(bars: Sequence, earlier: int, current: int) -> bool:
    """同一段上升，且中间没有结构间断。间断落在 earlier 自己身上时，那段记忆从它重新开始。"""
    if bars[earlier].structure != UPTREND or bars[current].structure != UPTREND:
        return False
    left = bars[earlier].direction_start_index
    right = bars[current].direction_start_index
    if left is not None and right is not None and left != right:
        return False
    for index in range(earlier + 1, current + 1):
        bar = bars[index]
        if bar.gap_before or bar.structure != UPTREND:
            return False
    return True


def _text(
    *,
    bar,
    price_axis,
    close,
    geometric,
    prior_high,
    low,
    stop,
    origin,
    lookback,
    break_lookback,
    stop_buffer,
    a_t,
) -> str:
    width = f"{format_number(stop_buffer)}×{format_number(a_t)}"
    if price_axis == "log":
        stop_gate = f"最低价 {format_number(low)} 乘以 e^(-{width})"
    else:
        stop_gate = f"最低价 {format_number(low)} 减去 {width}"
    return (
        f"策略 B：{state_text(bar)}。"
        f"此前 {lookback} 根里有同一段上升的上涨回调，起点序号 {origin}。"
        f"几何收盘价 {format_number(geometric)} 高于 EMA10 {format_number(bar.ema)}，"
        f"收盘价 {format_number(close)} 高于此前 {break_lookback} 根最高价 {format_number(prior_high)}。"
        f"a_t 为 {format_number(a_t)}。止损取回调起点到当根的{stop_gate}，复权止损 {format_number(stop)}。"
    )
