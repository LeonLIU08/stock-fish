"""策略 D1：策略 D0 再加前高回撤止损。

进出和 D0 相同。止损是持有以来最高价乘以 (1 - 回撤比例)，不是成本价的固定亏损。
买入当根的前高先用成交价，当根最低价可以打到这条止损。当根最高价在收盘后
才抬高前高，新止损下一根生效。
"""
from __future__ import annotations

from typing import Optional, Sequence

from analysis.backtest.trend.config import StrategySpec
from analysis.backtest.trend.entries.base import (
    EntrySignal,
    format_number,
    scale_label,
)
from analysis.backtest.trend.entries.d0 import D0Tactic
from analysis.backtest.trend.entries.registry import register_entry
from analysis.backtest.trend.risk import format_percent
from analysis.structure.methods.phase import UPTREND
from analysis.structure.methods.phase import _STRUCTURE_NAME as STRUCTURE_NAMES


@register_entry
class D1Tactic(D0Tactic):
    name = "d1"

    def __init__(self, strategy: StrategySpec, *, price_axis: str) -> None:
        if strategy.exit.drawdown is None:
            raise TypeError("策略 D1 需要前高回撤止损")
        super().__init__(strategy, price_axis=price_axis)
        self.drawdown = float(strategy.exit.drawdown)

    def trigger(self, bars: Sequence) -> Optional[EntrySignal]:
        if self.skip_reason(bars) is not None or not bars:
            return None
        bar = bars[-1]
        if bar.structure != UPTREND or bar.direction_start_index != len(bars) - 1:
            return None
        return EntrySignal(None, _text(bar, self.scale, self.drawdown))


def _text(bar, scale: str, drawdown: float) -> str:
    name = scale_label(scale)
    current = STRUCTURE_NAMES.get(
        bar.structure,
        "不可用" if bar.structure is None else str(bar.structure),
    )
    width = "尚未就绪" if bar.a_t is None else format_number(bar.a_t)
    return (
        f"策略 D1：{name}上升趋势在当根确认，下一根开盘买入。"
        f"不看主状态。持有以来最高价回撤 {format_percent(drawdown)} 止损。"
        f"a_t 为 {width}。"
        f"卖出也等待{name}下降趋势出现，当时{name}方向为{current}。"
    )
