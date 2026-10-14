"""入场插件只回答当根有没有信号，以及结构止损。它不改账户。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from analysis.structure.methods.phase import _NAMES as PHASE_NAMES
from analysis.structure.methods.phase import _STRUCTURE_NAME as STRUCTURE_NAMES


@dataclass(frozen=True)
class EntrySignal:
    """`adjusted_stop` 是信号当根的复权止损。策略 C 不设价格止损，这里是 None。"""

    adjusted_stop: Optional[float]
    text: str


SCALE_LABELS = {
    "short": "短期",
    "mid": "中期",
    "long": "长期",
}


def scale_label(scale: Optional[str]) -> str:
    if not scale:
        return "该尺度"
    return SCALE_LABELS.get(scale, scale)


class EntryTactic:
    """一个账户只装一个插件。`bars` 停在信号当根，不含以后的 K 线。

    没有价格止损、也没有 `exit_kind == scale_downtrend` 的插件拒绝加载。
    `allows_add` 留给以后的加仓，第一期会话不用。
    """

    name = ""
    provides_stop = False
    allows_add = False
    exit_kind = ""

    def propose(self, bars: Sequence) -> Optional[EntrySignal]:
        if not bars or not self.allows(bars[-1]):
            return None
        signal = self.trigger(bars)
        if signal is None:
            return None
        if not isinstance(signal, EntrySignal):
            raise TypeError("插件必须返回 EntrySignal 或 None")
        stop = signal.adjusted_stop
        if self.provides_stop:
            if isinstance(stop, bool) or not isinstance(stop, (int, float)) or not math.isfinite(float(stop)):
                raise ValueError("入场插件必须给出有限止损")
        elif stop is not None:
            raise ValueError("不设价格止损的插件不能返回止损价")
        return signal

    def allows(self, bar) -> bool:
        """当根的方向和主状态是否允许这个插件寻找入场。"""
        return False

    def environment_block(self, bar) -> Optional[str]:
        """环境不允许时返回原因。`propose` 被覆盖的测试插件不受影响。"""
        if self.allows(bar):
            return None
        return "环境不允许"

    def trigger(self, bars: Sequence) -> Optional[EntrySignal]:
        raise NotImplementedError


def require_price_axis(price_axis: str) -> str:
    if price_axis not in ("log", "uniform"):
        raise ValueError(f"价格轴必须是 log 或 uniform，收到 {price_axis}")
    return price_axis


def adjusted_high(bar) -> float:
    return float(bar.high) * float(bar.factor)


def adjusted_low(bar) -> float:
    return float(bar.low) * float(bar.factor)


def adjusted_close(bar) -> float:
    return float(bar.close) * float(bar.factor)


def geometric_close(close: float, price_axis: str) -> float:
    """对数轴用对数，均匀轴用价格。EMA10 和这条收盘价在同一套坐标里比较。"""
    require_price_axis(price_axis)
    if price_axis == "log":
        if close <= 0:
            raise ValueError("对数轴的收盘价必须为正")
        return math.log(close)
    return close


def offset_price(level: float, a_t: float, coefficient: float, price_axis: str) -> float:
    """对数轴乘以 e^{k a_t}，均匀轴加上 k a_t。k 为负时就是往下留出缓冲。"""
    require_price_axis(price_axis)
    if a_t < 0:
        raise ValueError("a_t 不能为负")
    if price_axis == "log":
        if level <= 0:
            raise ValueError("对数轴的价格必须为正")
        return float(level) * math.exp(float(coefficient) * float(a_t))
    return float(level) + float(coefficient) * float(a_t)


def format_number(value: float) -> str:
    text = f"{float(value):.8f}".rstrip("0").rstrip(".")
    return text or "0"


def state_text(bar) -> str:
    structure = STRUCTURE_NAMES.get(bar.structure, "不可用" if bar.structure is None else str(bar.structure))
    category = PHASE_NAMES.get(bar.category, "不可用" if bar.category is None else str(bar.category))
    return f"{structure}、{category}"
