"""入场插件只回答当根有没有信号，以及结构止损。它不改账户。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class EntrySignal:
    """`adjusted_stop` 是信号当根的复权止损。"""

    adjusted_stop: float
    text: str


class EntryTactic:
    """一个账户只装一个插件。`bars` 停在信号当根，不含以后的 K 线。"""

    name = ""

    def propose(self, bars: Sequence) -> Optional[EntrySignal]:
        raise NotImplementedError
