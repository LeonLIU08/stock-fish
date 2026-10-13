"""趋势回测的入场插件。一个账户只装注册表里的一个。"""

from analysis.backtest.trend.entries.base import EntrySignal, EntryTactic
from analysis.backtest.trend.entries.registry import create_entry, list_entries, register_entry

__all__ = [
    "EntrySignal",
    "EntryTactic",
    "create_entry",
    "list_entries",
    "register_entry",
]
