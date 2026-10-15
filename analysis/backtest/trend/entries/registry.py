"""入场插件注册表。一个账户只按名字装一个。"""
from __future__ import annotations

from typing import Dict, Type

from analysis.backtest.trend.config import StrategySpec
from analysis.backtest.trend.entries.base import EntryTactic

_REGISTRY: Dict[str, Type[EntryTactic]] = {}
_LOADED = False


def register_entry(cls: Type[EntryTactic]) -> Type[EntryTactic]:
    if not getattr(cls, "name", ""):
        raise ValueError(f"入场插件 {cls.__name__} 必须定义 name")
    key = cls.name.strip().lower()
    existing = _REGISTRY.get(key)
    if existing is not None and existing is not cls:
        raise ValueError(f"入场策略名冲突: {key}")
    _REGISTRY[key] = cls
    return cls


def list_entries() -> list[str]:
    _ensure_loaded()
    return sorted(_REGISTRY)


def create_entry(name: str, strategy: StrategySpec, *, price_axis: str) -> EntryTactic:
    """按策略名新建一个插件。每次都是新实例，账户之间不共享状态。"""
    _ensure_loaded()
    key = (name or "").strip().lower()
    cls = _REGISTRY.get(key)
    if cls is None:
        raise KeyError(f"未知入场策略 {name!r}，已注册: {list_entries()}")
    if not cls.provides_stop and getattr(cls, "exit_kind", "") not in ("scale_downtrend", "scale_book"):
        raise ValueError(f"没有止损的插件拒绝加载: {key}")
    return cls(strategy, price_axis=price_axis)


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    from analysis.backtest.trend.entries import breakout, d0, d1, pullback, scale, trend  # noqa: F401

    _LOADED = True
