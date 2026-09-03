"""Strategy registration and lookup."""
from __future__ import annotations

from typing import Dict, Type

from analysis.backtest.strategy.base import Strategy

_REGISTRY: Dict[str, Type[Strategy]] = {}


def register_strategy(cls: Type[Strategy]) -> Type[Strategy]:
    if not cls.name:
        raise ValueError(f"策略类 {cls.__name__} 必须定义 name")
    key = cls.name.strip().lower()
    if key in _REGISTRY and _REGISTRY[key] is not cls:
        raise ValueError(f"策略名冲突: {key}")
    _REGISTRY[key] = cls
    return cls


def list_strategies() -> list[str]:
    return sorted(_REGISTRY)


def get_strategy_class(name: str) -> Type[Strategy]:
    key = (name or "").strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"未知策略 {name!r}，已注册: {list_strategies()}")
    return _REGISTRY[key]


def create_strategy(name: str) -> Strategy:
    return get_strategy_class(name)()
