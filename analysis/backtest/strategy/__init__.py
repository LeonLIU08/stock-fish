"""Strategy plugin layer."""

from analysis.backtest.strategy.base import Strategy, StrategyVariant
from analysis.backtest.strategy.registry import (
    create_strategy,
    get_strategy_class,
    list_strategies,
    register_strategy,
)

__all__ = [
    "Strategy",
    "StrategyVariant",
    "create_strategy",
    "get_strategy_class",
    "list_strategies",
    "register_strategy",
]
