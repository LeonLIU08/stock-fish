"""Built-in strategy implementations (auto-register on import)."""

from analysis.backtest.strategies import ma_cross  # noqa: F401
from analysis.backtest.strategies import monday_open  # noqa: F401

__all__ = ["ma_cross", "monday_open"]
