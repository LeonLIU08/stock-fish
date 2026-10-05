"""Built-in strategy implementations (auto-register on import)."""

from analysis.backtest.strategies import ma_cross  # noqa: F401

__all__ = ["ma_cross"]
