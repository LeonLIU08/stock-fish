"""Extensible rule-based backtest framework."""

from analysis.backtest import strategies as _builtin_strategies  # noqa: F401
from analysis.backtest.config import (
    BAR_INTERVALS,
    BENCHMARKS,
    DEFAULT_CONFIG_PATH,
    DEFAULT_SYMBOLS,
    BacktestConfig,
    CostModel,
    load_backtest_config,
)
from analysis.backtest.runner import run_backtest
from analysis.backtest.signals import (
    ChartOverlay,
    PreparedBars,
    SignalAction,
    attach_signals,
    ensure_signal_columns,
    validate_signal_columns,
)
from analysis.backtest.strategy import (
    Strategy,
    StrategyVariant,
    create_strategy,
    get_strategy_class,
    list_strategies,
    register_strategy,
)
from analysis.backtest.strategies.ma_cross import DEFAULT_MA_SCHEMES, MA_SCHEMES

__all__ = [
    "BAR_INTERVALS",
    "BENCHMARKS",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_MA_SCHEMES",
    "DEFAULT_SYMBOLS",
    "MA_SCHEMES",
    "BacktestConfig",
    "ChartOverlay",
    "CostModel",
    "PreparedBars",
    "SignalAction",
    "Strategy",
    "StrategyVariant",
    "attach_signals",
    "create_strategy",
    "ensure_signal_columns",
    "get_strategy_class",
    "list_strategies",
    "load_backtest_config",
    "register_strategy",
    "run_backtest",
    "validate_signal_columns",
]
