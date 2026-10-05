"""Strategy plugin interface for the backtest framework."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, TYPE_CHECKING

import pandas as pd

from analysis.backtest.signals import ChartOverlay, PreparedBars

if TYPE_CHECKING:
    from analysis.backtest.config import BacktestConfig


@dataclass(frozen=True)
class StrategyVariant:
    """One runnable parameter set for a strategy."""

    key: str
    label: str
    params: Dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """Compute indicators + trading signals from OHLCV bars."""

    name: str = ""
    display_name: str = ""

    @abstractmethod
    def variants(self, config: "BacktestConfig") -> List[StrategyVariant]:
        """Return parameter variants selected for this run."""

    @abstractmethod
    def warmup_bars(self, config: "BacktestConfig", variant: StrategyVariant) -> int:
        """Minimum historical bars required before signals are valid."""

    @abstractmethod
    def prepare(
        self,
        bars: pd.DataFrame,
        config: "BacktestConfig",
        variant: StrategyVariant,
    ) -> PreparedBars:
        """Augment bars with indicators and standard signal columns."""

    def chart_overlays(self, variant: StrategyVariant) -> List[ChartOverlay]:
        """Optional price-chart overlays (e.g. moving averages)."""
        return []

    def result_metadata(self, variant: StrategyVariant) -> Dict[str, Any]:
        """Extra fields stored in per-run result payloads."""
        return {"variant_params": dict(variant.params)}
