"""Golden / death cross on two moving averages."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from analysis.backtest.config import BacktestConfig
from analysis.backtest.signals import (
    ChartOverlay,
    PreparedBars,
    SignalAction,
    attach_signals,
)
from analysis.backtest.strategy.base import Strategy, StrategyVariant
from analysis.backtest.strategy.registry import register_strategy

DEFAULT_MA_SCHEMES: Dict[str, Tuple[int, int]] = {
    "ma5_ma10": (5, 10),
    "ma5_ma20": (5, 20),
    "ma10_ma20": (10, 20),
}

# Back-compat alias used by tests / external imports.
MA_SCHEMES = DEFAULT_MA_SCHEMES


def _scheme_map(config: BacktestConfig) -> Dict[str, Tuple[int, int]]:
    raw = config.strategy_params.get("schemes") or DEFAULT_MA_SCHEMES
    out: Dict[str, Tuple[int, int]] = {}
    for key, pair in raw.items():
        if isinstance(pair, dict):
            fast, slow = int(pair["fast"]), int(pair["slow"])
        else:
            fast, slow = int(pair[0]), int(pair[1])
        out[str(key)] = (fast, slow)
    return out


def compute_ma_cross_signals(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    *,
    variant_key: str = "",
) -> pd.DataFrame:
    """Append MA columns and translate crosses into standard signal columns."""
    if fast < 1 or slow < 1 or fast >= slow:
        raise ValueError(f"均线窗口非法: fast={fast}, slow={slow}")
    out = df.copy()
    close = out["close"].astype(float)
    out["ma_fast"] = close.rolling(fast, min_periods=fast).mean()
    out["ma_slow"] = close.rolling(slow, min_periods=slow).mean()
    prev_fast = out["ma_fast"].shift(1)
    prev_slow = out["ma_slow"].shift(1)
    valid = out["ma_fast"].notna() & out["ma_slow"].notna() & prev_fast.notna() & prev_slow.notna()
    golden = valid & (prev_fast <= prev_slow) & (out["ma_fast"] > out["ma_slow"])
    death = valid & (prev_fast >= prev_slow) & (out["ma_fast"] < out["ma_slow"])

    actions: List[str] = []
    reasons: List[str] = []
    for is_golden, is_death in zip(golden.tolist(), death.tolist()):
        if is_golden:
            actions.append(SignalAction.BUY.value)
            reasons.append("golden_cross")
        elif is_death:
            actions.append(SignalAction.SELL.value)
            reasons.append("death_cross")
        else:
            actions.append(SignalAction.HOLD.value)
            reasons.append("")

    out = attach_signals(out, actions, reasons=reasons)
    out["variant"] = variant_key
    out["ma_fast_n"] = fast
    out["ma_slow_n"] = slow
    # Legacy boolean columns kept for analytics / debugging.
    out["golden"] = golden
    out["death"] = death
    return out


@register_strategy
class MaCrossStrategy(Strategy):
    name = "ma_cross"
    display_name = "均线金叉银叉"

    def variants(self, config: BacktestConfig) -> List[StrategyVariant]:
        table = _scheme_map(config)
        selected = config.variants or list(table.keys())
        unknown = [key for key in selected if key not in table]
        if unknown:
            raise ValueError(f"未知均线方案 {unknown}，可选: {list(table)}")
        result: List[StrategyVariant] = []
        for key in selected:
            fast, slow = table[key]
            result.append(
                StrategyVariant(
                    key=key,
                    label=f"MA{fast}/MA{slow}",
                    params={"fast": fast, "slow": slow, "scheme": key},
                )
            )
        return result

    def warmup_bars(self, config: BacktestConfig, variant: StrategyVariant) -> int:
        slow = int(variant.params.get("slow", 1))
        return slow + 5

    def prepare(
        self,
        bars: pd.DataFrame,
        config: BacktestConfig,
        variant: StrategyVariant,
    ) -> PreparedBars:
        fast = int(variant.params["fast"])
        slow = int(variant.params["slow"])
        signaled = compute_ma_cross_signals(bars, fast, slow, variant_key=variant.key)
        return PreparedBars(
            bars=signaled,
            overlay_columns=["ma_fast", "ma_slow"],
            context={
                "fast": fast,
                "slow": slow,
                "scheme": variant.key,
            },
        )

    def chart_overlays(self, variant: StrategyVariant) -> List[ChartOverlay]:
        fast = int(variant.params.get("fast", 0))
        slow = int(variant.params.get("slow", 0))
        return [
            ChartOverlay(column="ma_fast", label=f"MA{fast}", color="#fbbf24"),
            ChartOverlay(column="ma_slow", label=f"MA{slow}", color="#60a5fa"),
        ]

    def result_metadata(self, variant: StrategyVariant) -> Dict[str, Any]:
        fast = int(variant.params.get("fast", 0))
        slow = int(variant.params.get("slow", 0))
        return {
            "variant_params": dict(variant.params),
            "ma": {"fast": fast, "slow": slow},
        }
