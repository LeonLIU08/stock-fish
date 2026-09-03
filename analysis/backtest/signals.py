"""Signal contract between strategies and the simulation engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

SIGNAL_ACTION_COL = "signal_action"
SIGNAL_SIZE_COL = "signal_size"
SIGNAL_REASON_COL = "signal_reason"


class SignalAction(str, Enum):
    HOLD = "hold"
    BUY = "buy"
    SELL = "sell"

    @classmethod
    def normalize(cls, value: Any) -> "SignalAction":
        if isinstance(value, SignalAction):
            return value
        text = str(value or "hold").strip().lower()
        for item in cls:
            if item.value == text:
                return item
        raise ValueError(f"未知信号动作: {value!r}，可选 {[a.value for a in cls]}")


@dataclass
class PreparedBars:
    """Strategy output consumed by the engine and reporting layer."""

    bars: pd.DataFrame
    overlay_columns: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChartOverlay:
    column: str
    label: str
    color: str = "#fbbf24"


def attach_signals(
    df: pd.DataFrame,
    actions: Sequence[Union[SignalAction, str]],
    *,
    reasons: Optional[Sequence[str]] = None,
    sizes: Optional[Sequence[Optional[float]]] = None,
) -> pd.DataFrame:
    """Write standard signal columns onto a bar table."""
    out = df.copy()
    normalized = [SignalAction.normalize(a).value for a in actions]
    if len(normalized) != len(out):
        raise ValueError(f"信号长度 {len(normalized)} 与 K 线行数 {len(out)} 不一致")
    out[SIGNAL_ACTION_COL] = normalized
    if reasons is not None:
        if len(reasons) != len(out):
            raise ValueError(f"reason 长度 {len(reasons)} 与 K 线行数 {len(out)} 不一致")
        out[SIGNAL_REASON_COL] = list(reasons)
    else:
        out[SIGNAL_REASON_COL] = ""
    if sizes is not None:
        if len(sizes) != len(out):
            raise ValueError(f"size 长度 {len(sizes)} 与 K 线行数 {len(out)} 不一致")
        out[SIGNAL_SIZE_COL] = [np.nan if s is None else float(s) for s in sizes]
    return out


def ensure_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee signal columns exist with safe defaults."""
    out = df.copy()
    if SIGNAL_ACTION_COL not in out.columns:
        out[SIGNAL_ACTION_COL] = SignalAction.HOLD.value
    else:
        out[SIGNAL_ACTION_COL] = out[SIGNAL_ACTION_COL].map(lambda v: SignalAction.normalize(v).value)
    if SIGNAL_REASON_COL not in out.columns:
        out[SIGNAL_REASON_COL] = ""
    if SIGNAL_SIZE_COL not in out.columns:
        out[SIGNAL_SIZE_COL] = np.nan
    return out


def validate_signal_columns(df: pd.DataFrame) -> None:
    missing = [col for col in (SIGNAL_ACTION_COL,) if col not in df.columns]
    if missing:
        raise ValueError(f"策略输出缺少信号列: {missing}")
