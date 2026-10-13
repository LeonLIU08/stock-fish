"""一个账户的空会话。

第一阶段只把评估窗口标成现金。入场、止损和成交留到后面的阶段。
"""
from __future__ import annotations

import pandas as pd

from analysis.backtest.engine import EngineResult

_COLUMNS = [
    "datetime",
    "equity",
    "equity_gross",
    "cash",
    "shares",
    "price",
    "position",
    "turnover",
    "leverage",
    "fee",
]


def run_empty_session(bars: pd.DataFrame, capital: float) -> EngineResult:
    """评估窗口内保持空仓。暖机不在这里出现，也不留下待执行单。"""
    if capital <= 0:
        raise ValueError("capital 必须 > 0")
    if bars is None or bars.empty:
        return EngineResult(
            equity=pd.DataFrame(columns=_COLUMNS),
            final_cash=float(capital),
        )

    rows = []
    last_price = 0.0
    for _, bar in bars.iterrows():
        price = float(bar["close"])
        last_price = price
        rows.append({
            "datetime": bar["datetime"],
            "equity": float(capital),
            "equity_gross": float(capital),
            "cash": float(capital),
            "shares": 0,
            "price": price,
            "position": 0,
            "turnover": 0.0,
            "leverage": 0.0,
            "fee": 0.0,
        })
    return EngineResult(
        equity=pd.DataFrame(rows),
        final_cash=float(capital),
        final_shares=0,
        final_price=last_price,
        cumulative_fees=0.0,
    )
