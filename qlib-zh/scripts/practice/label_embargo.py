"""从标签表达式推导决策时还没有成熟的未来交易日数。"""

from __future__ import annotations

import re

ROBUST_ALPHA158_LABEL = "Ref($close, -2)/Ref($close, -1) - 1"
_REF_FUTURE = re.compile(r"Ref\([^,]+,\s*-(\d+)\s*\)")


def label_embargo_days(expression: str) -> int:
    """返回表达式里 Ref(field, -N) 的最大 N。

    Ref($close, -2) 表示向后 2 个交易日，因此决策日 T 能用的最新标签
    样本是 T 往前第 2 个交易日。
    """
    if not expression or not str(expression).strip():
        raise ValueError("label expression is empty")
    horizons = [int(match) for match in _REF_FUTURE.findall(str(expression))]
    if not horizons:
        raise ValueError(f"label expression has no future Ref horizon: {expression}")
    embargo = max(horizons)
    if embargo < 1:
        raise ValueError("label embargo must be at least 1 trading day")
    return embargo


def label_expression_for_model_mode(model_mode: str) -> str:
    mode = (model_mode or "robust").strip().lower()
    if mode == "multihorizon":
        from multi_horizon_label_handler import _MULTI_HORIZON_LABEL_EXPRS

        return " ; ".join(_MULTI_HORIZON_LABEL_EXPRS)
    return ROBUST_ALPHA158_LABEL
