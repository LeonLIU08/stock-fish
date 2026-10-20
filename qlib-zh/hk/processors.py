"""训练样本在截面标准化之后，丢掉标签里的 NaN 和无穷大。"""

from __future__ import annotations

import numpy as np
from qlib.data.dataset.processor import Processor, get_group_columns


class DropInvalidLabel(Processor):
    """只用于学习集。标签非有限值的样本不参与拟合。特征里的无穷大改成缺失。"""

    def is_for_infer(self) -> bool:
        return False

    def __call__(self, df):
        label_cols = get_group_columns(df, "label")
        feature_cols = get_group_columns(df, "feature")
        if len(feature_cols):
            df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        df[label_cols] = df[label_cols].replace([np.inf, -np.inf], np.nan)
        return df.dropna(subset=label_cols)


class SanitizeFeature(Processor):
    """推理时把特征里的无穷大改成缺失，避免预测阶段再碰到过大的数。"""

    def __call__(self, df):
        feature_cols = get_group_columns(df, "feature")
        if len(feature_cols):
            df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        return df
