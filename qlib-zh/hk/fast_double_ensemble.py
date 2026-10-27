"""Double Ensemble 的 CPU 实现。

Qlib 自带的特征筛选对每一列做一次 pandas `.loc` 赋值。训练集是多层索引，
赋值会整表对齐，32 个核里只有主线程在忙。这里改成 numpy 上打乱一列，
并用线程池并行各列的预测。随机数仍在主线程按原来的顺序取出，打乱结果和
抽样顺序与单线程实现一致。
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from qlib.contrib.model.double_ensemble import DEnsembleModel as _QlibDEnsembleModel

_BUFFER_BUDGET_BYTES = 6 * 1024**3


class DEnsembleModel(_QlibDEnsembleModel):
    def fit(self, dataset):
        self._limit_train_threads()
        return super().fit(dataset)

    def _limit_train_threads(self) -> None:
        """多折同时训练时，每折少用几个 LightGBM 线程，避免互相抢核。"""
        fold_workers = max(1, int(os.environ.get("QLIB_FOLD_WORKERS", "1")))
        if fold_workers <= 1:
            return
        cpus = os.cpu_count() or 1
        self.params["num_threads"] = max(1, cpus // fold_workers)
        self.logger.info(
            "同时训练 %d 折，本折 LightGBM 使用 %d 个线程",
            fold_workers,
            self.params["num_threads"],
        )

    def _worker_count(self, nbytes: int, n_items: int) -> int:
        explicit = os.environ.get("QLIB_FS_THREADS")
        if explicit:
            return max(1, min(n_items, int(explicit)))
        fold_workers = max(1, int(os.environ.get("QLIB_FOLD_WORKERS", "1")))
        cpus = os.cpu_count() or 1
        by_cpu = max(1, cpus // fold_workers)
        by_mem = max(1, _BUFFER_BUDGET_BYTES // max(int(nbytes), 1))
        return max(1, min(n_items, by_cpu, by_mem))

    def retrieve_loss_curve(self, model, df_train, features):
        if self.base_model != "gbm":
            raise ValueError("not implemented yet")
        x_train = df_train["feature"].loc[:, features]
        y_train = np.squeeze(df_train["label"].to_numpy())
        x = np.ascontiguousarray(x_train.to_numpy())
        num_trees = int(model.num_trees())
        # 按树预测会改 booster 的内部状态，不能并行。28 棵树相对特征筛选很短。
        pred = np.zeros(x.shape[0], dtype=np.float64)
        curve = np.empty((x.shape[0], num_trees), dtype=np.float64)
        for i_tree in range(num_trees):
            pred += model.predict(x, start_iteration=i_tree, num_iteration=1, num_threads=1)
            curve[:, i_tree] = self.get_loss(y_train, pred)
        return pd.DataFrame(curve)

    def feature_selection(self, df_train, loss_values):
        x_train, y_train = df_train["feature"], df_train["label"]
        features = x_train.columns
        values = np.ascontiguousarray(x_train.to_numpy())
        y = np.squeeze(y_train.to_numpy())
        loss = np.asarray(loss_values, dtype=np.float64).reshape(-1)
        n_rows, n_features = values.shape
        # 先按原顺序取出全部打乱索引，后面并行预测不再碰随机数。
        permutations = [np.random.permutation(n_rows) for _ in range(n_features)]
        positions = {name: i for i, name in enumerate(features)}
        sub_cols = [
            np.fromiter((positions[name] for name in feats), dtype=np.intp, count=len(feats))
            for feats in self.sub_features
        ]
        workers = self._worker_count(values.nbytes, n_features)
        self.logger.info(
            "特征筛选 %d 列，矩阵 %s %s，%d 个线程",
            n_features,
            values.shape,
            values.dtype,
            workers,
        )
        g_values = self._g_values(values, y, loss, permutations, sub_cols, workers)

        g = pd.DataFrame({"g_value": g_values})
        g["g_value"] = g["g_value"].replace(np.nan, 0)
        g["bins"] = pd.cut(g["g_value"], self.bins_fs)
        res_feat = []
        sorted_bins = sorted(g["bins"].unique(), reverse=True)
        for i_b, b in enumerate(sorted_bins):
            b_feat = features[g["bins"] == b]
            num_feat = int(np.ceil(self.sample_ratios[i_b] * len(b_feat)))
            res_feat = res_feat + np.random.choice(b_feat, size=num_feat, replace=False).tolist()
        return pd.Index(set(res_feat))

    def _g_values(self, values, y, loss, permutations, sub_cols, workers: int) -> np.ndarray:
        n_features = values.shape[1]
        if workers == 1:
            scores = self._score_chunk(values, y, loss, permutations, sub_cols, range(n_features))
        else:
            chunks = [[] for _ in range(workers)]
            for i_f in range(n_features):
                chunks[i_f % workers].append(i_f)
            chunks = [chunk for chunk in chunks if chunk]
            with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
                parts = pool.map(
                    lambda chunk: self._score_chunk(values, y, loss, permutations, sub_cols, chunk),
                    chunks,
                )
            scores = [item for part in parts for item in part]
        g_values = np.zeros(n_features, dtype=np.float64)
        for i_f, score in scores:
            g_values[i_f] = score
        return g_values

    def _score_chunk(self, values, y, loss, permutations, sub_cols, columns) -> list[tuple[int, float]]:
        buf = np.array(values, copy=True, order="C")
        models = self.ensemble
        weight = 1.0 / len(models)
        out = []
        for i_f in columns:
            column = buf[:, i_f].copy()
            buf[:, i_f] = column[permutations[i_f]]
            pred = np.zeros(buf.shape[0], dtype=np.float64)
            for i_s, model in enumerate(models):
                cols = sub_cols[i_s]
                if cols.size == buf.shape[1] and np.all(cols == np.arange(cols.size)):
                    data = buf
                else:
                    data = np.ascontiguousarray(buf[:, cols])
                pred += model.predict(data, num_threads=1) * weight
            diff = self.get_loss(y, pred) - loss
            out.append((i_f, float(np.mean(diff) / (np.std(diff) + 1e-7))))
            buf[:, i_f] = column
        return out
