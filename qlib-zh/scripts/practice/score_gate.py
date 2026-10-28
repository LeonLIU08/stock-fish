"""把模型原始预测校准成成本后预期超额收益。

校准只用 inner purged OOF 样本。分箱均值经过保序回归，置信度由
样本量与标准误决定，不能用分数绝对值代替。
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROFILE_QUANTILES = {
    "conservative": 0.75,
    "balanced": 0.50,
    "aggressive": 0.25,
}
PROFILE_MIN_CONFIDENCE = {
    "conservative": 0.60,
    "balanced": 0.40,
    "aggressive": 0.20,
}
HYSTERESIS = 0.005


@dataclass
class BinMapping:
    edges: list[float]
    means: list[float]
    counts: list[int]
    stds: list[float]
    confidences: list[float]

    def predict(self, raw: float) -> tuple[float, float]:
        if not self.edges:
            return 0.0, 0.0
        index = int(np.searchsorted(self.edges, raw, side="right") - 1)
        index = min(max(index, 0), len(self.means) - 1)
        return float(self.means[index]), float(self.confidences[index])


def _pool_adjacent_violators(values: Sequence[float], weights: Sequence[float]) -> list[float]:
    """保序回归：相邻违反单调的分箱合并后再取加权均值。"""
    blocks: list[dict[str, float]] = []
    for value, weight in zip(values, weights):
        blocks.append({"sum": float(value) * float(weight), "weight": float(weight), "count": 1.0})
        while len(blocks) >= 2:
            left = blocks[-2]["sum"] / blocks[-2]["weight"]
            right = blocks[-1]["sum"] / blocks[-1]["weight"]
            if left <= right + 1e-15:
                break
            merged = {
                "sum": blocks[-2]["sum"] + blocks[-1]["sum"],
                "weight": blocks[-2]["weight"] + blocks[-1]["weight"],
                "count": blocks[-2]["count"] + blocks[-1]["count"],
            }
            blocks = blocks[:-2] + [merged]
    expanded: list[float] = []
    for block in blocks:
        mean = block["sum"] / block["weight"] if block["weight"] else 0.0
        expanded.extend([mean] * int(block["count"]))
    return expanded


def _confidence(count: int, std: float, mean: float) -> float:
    if count <= 0:
        return 0.0
    se = float(std) / np.sqrt(count) if count else float(std)
    sample_factor = float(count) / float(count + 5.0)
    scale = max(abs(float(mean)), 1e-4)
    noise_factor = 1.0 / (1.0 + se / scale)
    return float(np.clip(sample_factor * noise_factor, 0.0, 1.0))


def fit_binned_mapping(raw: Sequence[float], target: Sequence[float], n_bins: int = 10) -> BinMapping:
    raw_arr = np.asarray(raw, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    mask = np.isfinite(raw_arr) & np.isfinite(target_arr)
    raw_arr = raw_arr[mask]
    target_arr = target_arr[mask]
    if len(raw_arr) == 0:
        raise ValueError("no finite samples for score calibration")
    order = np.argsort(raw_arr, kind="mergesort")
    raw_arr = raw_arr[order]
    target_arr = target_arr[order]
    bins = min(int(n_bins), len(raw_arr))
    chunks = np.array_split(np.arange(len(raw_arr)), bins)
    edges: list[float] = []
    means: list[float] = []
    counts: list[int] = []
    stds: list[float] = []
    weights: list[float] = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        values = target_arr[chunk]
        edges.append(float(raw_arr[chunk[0]]))
        means.append(float(np.mean(values)))
        counts.append(int(len(chunk)))
        stds.append(float(np.std(values, ddof=1)) if len(chunk) > 1 else 0.0)
        weights.append(float(len(chunk)))
    monotone = _pool_adjacent_violators(means, weights)
    confidences = [
        _confidence(count, std, mean) for count, std, mean in zip(counts, stds, monotone)
    ]
    return BinMapping(edges=edges, means=monotone, counts=counts, stds=stds, confidences=confidences)


@dataclass
class ScoreCalibrator:
    as_of: str
    mappings: dict[str, BinMapping]
    ensemble_weights: dict[str, float]
    thresholds: dict[str, dict[str, float]]
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = ""

    def predict(self, raw_by_model: Mapping[str, float]) -> tuple[float, float]:
        expected = 0.0
        confidence = 0.0
        weight_sum = 0.0
        for name, weight in self.ensemble_weights.items():
            if name not in self.mappings or name not in raw_by_model:
                continue
            pred, conf = self.mappings[name].predict(float(raw_by_model[name]))
            expected += float(weight) * pred
            confidence += float(weight) * conf
            weight_sum += float(weight)
        if weight_sum <= 0:
            raise KeyError("no overlapping models between ensemble weights and raw predictions")
        return expected / weight_sum, confidence / weight_sum

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "as_of": self.as_of,
            "version": self.version,
            "ensemble_weights": self.ensemble_weights,
            "thresholds": self.thresholds,
            "metadata": self.metadata,
            "mappings": {
                name: {
                    "edges": mapping.edges,
                    "means": mapping.means,
                    "counts": mapping.counts,
                    "stds": mapping.stds,
                    "confidences": mapping.confidences,
                }
                for name, mapping in self.mappings.items()
            },
        }
        (path / "score_gate_calibrator.pkl").write_bytes(pickle.dumps(payload))
        (path / "score_gate_thresholds.json").write_text(
            json.dumps(
                {
                    "as_of": self.as_of,
                    "version": self.version,
                    "profiles": self.thresholds,
                    "metadata": self.metadata,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "ScoreCalibrator":
        path = Path(directory)
        payload = pickle.loads((path / "score_gate_calibrator.pkl").read_bytes())
        mappings = {name: BinMapping(**body) for name, body in payload["mappings"].items()}
        return cls(
            as_of=payload["as_of"],
            mappings=mappings,
            ensemble_weights=payload["ensemble_weights"],
            thresholds=payload["thresholds"],
            metadata=payload.get("metadata") or {},
            version=payload.get("version") or "",
        )


def _version_for(as_of: str, mappings: Mapping[str, BinMapping], thresholds: Mapping[str, Any]) -> str:
    canonical = {
        "as_of": as_of,
        "mappings": {
            name: {"edges": mapping.edges, "means": mapping.means}
            for name, mapping in sorted(mappings.items())
        },
        "thresholds": thresholds,
    }
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


def select_thresholds(
    net_excess: Sequence[float],
    *,
    safety_margin_bps: float = 0.0,
) -> dict[str, dict[str, float]]:
    """在 OOF 成本后超额上冻结三档门槛。分位数越高，入场越严格。"""
    values = np.asarray(list(net_excess), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("no OOF net excess values for threshold selection")
    output: dict[str, dict[str, float]] = {}
    for profile, quantile in PROFILE_QUANTILES.items():
        entry = float(np.quantile(values, quantile))
        exit_threshold = float(entry - HYSTERESIS)
        output[profile] = {
            "entry_threshold": entry,
            "exit_threshold": exit_threshold,
            "min_confidence": PROFILE_MIN_CONFIDENCE[profile],
            "safety_margin_bps": float(safety_margin_bps),
        }
    ordered = [output[name]["entry_threshold"] for name in ("aggressive", "balanced", "conservative")]
    if ordered != sorted(ordered):
        raise AssertionError("profile entry thresholds must be monotonic")
    return output


def fit_score_calibrator(
    oof: pd.DataFrame,
    *,
    as_of: str,
    ensemble_weights: Mapping[str, float],
    raw_column: str = "raw_prediction",
    model_column: str = "model",
    target_column: str = "future_excess",
    n_bins: int = 10,
    round_trip_cost: float = 0.0,
    safety_margin_bps: float = 0.0,
    metadata: Mapping[str, Any] | None = None,
) -> ScoreCalibrator:
    """用 OOF 行拟合每个模型的单调映射，并在同一批 OOF 上冻结门槛。"""
    if oof.empty:
        raise ValueError("OOF frame is empty")
    mappings: dict[str, BinMapping] = {}
    for name in ensemble_weights:
        part = oof[oof[model_column].astype(str) == str(name)]
        if part.empty:
            raise ValueError(f"OOF frame has no rows for model {name}")
        mappings[name] = fit_binned_mapping(part[raw_column], part[target_column], n_bins=n_bins)
    weights = {str(name): float(weight) for name, weight in ensemble_weights.items()}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("ensemble weights must sum to a positive number")
    weights = {name: weight / total for name, weight in weights.items()}
    net_values: list[float] = []
    safety = float(safety_margin_bps) / 10000.0
    grouped = oof.groupby(list(oof.columns.intersection(["row_id", "instrument", "datetime"])) or [model_column])
    # 逐行合成。没有行号时按索引分组会退化为单模型行，门槛仍由各行净超额决定。
    if {"instrument", "datetime"}.issubset(oof.columns):
        for (_, _), part in oof.groupby(["datetime", "instrument"], sort=False):
            raw_by_model = {
                str(row[model_column]): float(row[raw_column]) for _, row in part.iterrows()
            }
            try:
                expected, _ = ScoreCalibrator(as_of, mappings, weights, {}, version="").predict(raw_by_model)
            except KeyError:
                continue
            net_values.append(expected - float(round_trip_cost) - safety)
    else:
        calibrator_tmp = ScoreCalibrator(as_of, mappings, weights, {}, version="")
        for _, row in oof.iterrows():
            expected, _ = calibrator_tmp.predict({str(row[model_column]): float(row[raw_column])})
            net_values.append(expected - float(round_trip_cost) - safety)
    thresholds = select_thresholds(net_values, safety_margin_bps=safety_margin_bps)
    version = _version_for(as_of, mappings, thresholds)
    return ScoreCalibrator(
        as_of=as_of,
        mappings=mappings,
        ensemble_weights=weights,
        thresholds=thresholds,
        metadata=dict(metadata or {}),
        version=version,
    )


def expected_net_excess(
    expected_excess_return: float,
    round_trip_cost: float,
    safety_margin_bps: float,
) -> float:
    return float(expected_excess_return) - float(round_trip_cost) - float(safety_margin_bps) / 10000.0
