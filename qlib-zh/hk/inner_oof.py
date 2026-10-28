"""用已经生成的 inner OOF 预测拟合并应用分数校准器。

这里不训练模型。训练发生在滚动任务里；本模块只检查样本外边界、
拟合最近一个校准窗口，并把原始预测变成门槛所需的预期超额和置信度。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import pandas as pd

_PRACTICE = Path(__file__).resolve().parents[1] / "scripts" / "practice"
if str(_PRACTICE) not in sys.path:
    sys.path.insert(0, str(_PRACTICE))

from rolling_schedule import shift_trading_days
from score_gate import ScoreCalibrator, fit_score_calibrator


def train_label_end(calendar: pd.DatetimeIndex, as_of: str, embargo: int) -> pd.Timestamp:
    return shift_trading_days(pd.DatetimeIndex(calendar), pd.Timestamp(as_of), -int(embargo))


def slice_calibration_rows(
    oof: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    as_of: str,
    embargo: int,
    calibration_years: float,
) -> pd.DataFrame:
    """只保留决策日已经成熟、并且落在最近校准窗口里的样本外行。"""
    if oof.empty:
        raise ValueError("OOF 预测是空的，无法拟合校准器")
    required = {"datetime", "model_train_label_end", "raw_prediction", "future_excess", "model"}
    missing = required - set(oof.columns)
    if missing:
        raise ValueError(f"OOF 预测缺少列: {sorted(missing)}")
    label_end = train_label_end(calendar, as_of, embargo)
    window_start = label_end - pd.DateOffset(years=float(calibration_years))
    frame = oof.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["model_train_label_end"] = pd.to_datetime(frame["model_train_label_end"])
    leaked = frame[frame["datetime"] <= frame["model_train_label_end"]]
    if not leaked.empty:
        sample = leaked.head(3)[["datetime", "model_train_label_end"]].to_dict(orient="records")
        raise ValueError(f"OOF 行落在对应模型的训练标签里: {sample}")
    chosen = frame[(frame["datetime"] >= window_start) & (frame["datetime"] <= label_end)]
    if chosen.empty:
        raise ValueError(
            f"{as_of} 的校准窗口 {window_start.date()} ~ {label_end.date()} 没有样本外预测"
        )
    return chosen


def fit_asof_calibrator(
    oof: pd.DataFrame,
    *,
    calendar: pd.DatetimeIndex,
    as_of: str,
    embargo: int,
    calibration_years: float,
    ensemble_weights: Mapping[str, float],
    out_dir: str | Path,
    round_trip_cost: float = 0.0,
    safety_margin_bps: float = 0.0,
    metadata: dict | None = None,
) -> ScoreCalibrator:
    chosen = slice_calibration_rows(oof, calendar, as_of, embargo, calibration_years)
    label_end = train_label_end(calendar, as_of, embargo)
    extra = dict(metadata or {})
    extra.update(
        {
            "prediction_date": pd.Timestamp(as_of).strftime("%Y-%m-%d"),
            "train_label_end": label_end.strftime("%Y-%m-%d"),
            "label_embargo_days": int(embargo),
            "calibration_start": pd.Timestamp(chosen["datetime"].min()).strftime("%Y-%m-%d"),
            "calibration_end": pd.Timestamp(chosen["datetime"].max()).strftime("%Y-%m-%d"),
            "round_trip_cost": float(round_trip_cost),
            "rows": int(len(chosen)),
        }
    )
    calibrator = fit_score_calibrator(
        chosen,
        as_of=pd.Timestamp(as_of).strftime("%Y-%m-%d"),
        ensemble_weights=ensemble_weights,
        round_trip_cost=round_trip_cost,
        safety_margin_bps=safety_margin_bps,
        metadata=extra,
    )
    calibrator.save(out_dir)
    return calibrator


def apply_calibrator(
    raw_prediction: pd.Series,
    calibrator: ScoreCalibrator,
    model_name: str,
    profile: str,
    round_trip_cost: float,
) -> pd.DataFrame:
    """把某一日模型的原始预测映射成校准门槛使用的列。"""
    if profile not in calibrator.thresholds:
        raise KeyError(f"校准器没有门槛档位 {profile}")
    thresholds = calibrator.thresholds[profile]
    rows: list[dict] = []
    series = raw_prediction.astype(float)
    for key, value in series.items():
        if not pd.notna(value):
            continue
        expected, confidence = calibrator.predict({model_name: float(value)})
        if isinstance(key, tuple):
            datetime, instrument = key
        else:
            datetime, instrument = None, key
        rows.append(
            {
                "datetime": pd.Timestamp(datetime),
                "instrument": str(instrument),
                "raw_prediction": float(value),
                "expected_excess_return": float(expected),
                "gate_confidence": float(confidence),
                "entry_threshold": float(thresholds["entry_threshold"]),
                "exit_threshold": float(thresholds["exit_threshold"]),
                "min_confidence": float(thresholds["min_confidence"]),
                "safety_margin_bps": float(thresholds.get("safety_margin_bps", 0.0)),
                "round_trip_cost": float(round_trip_cost),
                "artifact_version": calibrator.version,
                "score_gate_profile": profile,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.set_index(["datetime", "instrument"]).sort_index()
