"""在滚动训练里生成按月样本外预测，并在每个预测日前冻结校准器。

最终模型仍由原来的滚动折训练，并且只预测该折的测试窗口。
样本外预测按月缓存，后面的预测日复用已经完成的月份。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from inner_oof import apply_calibrator, fit_asof_calibrator
from run_baseline import cache_dir, trade_costs


def policy_requests_calibrator(cfg: dict) -> bool:
    policy = cfg.get("portfolio_policy") or {}
    gate = policy.get("score_gate") or {}
    enabled = bool(policy.get("enabled")) or os.environ.get("PORTFOLIO_POLICY_ENABLED") == "1"
    mode = os.environ.get("SCORE_GATE_MODE") or gate.get("mode") or "calibrated_net_excess"
    return enabled and mode == "calibrated_net_excess"


def score_gate_profile(cfg: dict) -> str:
    gate = (cfg.get("portfolio_policy") or {}).get("score_gate") or {}
    return os.environ.get("SCORE_GATE_PROFILE") or gate.get("profile") or "balanced"


def calibration_years(cfg: dict) -> float:
    rolling = (cfg.get("portfolio_policy") or {}).get("rolling") or {}
    raw = os.environ.get("CALIBRATION_WINDOW_YEARS") or rolling.get("calibration_window_years") or 1
    return float(raw)


def _embargo(cfg: dict) -> int:
    from label_embargo import label_embargo_days

    expression = str(cfg["label"]["expr"])
    parsed = label_embargo_days(expression)
    configured = int(cfg["label"].get("horizon_trading_days") or parsed)
    if configured != parsed:
        raise ValueError(
            f"label.horizon_trading_days={configured} 和标签表达式推导出的 embargo={parsed} 不一致"
        )
    return parsed


def _oof_dir(cfg: dict, model_name: str) -> Path:
    root = cache_dir(str(cfg["data"]["freeze_dirname"]), cfg) / "score_gate_oof" / model_name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _month_slices(calendar: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp):
    days = pd.DatetimeIndex(calendar)
    days = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
    if len(days) == 0:
        return
    grouped = pd.Series(days).groupby(pd.Series(days).dt.to_period("M"))
    for _period, group in grouped:
        yield pd.Timestamp(group.iloc[0]), pd.Timestamp(group.iloc[-1])


def _label_frame(cfg: dict, start: str, end: str) -> pd.DataFrame:
    from panel import load_panel

    frame = load_panel(cfg, start, end)
    column = ("label", str(cfg["label"]["column"]))
    if column not in frame.columns:
        raise KeyError(f"Alpha158 缓存没有标签列 {column}")
    label = frame[column].rename("label").reset_index()
    label["datetime"] = pd.to_datetime(label["datetime"])
    label["instrument"] = label["instrument"].astype(str)
    label["future_excess"] = label["label"] - label.groupby("datetime")["label"].transform("mean")
    return label.dropna(subset=["future_excess"])


def _train_month(
    cfg: dict,
    spec: dict,
    calendar: pd.DatetimeIndex,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    model_name: str,
) -> pd.DataFrame:
    from rolling_schedule import shift_trading_days

    from run_rolling import _fit_and_predict

    embargo = _embargo(cfg)
    label_end = shift_trading_days(calendar, month_start, -embargo)
    train_start = pd.Timestamp(cfg["segments"]["train"]["start"])
    if label_end <= train_start:
        return pd.DataFrame()
    early_days = int(cfg["rolling"]["early_stop_days"])
    valid_start = shift_trading_days(calendar, label_end, -(early_days - 1))
    train_end = shift_trading_days(calendar, valid_start, -1)
    if train_end <= train_start:
        return pd.DataFrame()
    task = {
        "model": {
            "class": spec["class"],
            "module_path": spec["module_path"],
            "kwargs": spec.get("kwargs") or {},
        },
        "dataset": {
            "kwargs": {
                "segments": {
                    "train": (train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")),
                    "valid": (valid_start.strftime("%Y-%m-%d"), label_end.strftime("%Y-%m-%d")),
                    "test": (month_start.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")),
                }
            }
        },
    }
    print(
        f"[{model_name}] 样本外 {month_start.date()} ~ {month_end.date()}，"
        f"训练标签截至 {label_end.date()}",
        flush=True,
    )
    prediction, _curves = _fit_and_predict(task, spec, cfg)
    labels = _label_frame(cfg, month_start.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d"))
    scored = prediction.rename("raw_prediction").reset_index()
    scored["datetime"] = pd.to_datetime(scored["datetime"])
    scored["instrument"] = scored["instrument"].astype(str)
    merged = scored.merge(labels[["datetime", "instrument", "future_excess"]], on=["datetime", "instrument"], how="inner")
    merged = merged[merged["datetime"] > label_end]
    merged["model"] = model_name
    merged["model_train_label_end"] = label_end
    return merged[["datetime", "instrument", "model", "raw_prediction", "future_excess", "model_train_label_end"]]


def _coverage_path(path: Path) -> Path:
    return path.with_suffix(".coverage.json")


def _month_target_end(
    calendar: pd.DatetimeIndex,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
) -> pd.Timestamp:
    """一个缓存文件覆盖从 month_start 到该月最后一个交易日。"""
    start = pd.Timestamp(month_start)
    requested = pd.Timestamp(month_end)
    days = pd.DatetimeIndex(calendar)
    in_month = days[(days >= start) & (days.to_period("M") == start.to_period("M"))]
    if len(in_month) == 0:
        return requested
    return max(requested, pd.Timestamp(in_month[-1]))


def _cached_coverage_end(path: Path) -> pd.Timestamp | None:
    """已写过的预测终点。旧文件没有旁路记录时，用表里的最后一天。"""
    meta = _coverage_path(path)
    if meta.exists():
        payload = json.loads(meta.read_text(encoding="utf-8"))
        if payload.get("end"):
            return pd.Timestamp(payload["end"])
    if not path.exists() or path.stat().st_size <= 0:
        return None
    frame = pd.read_parquet(path)
    if frame.empty or "datetime" not in frame.columns:
        return None
    return pd.Timestamp(frame["datetime"].max())


def _write_oof(path: Path, frame: pd.DataFrame, covered_end: pd.Timestamp) -> None:
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    meta = _coverage_path(path)
    meta_tmp = meta.with_suffix(".tmp")
    meta_tmp.write_text(
        json.dumps({"end": pd.Timestamp(covered_end).strftime("%Y-%m-%d")}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    meta_tmp.replace(meta)


def ensure_oof_month(
    cfg: dict,
    spec: dict,
    calendar: pd.DatetimeIndex,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    model_name: str,
) -> pd.DataFrame:
    """按月缓存样本外预测。第一次写入就覆盖到该月最后一个交易日。

    缓存名只用月份起点。窗口刚碰到这个月时，请求终点往往只是第一天；
    若把这天当成整月，后面的预测日会一直缺样本。
    """
    import fcntl

    path = _oof_dir(cfg, model_name) / f"{month_start.strftime('%Y-%m-%d')}.parquet"
    target = _month_target_end(calendar, month_start, month_end)
    lock_path = path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        covered = _cached_coverage_end(path)
        if covered is not None and covered >= target:
            return pd.read_parquet(path)
        frame = _train_month(cfg, spec, calendar, month_start, target, model_name)
        _write_oof(path, frame, target)
        return frame


def load_oof_for_asof(
    cfg: dict,
    spec: dict,
    calendar: pd.DatetimeIndex,
    as_of: str,
    model_name: str,
) -> pd.DataFrame:
    from inner_oof import train_label_end

    embargo = _embargo(cfg)
    label_end = train_label_end(calendar, as_of, embargo)
    window_start = label_end - pd.DateOffset(years=calibration_years(cfg))
    parts = []
    for month_start, month_end in _month_slices(calendar, window_start, label_end):
        part = ensure_oof_month(cfg, spec, calendar, month_start, month_end, model_name)
        if not part.empty:
            parts.append(part)
    if not parts:
        raise RuntimeError(f"{as_of} 之前没有可用的样本外月份")
    return pd.concat(parts, ignore_index=True)


def round_trip_cost(cfg: dict) -> float:
    buy, sell = trade_costs(cfg["backtest"]["costs"])
    return float(buy) + float(sell)


def freeze_asof_calibrator(
    cfg: dict,
    spec: dict,
    calendar: pd.DatetimeIndex,
    as_of: str,
    out_dir: Path,
    model_name: str,
):
    """先完成样本外预测和校准器，再允许最终模型重训。"""
    oof = load_oof_for_asof(cfg, spec, calendar, as_of, model_name)
    artifact_dir = out_dir / "rolling_models" / pd.Timestamp(as_of).strftime("%Y%m%d")
    calibrator = fit_asof_calibrator(
        oof,
        calendar=calendar,
        as_of=as_of,
        embargo=_embargo(cfg),
        calibration_years=calibration_years(cfg),
        ensemble_weights={model_name: 1.0},
        out_dir=artifact_dir,
        round_trip_cost=round_trip_cost(cfg),
        metadata={"model": model_name, "final_refit": True, "retrained": True},
    )
    metadata = dict(calibrator.metadata)
    metadata["checkpoint_as_of"] = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    metadata["retrained"] = True
    (artifact_dir / "rolling_asof_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return calibrator


def prepare_fold_gate(
    cfg: dict,
    spec: dict,
    calendar: pd.DatetimeIndex,
    as_of: str,
    raw_prediction: pd.Series,
    out_dir: Path,
    model_name: str,
) -> pd.DataFrame:
    """拟合该预测日的校准器，并映射这折模型的原始预测。"""
    calibrator = freeze_asof_calibrator(cfg, spec, calendar, as_of, out_dir, model_name)
    return apply_calibrator(
        raw_prediction,
        calibrator,
        model_name,
        score_gate_profile(cfg),
        round_trip_cost(cfg),
    )
