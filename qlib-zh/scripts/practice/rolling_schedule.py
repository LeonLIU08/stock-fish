"""严格单日 outer walk-forward 的日期边界。

每个预测日 T 只预测 T。训练样本停在标签 embargo 之前，
最终模型会重新使用这段已成熟标签；校准只用更早模型产生的 inner OOF。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd

from label_embargo import label_embargo_days, label_expression_for_model_mode


class StaleCheckpointError(RuntimeError):
    pass


class LeakageError(RuntimeError):
    pass


@dataclass
class InnerOofFold:
    test_start: str
    test_end: str
    train_start: str
    train_label_end: str

    def covers(self, row_date: pd.Timestamp) -> bool:
        return pd.Timestamp(self.test_start) <= pd.Timestamp(row_date) <= pd.Timestamp(self.test_end)


@dataclass
class DailyFoldPlan:
    prediction_date: str
    execution_date: str
    signal_date: str
    pred_date: str
    test_start: str
    test_end: str
    oos_end: str
    train_start: str
    train_end: str
    train_label_end: str
    valid_start: str
    valid_end: str
    calibration_start: str
    calibration_end: str
    label_embargo_days: int
    checkpoint_as_of: str
    final_refit: bool = True
    allow_valid_inside_train: bool = True
    retrained: bool = True
    inner_oof: list[InnerOofFold] = field(default_factory=list)

    def as_fold_dict(self) -> dict[str, str]:
        payload = {
            "signal_date": self.signal_date,
            "pred_date": self.pred_date,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "oos_end": self.oos_end,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "train_label_end": self.train_label_end,
            "valid_start": self.valid_start,
            "valid_end": self.valid_end,
            "calibration_start": self.calibration_start,
            "calibration_end": self.calibration_end,
            "label_embargo_days": str(self.label_embargo_days),
            "checkpoint_as_of": self.checkpoint_as_of,
            "final_refit": "true" if self.final_refit else "false",
            "allow_valid_inside_train": "1" if self.allow_valid_inside_train else "0",
        }
        return payload


def _fmt(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _index_on_or_before(calendar: pd.DatetimeIndex, target: pd.Timestamp) -> int:
    position = int(calendar.searchsorted(pd.Timestamp(target), side="right")) - 1
    if position < 0:
        raise IndexError(f"{target.date()} is before the calendar")
    return position


def shift_trading_days(calendar: pd.DatetimeIndex, date: pd.Timestamp, offset: int) -> pd.Timestamp:
    position = _index_on_or_before(calendar, pd.Timestamp(date))
    moved = position + int(offset)
    if moved < 0 or moved >= len(calendar):
        raise IndexError(f"cannot shift {date.date()} by {offset} trading days")
    return pd.Timestamp(calendar[moved])


def latest_labeled_sample(calendar: pd.DatetimeIndex, prediction_date: pd.Timestamp, embargo: int) -> pd.Timestamp:
    return shift_trading_days(calendar, prediction_date, -int(embargo))


def _month_starts(dates: Sequence[pd.Timestamp]) -> list[pd.Timestamp]:
    seen: set[tuple[int, int]] = set()
    starts: list[pd.Timestamp] = []
    for value in dates:
        stamp = pd.Timestamp(value)
        key = (stamp.year, stamp.month)
        if key in seen:
            continue
        seen.add(key)
        starts.append(stamp)
    return starts


def _align_start(calendar: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp:
    position = int(calendar.searchsorted(pd.Timestamp(target), side="left"))
    if position >= len(calendar):
        raise IndexError(f"{target.date()} is after the calendar")
    return pd.Timestamp(calendar[position])


def build_inner_oof_folds(
    calendar: pd.DatetimeIndex,
    train_start: pd.Timestamp,
    train_label_end: pd.Timestamp,
    embargo: int,
    step: str = "month",
) -> list[InnerOofFold]:
    if step != "month":
        raise ValueError("inner_oof_step currently supports month")
    window = [pd.Timestamp(day) for day in calendar if train_start <= pd.Timestamp(day) <= train_label_end]
    if not window:
        return []
    folds: list[InnerOofFold] = []
    anchors = _month_starts(window)
    for index, start in enumerate(anchors):
        end = anchors[index + 1] - pd.Timedelta(days=1) if index + 1 < len(anchors) else pd.Timestamp(train_label_end)
        end_pos = _index_on_or_before(calendar, end)
        end_ts = pd.Timestamp(calendar[end_pos])
        if end_ts < start:
            continue
        try:
            model_label_end = shift_trading_days(calendar, start, -int(embargo))
        except IndexError:
            continue
        if model_label_end < train_start:
            continue
        folds.append(
            InnerOofFold(
                test_start=_fmt(start),
                test_end=_fmt(min(end_ts, pd.Timestamp(train_label_end))),
                train_start=_fmt(train_start),
                train_label_end=_fmt(model_label_end),
            )
        )
    return folds


def build_daily_plan(
    calendar: pd.DatetimeIndex,
    prediction_date: str,
    *,
    label_expression: str,
    train_window_years: float = 5,
    calibration_window_years: float = 1,
    inner_oof_step: str = "month",
    valid_trading_days: int = 20,
) -> DailyFoldPlan:
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values().unique()
    embargo = label_embargo_days(label_expression)
    prediction = pd.Timestamp(prediction_date)
    prediction_pos = _index_on_or_before(calendar, prediction)
    if pd.Timestamp(calendar[prediction_pos]) != prediction.normalize():
        raise IndexError(f"{prediction_date} is not a trading day")
    label_end = latest_labeled_sample(calendar, prediction, embargo)
    if label_end > prediction:
        raise LeakageError("train_label_end is after the prediction date")
    train_start_target = label_end - pd.DateOffset(years=float(train_window_years))
    train_start = _align_start(calendar, train_start_target)
    if train_start > label_end:
        raise ValueError("training window is empty")
    calibration_target = label_end - pd.DateOffset(years=float(calibration_window_years))
    calibration_start = _align_start(calendar, calibration_target)
    if calibration_start < train_start:
        calibration_start = train_start
    label_pos = _index_on_or_before(calendar, label_end)
    valid_start_pos = max(int(calendar.searchsorted(train_start, side="left")), label_pos - valid_trading_days + 1)
    valid_start = pd.Timestamp(calendar[valid_start_pos])
    execution = pd.Timestamp(calendar[prediction_pos + 1]) if prediction_pos + 1 < len(calendar) else prediction
    inner = build_inner_oof_folds(calendar, train_start, label_end, embargo, step=inner_oof_step)
    plan = DailyFoldPlan(
        prediction_date=_fmt(prediction),
        execution_date=_fmt(execution),
        signal_date=_fmt(prediction),
        pred_date=_fmt(prediction),
        test_start=_fmt(prediction),
        test_end=_fmt(prediction),
        oos_end=_fmt(prediction),
        train_start=_fmt(train_start),
        train_end=_fmt(label_end),
        train_label_end=_fmt(label_end),
        valid_start=_fmt(valid_start),
        valid_end=_fmt(label_end),
        calibration_start=_fmt(calibration_start),
        calibration_end=_fmt(label_end),
        label_embargo_days=embargo,
        checkpoint_as_of=_fmt(prediction),
        inner_oof=inner,
    )
    assert_plan_no_leakage(plan, calendar)
    return plan


def assert_plan_no_leakage(plan: DailyFoldPlan, calendar: pd.DatetimeIndex) -> None:
    prediction = pd.Timestamp(plan.prediction_date)
    label_end = pd.Timestamp(plan.train_label_end)
    allowed = latest_labeled_sample(calendar, prediction, plan.label_embargo_days)
    if label_end > allowed:
        raise LeakageError(
            f"train_label_end {plan.train_label_end} is after embargo boundary { _fmt(allowed) }"
        )
    if pd.Timestamp(plan.train_end) != label_end:
        raise LeakageError("final refit train_end must equal train_label_end")
    if plan.test_start != plan.test_end or plan.test_start != plan.prediction_date:
        raise LeakageError("strict daily outer test must contain only the prediction date")
    if pd.Timestamp(plan.valid_end) > label_end or pd.Timestamp(plan.valid_start) < pd.Timestamp(plan.train_start):
        raise LeakageError("validation window must stay inside the mature training window")
    if pd.Timestamp(plan.calibration_end) > label_end:
        raise LeakageError("calibration window ends after mature labels")
    for fold in plan.inner_oof:
        if pd.Timestamp(fold.train_label_end) >= pd.Timestamp(fold.test_start):
            raise LeakageError(
                f"inner model label end {fold.train_label_end} overlaps OOF start {fold.test_start}"
            )


def assert_oof_rows_not_in_training_labels(oof_rows: pd.DataFrame) -> None:
    if oof_rows.empty:
        return
    required = {"row_date", "model_train_label_end"}
    missing = required - set(oof_rows.columns)
    if missing:
        raise ValueError(f"OOF leakage check missing columns: {sorted(missing)}")
    row_dates = pd.to_datetime(oof_rows["row_date"])
    train_ends = pd.to_datetime(oof_rows["model_train_label_end"])
    leaked = oof_rows.loc[row_dates <= train_ends]
    if not leaked.empty:
        sample = leaked.head(3).to_dict(orient="records")
        raise LeakageError(f"OOF rows were visible to their model labels: {sample}")


def build_strict_daily_folds(
    calendar: pd.DatetimeIndex,
    *,
    start_date: str,
    end_date: str,
    label_expression: str | None = None,
    model_mode: str = "robust",
    train_window_years: float = 5,
    calibration_window_years: float = 1,
    inner_oof_step: str = "month",
) -> list[dict[str, str]]:
    expression = label_expression or label_expression_for_model_mode(model_mode)
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values().unique()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    embargo = label_embargo_days(expression)
    folds: list[dict[str, str]] = []
    for day in calendar:
        stamp = pd.Timestamp(day)
        if stamp < start or stamp > end:
            continue
        position = _index_on_or_before(calendar, stamp)
        if position - embargo < 0:
            continue
        try:
            plan = build_daily_plan(
                calendar,
                _fmt(stamp),
                label_expression=expression,
                train_window_years=train_window_years,
                calibration_window_years=calibration_window_years,
                inner_oof_step=inner_oof_step,
            )
        except (IndexError, ValueError):
            continue
        folds.append(plan.as_fold_dict())
    return folds


def must_run_final_refit(
    *,
    feature_cache_hit: bool = False,
    oof_cache_hit: bool = False,
    checkpoint_as_of: str | None = None,
    prediction_date: str | None = None,
) -> bool:
    """特征或 OOF 缓存命中都不能代替当日最终重训。"""
    del feature_cache_hit, oof_cache_hit, checkpoint_as_of, prediction_date
    return True


def trading_day_distance(calendar: pd.DatetimeIndex, earlier: str, later: str) -> int:
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values().unique()
    start = _index_on_or_before(calendar, pd.Timestamp(earlier))
    end = _index_on_or_before(calendar, pd.Timestamp(later))
    return int(end - start)


def assert_checkpoint_fresh(
    checkpoint_as_of: str,
    prediction_date: str,
    *,
    allow_stale_days: int = 0,
    calendar: pd.DatetimeIndex | None = None,
) -> None:
    if calendar is None:
        if allow_stale_days == 0 and checkpoint_as_of != prediction_date:
            raise StaleCheckpointError(
                f"checkpoint as-of {checkpoint_as_of} does not match prediction date {prediction_date}"
            )
        return
    age = trading_day_distance(calendar, checkpoint_as_of, prediction_date)
    if age > int(allow_stale_days):
        raise StaleCheckpointError(
            f"checkpoint age {age} trading days exceeds allow_stale_checkpoint_days={allow_stale_days}"
        )


def write_asof_metadata(directory: str | Path, plan: DailyFoldPlan, *, extra: dict | None = None) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    payload = asdict(plan)
    payload["inner_oof"] = [asdict(item) for item in plan.inner_oof]
    payload["retrained"] = True
    if extra:
        payload.update(extra)
    target = path / "rolling_asof_metadata.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_asof_metadata(directory: str | Path) -> dict:
    return json.loads((Path(directory) / "rolling_asof_metadata.json").read_text(encoding="utf-8"))


def assert_strict_checkpoint(output_root: str | Path, prediction_date: str, *, allow_stale_days: int = 0) -> dict:
    stamp = pd.Timestamp(prediction_date).strftime("%Y%m%d")
    directory = Path(output_root) / "rolling_models" / stamp
    metadata_path = directory / "rolling_asof_metadata.json"
    if not metadata_path.exists():
        raise StaleCheckpointError(
            f"strict daily inference requires {metadata_path}; predict-only cannot reuse an older fold"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("retrained", False):
        raise StaleCheckpointError(f"checkpoint {directory} is not marked retrained")
    assert_checkpoint_fresh(
        str(metadata.get("checkpoint_as_of")),
        pd.Timestamp(prediction_date).strftime("%Y-%m-%d"),
        allow_stale_days=allow_stale_days,
    )
    return metadata
