"""目标仓位、实际仓位和文本/HTML 摘要。

目标从执行日开始生效。DATA_ERROR 延续上一目标，不能画成清仓。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import pandas as pd

from rolling_schedule import LeakageError, latest_labeled_sample


EXPOSURE_COLUMNS = [
    "date",
    "fold_id",
    "decision_date",
    "effective_target_date",
    "decision_status",
    "score_gate_profile",
    "checkpoint_as_of",
    "retrained",
    "train_sample_start",
    "train_label_end",
    "calibration_start",
    "calibration_end",
    "label_embargo_days",
    "checkpoint_age_trading_days",
    "max_holdings",
    "raw_topk_count",
    "model_reject_count",
    "hard_reject_count",
    "target_holding_count",
    "actual_holding_count",
    "target_gross_exposure",
    "actual_gross_exposure",
    "target_cash_weight",
    "actual_cash_weight",
    "holding_count_gap",
    "exposure_gap",
    "blocked_buy_count",
    "blocked_sell_count",
    "daily_turnover",
    "transaction_cost",
    "equity",
]


def _positive_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    return {
        str(code): float(weight)
        for code, weight in (weights or {}).items()
        if weight is not None and float(weight) > 0
    }


def build_target_position_rows(decisions: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for decision in decisions:
        weights = _positive_weights(decision.get("target_positions") or decision.get("target_weights"))
        details = {row.get("instrument"): row for row in decision.get("gate_details") or []}
        actions = {row.get("instrument"): row for row in decision.get("planned_actions") or []}
        for code, weight in weights.items():
            detail = details.get(code) or {}
            action = actions.get(code) or {}
            rows.append(
                {
                    "decision_date": decision.get("as_of_time") or decision.get("decision_date"),
                    "execution_date": decision.get("execution_time") or decision.get("execution_date"),
                    "instrument": code,
                    "rank": detail.get("rank"),
                    "rank_score": detail.get("rank_score"),
                    "gate_value": detail.get("expected_net_excess"),
                    "gate_confidence": detail.get("gate_confidence"),
                    "entry_threshold": detail.get("entry_threshold"),
                    "exit_threshold": detail.get("exit_threshold"),
                    "target_weight": weight,
                    "action": action.get("action", "keep"),
                    "reason": action.get("reason"),
                }
            )
    return pd.DataFrame(rows)


def build_exposure_frame(
    decisions: Sequence[Mapping[str, Any]],
    actuals: Sequence[Mapping[str, Any]] | None = None,
    *,
    calendar: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """把调仓决策展开成逐交易日目标，并和实际持仓对齐。"""
    actual_by_date = {
        pd.Timestamp(row["date"]): row for row in (actuals or []) if row.get("date") is not None
    }
    ordered = sorted(
        decisions,
        key=lambda item: pd.Timestamp(item.get("execution_time") or item.get("execution_date")),
    )
    if calendar is None:
        dates: list[pd.Timestamp] = []
        for decision in ordered:
            dates.append(pd.Timestamp(decision.get("execution_time") or decision.get("execution_date")))
        dates.extend(actual_by_date)
        calendar_index = pd.DatetimeIndex(sorted(set(dates)))
    else:
        calendar_index = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values().unique()

    rows: list[dict[str, Any]] = []
    current_target: Mapping[str, Any] | None = None
    decision_iter = iter(ordered)
    pending = next(decision_iter, None)
    for day in calendar_index:
        while pending is not None and pd.Timestamp(pending.get("execution_time") or pending.get("execution_date")) <= day:
            status = pending.get("decision_status")
            if status in {"VALID", "VALID_EMPTY"}:
                current_target = pending
            elif status == "DATA_ERROR":
                if current_target is None:
                    current_target = {**pending, "target_gross_exposure": None, "target_cash_weight": None}
                else:
                    current_target = {**current_target, "decision_status": "DATA_ERROR", "decision_date": pending.get("as_of_time")}
            pending = next(decision_iter, None)
        actual = actual_by_date.get(pd.Timestamp(day), {})
        _check_actual_identity(actual)
        target_gross = None if current_target is None else current_target.get("target_gross_exposure")
        target_cash = None if target_gross is None else 1.0 - float(target_gross)
        target_count = None if target_gross is None else int(current_target.get("selected_count") or 0)
        if current_target is not None and current_target.get("decision_status") == "VALID_EMPTY":
            target_gross = 0.0
            target_cash = 1.0
            target_count = 0
        actual_count = actual.get("actual_holding_count")
        actual_gross = actual.get("actual_gross_exposure")
        actual_cash = actual.get("actual_cash_weight")
        source = current_target or {}
        rows.append(
            {
                "date": pd.Timestamp(day).strftime("%Y-%m-%d"),
                "fold_id": source.get("fold_id"),
                "decision_date": source.get("as_of_time") or source.get("decision_date"),
                "effective_target_date": None
                if source.get("execution_time") is None and source.get("execution_date") is None
                else pd.Timestamp(source.get("execution_time") or source.get("execution_date")).strftime("%Y-%m-%d"),
                "decision_status": None if current_target is None else current_target.get("decision_status"),
                "score_gate_profile": source.get("score_gate_profile") or (source.get("diagnostics") or {}).get("score_gate_profile"),
                "checkpoint_as_of": source.get("checkpoint_as_of"),
                "retrained": source.get("retrained"),
                "train_sample_start": source.get("train_sample_start"),
                "train_label_end": source.get("train_label_end"),
                "calibration_start": source.get("calibration_start"),
                "calibration_end": source.get("calibration_end"),
                "label_embargo_days": source.get("label_embargo_days"),
                "checkpoint_age_trading_days": source.get("checkpoint_age_trading_days", 0),
                "max_holdings": source.get("max_holdings"),
                "raw_topk_count": len(source.get("raw_topk") or []),
                "model_reject_count": source.get("model_reject_count"),
                "hard_reject_count": source.get("hard_reject_count"),
                "target_holding_count": target_count,
                "actual_holding_count": actual_count,
                "target_gross_exposure": target_gross,
                "actual_gross_exposure": actual_gross,
                "target_cash_weight": target_cash,
                "actual_cash_weight": actual_cash,
                "holding_count_gap": None
                if target_count is None or actual_count is None
                else int(actual_count) - int(target_count),
                "exposure_gap": None
                if target_gross is None or actual_gross is None
                else float(actual_gross) - float(target_gross),
                "blocked_buy_count": actual.get("blocked_buy_count", source.get("blocked_buy_count", 0)),
                "blocked_sell_count": actual.get("blocked_sell_count", source.get("blocked_exit_count", 0)),
                "daily_turnover": actual.get("daily_turnover"),
                "transaction_cost": actual.get("transaction_cost"),
                "equity": actual.get("equity"),
            }
        )
    frame = pd.DataFrame(rows, columns=EXPOSURE_COLUMNS)
    validate_exposure_frame(frame, calendar=calendar_index)
    return frame


def validate_exposure_frame(frame: pd.DataFrame, *, calendar: pd.DatetimeIndex | None = None) -> None:
    for row in frame.itertuples(index=False):
        status = row.decision_status
        if status == "VALID_EMPTY":
            if row.target_gross_exposure not in (0, 0.0) or row.target_cash_weight not in (1, 1.0):
                raise AssertionError("VALID_EMPTY must target 0% stocks and 100% cash")
        if status == "DATA_ERROR" and row.target_gross_exposure == 0 and row.decision_date is None:
            raise AssertionError("DATA_ERROR without a previous target must stay unknown, not flat cash")
        if row.target_gross_exposure is not None and row.target_cash_weight is not None:
            if abs(float(row.target_gross_exposure) + float(row.target_cash_weight) - 1.0) > 1e-8:
                raise AssertionError("target exposure and cash must sum to 1")
        if row.actual_gross_exposure is not None and row.actual_cash_weight is not None:
            if abs(float(row.actual_gross_exposure) + float(row.actual_cash_weight) - 1.0) > 1e-6:
                raise AssertionError("actual exposure and cash must sum to 1")
        embargo = row.label_embargo_days
        if (
            calendar is not None
            and pd.notna(row.train_label_end)
            and pd.notna(row.decision_date)
            and pd.notna(embargo)
        ):
            allowed = latest_labeled_sample(
                calendar, pd.Timestamp(row.decision_date), int(embargo)
            )
            if pd.Timestamp(row.train_label_end) > allowed:
                raise LeakageError(
                    f"train_label_end {row.train_label_end} is after embargo boundary {allowed.date()}"
                )


def _check_actual_identity(actual: Mapping[str, Any]) -> None:
    equity = actual.get("equity")
    cash = actual.get("cash")
    position_value = actual.get("position_value")
    if equity is None or cash is None or position_value is None:
        return
    if abs(float(equity) - float(cash) - float(position_value)) > 1e-4:
        raise AssertionError("equity must equal cash plus position value")


def actual_position_value(row: Any) -> float | None:
    equity = getattr(row, "equity", None)
    cash_weight = getattr(row, "actual_cash_weight", None)
    if equity is None or cash_weight is None or (isinstance(cash_weight, float) and math.isnan(cash_weight)):
        return None
    cash = float(equity) * float(cash_weight)
    gross = getattr(row, "actual_gross_exposure", None)
    if gross is None:
        return None
    return cash + float(equity) * float(gross)


def build_actual_position_rows(snapshots: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        for position in snapshot.get("positions") or []:
            shares = float(position.get("shares") or 0.0)
            if shares == 0:
                continue
            price = float(position.get("unadjusted_close") or 0.0)
            value = shares * price
            rows.append(
                {
                    "date": snapshot.get("date"),
                    "instrument": position.get("instrument"),
                    "shares": shares,
                    "unadjusted_close": price,
                    "market_value": value,
                    "actual_weight": position.get("actual_weight"),
                    "target_weight": position.get("target_weight"),
                    "weight_gap": None
                    if position.get("actual_weight") is None or position.get("target_weight") is None
                    else float(position["actual_weight"]) - float(position["target_weight"]),
                    "status": position.get("status", "NORMAL"),
                }
            )
    return pd.DataFrame(rows)


def summarize_exposure(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    target_count = pd.to_numeric(frame["target_holding_count"], errors="coerce")
    actual_count = pd.to_numeric(frame["actual_holding_count"], errors="coerce")
    target_cash = pd.to_numeric(frame["target_cash_weight"], errors="coerce")
    actual_cash = pd.to_numeric(frame["actual_cash_weight"], errors="coerce")
    gap = pd.to_numeric(frame["exposure_gap"], errors="coerce")
    max_holdings = pd.to_numeric(frame["max_holdings"], errors="coerce")
    full_days = int(((target_count >= max_holdings) & max_holdings.notna()).sum()) if len(frame) else 0
    return {
        "days": int(len(frame)),
        "avg_target_holdings": float(target_count.mean()),
        "min_target_holdings": float(target_count.min()),
        "max_target_holdings": float(target_count.max()),
        "avg_actual_holdings": None if actual_count.dropna().empty else float(actual_count.mean()),
        "min_actual_holdings": None if actual_count.dropna().empty else float(actual_count.min()),
        "max_actual_holdings": None if actual_count.dropna().empty else float(actual_count.max()),
        "avg_target_cash": float(target_cash.mean()),
        "avg_actual_cash": None if actual_cash.dropna().empty else float(actual_cash.mean()),
        "full_slot_day_ratio": full_days / len(frame),
        "empty_days": int((target_count == 0).sum()),
        "data_error_days": int((frame["decision_status"] == "DATA_ERROR").sum()),
        "max_abs_exposure_gap": None if gap.dropna().empty else float(gap.abs().max()),
        "mean_model_reject_count": float(pd.to_numeric(frame["model_reject_count"], errors="coerce").mean()),
        "mean_hard_reject_count": float(pd.to_numeric(frame["hard_reject_count"], errors="coerce").mean()),
    }


def render_exposure_text(frame: pd.DataFrame, *, k: int | None = None, profile: str | None = None) -> str:
    summary = summarize_exposure(frame)
    lines = [
        "持仓与仓位变化",
        "====================================",
        f"最大持仓槽位 K: {k if k is not None else '见逐日文件'}",
        f"score gate profile: {profile or '见逐日文件'}",
        f"交易日: {summary.get('days', 0)}",
        f"目标持仓数 平均/最小/最大: {summary.get('avg_target_holdings')} / {summary.get('min_target_holdings')} / {summary.get('max_target_holdings')}",
        f"实际持仓数 平均/最小/最大: {summary.get('avg_actual_holdings')} / {summary.get('min_actual_holdings')} / {summary.get('max_actual_holdings')}",
        f"平均目标现金: {summary.get('avg_target_cash')}",
        f"平均实际现金: {summary.get('avg_actual_cash')}",
        f"满槽位日比例: {summary.get('full_slot_day_ratio')}",
        f"全现金日: {summary.get('empty_days')}",
        f"数据异常日: {summary.get('data_error_days')}",
        f"最大目标/实际仓位偏差: {summary.get('max_abs_exposure_gap')}",
        f"S_score 均值: {summary.get('mean_model_reject_count')}",
        f"S_hard 均值: {summary.get('mean_hard_reject_count')}",
        "",
    ]
    return "\n".join(lines)


def render_exposure_html(frame: pd.DataFrame) -> str:
    summary = summarize_exposure(frame)
    if frame.empty:
        return "<div class=\"card\"><h2>持仓与仓位变化</h2><p>没有仓位决策。</p></div>"
    points = []
    for index, cash in enumerate(pd.to_numeric(frame["target_cash_weight"], errors="coerce").fillna(0)):
        points.append(f"{index * 8},{20 + float(cash) * 60:.2f}")
    spark = " ".join(points)
    preview = frame.head(8).to_html(index=False)
    return f"""
<div class="card" id="position-exposure">
  <h2>持仓与仓位变化</h2>
  <p>平均目标持仓 {summary.get('avg_target_holdings')}，平均目标现金 {summary.get('avg_target_cash')}，全现金日 {summary.get('empty_days')}，数据异常日 {summary.get('data_error_days')}。</p>
  <svg viewBox="0 0 800 100" width="100%" height="100" role="img" aria-label="目标现金比例">
    <polyline fill="none" stroke="#0f766e" stroke-width="2" points="{spark}"></polyline>
  </svg>
  <p class="muted">VALID_EMPTY 显示目标 0% 股票、100% 现金。DATA_ERROR 延续上一目标，不画成清仓。</p>
  {preview}
</div>
"""


def write_exposure_bundle(
    directory,
    decisions: Sequence[Mapping[str, Any]],
    actuals: Sequence[Mapping[str, Any]] | None = None,
    *,
    calendar: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    from pathlib import Path

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    frame = build_exposure_frame(decisions, actuals, calendar=calendar)
    frame.to_csv(root / "position_exposure_daily.csv", index=False, encoding="utf-8-sig")
    build_target_position_rows(decisions).to_csv(root / "target_positions.csv", index=False, encoding="utf-8-sig")
    build_actual_position_rows(actuals or []).to_csv(root / "actual_positions.csv", index=False, encoding="utf-8-sig")
    (root / "position_exposure_report.txt").write_text(render_exposure_text(frame), encoding="utf-8")
    (root / "position_exposure_section.html").write_text(render_exposure_html(frame), encoding="utf-8")
    return frame
