"""把组合策略接到 practice 回测、基线回测和推理结果上。

默认不启用。只有 portfolio_policy.enabled 或 PORTFOLIO_POLICY_ENABLED=1
时才替换原有 TopK 权重。
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from portfolio_policy import (
    PortfolioPolicyConfig,
    build_target,
    config_from_dict,
    config_from_env,
    portfolio_policy_enabled,
)
from position_report import render_exposure_html, render_exposure_text, write_exposure_bundle


def load_policy(payload: Mapping[str, Any] | None = None) -> PortfolioPolicyConfig:
    config = config_from_dict(payload)
    if any(
        os_value is not None
        for os_value in (
            _env("PORTFOLIO_POLICY_ENABLED"),
            _env("PORTFOLIO_MAX_HOLDINGS"),
            _env("SCORE_GATE_MODE"),
        )
    ):
        config = config_from_env(config)
    return config


def _env(name: str) -> str | None:
    import os

    return os.environ.get(name)


def _config_for_day(config: PortfolioPolicyConfig, day_rows: pd.DataFrame) -> PortfolioPolicyConfig:
    """同一天的校准门槛来自该日冻结的产物，不沿用一个全局常数。"""
    if config.score_gate.mode == "disabled":
        return config
    if "entry_threshold" not in day_rows.columns or day_rows["entry_threshold"].dropna().empty:
        return config
    updated = deepcopy(config)
    gate = updated.score_gate
    sample = day_rows.dropna(subset=["entry_threshold"]).iloc[0]
    gate.mode = "calibrated_net_excess"
    gate.artifact_available = True
    gate.thresholds_locked = True
    gate.entry_threshold = float(sample["entry_threshold"])
    gate.exit_threshold = float(sample["exit_threshold"])
    gate.min_confidence = float(sample["min_confidence"])
    if "safety_margin_bps" in day_rows.columns and pd.notna(sample.get("safety_margin_bps")):
        gate.safety_margin_bps = float(sample["safety_margin_bps"])
    if "round_trip_cost" in day_rows.columns and pd.notna(sample.get("round_trip_cost")):
        gate.default_round_trip_cost = float(sample["round_trip_cost"])
    version = sample.get("artifact_version")
    if pd.notna(version):
        gate.artifact_version = str(version)
        gate.expected_artifact_version = str(version)
    profile = sample.get("score_gate_profile")
    if pd.notna(profile):
        gate.profile = str(profile)
    updated.validate()
    return updated


def _rebalance_dates(dates: pd.DatetimeIndex, freq: str) -> set[pd.Timestamp]:
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values().unique()
    if freq == "daily":
        return set(pd.Timestamp(value) for value in dates)
    series = pd.Series(dates, index=dates)
    if freq == "weekly":
        grouped = series.groupby([dates.isocalendar().year, dates.isocalendar().week]).max()
    elif freq == "monthly":
        grouped = series.groupby([dates.year, dates.month]).max()
    elif freq == "yearly":
        grouped = series.groupby(dates.year).max()
    else:
        grouped = series
    return set(pd.Timestamp(value) for value in grouped.tolist())


def _next_trade_date(calendar: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp:
    position = int(calendar.searchsorted(pd.Timestamp(day), side="right"))
    if position >= len(calendar):
        return pd.Timestamp(day)
    return pd.Timestamp(calendar[position])


def build_policy_signal(
    raw_signal: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    config: PortfolioPolicyConfig,
    *,
    freq: str = "daily",
    eligibility_by_date: Mapping | None = None,
    price_cap: float | None = None,
    closes: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """从原始分数生成执行日目标权重。权重和可以小于 1。"""
    frame = raw_signal.reset_index()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["instrument"] = frame["instrument"].astype(str)
    score_column = "score" if "score" in frame.columns else "rank_score"
    frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce")
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values().unique()
    decision_days = _rebalance_dates(pd.DatetimeIndex(frame["datetime"].unique()), freq)
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    liquidate_dates: set[str] = set()
    freeze_dates: set[str] = set()
    status_by_date: dict[str, str] = {}
    held: list[dict[str, Any]] = []

    for day in sorted(decision_days):
        day_rows = frame[frame["datetime"] == day].copy()
        scores = []
        for row in day_rows.reset_index().to_dict(orient="records"):
            rank_score = row.get(score_column)
            if rank_score is None or pd.isna(rank_score):
                continue
            scores.append(
                {
                    "instrument": row.get("instrument"),
                    "rank_score": float(rank_score),
                    "expected_excess_return": None if pd.isna(row.get("expected_excess_return")) else float(row.get("expected_excess_return")),
                    "gate_confidence": None if pd.isna(row.get("gate_confidence")) else float(row.get("gate_confidence")),
                    "raw_prediction": None if pd.isna(row.get("raw_prediction")) else float(row.get("raw_prediction")),
                }
            )
        eligibility = dict((eligibility_by_date or {}).get(pd.Timestamp(day), {}))
        if price_cap is not None and closes is not None and not closes.empty:
            close_day = closes[closes["datetime"] == day]
            for item in close_day.itertuples(index=False):
                current = dict(eligibility.get(item.instrument, {}))
                if float(item.close) > float(price_cap):
                    current["hard_entry_eligible"] = False
                    current["hard_entry_reason"] = "LOT_UNAFFORDABLE"
                eligibility[str(item.instrument)] = current
        execution = _next_trade_date(calendar, pd.Timestamp(day))
        day_config = _config_for_day(config, day_rows)
        decision = build_target(
            scores,
            current_positions=held,
            eligibility=eligibility,
            config=day_config,
            as_of_time=pd.Timestamp(day).strftime("%Y-%m-%d"),
            execution_time=execution.strftime("%Y-%m-%d"),
        )
        payload = decision.to_dict()
        decisions.append(payload)
        status_by_date[execution.strftime("%Y-%m-%d")] = decision.decision_status
        if decision.decision_status == "VALID_EMPTY":
            liquidate_dates.add(execution.strftime("%Y-%m-%d"))
        if decision.decision_status == "DATA_ERROR":
            freeze_dates.add(execution.strftime("%Y-%m-%d"))
        if decision.target_positions:
            for instrument, weight in decision.target_positions.items():
                rank_score = next((item["rank_score"] for item in scores if item["instrument"] == instrument), 0.0)
                rows.append(
                    {
                        "datetime": execution,
                        "instrument": instrument,
                        "score": rank_score,
                        "weight": weight,
                    }
                )
        else:
            rows.append(
                {
                    "datetime": execution,
                    "instrument": "__CASH_SLOT__",
                    "score": 0.0,
                    "weight": 0.0,
                }
            )
        held = [
            {"instrument": instrument, "weight": weight}
            for instrument, weight in decision.target_positions.items()
        ]

    signal = pd.DataFrame(rows)
    if signal.empty:
        signal = pd.DataFrame(columns=["score", "weight"])
    else:
        signal = signal.set_index(["datetime", "instrument"]).sort_index()
    state = {
        "enabled": True,
        "construction": (
            f"最多 {config.max_holdings} 个槽位，未通过准入的槽位保留现金，"
            f"score_gate={config.score_gate.mode}/{config.score_gate.profile}"
        ),
        "max_holdings": int(config.max_holdings),
        "decisions": decisions,
        "liquidate_dates": liquidate_dates,
        "freeze_dates": freeze_dates,
        "decision_status_by_date": status_by_date,
        "risk_degree": 1.0,
    }
    return signal, state


def write_policy_artifacts(directory: str | Path, state: Mapping[str, Any], calendar: pd.DatetimeIndex | None = None) -> None:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    decisions = list(state.get("decisions") or [])
    pd.DataFrame(
        [
            {
                "decision_date": item.get("as_of_time"),
                "execution_date": item.get("execution_time"),
                "decision_status": item.get("decision_status"),
                "max_holdings": item.get("max_holdings"),
                "model_reject_count": item.get("model_reject_count"),
                "hard_reject_count": item.get("hard_reject_count"),
                "selected_count": item.get("selected_count"),
                "target_gross_exposure": item.get("target_gross_exposure"),
                "target_cash_weight": item.get("target_cash_weight"),
            }
            for item in decisions
        ]
    ).to_csv(root / "portfolio_decisions.csv", index=False, encoding="utf-8-sig")
    gate_rows = []
    for item in decisions:
        for detail in item.get("gate_details") or []:
            gate_rows.append({"decision_date": item.get("as_of_time"), **detail})
    pd.DataFrame(gate_rows).to_csv(root / "score_gate_decisions.csv", index=False, encoding="utf-8-sig")
    frame = write_exposure_bundle(root, decisions, calendar=calendar)
    (root / "position_exposure_section.html").write_text(render_exposure_html(frame), encoding="utf-8")
    summary = render_exposure_text(
        frame,
        k=state.get("max_holdings"),
        profile=(decisions[0].get("diagnostics") or {}).get("score_gate_profile") if decisions else None,
    )
    (root / "position_exposure_report.txt").write_text(summary, encoding="utf-8")


def apply_policy_to_ranked_rows(
    rows: list[dict[str, Any]],
    *,
    max_holdings: int,
    score_gate_profile: str | None = None,
    score_gate_mode: str | None = None,
    holdings: list[str] | None = None,
    artifact_available: bool = False,
    thresholds_locked: bool = False,
    entry_threshold: float = 0.0,
    exit_threshold: float = 0.0,
    min_confidence: float = 0.0,
) -> dict[str, Any]:
    """推理和调仓建议共用的组合结果。模型排名仍由调用方单独返回。"""
    config = config_from_dict(
        {
            "enabled": True,
            "max_holdings": int(max_holdings),
            "score_gate": {
                "mode": score_gate_mode or ("disabled" if not artifact_available else "calibrated_net_excess"),
                "profile": score_gate_profile or "balanced",
                "artifact_available": artifact_available,
                "thresholds_locked": thresholds_locked or score_gate_mode == "disabled" or not artifact_available,
                "entry_threshold": entry_threshold,
                "exit_threshold": exit_threshold,
                "min_confidence": min_confidence,
                "missing_artifact_policy": "freeze_no_buy",
            },
        }
    )
    if score_gate_mode is None and not artifact_available:
        config.score_gate.mode = "disabled"
        config.score_gate.thresholds_locked = True
        config.validate()
    decision = build_target(
        rows,
        current_positions=holdings or [],
        config=config,
    )
    payload = decision.to_dict()
    payload["model_ranking"] = [
        {
            "instrument": row.get("instrument") or row.get("code"),
            "rank_score": row.get("rank_score", row.get("score")),
            "rank": row.get("rank"),
        }
        for row in rows
    ]
    payload["selected_codes"] = list(decision.selected)
    return payload


def strategy_from_score_frame(score_frame: pd.DataFrame, config: PortfolioPolicyConfig, calendar: pd.DatetimeIndex):
    signal, state = build_policy_signal(score_frame, calendar, config)
    from variable_holding_strategy import create_qlib_weight_strategy

    return create_qlib_weight_strategy(
        signal,
        liquidate_dates=state["liquidate_dates"],
        freeze_dates=state["freeze_dates"],
        max_holdings=state["max_holdings"],
        decision_status_by_date=state["decision_status_by_date"],
    )


def write_runtime_fingerprint(config: Mapping[str, Any], directory: str | Path) -> None:
    runtime = dict(config.get("runtime") or {})
    payload = {
        "docker_image": runtime.get("docker_image"),
        "image_digest": runtime.get("image_digest"),
        "note": "可变持仓回测与基线共用这一镜像标识；digest 为空表示镜像尚未固定。",
    }
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / "runtime_image.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
