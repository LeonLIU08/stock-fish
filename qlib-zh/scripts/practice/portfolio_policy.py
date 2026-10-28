"""可配置最大槽位的组合决策。

本模块不调用 Qlib。它只根据当日排名、模型准入、硬资格和当前持仓，
生成目标权重。空槽位留在现金里，不把剩余股票重新归一化成满仓。
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd


VALID = "VALID"
VALID_EMPTY = "VALID_EMPTY"
DATA_ERROR = "DATA_ERROR"
NO_REBALANCE = "NO_REBALANCE"

MODEL_SCORE_GATE_FAILED = "MODEL_SCORE_GATE_FAILED"
NO_SCORE = "NO_SCORE"
OUT_OF_UNIVERSE = "OUT_OF_UNIVERSE"
NEWLY_LISTED = "NEWLY_LISTED"
ST_OR_DELISTING = "ST_OR_DELISTING"
SUSPENDED = "SUSPENDED"
LOW_LIQUIDITY = "LOW_LIQUIDITY"
RISK_VETO = "RISK_VETO"
LOT_UNAFFORDABLE = "LOT_UNAFFORDABLE"
SCORE_GATE_FAILED = "SCORE_GATE_FAILED"
MISSING_REQUIRED_DATA = "MISSING_REQUIRED_DATA"
CAPACITY_EXIT = "CAPACITY_EXIT"
RANK_BUFFER_EXIT = "RANK_BUFFER_EXIT"
RANK_UPGRADE = "RANK_UPGRADE"
MANDATORY_EXIT = "MANDATORY_EXIT"
HOLD = "HOLD"
BUY = "BUY"

CASH_DISCLAIMER = (
    "当前现金仓位只来自硬过滤或执行限制，不代表模型判断市场整体不值得持有。"
)
RELATIVE_DISCLAIMER = (
    "relative_score_threshold 只控制相对持仓数量，不代表绝对收益判断。"
)

_MISSING = object()


def strict_qualification_enabled() -> bool:
    """开启后，筛选链路不得放宽条件、回填否决股票或把空仓补满。"""
    return os.environ.get("PORTFOLIO_POLICY_ENABLED", "0") == "1" or os.environ.get(
        "STRICT_NO_RELAXATION", "0"
    ) == "1"


def portfolio_policy_enabled() -> bool:
    return os.environ.get("PORTFOLIO_POLICY_ENABLED", "0") == "1"


@dataclass
class ScoreGateConfig:
    mode: str = "calibrated_net_excess"
    profile: str = "balanced"
    entry_threshold: float = 0.0
    exit_threshold: float = 0.0
    min_confidence: float = 0.0
    safety_margin_bps: float = 0.0
    default_round_trip_cost: float = 0.0
    missing_artifact_policy: str = "freeze_no_buy"
    artifact_available: bool = False
    artifact_version: str | None = None
    expected_artifact_version: str | None = None
    thresholds_locked: bool = False
    thresholds_by_profile: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class PortfolioPolicyConfig:
    enabled: bool = False
    max_holdings: int = 50
    allow_new_buys_outside_topk: bool = False
    master_candidate_pool_multiplier: float = 3.0
    unfilled_slot_policy: str = "cash"
    base_cash_ratio: float = 0.0
    risk_scale: float = 1.0
    max_regular_replacements: int = 5
    hold_rank_buffer_multiplier: float = 2.0
    mandatory_exit_bypasses_limit: bool = True
    allocation_method: str = "equal"
    risk_degree: float = 1.0
    score_gate: ScoreGateConfig = field(default_factory=ScoreGateConfig)

    def validate(self) -> None:
        if int(self.max_holdings) < 1:
            raise ValueError("max_holdings must be a positive integer")
        if self.max_holdings != int(self.max_holdings):
            raise ValueError("max_holdings must be an integer")
        self.max_holdings = int(self.max_holdings)
        if self.allow_new_buys_outside_topk:
            raise ValueError("allow_new_buys_outside_topk must be false")
        if float(self.master_candidate_pool_multiplier) < 1.0:
            raise ValueError("master_candidate_pool_multiplier must be >= 1")
        if self.unfilled_slot_policy != "cash":
            raise ValueError("unfilled_slot_policy must be cash")
        if not 0.0 <= float(self.base_cash_ratio) < 1.0:
            raise ValueError("base_cash_ratio must be in [0, 1)")
        if not 0.0 <= float(self.risk_scale) <= 1.0:
            raise ValueError("risk_scale must be in [0, 1]")
        if int(self.max_regular_replacements) < 0:
            raise ValueError("max_regular_replacements must be >= 0")
        if float(self.hold_rank_buffer_multiplier) < 1.0:
            raise ValueError("hold_rank_buffer_multiplier must be >= 1")
        gate = self.score_gate
        if float(gate.exit_threshold) > float(gate.entry_threshold) + 1e-15:
            raise ValueError("exit_threshold must be <= entry_threshold")
        if not 0.0 <= float(gate.min_confidence) <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if gate.mode not in {"calibrated_net_excess", "disabled", "relative_score_threshold"}:
            raise ValueError(f"unknown score_gate.mode: {gate.mode}")
        if gate.missing_artifact_policy not in {"freeze_no_buy", "disabled"}:
            raise ValueError("missing_artifact_policy must be freeze_no_buy or disabled")
        if abs(float(self.risk_degree) - 1.0) > 1e-12:
            raise ValueError("risk_degree must be 1.0; cash is expressed only by target weights")


def master_candidate_pool_size(config: PortfolioPolicyConfig) -> int:
    config.validate()
    pool = int(math.ceil(config.max_holdings * float(config.master_candidate_pool_multiplier)))
    return max(pool, int(config.max_holdings))


def config_from_dict(payload: Mapping[str, Any] | None) -> PortfolioPolicyConfig:
    data = dict(payload or {})
    gate_payload = dict(data.get("score_gate") or {})
    allocation = dict(data.get("allocation") or {})
    turnover = dict(data.get("turnover") or {})
    execution = dict(data.get("execution") or {})
    gate = ScoreGateConfig(
        mode=str(gate_payload.get("mode", "calibrated_net_excess")),
        profile=str(gate_payload.get("profile", "balanced")),
        entry_threshold=float(gate_payload.get("entry_threshold") or gate_payload.get("entry_threshold_override") or 0.0),
        exit_threshold=float(gate_payload.get("exit_threshold") or gate_payload.get("exit_threshold_override") or 0.0),
        min_confidence=float(gate_payload.get("min_confidence") or gate_payload.get("min_confidence_override") or 0.0),
        safety_margin_bps=float(
            gate_payload.get("safety_margin_bps")
            if gate_payload.get("safety_margin_bps") is not None
            else gate_payload.get("safety_margin_bps_override") or 0.0
        ),
        default_round_trip_cost=float(gate_payload.get("default_round_trip_cost", 0.0) or 0.0),
        missing_artifact_policy=str(gate_payload.get("missing_artifact_policy", "freeze_no_buy")),
        artifact_available=bool(gate_payload.get("artifact_available", False)),
        artifact_version=gate_payload.get("artifact_version"),
        expected_artifact_version=gate_payload.get("expected_artifact_version"),
        thresholds_locked=bool(gate_payload.get("thresholds_locked", False)),
        thresholds_by_profile=dict(gate_payload.get("thresholds_by_profile") or {}),
    )
    if gate_payload.get("entry_threshold_override") is not None or gate_payload.get("entry_threshold") is not None:
        if "entry_threshold" in gate_payload or gate_payload.get("entry_threshold_override") is not None:
            gate.thresholds_locked = True
    config = PortfolioPolicyConfig(
        enabled=bool(data.get("enabled", False)),
        max_holdings=int(data.get("max_holdings", 50)),
        allow_new_buys_outside_topk=bool(data.get("allow_new_buys_outside_topk", False)),
        master_candidate_pool_multiplier=float(data.get("master_candidate_pool_multiplier", 3.0)),
        unfilled_slot_policy=str(data.get("unfilled_slot_policy", "cash")),
        base_cash_ratio=float(allocation.get("base_cash_ratio", data.get("base_cash_ratio", 0.0)) or 0.0),
        risk_scale=float(allocation.get("risk_scale", data.get("risk_scale", 1.0))),
        max_regular_replacements=int(turnover.get("max_regular_replacements", data.get("max_regular_replacements", 5))),
        hold_rank_buffer_multiplier=float(
            turnover.get("hold_rank_buffer_multiplier", data.get("hold_rank_buffer_multiplier", 2.0))
        ),
        mandatory_exit_bypasses_limit=bool(
            turnover.get("mandatory_exit_bypasses_limit", data.get("mandatory_exit_bypasses_limit", True))
        ),
        allocation_method=str(allocation.get("method", data.get("allocation_method", "equal"))),
        risk_degree=float(execution.get("risk_degree", data.get("risk_degree", 1.0))),
        score_gate=gate,
    )
    config.validate()
    return config


def config_from_env(base: PortfolioPolicyConfig | None = None) -> PortfolioPolicyConfig:
    """环境变量只在迁移期覆盖配置，不替代配置对象。"""
    config = base or PortfolioPolicyConfig()
    if os.environ.get("PORTFOLIO_POLICY_ENABLED") is not None:
        config.enabled = os.environ.get("PORTFOLIO_POLICY_ENABLED", "0") == "1"
    if os.environ.get("PORTFOLIO_MAX_HOLDINGS"):
        config.max_holdings = int(os.environ["PORTFOLIO_MAX_HOLDINGS"])
    if os.environ.get("SCORE_GATE_MODE"):
        config.score_gate.mode = os.environ["SCORE_GATE_MODE"]
    if os.environ.get("SCORE_GATE_PROFILE"):
        config.score_gate.profile = os.environ["SCORE_GATE_PROFILE"]
    if os.environ.get("SCORE_GATE_ENTRY"):
        config.score_gate.entry_threshold = float(os.environ["SCORE_GATE_ENTRY"])
        config.score_gate.thresholds_locked = True
    if os.environ.get("SCORE_GATE_EXIT"):
        config.score_gate.exit_threshold = float(os.environ["SCORE_GATE_EXIT"])
        config.score_gate.thresholds_locked = True
    if os.environ.get("SCORE_GATE_MIN_CONFIDENCE"):
        config.score_gate.min_confidence = float(os.environ["SCORE_GATE_MIN_CONFIDENCE"])
        config.score_gate.thresholds_locked = True
    if os.environ.get("BASE_CASH_RATIO"):
        config.base_cash_ratio = float(os.environ["BASE_CASH_RATIO"])
    if os.environ.get("RISK_SCALE"):
        config.risk_scale = float(os.environ["RISK_SCALE"])
    config.validate()
    return config


@dataclass
class ScoreInput:
    instrument: str
    rank_score: float
    expected_excess_return: float | None = None
    gate_confidence: float | None = None
    estimated_round_trip_cost: float | None = None
    raw_prediction: float | None = None


@dataclass
class HardEligibility:
    hard_entry_eligible: bool = True
    hard_hold_eligible: bool = True
    hard_entry_reason: str | None = None
    hard_hold_reason: str | None = None
    tradable_buy: bool = True
    tradable_sell: bool = True


@dataclass
class HeldPosition:
    instrument: str
    weight: float | None = None
    shares: float = 0.0


@dataclass
class PortfolioDecision:
    decision_status: str
    max_holdings: int
    raw_topk: list[str]
    model_rejects: list[str]
    model_reject_count: int
    hard_rejects: list[str]
    hard_reject_count: int
    unfilled_score_slots: int
    selected: list[str]
    selected_count: int
    eligible_count: int
    blocked_exit_count: int
    target_gross_exposure: float
    target_cash_weight: float
    target_positions: dict[str, float]
    exclusions: list[dict[str, Any]]
    planned_actions: list[dict[str, Any]]
    planned_sells: list[str]
    planned_buys: list[str]
    gate_details: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    as_of_time: str | None = None
    execution_time: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_weights"] = dict(self.target_positions)
        payload["gate_value"] = {
            row["instrument"]: row.get("expected_net_excess") for row in self.gate_details
        }
        payload["gate_confidence"] = {
            row["instrument"]: row.get("gate_confidence") for row in self.gate_details
        }
        return payload


def _blank_decision(
    config: PortfolioPolicyConfig,
    status: str,
    *,
    selected: Sequence[str],
    weights: Mapping[str, float],
    diagnostics: dict[str, Any],
    as_of_time: str | None,
    execution_time: str | None,
    planned_sells: Sequence[str] | None = None,
    planned_buys: Sequence[str] | None = None,
) -> PortfolioDecision:
    gross = float(sum(weights.values()))
    return PortfolioDecision(
        decision_status=status,
        max_holdings=int(config.max_holdings),
        raw_topk=[],
        model_rejects=[],
        model_reject_count=0,
        hard_rejects=[],
        hard_reject_count=0,
        unfilled_score_slots=0,
        selected=list(selected),
        selected_count=len(list(selected)),
        eligible_count=0,
        blocked_exit_count=0,
        target_gross_exposure=gross,
        target_cash_weight=1.0 - gross,
        target_positions={str(k): float(v) for k, v in weights.items() if float(v) > 0},
        exclusions=[],
        planned_actions=[],
        planned_sells=list(planned_sells or []),
        planned_buys=list(planned_buys or []),
        gate_details=[],
        diagnostics=diagnostics,
        as_of_time=as_of_time,
        execution_time=execution_time,
    )


def _coerce_scores(scores: Any) -> list[ScoreInput]:
    if scores is None:
        return []
    if isinstance(scores, pd.DataFrame):
        frame = scores.reset_index()
        rows = frame.to_dict(orient="records")
    elif isinstance(scores, Mapping):
        rows = []
        for instrument, value in scores.items():
            if isinstance(value, Mapping):
                item = dict(value)
                item.setdefault("instrument", instrument)
                rows.append(item)
            else:
                rows.append({"instrument": instrument, "rank_score": value})
    else:
        rows = list(scores)
    parsed: list[ScoreInput] = []
    for row in rows:
        instrument = str(row.get("instrument") or row.get("code") or "")
        if not instrument:
            continue
        rank_score = row.get("rank_score", row.get("score"))
        if rank_score is None or (isinstance(rank_score, float) and math.isnan(rank_score)):
            continue
        parsed.append(
            ScoreInput(
                instrument=instrument,
                rank_score=float(rank_score),
                expected_excess_return=_optional_float(
                    row.get("expected_excess_return", row.get("gate_value"))
                ),
                gate_confidence=_optional_float(row.get("gate_confidence")),
                estimated_round_trip_cost=_optional_float(row.get("estimated_round_trip_cost")),
                raw_prediction=_optional_float(row.get("raw_prediction")),
            )
        )
    return parsed


def _optional_float(value: Any) -> float | None:
    if value is None or value is _MISSING:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _coerce_eligibility(eligibility: Any) -> dict[str, HardEligibility]:
    if not eligibility:
        return {}
    output: dict[str, HardEligibility] = {}
    items = eligibility.items() if isinstance(eligibility, Mapping) else []
    for instrument, value in items:
        if isinstance(value, HardEligibility):
            output[str(instrument)] = value
            continue
        payload = value if isinstance(value, Mapping) else {}
        output[str(instrument)] = HardEligibility(
            hard_entry_eligible=bool(payload.get("hard_entry_eligible", True)),
            hard_hold_eligible=bool(payload.get("hard_hold_eligible", True)),
            hard_entry_reason=payload.get("hard_entry_reason"),
            hard_hold_reason=payload.get("hard_hold_reason"),
            tradable_buy=bool(payload.get("tradable_buy", True)),
            tradable_sell=bool(payload.get("tradable_sell", True)),
        )
    return output


def _coerce_positions(current_positions: Any) -> list[HeldPosition]:
    if not current_positions:
        return []
    output: list[HeldPosition] = []
    for item in current_positions:
        if isinstance(item, HeldPosition):
            output.append(item)
        elif isinstance(item, str):
            output.append(HeldPosition(instrument=item))
        elif isinstance(item, Mapping):
            output.append(
                HeldPosition(
                    instrument=str(item.get("instrument") or item.get("code")),
                    weight=_optional_float(item.get("weight")),
                    shares=float(item.get("shares") or 0.0),
                )
            )
        else:
            output.append(HeldPosition(instrument=str(item)))
    return output


def _resolve_gate_thresholds(config: PortfolioPolicyConfig) -> tuple[float, float, float, str | None]:
    gate = config.score_gate
    profile = gate.thresholds_by_profile.get(gate.profile) or {}
    if gate.thresholds_locked or not profile:
        entry = float(gate.entry_threshold)
        exit_threshold = float(gate.exit_threshold)
        min_confidence = float(gate.min_confidence)
    else:
        entry = float(profile.get("entry_threshold", gate.entry_threshold))
        exit_threshold = float(profile.get("exit_threshold", gate.exit_threshold))
        min_confidence = float(profile.get("min_confidence", gate.min_confidence))
    if exit_threshold > entry + 1e-15:
        return entry, exit_threshold, min_confidence, "exit_threshold above entry_threshold"
    return entry, exit_threshold, min_confidence, None


def _gate_status(config: PortfolioPolicyConfig) -> tuple[str, str | None]:
    """返回 (effective_mode, error)。error 非空时整天冻结。"""
    gate = config.score_gate
    if gate.mode == "disabled":
        return "disabled", None
    if gate.mode == "relative_score_threshold":
        return "relative_score_threshold", None
    if gate.expected_artifact_version and gate.artifact_version != gate.expected_artifact_version:
        return "calibrated_net_excess", "score gate artifact version mismatch"
    if not gate.artifact_available:
        if gate.missing_artifact_policy == "disabled":
            return "disabled", None
        return "calibrated_net_excess", "score gate calibrator unavailable"
    if not gate.thresholds_locked and gate.profile not in gate.thresholds_by_profile:
        return "calibrated_net_excess", "score gate thresholds unavailable"
    return "calibrated_net_excess", None


def _evaluate_one(
    row: ScoreInput | None,
    *,
    mode: str,
    entry: float,
    exit_threshold: float,
    min_confidence: float,
    safety_margin: float,
    default_cost: float,
) -> dict[str, Any]:
    if row is None:
        return {
            "expected_net_excess": None,
            "gate_confidence": None,
            "entry_pass": False,
            "hold_pass": False,
            "missing": True,
        }
    cost = default_cost if row.estimated_round_trip_cost is None else float(row.estimated_round_trip_cost)
    confidence = 1.0 if row.gate_confidence is None and mode != "calibrated_net_excess" else row.gate_confidence
    if mode == "disabled":
        net = None if row.expected_excess_return is None else float(row.expected_excess_return) - cost - safety_margin
        return {
            "expected_net_excess": net,
            "gate_confidence": confidence,
            "entry_pass": True,
            "hold_pass": True,
            "missing": False,
        }
    if mode == "relative_score_threshold":
        return {
            "expected_net_excess": float(row.rank_score),
            "gate_confidence": 1.0 if confidence is None else float(confidence),
            "entry_pass": float(row.rank_score) >= entry and (confidence is None or float(confidence) >= min_confidence),
            "hold_pass": float(row.rank_score) >= exit_threshold and (confidence is None or float(confidence) >= min_confidence),
            "missing": False,
        }
    if row.expected_excess_return is None or row.gate_confidence is None:
        return {
            "expected_net_excess": None,
            "gate_confidence": row.gate_confidence,
            "entry_pass": False,
            "hold_pass": False,
            "missing": True,
        }
    net = float(row.expected_excess_return) - cost - safety_margin
    conf = float(row.gate_confidence)
    return {
        "expected_net_excess": net,
        "gate_confidence": conf,
        "entry_pass": net >= entry and conf >= min_confidence,
        "hold_pass": net >= exit_threshold and conf >= min_confidence,
        "missing": False,
    }


def _worst(instruments: Sequence[str], rank_of: Mapping[str, int], count: int) -> list[str]:
    # rank 大的更差；同排名按代码降序，保证稳定。
    ordered = sorted(instruments, key=lambda code: (int(rank_of.get(code, 10**12)), code), reverse=True)
    return ordered[: max(int(count), 0)]


def _best(instruments: Sequence[str], rank_of: Mapping[str, int], count: int) -> list[str]:
    ordered = sorted(set(instruments), key=lambda code: (int(rank_of.get(code, 10**12)), code))
    return ordered[: max(int(count), 0)]


def _relative_weights(
    selected: Sequence[str],
    rank_score: Mapping[str, float],
    method: str,
    provided: Mapping[str, float] | None,
) -> dict[str, float]:
    if not selected:
        return {}
    if method in {"preserve_strategy_allocator", "provided"} and provided:
        raw = {code: max(float(provided.get(code, 0.0)), 0.0) for code in selected}
        if sum(raw.values()) <= 0:
            raw = {code: 1.0 for code in selected}
    elif method == "score":
        values = {code: float(rank_score.get(code, 0.0)) for code in selected}
        floor = min(values.values()) if values else 0.0
        raw = {code: value - floor + 0.01 for code, value in values.items()}
    else:
        raw = {code: 1.0 for code in selected}
    total = float(sum(raw.values()))
    if total <= 0:
        return {code: 1.0 / len(selected) for code in selected}
    return {code: value / total for code, value in raw.items()}


def _scale_weights(relative: Mapping[str, float], gross: float) -> dict[str, float]:
    return {code: float(weight) * float(gross) for code, weight in relative.items() if float(weight) > 0}


def _freeze_weights(positions: Sequence[HeldPosition]) -> dict[str, float]:
    weighted = {item.instrument: float(item.weight) for item in positions if item.weight is not None and item.weight > 0}
    if weighted:
        return weighted
    if not positions:
        return {}
    equal = 1.0 / len(positions)
    return {item.instrument: equal for item in positions}


def build_target(
    scores: Any,
    current_positions: Any = None,
    eligibility: Any = None,
    config: PortfolioPolicyConfig | None = None,
    *,
    as_of_time: str | None = None,
    execution_time: str | None = None,
    cross_section_status: str = VALID,
    rebalance: bool = True,
    relative_weights: Mapping[str, float] | None = None,
    account_equity: float | None = None,
) -> PortfolioDecision:
    """生成一个调仓日的目标组合。"""
    config = config or PortfolioPolicyConfig()
    config.validate()
    held = _coerce_positions(current_positions)
    held_codes = [item.instrument for item in held]
    diagnostics: dict[str, Any] = {
        "account_equity": account_equity,
        "allocation_method": config.allocation_method,
        "score_gate_mode": config.score_gate.mode,
        "score_gate_profile": config.score_gate.profile,
        "risk_degree": 1.0,
    }
    if not rebalance or cross_section_status == NO_REBALANCE:
        weights = _freeze_weights(held)
        diagnostics["carry_forward"] = True
        return _blank_decision(
            config,
            NO_REBALANCE,
            selected=held_codes,
            weights=weights,
            diagnostics=diagnostics,
            as_of_time=as_of_time,
            execution_time=execution_time,
        )

    mode, gate_error = _gate_status(config)
    if cross_section_status == DATA_ERROR or gate_error:
        weights = _freeze_weights(held)
        diagnostics["error"] = gate_error or "cross section data error"
        diagnostics["freeze_no_buy"] = True
        return _blank_decision(
            config,
            DATA_ERROR,
            selected=held_codes,
            weights=weights,
            diagnostics=diagnostics,
            as_of_time=as_of_time,
            execution_time=execution_time,
        )

    if mode == "disabled":
        diagnostics["cash_source_disclaimer"] = CASH_DISCLAIMER
    elif mode == "relative_score_threshold":
        diagnostics["cash_source_disclaimer"] = RELATIVE_DISCLAIMER

    entry, exit_threshold, min_confidence, threshold_error = _resolve_gate_thresholds(config)
    if threshold_error and mode == "calibrated_net_excess":
        weights = _freeze_weights(held)
        diagnostics["error"] = threshold_error
        return _blank_decision(
            config, DATA_ERROR, selected=held_codes, weights=weights, diagnostics=diagnostics,
            as_of_time=as_of_time, execution_time=execution_time,
        )
    diagnostics["entry_threshold"] = entry
    diagnostics["exit_threshold"] = exit_threshold
    diagnostics["min_confidence"] = min_confidence

    parsed = _coerce_scores(scores)
    ranked = sorted(parsed, key=lambda row: (-row.rank_score, row.instrument))
    k = int(config.max_holdings)
    raw = ranked[:k]
    raw_codes = [row.instrument for row in raw]
    by_code = {row.instrument: row for row in ranked}
    rank_of = {row.instrument: index + 1 for index, row in enumerate(ranked)}
    for code in held_codes:
        rank_of.setdefault(code, 10**12)

    eligibility_map = _coerce_eligibility(eligibility)
    safety_margin = float(config.score_gate.safety_margin_bps) / 10000.0
    needed = set(raw_codes) | set(held_codes)
    gate_result: dict[str, dict[str, Any]] = {}
    for code in needed:
        gate_result[code] = _evaluate_one(
            by_code.get(code),
            mode=mode,
            entry=entry,
            exit_threshold=exit_threshold,
            min_confidence=min_confidence,
            safety_margin=safety_margin,
            default_cost=float(config.score_gate.default_round_trip_cost),
        )
    if mode == "calibrated_net_excess" and any(gate_result[code]["missing"] for code in needed):
        weights = _freeze_weights(held)
        diagnostics["error"] = "missing gate inputs"
        diagnostics["missing_gate_instruments"] = sorted(
            code for code in needed if gate_result[code]["missing"]
        )
        return _blank_decision(
            config, DATA_ERROR, selected=held_codes, weights=weights, diagnostics=diagnostics,
            as_of_time=as_of_time, execution_time=execution_time,
        )

    def hard_of(code: str) -> HardEligibility:
        return eligibility_map.get(code, HardEligibility())

    model_rejects = [code for code in raw_codes if not gate_result[code]["entry_pass"]]
    hard_rejects = [
        code
        for code in raw_codes
        if gate_result[code]["entry_pass"] and not hard_of(code).hard_entry_eligible
    ]
    entry_pool = [
        code
        for code in raw_codes
        if gate_result[code]["entry_pass"] and hard_of(code).hard_entry_eligible
    ]
    unfilled = max(k - len(raw_codes), 0)
    buffer_rank = int(math.ceil(k * float(config.hold_rank_buffer_multiplier)))
    drop_budget = int(config.max_regular_replacements)

    current = set(held_codes)
    mandatory = [
        code
        for code in held_codes
        if (not gate_result.get(code, {}).get("hold_pass", False)) or (not hard_of(code).hard_hold_eligible)
    ]
    retainable = [code for code in held_codes if code not in set(mandatory)]
    capacity = _worst(retainable, rank_of, max(len(retainable) - k, 0))
    retainable_set = set(retainable) - set(capacity)
    outside = [code for code in retainable_set if rank_of.get(code, 10**12) > buffer_rank]
    upgrade_candidates = [code for code in retainable_set if code not in set(entry_pool)]
    available_entrants = [
        code
        for code in entry_pool
        if code not in current and hard_of(code).tradable_buy
    ]
    buffer_sells = _worst(outside, rank_of, drop_budget)
    remaining_drop = drop_budget - len(buffer_sells)
    upgrade_pool = [code for code in upgrade_candidates if code not in set(buffer_sells)]
    upgrade_sells = _worst(upgrade_pool, rank_of, min(remaining_drop, len(available_entrants)))
    regular_sells = buffer_sells + upgrade_sells
    planned_sells = list(dict.fromkeys(mandatory + capacity + regular_sells))
    retained = _best([code for code in retainable_set if code not in set(regular_sells)], rank_of, k)
    vacancies = max(k - len(retained), 0)
    planned_buys = _best(
        [code for code in available_entrants if code not in set(planned_sells)],
        rank_of,
        vacancies,
    )
    selected = _best(list(retained) + list(planned_buys), rank_of, k)
    if len(selected) > k:
        selected = selected[:k]
    relative = _relative_weights(selected, {row.instrument: row.rank_score for row in ranked}, config.allocation_method, relative_weights)
    gross = (1.0 - float(config.base_cash_ratio)) * float(config.risk_scale) * (len(selected) / float(k))
    weights = _scale_weights(relative, gross)
    cash = 1.0 - gross
    base_cash = float(config.base_cash_ratio)
    risk_cash = (1.0 - float(config.base_cash_ratio)) * (1.0 - float(config.risk_scale))
    dynamic_cash = (1.0 - float(config.base_cash_ratio)) * float(config.risk_scale) * ((k - len(selected)) / float(k))

    exclusions: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for code in model_rejects:
        exclusions.append({"instrument": code, "reason": MODEL_SCORE_GATE_FAILED, "stage": "score_gate"})
    for code in hard_rejects:
        reason = hard_of(code).hard_entry_reason or RISK_VETO
        exclusions.append({"instrument": code, "reason": reason, "stage": "hard_entry"})
    if unfilled:
        exclusions.append({"instrument": "", "reason": NO_SCORE, "stage": "universe", "count": unfilled})
    for code in mandatory:
        reason = MANDATORY_EXIT
        if not hard_of(code).hard_hold_eligible:
            reason = hard_of(code).hard_hold_reason or MANDATORY_EXIT
        elif not gate_result[code]["hold_pass"]:
            reason = MODEL_SCORE_GATE_FAILED
        actions.append({"instrument": code, "action": "mandatory_sell", "reason": reason})
    for code in capacity:
        actions.append({"instrument": code, "action": "sell", "reason": CAPACITY_EXIT})
    for code in buffer_sells:
        actions.append({"instrument": code, "action": "sell", "reason": RANK_BUFFER_EXIT})
    for code in upgrade_sells:
        actions.append({"instrument": code, "action": "sell", "reason": RANK_UPGRADE})
    for code in retained:
        actions.append({"instrument": code, "action": "keep", "reason": HOLD})
    for code in planned_buys:
        actions.append({"instrument": code, "action": "buy", "reason": BUY})

    gate_details = []
    for code in sorted(needed):
        row = by_code.get(code)
        result = gate_result[code]
        gate_details.append(
            {
                "instrument": code,
                "rank": None if rank_of.get(code, 10**12) >= 10**12 else int(rank_of[code]),
                "rank_score": None if row is None else row.rank_score,
                "expected_net_excess": result["expected_net_excess"],
                "gate_confidence": result["gate_confidence"],
                "entry_pass": bool(result["entry_pass"]),
                "hold_pass": bool(result["hold_pass"]),
                "entry_threshold": entry,
                "exit_threshold": exit_threshold,
                "min_confidence": min_confidence,
            }
        )

    blocked_exit = [
        code
        for code in planned_sells
        if not hard_of(code).tradable_sell
    ]
    diagnostics.update(
        {
            "buffer_rank": buffer_rank,
            "max_regular_replacements": drop_budget,
            "base_cash": base_cash,
            "risk_cash": risk_cash,
            "dynamic_slot_cash": dynamic_cash,
            "master_candidate_pool_size": master_candidate_pool_size(config),
            "new_buys_outside_topk": False,
        }
    )
    status = VALID_EMPTY if not selected else VALID
    decision = PortfolioDecision(
        decision_status=status,
        max_holdings=k,
        raw_topk=raw_codes,
        model_rejects=model_rejects,
        model_reject_count=len(model_rejects),
        hard_rejects=hard_rejects,
        hard_reject_count=len(hard_rejects),
        unfilled_score_slots=unfilled,
        selected=selected,
        selected_count=len(selected),
        eligible_count=len(entry_pool),
        blocked_exit_count=len(blocked_exit),
        target_gross_exposure=gross,
        target_cash_weight=cash,
        target_positions=weights,
        exclusions=exclusions,
        planned_actions=actions,
        planned_sells=planned_sells,
        planned_buys=planned_buys,
        gate_details=gate_details,
        diagnostics=diagnostics,
        as_of_time=as_of_time,
        execution_time=execution_time,
    )
    _assert_decision_invariants(decision)
    return decision


def _assert_decision_invariants(decision: PortfolioDecision) -> None:
    if decision.decision_status in {VALID, VALID_EMPTY}:
        if decision.selected_count > decision.max_holdings:
            raise AssertionError("selected holdings exceed max_holdings")
        if abs(decision.target_gross_exposure + decision.target_cash_weight - 1.0) > 1e-9:
            raise AssertionError("target weights and cash must sum to 1")
        if abs(sum(decision.target_positions.values()) - decision.target_gross_exposure) > 1e-8:
            raise AssertionError("target position weights must equal gross exposure")
        selected = set(decision.selected)
        raw = set(decision.raw_topk)
        held_keeps = {
            row["instrument"]
            for row in decision.planned_actions
            if row["action"] == "keep"
        }
        for code in decision.planned_buys:
            if code not in raw:
                raise AssertionError(f"new buy {code} is outside raw topk")
        for code in selected - held_keeps:
            if code not in raw and code in decision.planned_buys:
                raise AssertionError(f"new buy {code} is outside raw topk")


def evaluate_hard_qualification(snapshot: Mapping[str, Any], *, min_listing_days: int = 60, min_amount: float | None = None) -> HardEligibility:
    """把一只股票的时点快照变成硬资格和原因码。"""
    if not snapshot.get("has_score", True):
        return HardEligibility(False, False, NO_SCORE, NO_SCORE)
    if snapshot.get("in_universe", True) is False:
        return HardEligibility(False, False, OUT_OF_UNIVERSE, OUT_OF_UNIVERSE)
    listing_days = snapshot.get("listing_days")
    if listing_days is not None and float(listing_days) < float(min_listing_days):
        return HardEligibility(False, True, NEWLY_LISTED, None)
    if snapshot.get("is_st") or snapshot.get("is_delisting"):
        return HardEligibility(False, False, ST_OR_DELISTING, ST_OR_DELISTING)
    if snapshot.get("is_suspended"):
        return HardEligibility(False, True, SUSPENDED, None, tradable_buy=False, tradable_sell=False)
    if min_amount is not None:
        amount = snapshot.get("avg_amount")
        if amount is None:
            return HardEligibility(False, True, MISSING_REQUIRED_DATA, None)
        if float(amount) < float(min_amount):
            return HardEligibility(False, True, LOW_LIQUIDITY, None)
    if snapshot.get("risk_veto"):
        return HardEligibility(False, False, RISK_VETO, RISK_VETO)
    if snapshot.get("lot_affordable") is False:
        return HardEligibility(False, True, LOT_UNAFFORDABLE, None)
    if snapshot.get("missing_required_data"):
        return HardEligibility(False, False, MISSING_REQUIRED_DATA, MISSING_REQUIRED_DATA)
    tradable_buy = bool(snapshot.get("tradable_buy", True))
    tradable_sell = bool(snapshot.get("tradable_sell", True))
    entry_reason = None if tradable_buy else SUSPENDED
    return HardEligibility(
        tradable_buy,
        True,
        entry_reason,
        None,
        tradable_buy=tradable_buy,
        tradable_sell=tradable_sell,
    )
