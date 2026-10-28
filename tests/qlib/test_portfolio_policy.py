"""可变持仓、分数门槛、单日滚动和仓位报告的单元测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PRACTICE = Path(__file__).resolve().parents[2] / "qlib-zh" / "scripts" / "practice"
sys.path.insert(0, str(PRACTICE))

from label_embargo import label_embargo_days, label_expression_for_model_mode
from portfolio_policy import (
    PortfolioPolicyConfig,
    ScoreGateConfig,
    build_target,
    config_from_dict,
    evaluate_hard_qualification,
    master_candidate_pool_size,
)
from position_report import build_exposure_frame, render_exposure_html, render_exposure_text
from rolling_schedule import (
    LeakageError,
    StaleCheckpointError,
    assert_checkpoint_fresh,
    assert_oof_rows_not_in_training_labels,
    assert_strict_checkpoint,
    build_daily_plan,
    must_run_final_refit,
    write_asof_metadata,
)
from score_gate import fit_binned_mapping, fit_score_calibrator
from variable_holding_strategy import (
    affordable_lots,
    finalize_signal_weights,
    lot_affordable,
    lot_size_for_instrument,
    plan_execution,
)


def policy(k=50, entry=0.0, exit_threshold=0.0, confidence=0.0, mode="calibrated_net_excess", **kwargs):
    gate_kwargs = dict(
        mode=mode,
        artifact_available=True,
        thresholds_locked=True,
        entry_threshold=entry,
        exit_threshold=exit_threshold,
        min_confidence=confidence,
    )
    gate_kwargs.update(kwargs.pop("gate", {}))
    return PortfolioPolicyConfig(
        enabled=True,
        max_holdings=k,
        score_gate=ScoreGateConfig(**gate_kwargs),
        **kwargs,
    )


def universe(k, pass_count, hard_fail=0, entry=0.01, confidence=0.8):
    """前 pass_count 只通过模型门槛，其中前 hard_fail 只再被硬过滤。"""
    scores = []
    eligibility = {}
    for index in range(k):
        passed = index < pass_count
        scores.append(
            {
                "instrument": f"S{index:03d}",
                "rank_score": float(k - index),
                "expected_excess_return": entry + 0.01 if passed else entry - 0.01,
                "gate_confidence": confidence,
            }
        )
        hard_ok = not (passed and index < hard_fail)
        eligibility[f"S{index:03d}"] = {
            "hard_entry_eligible": hard_ok,
            "hard_hold_eligible": hard_ok,
            "hard_entry_reason": None if hard_ok else "RISK_VETO",
            "hard_hold_reason": None if hard_ok else "RISK_VETO",
        }
    return scores, eligibility


class PortfolioPolicyTests(unittest.TestCase):
    def test_examples_scale_cash_by_empty_slots(self):
        cases = [
            (50, 49, 1, 48, 0.96),
            (50, 43, 3, 40, 0.80),
            (50, 18, 3, 15, 0.30),
            (30, 22, 2, 20, 20 / 30),
            (50, 0, 0, 0, 0.0),
        ]
        for k, passed, hard_fail, expect_n, expect_gross in cases:
            scores, eligibility = universe(k, passed, hard_fail)
            decision = build_target(scores, eligibility=eligibility, config=policy(k=k, entry=0.01))
            self.assertEqual(decision.selected_count, expect_n)
            self.assertAlmostEqual(decision.target_gross_exposure, expect_gross)
            self.assertAlmostEqual(decision.target_cash_weight, 1 - expect_gross)
            self.assertLessEqual(decision.selected_count, k)
            self.assertEqual(decision.model_reject_count, k - passed)
            self.assertEqual(decision.hard_reject_count, hard_fail if passed else 0)

    def test_many_k_and_pass_counts(self):
        for k in (20, 50, 100):
            for passed in (0, 1, k - 1, k):
                scores, eligibility = universe(k, passed, 0)
                decision = build_target(scores, eligibility=eligibility, config=policy(k=k, entry=0.01))
                self.assertEqual(decision.selected_count, passed)
                self.assertAlmostEqual(sum(decision.target_positions.values()), passed / k)

    def test_partial_book_is_not_renormalized_to_full_investment(self):
        scores, eligibility = universe(50, 49, 0)
        decision = build_target(scores, eligibility=eligibility, config=policy(k=50, entry=0.01))
        self.assertEqual(decision.selected_count, 49)
        self.assertAlmostEqual(sum(decision.target_positions.values()), 49 / 50)
        self.assertLess(sum(decision.target_positions.values()), 0.99)

    def test_gate_and_confidence_boundaries(self):
        scores = [
            {"instrument": "A", "rank_score": 3, "expected_excess_return": 0.02, "gate_confidence": 0.50},
            {"instrument": "B", "rank_score": 2, "expected_excess_return": 0.019999, "gate_confidence": 0.50},
            {"instrument": "C", "rank_score": 1, "expected_excess_return": 0.03, "gate_confidence": 0.4999},
        ]
        decision = build_target(scores, config=policy(k=3, entry=0.02, exit_threshold=0.0, confidence=0.50))
        self.assertEqual(decision.selected, ["A"])
        self.assertIn("B", decision.model_rejects)
        self.assertIn("C", decision.model_rejects)

    def test_hold_threshold_keeps_existing_name_that_cannot_be_bought(self):
        scores = [
            {"instrument": "A", "rank_score": 3, "expected_excess_return": 0.03, "gate_confidence": 0.9},
            {"instrument": "B", "rank_score": 2, "expected_excess_return": 0.01, "gate_confidence": 0.9},
        ]
        decision = build_target(
            scores,
            current_positions=["B"],
            config=policy(k=1, entry=0.02, exit_threshold=0.0, confidence=0.5, max_regular_replacements=0),
        )
        self.assertEqual(decision.selected, ["B"])
        self.assertEqual(decision.planned_buys, [])

    def test_hard_exit_ignores_replacement_budget_and_capacity_exit_runs(self):
        scores = [
            {"instrument": code, "rank_score": 10 - index, "expected_excess_return": 0.05, "gate_confidence": 0.9}
            for index, code in enumerate(["C", "B", "A"])
        ]
        eligibility = {
            "A": {"hard_entry_eligible": False, "hard_hold_eligible": False, "hard_hold_reason": "RISK_VETO"},
            "B": {"hard_entry_eligible": True, "hard_hold_eligible": True},
            "C": {"hard_entry_eligible": True, "hard_hold_eligible": True},
        }
        decision = build_target(
            scores,
            current_positions=["A", "B", "C"],
            eligibility=eligibility,
            config=policy(k=1, entry=0.0, max_regular_replacements=0),
        )
        self.assertNotIn("A", decision.selected)
        self.assertLessEqual(decision.selected_count, 1)
        self.assertIn("A", decision.planned_sells)

    def test_no_backfill_beyond_topk_or_after_veto(self):
        scores = [
            {"instrument": "A", "rank_score": 3, "expected_excess_return": -0.02, "gate_confidence": 0.9},
            {"instrument": "B", "rank_score": 2, "expected_excess_return": -0.02, "gate_confidence": 0.9},
            {"instrument": "C", "rank_score": 1, "expected_excess_return": 0.20, "gate_confidence": 0.9},
        ]
        decision = build_target(scores, config=policy(k=2, entry=0.0))
        self.assertNotIn("C", decision.selected)
        self.assertEqual(decision.decision_status, "VALID_EMPTY")
        veto_scores, eligibility = universe(10, 10, 2)
        vetoed = build_target(veto_scores, eligibility=eligibility, config=policy(k=10, entry=0.01))
        self.assertNotIn("S000", vetoed.selected)
        self.assertNotIn("S001", vetoed.selected)
        self.assertEqual(vetoed.selected_count, 8)

    def test_short_universe_leaves_cash(self):
        scores = [
            {"instrument": f"S{index}", "rank_score": float(index), "expected_excess_return": 0.02, "gate_confidence": 0.8}
            for index in range(10)
        ]
        decision = build_target(scores, config=policy(k=50, entry=0.0))
        self.assertEqual(decision.selected_count, 10)
        self.assertEqual(decision.unfilled_score_slots, 40)
        self.assertAlmostEqual(decision.target_gross_exposure, 0.2)

    def test_raising_threshold_does_not_increase_holdings(self):
        scores = [
            {
                "instrument": f"S{index:02d}",
                "rank_score": float(20 - index),
                "expected_excess_return": index / 100,
                "gate_confidence": 0.9,
            }
            for index in range(20)
        ]
        loose = build_target(scores, config=policy(k=20, entry=0.05))
        strict = build_target(scores, config=policy(k=20, entry=0.12))
        self.assertLessEqual(strict.selected_count, loose.selected_count)
        self.assertTrue(set(strict.selected).issubset(set(loose.selected)))

    def test_existing_book_above_k_is_cut_without_using_drop_budget(self):
        scores = [
            {
                "instrument": f"S{index:03d}",
                "rank_score": float(60 - index),
                "expected_excess_return": 0.05,
                "gate_confidence": 0.9,
            }
            for index in range(60)
        ]
        decision = build_target(
            scores,
            current_positions=[f"S{index:03d}" for index in range(60)],
            config=policy(k=50, entry=0.0, max_regular_replacements=0),
        )
        self.assertEqual(decision.selected_count, 50)
        self.assertTrue(set(decision.selected).isdisjoint({f"S{index:03d}" for index in range(50, 60)}))

    def test_buffer_sell_can_leave_cash_without_replacement(self):
        scores = [
            {"instrument": "KEEP", "rank_score": 5, "expected_excess_return": 0.05, "gate_confidence": 0.9},
            {"instrument": "OUT", "rank_score": 0, "expected_excess_return": 0.05, "gate_confidence": 0.9},
        ]
        decision = build_target(
            scores,
            current_positions=["KEEP", "OUT"],
            config=policy(k=1, entry=0.0, max_regular_replacements=1, hold_rank_buffer_multiplier=1.0),
        )
        self.assertNotIn("OUT", decision.selected)
        self.assertIn("KEEP", decision.selected)

    def test_missing_calibrator_version_and_gate_input_freeze(self):
        scores, _ = universe(4, 4)
        frozen = policy(k=4, entry=0.01)
        frozen.score_gate.artifact_available = False
        decision = build_target(scores, current_positions=["S000"], config=frozen)
        self.assertEqual(decision.decision_status, "DATA_ERROR")
        self.assertEqual(decision.planned_buys, [])
        self.assertEqual(decision.selected, ["S000"])

        mismatched = policy(k=4, entry=0.01)
        mismatched.score_gate.expected_artifact_version = "new"
        mismatched.score_gate.artifact_version = "old"
        mismatch = build_target(scores, current_positions=["S000"], config=mismatched)
        self.assertEqual(mismatch.decision_status, "DATA_ERROR")

        broken = [dict(row) for row in scores]
        broken[0]["expected_excess_return"] = None
        missing = build_target(broken, current_positions=["S001"], config=policy(k=4, entry=0.01))
        self.assertEqual(missing.decision_status, "DATA_ERROR")
        self.assertEqual(missing.planned_sells, [])

    def test_valid_empty_data_error_and_no_rebalance_are_distinct(self):
        scores, _ = universe(3, 0)
        empty = build_target(scores, config=policy(k=3, entry=0.01))
        self.assertEqual(empty.decision_status, "VALID_EMPTY")
        self.assertEqual(empty.target_cash_weight, 1.0)
        self.assertEqual(empty.planned_buys, [])

        paused = build_target(scores, current_positions=["S000"], config=policy(k=3), rebalance=False)
        self.assertEqual(paused.decision_status, "NO_REBALANCE")
        self.assertEqual(paused.selected, ["S000"])

        broken = build_target(scores, current_positions=["S000"], config=policy(k=3), cross_section_status="DATA_ERROR")
        self.assertEqual(broken.decision_status, "DATA_ERROR")
        self.assertNotEqual(broken.target_positions, {})

    def test_cash_layers_and_config_guards(self):
        scores, eligibility = universe(50, 37, 0)
        decision = build_target(
            scores,
            eligibility=eligibility,
            config=policy(k=50, entry=0.01, base_cash_ratio=0.10, risk_scale=0.8),
        )
        self.assertAlmostEqual(decision.target_gross_exposure, 0.9 * 0.8 * 37 / 50)
        self.assertAlmostEqual(decision.diagnostics["base_cash"] + decision.diagnostics["risk_cash"] + decision.diagnostics["dynamic_slot_cash"], decision.target_cash_weight)
        with self.assertRaises(ValueError):
            master_candidate_pool_size(policy(k=50, master_candidate_pool_multiplier=0.5))
        self.assertEqual(master_candidate_pool_size(policy(k=50, master_candidate_pool_multiplier=3)), 150)
        with self.assertRaises(ValueError):
            policy(k=50, entry=0.01, exit_threshold=0.02).validate()
        loaded = config_from_dict(
            {
                "enabled": False,
                "max_holdings": 50,
                "allow_new_buys_outside_topk": False,
                "master_candidate_pool_multiplier": 3,
                "unfilled_slot_policy": "cash",
                "allocation": {"base_cash_ratio": 0, "risk_scale": 1, "method": "equal"},
                "turnover": {"max_regular_replacements": 5, "hold_rank_buffer_multiplier": 2},
                "execution": {"risk_degree": 1},
                "score_gate": {"mode": "disabled", "exit_threshold": 0, "entry_threshold": 0},
            }
        )
        self.assertFalse(loaded.enabled)

    def test_disabled_gate_marks_relative_cash_source(self):
        scores, eligibility = universe(5, 5)
        for row in scores:
            row["expected_excess_return"] = None
        decision = build_target(scores, eligibility=eligibility, config=policy(k=5, mode="disabled"))
        self.assertEqual(decision.selected_count, 5)
        self.assertIn("不代表模型判断市场整体不值得持有", decision.diagnostics["cash_source_disclaimer"])

    def test_hard_qualification_reason_codes(self):
        self.assertEqual(evaluate_hard_qualification({"is_st": True}).hard_entry_reason, "ST_OR_DELISTING")
        self.assertEqual(evaluate_hard_qualification({"listing_days": 10}).hard_entry_reason, "NEWLY_LISTED")
        self.assertTrue(evaluate_hard_qualification({"listing_days": 10}).hard_hold_eligible)
        self.assertEqual(
            evaluate_hard_qualification({"avg_amount": 1}, min_amount=10).hard_entry_reason,
            "LOW_LIQUIDITY",
        )
        self.assertEqual(evaluate_hard_qualification({"risk_veto": True}).hard_entry_reason, "RISK_VETO")
        self.assertEqual(evaluate_hard_qualification({"lot_affordable": False}).hard_entry_reason, "LOT_UNAFFORDABLE")


class ExecutionTests(unittest.TestCase):
    def test_cash_is_preserved_below_and_near_one(self):
        self.assertAlmostEqual(sum(finalize_signal_weights({"A": 0.4, "B": 0.4}, preserve_cash=True).values()), 0.8)
        near = finalize_signal_weights({"A": 0.5, "B": 0.495}, preserve_cash=True)
        self.assertAlmostEqual(sum(near.values()), 0.995)
        renormalized = finalize_signal_weights({"A": 0.5, "B": 0.495}, preserve_cash=False)
        self.assertAlmostEqual(sum(renormalized.values()), 1.0)

    def test_blocked_sell_blocks_replacement_and_failed_buy_stays_cash(self):
        execution = plan_execution(
            decision_status="VALID",
            desired_weights={"C": 1.0},
            current_weights={"A": 1.0},
            tradable_sell={"A": False},
            tradable_buy={"C": True},
            max_holdings=1,
        )
        self.assertEqual(execution.buy_instruments, [])
        self.assertEqual(execution.blocked_sells, ["A"])
        self.assertNotIn("C", execution.target_weights)

        priced = plan_execution(
            decision_status="VALID",
            desired_weights={"A": 0.5, "B": 0.5},
            current_weights={},
            max_holdings=2,
            prices={"A": 10.0, "B": 10.0},
            equity=1000,
            fee_rate=0.01,
            lot_sizes={"A": 100, "B": 100},
        )
        self.assertNotIn("A", priced.buy_instruments)
        self.assertNotIn("B", priced.buy_instruments)
        self.assertGreater(priced.dropped_buy_weight, 0)
        self.assertAlmostEqual(sum(priced.target_weights.values()), 0.0)

    def test_lot_boundary_and_market_lot_sizes(self):
        self.assertFalse(lot_affordable(10, 100, 1000, 0.001))
        self.assertTrue(lot_affordable(10, 100, 1001, 0.001))
        self.assertEqual(affordable_lots(10, 100, 1001, 0.001), 1)
        self.assertEqual(lot_size_for_instrument("SH688001"), 200)
        self.assertEqual(lot_size_for_instrument("SZ000001"), 100)
        self.assertEqual(lot_size_for_instrument("00700", {"00700": 500}), 500)

    def test_valid_empty_sells_and_data_error_does_not(self):
        liquidated = plan_execution(
            decision_status="VALID_EMPTY",
            desired_weights={},
            current_weights={"A": 0.4, "B": 0.4},
            max_holdings=2,
        )
        self.assertEqual(sorted(liquidated.sell_instruments), ["A", "B"])
        self.assertEqual(liquidated.buy_instruments, [])
        frozen = plan_execution(
            decision_status="DATA_ERROR",
            desired_weights={},
            current_weights={"A": 0.4},
            max_holdings=2,
        )
        self.assertEqual(frozen.sell_instruments, [])
        self.assertEqual(frozen.buy_instruments, [])
        self.assertIn("A", frozen.target_weights)


class ScoreAndScheduleTests(unittest.TestCase):
    def test_calibration_is_monotone_and_profiles_are_ordered(self):
        raw = list(range(100))
        target = [value / 100 + (0.1 if value % 7 == 0 else 0) for value in raw]
        mapping = fit_binned_mapping(raw, target, n_bins=8)
        predictions = [mapping.predict(value)[0] for value in raw]
        self.assertEqual(predictions, sorted(predictions))
        frame = pd.DataFrame(
            {
                "datetime": ["2024-01-02"] * 40 + ["2024-01-03"] * 40,
                "instrument": [f"S{index:03d}" for index in range(40)] * 2,
                "model": ["alpha"] * 80,
                "raw_prediction": list(range(40)) + list(range(40)),
                "future_excess": [index / 200 for index in range(40)] * 2,
            }
        )
        calibrator = fit_score_calibrator(
            frame,
            as_of="2024-01-03",
            ensemble_weights={"alpha": 1.0},
        )
        ordered = [
            calibrator.thresholds[name]["entry_threshold"]
            for name in ("aggressive", "balanced", "conservative")
        ]
        self.assertEqual(ordered, sorted(ordered))
        for profile in calibrator.thresholds.values():
            self.assertLessEqual(profile["exit_threshold"], profile["entry_threshold"])
            self.assertGreaterEqual(profile["min_confidence"], 0)
            self.assertLessEqual(profile["min_confidence"], 1)
        with tempfile.TemporaryDirectory() as directory:
            calibrator.save(directory)
            loaded = type(calibrator).load(directory)
            self.assertEqual(loaded.version, calibrator.version)

    def test_label_embargo_for_robust_and_multihorizon(self):
        self.assertEqual(label_embargo_days("Ref($close, -2)/Ref($close, -1) - 1"), 2)
        expression = label_expression_for_model_mode("multihorizon")
        self.assertEqual(label_embargo_days(expression), 22)
        self.assertEqual(label_embargo_days(label_expression_for_model_mode("robust")), 2)

    def test_daily_plan_moves_forward_and_ignores_future_calendar_tail(self):
        calendar = pd.bdate_range("2018-01-01", "2026-10-30")
        plan = build_daily_plan(
            calendar,
            "2026-10-20",
            label_expression="Ref($close, -2)/Ref($close, -1) - 1",
            train_window_years=5,
            calibration_window_years=1,
        )
        self.assertEqual(plan.prediction_date, "2026-10-20")
        self.assertEqual(plan.execution_date, "2026-10-21")
        self.assertEqual(plan.train_label_end, "2026-10-16")
        self.assertEqual(plan.test_start, plan.test_end)
        self.assertEqual(plan.label_embargo_days, 2)
        longer = pd.bdate_range("2018-01-01", "2027-01-29")
        later = build_daily_plan(
            longer,
            "2026-10-20",
            label_expression="Ref($close, -2)/Ref($close, -1) - 1",
        )
        self.assertEqual(later.train_label_end, plan.train_label_end)
        self.assertEqual(later.test_end, plan.test_end)
        nxt = build_daily_plan(
            calendar,
            "2026-10-21",
            label_expression="Ref($close, -2)/Ref($close, -1) - 1",
        )
        self.assertGreater(pd.Timestamp(nxt.train_label_end), pd.Timestamp(plan.train_label_end))
        self.assertNotEqual(nxt.checkpoint_as_of, plan.checkpoint_as_of)
        self.assertTrue(must_run_final_refit(feature_cache_hit=True, oof_cache_hit=True))

    def test_inner_oof_rows_cannot_sit_inside_their_training_labels(self):
        calendar = pd.bdate_range("2024-01-01", "2026-10-30")
        plan = build_daily_plan(
            calendar,
            "2026-10-20",
            label_expression="Ref($close, -2)/Ref($close, -1) - 1",
            train_window_years=2,
            calibration_window_years=1,
        )
        rows = []
        for fold in plan.inner_oof:
            rows.append({"row_date": fold.test_start, "model_train_label_end": fold.train_label_end})
        assert_oof_rows_not_in_training_labels(pd.DataFrame(rows))
        leaked = pd.DataFrame([{"row_date": "2026-06-30", "model_train_label_end": "2026-06-30"}])
        with self.assertRaises(LeakageError):
            assert_oof_rows_not_in_training_labels(leaked)

    def test_stale_checkpoint_is_rejected(self):
        calendar = pd.bdate_range("2026-10-01", "2026-10-30")
        with self.assertRaises(StaleCheckpointError):
            assert_checkpoint_fresh("2026-10-20", "2026-10-21", allow_stale_days=0, calendar=calendar)
        assert_checkpoint_fresh("2026-10-21", "2026-10-21", allow_stale_days=0, calendar=calendar)
        with tempfile.TemporaryDirectory() as directory:
            plan = build_daily_plan(
                pd.bdate_range("2024-01-01", "2026-10-30"),
                "2026-10-20",
                label_expression="Ref($close, -2)/Ref($close, -1) - 1",
                train_window_years=1,
            )
            write_asof_metadata(Path(directory) / "rolling_models" / "20261020", plan)
            assert_strict_checkpoint(directory, "2026-10-20")
            with self.assertRaises(StaleCheckpointError):
                assert_strict_checkpoint(directory, "2026-10-21")


class ReportTests(unittest.TestCase):
    def test_exposure_carries_targets_and_checks_identity(self):
        calendar = pd.bdate_range("2026-10-01", "2026-10-23")
        decisions = [
            {
                "decision_status": "VALID",
                "as_of_time": "2026-10-19",
                "execution_time": "2026-10-20",
                "selected_count": 4,
                "max_holdings": 5,
                "target_gross_exposure": 0.8,
                "target_cash_weight": 0.2,
                "model_reject_count": 1,
                "hard_reject_count": 0,
                "raw_topk": ["A", "B", "C", "D", "E"],
                "target_positions": {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2},
                "diagnostics": {"score_gate_profile": "balanced"},
                "checkpoint_as_of": "2026-10-19",
                "train_label_end": "2026-10-15",
                "label_embargo_days": 2,
                "decision_date": "2026-10-19",
            },
            {
                "decision_status": "DATA_ERROR",
                "as_of_time": "2026-10-20",
                "execution_time": "2026-10-21",
                "selected_count": 0,
                "max_holdings": 5,
                "target_gross_exposure": 0.0,
                "model_reject_count": 0,
                "hard_reject_count": 0,
                "raw_topk": [],
                "target_positions": {},
            },
            {
                "decision_status": "VALID_EMPTY",
                "as_of_time": "2026-10-21",
                "execution_time": "2026-10-22",
                "selected_count": 0,
                "max_holdings": 5,
                "target_gross_exposure": 0.0,
                "target_cash_weight": 1.0,
                "model_reject_count": 5,
                "hard_reject_count": 0,
                "raw_topk": ["A"],
                "target_positions": {},
            },
        ]
        actuals = [
            {
                "date": "2026-10-20",
                "actual_holding_count": 4,
                "actual_gross_exposure": 0.75,
                "actual_cash_weight": 0.25,
                "equity": 1000,
                "cash": 250,
                "position_value": 750,
                "positions": [
                    {
                        "instrument": "A",
                        "shares": 100,
                        "unadjusted_close": 7.5,
                        "actual_weight": 0.75,
                        "target_weight": 0.2,
                        "status": "NORMAL",
                    }
                ],
            }
        ]
        frame = build_exposure_frame(decisions, actuals, calendar=calendar)
        carried = frame[frame["date"] == "2026-10-21"].iloc[0]
        self.assertEqual(carried.decision_status, "DATA_ERROR")
        self.assertAlmostEqual(carried.target_gross_exposure, 0.8)
        emptied = frame[frame["date"] == "2026-10-22"].iloc[0]
        self.assertEqual(emptied.decision_status, "VALID_EMPTY")
        self.assertEqual(emptied.target_gross_exposure, 0.0)
        self.assertEqual(emptied.target_cash_weight, 1.0)
        text = render_exposure_text(frame, k=5, profile="balanced")
        html = render_exposure_html(frame)
        self.assertIn("S_score", text)
        self.assertIn("持仓与仓位变化", html)
        self.assertIn("DATA_ERROR", html)

    def test_report_fails_when_training_label_crosses_embargo(self):
        calendar = pd.bdate_range("2026-10-01", "2026-10-30")
        decisions = [
            {
                "decision_status": "VALID",
                "as_of_time": "2026-10-20",
                "execution_time": "2026-10-21",
                "decision_date": "2026-10-20",
                "selected_count": 1,
                "max_holdings": 1,
                "target_gross_exposure": 1.0,
                "model_reject_count": 0,
                "hard_reject_count": 0,
                "raw_topk": ["A"],
                "target_positions": {"A": 1.0},
                "train_label_end": "2026-10-20",
                "label_embargo_days": 2,
            }
        ]
        with self.assertRaises(LeakageError):
            build_exposure_frame(decisions, calendar=calendar)


class CalibratorWiringTests(unittest.TestCase):
    def test_calibration_window_drops_future_and_leaked_rows(self):
        import sys
        from pathlib import Path

        hk = Path(__file__).resolve().parents[2] / "qlib-zh" / "hk"
        sys.path.insert(0, str(hk))
        from inner_oof import slice_calibration_rows

        calendar = pd.bdate_range("2024-01-01", "2026-10-30")
        rows = pd.DataFrame(
            [
                {"datetime": "2025-10-16", "model_train_label_end": "2025-10-14", "raw_prediction": 0.2, "future_excess": 0.01, "model": "xgboost"},
                {"datetime": "2026-10-20", "model_train_label_end": "2026-10-16", "raw_prediction": 0.3, "future_excess": 0.02, "model": "xgboost"},
                {"datetime": "2026-06-30", "model_train_label_end": "2026-06-30", "raw_prediction": 0.1, "future_excess": 0.0, "model": "xgboost"},
            ]
        )
        with self.assertRaises(ValueError):
            slice_calibration_rows(rows, calendar, "2026-10-20", 2, 1)
        clean = rows.iloc[:2]
        chosen = slice_calibration_rows(clean, calendar, "2026-10-20", 2, 1)
        self.assertNotIn(pd.Timestamp("2026-10-20"), set(pd.to_datetime(chosen["datetime"])))
        self.assertIn(pd.Timestamp("2025-10-16"), set(pd.to_datetime(chosen["datetime"])))

    def test_saved_threshold_is_used_for_that_day(self):
        from portfolio_backtest import build_policy_signal

        day = pd.Timestamp("2026-10-20")
        frame = pd.DataFrame(
            [
                {
                    "datetime": day,
                    "instrument": "A",
                    "score": 2.0,
                    "expected_excess_return": 0.03,
                    "gate_confidence": 0.9,
                    "entry_threshold": 0.0,
                    "exit_threshold": -0.01,
                    "min_confidence": 0.4,
                    "safety_margin_bps": 0.0,
                    "round_trip_cost": 0.0,
                    "artifact_version": "abc",
                    "score_gate_profile": "balanced",
                },
                {
                    "datetime": day,
                    "instrument": "B",
                    "score": 1.0,
                    "expected_excess_return": -0.01,
                    "gate_confidence": 0.9,
                    "entry_threshold": 0.0,
                    "exit_threshold": -0.01,
                    "min_confidence": 0.4,
                    "safety_margin_bps": 0.0,
                    "round_trip_cost": 0.0,
                    "artifact_version": "abc",
                    "score_gate_profile": "balanced",
                },
            ]
        ).set_index(["datetime", "instrument"])
        config = policy(k=2, entry=0.0, mode="calibrated_net_excess")
        config.score_gate.artifact_available = False
        _signal, state = build_policy_signal(frame, pd.bdate_range("2026-10-20", "2026-10-21"), config)
        decision = state["decisions"][0]
        self.assertEqual(decision["decision_status"], "VALID")
        self.assertEqual(decision["selected"], ["A"])
        self.assertAlmostEqual(decision["target_gross_exposure"], 0.5)


class WiringTests(unittest.TestCase):
    def test_inference_ranking_is_separate_from_portfolio_target(self):
        from portfolio_backtest import apply_policy_to_ranked_rows

        rows = [{"instrument": f"S{index}", "rank_score": 10 - index, "rank": index + 1} for index in range(6)]
        decision = apply_policy_to_ranked_rows(rows, max_holdings=4)
        self.assertEqual(decision["selected_count"], 4)
        self.assertEqual(len(decision["model_ranking"]), 6)
        self.assertAlmostEqual(decision["target_gross_exposure"], 1.0)
        blocked = apply_policy_to_ranked_rows(
            rows,
            max_holdings=4,
            score_gate_mode="calibrated_net_excess",
            artifact_available=False,
        )
        self.assertEqual(blocked["decision_status"], "DATA_ERROR")
        self.assertEqual(blocked["planned_buys"], [])

    def test_baseline_config_keeps_policy_disabled(self):
        import yaml

        document = yaml.safe_load((Path(__file__).resolve().parents[2] / "qlib-zh" / "cn" / "baseline.yaml").read_text())
        config = config_from_dict(document["portfolio_policy"])
        self.assertFalse(config.enabled)
        self.assertEqual(config.max_holdings, 50)
        self.assertEqual(config.risk_degree, 1.0)

    def test_final_refit_yaml_allows_validation_inside_training(self):
        from gen_practice_yaml import patch_yaml

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.yaml"
            template.write_text(
                """
data_handler_config: {}
market: csi300
benchmark: SH000300
task:
  model:
    kwargs: {}
  dataset:
    kwargs:
      segments:
        train: ["2018-01-01", "2018-12-31"]
        valid: ["2019-01-01", "2019-06-30"]
        test: ["2019-07-01", "2019-12-31"]
      handler:
        class: Alpha158
        module_path: qlib.contrib.data.handler
        kwargs: {}
  record: []
port_analysis_config:
  strategy:
    kwargs: {}
  backtest: {}
""",
                encoding="utf-8",
            )
            output = root / "out.yaml"
            patch_yaml(
                str(template),
                str(output),
                {
                    "train_start": "2021-10-18",
                    "train_end": "2026-10-16",
                    "valid_start": "2026-09-18",
                    "valid_end": "2026-10-16",
                    "test_start": "2026-10-20",
                    "test_end": "2026-10-20",
                },
                model_mode="default",
                allow_valid_inside_train=True,
            )
            written = output.read_text(encoding="utf-8")
            self.assertIn("2026-10-20", written)


if __name__ == "__main__":
    unittest.main()
