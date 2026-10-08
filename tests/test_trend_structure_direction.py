"""方向判断和判断方法接口的验收。合成收盘序列，不访问网络。"""
from __future__ import annotations

import math
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from analysis.structure.config import (
    PARAM_VERSION,
    PARAM_VERSION_V1,
    PARAM_VERSION_V2,
    SCALE_NAMES,
    DirectionParams,
    default_params,
)
from analysis.structure.context import build_context
from analysis.structure.html_report import render_report
from analysis.structure.methods.base import (
    KIND_INSTANCE,
    KIND_RUN,
    MARK_KINDS,
    CategorySpec,
    MethodResult,
    make_mark,
)
from analysis.structure.methods.direction import DirectionMethod, _volatility_at
from analysis.structure.methods.registry import run_methods, validate_registry
from analysis.structure.pivots import detect_pivots
from analysis.structure.series import build_series
from analysis.structure.snapshot import build_snapshot
from analysis.structure.view_model import build_view_model, display_snapshot
from analysis.structure.volatility import compute_volatility

_SWAP = {
    "uptrend": "downtrend",
    "downtrend": "uptrend",
    "range": "range",
    "": "",
    "undecided": "undecided",
    "higher": "lower",
    "lower": "higher",
    "flat": "flat",
    "unknown": "unknown",
    "widening": "narrowing",
    "narrowing": "widening",
    "mixed": "mixed",
    "trend_break": "trend_break",
    "stall": "stall",
    "high": "low",
    "low": "high",
    "pivot": "pivot",
    "cross": "cross",
    "progress": "progress",
    "gap": "gap",
    None: None,
}


def _times(count, extra_days=None):
    extra_days = extra_days or {}
    cursor = datetime(2024, 1, 2, 16, 0)
    times = []
    for index in range(count):
        if index:
            cursor = cursor + timedelta(days=1 + extra_days.get(index, 0))
        times.append(cursor)
    return times


def _plateaus(pairs):
    levels = [0.0, 0.001]
    levels.extend([0.001] * 28)
    for value, count in pairs:
        levels.extend([value] * count)
    return levels


def _closes(levels):
    return [100.0 * math.exp(value) for value in levels]


def _params(axis="log", stall=None):
    params = default_params(axis, version=PARAM_VERSION_V2)
    if stall is not None:
        params = replace(params, direction=DirectionParams(stall_bars=stall))
    return params


def _snapshot(levels, axis="log", stall=None, extra_days=None, version=PARAM_VERSION_V2):
    closes = _closes(levels)
    params = _params(axis, stall) if version == PARAM_VERSION_V2 else default_params(axis, version=version)
    series = build_series(
        _times(len(closes), extra_days),
        closes,
        interval="1d",
        params=params,
    )
    snapshot = build_snapshot(series, symbol="DIR", engine_version="test")
    return snapshot, series


def _short(snapshot):
    return [
        item
        for item in snapshot["methods"]["direction"]["objects"]
        if item["scale"] == "short"
    ]


def _signature(segment):
    origin = segment.get("structure_origin") or {}
    return (
        segment["start_index"],
        segment["end_index"],
        segment.get("category") or "",
        segment.get("structure"),
        segment.get("trigger"),
        segment.get("cause"),
        origin.get("role"),
        origin.get("extreme_index"),
    )


def _swapped(signature):
    start, end, category, structure, trigger, cause, role, index = signature
    return (
        start,
        end,
        _SWAP[category],
        _SWAP[structure],
        _SWAP[trigger],
        _SWAP[cause],
        _SWAP[role],
        index,
    )


class DirectionAcceptanceTests(unittest.TestCase):
    def test_rising_highs_and_lows_become_an_uptrend(self):
        levels = _plateaus([(0.05, 4), (0.01, 4), (0.09, 4), (0.04, 4), (0.13, 4)])
        for axis in ("log", "uniform"):
            with self.subTest(axis=axis):
                snapshot, series = _snapshot(levels, axis)
                segments = _short(snapshot)
                up = next(item for item in segments if item["category"] == "uptrend")
                self.assertEqual(up["structure"], "uptrend")
                origin = up["structure_origin"]
                self.assertEqual(origin["role"], "low")
                lows = [
                    pivot
                    for pivot in detect_pivots(series).confirmed
                    if pivot.scale == "short" and pivot.role == "low" and pivot.extreme_index <= up["start_index"]
                ]
                self.assertEqual(origin["extreme_index"], min(pivot.extreme_index for pivot in lows))
                first = up["protection"][0]
                self.assertEqual(first["source"], "inherit")
                self.assertEqual(first["available_index"], up["start_index"])
                protected = next(
                    pivot
                    for pivot in detect_pivots(series).confirmed
                    if pivot.scale == "short"
                    and pivot.role == "low"
                    and pivot.extreme_index == first["pivot"]["extreme_index"]
                )
                self.assertEqual(first["geometric_price"], protected.geometric_price)
                self.assertGreater(first["pivot"]["extreme_index"], origin["extreme_index"])
                mirrored, _mirror_series = _snapshot([-value for value in levels], axis)
                self.assertEqual(
                    [_swapped(_signature(item)) for item in _short(snapshot)],
                    [_signature(item) for item in _short(mirrored)],
                )

    def test_failure_swing_turns_directly_into_a_downtrend(self):
        levels = _plateaus(
            [(0.05, 4), (0.00, 4), (0.09, 4), (0.03, 4), (0.07, 4), (0.06, 4), (0.04, 2), (0.00, 4)]
        )
        for axis in ("log", "uniform"):
            with self.subTest(axis=axis):
                snapshot, _series = _snapshot(levels, axis)
                segments = _short(snapshot)
                up = next(item for item in segments if item["category"] == "uptrend")
                down = next(item for item in segments if item["category"] == "downtrend")
                self.assertEqual(down["start_index"], up["end_index"] + 1)
                self.assertEqual(down["trigger"], "cross")
                self.assertIsNone(down["cause"])
                self.assertEqual(down["structure_origin"]["role"], "high")
                self.assertEqual(up["stats"]["failed_swing_count"], 1)
                event = next(
                    item
                    for item in snapshot["events"]
                    if item.get("object_id") == down["id"]
                )
                self.assertEqual(event["event_index"], down["start_index"])
                self.assertEqual(event["price_cross_index"], down["start_index"] - 1)
                self.assertEqual(event["label_before"], "uptrend")
                self.assertEqual(event["label_after"], "downtrend")
                mirrored, _mirror_series = _snapshot([-value for value in levels], axis)
                self.assertEqual(
                    [_swapped(_signature(item)) for item in segments],
                    [_signature(item) for item in _short(mirrored)],
                )

    def test_non_failure_breaks_into_a_range_before_the_downtrend(self):
        levels = _plateaus(
            [(0.05, 4), (0.00, 4), (0.09, 4), (0.03, 4), (0.13, 4), (0.08, 2), (0.00, 3), (0.06, 4), (0.02, 3)]
        )
        for axis in ("log", "uniform"):
            with self.subTest(axis=axis):
                snapshot, _series = _snapshot(levels, axis)
                segments = _short(snapshot)
                categories = [item.get("category") or "" for item in segments]
                self.assertIn("uptrend", categories)
                broken = next(item for item in segments if item.get("cause") == "trend_break")
                self.assertEqual(broken["category"], "range")
                self.assertEqual(broken["structure"], "range")
                self.assertEqual(broken["trigger"], "cross")
                later = next(item for item in segments if item["start_index"] > broken["start_index"])
                self.assertEqual(later["category"], "downtrend")
                self.assertEqual(later["trigger"], "pivot")
                mirrored, _mirror_series = _snapshot([-value for value in levels], axis)
                self.assertEqual(
                    [_swapped(_signature(item)) for item in segments],
                    [_signature(item) for item in _short(mirrored)],
                )

    def test_cross_needs_a_strict_buffer_and_a_full_run(self):
        for axis in ("log", "uniform"):
            with self.subTest(axis=axis):
                base = _plateaus([(0.05, 4), (0.00, 4)])
                prefix, _series = _snapshot(base, axis)
                high = [
                    pivot
                    for pivot in detect_pivots(_series).confirmed
                    if pivot.scale == "short" and pivot.role == "high"
                ][-1]
                epsilon = 0.5
                span = epsilon * high.volatility
                equal = _close_not_above(high.geometric_price, span, axis)
                over = _close_above(high.geometric_price, span, axis)
                self.assertEqual(_crosses(_extend(base, [equal] * 3, axis), high), [])
                self.assertEqual(_crosses(_extend(base, [over], axis), high), [])
                done = _crosses(_extend(base, [over, over], axis), high)
                self.assertEqual(len(done), 1)
                self.assertEqual(done[0]["result"], "higher")
                self.assertEqual(done[0]["available_index"], len(base) + 1)
                self.assertEqual(done[0]["price_cross_index"], len(base))

    def test_equal_epsilon_is_flat_and_a_little_more_is_higher(self):
        for axis in ("log", "uniform"):
            with self.subTest(axis=axis):
                base = _plateaus([(0.05, 4), (0.00, 4)])
                _prefix, series = _snapshot(base, axis)
                high = [
                    pivot
                    for pivot in detect_pivots(series).confirmed
                    if pivot.scale == "short" and pivot.role == "high"
                ][-1]
                epsilon = 0.5
                span = epsilon * high.volatility
                flat_close = _close_not_above(high.geometric_price, span, axis)
                higher_close = _close_above(high.geometric_price, span, axis)
                drop = _close_for(high.geometric_price - high.threshold * 3, axis)
                flat = _high_compare(_extend(base, [flat_close] * 4 + [drop] * 4, axis))
                higher = _high_compare(_extend(base, [higher_close] * 4 + [drop] * 4, axis))
                self.assertEqual(flat["result"], "flat")
                self.assertLessEqual(flat["d"], epsilon)
                self.assertGreater(flat["d"], epsilon - 1e-6)
                self.assertNotIn("uptrend", _categories(_extend(base, [flat_close] * 4 + [drop] * 4, axis)))
                self.assertEqual(higher["result"], "higher")
                self.assertGreater(higher["d"], epsilon)

    def test_a_later_confirmation_stays_higher_after_the_cross(self):
        levels = _plateaus([(0.05, 4), (0.01, 4), (0.09, 4), (0.04, 4), (0.13, 4), (0.08, 4), (0.17, 4)])
        for axis in ("log", "uniform"):
            with self.subTest(axis=axis):
                snapshot, _series = _snapshot(levels, axis)
                rows = [
                    row
                    for row in snapshot["methods"]["direction"]["extra"]["swing_comparisons"]
                    if row["scale"] == "short" and row["side"] == "high"
                ]
                pairs = []
                for row in rows:
                    if row["mode"] != "cross" or row["result"] != "higher":
                        continue
                    later = [
                        item
                        for item in rows
                        if item["mode"] == "pivot"
                        and item["reference"] == row["reference"]
                        and item["available_index"] >= row["available_index"]
                    ]
                    if later:
                        pairs.append(later[0])
                self.assertTrue(pairs)
                self.assertTrue(all(item["result"] == "higher" for item in pairs))

    def test_stall_relabels_without_changing_structure_until_price_moves_or_breaks(self):
        base = _plateaus([(0.05, 4), (0.00, 4), (0.09, 4), (0.03, 4), (0.12, 4)])
        stall = (6, 20, 40)
        for axis in ("log", "uniform"):
            with self.subTest(axis=axis):
                flat, _series = _snapshot(base + [0.12] * 20, axis, stall=stall)
                up, stalled = _short(flat)[-2:]
                self.assertEqual(up["category"], "uptrend")
                self.assertEqual(stalled["category"], "range")
                self.assertEqual(stalled["structure"], "uptrend")
                self.assertEqual(stalled["cause"], "stall")
                self.assertEqual(stalled["start_index"], up["start_index"] + 5)
                resumed, _resumed_series = _snapshot(base + [0.12] * 8 + [0.30], axis, stall=stall)
                self.assertEqual(_short(resumed)[-1]["trigger"], "progress")
                self.assertEqual(_short(resumed)[-1]["category"], "uptrend")
                during, _during_series = _snapshot(base + [0.12, -0.05, -0.05], axis, stall=stall)
                self.assertEqual(_short(during)[-2]["cause"], "stall")
                self.assertEqual(_short(during)[-1]["cause"], "trend_break")
                self.assertEqual(_short(during)[-1]["structure"], "range")

    def test_progress_on_the_stall_bar_keeps_the_uptrend_and_a_reversal_does_not_also_stall(self):
        base = _plateaus([(0.05, 4), (0.00, 4), (0.09, 4), (0.03, 4), (0.12, 4)])
        stall = (6, 20, 40)
        failure = _plateaus(
            [(0.05, 4), (0.00, 4), (0.09, 4), (0.03, 4), (0.07, 4), (0.06, 4), (0.04, 2), (0.00, 4)]
        )
        for axis in ("log", "uniform"):
            with self.subTest(axis=axis):
                jumped, _series = _snapshot(base + [0.12, 0.30, 0.30], axis, stall=stall)
                self.assertNotIn("stall", [item.get("cause") for item in _short(jumped)])
                self.assertEqual(_short(jumped)[-1]["category"], "uptrend")
                self.assertGreater(_short(jumped)[-1]["stats"]["progress_count"], 0)
                turned, _turned_series = _snapshot(base + [-0.05, -0.05], axis, stall=stall)
                self.assertNotIn("stall", [item.get("cause") for item in _short(turned)])
                self.assertEqual(_short(turned)[-1]["cause"], "trend_break")
                direct, _direct_series = _snapshot(failure, axis, stall=(12, 40, 80))
                self.assertNotIn("stall", [item.get("cause") for item in _short(direct)])
                self.assertEqual(_short(direct)[-1]["category"], "downtrend")

    def test_gap_restarts_from_undecided_and_drops_old_pivots(self):
        levels = _plateaus([(0.05, 4), (0.01, 4), (0.09, 4), (0.04, 4), (0.13, 4), (0.08, 4)])
        for axis in ("log", "uniform"):
            with self.subTest(axis=axis):
                snapshot, series = _snapshot(levels, axis, extra_days={45: 8})
                gap = series.gap_indices[0]
                segments = _short(snapshot)
                restarted = next(item for item in segments if item.get("trigger") == "gap")
                self.assertEqual(restarted["start_index"], gap)
                self.assertEqual(restarted["structure"], "undecided")
                self.assertFalse(restarted.get("category"))
                for row in snapshot["methods"]["direction"]["extra"]["swing_comparisons"]:
                    if row["scale"] != "short" or row["available_index"] < gap:
                        continue
                    self.assertGreaterEqual(row["reference"]["extreme_index"], gap)
                    if row.get("subject_pivot"):
                        self.assertGreaterEqual(row["subject_pivot"]["extreme_index"], gap)
                for item in segments:
                    self.assertFalse(item["start_index"] < gap <= item["end_index"])

    def test_missing_volatility_uses_the_floor(self):
        levels = _plateaus([(0.05, 4), (0.00, 4), (0.09, 4), (0.03, 4), (0.12, 4)]) + [0.12] * 4
        for axis in ("log", "uniform"):
            with self.subTest(axis=axis):
                _snapshot_value, series = _snapshot(levels, axis, stall=(30, 60, 125))
                floor = series.params.volatility_floor
                volatility = compute_volatility(series)
                self.assertIsNone(volatility[0])
                self.assertEqual(_volatility_at(volatility, 0, floor), floor)
                blank = tuple(None for _value in volatility)
                quiet = _judge(series, blank)
                bumped_series = _bump(series, axis)
                bumped = _judge(bumped_series, blank + (None,))
                quiet_up = [item for item in quiet.objects if item["scale"] == "short" and item["structure"] == "uptrend"]
                bumped_up = [item for item in bumped.objects if item["scale"] == "short" and item["structure"] == "uptrend"]
                self.assertEqual(sum(item["stats"]["progress_count"] for item in quiet_up), 0)
                self.assertGreater(sum(item["stats"]["progress_count"] for item in bumped_up), 0)

    def test_new_uptrend_records_the_current_low_as_protection(self):
        levels = _plateaus([(0.05, 4), (0.01, 4), (0.09, 4), (0.04, 4)])
        snapshot, series = _snapshot(levels)
        up = next(item for item in _short(snapshot) if item["category"] == "uptrend")
        row = up["protection"][0]
        pivot = next(
            pivot
            for pivot in detect_pivots(series).confirmed
            if pivot.scale == "short" and pivot.extreme_index == row["pivot"]["extreme_index"]
        )
        self.assertEqual(row["source"], "inherit")
        self.assertEqual(row["geometric_price"], pivot.geometric_price)
        self.assertEqual(row["pivot"]["role"], "low")


class DirectionContractTests(unittest.TestCase):
    def test_prefix_matches_the_as_of_view(self):
        levels = _plateaus(
            [(0.05, 4), (0.00, 4), (0.09, 4), (0.03, 4), (0.13, 4), (0.08, 2), (0.00, 3), (0.06, 4)]
        )
        for axis in ("log", "uniform"):
            snapshot, series = _snapshot(levels, axis)
            interesting = {1, len(series) // 2, len(series) - 1}
            for pivot in detect_pivots(series).confirmed:
                if pivot.scale == "short":
                    interesting.add(pivot.confirm_index + 1)
            for event in snapshot["events"]:
                if event.get("object_kind") == "direction":
                    interesting.add(int(event["available_index"]) + 1)
            for count in sorted(index for index in interesting if 1 <= index <= len(series)):
                with self.subTest(axis=axis, count=count):
                    prefix_snapshot, prefix = _snapshot(levels[:count], axis)
                    shown = _as_of(snapshot, prefix.bars[-1].timestamp)
                    self._assert_prefix(prefix_snapshot, shown, count)

    def test_objects_are_available_after_their_evidence_and_do_not_cross_gaps(self):
        levels = _plateaus([(0.05, 4), (0.01, 4), (0.09, 4), (0.04, 4), (0.13, 4)])
        snapshot, series = _snapshot(levels, extra_days={40: 8})
        gaps = set(series.gap_indices)
        pivots = {
            (pivot.scale, pivot.role, pivot.extreme_index): pivot
            for pivot in detect_pivots(series).confirmed
        }
        for segment in snapshot["methods"]["direction"]["objects"]:
            self.assertFalse(any(segment["start_index"] < gap <= segment["end_index"] for gap in gaps))
            for ref in segment.get("evidence") or []:
                pivot = pivots[(ref["scale"], ref["role"], ref["extreme_index"])]
                self.assertGreaterEqual(segment["available_index"], pivot.confirm_index)
            for event in snapshot["events"]:
                if event.get("object_id") == segment["id"]:
                    self.assertGreaterEqual(event["available_index"], segment["available_index"])

    def test_direction_does_not_recompute_pivots_or_boundaries(self):
        levels = _plateaus([(0.05, 4), (0.01, 4), (0.09, 4)])
        _snapshot_value, series = _snapshot(levels)
        context = _context(series)
        modules = (
            "analysis.structure.pivots",
            "analysis.structure.boundaries",
            "analysis.structure.zones",
            "analysis.structure.lifecycle",
            "analysis.structure.segments",
        )
        originals = {}
        for name in modules:
            module = __import__(name, fromlist=["detect_pivots"])
            for attr in ("detect_pivots", "build_boundaries", "build_zones", "build_lifecycle", "build_segments"):
                if hasattr(module, attr):
                    originals[(name, attr)] = getattr(module, attr)
                    setattr(module, attr, _refuse)
        try:
            result = DirectionMethod().judge(context, series.params.direction.to_dict(), {})
        finally:
            for (name, attr), original in originals.items():
                setattr(__import__(name, fromlist=[attr]), attr, original)
        self.assertTrue(result.objects)

    def test_segments_cover_every_bar_once(self):
        levels = _plateaus([(0.05, 4), (0.01, 4), (0.09, 4), (0.04, 4)])
        snapshot, series = _snapshot(levels, extra_days={20: 8})
        for scale in SCALE_NAMES:
            segments = [
                item
                for item in snapshot["methods"]["direction"]["objects"]
                if item["scale"] == scale
            ]
            self.assertEqual(segments[0]["start_index"], 0)
            self.assertEqual(segments[-1]["end_index"], len(series) - 1)
            for left, right in zip(segments, segments[1:]):
                self.assertEqual(right["start_index"], left["end_index"] + 1)
            self.assertEqual(segments[-1]["status"], "open")
            self.assertTrue(all(item["status"] == "closed" for item in segments[:-1]))

    def test_both_axes_render_and_v0_has_no_direction(self):
        levels = _plateaus([(0.05, 4), (0.01, 4), (0.09, 4)])
        for axis in ("log", "uniform"):
            snapshot, _series = _snapshot(levels, axis)
            html = render_report(snapshot)
            self.assertIn("方向", html)
            self.assertIn("method-direction", html)
            model = build_view_model(snapshot)
            self.assertTrue(model["marks"])
            self.assertTrue(all(mark["kind"] in MARK_KINDS for mark in model["marks"]))
        for version in (PARAM_VERSION, PARAM_VERSION_V1):
            snapshot, _series = _snapshot(levels, version=version)
            self.assertNotIn("methods", snapshot)
            self.assertNotIn("direction", snapshot["params"])
            self.assertNotIn("visible_methods", snapshot["identity"])
            html = render_report(snapshot)
            self.assertNotIn('id="method-direction"', html)

    def test_hand_built_snapshot_keeps_every_table_row(self):
        snapshot = _hand_direction_snapshot(3)
        model = build_view_model(snapshot)
        self.assertEqual(len(model["method_objects"]["direction"]), 3)
        html = render_report(snapshot)
        table = html.split('id="method-direction-segments"', 1)[1].split("</table>", 1)[0]
        self.assertEqual(table.count("data-object-id"), 3)
        snapshot["identity"]["as_of"] = "2024-01-08T16:00:00"
        shown = display_snapshot(snapshot)
        self.assertEqual(len(shown["methods"]["direction"]["objects"]), 2)

    def test_categories_and_marks_stay_inside_the_declaration(self):
        method = DirectionMethod()
        declared = {item.key for item in method.categories}
        levels = _plateaus([(0.05, 4), (0.01, 4), (0.09, 4), (0.04, 4), (0.13, 4)])
        snapshot, _series = _snapshot(levels)
        model = build_view_model(snapshot)
        for item in snapshot["methods"]["direction"]["objects"]:
            if item.get("category"):
                self.assertIn(item["category"], declared)
        for mark in model["marks"]:
            self.assertIn(mark["kind"], MARK_KINDS)

    def _assert_prefix(self, prefix, shown, count):
        for scale in SCALE_NAMES:
            left = _scale_objects(prefix, scale)
            right = _scale_objects(shown, scale)
            self.assertEqual([item["id"] for item in left], [item["id"] for item in right])
            stop = count - 1
            for item, other in zip(left, right):
                self.assertEqual(_frozen(item), _frozen(other))
                self.assertEqual(item["status"], other["status"])
                self.assertEqual(item["end_index"], other["end_index"])
                self.assertEqual(_rows_until(item.get("protection"), stop), _rows_until(other.get("protection"), stop))
        self.assertEqual(
            _rows_until(prefix["methods"]["direction"]["extra"]["swing_comparisons"], count - 1),
            _rows_until(shown["methods"]["direction"]["extra"]["swing_comparisons"], count - 1),
        )
        self.assertEqual(
            [event for event in prefix["events"] if event.get("object_kind") == "direction"],
            [event for event in shown["events"] if event.get("object_kind") == "direction"],
        )


class MethodInterfaceTests(unittest.TestCase):
    def test_registry_rejects_duplicates_missing_inputs_and_cycles(self):
        direction = DirectionMethod()
        with self.assertRaises(ValueError):
            validate_registry((direction, direction))
        missing = _method("orphan", requires=("not-a-real-input",))
        with self.assertRaises(ValueError):
            validate_registry((missing,))
        left = _method("left", requires=("right",))
        right = _method("right", requires=("left",))
        with self.assertRaises(ValueError):
            validate_registry((left, right))

    def test_one_method_failing_leaves_the_others_and_the_base_fields(self):
        levels = _plateaus([(0.05, 4), (0.01, 4)])
        _snapshot_value, series = _snapshot(levels)
        context = _context(series)
        outcome = run_methods(
            context,
            series.params,
            ("boom", "direction"),
            methods=(_method("boom", raises=True), DirectionMethod()),
        )
        self.assertEqual(outcome["records"]["boom"]["status"], "failed")
        self.assertEqual(outcome["records"]["direction"]["status"], "completed")
        self.assertTrue(outcome["records"]["direction"]["objects"])
        original = DirectionMethod.judge
        DirectionMethod.judge = lambda self, context, params, upstream: (_ for _ in ()).throw(RuntimeError("故意失败"))
        try:
            snapshot, _series = _snapshot(levels)
        finally:
            DirectionMethod.judge = original
        self.assertEqual(snapshot["methods"]["direction"]["status"], "failed")
        self.assertIn("pivots", snapshot)
        self.assertIn("boundaries", snapshot)
        self.assertNotIn("direction_change", [item.get("event_type") for item in snapshot["events"]])

    def test_fixture_method_emits_every_mark_kind(self):
        method = _fixture_method()
        record = {
            "id": "fixture:mid:0",
            "method": "fixture",
            "scale": "mid",
            "category": "demo",
            "status": "validated",
            "start_index": 0,
            "end_index": 4,
            "available_index": 0,
        }
        marks = method.present(record, {"price_axis": "log"})
        self.assertEqual(sorted(mark["kind"] for mark in marks), sorted(MARK_KINDS))
        chart = (
            Path(__file__).resolve().parents[1] / "analysis/structure/assets/structure-chart.js"
        ).read_text(encoding="utf-8")
        for kind in MARK_KINDS:
            self.assertIn(kind, chart)


def _close_for(geometric, axis):
    if axis == "log":
        return math.exp(geometric)
    return geometric


def _geometric(close, axis):
    if axis == "log":
        return math.log(close)
    return close


def _close_not_above(reference_geometric, span, axis):
    """几何差不超过 span。浮点把「刚好等于」抬过线时，退到仍不超过的那一档。"""
    close = _close_for(reference_geometric + span, axis)
    while _geometric(close, axis) - reference_geometric > span:
        close = math.nextafter(close, 0.0)
    return close


def _close_above(reference_geometric, span, axis):
    close = _close_for(reference_geometric + span, axis)
    while _geometric(close, axis) - reference_geometric <= span:
        close = math.nextafter(close, math.inf)
    return close


def _extend(base_levels, closes, axis):
    base_closes = _closes(base_levels)
    params = _params(axis)
    series = build_series(
        _times(len(base_closes) + len(closes)),
        base_closes + list(closes),
        interval="1d",
        params=params,
    )
    return build_snapshot(series, symbol="DIR", engine_version="test")


def _crosses(snapshot, high):
    return [
        row
        for row in snapshot["methods"]["direction"]["extra"]["swing_comparisons"]
        if row["scale"] == "short"
        and row["mode"] == "cross"
        and row["reference"]["extreme_index"] == high.extreme_index
    ]


def _high_compare(snapshot):
    rows = [
        row
        for row in snapshot["methods"]["direction"]["extra"]["swing_comparisons"]
        if row["scale"] == "short" and row["side"] == "high" and row["mode"] == "pivot"
    ]
    return rows[-1]


def _categories(snapshot):
    return [item.get("category") or "" for item in _short(snapshot)]


def _context(series):
    pivots = detect_pivots(series)
    return build_context(
        series,
        volatility=compute_volatility(series),
        pivots=pivots,
        segments=(),
        boundaries=(),
        zones=(),
        events=(),
    )


def _judge(series, volatility):
    pivots = detect_pivots(series)
    context = build_context(
        series,
        volatility=volatility,
        pivots=pivots,
        segments=(),
        boundaries=(),
        zones=(),
        events=(),
    )
    return DirectionMethod().judge(context, series.params.direction.to_dict(), {})


def _bump(series, axis):
    closes = [bar.close for bar in series.bars]
    last = closes[-1]
    if axis == "log":
        closes.append(last * math.exp(0.02))
    else:
        closes.append(last + 2.0)
    return build_series(
        _times(len(closes)),
        closes,
        interval="1d",
        params=series.params,
    )


def _as_of(snapshot, when):
    copied = dict(snapshot)
    identity = dict(snapshot["identity"])
    identity["as_of"] = when.isoformat()
    copied["identity"] = identity
    return display_snapshot(copied)


def _scale_objects(snapshot, scale):
    return [
        item
        for item in snapshot["methods"]["direction"]["objects"]
        if item["scale"] == scale
    ]


def _frozen(segment):
    hidden = {"status", "end_index", "end_time", "stats", "protection"}
    return {key: value for key, value in segment.items() if key not in hidden}


def _rows_until(rows, stop):
    return [row for row in rows or [] if int(row.get("available_index") or 0) <= stop]


def _refuse(*_args, **_kwargs):
    raise AssertionError("方向判断不应重算拐点或边界")


def _method(key, requires=("closes",), raises=False):
    class _Fake:
        version = "fake-v1"
        kind = KIND_RUN
        categories = ()
        replaces = ()

        def default_params(self):
            return {}

        def check_params(self, params):
            return None

        def judge(self, context, params, upstream):
            if raises:
                raise RuntimeError("故意失败")
            return MethodResult()

        def present(self, record, view):
            return []

        def tables(self):
            return ()

        def samples(self):
            return ()

    fake = _Fake()
    fake.key = key
    fake.name = key
    fake.section = "test"
    fake.requires = requires
    return fake


def _fixture_method():
    class _Fixture:
        key = "fixture"
        name = "检查"
        section = "5"
        version = "fixture-v1"
        kind = KIND_INSTANCE
        requires = ("closes",)
        categories = (CategorySpec("demo", "示例", "#7EB6FF"),)
        replaces = ()

        def default_params(self):
            return {}

        def check_params(self, params):
            return None

        def judge(self, context, params, upstream):
            return MethodResult()

        def present(self, record, view):
            shared = dict(
                object_id=record["id"],
                method=self.key,
                scale=record["scale"],
                category=record["category"],
                color="#7EB6FF",
                legend="检查 · 中尺度",
                start_index=0,
                end_index=4,
                available_index=0,
            )
            return [
                make_mark(kind="label_bar", anchor="lane", **shared),
                make_mark(kind="band", points=[[0, 100], [4, 101], [4, 110], [0, 109]], **shared),
                make_mark(kind="segment", points=[[0, 100], [4, 110]], **shared),
                make_mark(kind="polyline", points=[[0, 100], [2, 105], [4, 103]], **shared),
                make_mark(kind="step_line", points=[[0, 100], [4, 100]], line_type="dashed", **shared),
                make_mark(kind="marker", anchor="price", index=2, price=105, **shared),
                make_mark(kind="text", text="示例", points=[[2, 108]], **shared),
            ]

        def tables(self):
            return ()

        def samples(self):
            return ()

    return _Fixture()


def _hand_direction_snapshot(count):
    params = default_params("log", version=PARAM_VERSION_V2)
    objects = []
    closes = []
    for index in range(count):
        start = index * 5
        objects.append(
            {
                "id": f"direction:mid:{start}",
                "method": "direction",
                "scale": "mid",
                "category": "uptrend",
                "status": "closed" if index < count - 1 else "open",
                "start_index": start,
                "end_index": start + 4,
                "start_time": f"2024-01-{2 + start:02d}T16:00:00",
                "end_time": f"2024-01-{6 + start:02d}T16:00:00",
                "available_index": start,
                "available_time": f"2024-01-{2 + start:02d}T16:00:00",
                "structure": "uptrend",
                "trigger": "pivot",
                "cause": None,
                "protection": [],
                "stats": {"bar_count": 5},
            }
        )
    for index in range(20):
        closes.append(
            {
                "index": index,
                "time": f"2024-01-{2 + index:02d}T16:00:00",
                "close": 100.0 + index,
                "gap_before": False,
            }
        )
    return {
        "kind": "trend_structure_snapshot",
        "identity": {
            "symbol": "HAND",
            "interval": "1d",
            "price_axis": "log",
            "price_axis_label": "对数价格",
            "param_version": PARAM_VERSION_V2,
            "as_of": None,
            "visible_scales": ["mid"],
            "visible_methods": ["direction"],
            "bar_count": 20,
        },
        "reading": "",
        "scale_definition_note": "",
        "quality": {"bar_count": 20, "gap_count": 0, "gaps": [], "grade": "正常", "grade_note": ""},
        "volatility": {"scales": [], "unit": "对数收益"},
        "pivot_stats": [],
        "pivots": [],
        "temporary_pivots": [],
        "segments": [],
        "temporary_segments": [],
        "boundaries": [],
        "zones": [],
        "events": [],
        "relations": {},
        "closes": closes,
        "params": params.to_dict(),
        "methods": {
            "direction": {
                "key": "direction",
                "name": "方向",
                "version": "direction-v1",
                "kind": "run",
                "section": "2.1",
                "requires": ["closes", "pivots", "volatility"],
                "categories": [{"key": "uptrend", "name": "上升趋势", "color": "#E85D4C"}],
                "replaces": [],
                "params": params.direction.to_dict(),
                "status": "completed",
                "reason": "",
                "objects": objects,
                "extra": {"swing_comparisons": []},
                "notes": [],
            }
        },
    }


if __name__ == "__main__":
    unittest.main()
