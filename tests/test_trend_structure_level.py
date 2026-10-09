"""级别：主趋势、次级折返、短期波动。合成序列，不访问网络。"""
from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta

from analysis.structure.config import (
    PARAM_VERSION,
    PARAM_VERSION_V4,
    PARAM_VERSION_V5,
    default_params,
)
from analysis.structure.context import build_context
from analysis.structure.html_report import render_report
from analysis.structure.methods.base import MARK_KINDS
from analysis.structure.methods.direction import DirectionMethod
from analysis.structure.methods.major_trend import MajorTrendMethod
from analysis.structure.methods.minor_swing import MinorSwingMethod
from analysis.structure.methods.retracement import RetracementMethod
from analysis.structure.pivots import ROLE_HIGH, ROLE_LOW, Pivot
from analysis.structure.segments import DIRECTION_DOWN, DIRECTION_FLAT, DIRECTION_UP, Segment, SegmentResult
from analysis.structure.series import build_series
from analysis.structure.snapshot import build_snapshot
from analysis.structure.view_model import build_view_model, display_snapshot
from analysis.structure.volatility import compute_volatility

_LEVELS = ("major_trend", "retracement", "minor_swing")
_SWAP = {
    "uptrend": "downtrend",
    "downtrend": "uptrend",
    "range": "range",
    "": "",
    "undecided": "undecided",
    "up": "down",
    "down": "up",
    "flat": "flat",
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


def _plateau():
    levels = [0.0, 0.001]
    levels.extend([0.001] * 28)
    for value, count in ((0.05, 4), (0.01, 4), (0.09, 4), (0.04, 4), (0.13, 4)):
        levels.extend([value] * count)
    return levels


def _closes(levels):
    return [100.0 * math.exp(value) for value in levels]


def _series_from_levels(levels, axis="log", extra_days=None, version=PARAM_VERSION_V5, quoted=False):
    closes = _closes(levels)
    params = default_params(axis, version=version)
    highs = [price * 1.01 for price in closes] if quoted else None
    lows = [price * 0.99 for price in closes] if quoted else None
    volumes = [1000.0] * len(closes) if quoted else None
    return build_series(
        _times(len(closes), extra_days),
        closes,
        interval="1d",
        params=params,
        highs=highs,
        lows=lows,
        volumes=volumes,
    )


def _flat_series(count, gap_at=None):
    extra = {gap_at: 8} if gap_at else None
    return build_series(
        _times(count, extra),
        [100.0] * count,
        interval="1d",
        params=default_params("log", version=PARAM_VERSION_V5),
    )


def _context(series, segments=()):
    count = len(series)
    volatility = tuple(None if index == 0 else 0.01 for index in range(count))
    return build_context(
        series,
        volatility=volatility,
        pivots=(),
        segments=SegmentResult(tuple(segments), ()),
        boundaries=(),
        zones=(),
        events=(),
    )


def _pivot(scale, role, extreme, confirm, geometric):
    extreme_time = datetime(2024, 1, 2) + timedelta(days=extreme)
    confirm_time = datetime(2024, 1, 2) + timedelta(days=confirm)
    return Pivot(
        scale=scale,
        role=role,
        extreme_index=extreme,
        confirm_index=confirm,
        extreme_time=extreme_time,
        confirm_time=confirm_time,
        available_time=confirm_time,
        price=100.0 + geometric,
        geometric_price=geometric,
        threshold=0.02,
        volatility=0.01,
        threshold_index=extreme,
        segment_start_index=0,
    )


def _segment(start, end, direction):
    change = end.geometric_price - start.geometric_price
    span = end.extreme_index - start.extreme_index
    return Segment(
        id=f"{end.scale}:{start.extreme_index}-{end.extreme_index}",
        scale=end.scale,
        direction=direction,
        start=start,
        end=end,
        bar_span=span,
        change=change,
        change_multiple=change / end.volatility,
        slope=change / span,
        max_deviation=0.0,
        adverse_excursion=0.0,
        volatility=end.volatility,
        confirm_time=end.confirm_time,
        available_time=end.available_time,
    )


def _direction_segment(start, end, structure, category, cause=None):
    return {
        "id": f"direction:long:{start}",
        "scale": "long",
        "structure": structure,
        "category": category,
        "cause": cause,
        "start_index": start,
        "end_index": end,
        "available_index": start,
    }


def _major(start, end, category):
    return {
        "id": f"major_trend:long:{start}",
        "scale": "long",
        "category": category,
        "structure": category or "undecided",
        "start_index": start,
        "end_index": end,
        "available_index": start,
        "status": "open" if end == start else "closed",
    }


class MajorTrendTests(unittest.TestCase):
    def test_stall_does_not_split_and_a_structure_change_does(self):
        series = _flat_series(25)
        upstream = {
            "direction": {
                "status": "completed",
                "objects": [
                    _direction_segment(0, 9, "uptrend", "uptrend"),
                    _direction_segment(10, 19, "uptrend", "range", cause="stall"),
                    _direction_segment(20, 24, "downtrend", "downtrend"),
                    {
                        "id": "direction:short:0",
                        "scale": "short",
                        "structure": "range",
                        "category": "range",
                        "start_index": 0,
                        "end_index": 24,
                        "available_index": 0,
                    },
                ],
            }
        }
        result = MajorTrendMethod().judge(_context(series), {}, upstream)
        self.assertEqual(
            [(item["start_index"], item["end_index"], item["category"], item["structure"]) for item in result.objects],
            [(0, 19, "uptrend", "uptrend"), (20, 24, "downtrend", "downtrend")],
        )
        self.assertEqual(result.objects[0]["direction_id"], "direction:long:0")
        self.assertEqual(result.objects[0]["evidence"], [{"object_kind": "direction", "id": "direction:long:0"}])
        self.assertEqual(result.objects[-1]["status"], "open")
        self.assertEqual(result.objects[0]["status"], "closed")
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0]["event_type"], "major_trend_change")
        self.assertEqual(result.events[0]["structure_before"], "uptrend")
        self.assertEqual(result.events[0]["structure_after"], "downtrend")
        self.assertEqual(result.events[0]["trigger"], "structure")
        self.assertIsNone(result.objects[0]["trigger"])

    def test_a_gap_splits_undecided_runs(self):
        series = _flat_series(16, gap_at=8)
        self.assertIn(8, series.gap_indices)
        upstream = {
            "direction": {
                "status": "completed",
                "objects": [
                    _direction_segment(0, 7, "undecided", ""),
                    _direction_segment(8, 15, "undecided", ""),
                ],
            }
        }
        result = MajorTrendMethod().judge(_context(series), {}, upstream)
        self.assertEqual([item["category"] for item in result.objects], ["", ""])
        self.assertEqual(result.objects[1]["start_index"], 8)
        self.assertEqual(result.objects[1]["trigger"], "gap")
        for item in result.objects:
            self.assertFalse(any(item["start_index"] < gap <= item["end_index"] for gap in series.gap_indices))
        covered = []
        for item in result.objects:
            covered.extend(range(item["start_index"], item["end_index"] + 1))
        self.assertEqual(covered, list(range(len(series))))

    def test_missing_direction_still_covers_every_bar(self):
        series = _flat_series(4)
        result = MajorTrendMethod().judge(_context(series), {}, {})
        self.assertEqual(len(result.objects), 1)
        self.assertEqual(result.objects[0]["category"], "")
        self.assertEqual(result.objects[0]["end_index"], 3)
        self.assertTrue(any("没有长尺度" in note for note in result.notes))

    def test_rejects_tunable_params(self):
        with self.assertRaises(ValueError):
            MajorTrendMethod().check_params({"epsilon": 0.5})


class RetracementTests(unittest.TestCase):
    def _pair(self, up_change, down_change):
        series = _flat_series(30)
        low = _pivot("mid", ROLE_LOW, 2, 4, 0.0)
        high = _pivot("mid", ROLE_HIGH, 8, 10, up_change)
        low_again = _pivot("mid", ROLE_LOW, 14, 16, up_change - down_change)
        rising = _segment(low, high, DIRECTION_UP)
        falling = _segment(high, low_again, DIRECTION_DOWN)
        return series, (rising, falling)

    def test_only_the_counter_trend_segment_is_kept_and_the_ratio_is_not_a_filter(self):
        series, segments = self._pair(2.0, 1.0)
        majors = [_major(0, len(series) - 1, "uptrend")]
        result = RetracementMethod().judge(
            _context(series, segments),
            {},
            {"major_trend": {"status": "completed", "objects": majors}},
        )
        self.assertEqual([item["id"] for item in result.objects], [f"retracement:{segments[1].id}"])
        item = result.objects[0]
        self.assertEqual(item["segment_direction"], "down")
        self.assertEqual(item["major_category"], "uptrend")
        self.assertAlmostEqual(item["retracement_ratio"], 0.5)
        self.assertEqual(item["dow_band"], "dow_inside")
        self.assertEqual(item["bars"], 6)
        self.assertEqual(item["calendar_days"], 6)
        self.assertEqual(item["status"], "validated")
        self.assertTrue(item["primary"])
        self.assertIsNone(item["alternate_of"])
        self.assertEqual(item["available_index"], 16)
        self.assertEqual([event["event_type"] for event in result.events], ["validated"])
        self.assertNotIn("breakout", [event["event_type"] for event in result.events])
        self.assertNotIn("expired", [event["event_type"] for event in result.events])

        wide, wide_segments = self._pair(10.0, 9.0)
        wide_result = RetracementMethod().judge(
            _context(wide, wide_segments),
            {},
            {"major_trend": {"status": "completed", "objects": [_major(0, len(wide) - 1, "uptrend")]}},
        )
        self.assertEqual(len(wide_result.objects), 1)
        self.assertAlmostEqual(wide_result.objects[0]["retracement_ratio"], 0.9)
        self.assertEqual(wide_result.objects[0]["dow_band"], "dow_outside")

        third, third_segments = self._pair(3.0, 1.0)
        third_result = RetracementMethod().judge(
            _context(third, third_segments),
            {},
            {"major_trend": {"status": "completed", "objects": [_major(0, len(third) - 1, "uptrend")]}},
        )
        self.assertEqual(third_result.objects[0]["dow_band"], "dow_inside")
        two_thirds, two_segments = self._pair(3.0, 2.0)
        two_result = RetracementMethod().judge(
            _context(two_thirds, two_segments),
            {},
            {"major_trend": {"status": "completed", "objects": [_major(0, len(two_thirds) - 1, "uptrend")]}},
        )
        self.assertEqual(two_result.objects[0]["dow_band"], "dow_inside")

    def test_range_flat_and_with_trend_segments_are_not_retracements(self):
        series, segments = self._pair(2.0, 1.0)
        flat = _segment(
            _pivot("mid", ROLE_HIGH, 18, 20, 1.0),
            _pivot("mid", ROLE_LOW, 22, 24, 1.0),
            DIRECTION_FLAT,
        )
        for category in ("range", ""):
            result = RetracementMethod().judge(
                _context(series, segments + (flat,)),
                {},
                {"major_trend": {"status": "completed", "objects": [_major(0, len(series) - 1, category)]}},
            )
            self.assertEqual(result.objects, [])
        downtrend = RetracementMethod().judge(
            _context(series, segments),
            {},
            {"major_trend": {"status": "completed", "objects": [_major(0, len(series) - 1, "downtrend")]}},
        )
        self.assertEqual([item["segment_direction"] for item in downtrend.objects], ["up"])
        self.assertEqual(downtrend.objects[0]["id"], f"retracement:{segments[0].id}")

    def test_ratio_stays_empty_without_a_prior_with_trend_leg_or_across_a_gap(self):
        series = _flat_series(24)
        only = _segment(
            _pivot("mid", ROLE_HIGH, 4, 6, 2.0),
            _pivot("mid", ROLE_LOW, 10, 12, 1.0),
            DIRECTION_DOWN,
        )
        result = RetracementMethod().judge(
            _context(series, (only,)),
            {},
            {"major_trend": {"status": "completed", "objects": [_major(0, 23, "uptrend")]}},
        )
        self.assertIsNone(result.objects[0]["retracement_ratio"])
        self.assertIsNone(result.objects[0]["dow_band"])

        gapped = _flat_series(30, gap_at=12)
        self.assertIn(12, gapped.gap_indices)
        prior = _segment(
            _pivot("mid", ROLE_LOW, 2, 4, 0.0),
            _pivot("mid", ROLE_HIGH, 8, 10, 2.0),
            DIRECTION_UP,
        )
        later = _segment(
            _pivot("mid", ROLE_HIGH, 14, 16, 2.0),
            _pivot("mid", ROLE_LOW, 18, 20, 1.0),
            DIRECTION_DOWN,
        )
        gapped_result = RetracementMethod().judge(
            _context(gapped, (prior, later)),
            {},
            {"major_trend": {"status": "completed", "objects": [_major(0, len(gapped) - 1, "uptrend")]}},
        )
        self.assertEqual(len(gapped_result.objects), 1)
        self.assertIsNone(gapped_result.objects[0]["retracement_ratio"])

        zero_prior = _segment(
            _pivot("mid", ROLE_LOW, 2, 4, 1.0),
            _pivot("mid", ROLE_HIGH, 8, 10, 1.0),
            DIRECTION_UP,
        )
        rebound = _segment(zero_prior.end, _pivot("mid", ROLE_LOW, 14, 16, 0.0), DIRECTION_DOWN)
        zero_result = RetracementMethod().judge(
            _context(series, (zero_prior, rebound)),
            {},
            {"major_trend": {"status": "completed", "objects": [_major(0, 23, "uptrend")]}},
        )
        self.assertEqual(len(zero_result.objects), 1)
        self.assertIsNone(zero_result.objects[0]["retracement_ratio"])

    def test_without_major_trend_there_are_no_objects(self):
        series, segments = self._pair(2.0, 1.0)
        result = RetracementMethod().judge(_context(series, segments), {}, {})
        self.assertEqual(result.objects, [])
        self.assertTrue(any("没有主趋势" in note for note in result.notes))


class MinorSwingTests(unittest.TestCase):
    def test_every_confirmed_short_segment_is_copied(self):
        series = _flat_series(30)
        rising = _segment(
            _pivot("short", ROLE_LOW, 2, 4, 0.0),
            _pivot("short", ROLE_HIGH, 6, 8, 1.0),
            DIRECTION_UP,
        )
        falling = _segment(rising.end, _pivot("short", ROLE_LOW, 11, 13, 0.4), DIRECTION_DOWN)
        flat = _segment(
            _pivot("short", ROLE_HIGH, 16, 18, 0.4),
            _pivot("short", ROLE_LOW, 20, 22, 0.4),
            DIRECTION_FLAT,
        )
        mid = _segment(
            _pivot("mid", ROLE_LOW, 2, 5, 0.0),
            _pivot("mid", ROLE_HIGH, 8, 11, 1.0),
            DIRECTION_UP,
        )
        result = MinorSwingMethod().judge(_context(series, (rising, falling, flat, mid)), {}, {})
        self.assertEqual(
            [item["id"] for item in result.objects],
            [f"minor_swing:{item.id}" for item in (rising, falling, flat)],
        )
        self.assertEqual([item["segment_direction"] for item in result.objects], ["up", "down", "flat"])
        self.assertTrue(all(item["status"] == "validated" and item["primary"] for item in result.objects))
        self.assertEqual(result.objects[0]["available_index"], rising.end.confirm_index)
        self.assertEqual(result.objects[0]["available_time"], rising.available_time.isoformat())
        self.assertEqual([event["event_type"] for event in result.events], ["validated", "validated", "validated"])
        self.assertEqual(result.objects[0]["change"], rising.change)
        self.assertEqual(result.objects[0]["bars"], rising.bar_span)


class LevelPipelineTests(unittest.TestCase):
    def test_same_run_keeps_direction_zone_shape_and_pause(self):
        series = _series_from_levels(_plateau(), quoted=True)
        snapshot = build_snapshot(series, symbol="LEVEL", engine_version="test")
        methods = snapshot["methods"]
        for key in ("direction", "zone_shape", "pause", *_LEVELS):
            self.assertEqual(methods[key]["status"], "completed", methods[key].get("reason"))
        self.assertTrue(methods["direction"]["objects"])
        self.assertTrue(methods["minor_swing"]["objects"])
        older = _series_from_levels(_plateau(), version=PARAM_VERSION_V4, quoted=True)
        older_snapshot = build_snapshot(older, symbol="LEVEL", engine_version="test")
        self.assertNotIn("major_trend", older_snapshot["methods"])
        self.assertEqual(
            _direction_signature(older_snapshot),
            _direction_signature(snapshot),
        )
        bare = build_snapshot(
            _series_from_levels(_plateau(), version=PARAM_VERSION),
            symbol="LEVEL",
            engine_version="test",
        )
        self.assertNotIn("methods", bare)

    def test_major_follows_long_structure_and_swings_follow_short_segments(self):
        for axis in ("log", "uniform"):
            series = _series_from_levels(_plateau(), axis=axis, extra_days={40: 8})
            snapshot = build_snapshot(series, symbol="LEVEL", engine_version="test")
            self._assert_follows_inputs(snapshot, series)
            mirrored = build_snapshot(
                _series_from_levels([-value for value in _plateau()], axis=axis, extra_days={40: 8}),
                symbol="LEVEL",
                engine_version="test",
            )
            self.assertEqual(
                [_swapped_major(item) for item in _objects(snapshot, "major_trend")],
                [_major_signature(item) for item in _objects(mirrored, "major_trend")],
            )
            self.assertEqual(
                [_swapped_retracement(item) for item in _objects(snapshot, "retracement")],
                [_retracement_signature(item) for item in _objects(mirrored, "retracement")],
            )

    def test_prefix_matches_the_as_of_view(self):
        levels = _plateau()
        series = _series_from_levels(levels, extra_days={40: 8})
        snapshot = build_snapshot(series, symbol="LEVEL", engine_version="test")
        interesting = {1, len(series) // 2, len(series) - 1}
        for event in snapshot["events"]:
            if event.get("object_kind") in _LEVELS:
                interesting.add(int(event["available_index"]) + 1)
        for count in sorted(index for index in interesting if 1 <= index <= len(series)):
            prefix_series = _series_from_levels(levels[:count], extra_days={40: 8})
            prefix = build_snapshot(prefix_series, symbol="LEVEL", engine_version="test")
            shown = _as_of(snapshot, prefix_series.bars[-1].timestamp)
            self._assert_prefix(prefix, shown, count)

    def test_methods_do_not_recompute_pivots_or_segments(self):
        series = _series_from_levels(_plateau())
        context = build_context(
            series,
            volatility=compute_volatility(series),
            pivots=(),
            segments=SegmentResult((), ()),
            boundaries=(),
            zones=(),
            events=(),
        )
        modules = (
            "analysis.structure.pivots",
            "analysis.structure.boundaries",
            "analysis.structure.zones",
            "analysis.structure.segments",
        )
        originals = {}
        for name in modules:
            module = __import__(name, fromlist=["detect_pivots"])
            for attr in ("detect_pivots", "build_boundaries", "build_zones", "build_segments"):
                if hasattr(module, attr):
                    originals[(name, attr)] = getattr(module, attr)
                    setattr(module, attr, _refuse)
        try:
            major = MajorTrendMethod().judge(context, {}, {"direction": {"status": "completed", "objects": []}})
            minor = MinorSwingMethod().judge(context, {}, {})
        finally:
            for (name, attr), original in originals.items():
                setattr(__import__(name, fromlist=[attr]), attr, original)
        self.assertTrue(major.objects)
        self.assertEqual(minor.objects, [])

    def test_direction_failure_skips_only_its_dependents(self):
        series = _series_from_levels(_plateau(), quoted=True)
        original = DirectionMethod.judge
        DirectionMethod.judge = lambda self, context, params, upstream: (_ for _ in ()).throw(RuntimeError("故意失败"))
        try:
            snapshot = build_snapshot(series, symbol="LEVEL", engine_version="test")
        finally:
            DirectionMethod.judge = original
        methods = snapshot["methods"]
        self.assertEqual(methods["direction"]["status"], "failed")
        self.assertEqual(methods["major_trend"]["status"], "skipped")
        self.assertEqual(methods["retracement"]["status"], "skipped")
        self.assertEqual(methods["minor_swing"]["status"], "completed")
        self.assertEqual(methods["zone_shape"]["status"], "completed")
        self.assertEqual(methods["pause"]["status"], "completed")
        self.assertIn("pivots", snapshot)
        self.assertIn("segments", snapshot)

    def test_both_axes_render_and_a_hand_built_table_keeps_every_row(self):
        for axis in ("log", "uniform"):
            snapshot = build_snapshot(
                _series_from_levels(_plateau(), axis=axis),
                symbol="LEVEL",
                engine_version="test",
            )
            html = render_report(snapshot)
            self.assertIn("主趋势", html)
            self.assertIn("次级折返", html)
            self.assertIn("短期波动", html)
            self.assertIn("方向", html)
            model = build_view_model(snapshot)
            self.assertTrue(all(mark["kind"] in MARK_KINDS for mark in model["marks"]))
            for key in _LEVELS:
                declared = {item.key for item in _method(key).categories}
                for item in snapshot["methods"][key]["objects"]:
                    if item.get("category"):
                        self.assertIn(item["category"], declared)
        hand = _hand_snapshot()
        model = build_view_model(hand)
        self.assertEqual(len(model["method_objects"]["major_trend"]), 3)
        html = render_report(hand)
        table = html.split('id="method-major_trend-segments"', 1)[1].split("</table>", 1)[0]
        self.assertEqual(table.count("data-object-id"), 3)

    def _assert_follows_inputs(self, snapshot, series):
        gaps = set(series.gap_indices)
        directions = _objects(snapshot, "direction")
        majors = _objects(snapshot, "major_trend")
        self.assertEqual(majors[0]["start_index"], 0)
        self.assertEqual(majors[-1]["end_index"], len(series) - 1)
        for left, right in zip(majors, majors[1:]):
            self.assertEqual(right["start_index"], left["end_index"] + 1)
        for index in range(len(series)):
            direction = _covering(directions, index, "long")
            major = _covering(majors, index, "long")
            structure = direction.get("structure") or "undecided"
            expected = "" if structure == "undecided" else structure
            self.assertEqual(major.get("category") or "", expected)
            self.assertEqual(major["structure"], structure)
        long_dirs = [item for item in directions if item["scale"] == "long"]
        for left, right in zip(long_dirs, long_dirs[1:]):
            if left.get("structure") != right.get("structure"):
                continue
            if int(right["start_index"]) != int(left["end_index"]) + 1:
                continue
            if any(int(left["end_index"]) < gap <= int(right["start_index"]) for gap in gaps):
                continue
            self.assertEqual(
                _covering(majors, int(left["end_index"]), "long")["id"],
                _covering(majors, int(right["start_index"]), "long")["id"],
            )
        for item in majors + _objects(snapshot, "retracement") + _objects(snapshot, "minor_swing"):
            self.assertFalse(any(item["start_index"] < gap <= item["end_index"] for gap in gaps))
        shorts = [item for item in snapshot["segments"] if item["scale"] == "short"]
        self.assertEqual(
            {item["id"] for item in _objects(snapshot, "minor_swing")},
            {f"minor_swing:{item['id']}" for item in shorts},
        )
        mids = [item for item in snapshot["segments"] if item["scale"] == "mid"]
        expected = []
        for segment in mids:
            major = _covering(majors, int(segment["start"]["extreme_index"]), "long")
            if _is_retracement(major.get("category") or "", segment["direction"]):
                expected.append(f"retracement:{segment['id']}")
        self.assertEqual({item["id"] for item in _objects(snapshot, "retracement")}, set(expected))
        for kind in ("retracement", "minor_swing"):
            types = {
                event["event_type"]
                for event in snapshot["events"]
                if event.get("object_kind") == kind
            }
            self.assertTrue(types <= {"validated"})

    def _assert_prefix(self, prefix, shown, count):
        stop = count - 1
        for key in _LEVELS:
            left = _objects(prefix, key)
            right = _objects(shown, key)
            if key == "major_trend":
                self.assertEqual([item["id"] for item in left], [item["id"] for item in right])
                for item, other in zip(left, right):
                    self.assertEqual(_frozen(item), _frozen(other))
                    self.assertEqual(item["status"], other["status"])
                    self.assertEqual(item["end_index"], other["end_index"])
            else:
                self.assertEqual(left, right)
        for kind in _LEVELS:
            self.assertEqual(
                [event for event in prefix["events"] if event.get("object_kind") == kind and event["available_index"] <= stop],
                [event for event in shown["events"] if event.get("object_kind") == kind],
            )


def _is_retracement(category, direction):
    return (category == "uptrend" and direction == "down") or (category == "downtrend" and direction == "up")


def _objects(snapshot, key):
    return list(snapshot["methods"][key]["objects"])


def _covering(objects, index, scale):
    found = [
        item
        for item in objects
        if item.get("scale") == scale and int(item["start_index"]) <= index <= int(item["end_index"])
    ]
    return found[-1]


def _direction_signature(snapshot):
    return [
        (
            item["id"],
            item.get("category") or "",
            item.get("structure"),
            item["start_index"],
            item["end_index"],
            item.get("trigger"),
            item.get("cause"),
        )
        for item in _objects(snapshot, "direction")
    ]


def _major_signature(item):
    return (item["start_index"], item["end_index"], item.get("category") or "", item.get("structure"), item.get("trigger"))


def _swapped_major(item):
    start, end, category, structure, trigger = _major_signature(item)
    return (start, end, _SWAP[category], _SWAP[structure], trigger)


def _retracement_signature(item):
    return (
        item["start_index"],
        item["end_index"],
        item.get("segment_direction"),
        item.get("major_category"),
        None if item.get("retracement_ratio") is None else round(float(item["retracement_ratio"]), 6),
    )


def _swapped_retracement(item):
    start, end, direction, major, ratio = _retracement_signature(item)
    return (start, end, _SWAP[direction], _SWAP[major], ratio)


def _frozen(segment):
    hidden = {"status", "end_index", "end_time", "stats"}
    return {key: value for key, value in segment.items() if key not in hidden}


def _as_of(snapshot, when):
    copied = dict(snapshot)
    identity = dict(snapshot["identity"])
    identity["as_of"] = when.isoformat()
    copied["identity"] = identity
    return display_snapshot(copied)


def _method(key):
    return {
        "major_trend": MajorTrendMethod,
        "retracement": RetracementMethod,
        "minor_swing": MinorSwingMethod,
    }[key]()


def _refuse(*_args, **_kwargs):
    raise AssertionError("级别判断不应重算拐点或线段")


def _hand_snapshot():
    params = default_params("log", version=PARAM_VERSION_V5)
    objects = []
    closes = []
    for index in range(3):
        start = index * 5
        objects.append(
            {
                "id": f"major_trend:long:{start}",
                "method": "major_trend",
                "scale": "long",
                "category": "uptrend",
                "status": "closed" if index < 2 else "open",
                "start_index": start,
                "end_index": start + 4,
                "start_time": f"2024-01-{2 + start:02d}T16:00:00",
                "end_time": f"2024-01-{6 + start:02d}T16:00:00",
                "available_index": start,
                "available_time": f"2024-01-{2 + start:02d}T16:00:00",
                "structure": "uptrend",
                "trigger": None if index == 0 else "structure",
                "direction_id": f"direction:long:{start}",
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
    record = {
        "key": "major_trend",
        "name": "主趋势",
        "version": "major_trend-v1",
        "kind": "run",
        "section": "2.4",
        "requires": ["direction"],
        "categories": [{"key": "uptrend", "name": "上升趋势", "color": "#E85D4C"}],
        "replaces": [],
        "params": {},
        "status": "completed",
        "reason": "",
        "objects": objects,
        "extra": {},
        "notes": [],
    }
    return {
        "kind": "trend_structure_snapshot",
        "identity": {
            "symbol": "HAND",
            "interval": "1d",
            "price_axis": "log",
            "price_axis_label": "对数价格",
            "param_version": PARAM_VERSION_V5,
            "as_of": None,
            "visible_scales": ["long"],
            "visible_methods": ["major_trend"],
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
        "methods": {"major_trend": record},
    }
