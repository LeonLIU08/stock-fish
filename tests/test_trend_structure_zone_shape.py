"""区间形状。合成收盘序列，不访问网络。"""
from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta

from analysis.structure.config import (
    PARAM_VERSION,
    PARAM_VERSION_V1,
    PARAM_VERSION_V2,
    PARAM_VERSION_V3,
    default_params,
    enabled_methods,
)
from analysis.structure.context import build_context
from analysis.structure.html_report import render_report
from analysis.structure.lifecycle import build_lifecycle
from analysis.structure.methods.base import MARK_KINDS
from analysis.structure.methods.zone_shape import (
    ZoneShapeMethod,
    ascending_triangle_levels,
    broadening_levels,
    classify_rails,
    rising_channel_levels,
    rising_wedge_levels,
    slow_rising_levels,
)
from analysis.structure.pivots import detect_pivots
from analysis.structure.segments import build_segments
from analysis.structure.series import build_series
from analysis.structure.snapshot import build_snapshot
from analysis.structure.view_model import build_view_model, display_snapshot
from analysis.structure.volatility import compute_volatility

_MIRROR = {
    "rising_channel": "falling_channel",
    "falling_channel": "rising_channel",
    "rising_wedge": "falling_wedge",
    "falling_wedge": "rising_wedge",
    "ascending_triangle": "descending_triangle",
    "descending_triangle": "ascending_triangle",
    "flat_channel": "flat_channel",
    "symmetrical_triangle": "symmetrical_triangle",
    "broadening": "broadening",
    "unclassified": "unclassified",
}


def _case(ru, rl, ratio):
    span = 10
    width = 1.0
    judged = classify_rails(
        upper_slope=ru * width / span,
        lower_slope=rl * width / span,
        start_index=0,
        end_index=span,
        start_width=width,
        width_ratio=ratio,
    )
    judged.update(upper_slope=ru * width / span, lower_slope=rl * width / span, width_ratio=ratio)
    return judged


class ZoneShapeClassifyTests(unittest.TestCase):
    def test_each_row_of_the_table_and_both_unclassified_cases(self):
        cases = {
            "rising_channel": _case(0.5, 0.5, 1.0),
            "falling_channel": _case(-0.5, -0.5, 1.0),
            "flat_channel": _case(0.0, 0.0, 1.0),
            "rising_wedge": _case(0.3, 0.8, 0.5),
            "falling_wedge": _case(-0.8, -0.3, 0.5),
            "symmetrical_triangle": _case(-0.4, 0.4, 0.2),
            "ascending_triangle": _case(0.0, 0.4, 0.6),
            "descending_triangle": _case(-0.4, 0.0, 0.6),
            "broadening": _case(0.4, -0.4, 1.8),
        }
        both_level_triangle = _case(-0.2, 0.2, 0.6)
        self.assertEqual(both_level_triangle["category"], "symmetrical_triangle")
        self.assertEqual(both_level_triangle["upper_course"], "level")
        self.assertEqual(both_level_triangle["lower_course"], "level")
        upper_level_broad = _case(0.0, -0.5, 1.5)
        self.assertEqual(upper_level_broad["category"], "broadening")
        for category, judged in cases.items():
            self.assertEqual(judged["category"], category, category)
        stable_mixed = _case(0.4, 0.2, 1.2)
        self.assertEqual(stable_mixed["width_course"], "stable")
        self.assertEqual(stable_mixed["upper_course"], "rising")
        self.assertEqual(stable_mixed["lower_course"], "level")
        self.assertEqual(stable_mixed["category"], "unclassified")
        widening_same_way = _case(0.8, 0.3, 1.5)
        self.assertEqual(widening_same_way["width_course"], "diverging")
        self.assertEqual(widening_same_way["upper_course"], "rising")
        self.assertEqual(widening_same_way["lower_course"], "rising")
        self.assertEqual(widening_same_way["category"], "unclassified")

    def test_threshold_edges_stay_level_and_stable(self):
        level = _case(0.2, -0.2, 1.0)
        self.assertEqual(level["upper_course"], "level")
        self.assertEqual(level["lower_course"], "level")
        higher = classify_rails(
            upper_slope=math.nextafter(0.2, math.inf) / 10.0,
            lower_slope=0.0,
            start_index=0,
            end_index=10,
            start_width=1.0,
            width_ratio=1.0,
        )
        self.assertEqual(higher["upper_course"], "rising")
        lower = classify_rails(
            upper_slope=0.0,
            lower_slope=math.nextafter(-0.2, -math.inf) / 10.0,
            start_index=0,
            end_index=10,
            start_width=1.0,
            width_ratio=1.0,
        )
        self.assertEqual(lower["lower_course"], "falling")
        on_low = _case(0.3, 0.5, 0.8)
        self.assertEqual(on_low["width_course"], "stable")
        self.assertEqual(on_low["category"], "rising_channel")
        just_under = _case(0.3, 0.5, math.nextafter(0.8, 0.0))
        self.assertEqual(just_under["width_course"], "converging")
        on_high = _case(0.45, 0.2, 1.25)
        self.assertEqual(on_high["width_course"], "stable")
        self.assertEqual(on_high["category"], "unclassified")
        just_over = _case(0.5, 0.0, math.nextafter(1.25, math.inf))
        self.assertEqual(just_over["width_course"], "diverging")
        self.assertEqual(just_over["category"], "broadening")

    def test_mirroring_prices_swaps_rising_and_falling(self):
        samples = [
            _case(0.5, 0.5, 1.0),
            _case(-0.5, -0.5, 1.0),
            _case(0.3, 0.8, 0.5),
            _case(-0.8, -0.3, 0.5),
            _case(0.0, 0.4, 0.6),
            _case(-0.4, 0.0, 0.6),
            _case(-0.4, 0.4, 0.2),
            _case(-0.2, 0.2, 0.6),
            _case(0.0, 0.0, 1.0),
            _case(0.4, -0.4, 1.8),
            _case(0.2, -0.2, 1.4),
            _case(0.4, 0.2, 1.2),
            _case(0.8, 0.3, 1.5),
            _case(-0.3, -0.8, 1.5),
        ]
        for judged in samples:
            flipped = classify_rails(
                upper_slope=-judged["lower_slope"],
                lower_slope=-judged["upper_slope"],
                start_index=0,
                end_index=10,
                start_width=1.0,
                width_ratio=judged["width_ratio"],
            )
            self.assertEqual(flipped["category"], _MIRROR[judged["category"]])
            self.assertEqual(flipped["width_course"], judged["width_course"])

    def test_bad_params_are_rejected(self):
        method = ZoneShapeMethod()
        with self.assertRaises(ValueError):
            method.check_params({"level_tolerance": 0, "converging_width_ratio": 0.8, "diverging_width_ratio": 1.25})
        with self.assertRaises(ValueError):
            method.check_params({"level_tolerance": 0.2, "converging_width_ratio": 1.3, "diverging_width_ratio": 0.8})


class ZoneShapeAcceptanceTests(unittest.TestCase):
    def test_v3_keeps_v2_and_adds_zone_shape(self):
        self.assertEqual(enabled_methods(PARAM_VERSION_V2), ("direction",))
        self.assertEqual(enabled_methods(PARAM_VERSION_V3), ("direction", "zone_shape", "pause"))
        directed = default_params(version=PARAM_VERSION_V2)
        shaped = default_params(version=PARAM_VERSION_V3)
        self.assertNotIn("zone_shape", directed.to_dict())
        self.assertEqual(shaped.zone_min_touch_clusters, directed.zone_min_touch_clusters)
        self.assertEqual(shaped.zone_credit_full_span, directed.zone_credit_full_span)
        self.assertEqual(shaped.zone_shape.level_tolerance, 0.2)
        self.assertEqual(shaped.zone_shape.converging_width_ratio, 0.8)
        self.assertEqual(shaped.zone_shape.diverging_width_ratio, 1.25)
        self.assertEqual(shaped.direction.stall_bars, directed.direction.stall_bars)
        for version in (PARAM_VERSION, PARAM_VERSION_V1):
            self.assertNotIn("zone_shape", default_params(version=version).to_dict())

    def test_width_ratio_matches_the_stored_zone_ratio(self):
        snapshot, _series = _snapshot(rising_channel_levels())
        for row in snapshot["methods"]["zone_shape"]["extra"]["classifications"]:
            formula = 1.0 + row["upper_displacement"] - row["lower_displacement"]
            self.assertTrue(
                math.isclose(formula, row["width_ratio"], rel_tol=1e-9, abs_tol=1e-9),
                f"{formula} != {row['width_ratio']} for {row['zone_id']}",
            )

    def test_full_run_finds_the_named_shapes(self):
        expectations = (
            (rising_wedge_levels(), "rising_wedge"),
            (ascending_triangle_levels(), "ascending_triangle"),
            (broadening_levels(), "broadening"),
        )
        for levels, category in expectations:
            snapshot, _series = _snapshot(levels)
            shapes = _shapes(snapshot)
            self.assertTrue(
                any(item["category"] == category and item["primary"] and item["scale"] == "short" for item in shapes),
                category,
            )
            self.assertNotIn("unclassified", {item["category"] for item in shapes})

    def test_slow_rise_is_a_channel_where_v1_said_sideways(self):
        snapshot, _series = _snapshot(slow_rising_levels())
        rows = _rows(snapshot)
        matched = [
            row
            for row in rows
            if row["legacy_label"] == "sideways" and row["category"] == "rising_channel"
        ]
        self.assertTrue(matched)
        for row in matched:
            if row["scale"] != "short" or not _is_primary_shape(snapshot, row["zone_id"]):
                continue
            self.assertEqual(row["upper_course"], "rising")
            self.assertEqual(row["lower_course"], "rising")
            self.assertEqual(row["width_course"], "stable")
            self.assertGreater(row["upper_displacement"], 0.2)
            break
        else:
            self.fail("没有短尺度主结果同时是横向区间和上升通道")

    def test_gentle_rise_under_a_flat_rail_stays_unclassified(self):
        snapshot, _series = _snapshot(rising_wedge_levels())
        matched = [
            row
            for row in _rows(snapshot)
            if row["legacy_label"] == "channel"
            and row["category"] == "unclassified"
            and row["upper_course"] == "level"
            and row["lower_course"] == "rising"
            and row["width_course"] == "stable"
        ]
        self.assertTrue(matched)
        shape_zones = {item["evidence"]["zone_id"] for item in _shapes(snapshot)}
        self.assertTrue(all(row["zone_id"] not in shape_zones for row in matched))

    def test_breakout_updates_the_shape_on_the_same_bar(self):
        base = rising_channel_levels()
        series = _series(base)
        zone = next(
            item
            for item in build_lifecycle(series).zones
            if item.scale == "short" and item.primary and item.label == "channel" and item.status == "validated"
        )
        snapshot, _extended = _snapshot(tuple(base) + _below(zone, len(base)))
        shape = next(item for item in _shapes(snapshot) if item["evidence"]["zone_id"] == zone.id)
        self.assertEqual(shape["category"], "rising_channel")
        self.assertEqual(shape["status"], "broken")
        shape_events = [event for event in snapshot["events"] if event["object_id"] == shape["id"]]
        zone_events = [
            event
            for event in snapshot["events"]
            if event["object_id"] == zone.id and event["object_kind"] == "zone"
        ]
        self.assertEqual([event["event_type"] for event in shape_events], ["validated", "breakout"])
        self.assertEqual(shape_events[-1]["event_index"], zone_events[-1]["event_index"])
        self.assertEqual(shape_events[-1]["reference_close"], zone_events[-1]["reference_close"])
        self.assertEqual(shape_events[-1]["price_cross_index"], zone_events[-1]["price_cross_index"])
        self.assertTrue(shape_events[-1]["reason"].startswith("上升通道："))
        self.assertEqual(shape["available_time"], next(item["available_time"] for item in snapshot["zones"] if item["id"] == zone.id))

    def test_expired_zone_keeps_an_expired_shape_event(self):
        snapshot, _series = _snapshot(rising_channel_levels())
        expired = next(item for item in _shapes(snapshot) if item["status"] == "expired")
        events = [event for event in snapshot["events"] if event["object_id"] == expired["id"]]
        self.assertEqual(events[-1]["event_type"], "expired")
        self.assertIn("：", events[-1]["reason"])

    def test_candidates_are_not_classified(self):
        snapshot, series = _snapshot(rising_channel_levels())
        live = {
            zone.id
            for zone in build_lifecycle(series).zones
            if zone.status in ("validated", "broken", "expired")
        }
        rows = _rows(snapshot)
        self.assertEqual({row["zone_id"] for row in rows}, live)
        classified = {row["zone_id"] for row in rows if row["category"] != "unclassified"}
        self.assertEqual({item["evidence"]["zone_id"] for item in _shapes(snapshot)}, classified)


class ZoneShapeContractTests(unittest.TestCase):
    def test_prefix_matches_the_as_of_view(self):
        base = rising_channel_levels()
        series = _series(base)
        zone = next(
            item
            for item in build_lifecycle(series).zones
            if item.scale == "short" and item.primary and item.label == "channel"
        )
        levels = tuple(base) + _below(zone, len(base))
        for axis in ("log", "uniform"):
            snapshot, full = _snapshot(levels, axis)
            interesting = {len(full) // 2, len(full)}
            confirms = sorted(
                {
                    item.confirm_index + 1
                    for item in build_lifecycle(full).zones
                    if item.scale == "short" and item.status in ("validated", "broken", "expired")
                }
            )
            interesting.update(confirms[:4])
            interesting.update(confirms[-2:])
            for event in snapshot["events"]:
                if event.get("object_kind") == "zone_shape" and event.get("event_type") == "breakout":
                    interesting.add(int(event["available_index"]))
                    interesting.add(int(event["available_index"]) + 1)
            for count in sorted(index for index in interesting if 8 <= index <= len(full)):
                with self.subTest(axis=axis, count=count):
                    prefix_snapshot, prefix = _snapshot(levels[:count], axis)
                    shown = _as_of(snapshot, prefix.bars[-1].timestamp)
                    self._assert_prefix(prefix_snapshot, shown, count)

    def test_available_after_the_zone_and_not_across_gaps(self):
        snapshot, series = _snapshot(ascending_triangle_levels(), extra_days={60: 12})
        self.assertTrue(series.gap_indices)
        self.assertTrue(_shapes(snapshot))
        zones = {zone["id"]: zone for zone in snapshot["zones"]}
        gaps = set(series.gap_indices)
        for shape in _shapes(snapshot):
            self.assertFalse(any(shape["start_index"] < gap <= shape["end_index"] for gap in gaps))
            zone = zones[shape["evidence"]["zone_id"]]
            self.assertGreaterEqual(shape["available_index"], zone["confirm_index"])
            self.assertEqual(shape["available_time"], zone["available_time"])
            self.assertEqual(shape["evidence"]["lower_id"], zone["lower_id"])
            self.assertEqual(shape["evidence"]["upper_id"], zone["upper_id"])
            self.assertEqual(shape["status"], zone["status"])
            self.assertEqual(shape["revision"], zone["revision"])
            self.assertEqual(shape["primary"], zone["primary"])
            self.assertEqual(shape["constraints"], zone["constraints"])
            for event in snapshot["events"]:
                if event.get("object_id") == shape["id"]:
                    self.assertGreaterEqual(event["available_index"], shape["available_index"])
                    self.assertEqual(event["object_kind"], "zone_shape")

    def test_shape_does_not_refit_or_rewrite_zones(self):
        _snapshot_value, series = _snapshot(ascending_triangle_levels())
        context = _context(series)
        before_zones = [zone.to_dict() for zone in context.zones]
        before_events = [event.to_dict() for event in context.events]
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
            result = ZoneShapeMethod().judge(context, series.params.zone_shape.to_dict(), {})
        finally:
            for (name, attr), original in originals.items():
                setattr(__import__(name, fromlist=[attr]), attr, original)
        self.assertTrue(result.objects)
        self.assertEqual(before_zones, [zone.to_dict() for zone in context.zones])
        self.assertEqual(before_events, [event.to_dict() for event in context.events])

    def test_both_axes_render_and_older_versions_omit_shapes(self):
        for axis in ("log", "uniform"):
            snapshot, _series = _snapshot(ascending_triangle_levels(), axis)
            html = render_report(snapshot)
            self.assertIn("区间形状", html)
            self.assertIn("method-zone_shape", html)
            model = build_view_model(snapshot)
            self.assertIn("zones", model["replaced_layers"])
            bands = [mark for mark in model["marks"] if mark["method"] == "zone_shape" and mark["kind"] == "band"]
            self.assertTrue(bands)
            self.assertTrue(all(mark["kind"] in MARK_KINDS for mark in model["marks"]))
            declared = {item.key for item in ZoneShapeMethod.categories}
            for item in _shapes(snapshot):
                self.assertIn(item["category"], declared)
        for version in (PARAM_VERSION, PARAM_VERSION_V1, PARAM_VERSION_V2):
            snapshot, _series = _snapshot(ascending_triangle_levels(), version=version)
            self.assertNotIn("zone_shape", snapshot.get("methods") or {})
            self.assertNotIn("zone_shape", snapshot["params"])
            html = render_report(snapshot)
            self.assertNotIn('id="method-zone_shape"', html)

    def test_direction_only_keeps_the_old_zone_bands(self):
        series = _series(ascending_triangle_levels())
        snapshot = build_snapshot(series, symbol="SHAPE", engine_version="test", visible_methods=["direction"])
        model = build_view_model(snapshot)
        self.assertEqual(model["replaced_layers"], [])
        self.assertIn("zone_shape", snapshot["methods"])
        self.assertEqual(model["default_methods"], ["direction"])

    def test_hand_built_snapshot_keeps_rows_when_a_zone_is_missing(self):
        snapshot = _hand_snapshot()
        model = build_view_model(snapshot)
        self.assertEqual(len(model["method_objects"]["zone_shape"]), 3)
        bands = [mark for mark in model["marks"] if mark["kind"] == "band"]
        texts = [mark for mark in model["marks"] if mark["kind"] == "text" and mark["method"] == "zone_shape"]
        self.assertEqual(len(bands), 1)
        self.assertEqual(bands[0]["object_id"], "zone_shape:short:zone:present")
        self.assertGreater(bands[0]["points"][0][1], 50)
        self.assertEqual(texts[0]["text"], "上升通道")
        self.assertEqual(bands[0]["line_type"], "dashed")
        html = render_report(snapshot)
        table = html.split('id="method-zone_shape-shapes"', 1)[1].split("</table>", 1)[0]
        self.assertEqual(table.count("data-object-id"), 3)
        chart = (
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "analysis/structure/assets/structure-chart.js"
        ).read_text(encoding="utf-8")
        self.assertIn("replaced_layers", chart)

    def _assert_prefix(self, prefix, shown, count):
        left = _shapes(prefix)
        right = _shapes(shown)
        self.assertEqual([item["id"] for item in left], [item["id"] for item in right])
        for item, other in zip(left, right):
            self.assertEqual(_frozen(item), _frozen(other))
            self.assertEqual(item["status"], other["status"])
        stop = count - 1
        self.assertEqual(_rows_until(_rows(prefix), stop), _rows(shown))
        self.assertEqual(
            [event for event in prefix["events"] if event.get("object_kind") == "zone_shape"],
            [event for event in shown["events"] if event.get("object_kind") == "zone_shape"],
        )


def _shapes(snapshot):
    return list(snapshot["methods"]["zone_shape"]["objects"])


def _rows(snapshot):
    return list(snapshot["methods"]["zone_shape"]["extra"]["classifications"])


def _rows_until(rows, stop):
    return [row for row in rows if int(row["available_index"]) <= stop]


def _is_primary_shape(snapshot, zone_id):
    return any(item["evidence"]["zone_id"] == zone_id and item["primary"] for item in _shapes(snapshot))


def _frozen(shape):
    hidden = {"status", "primary", "alternate_of", "end_index", "end_time", "stats"}
    return {key: value for key, value in shape.items() if key not in hidden}


def _series(levels, axis="log", extra_days=None, version=PARAM_VERSION_V3):
    closes = [100.0 * math.exp(level) for level in levels]
    return build_series(
        _times(len(levels), extra_days),
        closes,
        interval="1d",
        price_axis=axis,
        params=default_params(axis, version=version),
    )


def _snapshot(levels, axis="log", extra_days=None, version=PARAM_VERSION_V3):
    series = _series(levels, axis, extra_days, version)
    return build_snapshot(series, symbol="SHAPE", engine_version="test"), series


def _context(series):
    pivots = detect_pivots(series)
    lifecycle = build_lifecycle(series, pivots)
    return build_context(
        series,
        volatility=compute_volatility(series),
        pivots=pivots,
        segments=build_segments(series, pivots),
        boundaries=lifecycle.boundaries,
        zones=lifecycle.zones,
        events=lifecycle.events,
    )


def _times(count, extra_days=None, start_year=2024):
    extra_days = extra_days or {}
    cursor = datetime(start_year, 1, 2, 16, 0)
    times = []
    for index in range(count):
        if index:
            cursor = cursor + timedelta(days=1 + extra_days.get(index, 0))
        times.append(cursor)
    return times


def _below(zone, start_index, count=2):
    levels = []
    for step in range(count):
        index = start_index + step
        geometric = zone.lower_at(index) - 0.8 * zone.volatility
        levels.append(geometric - math.log(100.0))
    return tuple(levels)


def _as_of(snapshot, when):
    copied = dict(snapshot)
    identity = dict(snapshot["identity"])
    identity["as_of"] = when.isoformat()
    copied["identity"] = identity
    return display_snapshot(copied)


def _refuse(*_args, **_kwargs):
    raise AssertionError("区间形状不应重算拐点或边界")


def _hand_snapshot():
    params = default_params("log", version=PARAM_VERSION_V3)
    low = math.log(100.0)
    zone = {
        "id": "short:zone:present",
        "revision": 1,
        "scale": "short",
        "label": "channel",
        "status": "broken",
        "primary": True,
        "alternate_of": None,
        "lower_id": "short:support:0-8",
        "upper_id": "short:resistance:0-8",
        "lower_slope": 0.001,
        "lower_intercept": low,
        "upper_slope": 0.001,
        "upper_intercept": low + 0.08,
        "effective_start": 0,
        "effective_end": 8,
        "available_time": "2024-01-02T16:00:00",
        "confirm_index": 4,
    }
    objects = [
        _hand_shape("zone_shape:short:zone:present", "rising_channel", True, "short:zone:present", "broken"),
        _hand_shape("zone_shape:short:zone:missing", "rising_wedge", True, "short:zone:missing", "validated"),
        _hand_shape("zone_shape:short:zone:alt", "broadening", False, "short:zone:present", "validated"),
    ]
    closes = [
        {"index": index, "time": f"2024-01-{2 + index:02d}T16:00:00", "close": 100.0 + index, "gap_before": False}
        for index in range(12)
    ]
    record = {
        "key": "zone_shape",
        "name": "区间形状",
        "version": "zone_shape-v1",
        "kind": "instance",
        "section": "2.2",
        "requires": ["zones", "events"],
        "categories": [
            {"key": item.key, "name": item.name, "color": item.color} for item in ZoneShapeMethod.categories
        ],
        "replaces": ["zones"],
        "params": params.zone_shape.to_dict(),
        "status": "completed",
        "reason": "",
        "objects": objects,
        "extra": {"classifications": []},
        "notes": [],
    }
    return {
        "kind": "trend_structure_snapshot",
        "identity": {
            "symbol": "HAND",
            "interval": "1d",
            "price_axis": "log",
            "price_axis_label": "对数价格",
            "param_version": PARAM_VERSION_V3,
            "as_of": None,
            "visible_scales": ["short", "mid", "long"],
            "visible_methods": ["zone_shape"],
            "bar_count": 12,
        },
        "reading": "",
        "scale_definition_note": "",
        "quality": {"bar_count": 12, "gap_count": 0, "gaps": [], "grade": "正常", "grade_note": ""},
        "volatility": {"scales": [], "unit": "对数收益"},
        "pivot_stats": [],
        "pivots": [],
        "temporary_pivots": [],
        "segments": [],
        "temporary_segments": [],
        "boundaries": [],
        "zones": [zone],
        "events": [
            {
                "event_type": "breakout",
                "event_index": 6,
                "available_index": 6,
                "available_time": "2024-01-08T16:00:00",
                "object_id": "zone_shape:short:zone:present",
                "object_kind": "zone_shape",
                "reference_close": 106.0,
                "reason": "上升通道：收盘价连续 2 根落在区间外",
                "revision": 1,
                "price_cross_index": 5,
            }
        ],
        "relations": {},
        "closes": closes,
        "params": params.to_dict(),
        "methods": {"zone_shape": record},
    }


def _hand_shape(shape_id, category, primary, zone_id, status):
    return {
        "id": shape_id,
        "method": "zone_shape",
        "scale": "short",
        "category": category,
        "status": status,
        "start_index": 0,
        "end_index": 8,
        "start_time": "2024-01-02T16:00:00",
        "end_time": "2024-01-10T16:00:00",
        "available_index": 4,
        "available_time": "2024-01-06T16:00:00",
        "revision": 1,
        "primary": primary,
        "alternate_of": None if primary else "short:zone:present",
        "evidence": {"zone_id": zone_id, "lower_id": "short:support:0-8", "upper_id": "short:resistance:0-8"},
        "constraints": [],
        "score": None,
        "stats": None,
        "upper_course": "rising",
        "lower_course": "rising",
        "width_course": "stable",
        "upper_displacement": 0.4,
        "lower_displacement": 0.4,
        "width_ratio": 1.0,
        "legacy_label": "channel",
    }


if __name__ == "__main__":
    unittest.main()
