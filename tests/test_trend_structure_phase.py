"""阶段：主状态、推进参照和整理分数。合成序列，不访问网络。"""
from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta

from analysis.structure.config import (
    PARAM_VERSION,
    PARAM_VERSION_V2,
    PARAM_VERSION_V3,
    PARAM_VERSION_V4,
    PARAM_VERSION_V5,
    PARAM_VERSION_V6,
    default_params,
    phase_params_from_dict,
)
from analysis.structure.context import build_context
from analysis.structure.html_report import render_report
from analysis.structure.methods.base import MARK_KINDS
from analysis.structure.methods.direction import DirectionMethod
from analysis.structure.methods.phase import PhaseMethod
from analysis.structure.pivots import Pivot
from analysis.structure.series import build_series
from analysis.structure.snapshot import build_snapshot
from analysis.structure.view_model import build_view_model, display_snapshot
from analysis.structure.volatility import compute_volatility

_MIRROR = {
    "up_birth": "down_birth",
    "up_continue": "down_continue",
    "up_pullback": "down_rebound",
    "up_exhaustion": "down_exhaustion",
    "down_birth": "up_birth",
    "down_continue": "up_continue",
    "down_rebound": "up_pullback",
    "down_exhaustion": "up_exhaustion",
    "suspected_accumulation": "suspected_distribution",
    "suspected_distribution": "suspected_accumulation",
    "neutral_range": "neutral_range",
    "transition_pending": "transition_pending",
    "insufficient": "insufficient",
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


def _series(closes, axis="uniform", version=PARAM_VERSION_V6, extra_days=None, highs=None, lows=None, volumes=None, opens=None):
    return build_series(
        _times(len(closes), extra_days),
        closes,
        interval="1d",
        params=default_params(axis, version=version),
        highs=highs,
        lows=lows,
        volumes=volumes,
        opens=opens,
    )


def _context(series, volatility=None, pivots=()):
    count = len(series)
    if volatility is None:
        volatility = tuple(1.0 for _index in range(count))
    return build_context(
        series,
        volatility=tuple(volatility),
        pivots=_PivotBag(pivots),
        segments=(),
        boundaries=(),
        zones=(),
        events=(),
    )


class _PivotBag:
    def __init__(self, confirmed):
        self.confirmed = tuple(confirmed)


def _pivot(scale, role, extreme, confirm, geometric):
    moment = datetime(2024, 1, 2) + timedelta(days=extreme)
    confirmed = datetime(2024, 1, 2) + timedelta(days=confirm)
    return Pivot(
        scale=scale,
        role=role,
        extreme_index=extreme,
        confirm_index=confirm,
        extreme_time=moment,
        confirm_time=confirmed,
        available_time=confirmed,
        price=float(geometric),
        geometric_price=float(geometric),
        threshold=1.0,
        volatility=1.0,
        threshold_index=extreme,
        segment_start_index=0,
    )


def _direction(scale, start, end, structure, category=None, cause=None, origin=None, progress=None, protection=None):
    if category is None:
        category = "" if structure == "undecided" else structure
    return {
        "id": f"direction:{scale}:{start}",
        "scale": scale,
        "structure": structure,
        "category": category,
        "cause": cause,
        "start_index": start,
        "end_index": end,
        "available_index": start,
        "structure_origin": origin,
        "progress": progress or [],
        "protection": protection or [],
    }


def _progress(index, price, source="establish"):
    return {
        "available_index": index,
        "available_time": "",
        "reference_index": index,
        "geometric_price": float(price),
        "source": source,
    }


def _origin(index, role="low", price=90.0):
    return {
        "scale": "short",
        "role": role,
        "extreme_index": index,
        "geometric_price": float(price),
        "volatility": 1.0,
    }


def _pack(**overrides):
    payload = phase_params_from_dict(None).to_dict()
    payload.update(overrides)
    return payload


def _run(closes, directions, comparisons=None, pivots=(), volatility=None, params=None, axis="uniform", **quotes):
    series = _series(closes, axis=axis, **quotes)
    upstream = {
        "direction": {
            "status": "completed",
            "objects": directions,
            "extra": {"swing_comparisons": comparisons or []},
        }
    }
    result = PhaseMethod().judge(_context(series, volatility, pivots), params or _pack(), upstream)
    return result, series


def _short(result):
    return [item for item in result.objects if item["scale"] == "short"]


def _at(segments, index):
    for item in segments:
        if int(item["start_index"]) <= index <= int(item["end_index"]):
            return item
    raise AssertionError(f"序号 {index} 没有阶段区段")


def _categories(segments):
    return [item["category"] for item in segments]


class PhaseAcceptanceTests(unittest.TestCase):
    def test_age_starts_at_the_structure_change_not_the_origin(self):
        count = 40
        closes = [100.0] * count
        origin = _origin(0)
        directions = [
            _direction("short", 0, 19, "undecided", ""),
            _direction(
                "short",
                20,
                39,
                "uptrend",
                origin=origin,
                progress=[_progress(20, 100.0)],
            ),
        ]
        result, _series_value = _run(closes, directions)
        segments = _short(result)
        birth = _at(segments, 20)
        self.assertEqual(birth["category"], "up_birth")
        self.assertEqual(birth["start_index"], 20)
        self.assertEqual(birth["confirm_index"], 20)
        self.assertEqual(birth["summary"]["age"], 1)
        self.assertEqual(birth["structure_origin"]["extreme_index"], 0)
        self.assertEqual(_at(segments, 24)["category"], "up_birth")
        continued = _at(segments, 25)
        self.assertEqual(continued["category"], "up_continue")
        self.assertEqual(continued["start_index"], 25)
        self.assertEqual(continued["confirm_index"], 20)
        self.assertEqual(continued["summary"]["age"], 6)
        self.assertEqual(continued["structure_origin"]["extreme_index"], 0)

    def test_pullback_waits_for_two_bars_and_ignores_an_equal_average(self):
        closes = [100.0] * 16
        closes[14] = 98.0
        closes[15] = 98.0
        directions = [
            _direction("short", 0, 15, "uptrend", origin=_origin(0), progress=[_progress(0, 110.0)]),
        ]
        result, _series_value = _run(closes, directions)
        segments = _short(result)
        self.assertEqual(_at(segments, 13)["category"], "up_continue")
        self.assertEqual(_at(segments, 14)["category"], "up_continue")
        pullback = _at(segments, 15)
        self.assertEqual(pullback["category"], "up_pullback")
        self.assertEqual(pullback["start_index"], 15)
        self.assertEqual(pullback["trigger"], "pullback")
        one = closes[:15] + [100.0]
        held, _ignored = _run(one, [_direction("short", 0, 15, "uptrend", progress=[_progress(0, 110.0)])])
        self.assertEqual(_at(_short(held), 14)["category"], "up_continue")
        self.assertNotIn("up_pullback", _categories(_short(held)))
        flat, _ignored = _run([100.0] * 20, [_direction("short", 0, 19, "uptrend", progress=[_progress(0, 110.0)])])
        self.assertEqual(_at(_short(flat), 19)["category"], "up_continue")
        self.assertEqual(_at(_short(flat), 19)["stats"]["reference"], 110.0)
        raised = [100.0] * 19 + [130.0]
        still, _ignored = _run(raised, [_direction("short", 0, 19, "uptrend", progress=[_progress(0, 110.0)])])
        self.assertEqual(_at(_short(still), 19)["stats"]["reference"], 110.0)

    def test_exhaustion_uses_each_item_and_exits_without_another_wait(self):
        closes = [100.0] * 12
        directions = [
            _direction(
                "short",
                0,
                11,
                "uptrend",
                progress=[_progress(0, 100.0), _progress(1, 110.0, "progress")],
            )
        ]
        fade, _series_value = _run(closes, directions, params=_pack(exhaustion_score=1, birth_bars=20))
        segments = _short(fade)
        self.assertEqual(_at(segments, 1)["category"], "up_birth")
        entered = _at(segments, 2)
        self.assertEqual(entered["category"], "up_exhaustion")
        self.assertEqual(entered["start_index"], 2)
        self.assertEqual(entered["structure"], "uptrend")
        self.assertTrue(entered["summary"]["exhaustion"]["progress_fade"])
        jumped = [100.0, 110.0, 110.0, 200.0, 200.0] + [200.0] * 7
        exited, _ignored = _run(
            jumped,
            [_direction("short", 0, 11, "uptrend", progress=[_progress(0, 100.0), _progress(1, 110.0, "progress")])],
            params=_pack(exhaustion_score=1, birth_bars=20),
        )
        back = _at(_short(exited), 4)
        self.assertEqual(back["category"], "up_birth")
        self.assertEqual(back["start_index"], 4)
        self.assertLessEqual(back["summary"]["age"], 20)
        swing = {
            "scale": "short",
            "side": "high",
            "mode": "pivot",
            "result": "higher",
            "available_index": 6,
            "reference": {"scale": "short", "role": "high", "extreme_index": 4},
            "subject_index": 6,
        }
        dipped = [120.0] * 6 + [100.0, 100.0]
        swung, _ignored = _run(
            dipped,
            [_direction("short", 0, 7, "uptrend", progress=[_progress(0, 120.0)])],
            comparisons=[swing],
            params=_pack(exhaustion_score=1, swing_retrace=1.5),
        )
        self.assertTrue(_at(_short(swung), 7)["summary"]["exhaustion"]["swing_retrace"])
        self.assertEqual(_at(_short(swung), 7)["category"], "up_exhaustion")

    def test_wick_and_volume_can_score_on_their_own(self):
        count = 8
        closes = [100.0] * count
        highs = [101.0] * count
        lows = [99.0] * count
        opens = [100.0] * count
        for index in (5, 6):
            highs[index] = 110.0
            lows[index] = 100.0
            opens[index] = 100.0
            closes[index] = 100.0
        wick, _series_value = _run(
            closes,
            [_direction("short", 0, count - 1, "uptrend", progress=[_progress(0, 100.0)])],
            params=_pack(exhaustion_score=1, wick_count=2, wick_amplitude=1.0),
            highs=highs,
            lows=lows,
            opens=opens,
            volumes=[1000.0] * count,
        )
        self.assertTrue(_at(_short(wick), 6)["summary"]["exhaustion"]["wick_reject"])
        volumes = [100.0] * 25 + [400.0] * 3
        quiet = [100.0] * 28
        volume, _ignored = _run(
            quiet,
            [_direction("short", 0, 27, "uptrend", progress=[_progress(0, 100.0)])],
            params=_pack(exhaustion_score=1, volume_multiple=1.5, volume_impulse=0.5),
            highs=quiet,
            lows=[price - 0.2 for price in quiet],
            opens=quiet,
            volumes=volumes,
        )
        exhausted = _at(_short(volume), 26)
        self.assertEqual(exhausted["category"], "up_exhaustion")
        self.assertTrue(exhausted["summary"]["exhaustion"]["volume_stall"])
        self.assertFalse(exhausted["summary"]["exhaustion"]["wick_reject"])

    def test_confirmed_exhaustion_outranks_birth(self):
        closes = [110.0] * 4
        directions = [
            _direction(
                "short",
                0,
                3,
                "uptrend",
                progress=[_progress(0, 100.0), _progress(1, 110.0, "progress")],
            )
        ]
        result, _series_value = _run(closes, directions, params=_pack(exhaustion_score=1, exhaustion_bars=1, birth_bars=20))
        self.assertEqual(_at(_short(result), 1)["category"], "up_exhaustion")
        self.assertLessEqual(_at(_short(result), 1)["summary"]["age"], 20)

    def test_stall_keeps_the_up_branch_and_the_age(self):
        closes = [100.0] * 16
        directions = [
            _direction("short", 0, 9, "uptrend", progress=[_progress(0, 100.0)]),
            _direction(
                "short",
                10,
                15,
                "uptrend",
                category="range",
                cause="stall",
                progress=[_progress(10, 100.0, "inherit")],
            ),
        ]
        result, _series_value = _run(closes, directions)
        segments = _short(result)
        current = _at(segments, 12)
        self.assertTrue(current["category"].startswith("up_"))
        self.assertEqual(current["confirm_index"], 0)
        self.assertGreater(current["stats"]["age"], 5)
        self.assertNotIn("suspected_accumulation", _categories(segments))
        self.assertEqual(current["stats"]["reference"], 100.0)

    def test_range_bounds_stay_frozen_and_scores_use_the_gap(self):
        count = 24
        closes = [105.8] * count
        highs = [106.0] * count
        lows = [104.0] * count
        opens = [105.8] * count
        volumes = [1000.0] * count
        highs[12] = 106.0
        lows[12] = 99.0
        closes[12] = 105.0
        opens[12] = 105.0
        lows[4] = 90.0
        pivots = [
            _pivot("short", "low", 2, 8, 100.0),
            _pivot("short", "high", 3, 9, 110.0),
            _pivot("short", "low", 11, 12, 101.0),
            _pivot("short", "high", 14, 16, 130.0),
        ]
        directions = [
            _direction("short", 0, 9, "downtrend"),
            _direction("short", 10, count - 1, "range"),
        ]
        quotes = {"highs": highs, "lows": lows, "opens": opens, "volumes": volumes}
        result, _series_value = _run(closes, directions, pivots=pivots, **quotes)
        neutral = _at(_short(result), 10)
        self.assertEqual(neutral["range_bounds"]["upper"], 110.0)
        self.assertEqual(neutral["range_bounds"]["lower"], 100.0)
        self.assertEqual(neutral["range_bounds"]["s0"], 1.0)
        self.assertEqual(neutral["category"], "neutral_range")
        accumulated = _at(_short(result), 13)
        self.assertEqual(accumulated["category"], "suspected_accumulation")
        self.assertEqual(accumulated["start_index"], 13)
        self.assertEqual(accumulated["range_bounds"]["upper"], 110.0)
        self.assertTrue(accumulated["summary"]["range"]["accumulation"]["fake_break"])
        self.assertTrue(accumulated["summary"]["range"]["accumulation"]["range_structure"])
        self.assertEqual(_at(_short(result), 16)["range_bounds"]["upper"], 110.0)
        early, _ignored = _run(
            closes[:11],
            [_direction("short", 0, 9, "downtrend"), _direction("short", 10, 10, "range")],
            pivots=pivots,
            highs=highs[:11],
            lows=lows[:11],
            opens=opens[:11],
            volumes=volumes[:11],
        )
        self.assertFalse(_at(_short(early), 10)["summary"]["range"]["accumulation"]["fake_break"])
        lower_high = [
            _pivot("short", "low", 2, 8, 100.0),
            _pivot("short", "high", 3, 7, 112.0),
            _pivot("short", "high", 4, 9, 110.0),
            _pivot("short", "low", 11, 12, 101.0),
            _pivot("short", "high", 14, 15, 109.0),
        ]
        quiet_highs = [106.0] * count
        quiet_lows = [104.0] * count
        quiet_highs[15] = 111.0
        quiet_lows[15] = 109.0
        closes[15] = 109.5
        narrow, _ignored = _run(
            closes,
            directions,
            pivots=lower_high,
            highs=quiet_highs,
            lows=quiet_lows,
            opens=closes,
            volumes=volumes,
        )
        self.assertEqual(_at(_short(narrow), 16)["category"], "neutral_range")
        self.assertEqual(_at(_short(narrow), 16)["summary"]["accumulation_score"], 3)
        self.assertEqual(_at(_short(narrow), 16)["summary"]["distribution_score"], 2)
        both, _ignored = _run(
            closes,
            directions,
            pivots=lower_high,
            highs=quiet_highs,
            lows=quiet_lows,
            opens=closes,
            volumes=volumes,
            params=_pack(range_score=2, score_gap=2),
        )
        self.assertEqual(_at(_short(both), 16)["category"], "neutral_range")
        missing, _ignored = _run(closes, directions, pivots=pivots)
        self.assertIsNone(_at(_short(missing), 23)["stats"]["volume_bias"])
        self.assertLess(_at(_short(missing), 23)["stats"]["accumulation_score"], 3)
        zeros, _ignored = _run(
            [105.8] * count,
            directions,
            pivots=pivots,
            highs=[106.0] * count,
            lows=[104.0] * count,
            opens=[105.8] * count,
            volumes=[0.0] * count,
        )
        self.assertIsNone(_at(_short(zeros), 23)["stats"]["volume_bias"])

    def test_a_range_break_latches_until_the_new_trend(self):
        count = 20
        closes = [105.0] * count
        closes[14] = 110.4
        closes[15] = 110.4
        closes[16] = 111.0
        pivots = [
            _pivot("short", "low", 2, 8, 100.0),
            _pivot("short", "high", 3, 9, 110.0),
            _pivot("short", "low", 11, 12, 101.0),
        ]
        highs = [112.0] * count
        lows = [99.0] * count
        opens = [105.0] * count
        volumes = [1000.0] * count
        directions = [
            _direction("short", 0, 9, "downtrend"),
            _direction("short", 10, 16, "range"),
            _direction("short", 17, 19, "uptrend", progress=[_progress(17, 111.0)]),
        ]
        result, _series_value = _run(
            closes,
            directions,
            pivots=pivots,
            highs=highs,
            lows=lows,
            opens=opens,
            volumes=volumes,
        )
        segments = _short(result)
        self.assertEqual(_at(segments, 13)["category"], "suspected_accumulation")
        broken = _at(segments, 15)
        self.assertEqual(broken["category"], "transition_pending")
        self.assertEqual(broken["trigger"], "range_break_up")
        self.assertEqual(broken["start_index"], 15)
        self.assertEqual(_at(segments, 13)["category"], "suspected_accumulation")
        born = _at(segments, 17)
        self.assertEqual(born["category"], "up_birth")
        self.assertEqual(born["summary"]["age"], 1)
        equal = list(closes)
        equal[14] = 110.3
        equal[15] = 110.3
        held, _ignored = _run(
            equal,
            directions,
            pivots=pivots,
            highs=highs,
            lows=lows,
            opens=opens,
            volumes=volumes,
        )
        self.assertNotEqual(_at(_short(held), 15)["trigger"], "range_break_up")

    def test_a_direct_reversal_is_birth_and_undecided_waits(self):
        closes = [100.0] * 12
        directions = [
            _direction("short", 0, 5, "uptrend", progress=[_progress(0, 100.0)]),
            _direction("short", 6, 11, "downtrend", progress=[_progress(6, 100.0)]),
        ]
        result, _series_value = _run(closes, directions)
        segments = _short(result)
        self.assertEqual(_at(segments, 4)["category"], "up_birth")
        self.assertEqual(_at(segments, 5)["category"], "up_continue")
        self.assertEqual(_at(segments, 6)["category"], "down_birth")
        self.assertEqual(_at(segments, 6)["summary"]["age"], 1)
        between = [item for item in segments if item["start_index"] == 6]
        self.assertEqual(between[0]["category"], "down_birth")
        undecided, _ignored = _run(
            closes,
            [
                _direction("short", 0, 5, "uptrend", progress=[_progress(0, 100.0)]),
                _direction("short", 6, 11, "undecided", ""),
            ],
        )
        self.assertEqual(_at(_short(undecided), 6)["category"], "transition_pending")
        self.assertEqual(_at(_short(undecided), 6)["trigger"], "unconfirmed")

    def test_bar_zero_uses_the_floor_and_a_nonpositive_scale_is_insufficient(self):
        result, series = _run(
            [100.0],
            [_direction("short", 0, 0, "uptrend", progress=[_progress(0, 100.0)])],
            volatility=(None,),
        )
        birth = _short(result)[0]
        self.assertEqual(birth["category"], "up_birth")
        self.assertEqual(birth["stats"]["s_t"], series.params.volatility_floor)
        bad, _ignored = _run(
            [100.0] * 3,
            [_direction("short", 0, 2, "uptrend", progress=[_progress(0, 100.0)])],
            volatility=(1.0, 0.0, 1.0),
        )
        self.assertEqual(_at(_short(bad), 1)["category"], "insufficient")
        self.assertEqual(_at(_short(bad), 2)["category"], "up_birth")

    def test_the_down_case_mirrors_the_up_case(self):
        up_closes = [100.0] * 16
        up_closes[14] = 98.0
        up_closes[15] = 98.0
        up, _ignored = _run(up_closes, [_direction("short", 0, 15, "uptrend", progress=[_progress(0, 110.0)])])
        down_closes = [100.0] * 16
        down_closes[14] = 102.0
        down_closes[15] = 102.0
        down, _ignored = _run(down_closes, [_direction("short", 0, 15, "downtrend", progress=[_progress(0, 90.0)])])
        self.assertEqual(
            [_MIRROR[item] for item in _categories(_short(up))],
            _categories(_short(down)),
        )

    def test_distribution_mirrors_accumulation(self):
        count = 18
        closes = [105.0] * count
        pivots = [
            _pivot("short", "low", 2, 8, 100.0),
            _pivot("short", "high", 3, 9, 110.0),
            _pivot("short", "low", 11, 12, 101.0),
        ]
        quotes = {
            "highs": [112.0] * count,
            "lows": [99.0] * count,
            "opens": [105.0] * count,
            "volumes": [1000.0] * count,
        }
        quotes["lows"][12] = 99.0
        acc, _ignored = _run(
            closes,
            [_direction("short", 0, 9, "downtrend"), _direction("short", 10, count - 1, "range")],
            pivots=pivots,
            **quotes,
        )
        mirror_pivots = [
            _pivot("short", "high", 2, 8, 200.0 - 100.0),
            _pivot("short", "low", 3, 9, 200.0 - 110.0),
            _pivot("short", "high", 11, 12, 200.0 - 101.0),
        ]
        mirror_quotes = {
            "highs": [200.0 - value for value in quotes["lows"]],
            "lows": [200.0 - value for value in quotes["highs"]],
            "opens": [200.0 - value for value in quotes["opens"]],
            "volumes": quotes["volumes"],
        }
        dist, _ignored = _run(
            [200.0 - value for value in closes],
            [_direction("short", 0, 9, "uptrend"), _direction("short", 10, count - 1, "range")],
            pivots=mirror_pivots,
            **mirror_quotes,
        )
        self.assertEqual(
            [_MIRROR[item] for item in _categories(_short(acc))],
            _categories(_short(dist)),
        )

    def test_prefix_does_not_backfill_the_switch(self):
        closes = [100.0] * 16
        closes[14] = 98.0
        closes[15] = 98.0
        directions = [_direction("short", 0, 15, "uptrend", progress=[_progress(0, 110.0)])]
        full, _series_value = _run(closes, directions)
        early_directions = [_direction("short", 0, 14, "uptrend", progress=[_progress(0, 110.0)])]
        early, _ignored = _run(closes[:15], early_directions)
        self.assertEqual(_at(_short(early), 14)["category"], "up_continue")
        self.assertEqual(_at(_short(full), 15)["start_index"], 15)
        self.assertEqual(_at(_short(full), 14)["category"], "up_continue")

    def test_log_and_uniform_share_geometric_bounds(self):
        geoms = [4.5 + index * 0.01 for index in range(18)]
        pivots = [
            _pivot("short", "low", 1, 2, 4.4),
            _pivot("short", "high", 3, 4, 4.8),
        ]
        directions = [
            _direction("short", 0, 5, "downtrend"),
            _direction("short", 6, 17, "range"),
        ]
        outcomes = {}
        for axis in ("log", "uniform"):
            closes = [math.exp(value) if axis == "log" else value for value in geoms]
            result, _series_value = _run(closes, directions, pivots=pivots, axis=axis, volatility=tuple(0.05 for _value in geoms))
            segment = _at(_short(result), 6)
            outcomes[axis] = (_categories(_short(result)), segment["range_bounds"]["upper"], segment["range_bounds"]["lower"])
        self.assertEqual(outcomes["log"][0], outcomes["uniform"][0])
        self.assertEqual(outcomes["log"][1], outcomes["uniform"][1])
        self.assertEqual(outcomes["log"][2], outcomes["uniform"][2])

    def test_missing_quote_columns_still_classify_and_note_the_gap(self):
        result, _series_value = _run(
            [100.0] * 12,
            [_direction("short", 0, 11, "uptrend", progress=[_progress(0, 100.0)])],
        )
        categories = _categories(_short(result))
        self.assertIn("up_birth", categories)
        self.assertIn("up_continue", categories)
        note = " ".join(result.notes)
        for name in ("最高价", "最低价", "开盘价", "成交量"):
            self.assertIn(name, note)
        self.assertFalse(_at(_short(result), 11)["summary"]["exhaustion"]["wick_reject"])
        self.assertFalse(_at(_short(result), 11)["summary"]["exhaustion"]["volume_stall"])
        self.assertIsNone(_at(_short(result), 11)["stats"]["volume_bias"])

    def test_published_scale_is_reused(self):
        volatility = tuple(0.2 if index else None for index in range(6))
        result, series = _run(
            [100.0] * 6,
            [_direction("short", 0, 5, "uptrend", progress=[_progress(0, 100.0)])],
            volatility=volatility,
        )
        self.assertEqual(_at(_short(result), 0)["summary"]["s_t"], series.params.volatility_floor)
        self.assertEqual(_at(_short(result), 5)["stats"]["s_t"], 0.2)

    def test_scales_do_not_share_a_direction(self):
        closes = [100.0] * 8
        directions = [
            _direction("short", 0, 7, "uptrend", progress=[_progress(0, 100.0)]),
            _direction("long", 0, 7, "downtrend", progress=[_progress(0, 100.0)]),
        ]
        result, _series_value = _run(closes, directions)
        short = _at([item for item in result.objects if item["scale"] == "short"], 4)
        long = _at([item for item in result.objects if item["scale"] == "long"], 4)
        self.assertTrue(short["category"].startswith("up_"))
        self.assertTrue(long["category"].startswith("down_"))

    def test_a_gap_splits_the_phase(self):
        result, series = _run(
            [100.0] * 20,
            [_direction("short", 0, 19, "uptrend", progress=[_progress(0, 100.0)])],
            extra_days={8: 10},
        )
        gap = series.gap_indices[0]
        segments = _short(result)
        self.assertTrue(any(item["start_index"] == gap for item in segments))
        for item in segments:
            self.assertFalse(item["start_index"] < gap <= item["end_index"])


class PhaseDirectionV2Tests(unittest.TestCase):
    def test_v6_records_progress_and_older_versions_do_not(self):
        levels = _plateaus([(0.05, 4), (0.01, 4), (0.09, 4), (0.04, 4), (0.13, 4)])
        older = _direction_snapshot(levels, PARAM_VERSION_V5)
        newer = _direction_snapshot(levels, PARAM_VERSION_V6)
        self.assertEqual(older["methods"]["direction"]["version"], "direction-v1")
        self.assertEqual(newer["methods"]["direction"]["version"], "direction-v2")
        for version in (PARAM_VERSION_V2, PARAM_VERSION_V3, PARAM_VERSION_V4, PARAM_VERSION_V5):
            snapshot = _direction_snapshot(levels, version)
            for item in snapshot["methods"]["direction"]["objects"]:
                self.assertNotIn("progress", item)
        left = older["methods"]["direction"]["objects"]
        right = newer["methods"]["direction"]["objects"]
        self.assertEqual(len(left), len(right))
        for before, after in zip(left, right):
            self.assertIn("progress", after)
            self.assertEqual(before, {key: value for key, value in after.items() if key != "progress"})
        self.assertEqual(
            older["methods"]["direction"]["extra"]["swing_comparisons"],
            newer["methods"]["direction"]["extra"]["swing_comparisons"],
        )
        up = next(item for item in right if item["scale"] == "short" and item["category"] == "uptrend")
        self.assertTrue(up["progress"])
        self.assertEqual(up["progress"][0]["available_index"], up["start_index"])
        self.assertEqual(up["progress"][0]["source"], "establish")

    def test_stall_inherits_the_reference_without_rewriting_the_earlier_row(self):
        base = _plateaus([(0.05, 4), (0.00, 4), (0.09, 4), (0.03, 4), (0.12, 4)])
        result = _direction_judge(base + [0.12] * 20, stall=(6, 20, 40))
        short = [item for item in result.objects if item["scale"] == "short"]
        up, stalled = short[-2:]
        self.assertEqual(stalled["cause"], "stall")
        self.assertEqual(stalled["structure"], "uptrend")
        self.assertEqual(stalled["progress"][0]["source"], "inherit")
        self.assertEqual(stalled["progress"][0]["geometric_price"], up["progress"][-1]["geometric_price"])
        self.assertEqual(up["progress"][0]["available_index"], up["start_index"])

    def test_v0_to_v5_hashes_ignore_opens(self):
        closes = [100.0 * math.exp(value) for value in _plateaus([(0.05, 4), (0.01, 4)])]
        times = _times(len(closes))
        highs = [price * 1.01 for price in closes]
        lows = [price * 0.99 for price in closes]
        volumes = [1000.0] * len(closes)
        opens = list(closes)
        older = build_series(
            times,
            closes,
            interval="1d",
            params=default_params("log", version=PARAM_VERSION_V5),
            highs=highs,
            lows=lows,
            volumes=volumes,
            opens=opens,
        )
        newer = build_series(
            times,
            closes,
            interval="1d",
            params=default_params("log", version=PARAM_VERSION_V6),
            highs=highs,
            lows=lows,
            volumes=volumes,
        )
        self.assertEqual(older.data_hash, newer.data_hash)
        with_open = build_series(
            times,
            closes,
            interval="1d",
            params=default_params("log", version=PARAM_VERSION_V6),
            highs=highs,
            lows=lows,
            volumes=volumes,
            opens=opens,
        )
        self.assertNotEqual(with_open.data_hash, newer.data_hash)
        close_only = build_series(times, closes, interval="1d", params=default_params("log", version=PARAM_VERSION))
        self.assertEqual(len(close_only.data_hash), 32)


class PhasePipelineTests(unittest.TestCase):
    def test_the_same_run_writes_every_method(self):
        levels = _plateaus([(0.05, 4), (0.01, 4), (0.09, 4), (0.04, 4)])
        snapshot = _quoted_snapshot(levels)
        for key in (
            "direction",
            "zone_shape",
            "pause",
            "major_trend",
            "retracement",
            "minor_swing",
            "phase",
        ):
            self.assertEqual(snapshot["methods"][key]["status"], "completed", key)
        self.assertTrue(snapshot["methods"]["phase"]["objects"])
        for scale in ("short", "mid", "long"):
            segments = [item for item in snapshot["methods"]["phase"]["objects"] if item["scale"] == scale]
            self.assertEqual(segments[0]["start_index"], 0)
            self.assertEqual(segments[-1]["end_index"], len(levels) - 1)
            for left, right in zip(segments, segments[1:]):
                self.assertEqual(right["start_index"], left["end_index"] + 1)

    def test_direction_failure_still_leaves_a_phase(self):
        levels = _plateaus([(0.05, 4), (0.01, 4)])
        original = DirectionMethod.judge
        DirectionMethod.judge = lambda self, context, params, upstream: (_ for _ in ()).throw(RuntimeError("故意失败"))
        try:
            snapshot = _quoted_snapshot(levels)
        finally:
            DirectionMethod.judge = original
        self.assertEqual(snapshot["methods"]["direction"]["status"], "failed")
        self.assertEqual(snapshot["methods"]["phase"]["status"], "completed")
        self.assertEqual(snapshot["methods"]["major_trend"]["status"], "skipped")
        self.assertTrue(snapshot["methods"]["zone_shape"]["objects"] or snapshot["methods"]["zone_shape"]["status"] == "completed")
        categories = {item["category"] for item in snapshot["methods"]["phase"]["objects"]}
        self.assertTrue(categories <= {"transition_pending", "insufficient"})

    def test_phase_failure_leaves_the_other_methods(self):
        levels = _plateaus([(0.05, 4), (0.01, 4)])
        original = PhaseMethod.judge
        PhaseMethod.judge = lambda self, context, params, upstream: (_ for _ in ()).throw(RuntimeError("阶段失败"))
        try:
            snapshot = _quoted_snapshot(levels)
        finally:
            PhaseMethod.judge = original
        self.assertEqual(snapshot["methods"]["phase"]["status"], "failed")
        self.assertEqual(snapshot["methods"]["direction"]["status"], "completed")
        self.assertTrue(snapshot["methods"]["direction"]["objects"])
        self.assertEqual(snapshot["methods"]["zone_shape"]["status"], "completed")
        self.assertEqual(snapshot["methods"]["pause"]["status"], "completed")

    def test_prefix_matches_the_as_of_view(self):
        levels = _plateaus([(0.05, 4), (0.01, 4), (0.09, 4), (0.04, 4), (0.13, 4), (0.08, 4)])
        snapshot = _quoted_snapshot(levels)
        for count in (8, len(levels) // 2, len(levels)):
            prefix = _quoted_snapshot(levels[:count])
            shown = _as_of(snapshot, prefix["closes"][-1]["time"])
            left = [item for item in prefix["methods"]["phase"]["objects"] if item["scale"] == "short"]
            right = [item for item in shown["methods"]["phase"]["objects"] if item["scale"] == "short"]
            self.assertEqual([item["id"] for item in left], [item["id"] for item in right])
            for item, other in zip(left, right):
                self.assertEqual(_frozen(item), _frozen(other))
                self.assertEqual(item["end_index"], other["end_index"])
                self.assertEqual(item["status"], other["status"])
            self.assertEqual(
                [row for row in prefix["methods"]["phase"]["extra"]["readings"] if row["scale"] == "short"],
                [row for row in shown["methods"]["phase"]["extra"]["readings"] if row["scale"] == "short"],
            )

    def test_phase_does_not_recompute_pivots(self):
        series = _series([100.0] * 6)
        context = _context(series)
        modules = (
            "analysis.structure.pivots",
            "analysis.structure.boundaries",
            "analysis.structure.zones",
            "analysis.structure.segments",
            "analysis.structure.volatility",
        )
        originals = {}
        for name in modules:
            module = __import__(name, fromlist=["detect_pivots"])
            for attr in ("detect_pivots", "build_boundaries", "build_zones", "build_segments", "compute_volatility"):
                if hasattr(module, attr):
                    originals[(name, attr)] = getattr(module, attr)
                    setattr(module, attr, _refuse)
        try:
            result = PhaseMethod().judge(
                context,
                _pack(),
                {"direction": {"status": "completed", "objects": [_direction("short", 0, 5, "uptrend", progress=[_progress(0, 100.0)])], "extra": {}}},
            )
        finally:
            for (name, attr), original in originals.items():
                setattr(__import__(name, fromlist=[attr]), attr, original)
        self.assertTrue(result.objects)

    def test_hand_built_snapshot_keeps_every_row_and_samples_render(self):
        snapshot = _hand_snapshot(3)
        model = build_view_model(snapshot)
        self.assertEqual(len(model["method_objects"]["phase"]), 3)
        html = render_report(snapshot)
        self.assertIn('aria-label="阶段颜色"', html)
        self.assertIn("上涨延续", html.split('aria-label="阶段颜色"', 1)[1].split("</ul>", 1)[0])
        self.assertNotIn("上涨初生", html.split('aria-label="阶段颜色"', 1)[1].split("</ul>", 1)[0])
        chart_js = (
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "analysis/structure/assets/structure-chart.js"
        ).read_text(encoding="utf-8")
        self.assertIn("lane-tip", chart_js)
        table = html.split('id="method-phase-segments"', 1)[1].split("</table>", 1)[0]
        self.assertEqual(table.count("data-object-id"), 3)
        declared = {item.key for item in PhaseMethod().categories}
        for mark in PhaseMethod().present(snapshot["methods"]["phase"]["objects"][0], {"price_axis": "log"}):
            self.assertIn(mark["kind"], MARK_KINDS)
            self.assertIn(mark["category"], declared)
        sample = PhaseMethod().samples()[0]
        page = build_snapshot(
            _series([100.0 * math.exp(value) for value in sample.levels], axis="log"),
            symbol=sample.symbol,
            engine_version="test",
        )
        self.assertIn("阶段", render_report(page))
        self.assertEqual(page["methods"]["phase"]["version"], "phase-v1")

    def test_illegal_params_raise(self):
        PhaseMethod().check_params(_pack())
        for bad in (
            {"fast_span": 0},
            {"fade_ratio": 0},
            {"fade_ratio": 1.1},
            {"exhaustion_score": 5},
            {"exhaustion_score": 0},
            {"pullback_multiple": 0},
            {"bias_minimum": 0},
        ):
            with self.assertRaises(ValueError):
                PhaseMethod().check_params(_pack(**bad))
        with self.assertRaises(ValueError):
            phase_params_from_dict({"fast_span": 10})


def _plateaus(pairs):
    levels = [0.0, 0.001]
    levels.extend([0.001] * 28)
    for value, count in pairs:
        levels.extend([value] * count)
    return levels


def _direction_snapshot(levels, version):
    closes = [100.0 * math.exp(value) for value in levels]
    series = build_series(
        _times(len(closes)),
        closes,
        interval="1d",
        params=default_params("log", version=version),
    )
    return build_snapshot(series, symbol="PHASE", engine_version="test")


def _direction_judge(levels, stall):
    closes = [100.0 * math.exp(value) for value in levels]
    series = build_series(
        _times(len(closes)),
        closes,
        interval="1d",
        params=default_params("log", version=PARAM_VERSION_V6),
    )
    volatility = compute_volatility(series)
    from analysis.structure.pivots import detect_pivots

    context = build_context(
        series,
        volatility=volatility,
        pivots=detect_pivots(series),
        segments=(),
        boundaries=(),
        zones=(),
        events=(),
    )
    pack = series.params.direction.to_dict()
    pack["stall_bars"] = {"short": stall[0], "mid": stall[1], "long": stall[2]}
    return DirectionMethod().judge(context, pack, {})


def _quoted_snapshot(levels):
    closes = [100.0 * math.exp(value) for value in levels]
    series = build_series(
        _times(len(closes)),
        closes,
        interval="1d",
        params=default_params("log", version=PARAM_VERSION_V6),
        highs=[price * 1.01 for price in closes],
        lows=[price * 0.99 for price in closes],
        volumes=[1000.0] * len(closes),
        opens=closes,
    )
    return build_snapshot(series, symbol="PHASE", engine_version="test")


def _as_of(snapshot, when):
    copied = dict(snapshot)
    identity = dict(snapshot["identity"])
    identity["as_of"] = when
    copied["identity"] = identity
    return display_snapshot(copied)


def _frozen(segment):
    hidden = {"status", "end_index", "end_time", "stats"}
    return {key: value for key, value in segment.items() if key not in hidden}


def _hand_snapshot(count):
    params = default_params("log", version=PARAM_VERSION_V6)
    objects = []
    closes = []
    for index in range(count):
        start = index * 4
        objects.append(
            {
                "id": f"phase:mid:{start}",
                "method": "phase",
                "scale": "mid",
                "category": "up_continue",
                "status": "closed" if index < count - 1 else "open",
                "start_index": start,
                "end_index": start + 3,
                "start_time": f"2024-01-{2 + start:02d}T16:00:00",
                "end_time": f"2024-01-{5 + start:02d}T16:00:00",
                "available_index": start,
                "available_time": f"2024-01-{2 + start:02d}T16:00:00",
                "structure": "uptrend",
                "trigger": "continue",
                "confirm_index": start,
                "summary": {"age": 4, "s_t": 1.0, "reference": 1.0, "z3": 0.0},
                "stats": {"age": 4, "s_t": 1.0},
                "range_bounds": None,
            }
        )
    for index in range(16):
        closes.append(
            {
                "index": index,
                "time": f"2024-01-{2 + index:02d}T16:00:00",
                "close": 100.0 + index,
                "gap_before": False,
            }
        )
    method = PhaseMethod()
    return {
        "kind": "trend_structure_snapshot",
        "identity": {
            "symbol": "HAND",
            "interval": "1d",
            "price_axis": "log",
            "price_axis_label": "对数价格",
            "param_version": PARAM_VERSION_V6,
            "as_of": None,
            "visible_scales": ["mid"],
            "visible_methods": ["phase"],
            "bar_count": 16,
        },
        "reading": "",
        "scale_definition_note": "",
        "quality": {"bar_count": 16, "gap_count": 0, "gaps": [], "grade": "正常", "grade_note": ""},
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
            "phase": {
                "key": "phase",
                "name": "阶段",
                "version": "phase-v1",
                "kind": "run",
                "section": "2.5",
                "requires": ["closes", "pivots", "volatility"],
                "categories": [{"key": item.key, "name": item.name, "color": item.color} for item in method.categories],
                "replaces": [],
                "params": params.phase.to_dict(),
                "status": "completed",
                "reason": "",
                "objects": objects,
                "extra": {"readings": []},
                "notes": [],
            }
        },
    }


def _refuse(*_args, **_kwargs):
    raise AssertionError("阶段判断不应重算拐点、边界或波动")


if __name__ == "__main__":
    unittest.main()
