"""合成收盘序列上的趋势结构阶段 0–5 测试。不访问网络。"""
from __future__ import annotations

import json
import math
import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from analysis.structure.config import (
    LONG,
    MID,
    PARAM_VERSION,
    PARAM_VERSION_V1,
    SCALE_NAMES,
    SHORT,
    default_params,
)
from analysis.structure.boundaries import build_boundaries, format_boundaries
from analysis.structure.html_report import render_report, write_result
from analysis.structure.lifecycle import build_lifecycle, format_events
from analysis.structure.snapshot import build_snapshot, resolve_requested_window
from analysis.structure.view_model import (
    READING_NOTE,
    SCALE_DEFINITION_NOTE,
    build_view_model,
    price_axis_distance,
)
from analysis.structure.zones import _effective_interval, build_zones, format_zones
from analysis.structure.pivots import PivotDetector, detect_pivots, format_trace
from analysis.structure.segments import build_segments, format_segments
from analysis.structure.series import (
    IDENTITY_FILENAME,
    NonPositivePriceError,
    build_series,
    result_directory,
    write_series_identity,
)
from analysis.structure.volatility import compute_volatility, reversal_threshold

HK = timezone(timedelta(hours=8))


def _daily_times(count, start=date(2024, 1, 2), extra_days=None):
    extra_days = extra_days or {}
    cursor = datetime(start.year, start.month, start.day, 16, 0)
    times = []
    for index in range(count):
        if index:
            cursor = cursor + timedelta(days=1 + extra_days.get(index, 0))
        times.append(cursor)
    return times


def _series_from_levels(levels, *, price_axis="log", extra_days=None, start=100.0, params=None):
    closes = [start * math.exp(level) for level in levels]
    return build_series(
        _daily_times(len(levels), extra_days=extra_days),
        closes,
        interval="1d",
        price_axis=price_axis,
        params=params,
    )


def _warm(count=30):
    """小波动热身，让波动尺度先落在 delta_min 以下，避免第一跳把门槛抬高。"""
    levels = [0.0, 0.001]
    levels.extend([0.001] * (count - 2))
    return levels


def _perturb(levels, amplitude=0.0002):
    shifted = []
    for index, level in enumerate(levels):
        sign = 1.0 if (index // 2) % 2 == 0 else -1.0
        if index % 5 == 0:
            sign *= 0.5
        shifted.append(level + sign * amplitude)
    return shifted


def _keys(pivots):
    return [(pivot.scale, pivot.role, pivot.extreme_index, pivot.confirm_index) for pivot in pivots]


class ParamsTests(unittest.TestCase):
    def test_v0_defaults_match_plan(self):
        params = default_params()
        self.assertEqual(params.version, PARAM_VERSION)
        self.assertEqual(params.version, "structure-params-v0")
        self.assertEqual(params.price_axis, "log")
        self.assertEqual(params.ewma_span, 20)
        self.assertEqual(params.vol_clip_multiple, 4.0)
        self.assertGreater(params.volatility_floor, 0.0)
        scales = {scale.name: scale for scale in params.scales}
        self.assertEqual([scale.name for scale in params.scales], [SHORT, MID, LONG])
        self.assertEqual(scales[SHORT].k, 1.5)
        self.assertEqual(scales[MID].k, 3.0)
        self.assertEqual(scales[LONG].k, 6.0)
        self.assertEqual(scales[SHORT].delta_min, 0.005)
        self.assertEqual(scales[MID].delta_min, 0.01)
        self.assertEqual(scales[LONG].delta_min, 0.02)
        self.assertEqual(scales[SHORT].max_span, 60)
        self.assertEqual(scales[MID].max_span, 120)
        self.assertEqual(scales[LONG].max_span, 250)
        self.assertEqual(params.touch_distance_ratio, 0.5)
        self.assertEqual(params.touch_cluster_window, 3)
        self.assertEqual(params.min_touch_clusters, 3)
        self.assertEqual(params.zone_min_touch_clusters, 3)
        self.assertFalse(params.zone_credit_full_span)
        self.assertEqual(params.hard_break_ratio, 1.0)
        self.assertEqual(params.breakout_buffer_ratio, 0.5)
        self.assertEqual(params.breakout_bars, 2)
        self.assertEqual(params.max_pivots_per_line, 8)
        self.assertEqual(params.channel_width_ratio_min, 0.8)
        self.assertEqual(params.channel_width_ratio_max, 1.25)
        self.assertEqual(params.convergence_width_ratio, 0.8)
        self.assertEqual(params.sideways_normalized_slope, 0.25)
        self.assertEqual(params.parallel_slope_gap, 0.35)
        self.assertEqual(params.expire_bars, 20)
        self.assertEqual(params.daily_gap_days, 5)
        self.assertEqual(params.intraday_gap_median_multiple, 3.0)
        self.assertEqual(params.max_zones_per_scale, 3)
        self.assertEqual(default_params("uniform").price_axis, "uniform")

    def test_v1_only_relaxes_zone_touch_counts(self):
        original = default_params()
        relaxed = default_params(version=PARAM_VERSION_V1)
        self.assertEqual(relaxed.version, "structure-params-v1")
        self.assertEqual(relaxed.min_touch_clusters, 3)
        self.assertEqual(relaxed.zone_min_touch_clusters, 2)
        self.assertTrue(relaxed.zone_credit_full_span)
        self.assertEqual(relaxed.touch_distance_ratio, original.touch_distance_ratio)
        self.assertEqual(relaxed.touch_cluster_window, original.touch_cluster_window)
        self.assertEqual([scale.to_dict() for scale in relaxed.scales], [scale.to_dict() for scale in original.scales])
        with self.assertRaises(ValueError):
            default_params(version="structure-params-v2")

    def test_unknown_price_axis_rejected(self):
        with self.assertRaises(ValueError):
            default_params("price")


class SeriesIdentityTests(unittest.TestCase):
    def test_hash_stable_and_sensitive(self):
        times = _daily_times(4)
        closes = [10.0, 11.0, 10.5, 12.0]
        first = build_series(times, closes, interval="1d", price_axis="log")
        second = build_series(list(times), list(closes), interval="1d", price_axis="log")
        self.assertEqual(first.data_hash, second.data_hash)
        self.assertEqual(first.price_basis, "close")
        self.assertEqual(first.param_version, PARAM_VERSION)

        changed = list(closes)
        changed[2] += 0.25
        self.assertNotEqual(
            build_series(times, changed, interval="1d").data_hash,
            first.data_hash,
        )
        moved = list(times)
        moved[1] = moved[1] + timedelta(minutes=5)
        self.assertNotEqual(
            build_series(moved, closes, interval="1d").data_hash,
            first.data_hash,
        )

    def test_hash_uses_instant_and_ignores_price_axis(self):
        utc = [
            datetime(2024, 1, 2, tzinfo=timezone.utc),
            datetime(2024, 1, 3, tzinfo=timezone.utc),
        ]
        hong_kong = [
            datetime(2024, 1, 2, 8, tzinfo=HK),
            datetime(2024, 1, 3, 8, tzinfo=HK),
        ]
        naive = [datetime(2024, 1, 2), datetime(2024, 1, 3)]
        closes = [100.0, 101.0]
        utc_series = build_series(utc, closes, price_axis="log")
        hk_series = build_series(hong_kong, closes, price_axis="uniform")
        self.assertEqual(utc_series.data_hash, hk_series.data_hash)
        self.assertNotEqual(
            build_series(naive, closes).data_hash,
            utc_series.data_hash,
        )
        self.assertEqual(utc_series.price_axis, "log")
        self.assertEqual(hk_series.price_axis, "uniform")

    def test_geometry_follows_price_axis(self):
        times = _daily_times(3)
        closes = [100.0, 110.0, 90.0]
        logged = build_series(times, closes, price_axis="log")
        uniform = build_series(times, closes, price_axis="uniform")
        for bar, close in zip(logged.bars, closes):
            self.assertEqual(bar.close, close)
            self.assertEqual(bar.geometric, math.log(close))
        for bar, close in zip(uniform.bars, closes):
            self.assertEqual(bar.geometric, close)
        self.assertEqual(
            [bar.index for bar in logged.bars],
            [0, 1, 2],
        )

    def test_directory_keeps_each_hash_and_overwrites_same_file(self):
        times = _daily_times(3)
        original = build_series(times, [10.0, 11.0, 12.0], price_axis="log")
        revised_closes = [10.0, 11.5, 12.0]
        revised = build_series(times, revised_closes, price_axis="log")
        same_again = build_series(times, [10.0, 11.0, 12.0], price_axis="log")
        self.assertEqual(original.data_hash, same_again.data_hash)
        self.assertNotEqual(original.data_hash, revised.data_hash)

        with TemporaryDirectory() as tmp:
            first_dir = result_directory(
                tmp, "00700", original.interval, original.price_axis, original.data_hash
            )
            second_dir = result_directory(
                tmp, "00700", revised.interval, revised.price_axis, revised.data_hash
            )
            self.assertEqual(
                first_dir.parts[-4:],
                (original.interval, PARAM_VERSION, "log", original.data_hash),
            )
            self.assertNotEqual(first_dir, second_dir)
            uniform_dir = result_directory(
                tmp, "00700", "1d", "uniform", original.data_hash
            )
            self.assertNotEqual(uniform_dir, first_dir)

            path = write_series_identity(tmp, "00700", original)
            self.assertEqual(path.name, IDENTITY_FILENAME)
            path.write_text("sentinel", encoding="utf-8")
            rewritten = write_series_identity(tmp, "00700", original)
            self.assertEqual(rewritten, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["data_hash"], original.data_hash)
            self.assertEqual(payload["param_version"], PARAM_VERSION)
            self.assertEqual(payload["price_basis"], "close")
            self.assertNotEqual(path.read_text(encoding="utf-8"), "sentinel")
            again = write_series_identity(tmp, "00700", same_again)
            self.assertEqual(again.read_bytes(), path.read_bytes())

            other = write_series_identity(tmp, "00700", revised)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["data_hash"],
                original.data_hash,
            )
            self.assertEqual(
                json.loads(other.read_text(encoding="utf-8"))["data_hash"],
                revised.data_hash,
            )

    def test_daily_gap_is_marked_without_filling_bars(self):
        times = [
            datetime(2024, 1, 2, 16, 0),
            datetime(2024, 1, 3, 16, 0),
            datetime(2024, 1, 20, 16, 0),
        ]
        closes = [100.0, 101.0, 103.0]
        series = build_series(times, closes, interval="1d")
        self.assertEqual(len(series), 3)
        self.assertEqual([bar.index for bar in series.bars], [0, 1, 2])
        self.assertEqual([bar.close for bar in series.bars], closes)
        self.assertEqual([bar.gap_before for bar in series.bars], [False, False, True])
        self.assertEqual(series.gap_indices, (2,))

        # 周五到下周五：日历跨 7 天，交易日正好 5 天，不记间断。
        exact = build_series(
            [datetime(2024, 1, 5), datetime(2024, 1, 12)],
            [100.0, 101.0],
            interval="1d",
        )
        self.assertEqual(exact.gap_indices, ())
        # 周四到下周三：日历跨 6 天，交易日 4 天，不记间断。
        calendar_span = build_series(
            [datetime(2024, 1, 4), datetime(2024, 1, 10)],
            [100.0, 101.0],
            interval="1d",
        )
        self.assertEqual(calendar_span.gap_indices, ())
        # 周五到再下一周的周一：交易日 6 天。
        over = build_series(
            [datetime(2024, 1, 5), datetime(2024, 1, 15)],
            [100.0, 101.0],
            interval="1d",
        )
        self.assertEqual(over.gap_indices, (1,))

    def test_weekend_is_not_a_gap_and_adds_no_bar(self):
        series = build_series(
            [datetime(2024, 1, 5, 16, 0), datetime(2024, 1, 8, 16, 0)],
            [100.0, 102.0],
            interval="1d",
        )
        self.assertEqual(len(series), 2)
        self.assertEqual(series.gap_indices, ())
        self.assertEqual(series.bars[1].index, 1)

    def test_intraday_gap_uses_prior_intervals_only(self):
        start = datetime(2024, 1, 2, 9, 30)
        times = [start + timedelta(minutes=offset) for offset in range(6)]
        times.append(times[-1] + timedelta(minutes=3))
        times.append(times[-1] + timedelta(minutes=4))
        closes = [100.0 + index for index in range(len(times))]
        series = build_series(times, closes, interval="1m")
        self.assertEqual(series.gap_indices, (len(times) - 1,))

        first_hole = build_series(
            [start, start + timedelta(minutes=10)],
            [100.0, 101.0],
            interval="1m",
        )
        self.assertEqual(first_hole.gap_indices, (1,))
        exact = build_series(
            [start, start + timedelta(minutes=3)],
            [100.0, 101.0],
            interval="1m",
        )
        self.assertEqual(exact.gap_indices, ())

    def test_nonpositive_closes_reject_the_whole_series(self):
        times = _daily_times(4)
        with self.assertRaises(NonPositivePriceError) as caught:
            build_series(times, [10.0, 0.0, -1.0, 12.0])
        self.assertEqual(caught.exception.count, 2)
        self.assertEqual(caught.exception.indices, (1, 2))
        with self.assertRaises(NonPositivePriceError):
            build_series(times, [10.0, float("nan"), 11.0, float("inf")])
        with self.assertRaises(ValueError):
            build_series(times, [10.0, 11.0])
        with self.assertRaises(ValueError):
            build_series([times[1], times[0]], [10.0, 11.0])

    def test_prefix_rebuild_matches_independent_input(self):
        times = _daily_times(6)
        closes = [10.0, 10.2, 10.1, 10.8, 10.4, 11.0]
        series = build_series(times, closes, price_axis="log")
        prefix = series.prefix(4)
        again = build_series(times[:4], closes[:4], price_axis="log")
        self.assertEqual(prefix.data_hash, again.data_hash)
        self.assertEqual(prefix.bars, again.bars)
        with self.assertRaises(ValueError):
            series.prefix(0)


class VolatilityTests(unittest.TestCase):
    def test_flat_prices_keep_positive_scale_and_delta_min(self):
        series = build_series(_daily_times(8), [100.0] * 8, price_axis="log")
        vol = compute_volatility(series)
        self.assertIsNone(vol[0])
        self.assertTrue(all(value is not None and value > 0 for value in vol[1:]))
        result = detect_pivots(series)
        self.assertEqual(result.confirmed, ())
        for scale_name in SCALE_NAMES:
            temporary = result.temporary_for(scale_name)
            self.assertEqual(len(temporary), 1)
            self.assertEqual(temporary[0].role, "pending")
            self.assertEqual(temporary[0].extreme_index, 7)
            self.assertEqual(temporary[0].threshold, series.params.scale(scale_name).delta_min)

    def test_clip_does_not_replace_the_close(self):
        levels = [0.0, 0.001] + [0.001] * 10
        levels.append(levels[-1] + 0.20)
        series = _series_from_levels(levels)
        vol = compute_volatility(series)
        raw_change = series.bars[-1].geometric - series.bars[-2].geometric
        self.assertGreater(raw_change, 0.15)
        self.assertLess(vol[-1], 0.05)
        self.assertEqual(series.bars[-1].close, 100.0 * math.exp(levels[-1]))
        self.assertEqual(series.bars[-1].geometric, math.log(series.bars[-1].close))

    def test_volatility_and_gaps_are_causal(self):
        levels = [0.0, 0.01, 0.02, 0.0, 0.03]
        extra = {3: 10}
        series = _series_from_levels(levels, extra_days=extra)
        self.assertIn(3, series.gap_indices)
        full_vol = compute_volatility(series)
        for count in range(2, len(series) + 1):
            prefix = series.prefix(count)
            self.assertEqual(
                [bar.gap_before for bar in series.bars[:count]],
                [bar.gap_before for bar in prefix.bars],
            )
            self.assertEqual(full_vol[:count], compute_volatility(prefix))


class PivotContractTests(unittest.TestCase):
    def test_flat_plateau_confirms_on_the_last_equal_bar(self):
        for price_axis in ("log", "uniform"):
            closes = [100.0] * 5 + [110.0]
            series = build_series(_daily_times(len(closes)), closes, price_axis=price_axis)
            result = detect_pivots(series)
            for scale_name in SCALE_NAMES:
                lows = [pivot for pivot in result.confirmed_for(scale_name) if pivot.role == "low"]
                self.assertEqual(
                    [pivot.extreme_index for pivot in lows],
                    [4],
                    format_trace(series, result),
                )
                pivot = lows[0]
                self.assertEqual(pivot.confirm_index, 5)
                self.assertEqual(pivot.extreme_time, series.bars[4].timestamp)
                self.assertEqual(pivot.confirm_time, series.bars[5].timestamp)
                self.assertEqual(pivot.available_time, pivot.confirm_time)
                self.assertEqual(pivot.price, 100.0)
                self.assertNotEqual(pivot.price, series.bars[5].close)
                anchor = series.bars[pivot.segment_start_index].close
                expected = reversal_threshold(
                    series.params.scale(scale_name),
                    pivot.volatility,
                    price_axis,
                    anchor,
                )
                self.assertEqual(pivot.threshold, expected)
                self.assertLess(pivot.threshold, math.log(110.0 / 100.0) if price_axis == "log" else 10.0)

    def test_high_and_low_plateaus_use_the_last_bar(self):
        for price_axis in ("log", "uniform"):
            levels = _warm()
            peak_indices = []
            for _ in range(3):
                peak_indices.append(len(levels))
                levels.append(0.06)
            low_indices = []
            for _ in range(3):
                low_indices.append(len(levels))
                levels.append(0.0)
            rally = len(levels)
            levels.append(0.06)
            series = _series_from_levels(levels, price_axis=price_axis)
            result = detect_pivots(series)
            for scale_name in SCALE_NAMES:
                highs = [pivot for pivot in result.confirmed_for(scale_name) if pivot.role == "high"]
                lows = [pivot for pivot in result.confirmed_for(scale_name) if pivot.role == "low"]
                self.assertEqual(
                    [pivot.extreme_index for pivot in highs],
                    [peak_indices[-1]],
                    format_trace(series, result),
                )
                self.assertEqual(highs[0].confirm_index, low_indices[0])
                self.assertNotEqual(highs[0].extreme_index, peak_indices[0])
                troughs = [pivot for pivot in lows if pivot.extreme_index in low_indices]
                self.assertEqual(
                    [pivot.extreme_index for pivot in troughs],
                    [low_indices[-1]],
                    format_trace(series, result),
                )
                self.assertEqual(troughs[0].confirm_index, rally)
                self.assertNotEqual(troughs[0].extreme_index, low_indices[0])
                self.assertEqual(troughs[0].available_time, series.bars[rally].timestamp)

    def test_threshold_stays_fixed_when_later_volatility_falls(self):
        levels = [0.0, 0.05]
        levels.extend([0.05] * 40)
        plateau = len(levels) - 1
        levels.append(0.01)
        held = _series_from_levels(levels)
        held_result = detect_pivots(held)
        self.assertEqual(held_result.confirmed_for(SHORT), ())
        temporary = [
            item for item in held_result.temporary_for(SHORT) if item.role == "high"
        ]
        self.assertEqual(len(temporary), 1, format_trace(held, held_result))
        candidate = temporary[0]
        self.assertEqual(candidate.extreme_index, plateau)
        self.assertEqual(candidate.threshold_index, 1)
        reversal = held.bars[candidate.extreme_index].geometric - held.bars[-1].geometric
        counterfactual = reversal_threshold(
            held.params.scale(SHORT),
            compute_volatility(held)[-1],
            "log",
            held.bars[candidate.segment_start_index].close,
        )
        self.assertGreaterEqual(reversal, counterfactual)
        self.assertLess(reversal, candidate.threshold)

        levels.append(0.05 - 0.09)
        finished = _series_from_levels(levels)
        result = detect_pivots(finished)
        highs = [pivot for pivot in result.confirmed_for(SHORT) if pivot.role == "high"]
        self.assertEqual(len(highs), 1, format_trace(finished, result))
        self.assertEqual(highs[0].extreme_index, plateau)
        self.assertEqual(highs[0].threshold_index, 1)
        self.assertEqual(highs[0].threshold, candidate.threshold)
        self.assertEqual(result.confirmed_for(MID), ())
        self.assertEqual(result.confirmed_for(LONG), ())

    def test_candidate_moves_and_then_freezes(self):
        for price_axis in ("log", "uniform"):
            levels = _warm()
            first = len(levels)
            levels.append(0.08)
            second = len(levels)
            levels.append(0.14)
            levels.append(0.06)
            series = _series_from_levels(levels, price_axis=price_axis)
            at_first = detect_pivots(series, bar_count=first + 1)
            at_second = detect_pivots(series, bar_count=second + 1)
            final = detect_pivots(series)
            self.assertEqual(at_first.confirmed, at_second.confirmed)
            self.assertTrue(at_first.confirmed)
            self.assertTrue(all(pivot.role == "low" for pivot in at_first.confirmed))
            for scale_name in SCALE_NAMES:
                early = at_first.temporary_for(scale_name)
                later = at_second.temporary_for(scale_name)
                self.assertEqual([item.role for item in early], ["high"])
                self.assertEqual(early[0].extreme_index, first)
                self.assertEqual(later[0].extreme_index, second)
                confirmed_highs = [
                    pivot.extreme_index
                    for pivot in final.confirmed_for(scale_name)
                    if pivot.role == "high"
                ]
                self.assertEqual(confirmed_highs, [second], format_trace(series, final))
            self.assertEqual(
                at_first.confirmed,
                tuple(pivot for pivot in final.confirmed if pivot.confirm_index <= first),
            )

    def test_prefix_matches_incremental_feed_and_full_replay(self):
        samples = []
        for price_axis in ("log", "uniform"):
            zigzag = _warm() + [0.08, 0.16, 0.08, 0.02, 0.10]
            samples.append(_series_from_levels(zigzag, price_axis=price_axis))
            samples.append(_series_from_levels(_plateau_levels(), price_axis=price_axis))
            gapped, _peak, _gap_at = _gapped_levels()
            samples.append(
                _series_from_levels(gapped[0], extra_days=gapped[1], price_axis=price_axis)
            )
        for series in samples:
            self._assert_prefix_consistent(series)

    def test_noise_keeps_mid_and_long_indices(self):
        base_levels = _warm() + [0.08, 0.16, 0.08, 0.02, 0.10]
        noisy_levels = _perturb(base_levels)
        for price_axis in ("log", "uniform"):
            base = detect_pivots(_series_from_levels(base_levels, price_axis=price_axis))
            noisy = detect_pivots(_series_from_levels(noisy_levels, price_axis=price_axis))
            for scale_name in (MID, LONG):
                self.assertEqual(_keys(base.confirmed_for(scale_name)), _keys(noisy.confirmed_for(scale_name)))
                self.assertTrue(base.confirmed_for(scale_name))

    def test_short_swing_inside_a_long_rise_does_not_change_long_pivots(self):
        # 计划原文写「低于短尺度门槛」。按 4.1，没到短尺度门槛就不会多出一个短拐点。
        # 这里的波动高于短尺度门槛、低于长尺度门槛。
        for price_axis in ("log", "uniform"):
            base_levels, local, peak = _rise_levels(dip=False)
            dip_levels, dip_local, dip_peak = _rise_levels(dip=True)
            self.assertEqual(local, dip_local)
            self.assertEqual(peak, dip_peak)
            base = detect_pivots(_series_from_levels(base_levels, price_axis=price_axis))
            dipped = detect_pivots(_series_from_levels(dip_levels, price_axis=price_axis))
            self.assertEqual(base.confirmed_for(LONG), dipped.confirmed_for(LONG))
            self.assertEqual(
                [pivot.extreme_index for pivot in dipped.confirmed_for(LONG) if pivot.role == "high"],
                [peak],
                format_trace(_series_from_levels(dip_levels, price_axis=price_axis), dipped),
            )
            dipped_short_highs = [
                pivot.extreme_index
                for pivot in dipped.confirmed_for(SHORT)
                if pivot.role == "high"
            ]
            base_short_highs = [
                pivot.extreme_index
                for pivot in base.confirmed_for(SHORT)
                if pivot.role == "high"
            ]
            self.assertIn(local, dipped_short_highs)
            self.assertNotIn(local, base_short_highs)
            self.assertNotIn(local, [
                pivot.extreme_index
                for pivot in dipped.confirmed_for(LONG)
                if pivot.role == "high"
            ])
            self.assertGreater(len(dipped.confirmed_for(SHORT)), len(base.confirmed_for(SHORT)))

    def test_gap_drops_the_unconfirmed_extreme(self):
        for price_axis in ("log", "uniform"):
            (levels, extra), peak, gap_at = _gapped_levels()
            series = _series_from_levels(levels, extra_days=extra, price_axis=price_axis)
            self.assertIn(gap_at, series.gap_indices)
            result = detect_pivots(series)
            confirmed_extremes = {pivot.extreme_index for pivot in result.confirmed}
            self.assertNotIn(peak, confirmed_extremes, format_trace(series, result))
            self.assertTrue(any(pivot.extreme_index >= gap_at for pivot in result.confirmed))
            for pivot in result.confirmed:
                for index in range(pivot.segment_start_index + 1, pivot.confirm_index + 1):
                    self.assertFalse(series.bars[index].gap_before)
                self.assertFalse(pivot.extreme_index < gap_at <= pivot.confirm_index)
            self._assert_prefix_consistent(series)

    def test_uniform_threshold_converts_delta_min_into_a_price_gap(self):
        closes = [100.0, 100.05] + [100.05] * 20 + [100.40]
        held = build_series(_daily_times(len(closes)), closes, price_axis="uniform")
        held_result = detect_pivots(held)
        self.assertEqual(held_result.confirmed_for(SHORT), ())
        closes.append(100.80)
        series = build_series(_daily_times(len(closes)), closes, price_axis="uniform")
        result = detect_pivots(series)
        lows = [pivot for pivot in result.confirmed_for(SHORT) if pivot.role == "low"]
        self.assertEqual(len(lows), 1, format_trace(series, result))
        expected = 100.0 * (math.exp(0.005) - 1.0)
        self.assertEqual(lows[0].threshold, expected)
        self.assertNotEqual(lows[0].threshold, 0.005)
        self.assertEqual(lows[0].extreme_index, 0)
        self.assertEqual(series.bars[3].geometric, series.bars[3].close)

    def test_below_short_threshold_stays_temporary(self):
        levels = _warm()
        levels.append(0.004)
        series = _series_from_levels(levels)
        result = detect_pivots(series)
        self.assertEqual(result.confirmed, ())
        highs = [item for item in result.temporary_for(SHORT) if item.role == "high"]
        self.assertEqual(len(highs), 1)
        self.assertEqual(highs[0].extreme_index, len(series) - 1)

    def _assert_prefix_consistent(self, series):
        detector = PivotDetector(series)
        full = detect_pivots(series)
        midpoint = max(1, len(series) // 2)
        mid_snapshot = None
        mid_length = None
        for count in range(1, len(series) + 1):
            stepped = detector.advance_to(count)
            one_shot = detect_pivots(series, bar_count=count)
            self.assertEqual(stepped, one_shot)
            self.assertEqual(
                one_shot.confirmed,
                tuple(pivot for pivot in full.confirmed if pivot.confirm_index < count),
            )
            if count == midpoint:
                mid_snapshot = stepped
                mid_length = len(stepped.confirmed)
        self.assertIsNotNone(mid_snapshot)
        detector.advance_to(len(series))
        self.assertEqual(len(mid_snapshot.confirmed), mid_length)
        again = detector.advance_to(len(series))
        self.assertEqual(again, full)


class SegmentTests(unittest.TestCase):
    def test_closed_segment_count_excludes_the_open_end(self):
        for price_axis in ("log", "uniform"):
            series = _series_from_levels(_warm() + [0.08], price_axis=price_axis)
            pivots = detect_pivots(series)
            before = [pivot.to_dict() for pivot in pivots.confirmed]
            result = build_segments(series, pivots)
            self.assertEqual(before, [pivot.to_dict() for pivot in pivots.confirmed])
            for scale_name in SCALE_NAMES:
                confirmed = pivots.confirmed_for(scale_name)
                segments = result.segments_for(scale_name)
                temporary = result.temporary_for(scale_name)
                self.assertEqual(len(confirmed), 1, format_trace(series, pivots))
                self.assertEqual(len(segments), 0)
                self.assertEqual(len(temporary), 1, format_segments(result))
                self.assertEqual(temporary[0].direction, "up")
                self.assertEqual(temporary[0].start.role, "low")
                self.assertEqual(temporary[0].end.role, "high")
                self.assertIs(temporary[0].start, confirmed[0])
                self.assertNotIn(temporary[0].end.extreme_index, [pivot.extreme_index for pivot in confirmed])

    def test_segment_fields_follow_the_two_confirmed_ends(self):
        for price_axis in ("log", "uniform"):
            series = _series_from_levels(_warm() + [0.08, 0.16, 0.08, 0.02, 0.10], price_axis=price_axis)
            pivots = detect_pivots(series)
            result = build_segments(series, pivots)
            self._assert_contracts(series, pivots, result)
            for scale_name in SCALE_NAMES:
                confirmed = pivots.confirmed_for(scale_name)
                segments = result.segments_for(scale_name)
                self.assertGreaterEqual(len(confirmed), 2)
                self.assertEqual(len(segments), len(confirmed) - 1, format_segments(result))
                self.assertIs(segments[0].start, confirmed[0])
                self.assertIs(segments[0].end, confirmed[1])

    def test_log_change_is_log_return_and_uniform_change_is_price_gap(self):
        levels = _warm() + [0.08, 0.02]
        logged = build_segments(_series_from_levels(levels, price_axis="log"))
        uniform = build_segments(_series_from_levels(levels, price_axis="uniform"))
        for segment in logged.segments_for(SHORT):
            self.assertAlmostEqual(segment.change, math.log(segment.end.price / segment.start.price))
            self.assertNotEqual(segment.change, segment.end.price - segment.start.price)
        for segment in uniform.segments_for(SHORT):
            self.assertAlmostEqual(segment.change, segment.end.price - segment.start.price)
            self.assertNotAlmostEqual(segment.change, math.log(segment.end.price / segment.start.price))

    def test_interior_dip_sets_adverse_excursion_without_splitting(self):
        levels = _warm()
        levels.extend([0.03, 0.028, 0.06, 0.0])
        series = _series_from_levels(levels, price_axis="log")
        result = build_segments(series)
        segment = result.segments_for(SHORT)[0]
        self.assertEqual(segment.start.extreme_index, 0)
        self.assertEqual(segment.end.extreme_index, len(_warm()) + 2)
        self.assertEqual(segment.direction, "up")
        self.assertAlmostEqual(segment.adverse_excursion, 0.002)
        self.assertGreater(segment.max_deviation, 0.0)
        self.assertEqual(len(result.segments_for(SHORT)), 1)
        self.assertEqual(len(result.segments_for(LONG)), 1)

    def test_one_interior_bar_off_the_chord_sets_deviation(self):
        series = _series_from_levels([0.0, 0.02, 0.20, 0.0], price_axis="log")
        result = build_segments(series)
        segment = result.segments_for(SHORT)[0]
        self.assertEqual((segment.start.extreme_index, segment.end.extreme_index), (0, 2))
        self.assertAlmostEqual(segment.change, 0.20)
        self.assertEqual(segment.bar_span, 2)
        self.assertAlmostEqual(segment.slope, 0.10)
        self.assertAlmostEqual(segment.max_deviation, 0.08)
        self.assertAlmostEqual(segment.adverse_excursion, 0.0)
        self.assertAlmostEqual(segment.change_multiple, segment.change / segment.end.volatility)

    def test_gap_keeps_each_side_separate(self):
        for price_axis in ("log", "uniform"):
            (levels, extra), _peak, gap_at = _gapped_levels()
            series = _series_from_levels(levels, extra_days=extra, price_axis=price_axis)
            pivots = detect_pivots(series)
            result = build_segments(series, pivots)
            self.assertIn(gap_at, series.gap_indices)
            for scale_name in SCALE_NAMES:
                confirmed = pivots.confirmed_for(scale_name)
                segments = result.segments_for(scale_name)
                self.assertEqual(
                    len(segments),
                    _connectable_pairs(series, confirmed),
                    format_segments(result),
                )
                self.assertNotEqual(len(segments), max(len(confirmed) - 1, 0))
                for segment in segments:
                    self.assertFalse(_range_crosses_gap(series, segment.start.extreme_index, segment.end.extreme_index))
                self.assertTrue(segments)
                self.assertTrue(any(pivot.extreme_index < gap_at for pivot in confirmed))
                self.assertTrue(any(pivot.extreme_index >= gap_at for pivot in confirmed))

    def test_finer_segments_inside_a_longer_one_are_linked_without_rewriting_pivots(self):
        for price_axis in ("log", "uniform"):
            levels = _warm()
            levels.extend([0.03, 0.04, 0.032, 0.08, 0.03])
            series = _series_from_levels(levels, price_axis=price_axis)
            pivots = detect_pivots(series)
            before = [pivot.to_dict() for pivot in pivots.confirmed]
            result = build_segments(series, pivots)
            self.assertEqual(before, [pivot.to_dict() for pivot in pivots.confirmed])

            short = result.segments_for(SHORT)
            mid = result.segments_for(MID)
            long = result.segments_for(LONG)
            self.assertEqual(len(short), 3, format_segments(result))
            self.assertEqual(len(mid), 1, format_segments(result))
            self.assertEqual(len(long), 1, format_segments(result))
            self.assertEqual(long[0].contains, (mid[0].id,))
            self.assertEqual(mid[0].contains, tuple(segment.id for segment in short))
            self.assertTrue(all(segment_id.startswith("short:") for segment_id in mid[0].contains))
            self.assertFalse(any(segment_id.startswith("short:") for segment_id in long[0].contains))
            for segment in short:
                self.assertEqual(segment.contains, ())
                self.assertGreaterEqual(segment.start.extreme_index, long[0].start.extreme_index)
                self.assertLessEqual(segment.end.extreme_index, long[0].end.extreme_index)
            short_highs = [
                pivot.extreme_index
                for pivot in pivots.confirmed_for(SHORT)
                if pivot.role == "high"
            ]
            long_highs = [
                pivot.extreme_index
                for pivot in pivots.confirmed_for(LONG)
                if pivot.role == "high"
            ]
            self.assertGreater(len(short_highs), len(long_highs))

    def test_prefix_segments_match_and_do_not_rewrite_earlier_ones(self):
        samples = []
        for price_axis in ("log", "uniform"):
            samples.append(_series_from_levels(_warm() + [0.08, 0.16, 0.08, 0.02, 0.10], price_axis=price_axis))
            dipped = _warm()
            dipped.extend([0.03, 0.04, 0.032, 0.08, 0.03])
            samples.append(_series_from_levels(dipped, price_axis=price_axis))
            gapped, _peak, _gap_at = _gapped_levels()
            samples.append(_series_from_levels(gapped[0], extra_days=gapped[1], price_axis=price_axis))
        for series in samples:
            full = build_segments(series)
            for count in range(1, len(series) + 1):
                partial = build_segments(series, bar_count=count)
                again = build_segments(series.prefix(count))
                self.assertEqual(partial, again)
                self.assertEqual(
                    partial.segments,
                    tuple(segment for segment in full.segments if segment.end.confirm_index < count),
                )

    def test_flat_series_has_no_segment(self):
        series = build_series(_daily_times(8), [100.0] * 8, price_axis="log")
        result = build_segments(series)
        self.assertEqual(result.segments, ())
        self.assertEqual(result.temporary, ())

    def _assert_contracts(self, series, pivots, result):
        self.assertTrue(result.segments, format_segments(result))
        for scale_name in SCALE_NAMES:
            confirmed = pivots.confirmed_for(scale_name)
            segments = result.segments_for(scale_name)
            self.assertEqual(len(segments), _connectable_pairs(series, confirmed), format_segments(result))
        for segment in result.segments:
            self.assertEqual(segment.bar_span, segment.end.extreme_index - segment.start.extreme_index)
            self.assertEqual(segment.change, segment.end.geometric_price - segment.start.geometric_price)
            self.assertEqual(segment.slope, segment.change / segment.bar_span)
            self.assertEqual(segment.change_multiple, segment.change / segment.volatility)
            self.assertEqual(segment.volatility, segment.end.volatility)
            self.assertEqual(segment.confirm_time, segment.end.confirm_time)
            self.assertEqual(segment.available_time, segment.end.available_time)
            self.assertEqual(segment.confirm_time, series.bars[segment.end.confirm_index].timestamp)
            self.assertIn(segment.direction, ("up", "down"))
            if segment.direction == "up":
                self.assertEqual((segment.start.role, segment.end.role), ("low", "high"))
                self.assertGreater(segment.change, 0.0)
            else:
                self.assertEqual((segment.start.role, segment.end.role), ("high", "low"))
                self.assertLess(segment.change, 0.0)
            if series.price_axis == "log":
                self.assertAlmostEqual(segment.change, math.log(segment.end.price / segment.start.price))
            else:
                self.assertEqual(segment.change, segment.end.price - segment.start.price)
            self.assertGreaterEqual(segment.max_deviation, 0.0)
            self.assertGreaterEqual(segment.adverse_excursion, 0.0)
            self.assertFalse(
                _range_crosses_gap(series, segment.start.extreme_index, segment.end.extreme_index)
            )
        confirmed_ids = {segment.id for segment in result.segments}
        for segment in result.temporary:
            self.assertTrue(segment.id.startswith("tmp:"))
            self.assertNotIn(segment.id, confirmed_ids)
            self.assertNotIn(segment.end.extreme_index, [
                pivot.extreme_index
                for pivot in pivots.confirmed_for(segment.scale)
            ])
        for segment in result.segments:
            self.assertTrue(set(segment.contains).issubset(confirmed_ids))
            if segment.scale == "long":
                self.assertTrue(all(item.startswith("mid:") for item in segment.contains))
            elif segment.scale == "mid":
                self.assertTrue(all(item.startswith("short:") for item in segment.contains))
            else:
                self.assertEqual(segment.contains, ())


class BoundaryTests(unittest.TestCase):
    def test_rising_lows_validate_an_upward_support(self):
        for price_axis in ("log", "uniform"):
            series = _series_from_levels(_rising_support_levels(), price_axis=price_axis)
            pivots = detect_pivots(series)
            before = [pivot.to_dict() for pivot in pivots.confirmed]
            result = build_boundaries(series, pivots)
            self.assertEqual(before, [pivot.to_dict() for pivot in pivots.confirmed])
            support = _boundary(result, "short", "support", 0, 53)
            self.assertEqual(support.status, "validated", format_boundaries(result))
            self.assertTrue(support.primary)
            self.assertTrue(support.emits_events)
            self.assertGreater(support.normalized_slope, 0.0)
            self.assertEqual(support.normalized_slope, support.slope / support.volatility)
            self.assertGreaterEqual(len(support.touch_clusters), 3)
            self.assertEqual(support.failed_constraints, ())
            self.assertEqual(support.simplicity_penalty, 0.0)
            self.assertEqual(support.context_fit, "未使用")
            self.assertIsNotNone(support.score)
            self.assertEqual(support.score.simplicity_penalty, 0.0)
            self.assertEqual(support.score.context_fit, "未使用")
            parts = (
                support.score.geometry,
                support.score.touch_quality,
                support.score.significance,
                support.score.path_integrity,
                support.score.span,
            )
            self.assertAlmostEqual(support.score.total, sum(parts) / len(parts))
            shorter = _boundary(result, "short", "support", 0, 45)
            self.assertEqual(shorter.status, "validated")
            self.assertFalse(shorter.primary)
            self.assertEqual(shorter.alternate_of, support.id)
            self.assertFalse(shorter.emits_events)
            pair = _boundary(result, "short", "support", 45, 53)
            self.assertEqual(pair.status, "candidate")
            self.assertEqual(pair.failed_constraints, ("min_touch_clusters",))
            self.assertIsNone(pair.score)
            self.assertFalse(pair.emits_events)

    def test_falling_highs_validate_downward_resistance(self):
        for price_axis in ("log", "uniform"):
            series = _series_from_levels(_falling_resistance_levels(), price_axis=price_axis)
            result = build_boundaries(series)
            matches = [
                item
                for item in result.validated("short", "resistance")
                if item.primary and item.normalized_slope < 0.0
            ]
            self.assertTrue(matches, format_boundaries(result))
            self.assertTrue(matches[0].emits_events)

    def test_deep_low_between_two_higher_lows_is_not_a_validated_support(self):
        for price_axis in ("log", "uniform"):
            series = _series_from_levels(_broken_support_levels(), price_axis=price_axis)
            result = build_boundaries(series)
            line = _boundary(result, "short", "support", 37, 53)
            self.assertNotEqual(line.status, "validated", format_boundaries(result))
            self.assertIn("hard_break", line.failed_constraints)
            self.assertIn("no_skipped_break", line.failed_constraints)
            self.assertGreater(line.max_break_depth, 1.0)
            self.assertFalse(line.emits_events)

    def test_break_depth_just_inside_validates_and_just_outside_names_the_constraint(self):
        for price_axis in ("log", "uniform"):
            clean = build_boundaries(_series_from_levels(_rising_support_levels(), price_axis=price_axis))
            reference = _boundary(clean, "short", "support", 0, 53)
            under = build_boundaries(_series_from_levels(
                _rising_support_levels(_middle_level(reference, 0.99, price_axis)),
                price_axis=price_axis,
            ))
            over = build_boundaries(_series_from_levels(
                _rising_support_levels(_middle_level(reference, 1.01, price_axis)),
                price_axis=price_axis,
            ))
            inside = _boundary(under, "short", "support", 0, 53)
            outside = _boundary(over, "short", "support", 0, 53)
            self.assertEqual(inside.status, "validated", format_boundaries(under))
            self.assertLessEqual(inside.max_break_depth, 1.0)
            self.assertEqual(inside.failed_constraints, ())
            self.assertEqual(outside.status, "rejected", format_boundaries(over))
            self.assertGreater(outside.max_break_depth, 1.0)
            self.assertIn("hard_break", outside.failed_constraints)
            self.assertIsNone(outside.score)

    def test_gap_lets_each_side_form_a_line_without_crossing(self):
        for price_axis in ("log", "uniform"):
            levels, extra = _gapped_support_levels()
            series = _series_from_levels(levels, price_axis=price_axis, extra_days=extra)
            result = build_boundaries(series)
            gap_at = series.gap_indices[0]
            supports = [item for item in result.for_scale("short") if item.role == "support"]
            self.assertTrue(supports)
            self.assertFalse(any(item.start_index < gap_at <= item.end_index for item in result.boundaries))
            self.assertTrue(any(item.end_index < gap_at and item.status == "validated" for item in supports))
            self.assertTrue(any(item.start_index >= gap_at and item.status == "validated" for item in supports))

    def test_prefix_keeps_slope_status_and_score(self):
        for price_axis in ("log", "uniform"):
            series = _series_from_levels(_rising_support_levels(), price_axis=price_axis)
            full = build_boundaries(series)
            for count in range(1, len(series) + 1):
                partial = build_boundaries(series, bar_count=count)
                earlier = {item.id: item for item in full.boundaries if item.confirm_index < count}
                self.assertEqual({item.id for item in partial.boundaries}, set(earlier))
                for item in partial.boundaries:
                    prior = earlier[item.id]
                    self.assertEqual(item.slope, prior.slope)
                    self.assertEqual(item.intercept, prior.intercept)
                    self.assertEqual(item.normalized_slope, prior.normalized_slope)
                    self.assertEqual(item.status, prior.status)
                    self.assertEqual(item.failed_constraints, prior.failed_constraints)
                    self.assertEqual(item.score, prior.score)
                    self.assertEqual(item.pivot_indices, prior.pivot_indices)
                    self.assertEqual(item.volatility, prior.volatility)

    def test_flat_series_has_no_boundary(self):
        series = build_series(_daily_times(8), [100.0] * 8, price_axis="log")
        self.assertEqual(build_boundaries(series).boundaries, ())


def _boundary(result, scale, role, start_index, end_index):
    matches = [
        item
        for item in result.for_scale(scale)
        if item.role == role and item.start_index == start_index and item.end_index == end_index
    ]
    if len(matches) != 1:
        raise AssertionError(format_boundaries(result))
    return matches[0]


def _rising_support_levels(middle_level=None, middle_index=45):
    """四个抬高低点落在同一条几何直线上。中间那个可以单独下移。"""
    slope = 0.0002
    later = (37, 45, 53)
    levels = _warm()
    for index in later:
        level = slope * index
        if middle_level is not None and index == middle_index:
            level = middle_level
        levels.extend([0.03] * 4)
        levels.extend([level] * 4)
    levels.extend([0.03] * 2)
    return levels


def _middle_level(reference, factor, price_axis):
    target = reference.line_at(45) - factor * reference.volatility
    if price_axis == "log":
        return target - math.log(100.0)
    return math.log(target / 100.0)


def _broken_support_levels():
    levels = _warm()
    for level in (0.02, 0.0, 0.02):
        levels.extend([0.03] * 4)
        levels.extend([level] * 4)
    levels.extend([0.03] * 2)
    return levels


def _falling_resistance_levels():
    levels = _warm()
    for high in (0.03, 0.02, 0.01):
        levels.extend([high] * 4)
        levels.extend([0.0] * 4)
    levels.extend([0.02])
    return levels


def _gapped_support_levels():
    left = _rising_support_levels()
    right = _rising_support_levels()
    return left + right, {len(left): 12}


def _connectable_pairs(series, pivots):
    count = 0
    for left, right in zip(pivots, pivots[1:]):
        if left.role == right.role or right.extreme_index <= left.extreme_index:
            continue
        if _range_crosses_gap(series, left.extreme_index, right.extreme_index):
            continue
        count += 1
    return count


def _range_crosses_gap(series, start_index, end_index):
    return any(series.bars[index].gap_before for index in range(start_index + 1, end_index + 1))


def _plateau_levels():
    levels = _warm()
    levels.extend([0.06, 0.06, 0.06, 0.0, 0.0, 0.0, 0.06])
    return levels


def _rise_levels(dip):
    levels = _warm()
    levels.append(0.08)
    local = len(levels)
    levels.append(0.11)
    levels.append(0.094 if dip else 0.14)
    peak = len(levels)
    levels.append(0.18)
    levels.append(0.10)
    return levels, local, peak


def _gapped_levels():
    levels = _warm()
    levels.append(0.08)
    peak = len(levels)
    levels.append(0.14)
    gap_at = len(levels)
    levels.extend([0.0, 0.08, 0.0])
    return (levels, {gap_at: 10}), peak, gap_at


class ZoneTests(unittest.TestCase):
    def test_v1_validates_a_channel_whose_overlap_keeps_only_some_touches(self):
        levels = _rail_levels(pairs=2)
        for price_axis in ("log", "uniform"):
            strict = build_zones(_series_from_levels(levels, price_axis=price_axis))
            relaxed = build_zones(
                _series_from_levels(
                    levels,
                    price_axis=price_axis,
                    params=default_params(price_axis, version=PARAM_VERSION_V1),
                )
            )
            before = next(zone for zone in strict.for_scale("short") if zone.primary and zone.label == "channel")
            after = next(zone for zone in relaxed.for_scale("short") if zone.primary and zone.label == "channel")
            self.assertIn("lower_touches", before.failed_constraints)
            self.assertEqual(before.status, "candidate")
            self.assertEqual(after.lower_touch_clusters, before.lower_touch_clusters)
            self.assertEqual(after.upper_touch_clusters, before.upper_touch_clusters)
            self.assertEqual(after.status, "validated", format_zones(relaxed))
            self.assertTrue(after.draw_on_main)
            lines = build_boundaries(
                _series_from_levels(
                    levels,
                    price_axis=price_axis,
                    params=default_params(price_axis, version=PARAM_VERSION_V1),
                )
            )
            thin = next(item for item in lines.for_scale("short") if len(item.touch_clusters) < 3)
            self.assertNotEqual(thin.status, "validated")

    def test_parallel_rising_rails_form_a_channel(self):
        for price_axis in ("log", "uniform"):
            series = _series_from_levels(_rail_levels(), price_axis=price_axis)
            pivots = detect_pivots(series)
            before = [pivot.to_dict() for pivot in pivots.confirmed]
            result = build_zones(series, pivots)
            self.assertEqual(before, [pivot.to_dict() for pivot in pivots.confirmed])
            channel = _primary(result, "short", "channel")
            self.assertEqual(channel.status, "validated", format_zones(result))
            self.assertGreater(channel.lower_slope, 0.0)
            self.assertGreater(channel.upper_slope, 0.0)
            self.assertGreaterEqual(channel.width_ratio, 0.8)
            self.assertLessEqual(channel.width_ratio, 1.25)
            self.assertLess(channel.normalized_slope_gap, 0.35)
            self.assertGreaterEqual(channel.lower_touch_clusters, 3)
            self.assertGreaterEqual(channel.upper_touch_clusters, 3)
            self.assertTrue(channel.draw_on_main)
            self.assertTrue(channel.emits_events)
            self.assertEqual(channel.simplicity_penalty, 0.0)
            self.assertEqual(channel.context_fit, "未使用")
            self.assertIsNotNone(channel.score)
            parts = (
                channel.score.geometry,
                channel.score.touch_quality,
                channel.score.significance,
                channel.score.path_integrity,
                channel.score.span,
            )
            self.assertAlmostEqual(channel.score.total, sum(parts) / len(parts))
            for index in range(channel.effective_start, channel.effective_end + 1):
                self.assertGreater(channel.width_at(index), 0.0)
            mid = _primary(result, "mid", "channel")
            overlap = [
                item
                for item in result.overlaps
                if set(item.zone_ids) == {channel.id, mid.id}
            ]
            self.assertEqual(len(overlap), 1)
            self.assertEqual(overlap[0].note, "不合并为一条证据")
            self.assertNotEqual(channel.score.total, mid.score.total + channel.score.total)

    def test_flat_parallel_rails_are_sideways_and_rising_floor_is_convergence(self):
        sideways = _primary(build_zones(_series_from_levels(_rail_levels(a=0.0))), "short", "sideways")
        self.assertEqual(sideways.status, "validated")
        self.assertTrue(sideways.draw_on_main)
        self.assertLess(abs(sideways.lower_slope), 1e-9)
        self.assertLess(abs(sideways.upper_slope), 1e-9)

        convergence = _primary(
            build_zones(_series_from_levels(_rail_levels(upper=lambda index: 0.12, lower=lambda index: 0.001 * index))),
            "short",
            "convergence",
        )
        self.assertEqual(convergence.status, "validated")
        self.assertLess(convergence.width_ratio, 0.8)
        self.assertGreater(convergence.end_width, 0.0)
        self.assertGreater(convergence.lower_slope, convergence.upper_slope)
        self.assertNotIn(convergence.label, ("triangle", "wedge"))

    def test_widening_pair_stays_an_other_boundary_pair(self):
        other = _primary(
            build_zones(_series_from_levels(_rail_levels(a=0.0, upper=lambda index: 0.04 + 0.003 * index, lower=lambda index: 0.0))),
            "short",
            "other",
        )
        self.assertEqual(other.status, "validated")
        self.assertGreater(other.width_ratio, 1.25)
        self.assertFalse(other.draw_on_main)
        self.assertTrue(other.emits_events)

    def test_nonpositive_width_stops_the_effective_interval(self):
        series = _series_from_levels(_rail_levels())
        zones = build_zones(series)
        channel = _primary(zones, "short", "channel")
        bounds = build_boundaries(series)
        lower = next(item for item in bounds.boundaries if item.id == channel.lower_id)
        upper = next(item for item in bounds.boundaries if item.id == channel.upper_id)
        start, end = channel.projected_start, channel.projected_end
        above = lower.line_at(start) + 0.05
        below = lower.line_at(end) - 0.01
        slope = (below - above) / (end - start)
        tilted = replace(upper, slope=slope, intercept=above - slope * start)
        interval = _effective_interval(series, lower, tilted, start, end, 1e9)
        self.assertIsNotNone(interval)
        effective_end = interval[1]
        self.assertLess(effective_end, end)
        self.assertGreater(tilted.line_at(effective_end) - lower.line_at(effective_end), 0.0)
        self.assertLessEqual(tilted.line_at(effective_end + 1) - lower.line_at(effective_end + 1), 0.0)

    def test_gap_keeps_zones_on_each_side(self):
        for price_axis in ("log", "uniform"):
            left = _rail_levels()
            levels = left + _rail_levels()
            series = _series_from_levels(levels, price_axis=price_axis, extra_days={len(left): 12})
            result = build_zones(series)
            gap_at = series.gap_indices[0]
            self.assertFalse(any(zone.effective_start < gap_at <= zone.effective_end for zone in result.zones))
            self.assertTrue(any(zone.effective_end < gap_at for zone in result.validated()))
            self.assertTrue(any(zone.effective_start >= gap_at for zone in result.validated()))

    def test_prefix_keeps_zone_geometry(self):
        series = _series_from_levels(_rail_levels(), price_axis="log")
        full = build_zones(series)
        for count in range(1, len(series) + 1):
            partial = build_zones(series, bar_count=count)
            earlier = {zone.id: zone for zone in full.zones if zone.confirm_index < count}
            self.assertEqual({zone.id for zone in partial.zones}, set(earlier))
            for zone in partial.zones:
                prior = earlier[zone.id]
                self.assertEqual(zone.label, prior.label)
                self.assertEqual(zone.status, prior.status)
                self.assertEqual((zone.effective_start, zone.effective_end), (prior.effective_start, prior.effective_end))
                self.assertEqual(zone.lower_slope, prior.lower_slope)
                self.assertEqual(zone.upper_slope, prior.upper_slope)
                self.assertEqual(zone.width_ratio, prior.width_ratio)
                self.assertEqual(zone.score, prior.score)
            partial_pairs = {item.zone_ids for item in partial.overlaps}
            earlier_ids = set(earlier)
            full_pairs = {item.zone_ids for item in full.overlaps if set(item.zone_ids).issubset(earlier_ids)}
            self.assertEqual(partial_pairs, full_pairs)


def _primary(result, scale, label):
    matches = [zone for zone in result.validated(scale, label) if zone.primary]
    if len(matches) != 1:
        raise AssertionError(format_zones(result))
    return matches[0]


def _rail_levels(a=0.001, width=0.08, pairs=4, step=6, upper=None, lower=None):
    """热身之后交替放高点和低点，中间用两条线的中线填充。"""
    warm = _warm()
    events = []
    cursor = len(warm)
    for _ in range(pairs):
        events.append(("H", cursor))
        cursor += step
        events.append(("L", cursor))
        cursor += step
    last_low = events[-1][1]
    confirm = last_low + step
    levels = [None] * (confirm + 1)
    for index, value in enumerate(warm):
        levels[index] = value

    def low_at(index):
        return a * index if lower is None else lower(index)

    def high_at(index):
        return a * index + width if upper is None else upper(index)

    for kind, index in events:
        levels[index] = high_at(index) if kind == "H" else low_at(index)
    levels[confirm] = high_at(confirm)
    for index in range(len(warm), len(levels)):
        if levels[index] is None:
            levels[index] = (low_at(index) + high_at(index)) / 2.0
    return levels


class LifecycleTests(unittest.TestCase):
    def test_two_closes_beyond_the_buffer_break_a_support_without_rewriting_it(self):
        base = _rising_support_levels()
        original = build_lifecycle(_series_from_levels(base))
        support = original.boundary("short:support:0-53")
        self.assertEqual(support.status, "validated")
        broken = build_lifecycle(_series_from_levels(base + _levels_from_line(support, len(base), 2, -0.8)))
        after = broken.boundary(support.id)
        self.assertEqual(after.status, "broken", format_events(broken))
        self.assertEqual(after.slope, support.slope)
        self.assertEqual(after.intercept, support.intercept)
        self.assertEqual(after.score, support.score)
        self.assertEqual(after.revision, support.revision)
        events = broken.events_for(support.id)
        self.assertEqual([event.event_type for event in events], ["validated", "breakout"])
        self.assertEqual(events[0].price_cross_index, None)
        self.assertGreaterEqual(events[0].available_index, support.confirm_index)
        self.assertEqual(events[1].price_cross_index, len(base))
        self.assertEqual(events[1].event_index, len(base) + 1)
        self.assertGreater(events[1].available_index, events[0].available_index)
        self.assertIn("支撑下方", events[1].reason)
        for boundary in broken.boundaries:
            if boundary.status == "candidate":
                self.assertFalse(any(event.event_type == "breakout" for event in broken.events_for(boundary.id)))

    def test_one_close_or_a_shallow_dip_does_not_break_the_line(self):
        base = _rising_support_levels()
        support = build_lifecycle(_series_from_levels(base)).boundary("short:support:0-53")
        for extra in (
            _levels_from_line(support, len(base), 1, -0.8),
            _levels_from_line(support, len(base), 2, -0.2),
        ):
            result = build_lifecycle(_series_from_levels(base + extra))
            self.assertEqual(result.boundary(support.id).status, "validated", format_events(result))
            self.assertEqual(
                [event.event_type for event in result.events_for(support.id)],
                ["validated"],
            )

    def test_no_breakout_before_the_expiry_window_marks_the_line_expired(self):
        base = _rising_support_levels()
        support = build_lifecycle(_series_from_levels(base)).boundary("short:support:0-53")
        held = build_lifecycle(_series_from_levels(base + _levels_from_line(support, len(base), 25, 0.2)))
        after = held.boundary(support.id)
        self.assertEqual(after.status, "expired", format_events(held))
        self.assertEqual(after.slope, support.slope)
        expired = held.events_for(support.id)[-1]
        self.assertEqual(expired.event_type, "expired")
        self.assertEqual(expired.event_index, support.end_index + 20)
        self.assertIsNone(expired.price_cross_index)
        self.assertNotIn("买入", expired.reason)
        self.assertNotIn("卖出", expired.reason)

    def test_zone_breakout_is_not_backfilled_and_leaves_the_main_chart(self):
        base = _rail_levels()
        original = build_lifecycle(_series_from_levels(base))
        zone = next(item for item in original.zones if item.scale == "short" and item.label == "channel" and item.primary)
        broken = build_lifecycle(_series_from_levels(base + _levels_from_line(zone, len(base), 2, -0.8, below_zone=True)))
        after = broken.zone(zone.id)
        self.assertEqual(after.status, "broken", format_events(broken))
        self.assertFalse(after.draw_on_main)
        self.assertEqual(after.lower_slope, zone.lower_slope)
        self.assertEqual(after.upper_slope, zone.upper_slope)
        events = broken.events_for(zone.id)
        self.assertEqual([event.event_type for event in events], ["validated", "breakout"])
        self.assertLess(events[1].price_cross_index, events[1].event_index)
        self.assertGreaterEqual(events[1].event_index, zone.confirm_index)
        self.assertEqual(events[1].object_kind, "zone")

    def test_prefix_events_and_revisions_stay_put_when_later_bars_arrive(self):
        base = _rising_support_levels()
        extended = base + _levels_from_line(
            build_lifecycle(_series_from_levels(base)).boundary("short:support:0-53"),
            len(base),
            2,
            -0.8,
        )
        full = build_lifecycle(_series_from_levels(extended))
        early = build_lifecycle(_series_from_levels(base))
        self.assertEqual(
            tuple(event for event in full.events if event.available_index < len(base)),
            early.events,
        )
        self.assertEqual(full.boundary("short:support:0-53").revision, early.boundary("short:support:0-53").revision)
        later = [item for item in full.boundaries if item.confirm_index >= len(base)]
        self.assertTrue(later)
        self.assertGreater(min(item.revision for item in later), early.boundary("short:support:0-53").revision)
        for left, right in zip(full.boundaries, full.boundaries[1:]):
            if left.confirm_index < right.confirm_index:
                self.assertLessEqual(left.revision, right.revision)
            elif left.confirm_index == right.confirm_index:
                self.assertEqual(left.revision, right.revision)

    def test_event_prefix_matches_a_shorter_replay(self):
        series = _series_from_levels(_rising_support_levels())
        full = build_lifecycle(series)
        for count in (20, 40, len(series)):
            partial = build_lifecycle(series, bar_count=count)
            self.assertEqual(
                partial.events,
                tuple(event for event in full.events if event.available_index < count),
            )


def _levels_from_line(structure, start_index, count, volatility_multiple, below_zone=False):
    """在已有直线的延长线上追加若干根，偏移量以该结构的波动尺度计。"""
    levels = []
    for step in range(count):
        index = start_index + step
        if below_zone:
            geometric = structure.lower_at(index) + volatility_multiple * structure.volatility
        else:
            geometric = structure.line_at(index) + volatility_multiple * structure.volatility
        levels.append(geometric - math.log(100.0))
    return levels


class ReportTests(unittest.TestCase):
    def test_hand_built_snapshot_renders_without_adding_lines(self):
        slope = 0.01
        intercept = math.log(100.0)
        snapshot = _one_support_snapshot(slope, intercept)
        snapshot["boundaries"].append(_rejected_boundary())
        model = build_view_model(snapshot)
        html = render_report(snapshot)
        embedded = _embedded_json(html, "structure-snapshot")
        view = _embedded_json(html, "structure-view-model")
        self.assertEqual(embedded["identity"]["data_hash"], snapshot["identity"]["data_hash"])
        validated = [
            item
            for item in embedded["boundaries"]
            if item["primary"] and item["status"] == "validated" and item["role"] == "support"
        ]
        self.assertEqual(len(validated), 1)
        drawn = [item for item in view["boundaries"] if item["role"] == "support" and item["status"] == "validated"]
        self.assertEqual(drawn, model["boundaries"])
        self.assertEqual(len(drawn), 1)
        self.assertEqual(drawn[0]["id"], "mid:support:0-10")
        self.assertEqual(len(drawn[0]["solid"]), 2)
        for index, price in drawn[0]["solid"]:
            self.assertAlmostEqual(price, math.exp(slope * index + intercept))
        self.assertEqual(html.count('data-table="primary" data-status="validated" data-role="support"'), 1)
        self.assertIn('id="validated-support-count" data-count="1"', html)
        self.assertIn("TrendChart.mount(chartRoot, viewModel)", html)
        self.assertIn(READING_NOTE, html)
        self.assertIn(SCALE_DEFINITION_NOTE, html)
        self.assertIn("横向区间", html)
        self.assertIn("归一化斜率差小于", html)

    def test_chart_keeps_broken_and_extra_trend_zones(self):
        snapshot = _one_support_snapshot(0.01, math.log(100.0))
        snapshot["zones"] = [
            _hand_zone("quota", "channel", "validated", main_chart=False),
            _hand_zone("broken", "convergence", "broken"),
            _hand_zone("ended", "sideways", "expired"),
            _hand_zone("aside", "other", "validated"),
            _hand_zone("alt", "channel", "validated", primary=False),
            _hand_zone("wait", "channel", "candidate"),
        ]
        drawn = {item["id"]: item["status"] for item in build_view_model(snapshot)["zones"]}
        self.assertEqual(drawn, {"quota": "validated", "broken": "broken", "ended": "expired"})
        html = render_report(snapshot)
        self.assertEqual(html.count(">主图</td>"), 3)
        self.assertEqual(html.count(">主图不画</td>"), 3)
        self.assertIn('class="swatch channel"', html)
        self.assertIn('class="swatch convergence"', html)
        self.assertIn('class="swatch sideways"', html)
        root = Path(__file__).resolve().parents[1]
        for name in ("detect_pivots", "build_boundaries", "build_zones", "build_lifecycle", "build_segments"):
            self.assertNotIn(name, (root / "analysis/structure/view_model.py").read_text(encoding="utf-8"))
            self.assertNotIn(name, (root / "analysis/structure/html_report.py").read_text(encoding="utf-8"))
            self.assertNotIn(name, (root / "analysis/structure/assets/structure-chart.js").read_text(encoding="utf-8"))

    def test_uniform_endpoints_use_the_price_itself(self):
        snapshot = _one_support_snapshot(1.5, 100.0, price_axis="uniform")
        line = build_view_model(snapshot)["boundaries"][0]
        self.assertEqual(line["solid"], [[0, 100.0], [10, 115.0]])

    def test_price_axis_distance_matches_the_axis(self):
        self.assertAlmostEqual(price_axis_distance(100, 200, "log"), price_axis_distance(200, 400, "log"))
        self.assertAlmostEqual(price_axis_distance(100, 110, "uniform"), price_axis_distance(400, 410, "uniform"))
        self.assertGreater(price_axis_distance(100, 200, "uniform"), price_axis_distance(200, 400, "log"))

    def test_requested_window_does_not_change_scale_definitions(self):
        short = build_snapshot(
            _series_from_levels(_warm(12)),
            symbol="WIN",
            requested_start="2024-01-01",
            requested_end="2024-02-01",
            visible_scales=["long"],
            engine_version="test",
        )
        long = build_snapshot(
            _series_from_levels(_rising_support_levels()),
            symbol="WIN",
            requested_start="2020-01-01",
            requested_end="2024-12-31",
            visible_scales=["short", "mid"],
            engine_version="test",
        )
        self.assertEqual(short["params"], long["params"])
        self.assertEqual(short["identity"]["scale_definition"], long["identity"]["scale_definition"])
        self.assertEqual([item["k"] for item in short["identity"]["scale_definition"]["scales"]], [1.5, 3.0, 6.0])
        self.assertEqual([item["max_span"] for item in short["identity"]["scale_definition"]["scales"]], [60, 120, 250])
        self.assertNotEqual(short["identity"]["requested_start"], long["identity"]["requested_start"])
        self.assertEqual(short["identity"]["visible_scales"], ["long"])
        self.assertEqual([item["name"] for item in short["params"]["scales"]], ["short", "mid", "long"])
        self.assertEqual(
            resolve_requested_window(start=date(2024, 1, 1), end=date(2024, 3, 1), years=None, today=date(2024, 6, 1)),
            (date(2024, 1, 1), date(2024, 3, 1)),
        )
        self.assertEqual(
            resolve_requested_window(start=None, end=None, years=1, today=date(2024, 6, 1))[0],
            date(2023, 6, 2),
        )
        with self.assertRaises(ValueError):
            resolve_requested_window(start=date(2024, 1, 1), end=None, years=2, today=date(2024, 6, 1))

    def test_as_of_hides_later_structures_without_dropping_closes(self):
        series = _series_from_levels(_rising_support_levels())
        snapshot = build_snapshot(series, symbol="ASOF", engine_version="test")
        support = next(item for item in snapshot["boundaries"] if item["id"] == "short:support:0-53")
        earlier = datetime.fromisoformat(support["available_time"]) - timedelta(seconds=1)
        snapshot["identity"]["as_of"] = earlier.isoformat()
        model = build_view_model(snapshot)
        self.assertNotIn(support["id"], [item["id"] for item in model["boundaries"]])
        self.assertEqual(len(model["closes"]), len(snapshot["closes"]))
        self.assertTrue(all(item["id"] in {row["id"] for row in snapshot["boundaries"]} for item in model["boundaries"]))

    def test_as_of_rolls_a_later_breakout_back_to_validated(self):
        base = _rising_support_levels()
        support = build_lifecycle(_series_from_levels(base)).boundary("short:support:0-53")
        snapshot = build_snapshot(
            _series_from_levels(base + _levels_from_line(support, len(base), 2, -0.8)),
            symbol="ROLL",
            engine_version="test",
        )
        validated = next(
            item
            for item in snapshot["events"]
            if item["object_id"] == support.id and item["event_type"] == "validated"
        )
        snapshot["identity"]["as_of"] = validated["available_time"]
        line = next(item for item in build_view_model(snapshot)["boundaries"] if item["id"] == support.id)
        self.assertEqual(line["status"], "validated")
        self.assertIsNone(line["breakout"])

    def test_gap_lowers_quality_without_a_crossing_line(self):
        series = _series_from_levels([0.0] * 6 + [0.01] * 6, extra_days={6: 12})
        snapshot = build_snapshot(series, symbol="GAP", engine_version="test")
        self.assertEqual(snapshot["quality"]["grade"], "降级")
        self.assertGreater(snapshot["quality"]["gap_count"], 0)
        self.assertEqual(snapshot["quality"]["crossing_ids"], [])
        self.assertEqual(snapshot["quality"]["rejected_non_positive"], 0)

    def test_report_files_follow_the_result_directory(self):
        series = _series_from_levels(_rising_support_levels(), price_axis="log")
        other = _series_from_levels(_rising_support_levels(), price_axis="uniform")
        with TemporaryDirectory() as folder:
            log_snapshot = build_snapshot(series, symbol="DIR", engine_version="test")
            uniform_snapshot = build_snapshot(other, symbol="DIR", engine_version="test")
            log_written = write_result(folder, log_snapshot)
            uniform_written = write_result(folder, uniform_snapshot)
            self.assertNotEqual(log_written["directory"], uniform_written["directory"])
            self.assertTrue(log_written["snapshot"].is_file())
            self.assertTrue(log_written["report"].is_file())
            saved = json.loads(log_written["snapshot"].read_text(encoding="utf-8"))
            self.assertEqual(saved["identity"]["data_hash"], series.data_hash)
            self.assertEqual(saved["identity"]["price_axis"], "log")
            self.assertIn("对数价格", log_written["report"].read_text(encoding="utf-8"))
            again = write_result(folder, log_snapshot)
            self.assertEqual(again["snapshot"], log_written["snapshot"])


def _one_support_snapshot(slope, intercept, price_axis="log"):
    params = default_params(price_axis)
    if price_axis == "log":
        end_price = math.exp(slope * 10 + intercept)
        start_price = math.exp(intercept)
    else:
        start_price = intercept
        end_price = slope * 10 + intercept
    return {
        "kind": "trend_structure_snapshot",
        "reading": READING_NOTE,
        "scale_definition_note": SCALE_DEFINITION_NOTE,
        "identity": {
            "symbol": "HAND",
            "interval": "1d",
            "start_time": "2024-01-02T16:00:00",
            "end_time": "2024-01-16T16:00:00",
            "requested_start": "2024-01-02",
            "requested_end": "2024-01-16",
            "param_version": PARAM_VERSION,
            "data_hash": "hand-built",
            "price_basis": "close",
            "price_axis": price_axis,
            "price_axis_label": "对数价格" if price_axis == "log" else "均匀价格",
            "engine_version": "",
            "engine_version_note": "取不到 git 短哈希，引擎代码版本留空。",
            "as_of": None,
            "visible_scales": ["mid"],
            "data_source": "synthetic",
            "bar_count": 2,
            "scale_definition": {
                "note": SCALE_DEFINITION_NOTE,
                "param_version": PARAM_VERSION,
                "scales": [scale.to_dict() for scale in params.scales],
            },
        },
        "quality": {
            "bar_count": 2,
            "gap_count": 0,
            "gaps": [],
            "rejected_non_positive": 0,
            "grade": "正常",
            "grade_note": "没有间断。",
            "crossing_ids": [],
        },
        "volatility": {"scales": [], "unit": "对数收益" if price_axis == "log" else "价格差"},
        "pivot_stats": [],
        "pivots": [],
        "temporary_pivots": [],
        "segments": [],
        "temporary_segments": [],
        "boundaries": [
            {
                "id": "mid:support:0-10",
                "revision": 1,
                "scale": "mid",
                "role": "support",
                "status": "validated",
                "primary": True,
                "alternate_of": None,
                "start_index": 0,
                "end_index": 10,
                "start_time": "2024-01-02T16:00:00",
                "end_time": "2024-01-16T16:00:00",
                "available_time": "2024-01-18T16:00:00",
                "slope": slope,
                "intercept": intercept,
                "normalized_slope": 0.2,
                "span": 10,
                "touch_clusters": [[0], [5], [10]],
                "touch_error_median": 0.1,
                "touch_error_p90": 0.2,
                "max_break_depth": 0.3,
                "longest_violation_bars": 0,
                "constraints": [{"name": "two_pivots", "passed": True, "detail": "两个拐点"}],
                "failed_constraints": [],
                "simplicity_penalty": 0.0,
                "context_fit": "未使用",
                "score": {
                    "geometry": 1.0,
                    "touch_quality": 1.0,
                    "significance": 1.0,
                    "path_integrity": 1.0,
                    "span": 1.0,
                    "simplicity_penalty": 0.0,
                    "context_fit": "未使用",
                    "total": 1.0,
                },
            }
        ],
        "zones": [],
        "events": [],
        "relations": {
            "boundary_groups": [],
            "zone_groups": [],
            "overlaps": [],
            "segment_contains": [],
            "overlap_note": "不合并为一条证据",
        },
        "closes": [
            {"index": 0, "time": "2024-01-02T16:00:00", "close": start_price, "gap_before": False},
            {"index": 10, "time": "2024-01-16T16:00:00", "close": end_price, "gap_before": False},
        ],
        "params": params.to_dict(),
    }


def _hand_zone(zone_id, label, status, *, primary=True, main_chart=False):
    return {
        "id": zone_id,
        "revision": 1,
        "scale": "mid",
        "label": label,
        "status": status,
        "primary": primary,
        "alternate_of": None if primary else "quota",
        "emits_events": False,
        "draw_on_main": False,
        "main_chart": main_chart,
        "lower_id": zone_id + ":lower",
        "upper_id": zone_id + ":upper",
        "lower_slope": 0.0,
        "lower_intercept": math.log(90.0),
        "upper_slope": 0.0,
        "upper_intercept": math.log(110.0),
        "volatility": 0.01,
        "effective_start": 0,
        "effective_end": 10,
        "projected_start": 0,
        "projected_end": 10,
        "start_time": "2024-01-02T16:00:00",
        "end_time": "2024-01-16T16:00:00",
        "confirm_index": 10,
        "confirm_time": "2024-01-16T16:00:00",
        "available_time": "2024-01-18T16:00:00",
        "start_width": 0.2,
        "end_width": 0.2,
        "width_ratio": 1.0,
        "slope_gap": 0.0,
        "normalized_slope_gap": 0.0,
        "lower_touch_clusters": 3,
        "upper_touch_clusters": 3,
        "max_break_depth": 0.1,
        "constraints": [],
        "failed_constraints": [],
        "simplicity_penalty": 0.0,
        "context_fit": "未使用",
        "score": None,
    }


def _rejected_boundary():
    return {
        "id": "mid:support:2-8",
        "revision": 1,
        "scale": "mid",
        "role": "support",
        "status": "rejected",
        "primary": True,
        "alternate_of": None,
        "start_index": 2,
        "end_index": 8,
        "start_time": "2024-01-04T16:00:00",
        "end_time": "2024-01-12T16:00:00",
        "available_time": "2024-01-14T16:00:00",
        "slope": 0.0,
        "intercept": math.log(90.0),
        "normalized_slope": -1.0,
        "span": 6,
        "touch_clusters": [[2], [8]],
        "touch_error_median": None,
        "touch_error_p90": None,
        "max_break_depth": 2.0,
        "longest_violation_bars": 3,
        "constraints": [{"name": "hard_break", "passed": False, "detail": "破坏过深"}],
        "failed_constraints": ["hard_break"],
        "simplicity_penalty": 0.0,
        "context_fit": "未使用",
        "score": None,
    }


def _embedded_json(html, element_id):
    marker = f'id="{element_id}"'
    start = html.index(marker)
    start = html.index(">", start) + 1
    end = html.index("</script>", start)
    return json.loads(html[start:end])


if __name__ == "__main__":
    unittest.main()
