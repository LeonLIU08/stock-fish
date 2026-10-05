"""合成收盘序列上的趋势结构阶段 0–1 测试。不访问网络。"""
from __future__ import annotations

import json
import math
import unittest
from datetime import date, datetime, timedelta, timezone
from tempfile import TemporaryDirectory

from analysis.structure.config import (
    LONG,
    MID,
    PARAM_VERSION,
    SCALE_NAMES,
    SHORT,
    default_params,
)
from analysis.structure.pivots import PivotDetector, detect_pivots, format_trace
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


def _series_from_levels(levels, *, price_axis="log", extra_days=None, start=100.0):
    closes = [start * math.exp(level) for level in levels]
    return build_series(
        _daily_times(len(levels), extra_days=extra_days),
        closes,
        interval="1d",
        price_axis=price_axis,
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


if __name__ == "__main__":
    unittest.main()
