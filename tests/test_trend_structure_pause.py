"""停顿：上涨之后的旗形和三角旗。合成高低收，不访问网络。"""
import math
import unittest
from dataclasses import replace
from datetime import datetime, timedelta

from analysis.structure.config import (
    PARAM_VERSION,
    PARAM_VERSION_V1,
    PARAM_VERSION_V2,
    PARAM_VERSION_V3,
    PauseParams,
    default_params,
)
from analysis.structure.context import build_context
from analysis.structure.html_report import render_report
from analysis.structure.methods.base import MARK_KINDS
from analysis.structure.methods.pause import (
    FLAG,
    PENNANT,
    PauseMethod,
    _Candidate,
    _choose,
    _exit_direction,
    example_geometry,
)
from analysis.structure.pivots import detect_pivots
from analysis.structure.series import build_series
from analysis.structure.snapshot import build_snapshot
from analysis.structure.view_model import build_view_model, display_snapshot


class PauseAcceptanceTests(unittest.TestCase):
    def test_flag_breaks_up_without_using_the_signal_bar(self):
        for axis in ("log", "uniform"):
            with self.subTest(axis=axis):
                geo = example_geometry("flag", "up")
                result = _judge(geo, axis=axis)
                obj = _one(self, result, "pause:15:22:flag")
                self.assertEqual(obj["category"], FLAG)
                self.assertEqual(obj["status"], "broken")
                reasons = _reasons(result, obj["id"])
                self.assertEqual(reasons[0], "旗形")
                self.assertTrue(reasons[-1].startswith("向上"))
                fit = obj["fits"][-1]
                self.assertEqual(fit["consolidation_end"], 34)
                self.assertTrue(all(point["index"] <= 35 - 1 - 2 for point in fit["highs"]))
                raised = example_geometry("flag", "up")
                raised["highs"][35] += 0.4
                again = _judge(raised, axis=axis)
                other = again.objects[0]["fits"][-1]
                for key in ("upper_slope", "lower_slope", "upper_intercept", "lower_intercept", "width_ratio"):
                    self.assertEqual(fit[key], other[key])

    def test_pennant_category(self):
        result = _judge(example_geometry("pennant", "up"))
        obj = result.objects[0]
        self.assertEqual(obj["category"], PENNANT)
        fit = obj["fits"][-1]
        self.assertLessEqual(fit["upper_slope"], -0.2 * obj["anchor"])
        self.assertGreaterEqual(fit["lower_slope"], 0.2 * obj["anchor"])
        self.assertGreaterEqual(fit["width_ratio"], 0.3)
        self.assertLessEqual(fit["width_ratio"], 0.75)
        self.assertEqual(obj["status"], "broken")

    def test_width_between_the_two_shapes_is_only_an_other_row(self):
        geo = example_geometry("flag", "forming")
        geo["highs"][31] -= 0.0012
        result = _judge(geo)
        self.assertEqual(result.objects, [])
        self.assertEqual(len(result.extra["others"]), 1)
        ratio = result.extra["others"][0]["width_ratio"]
        self.assertGreater(ratio, 0.75)
        self.assertLess(ratio, 0.85)

    def test_pole_thresholds_and_a_flat_path(self):
        pack = _pack(pole_min=8, pole_max=8, consolidation_min=12, consolidation_max=12)
        exact = example_geometry("flag")
        for step, index in enumerate(range(15, 23)):
            exact["closes"][index] = 1.0 + 0.04 * step / 7.0
            exact["highs"][index] = exact["closes"][index] + 0.002
            exact["lows"][index] = exact["closes"][index] - 0.002
        self.assertTrue(_judge(exact, pause=pack).objects)
        short = example_geometry("flag")
        for step, index in enumerate(range(15, 23)):
            short["closes"][index] = 1.0 + 0.04 * step / 7.0
            short["highs"][index] = short["closes"][index] + 0.002
            short["lows"][index] = short["closes"][index] - 0.002
        short["closes"][22] -= 0.0001
        short["highs"][22] = short["closes"][22] + 0.002
        short["lows"][22] = short["closes"][22] - 0.002
        self.assertFalse(_judge(short, pause=pack).objects)

        high = example_geometry("flag")
        high["highs"][22] = high["closes"][22] + 0.01
        self.assertTrue(_judge(high, pause=pack).objects)
        high["highs"][22] = high["closes"][22] + 0.011
        self.assertFalse(_judge(high, pause=pack).objects)

        flat = example_geometry("flag")
        for index in range(15, 23):
            flat["closes"][index] = 1.0
            flat["highs"][index] = 1.002
            flat["lows"][index] = 0.998
        self.assertFalse(_judge(flat, pause=pack).objects)

        kinked = example_geometry("flag")
        # 把中间一根收到前一根之下，使路径刚好等于净涨 / 0.65。
        previous = kinked["closes"][17]
        following = kinked["closes"][19]
        other_path = sum(abs(kinked["closes"][i] - kinked["closes"][i - 1]) for i in range(16, 23))
        other_path -= abs(kinked["closes"][18] - previous) + abs(following - kinked["closes"][18])
        pole = kinked["closes"][22] - kinked["closes"][15]
        target_path = pole / 0.65
        dip = (target_path - other_path - (previous - kinked["closes"][18]) - (following - kinked["closes"][18])) / 2
        kinked["closes"][18] -= dip
        kinked["highs"][18] = kinked["closes"][18] + 0.002
        kinked["lows"][18] = kinked["closes"][18] - 0.002
        passed = _judge(kinked, pause=pack)
        self.assertTrue(passed.objects)
        self.assertAlmostEqual(passed.objects[0]["efficiency"], 0.65, places=6)
        kinked["closes"][18] -= 0.002
        kinked["highs"][18] = kinked["closes"][18] + 0.002
        kinked["lows"][18] = kinked["closes"][18] - 0.002
        self.assertFalse(_judge(kinked, pause=pack).objects)

    def test_drawdown_and_range_boundaries(self):
        pack = _pack(pole_min=8, pole_max=8, consolidation_min=12, consolidation_max=12)
        lifted = example_geometry("flag", "forming")
        for index in range(23, 35):
            lifted["highs"][index] += 0.02
            lifted["lows"][index] += 0.02
            lifted["closes"][index] += 0.02
        result = _judge(lifted, pause=pack)
        self.assertEqual(result.objects[0]["fits"][-1]["drawdown"], 0.0)

        exact = example_geometry("flag", "forming")
        old_low = min(exact["lows"][23:35])
        pole = exact["closes"][22] - exact["closes"][15]
        shift = old_low - (exact["closes"][22] - 0.5 * pole)
        # 浮点会把刚好 0.5 抬过线，退到仍不超过的那一档。
        while True:
            trial = example_geometry("flag", "forming")
            for index in range(23, 35):
                trial["highs"][index] -= shift
                trial["lows"][index] -= shift
                trial["closes"][index] -= shift
            drawdown = (exact["closes"][22] - min(trial["lows"][23:35])) / pole
            if drawdown <= 0.5:
                break
            shift = math.nextafter(shift, 0.0)
        held = _judge(trial, pause=pack)
        self.assertTrue(held.objects)
        self.assertLessEqual(held.objects[0]["fits"][-1]["drawdown"], 0.5)
        self.assertAlmostEqual(held.objects[0]["fits"][-1]["drawdown"], 0.5, places=6)
        for index in range(23, 35):
            trial["highs"][index] -= 0.002
            trial["lows"][index] -= 0.002
            trial["closes"][index] -= 0.002
        self.assertFalse(_judge(trial, pause=pack).objects)

        wide = example_geometry("flag", "forming")
        pole = wide["closes"][22] - wide["closes"][15]
        limit = 0.6 * pole
        span = max(wide["highs"][23:35]) - min(wide["lows"][23:35])
        pad = (limit - span) / 2.0
        while True:
            trial = example_geometry("flag", "forming")
            for index in range(23, 35):
                trial["highs"][index] += pad
                trial["lows"][index] -= pad
            amplitude = max(trial["highs"][23:35]) - min(trial["lows"][23:35])
            if amplitude <= limit:
                break
            pad = math.nextafter(pad, 0.0)
        held = _judge(trial, pause=pack)
        self.assertTrue(held.objects)
        self.assertAlmostEqual(held.objects[0]["fits"][-1]["amplitude"], limit, places=6)
        for index in range(23, 35):
            trial["highs"][index] += 0.002
            trial["lows"][index] -= 0.002
        self.assertFalse(_judge(trial, pause=pack).objects)

    def test_local_high_waits_for_the_right_side_and_ignores_short_pivots(self):
        geo = example_geometry("flag", "up")
        result = _judge(geo)
        highs = [point["index"] for point in result.objects[0]["fits"][-1]["highs"]]
        self.assertNotIn(33, highs)
        self.assertLessEqual(max(highs), 32)
        series = _series(geo)
        confirmed = [
            pivot.extreme_index
            for pivot in detect_pivots(series).confirmed
            if pivot.scale == "short" and 23 <= pivot.extreme_index <= 34
        ]
        self.assertEqual(confirmed, [])
        self.assertTrue(result.objects)

    def test_anchor_uses_the_bars_before_the_pole(self):
        base = _judge(example_geometry("flag"))
        wider = example_geometry("flag")
        for index in (25, 31):
            wider["highs"][index] += 0.001
        other = _judge(wider)
        self.assertEqual(base.objects[0]["anchor"], other.objects[0]["anchor"])
        self.assertAlmostEqual(base.objects[0]["anchor"], 0.01)

    def test_down_break_and_a_crossed_extension_are_not_up(self):
        down = _judge(example_geometry("flag", "down"))
        obj = _one(self, down, "pause:15:22:flag")
        self.assertEqual(obj["status"], "broken")
        self.assertTrue(_reasons(down, obj["id"])[-1].startswith("向下"))
        self.assertEqual(obj["exit"][-1]["direction"], "down")

        geo = _crossing_pennant()
        pack = _pack(
            pole_min=8,
            pole_max=8,
            consolidation_min=12,
            consolidation_max=12,
            pennant_width_min=0.05,
            pennant_width_max=0.2,
        )
        result = _judge(geo, pause=pack)
        self.assertFalse(any(event["reason"].startswith("向上") for event in result.events))
        self.assertFalse(any(item["exit"] and item["exit"][-1]["direction"] == "up" for item in result.objects))
        self.assertEqual(
            _exit_direction(
                [0] * 34 + [1.05, 1.2],
                [1000.0] * 36,
                23,
                34,
                35,
                1.06,
                -0.0099,
                1.046,
                0.003,
                0.002,
                pack,
            ),
            "void",
        )

    def test_a_close_already_outside_does_not_count_as_a_new_breakout(self):
        geo = example_geometry("flag", "up")
        geo["closes"][34] = 1.0545
        geo["highs"][34] = 1.0548
        geo["lows"][34] = 1.05
        result = _judge(geo)
        obj = _one(self, result, "pause:15:22:flag")
        self.assertEqual(obj["status"], "validated")
        self.assertEqual(_reasons(result, obj["id"]), ["旗形"])
        self.assertEqual(obj["exit"], [])

    def test_forming_expires_when_the_consolidation_runs_long_and_on_a_gap(self):
        pack = _pack(pole_min=8, pole_max=8, consolidation_min=12, consolidation_max=12)
        geo = example_geometry("flag", "forming")
        for _ in range(4):
            geo["closes"].append(1.046)
            geo["highs"].append(1.048)
            geo["lows"].append(1.044)
            geo["volumes"].append(1000.0)
        result = _judge(geo, pause=pack)
        obj = _one(self, result, "pause:15:22:flag")
        self.assertEqual(obj["status"], "expired")
        self.assertIn("整理超过最长根数", _reasons(result, obj["id"]))

        gapped = example_geometry("flag", "forming")
        gapped["closes"].append(1.046)
        gapped["highs"].append(1.048)
        gapped["lows"].append(1.044)
        gapped["volumes"].append(1000.0)
        series = _series(gapped, extra_days={36: 12})
        self.assertIn(36, series.gap_indices)
        result = _judge_series(series)
        obj = _one(self, result, "pause:15:22:flag")
        self.assertEqual(obj["status"], "expired")
        self.assertIn("间断", _reasons(result, obj["id"]))
        self.assertTrue(obj["end_index"] < 36)
        self.assertFalse(any(item["start_index"] < 36 <= item["end_index"] for item in result.objects))

    def test_pullback_is_recorded_without_reopening_the_break(self):
        geo = example_geometry("flag", "up")
        geo["closes"].append(1.04)
        geo["highs"].append(1.045)
        geo["lows"].append(1.035)
        geo["volumes"].append(1000.0)
        result = _judge(geo)
        obj = result.objects[0]
        self.assertEqual(obj["status"], "broken")
        self.assertEqual([row["index"] for row in obj["pullbacks"]], [36])
        quiet = example_geometry("flag", "up")
        quiet["closes"].append(1.08)
        quiet["highs"].append(1.085)
        quiet["lows"].append(1.075)
        quiet["volumes"].append(1000.0)
        stayed = _judge(quiet).objects[0]
        self.assertEqual(stayed["status"], "broken")
        self.assertEqual(stayed["pullbacks"], [])

    def test_smaller_fit_error_is_chosen_before_efficiency(self):
        loose = _candidate(0.2, 0.9, 10, 8)
        tight = _candidate(0.05, 0.7, 12, 6)
        picked = _choose([loose, tight], 0.5)
        self.assertEqual([item.a for item in picked], [12])

    def test_overlap_keeps_the_straighter_pole_and_ignores_later_closes(self):
        pack = _pack(pole_min=7, pole_max=8, consolidation_min=12, consolidation_max=12)
        geo = example_geometry("flag")
        geo["closes"][16] = 0.99
        for step, index in enumerate(range(16, 23)):
            geo["closes"][index] = 0.99 + 0.06 * step / 6.0
            geo["highs"][index] = geo["closes"][index] + 0.002
            geo["lows"][index] = geo["closes"][index] - 0.002
        result = _judge(geo, pause=pack)
        ids = [item["id"] for item in result.objects]
        self.assertIn("pause:16:22:flag", ids)
        self.assertNotIn("pause:15:22:flag", ids)
        extended = example_geometry("flag")
        extended["closes"][16] = 0.99
        for step, index in enumerate(range(16, 23)):
            extended["closes"][index] = 0.99 + 0.06 * step / 6.0
            extended["highs"][index] = extended["closes"][index] + 0.002
            extended["lows"][index] = extended["closes"][index] - 0.002
        extended["closes"].append(0.5)
        extended["highs"].append(0.55)
        extended["lows"].append(0.45)
        extended["volumes"].append(1000.0)
        later = _judge(extended, pause=pack)
        born = [item["id"] for item in later.objects if item["available_index"] == 35]
        self.assertEqual(born, ["pause:16:22:flag"])

    def test_the_same_pole_can_change_from_flag_to_pennant(self):
        geo = _flag_then_pennant()
        pack = _pack(pole_min=8, pole_max=8, min_width=0.2, pennant_width_min=0.1, pennant_width_max=0.7)
        early = _judge(_prefix_geometry(geo, 37), pause=pack)
        full = _judge(geo, pause=pack)
        flag = _one(self, full, "pause:15:22:flag")
        pennant = _one(self, full, "pause:15:22:pennant")
        self.assertEqual(flag["status"], "expired")
        self.assertIn("改成了三角旗", _reasons(full, flag["id"]))
        self.assertEqual(pennant["category"], PENNANT)
        self.assertNotEqual(flag["id"], pennant["id"])
        self.assertEqual(
            [(fit["consolidation_end"], fit["upper_slope"]) for fit in flag["fits"]],
            [(fit["consolidation_end"], fit["upper_slope"]) for fit in early.objects[0]["fits"]],
        )

    def test_volume_filters_are_off_until_asked(self):
        loud = example_geometry("flag")
        for index in range(23, 35):
            loud["volumes"][index] = 5000
        self.assertTrue(_judge(loud).objects)
        pack = _pack(volume_filter=True, volume_ratio=0.8)
        quiet = example_geometry("flag")
        for index in range(23, 35):
            quiet["volumes"][index] = 800
        held = _judge(quiet, pause=pack)
        self.assertAlmostEqual(held.objects[0]["fits"][-1]["volume_ratio"], 0.8)
        for index in range(23, 35):
            quiet["volumes"][index] = 801
        self.assertFalse(_judge(quiet, pause=pack).objects)

        pack = _pack(breakout_volume_filter=True, breakout_volume=1.3)
        thin = example_geometry("flag", "up")
        thin["volumes"][35] = 1000
        self.assertFalse(_judge(thin, pause=pack).objects)
        thin["volumes"][35] = 1300
        broken = _judge(thin, pause=pack)
        self.assertTrue(_reasons(broken, broken.objects[0]["id"])[-1].startswith("向上"))

    def test_log_and_uniform_share_the_geometric_boundary(self):
        geo = example_geometry("flag")
        left = _judge(geo, axis="log").objects[0]
        right = _judge(geo, axis="uniform").objects[0]
        self.assertEqual(left["id"], right["id"])
        self.assertEqual(left["category"], right["category"])
        self.assertEqual(left["start_index"], right["start_index"])
        self.assertEqual(left["fits"][-1]["upper_slope"], right["fits"][-1]["upper_slope"])
        self.assertEqual(left["fits"][-1]["lower_intercept"], right["fits"][-1]["lower_intercept"])

    def test_missing_or_invalid_quotes_skip_without_stopping_direction(self):
        geo = example_geometry("flag")
        times = _times(len(geo["closes"]))
        params = default_params("log", version=PARAM_VERSION_V3)
        bare = build_series(times, [math.exp(value) for value in geo["closes"]], interval="1d", params=params)
        snapshot = build_snapshot(bare, symbol="PAUSE", engine_version="test")
        self.assertEqual(snapshot["methods"]["pause"]["status"], "skipped")
        self.assertIn("缺少最高价、最低价或成交量", snapshot["methods"]["pause"]["reason"])
        self.assertEqual(snapshot["methods"]["direction"]["status"], "completed")

        broken = example_geometry("flag")
        broken["highs"][4] = broken["lows"][4] - 0.01
        series = _series(broken)
        snapshot = build_snapshot(series, symbol="PAUSE", engine_version="test")
        self.assertEqual(snapshot["methods"]["pause"]["status"], "skipped")
        self.assertIn("最高价", snapshot["methods"]["pause"]["reason"])

    def test_older_versions_keep_the_close_only_hash(self):
        geo = example_geometry("flag")
        times = _times(len(geo["closes"]))
        closes = [math.exp(value) for value in geo["closes"]]
        highs = [math.exp(value) for value in geo["highs"]]
        lows = [math.exp(value) for value in geo["lows"]]
        for version in (PARAM_VERSION, PARAM_VERSION_V1, PARAM_VERSION_V2):
            params = default_params("log", version=version)
            plain = build_series(times, closes, interval="1d", params=params)
            quoted = build_series(
                times,
                closes,
                interval="1d",
                params=params,
                highs=highs,
                lows=lows,
                volumes=geo["volumes"],
            )
            self.assertEqual(plain.data_hash, quoted.data_hash)
            self.assertIsNone(plain.bars[0].high)
            snapshot = build_snapshot(plain, symbol="PAUSE", engine_version="test")
            self.assertNotIn("pause", snapshot.get("methods") or {})
            self.assertNotIn("pause", snapshot["params"])
        params = default_params("log", version=PARAM_VERSION_V3)
        plain = build_series(times, closes, interval="1d", params=params)
        quoted = build_series(
            times,
            closes,
            interval="1d",
            params=params,
            highs=highs,
            lows=lows,
            volumes=geo["volumes"],
        )
        self.assertNotEqual(plain.data_hash, quoted.data_hash)
        head = quoted.prefix(10)
        again = build_series(
            times[:10],
            closes[:10],
            interval="1d",
            params=params,
            highs=highs[:10],
            lows=lows[:10],
            volumes=geo["volumes"][:10],
        )
        self.assertEqual(head.data_hash, again.data_hash)
        self.assertEqual(head.bars, again.bars)


class PauseContractTests(unittest.TestCase):
    def test_prefix_matches_the_as_of_view(self):
        geo = example_geometry("flag", "up")
        for axis in ("log", "uniform"):
            snapshot, series = _snapshot(geo, axis)
            for count in (20, 34, 35, 36):
                with self.subTest(axis=axis, count=count):
                    prefix_snapshot, prefix = _snapshot(_prefix_geometry(geo, count), axis)
                    shown = _as_of(snapshot, prefix.bars[-1].timestamp)
                    self._assert_prefix(prefix_snapshot, shown, count)

    def test_pause_does_not_recompute_pivots_or_boundaries(self):
        series = _series(example_geometry("flag"))
        context = build_context(
            series,
            volatility=tuple(None for _ in series.bars),
            pivots=(),
            segments=(),
            boundaries=(),
            zones=(),
            events=(),
        )
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
            result = PauseMethod().judge(context, series.params.pause.to_dict(), {})
        finally:
            for (name, attr), original in originals.items():
                setattr(__import__(name, fromlist=[attr]), attr, original)
        self.assertTrue(result.objects)

    def test_objects_do_not_cross_gaps(self):
        geo = example_geometry("flag", "forming")
        geo["closes"].append(1.046)
        geo["highs"].append(1.048)
        geo["lows"].append(1.044)
        geo["volumes"].append(1000.0)
        snapshot, series = _snapshot(geo, extra_days={36: 12})
        gaps = set(series.gap_indices)
        self.assertTrue(gaps)
        for obj in snapshot["methods"]["pause"]["objects"]:
            self.assertFalse(any(obj["start_index"] < gap <= obj["end_index"] for gap in gaps))
            for event in snapshot["events"]:
                if event.get("object_id") == obj["id"]:
                    self.assertGreaterEqual(event["available_index"], obj["available_index"])

    def test_both_axes_render_and_older_versions_omit_pause(self):
        for axis in ("log", "uniform"):
            snapshot, _series = _snapshot(example_geometry("flag"), axis)
            html = render_report(snapshot)
            self.assertIn("停顿", html)
            self.assertIn("method-pause", html)
            model = build_view_model(snapshot)
            bands = [mark for mark in model["marks"] if mark["method"] == "pause" and mark["kind"] == "band"]
            segments = [mark for mark in model["marks"] if mark["method"] == "pause" and mark["kind"] == "segment"]
            self.assertTrue(bands)
            self.assertGreaterEqual(len(segments), 2)
            self.assertTrue(all(mark["kind"] in MARK_KINDS for mark in model["marks"]))
            declared = {item.key for item in PauseMethod.categories}
            for item in snapshot["methods"]["pause"]["objects"]:
                self.assertIn(item["category"], declared)
        for version in (PARAM_VERSION, PARAM_VERSION_V1, PARAM_VERSION_V2):
            params = default_params("log", version=version)
            geo = example_geometry("flag")
            series = build_series(_times(len(geo["closes"])), [math.exp(v) for v in geo["closes"]], interval="1d", params=params)
            snapshot = build_snapshot(series, symbol="PAUSE", engine_version="test")
            self.assertNotIn("pause", snapshot.get("methods") or {})
            html = render_report(snapshot)
            self.assertNotIn('id="method-pause"', html)

    def test_hand_built_snapshot_keeps_every_table_row(self):
        snapshot = _hand_snapshot()
        model = build_view_model(snapshot)
        self.assertEqual(len(model["method_objects"]["pause"]), 3)
        bands = [mark for mark in model["marks"] if mark["method"] == "pause" and mark["kind"] == "band"]
        self.assertEqual(len(bands), 2)
        self.assertEqual({mark["object_id"] for mark in bands}, {"pause:4:9:flag", "pause:4:9:pennant"})
        html = render_report(snapshot)
        table = html.split('id="method-pause-pauses"', 1)[1].split("</table>", 1)[0]
        self.assertEqual(table.count("data-object-id"), 3)

    def test_samples_render_a_flag_and_a_pennant(self):
        method = PauseMethod()
        samples = method.samples()
        self.assertGreaterEqual(len(samples), 1)
        for sample in samples:
            series = build_series(
                _times(len(sample.closes)),
                sample.closes,
                interval="1d",
                params=default_params(sample.price_axis, version=PARAM_VERSION_V3),
                highs=sample.highs,
                lows=sample.lows,
                volumes=sample.volumes,
            )
            snapshot = build_snapshot(series, symbol=sample.symbol, engine_version="test")
            categories = {item["category"] for item in snapshot["methods"]["pause"]["objects"]}
            self.assertTrue(categories)
            html = render_report(snapshot)
            self.assertIn("停顿", html)

    def test_illegal_params_raise(self):
        with self.assertRaises(ValueError):
            PauseParams(pole_min=1)
        with self.assertRaises(ValueError):
            PauseParams(pennant_width_max=0.9)
        with self.assertRaises(ValueError):
            PauseParams(consolidation_min=3)
        PauseMethod().check_params(PauseParams().to_dict())

    def _assert_prefix(self, prefix, shown, count):
        left = prefix["methods"]["pause"]["objects"]
        right = shown["methods"]["pause"]["objects"]
        self.assertEqual([item["id"] for item in left], [item["id"] for item in right])
        stop = count - 1
        for item, other in zip(left, right):
            self.assertEqual(_frozen(item), _frozen(other))
            self.assertEqual(item["status"], other["status"])
            self.assertEqual(_rows_until(item.get("fits"), stop), _rows_until(other.get("fits"), stop))
            self.assertEqual(_rows_until(item.get("exit"), stop), _rows_until(other.get("exit"), stop))
            self.assertEqual(_rows_until(item.get("pullbacks"), stop), _rows_until(other.get("pullbacks"), stop))
        self.assertEqual(
            _rows_until(prefix["methods"]["pause"]["extra"]["others"], stop),
            _rows_until(shown["methods"]["pause"]["extra"]["others"], stop),
        )
        self.assertEqual(
            [event for event in prefix["events"] if event.get("object_kind") == "pause"],
            [event for event in shown["events"] if event.get("object_kind") == "pause"],
        )


def _candidate(error: float, efficiency: float, anchor: int, pole: int) -> _Candidate:
    return _Candidate(
        a=anchor,
        e=anchor + pole - 1,
        s=anchor + pole,
        c=40,
        t=40,
        p=pole,
        anchor=0.01,
        pole_rise=0.05,
        efficiency=efficiency,
        category=FLAG,
        exit_direction=None,
        mean_error=error,
        fit={},
    )


def _pack(**kwargs) -> PauseParams:
    base = default_params(version=PARAM_VERSION_V3).pause
    return replace(base, **kwargs)


def _judge(geo, axis="log", pause=None, extra_days=None):
    return _judge_series(_series(geo, axis, pause, extra_days))


def _judge_series(series):
    context = build_context(
        series,
        volatility=tuple(None for _ in series.bars),
        pivots=(),
        segments=(),
        boundaries=(),
        zones=(),
        events=(),
    )
    return PauseMethod().judge(context, series.params.pause.to_dict(), {})


def _snapshot(geo, axis="log", extra_days=None):
    series = _series(geo, axis, extra_days=extra_days)
    return build_snapshot(series, symbol="PAUSE", engine_version="test"), series


def _series(geo, axis="log", pause=None, extra_days=None):
    def convert(values):
        if axis == "log":
            return [math.exp(value) for value in values]
        return list(values)

    params = default_params(axis, version=PARAM_VERSION_V3)
    if pause is not None:
        params = replace(params, pause=pause)
    return build_series(
        _times(len(geo["closes"]), extra_days),
        convert(geo["closes"]),
        interval="1d",
        params=params,
        highs=convert(geo["highs"]),
        lows=convert(geo["lows"]),
        volumes=list(geo["volumes"]),
    )


def _times(count, extra_days=None):
    extra_days = extra_days or {}
    cursor = datetime(2024, 1, 2, 16, 0)
    times = []
    for index in range(count):
        if index:
            cursor = cursor + timedelta(days=1 + extra_days.get(index, 0))
        times.append(cursor)
    return times


def _prefix_geometry(geo, count):
    return {
        "highs": list(geo["highs"][:count]),
        "lows": list(geo["lows"][:count]),
        "closes": list(geo["closes"][:count]),
        "volumes": list(geo["volumes"][:count]),
    }


def _one(test, result, object_id):
    matches = [item for item in result.objects if item["id"] == object_id]
    test.assertEqual(len(matches), 1, [item["id"] for item in result.objects])
    return matches[0]


def _reasons(result, object_id):
    return [event["reason"] for event in result.events if event["object_id"] == object_id]


def _frozen(obj):
    hidden = {
        "status",
        "primary",
        "alternate_of",
        "end_index",
        "end_time",
        "stats",
        "fits",
        "exit",
        "pullbacks",
    }
    return {key: value for key, value in obj.items() if key not in hidden}


def _rows_until(rows, stop):
    return [row for row in rows or [] if int(row.get("available_index") or 0) <= stop]


def _as_of(snapshot, when):
    copied = dict(snapshot)
    identity = dict(snapshot["identity"])
    identity["as_of"] = when.isoformat()
    copied["identity"] = identity
    return display_snapshot(copied)


def _refuse(*_args, **_kwargs):
    raise AssertionError("停顿不应重算拐点或边界")


def _crossing_pennant():
    """两条边界在整理内仍分开，延伸到信号根时已经交叉。"""
    count = 36
    closes = [1.02] * count
    highs = [1.021] * count
    lows = [1.019] * count
    volumes = [1000.0] * count
    for step, index in enumerate(range(15, 23)):
        closes[index] = 1.0 + 0.05 * step / 7.0
        highs[index] = closes[index] + 0.001
        lows[index] = closes[index] - 0.001
    upper_intercept, upper_slope = 1.06, -0.0099
    lower_intercept, lower_slope = 1.046, 0.003

    def upper(x_value):
        return upper_intercept + upper_slope * x_value

    def lower(x_value):
        return lower_intercept + lower_slope * x_value

    start, end = 23, 34
    span = end - start
    for index in range(start, end + 1):
        x_value = (index - start) / span
        top, bottom = upper(x_value), lower(x_value)
        width = top - bottom
        highs[index] = top - 0.2 * width
        lows[index] = bottom + 0.2 * width
        closes[index] = (top + bottom) / 2.0
    for index in (25, 31):
        highs[index] = upper((index - start) / span)
    for index in (26, 32):
        lows[index] = lower((index - start) / span)
    for index in (25, 31):
        for neighbor in range(index - 2, index + 3):
            if neighbor == index or neighbor < start or neighbor > end:
                continue
            highs[neighbor] = min(highs[neighbor], highs[index] - 0.0008)
    for index in (26, 32):
        for neighbor in range(index - 2, index + 3):
            if neighbor == index or neighbor < start or neighbor > end:
                continue
            lows[neighbor] = max(lows[neighbor], lows[index] + 0.0008)
    for index in range(start, end + 1):
        closes[index] = min(max(closes[index], lows[index]), highs[index])
    closes[35] = 1.2
    highs[35] = 1.22
    lows[35] = 1.18
    return {"highs": highs, "lows": lows, "closes": closes, "volumes": volumes}


def _flag_then_pennant():
    geo = example_geometry("flag", "forming")
    for _ in range(36, 48):
        geo["closes"].append(1.0478)
        geo["highs"].append(1.0485)
        geo["lows"].append(1.0472)
        geo["volumes"].append(1000.0)
    geo["highs"][42] = 1.0505
    geo["lows"][42] = 1.0468
    geo["closes"][42] = 1.0485
    geo["lows"][45] = 1.0462
    geo["highs"][45] = 1.0488
    geo["closes"][45] = 1.0474
    return geo


def _hand_snapshot():
    params = default_params("log", version=PARAM_VERSION_V3)
    closes = [
        {"index": index, "time": f"2024-01-{2 + index:02d}T16:00:00", "close": 100.0 + index, "gap_before": False}
        for index in range(24)
    ]
    objects = [
        _hand_object("pause:4:9:flag", "flag", True, "validated", None),
        _hand_object("pause:1:6:flag", "flag", False, "validated", "pause:4:9:flag"),
        _hand_object("pause:4:9:pennant", "pennant", False, "broken", "pause:1:6:flag"),
    ]
    record = {
        "key": "pause",
        "name": "停顿",
        "version": "pause-v1",
        "kind": "instance",
        "section": "2.3",
        "requires": ["closes", "highs", "lows", "volumes"],
        "categories": [{"key": item.key, "name": item.name, "color": item.color} for item in PauseMethod.categories],
        "replaces": [],
        "params": params.pause.to_dict(),
        "status": "completed",
        "reason": "",
        "objects": objects,
        "extra": {"fits": [], "others": []},
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
            "visible_scales": ["mid"],
            "visible_methods": ["pause"],
            "bar_count": 24,
        },
        "reading": "",
        "scale_definition_note": "",
        "quality": {"bar_count": 24, "gap_count": 0, "gaps": [], "grade": "正常", "grade_note": ""},
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
        "methods": {"pause": record},
    }


def _hand_object(object_id, category, primary, status, alternate_of):
    upper = math.log(110.0)
    lower = math.log(100.0)
    fit = {
        "available_index": 12,
        "available_time": "2024-01-14T16:00:00",
        "consolidation_end": 11,
        "upper_intercept": upper,
        "upper_slope": -0.01,
        "lower_intercept": lower,
        "lower_slope": -0.01,
        "width_start": upper - lower,
        "width_end": upper - lower,
        "width_ratio": 1.0,
        "midline": -0.01,
        "upper_error": 0.0,
        "lower_error": 0.0,
        "coverage": 1.0,
        "drawdown": 0.2,
        "amplitude": 0.05,
        "volume_ratio": 0.7,
        "mean_error": 0.0,
        "highs": [],
        "lows": [],
    }
    return {
        "id": object_id,
        "method": "pause",
        "scale": "multi",
        "category": category,
        "status": status,
        "start_index": 4,
        "end_index": 11,
        "start_time": "2024-01-06T16:00:00",
        "end_time": "2024-01-13T16:00:00",
        "available_index": 12,
        "available_time": "2024-01-14T16:00:00",
        "revision": 1,
        "primary": primary,
        "alternate_of": alternate_of,
        "evidence": None,
        "constraints": None,
        "score": None,
        "stats": None,
        "pole_end": 9,
        "consolidation_start": 10,
        "anchor": 0.01,
        "pole_rise": 0.05,
        "efficiency": 0.9,
        "rise_multiple": 5.0,
        "fits": [fit],
        "exit": [{"available_index": 12, "direction": "up", "index": 12}] if status == "broken" else [],
        "pullbacks": [],
    }


if __name__ == "__main__":
    unittest.main()
