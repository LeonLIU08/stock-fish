"""趋势策略第一阶段：配置校验，以及合成 K 线上的空会话报告。"""
from __future__ import annotations

import copy
import math
import subprocess
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from analysis.backtest.trend.config import load_trend_config, load_trend_yaml, trend_config_from_mapping
from analysis.backtest.trend.prices import CorporateAction, adjust_prices
from analysis.backtest.trend.runner import run_trend_backtest
from analysis.backtest.trend.session import run_empty_session
from analysis.backtest.trend.tape import (
    build_tape,
    canonical_row,
    geometric_emas,
    true_range_units,
)
from analysis.structure.config import PARAM_VERSION_V6, default_params
from analysis.structure.methods.phase import _Run
from analysis.structure.series import build_series
from analysis.structure.snapshot import build_snapshot
from analysis.structure.view_model import display_snapshot

HK = ZoneInfo("Asia/Hong_Kong")
ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/Users/leon/PyVenv/q_stock/bin/python"


def _raw():
    return copy.deepcopy(load_trend_yaml())


def _bars(count=80):
    cursor = datetime(2024, 1, 2, 16, 0, tzinfo=HK)
    rows = []
    while len(rows) < count:
        if cursor.weekday() < 5:
            close = 100.0 * (1.001 ** len(rows))
            open_ = close / 1.001
            rows.append({
                "datetime": pd.Timestamp(cursor),
                "open": open_,
                "high": close * 1.01,
                "low": open_ * 0.99,
                "close": close,
                "volume": 100000.0,
                "amount": close * 100000.0,
            })
        cursor += timedelta(days=1)
    return pd.DataFrame(rows)


class ConfigTests(unittest.TestCase):
    def test_default_file_loads_four_runs(self):
        config = load_trend_config(today=date(2026, 10, 13))
        self.assertEqual(config.version, "trend-strategy-v0")
        self.assertEqual(config.capital_text(), "港股 1000000 HKD；美股 100000 USD")
        self.assertEqual([item.code for item in config.symbols], ["00700", "09988", "NVDA"])
        self.assertEqual(config.symbols[0].lot_size, 100)
        self.assertEqual(config.symbols[0].lot_size_source, "config")
        self.assertEqual(config.symbols[2].lot_size, 1)
        self.assertEqual(config.symbols[2].lot_size_source, "market_default")
        self.assertEqual(config.symbols[2].dividend_withholding, 0.3)
        self.assertEqual(config.symbols[0].dividend_withholding, 0.0)
        self.assertEqual(config.structure.scale, "mid")
        self.assertEqual(config.structure.param_version, "structure-params-v6")
        self.assertEqual([run.key for run in config.runs], [
            "breakout",
            "breakout_no_reduce",
            "pullback",
            "pullback_no_reduce",
        ])
        by_key = {run.key: run for run in config.runs}
        self.assertEqual(by_key["breakout"].exhaustion_action, "reduce")
        self.assertEqual(by_key["breakout_no_reduce"].exhaustion_action, "none")
        self.assertEqual(config.strategies["breakout"].exit.exhaustion_action, "reduce")
        self.assertIsNone(config.strategies["breakout"].risk.point_value)

    def test_symbol_codes_are_canonical(self):
        data = _raw()
        data["universe"]["symbols"] = [
            {"code": "700.HK", "name": "腾讯控股", "lot_size": 100},
            {"code": "9988.HK", "name": "阿里巴巴", "lot_size": 100},
            {"code": "nvda.us", "name": "英伟达"},
        ]
        config = trend_config_from_mapping(data, today=date(2026, 10, 13))
        self.assertEqual([item.code for item in config.symbols], ["00700", "09988", "NVDA"])
        self.assertEqual([item.market for item in config.symbols], ["hk", "hk", "us"])

    def test_unknown_key_fails(self):
        data = _raw()
        data["portfolio"] = {}
        with self.assertRaisesRegex(ValueError, "未登记的字段: portfolio"):
            trend_config_from_mapping(data)

    def test_adds_fails(self):
        data = _raw()
        data["strategies"]["breakout"]["adds"] = {"enabled": False}
        with self.assertRaisesRegex(ValueError, "adds"):
            trend_config_from_mapping(data)

    def test_missing_phase_fails(self):
        data = _raw()
        data["structure"]["param_version"] = "structure-params-v5"
        with self.assertRaisesRegex(ValueError, "没有启用 phase"):
            trend_config_from_mapping(data)

    def test_point_value_fails(self):
        data = _raw()
        data["strategies"]["pullback"]["risk"]["point_value"] = 1
        with self.assertRaisesRegex(ValueError, "point_value"):
            trend_config_from_mapping(data)

    def test_allow_short_fails(self):
        data = _raw()
        data["universe"]["allow_short"] = True
        with self.assertRaisesRegex(ValueError, "allow_short"):
            trend_config_from_mapping(data)

    def test_a_share_code_fails(self):
        data = _raw()
        data["universe"]["symbols"] = [{"code": "600519", "name": "贵州茅台", "lot_size": 100}]
        with self.assertRaisesRegex(ValueError, "A 股"):
            trend_config_from_mapping(data)

    def test_interval_must_be_daily(self):
        data = _raw()
        data["window"]["interval"] = "5m"
        with self.assertRaisesRegex(ValueError, "1d"):
            trend_config_from_mapping(data)

    def test_hk_without_lot_size_fails(self):
        data = _raw()
        data["universe"]["symbols"] = [{"code": "00700", "name": "腾讯控股"}]
        with self.assertRaisesRegex(ValueError, "lot_size"):
            trend_config_from_mapping(data)

    def test_one_lot_max_risk_cannot_be_below_trade_risk(self):
        data = _raw()
        data["strategies"]["breakout"]["risk"]["one_lot_max_risk"] = 0.001
        with self.assertRaisesRegex(ValueError, "one_lot_max_risk"):
            trend_config_from_mapping(data)
        data = _raw()
        data["strategies"]["breakout"]["risk"]["one_lot_max_risk"] = 0
        config = trend_config_from_mapping(data, today=date(2026, 10, 13))
        self.assertEqual(config.strategies["breakout"].risk.one_lot_max_risk, 0)

    def test_run_cannot_override_other_keys(self):
        data = _raw()
        data["runs"][0]["risk_per_trade"] = 0.02
        with self.assertRaisesRegex(ValueError, "exhaustion_action"):
            trend_config_from_mapping(data)

    def test_tighten_is_rejected(self):
        data = _raw()
        data["strategies"]["breakout"]["exit"]["exhaustion_action"] = "tighten"
        with self.assertRaisesRegex(ValueError, "reduce 或 none"):
            trend_config_from_mapping(data)

    def test_overrides_select_symbol_and_run(self):
        config = load_trend_config(
            overrides={"symbols": ["AAPL"], "runs": ["pullback"], "lookback_years": 2},
            today=date(2026, 10, 13),
        )
        self.assertEqual(config.symbols[0].code, "AAPL")
        self.assertEqual(config.symbols[0].lot_size, 1)
        self.assertEqual(config.symbols[0].dividend_withholding, 0.3)
        self.assertEqual([run.key for run in config.runs], ["pullback"])
        self.assertEqual(config.window.end, date(2026, 10, 13))
        self.assertEqual(
            config.window.start,
            date(2026, 10, 13) - timedelta(days=int(round(2 * 365))),
        )

    def test_unknown_hk_symbol_fails_without_lot_size(self):
        with self.assertRaisesRegex(ValueError, "lot_size"):
            load_trend_config(overrides={"symbols": ["00005"]})

    def test_cli_rejects_a_share_before_fetching(self):
        result = subprocess.run(
            [PYTHON, "scripts/run_trend_backtest.py", "--symbols", "600519"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("A 股", result.stderr)


class EmptySessionTests(unittest.TestCase):
    def test_accounts_keep_their_own_cash(self):
        bars = _bars(5)
        hk = run_empty_session(bars, 1_000_000)
        us = run_empty_session(bars, 100_000)
        self.assertEqual(hk.final_cash, 1_000_000)
        self.assertEqual(hk.final_shares, 0)
        self.assertEqual(len(hk.trades), 0)
        self.assertEqual(us.final_cash, 100_000)
        self.assertEqual(us.final_shares, 0)

    def test_empty_run_writes_existing_report(self):
        bars = _bars(80)
        data = _raw()
        data["universe"]["symbols"] = [{"code": "00700", "name": "腾讯控股", "lot_size": 100}]
        data["window"]["start"] = bars["datetime"].iloc[40].date().isoformat()
        data["window"]["end"] = bars["datetime"].iloc[-1].date().isoformat()
        data["window"]["warmup_bars"] = 30
        data["runs"] = [data["runs"][0], data["runs"][2]]
        with tempfile.TemporaryDirectory() as tmp:
            data["output_dir"] = tmp
            config = trend_config_from_mapping(data, source="test")
            payload = run_trend_backtest(config, frames={"00700": bars}, load_benchmarks=False)
            out = Path(payload["output_dir"])
            self.assertTrue(out.name.endswith("_1d_trend_state"))
            self.assertTrue((out / "summary.md").is_file())
            self.assertTrue((out / "summary.json").is_file())
            self.assertTrue((out / "index.html").is_file())
            self.assertEqual(len(payload["results"]), 2)
            folders = sorted(path.name for path in out.iterdir() if path.is_dir())
            self.assertEqual(folders, ["00700_breakout", "00700_pullback"])
            for row in payload["results"]:
                self.assertEqual(row["metrics"]["n_trades"], 0)
                self.assertEqual(row["metrics"]["final_equity"], 1_000_000)
                self.assertEqual(row["metrics"]["final_cash"], 1_000_000)
                self.assertEqual(row["structure"]["param_version"], "structure-params-v6")
                self.assertEqual(row["structure"]["bar_count"], 80)
                self.assertIn("phase", row["structure"]["methods"])
                self.assertTrue(row["structure"]["data_hash"])
                self.assertEqual(row["structure"]["tape_bars"], 40)
                report = Path(row["report_html"])
                text = report.read_text(encoding="utf-8")
                self.assertIn("关键指标", text)
                self.assertIn("无成交", text)
                self.assertIn("空会话", text)
                decisions = report.parent / "decisions.jsonl"
                self.assertEqual(decisions.read_text(encoding="utf-8"), "")
            hashes = {row["structure"]["tape_hash"] for row in payload["results"]}
            self.assertEqual(len(hashes), 1)
            summary = (out / "summary.md").read_text(encoding="utf-8")
            self.assertIn("策略 A", summary)
            self.assertIn("策略 B", summary)


def _ohlc(closes):
    opens = [closes[0]] + list(closes[:-1])
    highs = [max(open_, close) * 1.01 for open_, close in zip(opens, closes)]
    lows = [min(open_, close) * 0.99 for open_, close in zip(opens, closes)]
    return opens, highs, lows


def _times(count, gap_at=None):
    cursor = datetime(2024, 1, 2, 16, 0, tzinfo=HK)
    times = []
    for index in range(count):
        if gap_at is not None and index == gap_at:
            cursor += timedelta(days=14)
        while cursor.weekday() >= 5:
            cursor += timedelta(days=1)
        times.append(cursor)
        cursor += timedelta(days=1)
    return times


def _make_series(closes, *, axis="log", gap_at=None, opens=None, highs=None, lows=None, volumes=None):
    if opens is None:
        opens, highs, lows = _ohlc(closes)
    params = default_params(axis, version=PARAM_VERSION_V6)
    return build_series(
        _times(len(closes), gap_at=gap_at),
        closes,
        interval="1d",
        params=params,
        opens=opens,
        highs=highs,
        lows=lows,
        volumes=volumes or [1000.0] * len(closes),
    )


def _snapshot(series):
    return build_snapshot(
        series,
        symbol="TAPE",
        requested_start=series.bars[0].timestamp.date(),
        requested_end=series.bars[-1].timestamp.date(),
        visible_scales=["mid"],
        data_source="synthetic",
        engine_version="test",
    )


def _wave(count=72):
    level = 0.0
    closes = []
    for index in range(count):
        level += 0.012 if (index // 15) % 2 == 0 else -0.009
        closes.append(100.0 * math.exp(level))
    return closes


class TapeTests(unittest.TestCase):
    def test_adjustment_keeps_past_prices_and_log_returns(self):
        closes = [100.0, 110.0, 105.0, 120.0]
        opens, highs, lows = _ohlc(closes)
        volumes = [1000.0, 1000.0, 1000.0, 1000.0]
        plain = adjust_prices(opens, highs, lows, closes, volumes)
        self.assertEqual(plain.factor, (1.0, 1.0, 1.0, 1.0))
        later = adjust_prices(
            opens,
            highs,
            lows,
            closes,
            volumes,
            [
                CorporateAction(2, "dividend", 5.0),
                CorporateAction(3, "split", 2.0),
            ],
        )
        self.assertEqual(later.factor[0], 1.0)
        self.assertEqual(later.factor[1], 1.0)
        self.assertEqual(later.close[:2], plain.close[:2])
        self.assertAlmostEqual(later.factor[2], 110.0 / 105.0)
        self.assertAlmostEqual(later.factor[3], (110.0 / 105.0) * 2.0)
        self.assertAlmostEqual(later.volume[3], 500.0)
        self.assertAlmostEqual(math.log(later.close[1] / later.close[0]), math.log(110.0 / 100.0))

    def test_true_range_uses_log_or_price_and_ignores_gap_jump(self):
        closes = [100.0, 100.0, 50.0]
        opens = [100.0, 100.0, 45.0]
        highs = [110.0, 100.0, 50.0]
        lows = [90.0, 100.0, 40.0]
        log_series = _make_series(closes, opens=opens, highs=highs, lows=lows, gap_at=2)
        units = true_range_units(log_series, span=3, floor=0.0)
        first = math.log(110.0 / 90.0)
        second = math.log(50.0 / 40.0)
        self.assertIsNone(units[0])
        self.assertIsNone(units[1])
        self.assertTrue(log_series.bars[2].gap_before)
        self.assertAlmostEqual(units[2], (first + 0.0 + second) / 3.0)
        jumped = math.log(100.0 / 40.0)
        self.assertNotAlmostEqual(units[2], (first + 0.0 + jumped) / 3.0)

        uniform = _make_series(closes, axis="uniform", opens=opens, highs=highs, lows=lows)
        width = true_range_units(uniform, span=1, floor=0.01)
        self.assertAlmostEqual(width[0], max(20.0, 0.01 * 100.0))
        self.assertAlmostEqual(width[1], max(0.0, 0.01 * 100.0))

        floored = _make_series(
            [1.0, 1.0, 1.0],
            opens=[1.0, 1.0, 1.0],
            highs=[math.exp(0.1), math.exp(0.1), math.exp(0.8)],
            lows=[1.0, 1.0, 1.0],
        )
        held = true_range_units(floored, span=2, floor=0.5)
        self.assertIsNone(held[0])
        self.assertAlmostEqual(held[1], 0.5)
        self.assertAlmostEqual(held[2], 0.5 + (0.8 - 0.5) / 2.0)

    def test_ema_matches_phase_fast_average_and_resets_on_gap(self):
        series = _make_series(_wave(30), gap_at=16)
        state = _Run()
        expected = []
        for bar in series.bars:
            if bar.index == 0 or bar.gap_before:
                state.begin_region()
            expected.append(state.update_ema(bar.geometric, 10))
        self.assertEqual(geometric_emas(series, 10), expected)
        self.assertIsNone(geometric_emas(series, 10)[16])
        self.assertIsNotNone(geometric_emas(series, 10)[15])

    def test_prefix_matches_as_of_fields(self):
        series = _make_series(_wave(72))
        snapshot = _snapshot(series)
        tape = build_tape(series, snapshot, scale="mid", span=14, floor=0.003)
        self.assertEqual(len(tape), len(series))
        self.assertTrue(all(row.scale == "mid" for row in tape))
        for count in (24, 48, 72):
            prefix = series.prefix(count)
            prefix_snapshot = _snapshot(prefix)
            prefix_tape = build_tape(prefix, prefix_snapshot, scale="mid", span=14, floor=0.003)
            self.assertEqual(canonical_row(prefix_tape[-1]), canonical_row(tape[count - 1]))
            covering = [
                item
                for item in prefix_snapshot["methods"]["phase"]["objects"]
                if item["scale"] == "mid" and int(item["start_index"]) <= count - 1 <= int(item["end_index"])
            ]
            self.assertEqual(prefix_tape[-1].category, covering[-1]["category"])
            self.assertEqual(prefix_tape[-1].structure, covering[-1]["structure"])
            self.assertEqual(prefix_tape[-1].phase_start_index, int(covering[-1]["start_index"]))
            self.assertEqual(prefix_tape[-1].phase_start_time, covering[-1]["start_time"])

    def test_future_prices_dividends_and_splits_do_not_change_past_rows(self):
        closes = _wave(70)
        opens, highs, lows = _ohlc(closes)
        volumes = [1000.0] * len(closes)
        original = adjust_prices(opens, highs, lows, closes, volumes)
        series = _make_series(
            list(original.close),
            opens=list(original.open),
            highs=list(original.high),
            lows=list(original.low),
            volumes=list(original.volume),
        )
        tape = build_tape(series, _snapshot(series), scale="mid", span=14, floor=0.003)
        changed_closes = list(closes)
        changed_highs = list(highs)
        changed_lows = list(lows)
        changed_opens = list(opens)
        for index in range(40, len(closes)):
            changed_closes[index] *= 1.5
            changed_highs[index] *= 1.5
            changed_lows[index] *= 1.5
            changed_opens[index] *= 1.5
        changed = adjust_prices(
            changed_opens,
            changed_highs,
            changed_lows,
            changed_closes,
            volumes,
            [
                CorporateAction(50, "dividend", 1.0),
                CorporateAction(60, "split", 3.0),
            ],
        )
        self.assertEqual(changed.close[:40], original.close[:40])
        changed_series = _make_series(
            list(changed.close),
            opens=list(changed.open),
            highs=list(changed.high),
            lows=list(changed.low),
            volumes=list(changed.volume),
        )
        changed_tape = build_tape(
            changed_series,
            _snapshot(changed_series),
            scale="mid",
            span=14,
            floor=0.003,
        )
        self.assertEqual(
            [canonical_row(row) for row in tape if row.index <= 39],
            [canonical_row(row) for row in changed_tape if row.index <= 39],
        )

    def test_truncated_segment_stats_are_not_read(self):
        series = _make_series(_wave(72))
        snapshot = _snapshot(series)
        for record in snapshot["methods"].values():
            for item in record.get("objects") or []:
                if isinstance(item.get("stats"), dict):
                    item["stats"]["fast_average"] = -1.0
                    item["stats"]["s_t"] = 999.0
        tape = build_tape(series, snapshot, scale="mid", span=14, floor=0.003)
        self.assertTrue(any(row.ema not in (None, -1.0) for row in tape))
        self.assertTrue(all(row.s_t != 999.0 for row in tape))
        self.assertNotIn("stats", canonical_row(tape[-1]))

        found = False
        for index in (20, 40, 55):
            copied = dict(snapshot)
            identity = dict(snapshot["identity"])
            identity["as_of"] = snapshot["closes"][index]["time"]
            copied["identity"] = identity
            shown = display_snapshot(copied)
            for item in shown["methods"]["phase"]["objects"]:
                if item.get("scale") != "mid" or int(item["end_index"]) != index:
                    continue
                raw = next(
                    obj
                    for obj in snapshot["methods"]["phase"]["objects"]
                    if obj["id"] == item["id"]
                )
                if int(raw["end_index"]) > index:
                    self.assertNotIn("stats", item)
                    self.assertIn("stats", raw)
                    found = True
        self.assertTrue(found)


if __name__ == "__main__":
    unittest.main()
