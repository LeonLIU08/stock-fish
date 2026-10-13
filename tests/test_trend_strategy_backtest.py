"""趋势策略第一阶段：配置校验，以及合成 K 线上的空会话报告。"""
from __future__ import annotations

import copy
import math
import subprocess
import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from analysis.backtest.trend.config import load_trend_config, load_trend_yaml, trend_config_from_mapping
from analysis.backtest.trend.markets import HK as MARKET_HK, US as MARKET_US, parse_symbol, trade_fees
from analysis.backtest.trend.prices import (
    CorporateAction,
    Holding,
    adjust_prices,
    dividend_credit,
    long_stop_hit,
    raw_from_adjusted,
    replay_holding,
    upward_gap_cancel,
)
from analysis.backtest.trend.runner import run_trend_backtest
from analysis.backtest.trend.entries.base import EntrySignal, EntryTactic
from analysis.backtest.trend.fills import entry_cancel_reason
from analysis.backtest.trend.exits import long_trend_failed, next_long_stop, reduce_shares
from analysis.backtest.trend.risk import estimate_cost_per_share, size_position
from analysis.backtest.trend.session import ReplayBar, run_account, run_empty_session
from analysis.backtest.trend.tape import (
    build_tape,
    canonical_row,
    geometric_emas,
    true_range_units,
)
from analysis.structure.config import PARAM_VERSION_V6, default_params
from analysis.structure.methods.phase import (
    DOWN_BIRTH,
    DOWNTREND,
    INSUFFICIENT,
    NEUTRAL,
    PENDING,
    RANGE,
    UNDECIDED,
    UP_BIRTH,
    UP_CONTINUE,
    UP_EXHAUSTION,
    UP_PULLBACK,
    UPTREND,
    _Run,
)
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


def _series_ohlc():
    closes = [100.0, 110.0, 105.0, 120.0, 130.0, 125.0]
    opens = [100.0, 100.0, 110.0, 105.0, 120.0, 130.0]
    highs = [101.0, 112.0, 106.0, 121.0, 132.0, 131.0]
    lows = [99.0, 108.0, 104.0, 119.0, 128.0, 124.0]
    volumes = [1000.0, 800.0, 900.0, 2000.0, 700.0, 600.0]
    return opens, highs, lows, closes, volumes


class PriceMarketTests(unittest.TestCase):
    def test_hand_calculated_factors_preserve_log_returns(self):
        opens, highs, lows, closes, volumes = _series_ohlc()
        adjusted = adjust_prices(
            opens[:4],
            highs[:4],
            lows[:4],
            closes[:4],
            volumes[:4],
            [
                CorporateAction(2, "dividend", 5.0),
                CorporateAction(3, "split", 2.0),
            ],
        )
        dividend_factor = 110.0 / 105.0
        self.assertEqual(adjusted.factor, (1.0, 1.0, dividend_factor, dividend_factor * 2.0))
        self.assertEqual(adjusted.split_ratio, (1.0, 1.0, 1.0, 2.0))
        self.assertAlmostEqual(adjusted.close[0], 100.0)
        self.assertAlmostEqual(adjusted.close[1], 110.0)
        self.assertAlmostEqual(adjusted.close[2], 105.0 * dividend_factor)
        self.assertAlmostEqual(adjusted.open[2], 110.0 * dividend_factor)
        self.assertAlmostEqual(adjusted.high[2], 106.0 * dividend_factor)
        self.assertAlmostEqual(adjusted.low[2], 104.0 * dividend_factor)
        self.assertAlmostEqual(adjusted.close[3], 120.0 * dividend_factor * 2.0)
        self.assertAlmostEqual(adjusted.open[3], 105.0 * dividend_factor * 2.0)
        self.assertEqual(adjusted.volume[2], 900.0)
        self.assertAlmostEqual(adjusted.volume[3], 1000.0)
        for index in range(1, 4):
            raw_return = math.log(closes[index] / closes[index - 1])
            adjusted_return = math.log(adjusted.close[index] / adjusted.close[index - 1])
            factor_return = math.log(adjusted.factor[index] / adjusted.factor[index - 1])
            self.assertAlmostEqual(adjusted_return, raw_return + factor_return)
        self.assertAlmostEqual(math.log(adjusted.close[1] / adjusted.close[0]), math.log(110.0 / 100.0))

    def test_future_dividend_does_not_change_past_prices_or_decisions(self):
        opens, highs, lows, closes, volumes = _series_ohlc()
        past = [
            CorporateAction(2, "dividend", 5.0),
            CorporateAction(3, "split", 2.0),
        ]
        holding = Holding(shares=100, cash=1000, adjusted_stop=80)
        original, original_rows, original_holding = replay_holding(
            opens,
            highs,
            lows,
            closes,
            volumes,
            past,
            holding,
            withholding=0.3,
            currency="HKD",
            times=[f"2024-01-0{index + 1}" for index in range(6)],
        )
        changed, changed_rows, _ = replay_holding(
            opens,
            highs,
            lows,
            closes,
            volumes,
            past + [CorporateAction(5, "dividend", 1.0)],
            holding,
            withholding=0.3,
            currency="HKD",
            times=[f"2024-01-0{index + 1}" for index in range(6)],
        )
        self.assertEqual(original.close[:4], changed.close[:4])
        self.assertEqual(original.open[:4], changed.open[:4])
        self.assertEqual(original.high[:4], changed.high[:4])
        self.assertEqual(original.low[:4], changed.low[:4])
        self.assertEqual(original.factor[:4], changed.factor[:4])
        past_rows = [row.canonical() for row in original_rows if row.index <= 3]
        changed_past = [row.canonical() for row in changed_rows if row.index <= 3]
        self.assertEqual(past_rows, changed_past)
        self.assertEqual(len(past_rows), 2)
        self.assertGreater(changed.factor[5], original.factor[5])
        dividend = original_rows[0]
        self.assertEqual(dividend.action, "派息入账")
        self.assertAlmostEqual(dividend.cash_credited, 350.0)
        self.assertAlmostEqual(dividend.tax, 150.0)
        self.assertEqual(dividend.shares_after, 100)
        self.assertAlmostEqual(dividend.cash_after, 1350.0)
        self.assertIn("入账现金 350 HKD", dividend.text)
        split = original_rows[1]
        self.assertEqual(split.shares_before, 100)
        self.assertEqual(split.shares_after, 200)
        self.assertEqual(split.cash_before, split.cash_after)
        self.assertAlmostEqual(split.raw_stop * split.factor, 80.0)
        self.assertAlmostEqual(original_holding.shares, 200)
        self.assertAlmostEqual(original_holding.cash, 1350)
        self.assertEqual(original_holding.adjusted_stop, 80)

    def test_ex_dividend_gap_does_not_stop_out_or_look_like_a_gap(self):
        opens = [100.0, 80.0]
        highs = [101.0, 81.0]
        lows = [99.0, 78.0]
        closes = [100.0, 80.0]
        volumes = [1000.0, 1000.0]
        adjusted = adjust_prices(
            opens,
            highs,
            lows,
            closes,
            volumes,
            [CorporateAction(1, "dividend", 20.0)],
        )
        self.assertAlmostEqual(adjusted.factor[1], 1.25)
        self.assertAlmostEqual(adjusted.open[1], 100.0)
        self.assertAlmostEqual(adjusted.close[1], 100.0)
        raw_stop = raw_from_adjusted(90.0, adjusted.factor[1])
        self.assertAlmostEqual(raw_stop, 72.0)
        self.assertGreater(lows[1], raw_stop)
        self.assertLess(lows[1], 90.0)
        self.assertFalse(long_stop_hit(lows[1], 90.0, adjusted.factor[1]))
        self.assertFalse(long_stop_hit(opens[1], 90.0, adjusted.factor[1]))
        self.assertTrue(long_stop_hit(72.0, 90.0, adjusted.factor[1]))
        a_t = 0.02
        self.assertGreater(math.log(closes[0] / opens[1]), 0.5 * a_t)
        self.assertFalse(upward_gap_cancel(opens[1], adjusted.factor[1], adjusted.close[0], a_t, 0.5))
        threshold = 100.0 * math.exp(0.5 * a_t)
        self.assertFalse(upward_gap_cancel(threshold, 1.0, 100.0, a_t, 0.5))
        self.assertTrue(upward_gap_cancel(threshold + 0.01, 1.0, 100.0, a_t, 0.5))
        self.assertFalse(upward_gap_cancel(101.0, 1.0, 100.0, 2.0, 0.5, price_axis="uniform"))
        self.assertTrue(upward_gap_cancel(101.01, 1.0, 100.0, 2.0, 0.5, price_axis="uniform"))

        credited = dividend_credit(200, 20, 0.0)
        self.assertEqual(credited.cash, 4000.0)
        self.assertEqual(credited.tax, 0.0)
        withheld = dividend_credit(200, 20, 0.3)
        self.assertEqual(withheld.cash, 2800.0)
        self.assertEqual(withheld.tax, 1200.0)

    def test_split_scales_shares_and_keeps_the_adjusted_stop(self):
        opens = [100.0, 50.0]
        highs = [101.0, 52.0]
        lows = [99.0, 48.0]
        closes = [100.0, 50.0]
        volumes = [1000.0, 2000.0]
        _, rows, holding = replay_holding(
            opens,
            highs,
            lows,
            closes,
            volumes,
            [CorporateAction(1, "split", 2.0)],
            Holding(shares=100, cash=5000, adjusted_stop=80),
            withholding=0.0,
            currency="USD",
        )
        self.assertEqual(holding.shares, 200)
        self.assertEqual(holding.cash, 5000)
        self.assertEqual(holding.adjusted_stop, 80)
        decision = rows[0]
        self.assertEqual(decision.action, "拆股")
        self.assertEqual(decision.cash_before, decision.cash_after)
        self.assertAlmostEqual(decision.factor, 2.0)
        self.assertAlmostEqual(decision.raw_stop, 40.0)
        self.assertAlmostEqual(decision.raw_stop * decision.factor, decision.adjusted_stop)
        self.assertFalse(long_stop_hit(lows[1], 80.0, decision.factor))
        self.assertLess(lows[1], 80.0)
        self.assertIn("现金不变仍为 5000 USD", decision.text)

        reverse = adjust_prices(
            [100.0, 200.0],
            [101.0, 202.0],
            [99.0, 198.0],
            [100.0, 200.0],
            [1000.0, 500.0],
            [CorporateAction(1, "split", 0.5)],
        )
        self.assertAlmostEqual(reverse.close[1], 100.0)
        self.assertAlmostEqual(reverse.volume[1], 1000.0)
        signal = reverse.close[0]
        self.assertGreater(200.0, signal * math.exp(0.5 * 0.02))
        self.assertFalse(upward_gap_cancel(200.0, reverse.factor[1], signal, 0.02, 0.5))

    def test_hk_and_us_trade_fees(self):
        config = load_trend_config(today=date(2026, 10, 13))
        hk = config.markets[MARKET_HK]
        us = config.markets[MARKET_US]
        buy = trade_fees(hk, side="buy", notional=10000, shares=100)
        sell = trade_fees(hk, side="sell", notional=10000, shares=100)
        self.assertEqual(buy.commission, 3.0)
        self.assertEqual(buy.stamp_duty, 10.0)
        self.assertAlmostEqual(buy.exchange_fee, 0.85)
        self.assertAlmostEqual(buy.settlement_fee, 0.2)
        self.assertEqual(buy.sec_fee, 0.0)
        self.assertEqual(buy.taf, 0.0)
        self.assertAlmostEqual(buy.total, 14.05)
        self.assertEqual(sell.stamp_duty, buy.stamp_duty)
        self.assertEqual(sell.commission, buy.commission)
        self.assertEqual(sell.sec_fee, 0.0)
        self.assertEqual(trade_fees(hk, side="buy", notional=10001, shares=100).stamp_duty, 11.0)
        self.assertEqual(trade_fees(hk, side="buy", notional=100, shares=100).commission, 3.0)
        self.assertEqual(trade_fees(hk, side="sell", notional=100, shares=100).stamp_duty, 1.0)
        with_platform = trade_fees(replace(hk, platform_fee=15), side="buy", notional=10000, shares=100)
        self.assertEqual(with_platform.platform_fee, 15.0)
        self.assertAlmostEqual(with_platform.total, 29.05)

        us_buy = trade_fees(us, side="buy", notional=5000, shares=100)
        us_sell = trade_fees(us, side="sell", notional=5000, shares=100)
        self.assertEqual(us_buy.commission, 1.0)
        self.assertEqual(us_buy.stamp_duty, 0.0)
        self.assertEqual(us_buy.sec_fee, 0.0)
        self.assertEqual(us_buy.taf, 0.0)
        self.assertEqual(us_sell.stamp_duty, 0.0)
        self.assertAlmostEqual(us_sell.sec_fee, 0.139)
        self.assertAlmostEqual(us_sell.taf, 0.0166)
        self.assertEqual(us_sell.commission, 1.0)
        large_buy = trade_fees(us, side="buy", notional=2_000_000, shares=100_000)
        large_sell = trade_fees(us, side="sell", notional=2_000_000, shares=100_000)
        self.assertEqual(large_buy.commission, 500.0)
        self.assertEqual(large_buy.sec_fee, 0.0)
        self.assertEqual(large_buy.taf, 0.0)
        self.assertEqual(large_sell.commission, 500.0)
        self.assertAlmostEqual(large_sell.sec_fee, 55.6)
        self.assertEqual(large_sell.taf, 8.3)
        capped = trade_fees(us, side="buy", notional=10, shares=1)
        self.assertAlmostEqual(capped.commission, 0.1)

    def test_symbol_codes_and_default_withholding(self):
        self.assertEqual(parse_symbol("700"), parse_symbol("00700.HK"))
        self.assertEqual(parse_symbol("700").code, "00700")
        self.assertEqual(parse_symbol("700").market, MARKET_HK)
        self.assertEqual(parse_symbol("NVDA").code, "NVDA")
        self.assertEqual(parse_symbol("NVDA.US").market, MARKET_US)
        self.assertEqual(parse_symbol("NVDA"), parse_symbol("NVDA.US"))
        with self.assertRaisesRegex(ValueError, "A 股"):
            parse_symbol("600519")
        config = load_trend_config(today=date(2026, 10, 13))
        self.assertEqual(config.markets[MARKET_HK].dividend_withholding, 0.0)
        self.assertEqual(config.markets[MARKET_US].dividend_withholding, 0.3)
        self.assertEqual(config.symbols[0].dividend_withholding, 0.0)
        self.assertEqual(config.symbols[2].dividend_withholding, 0.3)
        self.assertEqual(dividend_credit(100, 2, config.markets[MARKET_HK].dividend_withholding).cash, 200)
        self.assertEqual(dividend_credit(100, 2, config.markets[MARKET_US].dividend_withholding).cash, 140)


class _Script(EntryTactic):
    name = "script"

    def __init__(self, signals):
        self.signals = signals
        self.calls = []

    def propose(self, bars):
        self.calls.append(len(bars) - 1)
        return self.signals.get(len(bars) - 1)


def _bar(time, open_, high, low, close, **kw):
    return ReplayBar(
        time=time,
        open=open_,
        high=high,
        low=low,
        close=close,
        structure=kw.pop("structure", UPTREND),
        category=kw.pop("category", UP_CONTINUE),
        a_t=kw.pop("a_t", 5.0),
        factor=kw.pop("factor", 1.0),
        dividend=kw.pop("dividend", 0.0),
        split=kw.pop("split", 1.0),
    )
    if kw:
        raise TypeError(sorted(kw))


def _calm(time, **kw):
    return _bar(time, 100, 102, 96, 101, **kw)


def _sig(stop=90.0, text="测试入场"):
    return EntrySignal(stop, text)


def _account(bars, signals, **kw):
    config = load_trend_config(today=date(2026, 10, 13))
    strategy = config.strategies["breakout"]
    risk = {"risk_per_trade": 0.02}
    risk.update(kw.pop("risk", {}))
    order = {"slippage_rate": 0.0}
    order.update(kw.pop("order", {}))
    exit_over = kw.pop("exit", {})
    strategy = replace(
        strategy,
        risk=replace(strategy.risk, **risk),
        order=replace(strategy.order, **order),
    )
    if exit_over:
        strategy = replace(strategy, exit=replace(strategy.exit, **exit_over))
    market = replace(
        config.markets[MARKET_HK],
        commission_rate=0.0,
        min_commission=0.0,
        platform_fee=0.0,
        stamp_duty_rate=0.0,
        exchange_fee_rate=0.0,
        settlement_rate=0.0,
    )
    market_over = kw.pop("market", {})
    if market_over:
        market = replace(market, **market_over)
    lot_size = kw.pop("lot_size", 100)
    capital = kw.pop("capital", 100_000.0)
    price_axis = kw.pop("price_axis", "uniform")
    withholding = kw.pop("withholding", 0.0)
    if kw:
        raise TypeError(sorted(kw))
    tactic = _Script(signals)
    result = run_account(
        bars,
        tactic,
        strategy,
        market,
        lot_size=lot_size,
        capital=capital,
        price_axis=price_axis,
        withholding=withholding,
    )
    return result, tactic


def _by(result, action):
    return [row for row in result.decisions if row.action == action]


class AccountRuleTests(unittest.TestCase):
    def test_position_size_takes_the_smallest_constraint_then_rounds_to_lots(self):
        market = self._market()
        sized = size_position(
            equity=100_000,
            cash=30_000,
            open_price=100,
            raw_stop=90,
            lot_size=100,
            risk_per_trade=0.05,
            one_lot_max_risk=0.05,
            max_gross_exposure=0.25,
            slippage_rate=0,
            market=market,
        )
        self.assertEqual((sized.q_risk, sized.q_cash, sized.q_cap), (500, 300, 250))
        self.assertEqual(sized.shares, 200)
        self.assertFalse(sized.one_lot)
        self.assertAlmostEqual(sized.planned_risk, 0.02)

        cash_bound = size_position(
            equity=100_000, cash=15_000, open_price=100, raw_stop=90, lot_size=100,
            risk_per_trade=0.05, one_lot_max_risk=0.05, max_gross_exposure=1,
            slippage_rate=0, market=market,
        )
        self.assertEqual(cash_bound.q_cash, 150)
        self.assertEqual(cash_bound.shares, 100)

        risk_bound = size_position(
            equity=100_000, cash=100_000, open_price=100, raw_stop=90, lot_size=100,
            risk_per_trade=0.01, one_lot_max_risk=0.02, max_gross_exposure=1,
            slippage_rate=0, market=market,
        )
        self.assertEqual(risk_bound.q_risk, 100)
        self.assertEqual(risk_bound.shares, 100)

        capped = size_position(
            equity=100_000, cash=100_000, open_price=100, raw_stop=90, lot_size=100,
            risk_per_trade=0.005, one_lot_max_risk=0.02, max_gross_exposure=0.05,
            slippage_rate=0, market=market,
        )
        self.assertEqual(capped.shares, 0)
        self.assertEqual(capped.reason, "占用上限")
        self.assertFalse(capped.one_lot)

    def test_estimated_cost_ignores_per_order_fees_and_changes_the_share_count(self):
        plain = self._market()
        rated = replace(plain, commission_rate=0.01, min_commission=50, platform_fee=15)
        self.assertAlmostEqual(estimate_cost_per_share(rated, 100, 90, 0.001), 1.99)
        self.assertAlmostEqual(
            estimate_cost_per_share(rated, 100, 90, 0.001),
            estimate_cost_per_share(replace(rated, min_commission=0, platform_fee=0), 100, 90, 0.001),
        )
        kwargs = dict(
            equity=100_000, cash=100_000, open_price=100, raw_stop=90, lot_size=1,
            risk_per_trade=0.01, one_lot_max_risk=0.02, max_gross_exposure=1, slippage_rate=0,
        )
        self.assertEqual(size_position(market=plain, **kwargs).shares, 100)
        self.assertEqual(size_position(market=replace(plain, commission_rate=0.01), **kwargs).shares, 84)

        tight = dict(
            equity=100_000, cash=10_010, open_price=100, raw_stop=90, lot_size=1,
            risk_per_trade=0.05, one_lot_max_risk=0.05, max_gross_exposure=1, slippage_rate=0,
        )
        self.assertEqual(size_position(market=plain, **tight).q_cash, 100)
        self.assertEqual(size_position(market=replace(plain, platform_fee=15), **tight).q_cash, 99)
        self.assertEqual(
            size_position(market=plain, **tight).q_risk,
            size_position(market=replace(plain, platform_fee=15), **tight).q_risk,
        )

        us = load_trend_config(today=date(2026, 10, 13)).markets[MARKET_US]
        self.assertAlmostEqual(estimate_cost_per_share(us, 100, 90, 0.0005), 0.057668)

    def test_one_lot_rule_buys_rounds_or_cancels(self):
        bought, _ = _account(
            [_calm("d0"), _calm("d1")],
            {0: _sig()},
            risk={"risk_per_trade": 0.005, "one_lot_max_risk": 0.01},
        )
        entry = _by(bought, "入场")[0]
        self.assertEqual(entry.shares_after, 100)
        self.assertTrue(entry.one_lot)
        self.assertAlmostEqual(entry.planned_risk, 0.01)
        self.assertIn("风险预算不够一手，按一手规则买入，计划风险", entry.text)

        closed, _ = _account(
            [_calm("d0"), _calm("d1")],
            {0: _sig()},
            risk={"risk_per_trade": 0.005, "one_lot_max_risk": 0},
        )
        self.assertEqual(_by(closed, "入场"), [])
        cancel = _by(closed, "取消")[0]
        self.assertEqual(cancel.reason, "风险预算")
        self.assertIn("一手规则已关闭", cancel.text)

        exceeded, _ = _account(
            [_calm("d0"), _calm("d1")],
            {0: _sig()},
            risk={"risk_per_trade": 0.005, "one_lot_max_risk": 0.009},
        )
        self.assertEqual(_by(exceeded, "取消")[0].reason, "风险预算")
        self.assertIn("一手规则也不够", _by(exceeded, "取消")[0].text)

    def test_account_floors_the_risk_budget_to_a_whole_lot(self):
        result, _ = _account(
            [_calm("d0"), _calm("d1")],
            {0: _sig()},
            risk={"risk_per_trade": 0.025},
        )
        entry = _by(result, "入场")[0]
        self.assertEqual(entry.shares_after, 200)
        self.assertFalse(entry.one_lot)
        self.assertAlmostEqual(entry.planned_risk, 0.02)

    def test_cash_and_exposure_each_cancel_on_their_own(self):
        cash, _ = _account(
            [_calm("d0"), _calm("d1")],
            {0: _sig()},
            capital=10_050,
            market={"min_commission": 60},
            risk={"risk_per_trade": 0.2, "one_lot_max_risk": 0.2},
        )
        self.assertEqual(_by(cash, "取消")[0].reason, "现金")
        self.assertEqual(_by(cash, "入场"), [])

        cap, _ = _account(
            [_calm("d0"), _calm("d1")],
            {0: _sig()},
            risk={"risk_per_trade": 0.05, "one_lot_max_risk": 0.05, "max_gross_exposure": 0.05},
        )
        self.assertEqual(_by(cap, "取消")[0].reason, "占用上限")

    def test_gap_distance_and_open_through_stop_cancel_before_a_fill(self):
        self.assertIsNone(entry_cancel_reason(
            raw_open=101, factor=1, adjusted_stop=90, adjusted_signal_close=100,
            a_t=2, gap_cancel=0.5, max_stop_distance=20, price_axis="uniform",
        ))
        self.assertEqual(entry_cancel_reason(
            raw_open=101.01, factor=1, adjusted_stop=90, adjusted_signal_close=100,
            a_t=2, gap_cancel=0.5, max_stop_distance=20, price_axis="uniform",
        ), "跳空")
        self.assertIsNone(entry_cancel_reason(
            raw_open=88, factor=1, adjusted_stop=80, adjusted_signal_close=88,
            a_t=2, gap_cancel=0.5, max_stop_distance=4, price_axis="uniform",
        ))
        self.assertEqual(entry_cancel_reason(
            raw_open=88.01, factor=1, adjusted_stop=80, adjusted_signal_close=88,
            a_t=2, gap_cancel=0.5, max_stop_distance=4, price_axis="uniform",
        ), "止损距离")
        self.assertEqual(entry_cancel_reason(
            raw_open=80, factor=1, adjusted_stop=80, adjusted_signal_close=100,
            a_t=5, gap_cancel=0.5, max_stop_distance=4, price_axis="uniform",
        ), "开盘价不高于止损")

        held, _ = _account(
            [_bar("d0", 100, 101, 99, 100, a_t=2), _bar("d1", 101, 102, 95, 101, a_t=2)],
            {0: _sig()},
            order={"max_stop_distance": 20},
        )
        self.assertEqual(len(_by(held, "入场")), 1)
        gapped, _ = _account(
            [_bar("d0", 100, 101, 99, 100, a_t=2), _bar("d1", 101.01, 102, 95, 101.01, a_t=2)],
            {0: _sig()},
            order={"max_stop_distance": 20},
        )
        self.assertEqual(_by(gapped, "取消")[0].reason, "跳空")
        self.assertEqual(_by(gapped, "入场"), [])

        exact, _ = _account(
            [_bar("d0", 88, 89, 87, 88, a_t=2), _bar("d1", 88, 89, 85, 88, a_t=2)],
            {0: _sig(80)},
        )
        self.assertEqual(len(_by(exact, "入场")), 1)
        far, _ = _account(
            [_bar("d0", 88, 89, 87, 88, a_t=2), _bar("d1", 88.01, 89, 85, 88.01, a_t=2)],
            {0: _sig(80)},
        )
        self.assertEqual(_by(far, "取消")[0].reason, "止损距离")

        touched, _ = _account(
            [_bar("d0", 100, 101, 99, 100), _bar("d1", 80, 81, 79, 80)],
            {0: _sig(80)},
        )
        self.assertEqual(_by(touched, "取消")[0].reason, "开盘价不高于止损")
        self.assertEqual(_by(touched, "止损"), [])

    def test_birth_does_not_trail_and_a_new_high_cannot_stop_out_the_same_bar(self):
        result, tactic = _account(
            [
                _bar("d0", 100, 101, 99, 100, category=UP_BIRTH),
                _bar("d1", 100, 140, 96, 120, category=UP_BIRTH),
                _bar("d2", 120, 140, 110, 130, category=UP_BIRTH),
                _bar("d3", 130, 140, 100, 135, category=UP_CONTINUE),
                _bar("d4", 130, 140, 126, 132, category=UP_CONTINUE, a_t=40),
            ],
            {0: _sig()},
        )
        self.assertEqual(tactic.calls, [0])
        self.assertEqual(_by(result, "止损"), [])
        self.assertEqual(result.equity[1].next_stop, 90)
        self.assertEqual(result.equity[2].next_stop, 90)
        self.assertGreater(result.equity[2].shares, 0)
        self.assertEqual(result.equity[3].effective_stop, 90)
        self.assertEqual(result.equity[3].next_stop, 125)
        self.assertEqual(result.equity[4].effective_stop, 125)
        self.assertEqual(result.equity[4].next_stop, 125)

        states = ("up_continue", "up_pullback")
        self.assertEqual(next_long_stop(
            90, 140, 5, buffer=3, price_axis="uniform", category=UP_BIRTH, trail_states=states,
        ), 90)
        self.assertEqual(next_long_stop(
            90, 140, 5, buffer=3, price_axis="uniform", category=UP_CONTINUE, trail_states=states,
        ), 125)
        self.assertEqual(next_long_stop(
            125, 140, 40, buffer=3, price_axis="uniform", category=UP_CONTINUE, trail_states=states,
        ), 125)
        self.assertEqual(next_long_stop(
            90, 130, 5, buffer=3, price_axis="uniform", category=UP_PULLBACK, trail_states=states,
        ), 115)

    def test_log_trail_is_multiplicative_and_waits_for_the_next_bar(self):
        expected = 110 * math.exp(-0.15)
        result, _ = _account(
            [
                _bar("d0", 100, 101, 99, 100, a_t=0.05),
                _bar("d1", 100, 110, 93, 105, a_t=0.05),
                _bar("d2", 93, 95, 92, 94, a_t=0.05),
            ],
            {0: _sig()},
            price_axis="log",
        )
        self.assertEqual(_by(result, "止损")[0].index, 2)
        self.assertEqual(_by(result, "止损")[0].reason, "跳空穿过止损")
        self.assertAlmostEqual(_by(result, "止损")[0].price, 93)
        self.assertEqual(result.equity[1].effective_stop, 90)
        self.assertAlmostEqual(result.equity[1].next_stop, expected)
        self.assertEqual([row.index for row in _by(result, "止损")], [2])

    def test_old_stop_still_fills_when_the_same_bar_makes_a_new_high(self):
        result, _ = _account(
            [_calm("d0"), _calm("d1"), _bar("d2", 100, 140, 89, 100)],
            {0: _sig()},
        )
        stop = _by(result, "止损")
        self.assertEqual(len(stop), 1)
        self.assertEqual(stop[0].index, 2)
        self.assertEqual(stop[0].reason, "盘中触及止损")
        self.assertEqual(stop[0].price, 90)
        self.assertEqual(stop[0].raw_stop, 90)

    def test_intraday_stop_happens_before_the_close_state(self):
        result, _ = _account(
            [
                _calm("d0"),
                _calm("d1"),
                _bar("d2", 100, 110, 89, 100, structure=RANGE, category=NEUTRAL),
                _calm("d3", structure=RANGE, category=NEUTRAL),
            ],
            {0: _sig()},
        )
        self.assertEqual(_by(result, "趋势退出"), [])
        stop = _by(result, "止损")[0]
        self.assertEqual(stop.index, 2)
        self.assertEqual(stop.price, 90)
        self.assertEqual(stop.reason, "盘中触及止损")

    def test_a_gap_through_the_stop_fills_at_the_open_and_can_lose_more_than_planned(self):
        result, _ = _account(
            [_calm("d0"), _calm("d1"), _bar("d2", 70, 72, 60, 71)],
            {0: _sig()},
            risk={"risk_per_trade": 0.01},
        )
        stop = _by(result, "止损")[0]
        self.assertEqual(stop.reason, "跳空穿过止损")
        self.assertEqual(stop.price, 70)
        self.assertEqual(stop.raw_stop, 90)
        self.assertAlmostEqual(stop.planned_risk, 0.01)
        self.assertAlmostEqual(stop.actual_loss_fraction, 0.03)
        self.assertGreater(stop.actual_loss_fraction, stop.planned_risk)
        self.assertAlmostEqual(result.final_cash, 97_000)

        equal, _ = _account(
            [_calm("d0"), _calm("d1"), _bar("d2", 90, 92, 80, 91)],
            {0: _sig()},
            risk={"risk_per_trade": 0.01},
        )
        touched = _by(equal, "止损")[0]
        self.assertEqual(touched.reason, "跳空穿过止损")
        self.assertEqual(touched.price, 90)

    def test_slippage_hurts_the_position_and_a_stop_ignores_the_daily_trade_cap(self):
        result, _ = _account(
            [_calm("d0"), _bar("d1", 100, 102, 89, 101)],
            {0: _sig()},
            order={"slippage_rate": 0.001, "max_trades_per_day": 1},
        )
        entry = _by(result, "入场")[0]
        stop = _by(result, "止损")[0]
        self.assertAlmostEqual(entry.price, 100.1)
        self.assertAlmostEqual(stop.price, 89.91)
        self.assertEqual(result.final_shares, 0)

    def test_exhaustion_reduces_once_and_a_new_position_can_reduce_again(self):
        once, _ = _account(
            [
                _calm("d0"),
                _calm("d1"),
                _calm("d2", category=UP_EXHAUSTION),
                _calm("d3", category=UP_EXHAUSTION),
                _calm("d4", category=UP_EXHAUSTION),
                _bar("d5", 100, 130, 96, 120),
            ],
            {0: _sig()},
            exit={"reduce_fraction": 0.5},
        )
        reduces = _by(once, "减仓")
        self.assertEqual([row.index for row in reduces], [3])
        self.assertEqual(reduces[0].reason, "衰竭减仓")
        self.assertEqual(reduces[0].shares_before, 200)
        self.assertEqual(reduces[0].shares_after, 100)
        self.assertEqual(once.equity[4].shares, 100)
        self.assertEqual(once.equity[5].shares, 100)
        self.assertEqual(once.equity[5].next_stop, 115)

        flat, _ = _account(
            [_calm("d0"), _calm("d1"), _calm("d2", category=UP_EXHAUSTION), _calm("d3", category=UP_EXHAUSTION)],
            {0: _sig()},
            risk={"risk_per_trade": 0.01},
        )
        flatten = _by(flat, "减仓")[0]
        self.assertEqual(flatten.reason, "取整全平")
        self.assertIn("取整", flatten.text)
        self.assertEqual(flatten.shares_after, 0)

        again, _ = _account(
            [
                _calm("d0"), _calm("d1"),
                _calm("d2", category=UP_EXHAUSTION),
                _calm("d3", category=UP_EXHAUSTION),
                _calm("d4"),
                _calm("d5", category=UP_EXHAUSTION),
                _calm("d6", category=UP_EXHAUSTION),
            ],
            {0: _sig(), 3: _sig()},
            risk={"risk_per_trade": 0.01},
        )
        self.assertEqual([row.index for row in _by(again, "入场")], [1, 4])
        self.assertEqual([row.index for row in _by(again, "减仓")], [3, 6])
        self.assertNotIn(3, [row.index for row in _by(again, "入场")])

        kept, _ = _account(
            [
                _calm("d0"), _calm("d1"),
                _calm("d2", category=UP_EXHAUSTION),
                _calm("d3", category=UP_EXHAUSTION),
            ],
            {0: _sig()},
            exit={"exhaustion_action": "none"},
        )
        self.assertEqual(_by(kept, "减仓"), [])
        self.assertEqual(kept.equity[-1].shares, 200)

        deferred, _ = _account(
            [
                _calm("d0"),
                _calm("day"),
                _calm("d2", category=UP_EXHAUSTION),
                _calm("day", category=UP_EXHAUSTION),
                _calm("d4", category=UP_EXHAUSTION),
            ],
            {0: _sig()},
            order={"max_trades_per_day": 1},
        )
        self.assertEqual([row.index for row in _by(deferred, "减仓")], [4])
        self.assertEqual(deferred.equity[3].shares, 200)
        self.assertEqual(deferred.equity[4].shares, 100)

    def test_pullback_holds_and_trends_the_stop_without_exiting(self):
        result, tactic = _account(
            [_calm("d0"), _calm("d1"), _bar("d2", 100, 130, 96, 120, category=UP_PULLBACK)],
            {0: _sig()},
        )
        self.assertEqual(tactic.calls, [0])
        self.assertEqual(_by(result, "趋势退出"), [])
        self.assertEqual(_by(result, "减仓"), [])
        self.assertEqual(result.equity[2].shares, 200)
        self.assertEqual(result.equity[2].next_stop, 115)

    def test_trend_failure_flattens_next_bar_and_does_not_open_a_short(self):
        cases = (
            (RANGE, NEUTRAL),
            (UNDECIDED, PENDING),
            (DOWNTREND, DOWN_BIRTH),
            (None, INSUFFICIENT),
        )
        for structure, category in cases:
            with self.subTest(structure=structure):
                result, tactic = _account(
                    [
                        _calm("d0"),
                        _calm("d1"),
                        _calm("d2", structure=structure, category=category),
                        _calm("d3", structure=structure, category=category),
                        _calm("d4", structure=structure, category=category),
                    ],
                    {0: _sig(), 3: _sig()},
                )
                exit_row = _by(result, "趋势退出")[0]
                self.assertEqual(exit_row.index, 3)
                self.assertEqual(exit_row.price, 100)
                self.assertGreater(exit_row.shares_before, 0)
                self.assertEqual(exit_row.shares_after, 0)
                self.assertIn("不开反向仓", exit_row.text)
                self.assertEqual([row.index for row in _by(result, "入场")], [1, 4])
                self.assertNotIn(3, [row.index for row in _by(result, "入场")])
                self.assertTrue(all(row.shares_after >= 0 for row in result.decisions))
                self.assertNotIn(1, tactic.calls)
                self.assertNotIn(2, tactic.calls)

        self.assertTrue(long_trend_failed(RANGE))
        self.assertTrue(long_trend_failed(None))
        self.assertFalse(long_trend_failed(UPTREND))

    def test_trend_failure_and_exhaustion_on_the_same_bar_flatten_once(self):
        result, _ = _account(
            [
                _calm("d0"),
                _calm("d1"),
                _calm("d2", structure=RANGE, category=UP_EXHAUSTION),
                _calm("d3", structure=RANGE, category=UP_EXHAUSTION),
            ],
            {0: _sig()},
        )
        self.assertEqual(_by(result, "减仓"), [])
        exit_row = _by(result, "趋势退出")[0]
        self.assertEqual(exit_row.shares_after, 0)
        self.assertIn("衰竭", exit_row.text)
        self.assertIn("不开反向仓", exit_row.text)

    def test_plugin_is_not_asked_while_a_position_is_open(self):
        result, tactic = _account(
            [_calm("d0"), _calm("d1"), _calm("d2"), _calm("d3")],
            {0: _sig(), 1: _sig(), 2: _sig(), 3: _sig()},
        )
        self.assertEqual(tactic.calls, [0])
        self.assertEqual(len(_by(result, "入场")), 1)
        self.assertEqual(result.equity[-1].shares, 200)

    def test_the_last_bar_does_not_invent_a_fill(self):
        pending, _ = _account([_calm("d0")], {0: _sig()})
        self.assertEqual(_by(pending, "取消")[0].reason, "没有下一根")
        self.assertEqual(pending.final_shares, 0)

        held, _ = _account(
            [_calm("d0"), _calm("d1"), _calm("d2", structure=RANGE, category=NEUTRAL)],
            {0: _sig()},
        )
        self.assertEqual(_by(held, "趋势退出"), [])
        self.assertGreater(held.final_shares, 0)
        self.assertEqual(held.equity[-1].shares, held.final_shares)

    def test_dividend_and_split_do_not_look_like_a_stop(self):
        dividend, _ = _account(
            [_calm("d0"), _calm("d1"), _bar("d2", 80, 82, 78, 80, factor=1.25, dividend=20)],
            {0: _sig()},
            risk={"risk_per_trade": 0.01},
            withholding=0.3,
        )
        self.assertEqual(_by(dividend, "止损"), [])
        self.assertEqual(dividend.equity[2].shares, 100)
        self.assertAlmostEqual(dividend.equity[2].cash, 91_400)
        self.assertEqual(dividend.equity[2].effective_stop, 90)
        self.assertAlmostEqual(dividend.equity[2].raw_stop, 72)
        self.assertIn("入账现金 1400", _by(dividend, "派息入账")[0].text)

        split, _ = _account(
            [_calm("d0"), _calm("d1"), _bar("d2", 50, 52, 48, 50, factor=2, split=2)],
            {0: _sig()},
            risk={"risk_per_trade": 0.01},
        )
        self.assertEqual(_by(split, "止损"), [])
        row = _by(split, "拆股")[0]
        self.assertEqual(row.shares_before, 100)
        self.assertEqual(row.shares_after, 200)
        self.assertAlmostEqual(split.equity[2].cash, 90_000)
        self.assertEqual(split.equity[2].effective_stop, 90)
        self.assertAlmostEqual(split.equity[2].raw_stop, 45)

    def test_reduce_share_counts(self):
        self.assertEqual(reduce_shares(200, 0.5, 100), (100, False))
        self.assertEqual(reduce_shares(300, 0.5, 100), (100, False))
        self.assertEqual(reduce_shares(100, 0.5, 100), (100, True))

    def _market(self):
        config = load_trend_config(today=date(2026, 10, 13))
        return replace(
            config.markets[MARKET_HK],
            commission_rate=0.0,
            min_commission=0.0,
            platform_fee=0.0,
            stamp_duty_rate=0.0,
            exchange_fee_rate=0.0,
            settlement_rate=0.0,
        )


if __name__ == "__main__":
    unittest.main()
