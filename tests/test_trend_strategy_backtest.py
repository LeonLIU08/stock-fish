"""趋势策略回测：配置、磁带、价格口径、账户规则和入场插件。"""
from __future__ import annotations

import copy
import importlib.util
import math
import subprocess
import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from analysis.backtest.trend.config import (
    load_trend_config,
    load_trend_yaml,
    read_symbol_file,
    trend_config_from_mapping,
)
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
from analysis.backtest.report import render_html_case
from analysis.backtest.trend.audit import audit_summary
from analysis.backtest.trend.baselines import simulate_buy_hold, simulate_even_dca, split_adjusted_price
from analysis.backtest.trend.entries import create_entry, list_entries
from analysis.backtest.trend.feed import enclose_ohlc, restore_traded_prices
from analysis.backtest.trend.fills import whole_shares
from analysis.backtest.trend.entries.base import EntrySignal, EntryTactic
from analysis.backtest.trend.entries.registry import register_entry
from analysis.backtest.trend.fills import entry_cancel_reason
from analysis.backtest.trend.exits import long_trend_failed, next_long_stop, reduce_shares
from analysis.backtest.trend.risk import estimate_cost_per_share, size_position
from analysis.backtest.trend.runner import _case_config, run_trend_backtest
from analysis.backtest.trend.session import ReplayBar, run_account, run_empty_session
from analysis.backtest.trend.tape import (
    build_tape,
    canonical_row,
    geometric_emas,
    true_range_units,
)
from analysis.structure.config import PARAM_VERSION_V6, default_params
from analysis.structure.methods.phase import (
    ACCUMULATION,
    DISTRIBUTION,
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
PYTHON = "/usr/bin/python3"


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
        self.assertEqual(config.structure.scale, "short")
        self.assertEqual(config.structure.param_version, "structure-params-v6")
        self.assertEqual([run.key for run in config.runs], [
            "breakout",
            "breakout_no_reduce",
            "pullback",
            "pullback_no_reduce",
            "trend",
            "d0",
            "d1",
        ])
        trend = config.strategies["trend"]
        self.assertEqual(trend.entry.buy_scale, "short")
        self.assertEqual(trend.entry.sell_scale, "mid")
        self.assertEqual(trend.risk.sizing, "exposure")
        self.assertEqual(trend.exit.mode, "scale_downtrend")
        d0 = config.strategies["d0"]
        self.assertEqual(d0.entry.scale, "short")
        self.assertEqual(d0.risk.sizing, "exposure")
        self.assertEqual(d0.exit.mode, "scale_downtrend")
        self.assertEqual(d0.exit.exhaustion_action, "none")
        d1 = config.strategies["d1"]
        self.assertEqual(d1.entry.scale, "short")
        self.assertEqual(d1.exit.drawdown, 0.15)
        self.assertEqual(d1.risk.sizing, "exposure")
        self.assertIsNone(trend.order.max_stop_distance)
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

    def test_symbol_file_fills_missing_hk_lots_from_the_lookup(self):
        listing = ROOT / "scripts" / "hkconnect_06Oct.txt"
        self.assertGreater(len(read_symbol_file(listing)), 100)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "names.txt"
            path.write_text("00005\n# skip\n\n00700\n00005\n", encoding="utf-8")
            self.assertEqual(read_symbol_file(path), ("00005", "00700"))
        config = load_trend_config(
            overrides={"symbols": ["00005", "00700"]},
            today=date(2026, 10, 13),
            lot_lookup=lambda codes: {"00005": (400, "汇丰控股")} if "00005" in codes else {},
        )
        self.assertEqual(config.symbols[0].code, "00005")
        self.assertEqual(config.symbols[0].lot_size, 400)
        self.assertEqual(config.symbols[0].lot_size_source, "longbridge")
        self.assertEqual(config.symbols[0].name, "汇丰控股")
        self.assertEqual(config.symbols[1].code, "00700")
        self.assertEqual(config.symbols[1].lot_size, 100)
        self.assertEqual(config.symbols[1].lot_size_source, "config")
        self.assertEqual(config.symbols[1].name, "腾讯控股")

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

    def test_finish_returns_one_after_skipped_symbols(self):
        path = ROOT / "scripts" / "run_trend_backtest.py"
        spec = importlib.util.spec_from_file_location("run_trend_backtest_cli", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        code = module._finish({
            "index_html": "index.html",
            "report_md": "summary.md",
            "output_dir": ".",
            "failures": [{"symbol": "00004", "error": "最高价和最低价包不住开盘价或收盘价"}],
        })
        self.assertEqual(code, 1)
        self.assertEqual(module._finish({
            "index_html": "index.html",
            "report_md": "summary.md",
            "output_dir": ".",
            "failures": [],
        }), 0)

    def test_trend_buy_scale_must_match_the_run_scale(self):
        data = _raw()
        data["structure"]["scale"] = "mid"
        with self.assertRaisesRegex(ValueError, "买入尺度必须与 structure.scale 相同"):
            trend_config_from_mapping(data, today=date(2026, 10, 13))

    def test_trend_scales_must_differ_and_reduce_is_rejected(self):
        data = _raw()
        data["strategies"]["trend"]["entry"]["sell_scale"] = "short"
        with self.assertRaisesRegex(ValueError, "不能相同"):
            trend_config_from_mapping(data, today=date(2026, 10, 13))
        data = _raw()
        data["runs"][-1]["exhaustion_action"] = "reduce"
        with self.assertRaisesRegex(ValueError, "不能减仓"):
            trend_config_from_mapping(data, today=date(2026, 10, 13))
        data = _raw()
        data["strategies"]["trend"]["order"]["max_stop_distance"] = 4
        with self.assertRaisesRegex(ValueError, "未登记的字段: max_stop_distance"):
            trend_config_from_mapping(data, today=date(2026, 10, 13))

    def test_yaml_without_trend_still_loads(self):
        data = _raw()
        del data["strategies"]["trend"]
        del data["strategies"]["d0"]
        del data["strategies"]["d1"]
        data["runs"] = [
            item for item in data["runs"] if item["strategy"] not in ("trend", "d0", "d1")
        ]
        config = trend_config_from_mapping(data, today=date(2026, 10, 13))
        self.assertNotIn("trend", config.strategies)
        self.assertNotIn("d0", config.strategies)
        self.assertNotIn("d1", config.strategies)
        self.assertEqual([item.key for item in config.runs], [
            "breakout",
            "breakout_no_reduce",
            "pullback",
            "pullback_no_reduce",
        ])

    def test_d0_scale_must_match_the_run_scale(self):
        data = _raw()
        data["structure"]["scale"] = "mid"
        data["strategies"]["trend"]["entry"]["buy_scale"] = "mid"
        data["strategies"]["trend"]["entry"]["sell_scale"] = "long"
        with self.assertRaisesRegex(ValueError, "策略 D0 的尺度必须与 structure.scale 相同"):
            trend_config_from_mapping(data, today=date(2026, 10, 13))
        data = _raw()
        data["runs"].append({
            "key": "d0_reduce",
            "label": "策略 D0 减仓",
            "strategy": "d0",
            "exhaustion_action": "reduce",
        })
        with self.assertRaisesRegex(ValueError, "不能减仓"):
            trend_config_from_mapping(data, today=date(2026, 10, 13))

    def test_d1_drawdown_is_a_fraction_below_the_high(self):
        data = _raw()
        data["strategies"]["d1"]["exit"]["drawdown"] = 1
        with self.assertRaisesRegex(ValueError, "drawdown"):
            trend_config_from_mapping(data, today=date(2026, 10, 13))
        data = _raw()
        data["structure"]["scale"] = "mid"
        data["strategies"]["trend"]["entry"]["buy_scale"] = "mid"
        data["strategies"]["trend"]["entry"]["sell_scale"] = "long"
        data["strategies"]["d0"]["entry"]["scale"] = "mid"
        with self.assertRaisesRegex(ValueError, "策略 D1 的尺度必须与 structure.scale 相同"):
            trend_config_from_mapping(data, today=date(2026, 10, 13))


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
                self.assertEqual(row["structure"]["param_version"], "structure-params-v6")
                self.assertEqual(row["structure"]["bar_count"], 80)
                self.assertIn("phase", row["structure"]["methods"])
                self.assertTrue(row["structure"]["data_hash"])
                self.assertEqual(row["structure"]["tape_bars"], 40)
                report = Path(row["report_html"])
                text = report.read_text(encoding="utf-8")
                self.assertIn("关键指标", text)
                self.assertIn("取消与跳过", text)
                self.assertIn("衰竭减仓", text)
                self.assertIn("信号使用当根 as-of", text)
                self.assertTrue((report.parent / "decisions.jsonl").is_file())
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

    def test_extra_scale_reads_that_scale_without_changing_the_primary(self):
        series = _make_series(_wave(72))
        snapshot = _snapshot(series)
        short = build_tape(series, snapshot, scale="short", span=14, floor=0.003, extra_scale="mid")
        mid = build_tape(series, snapshot, scale="mid", span=14, floor=0.003)
        primary = build_tape(series, snapshot, scale="short", span=14, floor=0.003)
        self.assertTrue(any(row.structure is not None for row in short))
        self.assertTrue(any(row.exit_structure is not None for row in short))
        for left, other, plain in zip(short, mid, primary):
            self.assertEqual(left.scale, "short")
            self.assertEqual(left.exit_scale, "mid")
            self.assertEqual(left.exit_structure, other.structure)
            self.assertEqual(left.exit_direction_start_index, other.direction_start_index)
            self.assertEqual(left.structure, plain.structure)
            self.assertEqual(left.category, plain.category)
            self.assertEqual(left.direction_start_index, plain.direction_start_index)
            self.assertIsNone(plain.exit_scale)
        with self.assertRaisesRegex(ValueError, "卖出尺度不能和买入尺度相同"):
            build_tape(series, snapshot, scale="short", span=14, floor=0.003, extra_scale="short")


def _series_ohlc():
    closes = [100.0, 110.0, 105.0, 120.0, 130.0, 125.0]
    opens = [100.0, 100.0, 110.0, 105.0, 120.0, 130.0]
    highs = [101.0, 112.0, 106.0, 121.0, 132.0, 131.0]
    lows = [99.0, 108.0, 104.0, 119.0, 128.0, 124.0]
    volumes = [1000.0, 800.0, 900.0, 2000.0, 700.0, 600.0]
    return opens, highs, lows, closes, volumes


class PriceMarketTests(unittest.TestCase):
    def test_ohlc_is_expanded_to_enclose_open_and_close(self):
        frame = pd.DataFrame({
            "open": [10.0, 10.0],
            "high": [10.2, 9.0],
            "low": [9.8, 9.0],
            "close": [10.1, 12.0],
        })
        repaired, count = enclose_ohlc(frame)
        self.assertEqual(count, 1)
        self.assertEqual(repaired.loc[1, "high"], 12.0)
        self.assertEqual(repaired.loc[1, "low"], 9.0)
        self.assertEqual(repaired.loc[0, "high"], 10.2)

    def test_split_share_count_tolerates_float_dust(self):
        self.assertEqual(whole_shares(Decimal("20300.00000000000000000000001")), 20300)
        self.assertEqual(whole_shares(Decimal("99.99999999999999999999999999")), 100)
        with self.assertRaises(ValueError):
            whole_shares(Decimal("10.5"))

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
    bar = ReplayBar(
        time=time,
        open=open_,
        high=high,
        low=low,
        close=close,
        structure=kw.pop("structure", UPTREND),
        category=kw.pop("category", UP_CONTINUE),
        a_t=kw.pop("a_t", 5.0),
        factor=kw.pop("factor", 1.0),
        ema=kw.pop("ema", None),
        dividend=kw.pop("dividend", 0.0),
        split=kw.pop("split", 1.0),
        gap_before=kw.pop("gap_before", False),
        phase_start_index=kw.pop("phase_start_index", None),
        direction_start_index=kw.pop("direction_start_index", None),
        exit_scale=kw.pop("exit_scale", None),
        exit_structure=kw.pop("exit_structure", None),
        exit_direction_start_index=kw.pop("exit_direction_start_index", None),
        up_breaks=tuple(kw.pop("up_breaks", ())),
        down_breaks=tuple(kw.pop("down_breaks", ())),
        support_valid=kw.pop("support_valid", False),
        phase_trigger=kw.pop("phase_trigger", None),
        range_floor=kw.pop("range_floor", None),
    )
    if kw:
        raise TypeError(sorted(kw))
    return bar


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


_KEEP = object()


def _rb(
    index,
    open_,
    high,
    low,
    close,
    *,
    structure=UPTREND,
    category=UP_CONTINUE,
    a_t=1.0,
    ema=None,
    factor=1.0,
    gap_before=False,
    phase_start_index=None,
    direction_start_index=0,
):
    if high < low or high < max(open_, close) or low > min(open_, close):
        raise AssertionError(f"d{index} 的 OHLC 不合法: {(open_, high, low, close)}")
    return ReplayBar(
        time=f"d{index:02d}",
        open=open_,
        high=high,
        low=low,
        close=close,
        structure=structure,
        category=category,
        a_t=a_t,
        factor=factor,
        ema=ema,
        gap_before=gap_before,
        phase_start_index=phase_start_index,
        direction_start_index=direction_start_index,
    )


def _prepared(name, price_axis="log", **entry):
    config = load_trend_config(today=date(2026, 10, 13))
    strategy = config.strategies[name]
    if entry:
        strategy = replace(strategy, entry=replace(strategy.entry, **entry))
    strategy = replace(
        strategy,
        order=replace(strategy.order, slippage_rate=0.0),
        risk=replace(strategy.risk, risk_per_trade=0.02, one_lot_max_risk=0.02),
    )
    market = replace(
        config.markets[MARKET_HK],
        commission_rate=0.0,
        min_commission=0.0,
        platform_fee=0.0,
        stamp_duty_rate=0.0,
        exchange_fee_rate=0.0,
        settlement_rate=0.0,
    )
    tactic = create_entry(name, strategy, price_axis=price_axis)
    return tactic, strategy, market


def _run_plugin(bars, name, price_axis="log", **entry):
    tactic, strategy, market = _prepared(name, price_axis, **entry)
    return run_account(
        bars,
        tactic,
        strategy,
        market,
        lot_size=1,
        capital=100_000.0,
        price_axis=price_axis,
    )


def _breakout_ready(category=UP_CONTINUE, structure=UPTREND, close=101.0, a_t=1.0, signal_a_t=_KEEP):
    bars = [_rb(index, 100, 100, 99, 100, a_t=a_t) for index in range(20)]
    bars.append(_rb(
        20,
        100,
        max(100.0, close),
        99,
        close,
        structure=structure,
        category=category,
        a_t=a_t if signal_a_t is _KEEP else signal_a_t,
    ))
    return bars


def _pullback_week(
    pullback_index=1,
    last_category=UP_CONTINUE,
    last_structure=UPTREND,
    close=101.0,
    ema=100.0,
    a_t=1.0,
    last_a_t=_KEEP,
    last_gap=False,
    direction=0,
):
    """7 根里，默认第 2 根是上涨回调，最后一根用来试信号。序号从 0 起。"""
    bars = []
    for index in range(6):
        pullback = index == pullback_index
        bars.append(_rb(
            index,
            100,
            100,
            90 if pullback else 99,
            99 if pullback else 100,
            category=UP_PULLBACK if pullback else UP_CONTINUE,
            a_t=a_t,
            phase_start_index=pullback_index if pullback else None,
            direction_start_index=direction,
        ))
    bars.append(_rb(
        6,
        100,
        max(100.0, close),
        99,
        close,
        structure=last_structure,
        category=last_category,
        a_t=a_t if last_a_t is _KEEP else last_a_t,
        ema=ema,
        gap_before=last_gap,
        direction_start_index=direction,
    ))
    return bars


class EntryPluginTests(unittest.TestCase):
    def test_registry_refuses_a_plugin_without_a_stop(self):
        config = load_trend_config(today=date(2026, 10, 13))
        self.assertEqual(list_entries(), ["break", "breakout", "d0", "d1", "pullback", "scale", "trend"])
        first = create_entry("breakout", config.strategies["breakout"], price_axis="log")
        second = create_entry("breakout", config.strategies["breakout"], price_axis="log")
        self.assertIsNot(first, second)
        self.assertTrue(first.allows_add)
        self.assertEqual(first.entry.lookback, 20)
        self.assertEqual(first.entry.buffer, 0.2)
        self.assertEqual(first.entry.stop_lookback, 10)
        self.assertEqual(first.entry.stop_buffer, 1.0)
        pullback = create_entry("pullback", config.strategies["pullback"], price_axis="log")
        self.assertFalse(pullback.allows_add)
        self.assertEqual(pullback.entry.lookback, 5)
        self.assertEqual(pullback.entry.break_lookback, 3)
        import analysis.backtest.trend.entries.pullback as pullback_module
        self.assertIn("跌回 EMA10", pullback_module.__doc__)
        self.assertIn("反弹最高价", pullback_module.__doc__)
        with self.assertRaisesRegex(KeyError, "未知入场策略"):
            create_entry("grid", config.strategies["breakout"], price_axis="log")
        with self.assertRaisesRegex(ValueError, "价格轴"):
            create_entry("breakout", config.strategies["breakout"], price_axis="raw")
        with self.assertRaises(TypeError):
            create_entry("pullback", config.strategies["breakout"], price_axis="log")

        class Bare(EntryTactic):
            name = "bare"
            provides_stop = False

        register_entry(Bare)
        try:
            with self.assertRaisesRegex(ValueError, "没有止损"):
                create_entry("bare", config.strategies["breakout"], price_axis="log")
        finally:
            from analysis.backtest.trend.entries.registry import _REGISTRY
            _REGISTRY.pop("bare", None)

    def test_wide_stop_snapshot_pins_the_measured_run(self):
        path = ROOT / "analysis" / "backtest" / "trend-strategy-wide-stop.yaml"
        config = load_trend_config(path, today=date(2026, 10, 14))
        self.assertEqual(config.version, "trend-strategy-v0")
        self.assertEqual(config.window.start, date(2025, 10, 14))
        self.assertEqual(config.window.end, date(2026, 10, 14))
        self.assertEqual([run.key for run in config.runs], ["breakout", "pullback"])
        breakout = config.strategies["breakout"]
        pullback = config.strategies["pullback"]
        self.assertEqual(breakout.entry.lookback, 20)
        self.assertEqual(breakout.entry.stop_lookback, 10)
        self.assertEqual(breakout.entry.stop_buffer, 1.0)
        self.assertEqual(breakout.order.max_stop_distance, 8.0)
        self.assertEqual(breakout.exit.trail_buffer, 5.0)
        self.assertEqual(pullback.entry.stop_buffer, 1.0)
        self.assertEqual(pullback.order.max_stop_distance, 8.0)
        self.assertEqual(pullback.exit.trail_buffer, 5.0)

    def test_breakout_triggers_only_in_uptrend_birth_or_continuation(self):
        tactic, _, _ = _prepared("breakout", "uniform")
        birth = tactic.propose(_breakout_ready(UP_BIRTH))
        continuation = tactic.propose(_breakout_ready(UP_CONTINUE))
        self.assertIsNotNone(birth)
        self.assertIsNotNone(continuation)
        self.assertIn("上涨初生", birth.text)
        self.assertIn("上涨延续", continuation.text)
        blocked = (
            (UPTREND, UP_PULLBACK),
            (UPTREND, UP_EXHAUSTION),
            (DOWNTREND, DOWN_BIRTH),
            (DOWNTREND, UP_CONTINUE),
        )
        for structure, category in blocked:
            with self.subTest(structure=structure, category=category):
                self.assertIsNone(tactic.propose(_breakout_ready(category, structure)))
        ready = _breakout_ready()
        self.assertIsNone(tactic.propose(ready[1:]))
        self.assertIsNone(tactic.propose(_breakout_ready(signal_a_t=None)))

    def test_pullback_needs_a_prior_pullback_in_the_same_uptrend(self):
        tactic, _, _ = _prepared("pullback", "uniform")
        signal = tactic.propose(_pullback_week())
        self.assertIsNotNone(signal)
        self.assertIn("上涨延续", signal.text)
        self.assertIn("起点序号 1", signal.text)
        self.assertAlmostEqual(signal.adjusted_stop, 89.7)
        self.assertIsNone(tactic.propose(_pullback_week(pullback_index=0)))
        self.assertIsNone(tactic.propose(_pullback_week(last_category=UP_BIRTH)))
        self.assertIsNone(tactic.propose(_pullback_week(last_category=UP_EXHAUSTION)))
        self.assertIsNone(tactic.propose(_pullback_week(last_a_t=None)))
        self.assertIsNone(tactic.propose(_pullback_week(ema=None)))
        first = [
            _rb(0, 100, 100, 99, 100),
            _rb(1, 100, 100, 99, 100),
            _rb(2, 100, 100, 99, 100),
            _rb(3, 100, 101, 96, 101, category=UP_PULLBACK, phase_start_index=3, ema=100),
        ]
        self.assertIsNone(tactic.propose(first))

    def test_pullback_can_trigger_while_the_pullback_is_still_underway(self):
        tactic, _, _ = _prepared("pullback", "uniform")
        bars = [
            _rb(0, 100, 100, 99, 100),
            _rb(1, 100, 100, 99, 100),
            _rb(2, 100, 100, 99, 100),
            _rb(3, 100, 100, 96, 99, category=UP_PULLBACK, phase_start_index=3),
            _rb(4, 100, 101, 97, 101, category=UP_PULLBACK, phase_start_index=3, ema=100),
        ]
        signal = tactic.propose(bars)
        self.assertIsNotNone(signal)
        self.assertIn("上涨回调", signal.text)
        self.assertAlmostEqual(signal.adjusted_stop, 95.7)

    def test_pullback_forgets_a_pullback_after_the_uptrend_breaks(self):
        tactic, _, _ = _prepared("pullback", "uniform")
        self.assertIsNotNone(tactic.propose(self._rejoined(False)))
        self.assertIsNone(tactic.propose(self._rejoined(True)))
        self.assertIsNotNone(tactic.propose(self._relabeled(False)))
        self.assertIsNone(tactic.propose(self._relabeled(True)))

    def test_gap_clears_pullback_memory(self):
        tactic, _, _ = _prepared("pullback", "uniform")
        cleared = self._after_gap(remember=False)
        self.assertIsNone(tactic.propose(cleared))
        fresh = self._after_gap(remember=True)
        signal = tactic.propose(fresh)
        self.assertIsNotNone(signal)
        self.assertAlmostEqual(signal.adjusted_stop, 95.7)
        self.assertIn("起点序号 2", signal.text)
        prefix = fresh[:-1]
        self.assertIsNone(tactic.propose(prefix + [_rb(
            6, 100, 101, 99, 101, ema=100, gap_before=True,
        )]))

    def test_pullback_stop_uses_the_latest_segment_start(self):
        tactic, _, _ = _prepared("pullback", "uniform")
        latest = tactic.propose([
            _rb(0, 100, 100, 99, 100),
            _rb(1, 100, 100, 80, 99, category=UP_PULLBACK, phase_start_index=1),
            _rb(2, 100, 100, 99, 100),
            _rb(3, 100, 100, 99, 100),
            _rb(4, 100, 100, 96, 99, category=UP_PULLBACK, phase_start_index=4),
            _rb(5, 100, 100, 98, 100),
            _rb(6, 100, 101, 97, 101, ema=100),
        ])
        self.assertAlmostEqual(latest.adjusted_stop, 95.7)
        self.assertIn("起点序号 4", latest.text)

        earlier_than_window = tactic.propose([
            _rb(0, 100, 100, 80, 99, category=UP_PULLBACK, phase_start_index=0),
            _rb(1, 100, 100, 95, 99, category=UP_PULLBACK, phase_start_index=0),
            _rb(2, 100, 100, 99, 100),
            _rb(3, 100, 100, 99, 100),
            _rb(4, 100, 100, 99, 100),
            _rb(5, 100, 100, 99, 100),
            _rb(6, 100, 101, 99, 101, ema=100),
        ])
        self.assertAlmostEqual(earlier_than_window.adjusted_stop, 79.7)
        self.assertIn("起点序号 0", earlier_than_window.text)

        bounded = tactic.propose([
            _rb(0, 100, 100, 70, 100),
            _rb(1, 100, 100, 90, 99, category=UP_PULLBACK, phase_start_index=1),
            _rb(2, 100, 100, 99, 100),
            _rb(3, 100, 100, 99, 100),
            _rb(4, 100, 100, 99, 100),
            _rb(5, 100, 100, 99, 100),
            _rb(6, 100, 101, 99, 101, ema=100),
        ])
        self.assertAlmostEqual(bounded.adjusted_stop, 89.7)
        self.assertIn("起点序号 1", bounded.text)

    def test_mismatched_pullback_origin_skips_the_bar(self):
        tactic, _, _ = _prepared("pullback", "uniform")
        bars = [
            _rb(0, 100, 100, 99, 100),
            _rb(1, 100, 100, 96, 99, category=UP_PULLBACK, phase_start_index=0),
            _rb(2, 100, 100, 99, 100),
            _rb(3, 100, 100, 99, 100),
            _rb(4, 100, 100, 99, 100),
            _rb(5, 100, 101, 97, 101, ema=100),
        ]
        self.assertIsNone(tactic.propose(bars))

    def test_range_and_pending_states_do_not_open(self):
        breakout, _, _ = _prepared("breakout", "uniform")
        pullback, _, _ = _prepared("pullback", "uniform")
        blocked = (
            (RANGE, ACCUMULATION),
            (RANGE, DISTRIBUTION),
            (RANGE, NEUTRAL),
            (UNDECIDED, PENDING),
            (UPTREND, ACCUMULATION),
            (UPTREND, DISTRIBUTION),
            (UPTREND, NEUTRAL),
            (UPTREND, PENDING),
            (None, INSUFFICIENT),
            (DOWNTREND, DOWN_BIRTH),
        )
        for structure, category in blocked:
            with self.subTest(structure=structure, category=category):
                self.assertIsNone(breakout.propose(_breakout_ready(category, structure)))
                self.assertIsNone(pullback.propose(_pullback_week(last_category=category, last_structure=structure)))

    def test_plugins_read_their_own_parameters(self):
        above = [
            _rb(0, 100, 100, 99, 100),
            _rb(1, 100, 100, 99, 100),
            _rb(2, 100, 100.1, 99, 100.1),
        ]
        loose, _, _ = _prepared("breakout", "uniform", lookback=2, buffer=0, stop_lookback=0, stop_buffer=0.3)
        tight, _, _ = _prepared("breakout", "uniform", lookback=2, buffer=0.2, stop_lookback=0, stop_buffer=0.3)
        self.assertIsNotNone(loose.propose(above))
        self.assertIsNone(tight.propose(above))

        stops = [
            _rb(0, 100, 100, 90, 100),
            _rb(1, 100, 100, 94, 100),
            _rb(2, 100, 100, 94, 100),
            _rb(3, 100, 100, 94, 100),
            _rb(4, 100, 101, 95, 100.5),
        ]
        wide, _, _ = _prepared("breakout", "uniform", lookback=4, buffer=0, stop_lookback=4, stop_buffer=0.3)
        narrow, _, _ = _prepared("breakout", "uniform", lookback=4, buffer=0, stop_lookback=0, stop_buffer=0.3)
        self.assertAlmostEqual(wide.propose(stops).adjusted_stop, 89.7)
        self.assertAlmostEqual(narrow.propose(stops).adjusted_stop, 94.7)

        memory = [
            _rb(0, 100, 100, 99, 100),
            _rb(1, 100, 100, 96, 99, category=UP_PULLBACK, phase_start_index=1),
            _rb(2, 100, 100, 99, 100),
            _rb(3, 100, 101, 99, 101, ema=100),
        ]
        short, _, _ = _prepared("pullback", "uniform", lookback=1, break_lookback=2, stop_buffer=0.3)
        longer, _, _ = _prepared("pullback", "uniform", lookback=2, break_lookback=2, stop_buffer=0.3)
        touched, _, _ = _prepared("pullback", "uniform", lookback=2, break_lookback=2, stop_buffer=0)
        self.assertIsNone(short.propose(memory))
        self.assertAlmostEqual(longer.propose(memory).adjusted_stop, 95.7)
        self.assertAlmostEqual(touched.propose(memory).adjusted_stop, 96)

    def test_signals_use_adjusted_prices(self):
        tactic, _, _ = _prepared(
            "breakout", "uniform", lookback=2, buffer=0.2, stop_lookback=1, stop_buffer=0.3,
        )
        signal = tactic.propose([
            _rb(0, 50, 50, 40, 50, factor=2),
            _rb(1, 50, 50, 40, 50, factor=2),
            _rb(2, 50, 50.2, 40, 50.2, factor=2),
        ])
        self.assertIsNotNone(signal)
        self.assertAlmostEqual(signal.adjusted_stop, 79.7)

    def test_log_axis_uses_a_multiplicative_buffer(self):
        a_t = 0.02
        threshold = 100 * math.exp(0.2 * a_t)
        additive = 100 + 0.2 * a_t
        self.assertGreater(100.2, additive)
        self.assertLess(100.2, threshold)
        self.assertGreater(100.41, threshold)
        tactic, _, _ = _prepared("breakout", "log")
        self.assertIsNone(tactic.propose(self._breakout_close(100.2, a_t)))
        self.assertIsNone(tactic.propose(self._breakout_close(threshold, a_t)))
        prefix = self._breakout_close(100.41, a_t)
        signal = tactic.propose(prefix)
        stop = 99 * math.exp(-0.3 * a_t)
        self.assertAlmostEqual(signal.adjusted_stop, stop)
        self.assertNotAlmostEqual(signal.adjusted_stop, 99 - 0.3 * a_t, places=4)
        self.assertIn("乘以 e^", signal.text)
        self.assertIn("策略 A", signal.text)
        filled = prefix + [_rb(21, 100.41, 100.8, 100.2, 100.41, a_t=a_t)]
        result = _run_plugin(filled, "breakout", "log")
        entry = _by(result, "入场")
        self.assertEqual(len(entry), 1)
        self.assertEqual(entry[0].index, 21)
        self.assertEqual(_by(result, "取消"), [])
        self.assertAlmostEqual(entry[0].adjusted_stop, stop)
        self.assertIn("乘以 e^", entry[0].text)
        self.assertEqual(result.equity[20].shares, 0)
        self.assertGreater(result.equity[21].shares, 0)

        pullback, _, _ = _prepared("pullback", "log")
        blocked = self._pullback_axis(a_t, math.log(110) + 0.01)
        self.assertIsNone(pullback.propose(blocked))
        b_prefix = self._pullback_axis(a_t, math.log(110) - 0.01)
        b_stop = 105 * math.exp(-0.3 * a_t)
        b_signal = pullback.propose(b_prefix)
        self.assertAlmostEqual(b_signal.adjusted_stop, b_stop)
        self.assertNotAlmostEqual(b_signal.adjusted_stop, 105 - 0.3 * a_t, places=3)
        self.assertIn("乘以 e^", b_signal.text)
        b_filled = b_prefix + [_rb(7, 110, 111, 109, 110, a_t=a_t)]
        b_result = _run_plugin(b_filled, "pullback", "log")
        b_entry = _by(b_result, "入场")[0]
        self.assertEqual(b_entry.index, 7)
        self.assertAlmostEqual(b_entry.adjusted_stop, b_stop)
        self.assertGreater(b_entry.shares_after, 0)

    def test_uniform_axis_uses_an_additive_buffer(self):
        a_t = 5.0
        multiplicative = 100 * math.exp(0.2 * a_t)
        self.assertLess(100.5, 101)
        self.assertGreater(102, 101)
        self.assertLess(102, multiplicative)
        tactic, _, _ = _prepared("breakout", "uniform")
        self.assertIsNone(tactic.propose(self._breakout_close(100.5, a_t)))
        self.assertIsNone(tactic.propose(self._breakout_close(101, a_t)))
        prefix = self._breakout_close(102, a_t)
        signal = tactic.propose(prefix)
        self.assertAlmostEqual(signal.adjusted_stop, 97.5)
        self.assertNotAlmostEqual(signal.adjusted_stop, 99 * math.exp(-0.3 * a_t), places=1)
        self.assertIn("加上", signal.text)
        self.assertIn("减去", signal.text)
        filled = prefix + [_rb(21, 102, 103, 100, 102, a_t=a_t)]
        result = _run_plugin(filled, "breakout", "uniform")
        entry = _by(result, "入场")[0]
        self.assertEqual(entry.index, 21)
        self.assertAlmostEqual(entry.adjusted_stop, 97.5)
        self.assertEqual(entry.shares_after, 444)
        self.assertEqual(result.equity[20].shares, 0)
        self.assertEqual(result.equity[21].shares, 444)

        pullback, _, _ = _prepared("pullback", "uniform")
        self.assertIsNone(pullback.propose(self._pullback_axis(2, 110, close=110)))
        self.assertIsNone(pullback.propose(self._pullback_axis(2, 100, close=108)))
        b_prefix = self._pullback_axis(2, 109)
        b_signal = pullback.propose(b_prefix)
        self.assertAlmostEqual(b_signal.adjusted_stop, 104.4)
        self.assertNotAlmostEqual(b_signal.adjusted_stop, 105 * math.exp(-0.6), places=1)
        self.assertIn("减去", b_signal.text)
        b_result = _run_plugin(b_prefix + [_rb(7, 110, 111, 109, 110, a_t=2)], "pullback", "uniform")
        b_entry = _by(b_result, "入场")[0]
        self.assertEqual(b_entry.index, 7)
        self.assertAlmostEqual(b_entry.adjusted_stop, 104.4)
        self.assertGreater(b_entry.shares_after, 0)

    def test_accounts_match_whether_run_alone_or_together(self):
        bars = [
            _rb(0, 100, 100, 99, 100, a_t=2),
            _rb(1, 100, 100, 99, 100, a_t=2),
            _rb(2, 100, 100, 99, 100, a_t=2),
            _rb(3, 100, 103, 99, 102, a_t=2),
            _rb(4, 102, 103, 101, 102, a_t=2),
            _rb(5, 102, 104, 100.5, 104, a_t=2, category=UP_PULLBACK, phase_start_index=5, ema=100),
            _rb(6, 104, 105, 100.5, 104.5, a_t=2, category=UP_PULLBACK, phase_start_index=5, ema=100),
            _rb(7, 104.5, 106, 103, 105, a_t=2, category=UP_PULLBACK, phase_start_index=5, ema=100),
        ]
        solo_a = self._account(bars, "breakout")
        solo_b = self._account(bars, "pullback")
        pair_a, pair_b = self._account(bars, "breakout"), self._account(bars, "pullback")
        again_a = self._account(bars, "breakout")
        again_b = self._account(bars, "pullback")
        self.assertEqual(_by(solo_a, "入场")[0].index, 4)
        self.assertEqual(_by(solo_b, "入场")[0].index, 7)
        self.assertAlmostEqual(_by(solo_a, "入场")[0].adjusted_stop, 98.4)
        self.assertAlmostEqual(_by(solo_b, "入场")[0].adjusted_stop, 99.9)
        self.assertEqual(_by(solo_a, "取消"), [])
        self.assertEqual(_by(solo_b, "取消"), [])
        self.assertEqual(_by(solo_a, "止损"), [])
        self.assertEqual(_by(solo_b, "止损"), [])
        self.assertNotEqual(solo_a, solo_b)
        self.assertEqual(pair_a, solo_a)
        self.assertEqual(pair_b, solo_b)
        self.assertEqual(again_a, solo_a)
        self.assertEqual(again_b, solo_b)
        self.assertEqual(pair_a.final_shares, solo_a.final_shares)
        self.assertEqual(pair_b.final_cash, solo_b.final_cash)

    def _account(self, bars, name):
        if name == "breakout":
            return _run_plugin(
                bars, name, "uniform", lookback=3, buffer=0.2, stop_lookback=1, stop_buffer=0.3,
            )
        return _run_plugin(bars, name, "uniform", lookback=3, break_lookback=2, stop_buffer=0.3)

    def _breakout_close(self, close, a_t):
        bars = [_rb(index, 100, 100, 99, 100, a_t=a_t) for index in range(20)]
        bars.append(_rb(20, 100, max(100.0, close), 99, close, a_t=a_t))
        return bars

    def _pullback_axis(self, a_t, ema, close=110.0):
        bars = []
        for index in range(6):
            pullback = index == 1
            bars.append(_rb(
                index,
                108,
                108,
                105 if pullback else 106,
                107 if pullback else 108,
                a_t=a_t,
                category=UP_PULLBACK if pullback else UP_CONTINUE,
                phase_start_index=1 if pullback else None,
            ))
        bars.append(_rb(6, 108, max(108.0, close), 106, close, a_t=a_t, ema=ema))
        return bars

    def _rejoined(self, broken):
        bars = []
        for index in range(7):
            if index <= 1:
                category, low, close, direction, structure, phase = UP_PULLBACK, 80 if index == 0 else 90, 99, 0, UPTREND, 0
            elif index == 6:
                category, low, close, direction, structure, phase = UP_CONTINUE, 99, 101, 3 if broken else 0, UPTREND, None
            elif index == 2 and broken:
                category, low, close, direction, structure, phase = NEUTRAL, 99, 100, 2, RANGE, None
            else:
                category = UP_BIRTH if index == 3 and broken else UP_CONTINUE
                low, close, direction, structure, phase = 99, 100, 3 if broken else 0, UPTREND, None
            bars.append(_rb(
                index,
                100,
                101 if index == 6 else 100,
                low,
                close,
                structure=structure,
                category=category,
                phase_start_index=phase,
                direction_start_index=direction,
                ema=100 if index == 6 else None,
            ))
        return bars

    def _relabeled(self, changed):
        bars = []
        for index in range(7):
            pullback = index == 1
            direction = 0 if index <= 1 or not changed else 7
            bars.append(_rb(
                index,
                100,
                101 if index == 6 else 100,
                90 if pullback else 99,
                101 if index == 6 else (99 if pullback else 100),
                category=UP_PULLBACK if pullback else UP_CONTINUE,
                phase_start_index=1 if pullback else None,
                direction_start_index=direction,
                ema=100 if index == 6 else None,
            ))
        return bars

    def _after_gap(self, remember):
        if not remember:
            return [
                _rb(0, 100, 100, 80, 99, category=UP_PULLBACK, phase_start_index=0),
                _rb(1, 100, 100, 90, 99, category=UP_PULLBACK, phase_start_index=0),
                _rb(2, 100, 100, 99, 100, gap_before=True),
                _rb(3, 100, 100, 99, 100),
                _rb(4, 100, 100, 99, 100),
                _rb(5, 100, 100, 99, 100),
                _rb(6, 100, 101, 99, 101, ema=100),
            ]
        return [
            _rb(0, 100, 100, 70, 100),
            _rb(1, 100, 100, 70, 100),
            _rb(2, 100, 100, 96, 99, category=UP_PULLBACK, phase_start_index=2, gap_before=True),
            _rb(3, 100, 100, 97, 99, category=UP_PULLBACK, phase_start_index=2),
            _rb(4, 100, 100, 99, 100),
            _rb(5, 100, 100, 99, 100),
            _rb(6, 100, 101, 99, 101, ema=100),
        ]


def _cbar(index, open_, close, *, structure, direction_start, exit_structure, exit_start, low=None, a_t=1.0, exit_scale="mid"):
    high = max(open_, close)
    if low is None:
        low = min(open_, close)
    return _bar(
        f"d{index}",
        open_,
        high,
        low,
        close,
        structure=structure,
        category=UP_CONTINUE if structure == UPTREND else NEUTRAL,
        a_t=a_t,
        direction_start_index=direction_start,
        exit_scale=exit_scale,
        exit_structure=exit_structure,
        exit_direction_start_index=exit_start,
    )


class TrendOnlyTests(unittest.TestCase):
    def test_buys_the_day_after_a_short_uptrend_and_sells_on_a_new_mid_downtrend(self):
        bars = [
            _cbar(0, 100, 100, structure=RANGE, direction_start=0, exit_structure=RANGE, exit_start=0),
            _cbar(1, 100, 100, structure=UPTREND, direction_start=1, exit_structure=RANGE, exit_start=0),
            _cbar(2, 100, 80, structure=UPTREND, direction_start=1, exit_structure=RANGE, exit_start=0, low=70),
            _cbar(3, 80, 80, structure=RANGE, direction_start=3, exit_structure=RANGE, exit_start=0),
            _cbar(4, 80, 80, structure=UPTREND, direction_start=4, exit_structure=DOWNTREND, exit_start=4),
            _cbar(5, 110, 110, structure=UPTREND, direction_start=4, exit_structure=DOWNTREND, exit_start=4),
            _cbar(6, 110, 110, structure=UPTREND, direction_start=6, exit_structure=DOWNTREND, exit_start=4),
            _cbar(7, 110, 110, structure=UPTREND, direction_start=6, exit_structure=RANGE, exit_start=7),
            _cbar(8, 100, 100, structure=UPTREND, direction_start=8, exit_structure=RANGE, exit_start=7),
            _cbar(9, 100, 100, structure=UPTREND, direction_start=8, exit_structure=RANGE, exit_start=7),
        ]
        result = _run_plugin(bars, "trend", "uniform")
        entries = _by(result, "入场")
        exits = _by(result, "趋势退出")
        self.assertEqual([item.time for item in entries], ["d2", "d9"])
        self.assertEqual(entries[0].shares_after, 1000)
        self.assertAlmostEqual(entries[0].planned_risk, 1.0)
        self.assertIsNone(entries[0].adjusted_stop)
        self.assertIn("策略 C", entries[0].text)
        self.assertIn("下一根开盘买入", entries[0].text)
        self.assertIn("没有价格止损", entries[0].text)
        self.assertEqual(_by(result, "止损"), [])
        self.assertEqual([item.time for item in exits], ["d5"])
        self.assertEqual(exits[0].exit_rule, "中期下降")
        self.assertEqual(exits[0].shares_after, 0)
        self.assertIn("不开反向仓", exits[0].text)
        self.assertEqual(result.blocked, (("中期下降，不开仓", 1),))
        self.assertEqual(result.equity[2].effective_stop, None)
        self.assertGreater(result.final_shares, 0)

    def test_a_gap_cancels_and_the_same_uptrend_is_not_retried(self):
        appeared = _cbar(0, 100, 100, structure=UPTREND, direction_start=0, exit_structure=RANGE, exit_start=0, a_t=10)
        cancelled = _run_plugin([
            appeared,
            _cbar(1, 106, 106, structure=UPTREND, direction_start=0, exit_structure=RANGE, exit_start=0, a_t=10),
        ], "trend", "uniform")
        self.assertEqual(_by(cancelled, "入场"), [])
        self.assertEqual(_by(cancelled, "取消")[0].reason, "跳空")
        filled = _run_plugin([
            appeared,
            _cbar(1, 104, 104, structure=UPTREND, direction_start=0, exit_structure=RANGE, exit_start=0, a_t=10),
        ], "trend", "uniform")
        self.assertEqual(len(_by(filled, "入场")), 1)

    def test_the_last_bar_and_warmup_do_not_leave_an_order(self):
        alone = _run_plugin([
            _cbar(0, 100, 100, structure=UPTREND, direction_start=0, exit_structure=RANGE, exit_start=0),
        ], "trend", "uniform")
        self.assertEqual(_by(alone, "取消")[0].reason, "没有下一根")
        self.assertEqual(alone.final_shares, 0)

        tactic, strategy, market = _prepared("trend", "uniform")
        missed = run_account(
            [
                _cbar(0, 100, 100, structure=UPTREND, direction_start=0, exit_structure=RANGE, exit_start=0),
                _cbar(1, 100, 100, structure=UPTREND, direction_start=0, exit_structure=RANGE, exit_start=0),
            ],
            tactic,
            strategy,
            market,
            lot_size=1,
            capital=100_000.0,
            price_axis="uniform",
            trade_from=1,
        )
        self.assertEqual(_by(missed, "入场"), [])
        self.assertEqual(missed.final_shares, 0)

    def test_exposure_cap_and_cash_cancel_without_a_stop(self):
        bars = [
            _cbar(0, 100, 100, structure=UPTREND, direction_start=0, exit_structure=RANGE, exit_start=0),
            _cbar(1, 100, 100, structure=UPTREND, direction_start=0, exit_structure=RANGE, exit_start=0),
        ]
        tactic, strategy, market = _prepared("trend", "uniform")
        half = run_account(
            bars, tactic, replace(strategy, risk=replace(strategy.risk, max_gross_exposure=0.5)),
            market, lot_size=1, capital=100_000.0, price_axis="uniform",
        )
        self.assertEqual(_by(half, "入场")[0].shares_after, 500)
        self.assertAlmostEqual(_by(half, "入场")[0].planned_risk, 0.5)

        cash = run_account(
            bars, tactic, strategy, market, lot_size=100, capital=50.0, price_axis="uniform",
        )
        self.assertEqual(_by(cash, "取消")[0].reason, "现金")
        capped = run_account(
            bars,
            tactic,
            replace(strategy, risk=replace(strategy.risk, max_gross_exposure=0.01)),
            market,
            lot_size=100,
            capital=100_000.0,
            price_axis="uniform",
        )
        self.assertEqual(_by(capped, "取消")[0].reason, "占用上限")

    def test_mid_downtrend_blocks_the_entry_and_the_sell_scale_comes_from_the_parameters(self):
        blocked = [
            _cbar(0, 100, 100, structure=UPTREND, direction_start=0, exit_structure=DOWNTREND, exit_start=0),
            _cbar(1, 100, 100, structure=UPTREND, direction_start=0, exit_structure=DOWNTREND, exit_start=0),
        ]
        result = _run_plugin(blocked, "trend", "uniform")
        self.assertEqual(_by(result, "入场"), [])
        self.assertEqual(result.blocked, (("中期下降，不开仓", 1),))

        tactic, strategy, _ = _prepared("trend", "uniform")
        wrong_scale = [
            _cbar(0, 100, 100, structure=UPTREND, direction_start=0, exit_structure=RANGE, exit_start=0, exit_scale="long"),
        ]
        self.assertIsNone(tactic.propose(wrong_scale))
        self.assertEqual(tactic.skip_reason(wrong_scale), "缺少卖出尺度")
        renamed = create_entry(
            "trend",
            replace(strategy, entry=replace(strategy.entry, sell_scale="long")),
            price_axis="uniform",
        )
        self.assertIsNotNone(renamed.propose(wrong_scale))
        self.assertIn("长期", renamed.propose(wrong_scale).text)

    def test_price_stops_and_the_short_trend_ending_do_not_change_strategy_a(self):
        held, _ = _account([
            _calm("d0"),
            _calm("d1", exit_structure=DOWNTREND, exit_direction_start_index=1, exit_scale="mid"),
            _calm("d2"),
        ], {0: _sig()})
        self.assertEqual(_by(held, "趋势退出"), [])
        self.assertGreater(held.final_shares, 0)


def _dbar(index, open_, close, *, structure, direction_start, a_t=1.0, high=None, low=None, exit_structure=None, exit_start=None):
    return _bar(
        f"d{index}",
        open_,
        max(open_, close) if high is None else high,
        min(open_, close) if low is None else low,
        close,
        structure=structure,
        category=UP_CONTINUE if structure == UPTREND else NEUTRAL,
        a_t=a_t,
        direction_start_index=direction_start,
        exit_scale="mid",
        exit_structure=exit_structure,
        exit_direction_start_index=exit_start,
    )


class D0Tests(unittest.TestCase):
    def test_buys_the_day_after_a_short_uptrend_and_sells_the_day_after_a_short_downtrend(self):
        bars = [
            _dbar(0, 100, 100, structure=RANGE, direction_start=0),
            _dbar(1, 100, 100, structure=UPTREND, direction_start=1),
            _dbar(2, 100, 101, structure=UPTREND, direction_start=1),
            _dbar(3, 101, 101, structure=RANGE, direction_start=3),
            _dbar(4, 101, 102, structure=UPTREND, direction_start=4, exit_structure=DOWNTREND, exit_start=4),
            _dbar(5, 102, 90, structure=DOWNTREND, direction_start=5),
            _dbar(6, 90, 90, structure=DOWNTREND, direction_start=5),
            _dbar(7, 90, 90, structure=UPTREND, direction_start=7),
            _dbar(8, 90, 91, structure=UPTREND, direction_start=7),
        ]
        result = _run_plugin(bars, "d0", "uniform")
        entries = _by(result, "入场")
        exits = _by(result, "趋势退出")
        self.assertEqual([item.time for item in entries], ["d2", "d8"])
        self.assertEqual(entries[0].shares_after, 1000)
        self.assertAlmostEqual(entries[0].planned_risk, 1.0)
        self.assertIsNone(entries[0].adjusted_stop)
        self.assertIn("策略 D0", entries[0].text)
        self.assertIn("下一根开盘买入", entries[0].text)
        self.assertIn("短期", entries[0].text)
        self.assertEqual([item.time for item in exits], ["d6"])
        self.assertEqual(exits[0].exit_rule, "短期下降")
        self.assertEqual(exits[0].shares_after, 0)
        self.assertIn("不开反向仓", exits[0].text)
        self.assertEqual(result.blocked, ())
        self.assertGreater(result.final_shares, 0)

    def test_a_gap_cancels_and_the_same_uptrend_is_not_retried(self):
        appeared = _dbar(0, 100, 100, structure=UPTREND, direction_start=0, a_t=10)
        cancelled = _run_plugin([
            appeared,
            _dbar(1, 106, 106, structure=UPTREND, direction_start=0, a_t=10),
        ], "d0", "uniform")
        self.assertEqual(_by(cancelled, "入场"), [])
        self.assertEqual(_by(cancelled, "取消")[0].reason, "跳空")
        filled = _run_plugin([
            appeared,
            _dbar(1, 104, 104, structure=UPTREND, direction_start=0, a_t=10),
        ], "d0", "uniform")
        self.assertEqual(len(_by(filled, "入场")), 1)

    def test_mid_downtrend_does_not_sell_or_block_the_entry(self):
        held = _run_plugin([
            _dbar(0, 100, 100, structure=UPTREND, direction_start=0, exit_structure=DOWNTREND, exit_start=0),
            _dbar(1, 100, 100, structure=UPTREND, direction_start=0, exit_structure=DOWNTREND, exit_start=0),
            _dbar(2, 100, 100, structure=UPTREND, direction_start=0, exit_structure=DOWNTREND, exit_start=2),
        ], "d0", "uniform")
        self.assertEqual(len(_by(held, "入场")), 1)
        self.assertEqual(_by(held, "趋势退出"), [])
        self.assertEqual(held.blocked, ())
        self.assertGreater(held.final_shares, 0)


class D1Tests(unittest.TestCase):
    def test_stop_is_fifteen_percent_under_the_holding_high(self):
        bars = [
            _dbar(0, 100, 100, structure=UPTREND, direction_start=0),
            _dbar(1, 100, 100, structure=UPTREND, direction_start=0),
            _dbar(2, 150, 180, structure=UPTREND, direction_start=0, high=200, low=150),
            _dbar(3, 180, 175, structure=UPTREND, direction_start=0, high=180, low=171),
            _dbar(4, 175, 172, structure=UPTREND, direction_start=0, high=175, low=170),
        ]
        result = _run_plugin(bars, "d1", "uniform")
        entry = _by(result, "入场")[0]
        self.assertEqual(entry.time, "d1")
        self.assertAlmostEqual(entry.adjusted_stop, 85)
        self.assertIn("前高", entry.text)
        self.assertIn("15%", entry.text)
        stopped = _by(result, "止损")
        self.assertEqual([item.time for item in stopped], ["d4"])
        self.assertAlmostEqual(stopped[0].price, 170)
        self.assertEqual(stopped[0].exit_rule, "前高回撤")
        self.assertEqual(_by(result, "趋势退出"), [])
        self.assertEqual(result.final_shares, 0)

    def test_the_same_bars_high_does_not_stop_that_bars_low(self):
        bars = [
            _dbar(0, 100, 100, structure=UPTREND, direction_start=0),
            _dbar(1, 100, 190, structure=UPTREND, direction_start=0, high=200, low=90),
            _dbar(2, 190, 165, structure=UPTREND, direction_start=0, high=190, low=160),
        ]
        result = _run_plugin(bars, "d1", "uniform")
        stopped = _by(result, "止损")
        self.assertEqual([item.time for item in stopped], ["d2"])
        self.assertAlmostEqual(stopped[0].price, 170)

    def test_short_downtrend_still_exits_while_the_drawdown_is_inside_fifteen_percent(self):
        bars = [
            _dbar(0, 100, 100, structure=UPTREND, direction_start=0),
            _dbar(1, 100, 100, structure=UPTREND, direction_start=0),
            _dbar(2, 100, 100, structure=DOWNTREND, direction_start=2),
            _dbar(3, 96, 96, structure=DOWNTREND, direction_start=2, high=96, low=96),
        ]
        result = _run_plugin(bars, "d1", "uniform")
        exits = _by(result, "趋势退出")
        self.assertEqual([item.time for item in exits], ["d3"])
        self.assertEqual(exits[0].exit_rule, "短期下降")
        self.assertAlmostEqual(exits[0].price, 96)
        self.assertEqual(_by(result, "止损"), [])

    def test_a_gap_through_the_stop_runs_before_the_trend_exit(self):
        bars = [
            _dbar(0, 100, 100, structure=UPTREND, direction_start=0),
            _dbar(1, 100, 100, structure=UPTREND, direction_start=0),
            _dbar(2, 100, 100, structure=DOWNTREND, direction_start=2),
            _dbar(3, 80, 80, structure=DOWNTREND, direction_start=2),
        ]
        result = _run_plugin(bars, "d1", "uniform")
        stopped = _by(result, "止损")
        self.assertEqual([item.time for item in stopped], ["d3"])
        self.assertEqual(stopped[0].exit_rule, "前高回撤")
        self.assertEqual(stopped[0].reason, "跳空穿过止损")
        self.assertAlmostEqual(stopped[0].price, 80)
        self.assertEqual(_by(result, "趋势退出"), [])

    def test_the_same_uptrend_is_not_bought_again_after_the_stop(self):
        bars = [
            _dbar(0, 100, 100, structure=UPTREND, direction_start=0),
            _dbar(1, 100, 100, structure=UPTREND, direction_start=0, high=100, low=80),
            _dbar(2, 90, 90, structure=UPTREND, direction_start=0),
            _dbar(3, 90, 90, structure=UPTREND, direction_start=0),
        ]
        result = _run_plugin(bars, "d1", "uniform")
        self.assertEqual([item.time for item in _by(result, "入场")], ["d1"])
        self.assertEqual([item.time for item in _by(result, "止损")], ["d1"])
        self.assertEqual(result.final_shares, 0)


class ReportTests(unittest.TestCase):
    def test_dividend_income_is_cash_and_is_not_also_left_in_the_price(self):
        market = self._flat_market()
        # 100 股买在 100，除息 2 之后不复权收盘 98。复权价仍是 100。
        # 总权益应回到 10000：价格少了 200，股息收入补回 200。
        dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
        hold = simulate_buy_hold(
            dates,
            [100, 100, 100],
            [100, 100, 100],
            [1, 1, 100 / 98],
            [1, 1, 1],
            [0, 0, 2],
            lot_size=1,
            capital=10_000,
            market=market,
            withholding=0,
        )
        self.assertEqual(hold.final_shares, 100)
        self.assertAlmostEqual(hold.dividend_income, 200)
        self.assertAlmostEqual(hold.final_equity, 10_000)
        self.assertAlmostEqual(hold.total_return, 0)
        self.assertAlmostEqual(split_adjusted_price(100, 100 / 98, 1), 98)

        taxed = simulate_buy_hold(
            dates, [100, 100, 100], [100, 100, 100], [1, 1, 100 / 98], [1, 1, 1], [0, 0, 2],
            lot_size=1, capital=10_000, market=market, withholding=0.3,
        )
        self.assertAlmostEqual(taxed.dividend_income, 140)
        self.assertAlmostEqual(taxed.final_equity, 9_940)

    def test_split_stays_inside_the_adjusted_price(self):
        market = self._flat_market()
        hold = simulate_buy_hold(
            ["2024-01-02", "2024-01-03"],
            [200, 200],
            [200, 200],
            [1, 2],
            [1, 2],
            [0, 0],
            lot_size=1,
            capital=200,
            market=market,
            withholding=0,
        )
        self.assertEqual(hold.final_shares, 2)
        self.assertAlmostEqual(hold.final_equity, 200)
        self.assertAlmostEqual(hold.dividend_income, 0)

        dca = simulate_even_dca(
            ["2024-01-02", "2024-01-03", "2024-01-04"],
            [100, 100, 100],
            [100, 100, 100],
            [1, 1, 100 / 98],
            [1, 1, 1],
            [0, 0, 2],
            lot_size=1,
            capital=250,
            market=market,
            withholding=0,
        )
        self.assertGreater(dca.dividend_income, 0)
        self.assertGreater(dca.final_shares, 0)

    def _flat_market(self):
        config = load_trend_config(today=date(2026, 10, 13))
        return replace(
            config.markets[MARKET_HK],
            commission_rate=0,
            min_commission=0,
            platform_fee=0,
            stamp_duty_rate=0,
            exchange_fee_rate=0,
            settlement_rate=0,
        )

    def test_restores_split_adjusted_yahoo_prices_to_the_traded_price(self):
        opens, highs, lows, closes, dividends = restore_traded_prices(
            [100, 100, 50],
            [110, 110, 55],
            [90, 90, 45],
            [100, 100, 50],
            [1, 0, 0],
            [0, 2, 0],
        )
        self.assertEqual(closes, [200, 100, 50])
        self.assertEqual(opens, [200, 100, 50])
        self.assertEqual(highs, [220, 110, 55])
        self.assertEqual(lows, [180, 90, 45])
        self.assertEqual(dividends, [2, 0, 0])

    def test_case_report_includes_basis_position_stop_and_risk(self):
        prefix = [_rb(index, 100, 100, 99, 100, a_t=1) for index in range(3)]
        prefix.append(_rb(3, 100, 102, 99, 102, a_t=1))
        prefix.append(_rb(4, 102, 103, 100, 102, a_t=1))
        result = _run_plugin(prefix, "breakout", "uniform", lookback=3, buffer=0.2, stop_lookback=1, stop_buffer=0.3)
        entry = _by(result, "入场")[0]
        self.assertIn("策略 A", entry.text)
        summary = audit_summary(result, exhaustion_enabled=True)
        config = load_trend_config(today=date(2026, 10, 13))
        case = _case_config(config, config.symbols[0], config.runs[0])
        row = {
            "symbol": "00700",
            "variant_label": "策略 A",
            "coverage": "合成",
            "bar_source": "synthetic",
            "metrics": {
                "dca_start_price": 100,
                "dca_n_lots": 1,
                "dca_n_trading_days": 5,
                "dca_interval": 1,
                "dca_n_lots_bought": 1,
                "dca_n_buy_days": 1,
                "dca_return": 0,
                "excess_vs_dca": 0,
                "max_consecutive_losses": summary["max_consecutive_losses"],
            },
            "split": {},
            "gaps": [],
            "extremes": {},
            "round_trips": summary["round_trips"],
            "trades": [{
                "timestamp": entry.time,
                "side": "buy",
                "action": "入场",
                "tactic": "breakout",
                "price": entry.price,
                "shares": entry.shares_after,
                "position_fraction": entry.position_fraction,
                "planned_risk": entry.planned_risk,
                "stop_price": entry.raw_stop,
                "basis": entry.text,
                "fee": 0,
                "equity_after": entry.equity_after,
            }],
            "audit": summary,
            "preamble": ["信号使用当根 as-of，成交在下一根开盘；盘中止损使用上一根收盘已经生效的止损"],
        }
        text = render_html_case(row, case, {})
        self.assertIn("关键指标", text)
        self.assertIn("仓位占比", text)
        self.assertIn("计划风险", text)
        self.assertIn("生效止损", text)
        self.assertIn("策略 A", text)
        self.assertIn("取消与跳过", text)
        self.assertIn("衰竭减仓", text)
        self.assertIn("最大连续亏损次数", text)
        self.assertIn("按一手规则成交", text)

    def test_summary_lines_up_runs_without_adding_accounts(self):
        bars = _bars(80)
        data = _raw()
        data["universe"]["symbols"] = [
            {"code": "00700", "name": "腾讯控股", "lot_size": 100},
            {"code": "NVDA", "name": "英伟达"},
        ]
        data["window"]["start"] = bars["datetime"].iloc[40].date().isoformat()
        data["window"]["end"] = bars["datetime"].iloc[-1].date().isoformat()
        data["window"]["warmup_bars"] = 30
        with tempfile.TemporaryDirectory() as tmp:
            data["output_dir"] = tmp
            config = trend_config_from_mapping(data, source="test")
            payload = run_trend_backtest(
                config,
                frames={"00700": bars, "NVDA": bars},
                load_benchmarks=False,
            )
            out = Path(payload["output_dir"])
            folders = sorted(path.name for path in out.iterdir() if path.is_dir())
            self.assertEqual(folders, [
                "00700_breakout",
                "00700_breakout_no_reduce",
                "00700_d0",
                "00700_d1",
                "00700_pullback",
                "00700_pullback_no_reduce",
                "00700_trend",
                "NVDA_breakout",
                "NVDA_breakout_no_reduce",
                "NVDA_d0",
                "NVDA_d1",
                "NVDA_pullback",
                "NVDA_pullback_no_reduce",
                "NVDA_trend",
            ])
            summary = (out / "summary.md").read_text(encoding="utf-8")
            index = (out / "index.html").read_text(encoding="utf-8")
            self.assertIn("账户对照", summary)
            self.assertIn("账户对照", index)
            self.assertNotIn("合计", summary)
            self.assertIn("00700", summary)
            self.assertIn("NVDA", summary)
            self.assertIn("HKD", summary)
            self.assertIn("USD", summary)
            self.assertIn("样本外", summary)
            self.assertIn("趋势回测汇总", index)
            self.assertIn('href="00700_breakout/report.html"', index)
            self.assertIn('href="00700_pullback/report.html"', index)
            self.assertIn('href="NVDA_breakout/report.html"', index)
            self.assertIn('href="NVDA_pullback_no_reduce/report.html"', index)
            self.assertIn('href="00700_trend/report.html"', index)
            trend_case = (out / "00700_trend" / "report.html").read_text(encoding="utf-8")
            self.assertIn("不设价格止损", trend_case)
            self.assertIn("中期", trend_case)
            case = (out / "00700_breakout" / "report.html").read_text(encoding="utf-8")
            self.assertIn('href="../index.html"', case)
            self.assertIn("返回汇总", case)

    def test_bar_that_does_not_enclose_is_repaired_and_kept(self):
        bars = _bars(80)
        bad = bars.copy()
        bad.loc[bad.index[50], "high"] = 1.0
        data = _raw()
        data["universe"]["symbols"] = [
            {"code": "00700", "name": "腾讯控股", "lot_size": 100},
            {"code": "00004", "name": "九龙仓集团", "lot_size": 100},
        ]
        data["window"]["start"] = bars["datetime"].iloc[40].date().isoformat()
        data["window"]["end"] = bars["datetime"].iloc[-1].date().isoformat()
        data["window"]["warmup_bars"] = 30
        data["runs"] = [data["runs"][0]]
        with tempfile.TemporaryDirectory() as tmp:
            data["output_dir"] = tmp
            config = trend_config_from_mapping(data, source="test")
            payload = run_trend_backtest(
                config,
                frames={"00700": bars, "00004": bad},
                load_benchmarks=False,
            )
            self.assertEqual(sorted({row["symbol"] for row in payload["results"]}), ["00004", "00700"])
            self.assertEqual(payload["failures"], [])
            repaired = [row for row in payload["results"] if row["symbol"] == "00004"]
            self.assertTrue(repaired)
            self.assertGreater(repaired[0]["repaired_bars"], 0)
            self.assertTrue(any("修补了" in gap for gap in repaired[0]["gaps"]))
            out = Path(payload["output_dir"])
            self.assertTrue((out / "00004_breakout" / "report.html").is_file())


def _run_scale(bars, price_axis="uniform"):
    config = load_trend_config(
        ROOT / "analysis" / "backtest" / "trend-strategy-scale.yaml",
        today=date(2026, 10, 15),
    )
    strategy = replace(config.strategies["scale"], order=replace(config.strategies["scale"].order, slippage_rate=0.0))
    market = replace(
        config.markets[MARKET_HK],
        commission_rate=0.0,
        min_commission=0.0,
        platform_fee=0.0,
        stamp_duty_rate=0.0,
        exchange_fee_rate=0.0,
        settlement_rate=0.0,
    )
    tactic = create_entry("scale", strategy, price_axis=price_axis)
    return run_account(
        bars, tactic, strategy, market, lot_size=1, capital=100_000.0, price_axis=price_axis,
    )


def _sbar(index, open_, close, *, structure=UPTREND, category=UP_CONTINUE, direction_start=0, a_t=1.0, high=None, low=None, **kw):
    return _bar(
        f"d{index}",
        open_,
        max(open_, close) if high is None else high,
        min(open_, close) if low is None else low,
        close,
        structure=structure,
        category=category,
        a_t=a_t,
        direction_start_index=direction_start,
        **kw,
    )


class ScaleTests(unittest.TestCase):
    def test_config_rejects_trailing_during_birth(self):
        path = ROOT / "analysis" / "backtest" / "trend-strategy-scale.yaml"
        config = load_trend_config(path, today=date(2026, 10, 15))
        scale = config.strategies["scale"]
        self.assertEqual(scale.exit.mode, "scale_book")
        self.assertEqual(scale.exit.trail_buffer, 3.0)
        self.assertEqual(scale.exit.max_adds, 2)
        self.assertEqual(scale.risk.risk_per_trade, 0.005)
        self.assertEqual(scale.risk.max_open_risk, 0.01)
        self.assertEqual(scale.order.min_stop_distance, 1.0)
        self.assertNotIn(UP_BIRTH, scale.exit.trail_states)
        data = load_trend_yaml(path)
        data["strategies"]["scale"]["exit"]["trail_categories"] = ["up_continue", "up_birth"]
        with self.assertRaisesRegex(ValueError, "up_birth"):
            trend_config_from_mapping(data, source="scale-test")

    def test_birth_does_not_raise_the_stop(self):
        result = _run_scale([
            _sbar(0, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=99),
            _sbar(1, 100, 102, category=UP_BIRTH, direction_start=0, high=103, low=99.5),
            _sbar(2, 102, 104, category=UP_BIRTH, direction_start=0, high=105, low=101),
            _sbar(3, 101.2, 104, category=UP_CONTINUE, direction_start=0, high=105, low=101.2),
            _sbar(4, 101.5, 101.5, category=UP_CONTINUE, direction_start=0, high=102, low=101.5),
        ])
        self.assertEqual([item.time for item in _by(result, "入场")], ["d1"])
        self.assertEqual([item.time for item in _by(result, "止损")], ["d4"])
        self.assertEqual(_by(result, "减仓"), [])
        self.assertAlmostEqual(result.equity[1].next_stop, 97.0)
        self.assertAlmostEqual(result.equity[2].next_stop, 97.0)
        self.assertAlmostEqual(result.equity[3].next_stop, 102.0)

    def test_add_waits_until_pullback_ends_and_the_stop_covers_cost(self):
        added = _run_scale([
            _sbar(0, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=98),
            _sbar(1, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=98),
            _sbar(2, 100, 103, category=UP_CONTINUE, direction_start=0, high=104, low=99.5),
            _sbar(3, 103, 102, category=UP_PULLBACK, direction_start=0, high=103.5, low=101.5),
            _sbar(4, 102, 103, category=UP_CONTINUE, direction_start=0, high=104, low=101.5),
            _sbar(5, 103, 103, category=UP_CONTINUE, direction_start=0, high=104, low=102),
        ])
        self.assertEqual([item.time for item in _by(added, "加仓")], ["d5"])
        self.assertGreater(added.final_shares, _by(added, "入场")[0].shares_after)
        uncovered = _run_scale([
            _sbar(0, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=98),
            _sbar(1, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=98),
            _sbar(2, 100, 101, category=UP_CONTINUE, direction_start=0, high=102, low=99.5),
            _sbar(3, 101, 100.5, category=UP_PULLBACK, direction_start=0, high=101, low=99.2),
            _sbar(4, 100.5, 101, category=UP_CONTINUE, direction_start=0, high=101.5, low=99.2),
            _sbar(5, 101, 101, category=UP_CONTINUE, direction_start=0, high=101.5, low=99.2),
        ])
        self.assertEqual(_by(uncovered, "加仓"), [])

    def test_pullback_on_the_confirmation_bar_does_not_open(self):
        result = _run_scale([
            _sbar(0, 100, 100, category=UP_PULLBACK, direction_start=0, high=101, low=99),
            _sbar(1, 100, 101, category=UP_CONTINUE, direction_start=0, high=102, low=99),
        ])
        self.assertEqual(_by(result, "入场"), [])
        self.assertIn("主状态不允许试仓", dict(result.blocked))

    def test_exhaustion_peels_once_and_down_birth_flattens(self):
        peeled = _run_scale([
            _sbar(0, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=98),
            _sbar(1, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=98),
            _sbar(2, 100, 103, category=UP_CONTINUE, direction_start=0, high=104, low=99.5),
            _sbar(3, 103, 102, category=UP_PULLBACK, direction_start=0, high=103.5, low=101.5),
            _sbar(4, 102, 103, category=UP_CONTINUE, direction_start=0, high=104, low=101.5),
            _sbar(5, 103, 103, category=UP_CONTINUE, direction_start=0, high=104, low=102),
            _sbar(6, 103, 103, category=UP_EXHAUSTION, direction_start=0, high=104, low=102),
            _sbar(7, 103, 103, category=UP_EXHAUSTION, direction_start=0, high=104, low=102),
            _sbar(8, 103, 103, category=UP_EXHAUSTION, direction_start=0, high=104, low=102),
        ])
        reductions = _by(peeled, "减仓")
        self.assertEqual([item.time for item in reductions], ["d7"])
        self.assertGreater(reductions[0].shares_after, 0)
        self.assertEqual(reductions[0].exit_rule, "上涨衰竭")
        flat = _run_scale([
            _sbar(0, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=98),
            _sbar(1, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=98),
            _sbar(2, 100, 100, category=DOWN_BIRTH, structure=DOWNTREND, direction_start=2, high=101, low=98),
            _sbar(3, 100, 100, category="down_continue", structure=DOWNTREND, direction_start=2, high=101, low=98),
        ])
        exits = _by(flat, "趋势退出")
        self.assertEqual([item.time for item in exits], ["d3"])
        self.assertEqual(exits[0].shares_after, 0)
        self.assertEqual(exits[0].exit_rule, "下跌初生")
        held = _run_scale([
            _sbar(0, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=98),
            _sbar(1, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=98),
            _sbar(2, 100, 100, category=NEUTRAL, structure=RANGE, direction_start=2, high=101, low=98),
            _sbar(3, 100, 100, category=NEUTRAL, structure=RANGE, direction_start=2, high=101, low=98),
        ])
        self.assertEqual(_by(held, "趋势退出"), [])
        self.assertGreater(held.final_shares, 0)

    def test_banking_the_starter_leaves_the_later_tranche(self):
        result = _run_scale([
            _sbar(0, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=98),
            _sbar(1, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=98),
            _sbar(2, 100, 103, category=UP_CONTINUE, direction_start=0, high=104, low=99.5),
            _sbar(3, 103, 102, category=UP_PULLBACK, direction_start=0, high=103.5, low=101.5),
            _sbar(4, 102, 103, category=UP_CONTINUE, direction_start=0, high=104, low=101.5),
            _sbar(5, 103, 103, category=UP_CONTINUE, direction_start=0, high=104, low=102),
            _sbar(6, 108, 110, category=UP_CONTINUE, direction_start=0, high=110, low=108),
            _sbar(7, 108, 108, category=UP_CONTINUE, direction_start=0, high=109, low=108),
        ])
        reductions = _by(result, "减仓")
        self.assertEqual([item.time for item in reductions], ["d7"])
        self.assertEqual(reductions[0].exit_rule, "试仓止盈")
        self.assertGreater(reductions[0].shares_after, 0)
        self.assertGreaterEqual(len(_by(result, "加仓")), 1)

    def test_wider_volatility_does_not_lower_the_stop(self):
        result = _run_scale([
            _sbar(0, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=99),
            _sbar(1, 100, 102, category=UP_BIRTH, direction_start=0, high=103, low=99.5),
            _sbar(2, 102, 103, category=UP_CONTINUE, direction_start=0, high=104, low=101.5),
            _sbar(3, 103, 103, category=UP_CONTINUE, direction_start=0, high=104, low=102, a_t=20),
        ])
        self.assertAlmostEqual(result.equity[2].next_stop, 101.0)
        self.assertAlmostEqual(result.equity[3].next_stop, 101.0)
        self.assertEqual(_by(result, "止损"), [])

    def test_add_budget_shrinks_and_then_skips(self):
        from analysis.backtest.trend.risk import add_risk_budget
        self.assertIsNone(add_risk_budget(0.008, 0.01, 0.005))
        self.assertAlmostEqual(add_risk_budget(0.006, 0.01, 0.005), 0.004)
        self.assertAlmostEqual(add_risk_budget(0.0, 0.01, 0.005), 0.005)


def _run_break(bars, price_axis="uniform"):
    config = load_trend_config(
        ROOT / "analysis" / "backtest" / "trend-strategy-break.yaml",
        today=date(2026, 10, 16),
    )
    strategy = replace(config.strategies["break"], order=replace(config.strategies["break"].order, slippage_rate=0.0))
    market = replace(
        config.markets[MARKET_HK],
        commission_rate=0.0,
        min_commission=0.0,
        platform_fee=0.0,
        stamp_duty_rate=0.0,
        exchange_fee_rate=0.0,
        settlement_rate=0.0,
    )
    tactic = create_entry("break", strategy, price_axis=price_axis)
    return run_account(
        bars, tactic, strategy, market, lot_size=1, capital=100_000.0, price_axis=price_axis,
    )


class BreakTests(unittest.TestCase):
    def test_config_uses_five_times_the_scale_budget(self):
        config = load_trend_config(
            ROOT / "analysis" / "backtest" / "trend-strategy-break.yaml",
            today=date(2026, 10, 16),
        )
        strategy = config.strategies["break"]
        self.assertEqual(strategy.exit.mode, "break_book")
        self.assertEqual(strategy.risk.risk_per_trade, 0.025)
        self.assertEqual(strategy.risk.max_open_risk, 0.05)
        self.assertEqual(strategy.risk.max_gross_exposure, 1.0)
        self.assertEqual(strategy.breakout.step, 0.0125)
        self.assertEqual(strategy.breakout.range_break_risk, 0.0125)
        self.assertEqual(strategy.exit.add_floor, 0.0125)
        self.assertEqual(config.runs[0].key, "break")
        data = load_trend_yaml(ROOT / "analysis" / "backtest" / "trend-strategy-break.yaml")
        data["strategies"]["break"]["breakout"]["short_down"] = 1
        with self.assertRaisesRegex(ValueError, "short_down"):
            trend_config_from_mapping(data, source="break-test")

    def test_plain_entry_is_five_times_the_old_tranche(self):
        result = _run_break([
            _sbar(0, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=99),
            _sbar(1, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=99),
        ])
        entry = _by(result, "入场")[0]
        self.assertAlmostEqual(entry.planned_risk, 0.025, places=3)

    def test_up_break_and_support_raise_the_tranche(self):
        up = _run_break([
            _sbar(0, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=80, a_t=10, up_breaks=("short|boundary|line|up",)),
            _sbar(1, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=80, a_t=10),
        ])
        both = _run_break([
            _sbar(
                0, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=80, a_t=10,
                up_breaks=("short|boundary|line|up",), support_valid=True,
            ),
            _sbar(1, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=80, a_t=10),
        ])
        self.assertAlmostEqual(_by(up, "入场")[0].planned_risk, 0.0375, places=3)
        self.assertAlmostEqual(_by(both, "入场")[0].planned_risk, 0.05, places=3)

    def test_down_break_blocks_the_entry_and_a_used_event_does_not_boost_the_next_order(self):
        blocked = _run_break([
            _sbar(
                0, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=99,
                up_breaks=("short|boundary|line|up",), down_breaks=("short|boundary|floor|down",),
            ),
            _sbar(1, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=99),
        ])
        self.assertEqual(_by(blocked, "入场"), [])
        reused = _run_break([
            _sbar(0, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=99, up_breaks=("short|boundary|line|up",)),
            _sbar(1, 101, 100, category=UP_BIRTH, direction_start=1, high=101, low=99),
            _sbar(2, 100, 100, category=UP_BIRTH, direction_start=1, high=100, low=99),
        ])
        self.assertEqual([item.action for item in reused.decisions if item.action in ("取消", "入场")], ["取消", "入场"])
        self.assertAlmostEqual(_by(reused, "入场")[0].planned_risk, 0.025, places=3)

    def test_short_down_peels_and_mid_down_flattens(self):
        peeled = _run_break([
            _sbar(0, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=99),
            _sbar(1, 100, 102, category=UP_CONTINUE, direction_start=0, high=103, low=99.5),
            _sbar(2, 102, 103, category=UP_CONTINUE, direction_start=0, high=104, low=101, down_breaks=("short|boundary|floor|down",)),
            _sbar(3, 103, 103, category=UP_CONTINUE, direction_start=0, high=104, low=102),
        ])
        self.assertEqual([item.time for item in _by(peeled, "减仓")], ["d3"])
        self.assertEqual(_by(peeled, "趋势退出"), [])
        flat = _run_break([
            _sbar(0, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=99),
            _sbar(1, 100, 102, category=UP_CONTINUE, direction_start=0, high=103, low=99.5),
            _sbar(2, 102, 103, category=UP_CONTINUE, direction_start=0, high=104, low=101, down_breaks=("mid|boundary|floor|down",)),
            _sbar(3, 103, 103, category=UP_CONTINUE, direction_start=0, high=104, low=102),
        ])
        self.assertEqual([item.time for item in _by(flat, "趋势退出")], ["d3"])
        self.assertEqual(flat.equity[3].shares, 0)

    def test_range_break_buys_the_small_tranche_at_the_tighter_stop(self):
        result = _run_break([
            _sbar(
                0, 100, 100, structure=RANGE, category=PENDING, direction_start=0,
                high=100, low=99, phase_start_index=0, phase_trigger="range_break_up", range_floor=98.5,
            ),
            _sbar(1, 100, 100, structure=RANGE, category=PENDING, direction_start=0, high=100, low=99),
        ])
        entry = _by(result, "入场")[0]
        self.assertEqual(entry.time, "d1")
        self.assertAlmostEqual(entry.planned_risk, 0.0125, places=3)
        self.assertAlmostEqual(entry.adjusted_stop, 98.5)

    def test_downtrend_phase_still_flattens(self):
        result = _run_break([
            _sbar(0, 100, 100, category=UP_BIRTH, direction_start=0, high=100, low=99),
            _sbar(1, 100, 102, category=UP_CONTINUE, direction_start=0, high=103, low=99.5),
            _sbar(2, 102, 102, structure=DOWNTREND, category=DOWN_BIRTH, direction_start=2, high=103, low=101.5),
            _sbar(3, 102, 101, structure=DOWNTREND, category=DOWN_BIRTH, direction_start=2, high=102, low=100.5),
        ])
        self.assertEqual([item.time for item in _by(result, "趋势退出")], ["d3"])

    def test_break_events_are_read_only_on_their_available_bar(self):
        from analysis.backtest.trend.breaks import read_breaks
        from analysis.structure.series import SeriesBar, StandardSeries

        params = default_params("uniform", version=PARAM_VERSION_V6)
        series = StandardSeries(
            (
                SeriesBar(0, datetime(2026, 1, 2, tzinfo=HK), 101.0, 101.0, False),
                SeriesBar(1, datetime(2026, 1, 5, tzinfo=HK), 101.0, 101.0, False),
            ),
            "close",
            "uniform",
            "1d",
            params.version,
            "test",
            params,
        )
        view = {
            "boundaries": [{
                "id": "short:resistance:0-4",
                "scale": "short",
                "role": "resistance",
                "status": "broken",
                "primary": True,
                "slope": 0.0,
                "intercept": 90.0,
                "volatility": 1.0,
            }, {
                "id": "short:support:0-4",
                "scale": "short",
                "role": "support",
                "status": "validated",
                "primary": True,
                "slope": 0.0,
                "intercept": 100.0,
                "volatility": 2.0,
            }],
            "zones": [],
            "events": [{
                "event_type": "breakout",
                "available_index": 0,
                "object_kind": "boundary",
                "object_id": "short:resistance:0-4",
            }],
            "pivots": [{
                "scale": "short",
                "role": "low",
                "extreme_index": 0,
                "confirm_index": 0,
                "geometric_price": 100.0,
            }],
            "methods": {
                "phase": {"objects": []},
                "direction": {"objects": [{
                    "scale": "short",
                    "structure": "uptrend",
                    "start_index": 0,
                    "end_index": 0,
                }]},
            },
        }
        fresh = read_breaks(view, series, 0, ("short", "mid"), entry_scale="short")
        later = read_breaks(view, series, 1, ("short", "mid"), entry_scale="short")
        self.assertEqual(fresh.up, ("short|boundary|short:resistance:0-4|up",))
        self.assertEqual(later.up, ())
        self.assertTrue(fresh.support_valid)


if __name__ == "__main__":
    unittest.main()
