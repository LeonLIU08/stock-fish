"""每日开盘止盈止损：路径依赖成交。"""
from __future__ import annotations

import unittest
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from analysis.backtest import strategies as _builtin_strategies  # noqa: F401
from analysis.backtest.config import BacktestConfig, CostModel, load_backtest_config
from analysis.backtest.strategies.monday_open import (
    breach_reason,
    simulate_monday_open,
)
from analysis.backtest.strategy import create_strategy, list_strategies

HK = ZoneInfo("Asia/Hong_Kong")


def _bar(ts: datetime, open_: float, close: float, low: float | None = None, high: float | None = None) -> dict:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=HK)
    low = close if low is None else low
    high = close if high is None else high
    return {
        "datetime": pd.Timestamp(ts),
        "open": float(open_),
        "high": float(max(high, open_, close)),
        "low": float(min(low, open_, close)),
        "close": float(close),
        "volume": 1000,
        "amount": float(close) * 1000,
    }


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _config(**kwargs) -> BacktestConfig:
    cost = kwargs.pop("cost", CostModel(commission_rate=0.0, stamp_duty_rate=0.0, lot_size=100))
    return BacktestConfig(
        symbols=["00001"],
        symbol_lot_sizes={"00001": 100},
        interval="60m",
        start=date(2025, 1, 1),
        end=date(2025, 2, 1),
        capital=kwargs.pop("capital", 1_000_000),
        cost=cost,
        strategy_name="monday_open",
        **kwargs,
    )


class BreachTests(unittest.TestCase):
    def test_exact_thresholds_do_not_fire(self):
        self.assertIsNone(breach_reason(110.0, 100.0, 0.10, 0.05))
        self.assertIsNone(breach_reason(95.0, 100.0, 0.10, 0.05))

    def test_strict_thresholds_fire(self):
        self.assertEqual(breach_reason(110.01, 100.0, 0.10, 0.05), "take_profit")
        self.assertEqual(breach_reason(94.99, 100.0, 0.10, 0.05), "stop_loss")


class MondayOpenTests(unittest.TestCase):
    def test_buys_monday_open_and_holds_inside_the_band(self):
        df = _frame([
            _bar(datetime(2025, 1, 6, 9, 30), 100, 101),
            _bar(datetime(2025, 1, 6, 10, 30), 101, 102),
            _bar(datetime(2025, 1, 7, 9, 30), 102, 103),
        ])
        result = simulate_monday_open(df, _config(), "00001")
        self.assertEqual([t.side for t in result.trades], ["buy"])
        self.assertEqual(result.trades[0].price, 100)
        self.assertEqual(result.trades[0].reason, "daily_open")
        self.assertGreater(result.final_shares, 0)

    def test_buys_the_first_session_even_when_it_is_not_monday(self):
        df = _frame([
            _bar(datetime(2025, 1, 7, 9, 30), 100, 100),
            _bar(datetime(2025, 1, 8, 9, 30), 100, 100),
            _bar(datetime(2025, 1, 13, 9, 30), 110, 110),
        ])
        result = simulate_monday_open(df, _config(), "00001")
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(pd.Timestamp(result.trades[0].timestamp).day, 7)
        self.assertEqual(result.trades[0].price, 100)

    def test_does_not_add_while_holding(self):
        df = _frame([
            _bar(datetime(2025, 1, 6, 9, 30), 100, 104),
            _bar(datetime(2025, 1, 13, 9, 30), 105, 106),
        ])
        result = simulate_monday_open(df, _config(), "00001")
        self.assertEqual([t.side for t in result.trades], ["buy"])

    def test_close_stop_fills_next_open_even_if_price_recovers(self):
        df = _frame([
            _bar(datetime(2025, 1, 6, 9, 30), 100, 100),
            _bar(datetime(2025, 1, 6, 10, 30), 100, 94),
            _bar(datetime(2025, 1, 6, 11, 30), 97, 97),
        ])
        result = simulate_monday_open(df, _config(), "00001")
        self.assertEqual([t.side for t in result.trades], ["buy", "sell"])
        sell = result.trades[1]
        self.assertEqual(sell.price, 97)
        self.assertEqual(sell.reason, "stop_loss")
        self.assertEqual(result.final_shares, 0)
        self.assertEqual(pd.Timestamp(sell.timestamp).hour, 11)

    def test_close_take_profit_fills_next_open(self):
        df = _frame([
            _bar(datetime(2025, 1, 6, 9, 30), 100, 100),
            _bar(datetime(2025, 1, 6, 10, 30), 110, 111),
            _bar(datetime(2025, 1, 6, 11, 30), 109, 109),
        ])
        result = simulate_monday_open(df, _config(), "00001")
        self.assertEqual(result.trades[1].reason, "take_profit")
        self.assertEqual(result.trades[1].price, 109)

    def test_exact_close_does_not_exit(self):
        df = _frame([
            _bar(datetime(2025, 1, 6, 9, 30), 100, 110),
            _bar(datetime(2025, 1, 6, 10, 30), 110, 95),
            _bar(datetime(2025, 1, 6, 11, 30), 95, 100),
        ])
        result = simulate_monday_open(df, _config(), "00001")
        self.assertEqual([t.side for t in result.trades], ["buy"])

    def test_intrabar_wick_does_not_exit(self):
        df = _frame([
            _bar(datetime(2025, 1, 6, 9, 30), 100, 100),
            _bar(datetime(2025, 1, 6, 10, 30), 100, 97, low=90, high=100),
            _bar(datetime(2025, 1, 6, 11, 30), 97, 98),
        ])
        result = simulate_monday_open(df, _config(), "00001")
        self.assertEqual([t.side for t in result.trades], ["buy"])
        self.assertGreater(result.final_shares, 0)

    def test_gap_stop_on_monday_open_does_not_rebuy_that_open(self):
        df = _frame([
            _bar(datetime(2025, 1, 6, 9, 30), 100, 100),
            _bar(datetime(2025, 1, 13, 9, 30), 80, 80),
            _bar(datetime(2025, 1, 14, 9, 30), 80, 80),
        ])
        result = simulate_monday_open(df, _config(), "00001")
        self.assertEqual([t.reason for t in result.trades], ["daily_open", "stop_loss_gap", "daily_open"])
        self.assertEqual(result.trades[1].price, 80)
        self.assertEqual(pd.Timestamp(result.trades[1].timestamp).day, 13)
        self.assertEqual(pd.Timestamp(result.trades[2].timestamp).day, 14)

    def test_intraday_stop_does_not_rebuy_until_the_next_session(self):
        df = _frame([
            _bar(datetime(2025, 1, 6, 9, 30), 100, 100),
            _bar(datetime(2025, 1, 6, 10, 30), 100, 94),
            _bar(datetime(2025, 1, 6, 11, 30), 97, 97),
            _bar(datetime(2025, 1, 6, 13, 0), 97, 97),
            _bar(datetime(2025, 1, 7, 9, 30), 97, 97),
        ])
        result = simulate_monday_open(df, _config(), "00001")
        self.assertEqual([t.reason for t in result.trades], ["daily_open", "stop_loss", "daily_open"])
        self.assertEqual(pd.Timestamp(result.trades[1].timestamp).day, 6)
        self.assertEqual(pd.Timestamp(result.trades[2].timestamp).day, 7)

    def test_friday_close_stop_sells_next_open_and_buys_the_following_day(self):
        df = _frame([
            _bar(datetime(2025, 1, 6, 9, 30), 100, 100),
            _bar(datetime(2025, 1, 10, 9, 30), 100, 90),
            _bar(datetime(2025, 1, 13, 9, 30), 92, 92),
            _bar(datetime(2025, 1, 14, 9, 30), 92, 92),
        ])
        result = simulate_monday_open(df, _config(), "00001")
        self.assertEqual([t.reason for t in result.trades], ["daily_open", "stop_loss", "daily_open"])
        self.assertEqual(result.trades[1].price, 92)
        self.assertEqual(pd.Timestamp(result.trades[1].timestamp).day, 13)
        self.assertEqual(pd.Timestamp(result.trades[2].timestamp).day, 14)

    def test_gap_take_profit_fills_at_the_open(self):
        df = _frame([
            _bar(datetime(2025, 1, 6, 9, 30), 100, 100),
            _bar(datetime(2025, 1, 7, 9, 30), 120, 121),
        ])
        result = simulate_monday_open(df, _config(), "00001")
        self.assertEqual(result.trades[1].reason, "take_profit_gap")
        self.assertEqual(result.trades[1].price, 120)

    def test_last_bar_breach_stays_open(self):
        df = _frame([
            _bar(datetime(2025, 1, 6, 9, 30), 100, 120),
        ])
        result = simulate_monday_open(df, _config(), "00001")
        self.assertEqual([t.side for t in result.trades], ["buy"])
        self.assertGreater(result.final_shares, 0)
        self.assertAlmostEqual(result.equity["equity"].iloc[-1], result.final_cash + result.final_shares * 120)

    def test_lot_larger_than_cash_is_skipped(self):
        df = _frame([
            _bar(datetime(2025, 1, 6, 9, 30), 100, 100),
        ])
        cfg = _config(capital=1_000, cost=CostModel(commission_rate=0.0, stamp_duty_rate=0.0, lot_size=100))
        cfg.symbol_lot_sizes["00001"] = 100
        result = simulate_monday_open(df, cfg, "00001")
        self.assertEqual(result.trades, [])
        self.assertEqual(len(result.skipped), 1)
        self.assertIn("一手", result.skipped[0].reason)
        self.assertEqual(result.final_shares, 0)
        self.assertEqual(result.final_cash, 1_000)

    def test_fees_and_lot_rounding(self):
        df = _frame([
            _bar(datetime(2025, 1, 6, 9, 30), 100, 100),
        ])
        cost = CostModel(commission_rate=0.0003, stamp_duty_rate=0.001, lot_size=100)
        cfg = _config(cost=cost)
        cfg.symbol_lot_sizes["00001"] = 100
        result = simulate_monday_open(df, cfg, "00001")
        qty = cost.max_shares(1_000_000, 100, lot_size=100)
        buy = result.trades[0]
        self.assertEqual(buy.shares, qty)
        self.assertEqual(buy.shares % 100, 0)
        fee = cost.fee(qty * 100)
        self.assertAlmostEqual(buy.fee, fee)
        self.assertAlmostEqual(result.final_cash, 1_000_000 - qty * 100 - fee)

    def test_refuses_daily_bars(self):
        cfg = BacktestConfig(
            symbols=["00001"],
            symbol_lot_sizes={"00001": 100},
            interval="1d",
            start=date(2025, 1, 1),
            end=date(2025, 2, 1),
            capital=1_000_000,
        )
        with self.assertRaisesRegex(ValueError, "60m"):
            simulate_monday_open(_frame([_bar(datetime(2025, 1, 6, 9, 30), 100, 100)]), cfg, "00001")


class RegistryTests(unittest.TestCase):
    def test_registered(self):
        self.assertIn("monday_open", list_strategies())
        strategy = create_strategy("monday_open")
        cfg = _config()
        variants = strategy.variants(cfg)
        self.assertEqual(variants[0].key, "band")
        self.assertEqual(strategy.warmup_bars(cfg, variants[0]), 0)

    def test_yaml_universe(self):
        cfg = load_backtest_config("analysis/backtest/monday_open.yaml")
        self.assertEqual(cfg.strategy_name, "monday_open")
        self.assertEqual(cfg.interval, "60m")
        self.assertEqual(cfg.capital, 1_000_000)
        self.assertGreater(len(cfg.symbols), 600)
        self.assertEqual(cfg.symbols[0], "00001")
        self.assertTrue(cfg.fetch_lot_sizes)
        self.assertTrue(cfg.skip_symbol_errors)
        self.assertTrue(cfg.write_charts)
        self.assertEqual(cfg.strategy_params["take_profit"], 0.10)
        self.assertEqual(cfg.strategy_params["stop_loss"], 0.05)


if __name__ == "__main__":
    unittest.main()
