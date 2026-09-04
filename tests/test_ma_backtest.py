"""Synthetic tests for the extensible backtest framework."""
from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from analysis.backtest import strategies as _builtin_strategies  # noqa: F401
from analysis.backtest.analytics import daily_frame, monthly_return_table, split_cost_table
from analysis.backtest.bars import to_yahoo_hk
from analysis.backtest.config import DEFAULT_CONFIG_PATH, BacktestConfig, CostModel, load_backtest_config
from analysis.backtest.engine import simulate
from analysis.backtest.signals import SIGNAL_ACTION_COL, SignalAction, attach_signals
from analysis.backtest.strategy import create_strategy, list_strategies
from analysis.backtest.strategies.ma_cross import compute_ma_cross_signals

HK = ZoneInfo("Asia/Hong_Kong")


def _bars(closes, interval_minutes=1440):
    start = datetime(2025, 1, 2, 16, 0, tzinfo=HK)
    rows = []
    price = float(closes[0])
    for i, close in enumerate(closes):
        ts = start + timedelta(minutes=interval_minutes * i)
        close = float(close)
        open_ = price
        rows.append({
            "datetime": pd.Timestamp(ts),
            "open": open_,
            "high": max(open_, close) * 1.001,
            "low": min(open_, close) * 0.999,
            "close": close,
            "volume": 1000,
            "amount": close * 1000,
        })
        price = close
    return pd.DataFrame(rows)


class SymbolTests(unittest.TestCase):
    def test_yahoo_hk_codes(self):
        self.assertEqual(to_yahoo_hk("00700"), "0700.HK")
        self.assertEqual(to_yahoo_hk("06869"), "6869.HK")
        self.assertEqual(to_yahoo_hk("09988"), "9988.HK")
        self.assertEqual(to_yahoo_hk("HK00700"), "0700.HK")


class StrategyRegistryTests(unittest.TestCase):
    def test_ma_cross_registered(self):
        self.assertIn("ma_cross", list_strategies())
        strategy = create_strategy("ma_cross")
        self.assertEqual(strategy.name, "ma_cross")


class MaSignalTests(unittest.TestCase):
    def test_golden_and_death_on_ma5_ma10(self):
        closes = [10.0] * 12 + [12.0] * 12 + [8.0] * 12
        df = compute_ma_cross_signals(_bars(closes), 5, 10)
        self.assertTrue((df[SIGNAL_ACTION_COL] == SignalAction.BUY.value).any())
        self.assertTrue((df[SIGNAL_ACTION_COL] == SignalAction.SELL.value).any())
        first_buy = df.index[df[SIGNAL_ACTION_COL] == SignalAction.BUY.value].min()
        first_sell = df.index[df[SIGNAL_ACTION_COL] == SignalAction.SELL.value].min()
        self.assertLess(first_buy, first_sell)


class EngineTests(unittest.TestCase):
    def test_long_only_next_open_fill(self):
        closes = [10.0] * 12 + [12.0] * 12 + [8.0] * 12
        df = compute_ma_cross_signals(_bars(closes), 5, 10)
        cfg = BacktestConfig(
            symbols=["TEST"],
            variants=["ma5_ma10"],
            interval="1d",
            start=date(2025, 1, 1),
            end=date(2025, 6, 1),
            capital=100_000,
            cost=CostModel(commission_rate=0.0, stamp_duty_rate=0.0),
        )
        result = simulate(df, cfg)
        sides = [t.side for t in result.trades]
        self.assertIn("buy", sides)
        self.assertIn("sell", sides)
        self.assertEqual(sides[0], "buy")
        buy = result.trades[0]
        self.assertGreater(buy.shares, 0)
        self.assertEqual(result.final_shares, 0)

    def test_max_two_trades_per_day_on_minute_bars(self):
        start = datetime(2025, 1, 2, 10, 0, tzinfo=HK)
        rows = []
        actions = []
        reasons = []
        for i in range(6):
            rows.append({
                "datetime": pd.Timestamp(start + timedelta(minutes=5 * i)),
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 1,
                "amount": 10,
            })
            if i in (0, 2, 4):
                actions.append(SignalAction.BUY.value)
                reasons.append("buy")
            elif i in (1, 3):
                actions.append(SignalAction.SELL.value)
                reasons.append("sell")
            else:
                actions.append(SignalAction.HOLD.value)
                reasons.append("")
        df = attach_signals(pd.DataFrame(rows), actions, reasons=reasons)
        cfg = BacktestConfig(
            symbols=["TEST"],
            variants=["ma5_ma10"],
            interval="5m",
            start=date(2025, 1, 1),
            end=date(2025, 1, 3),
            capital=100_000,
            max_trades_per_day=2,
            cost=CostModel(commission_rate=0.0, stamp_duty_rate=0.0),
        )
        result = simulate(df, cfg)
        self.assertEqual([t.side for t in result.trades], ["buy", "sell"])
        self.assertTrue(any("当日交易次数已达上限" in t.reason for t in result.skipped))
        self.assertIn("equity_gross", result.equity.columns)
        self.assertIn("leverage", result.equity.columns)

    def test_lot_size_rounds_down_buy_quantity(self):
        closes = [10.0] * 12 + [12.0] * 12 + [8.0] * 12
        df = compute_ma_cross_signals(_bars(closes), 5, 10)
        cfg = BacktestConfig(
            symbols=["TEST"],
            symbol_lot_sizes={"TEST": 100},
            variants=["ma5_ma10"],
            interval="1d",
            start=date(2025, 1, 1),
            end=date(2025, 6, 1),
            capital=10_500,
            cost=CostModel(commission_rate=0.0, stamp_duty_rate=0.0, lot_size=100),
        )
        result = simulate(df, cfg, symbol="TEST")
        buy = next(t for t in result.trades if t.side == "buy")
        expected = cfg.cost.max_shares(cfg.capital, buy.price, lot_size=100)
        self.assertEqual(buy.shares, expected)
        self.assertEqual(buy.shares % 100, 0)


class ConfigTests(unittest.TestCase):
    def test_load_yaml_defaults(self):
        cfg = load_backtest_config(DEFAULT_CONFIG_PATH)
        self.assertEqual(cfg.symbols, ["00700", "09988", "06869"])
        self.assertEqual(cfg.capital, 300_000)
        self.assertEqual(cfg.max_trades_per_day, 2)
        self.assertEqual(cfg.strategy_name, "ma_cross")
        self.assertIn("hsi", cfg.benchmarks)
        self.assertIn("hstech", cfg.benchmarks)
        self.assertEqual(set(cfg.variants), {"ma5_ma10", "ma5_ma20", "ma10_ma20"})

    def test_cli_overrides(self):
        cfg = load_backtest_config(
            DEFAULT_CONFIG_PATH,
            overrides={"symbols": ["00700"], "variants": ["ma5_ma10"], "capital": 100000, "interval": "5m"},
        )
        self.assertEqual(cfg.symbols, ["00700"])
        self.assertEqual(cfg.variants, ["ma5_ma10"])
        self.assertEqual(cfg.capital, 100000)
        self.assertEqual(cfg.interval, "5m")

    def test_legacy_schemes_override(self):
        cfg = load_backtest_config(
            DEFAULT_CONFIG_PATH,
            overrides={"schemes": ["ma5_ma10"]},
        )
        self.assertEqual(cfg.variants, ["ma5_ma10"])

    def test_per_symbol_lot_sizes_from_yaml(self):
        cfg = load_backtest_config(DEFAULT_CONFIG_PATH)
        self.assertEqual(cfg.lot_size_for("00700"), 100)
        self.assertEqual(cfg.lot_size_for("09988"), 100)
        self.assertEqual(cfg.lot_size_for("06869"), 500)


class AnalyticsTests(unittest.TestCase):
    def test_monthly_heatmap_and_split(self):
        idx = pd.date_range("2025-01-02", periods=40, freq="B", tz="Asia/Hong_Kong")
        equity = pd.DataFrame({
            "datetime": idx,
            "equity": 100_000 * (1.01 ** pd.Series(range(40))).values,
            "equity_gross": 100_000 * (1.012 ** pd.Series(range(40))).values,
            "turnover": 0.0,
            "position": 1,
            "leverage": 1.0,
        })
        daily = daily_frame(equity)
        heat = monthly_return_table(daily["ret_net"])
        self.assertFalse(heat.empty)
        cfg = BacktestConfig(
            symbols=["TEST"],
            variants=["ma5_ma10"],
            start=date(2025, 1, 2),
            end=date(2025, 2, 28),
            oos_ratio=0.3,
        )
        split = split_cost_table(daily, [], cfg)
        self.assertIn("full_post_cost", split)
        self.assertIsNotNone(split["full_post_cost"]["total_return"])
        self.assertGreater(split["full_pre_cost"]["total_return"], split["full_post_cost"]["total_return"])


class EvenDcaTests(unittest.TestCase):
    def test_schedule_constant_price(self):
        from analysis.backtest.analytics import simulate_even_dca

        # 10 个交易日、价格恒为 10；本金 5000、每手 100 股 → 可买 5 手，间隔 = 2
        closes = [10.0] * 10
        df = _bars(closes)
        cfg = BacktestConfig(
            symbols=["TEST"],
            symbol_lot_sizes={"TEST": 100},
            interval="1d",
            start=date(2025, 1, 1),
            end=date(2025, 6, 1),
            capital=5_000,
            cost=CostModel(commission_rate=0.0, stamp_duty_rate=0.0, lot_size=100),
        )
        result = simulate_even_dca(df, cfg, "TEST")
        self.assertEqual(result.n_lots, 5)
        self.assertEqual(result.n_trading_days, 10)
        self.assertEqual(result.interval, 2.0)
        self.assertEqual(result.n_lots_bought, 5)
        self.assertEqual(result.n_buy_days, 5)
        self.assertEqual(result.final_shares, 500)
        self.assertAlmostEqual(result.final_cash, 0.0, places=6)
        self.assertAlmostEqual(result.total_return, 0.0, places=6)
        shares = result.equity["shares"].tolist()
        # 买点在第 0/2/4/6/8 根（间隔 2）
        self.assertEqual(shares, [100, 100, 200, 200, 300, 300, 400, 400, 500, 500])

    def test_rising_price_lags_buy_hold(self):
        from analysis.backtest.analytics import simulate_even_dca
        from analysis.backtest.metrics import _buy_hold_return

        closes = [10.0 + 0.05 * i for i in range(10)]
        df = _bars(closes)
        cfg = BacktestConfig(
            symbols=["TEST"],
            symbol_lot_sizes={"TEST": 100},
            interval="1d",
            start=date(2025, 1, 1),
            end=date(2025, 6, 1),
            capital=3_500,
            cost=CostModel(commission_rate=0.0, stamp_duty_rate=0.0, lot_size=100),
        )
        result = simulate_even_dca(df, cfg, "TEST")
        # 起点收盘 10，每手 1000，N=3；间隔 10/3；买点 int(i * 10/3) = 0, 3, 6
        self.assertEqual(result.n_lots, 3)
        self.assertAlmostEqual(result.interval, 10 / 3)
        self.assertEqual(result.n_lots_bought, 3)
        self.assertEqual(result.final_shares, 300)
        bh = _buy_hold_return(df, df)
        self.assertIsNotNone(bh)
        self.assertLess(result.total_return, bh)

    def test_cannot_afford_one_lot(self):
        from analysis.backtest.analytics import simulate_even_dca

        df = _bars([100.0] * 8)
        cfg = BacktestConfig(
            symbols=["TEST"],
            symbol_lot_sizes={"TEST": 100},
            interval="1d",
            start=date(2025, 1, 1),
            end=date(2025, 6, 1),
            capital=1_000,
            cost=CostModel(commission_rate=0.0, stamp_duty_rate=0.0, lot_size=100),
        )
        result = simulate_even_dca(df, cfg, "TEST")
        self.assertEqual(result.n_lots, 0)
        self.assertIsNone(result.interval)
        self.assertEqual(result.n_lots_bought, 0)
        self.assertEqual(result.final_shares, 0)
        self.assertAlmostEqual(result.total_return, 0.0, places=6)


class ChartSmokeTests(unittest.TestCase):
    def test_charts_write_png(self):
        import tempfile
        from pathlib import Path

        from analysis.backtest.analytics import factor_exposures, nav_from_price
        from analysis.backtest.charts import (
            plot_daily_kline_with_trades,
            plot_factor_exposure,
            plot_monthly_heatmap,
            plot_nav_and_underwater,
            plot_position_turnover_leverage,
            plot_rolling,
        )
        from analysis.backtest.signals import ChartOverlay

        idx = pd.date_range("2025-01-02", periods=80, freq="B", tz="Asia/Hong_Kong")
        equity = pd.DataFrame({
            "datetime": idx,
            "equity": 100_000 * (1.002 ** pd.Series(range(80))).values,
            "equity_gross": 100_000 * (1.0022 ** pd.Series(range(80))).values,
            "turnover": [0.1 if i % 10 == 0 else 0 for i in range(80)],
            "position": [1 if i % 7 < 4 else 0 for i in range(80)],
            "leverage": [1 if i % 7 < 4 else 0 for i in range(80)],
            "ma_fast": range(80),
            "ma_slow": [x + 2 for x in range(80)],
        })
        daily = daily_frame(equity)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            bars = pd.DataFrame({
                "datetime": idx,
                "open": [100 + i * 0.2 for i in range(80)],
                "high": [101 + i * 0.2 for i in range(80)],
                "low": [99 + i * 0.2 for i in range(80)],
                "close": [100.5 + i * 0.2 for i in range(80)],
                "volume": [1000 + i * 10 for i in range(80)],
                "ma_fast": range(80),
                "ma_slow": [x + 2 for x in range(80)],
                "ma_fast_n": 5,
                "ma_slow_n": 10,
            })
            trades = [
                {"timestamp": idx[10], "side": "buy", "price": 102.5, "shares": 100, "skipped": False},
                {"timestamp": idx[30], "side": "sell", "price": 106.5, "shares": 100, "skipped": False},
            ]
            nav = nav_from_price(daily["equity"])
            bench = nav_from_price(daily["equity"] * 0.99)
            overlays = [
                ChartOverlay(column="ma_fast", label="MA5", color="#fbbf24"),
                ChartOverlay(column="ma_slow", label="MA10", color="#60a5fa"),
            ]
            self.assertIsNotNone(plot_daily_kline_with_trades(bars, trades, out, overlays=overlays))
            self.assertTrue((out / "daily_kline_trades.png").is_file())
            self.assertIsNotNone(plot_nav_and_underwater(nav, {"恒生指数": bench}, out))
            self.assertIsNotNone(plot_monthly_heatmap(monthly_return_table(daily["ret_net"]), out))
            from analysis.backtest.analytics import rolling_stats
            roll = rolling_stats(daily["ret_net"], {"hsi": daily["equity"].pct_change()}, 40)
            self.assertIsNotNone(plot_rolling(roll, 40, out))
            self.assertIsNotNone(plot_position_turnover_leverage(daily, out))
            self.assertIsNotNone(plot_factor_exposure(factor_exposures(daily), out))
            self.assertTrue((out / "nav_underwater.png").is_file())


if __name__ == "__main__":
    unittest.main()
