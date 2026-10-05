"""Backtest configuration: YAML file + in-memory BacktestConfig."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

BAR_INTERVALS = ("1m", "5m", "15m", "60m", "1d")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "ma_cross.yaml"

# Approximate HK cash-session bars per trading day (09:30-12:00 + 13:00-16:00).
BARS_PER_DAY = {
    "1m": 330,
    "5m": 66,
    "15m": 22,
    "60m": 6,
    "1d": 1,
}

DEFAULT_SYMBOLS = ("00700", "09988", "06869")
DEFAULT_SYMBOL_NAMES = {
    "00700": "腾讯控股",
    "09988": "阿里巴巴",
    "06869": "长飞光纤",
}
DEFAULT_SYMBOL_LOT_SIZES = {
    "00700": 100,
    "09988": 100,
    "06869": 500,
}

DEFAULT_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "hsi": {
        "name": "恒生指数",
        "yahoo": "^HSI",
        "fallbacks": ["HSI.HI"],
    },
    "hstech": {
        "name": "恒生科技指数",
        "yahoo": "HSTECH.HK",
        "fallbacks": ["3033.HK"],
    },
}

BENCHMARKS = DEFAULT_BENCHMARKS


@dataclass
class CostModel:
    """Hong Kong cash-equity costs. Stamp duty is charged on both sides."""

    commission_rate: float = 0.0003
    stamp_duty_rate: float = 0.001
    lot_size: int = 1

    @property
    def round_trip_friction(self) -> float:
        return 2.0 * (self.commission_rate + self.stamp_duty_rate)

    def fee(self, notional: float) -> float:
        if notional <= 0:
            return 0.0
        return abs(notional) * (self.commission_rate + self.stamp_duty_rate)

    def max_shares(self, cash: float, price: float, lot_size: Optional[int] = None) -> int:
        if cash <= 0 or price <= 0:
            return 0
        lot = self.lot_size if lot_size is None else lot_size
        if lot < 1:
            raise ValueError(f"lot_size 必须 >= 1，收到 {lot}")
        cost_rate = self.commission_rate + self.stamp_duty_rate
        raw = int(cash / (price * (1.0 + cost_rate)))
        if lot > 1:
            raw = (raw // lot) * lot
        return max(raw, 0)


@dataclass
class BacktestConfig:
    symbols: List[str] = field(default_factory=lambda: list(DEFAULT_SYMBOLS))
    symbol_names: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SYMBOL_NAMES))
    symbol_lot_sizes: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SYMBOL_LOT_SIZES))
    strategy_name: str = "ma_cross"
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    variants: List[str] = field(default_factory=list)
    interval: str = "1d"
    start: date = field(default_factory=lambda: date.today() - timedelta(days=365))
    end: date = field(default_factory=lambda: date.today())
    lookback_years: float = 1.0
    capital: float = 300_000.0
    independent: bool = True
    max_trades_per_day: int = 2
    long_only: bool = True
    signal_on: str = "close"
    fill_on: str = "next_open"
    position_mode: str = "all_in"
    cost: CostModel = field(default_factory=CostModel)
    timezone: str = "Asia/Hong_Kong"
    benchmarks: Dict[str, Dict[str, Any]] = field(default_factory=lambda: _copy_benchmarks())
    oos_start: Optional[date] = None
    oos_ratio: float = 0.3
    rolling_window_days: int = 252
    extreme_days: int = 5
    output_dir: str = "backtest_results"
    config_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.symbols = [str(s).strip() for s in self.symbols if str(s).strip()]
        self.strategy_name = str(self.strategy_name or "ma_cross").strip().lower()
        self.strategy_params = dict(self.strategy_params or {})
        self.variants = [str(v).strip() for v in (self.variants or []) if str(v).strip()]
        if self.interval not in BAR_INTERVALS:
            raise ValueError(f"interval 必须是 {BAR_INTERVALS} 之一，收到 {self.interval}")
        if not self.symbols:
            raise ValueError("symbols 不能为空")
        for symbol, lot in self.symbol_lot_sizes.items():
            if lot < 1:
                raise ValueError(f"{symbol} 的 lot_size 必须 >= 1，收到 {lot}")
        for symbol in self.symbols:
            if symbol not in self.symbol_lot_sizes:
                self.symbol_lot_sizes[symbol] = self.cost.lot_size
        if self.capital <= 0:
            raise ValueError("capital 必须 > 0")
        if self.max_trades_per_day < 1:
            raise ValueError("max_trades_per_day 必须 >= 1")
        if self.end < self.start:
            raise ValueError("end 不能早于 start")
        if self.signal_on != "close":
            raise ValueError("当前仅支持 signal_on=close")
        if self.fill_on != "next_open":
            raise ValueError("当前仅支持 fill_on=next_open")
        if self.position_mode not in ("all_in", "fraction"):
            raise ValueError("position_mode 必须是 all_in 或 fraction")
        if not self.independent:
            raise ValueError("当前仅支持分股独立核算")
        if not 0 <= self.oos_ratio < 1:
            raise ValueError("oos_ratio 必须在 [0, 1)")
        if self.rolling_window_days < 20:
            raise ValueError("rolling_window_days 必须 >= 20")
        self.timezone = self.timezone or "Asia/Hong_Kong"
        if not self.benchmarks:
            self.benchmarks = _copy_benchmarks()

    def warmup_calendar_days(self, warmup_bars: int) -> int:
        bars_per_day = max(BARS_PER_DAY.get(self.interval, 1), 1)
        bar_days = int((max(warmup_bars, 1) + bars_per_day - 1) / bars_per_day)
        return max(bar_days + 2, 5 if self.interval == "1d" else 2)

    def lot_size_for(self, symbol: str) -> int:
        return int(self.symbol_lot_sizes.get(symbol, self.cost.lot_size))

    def resolved_oos_start(self) -> date:
        if self.oos_start is not None:
            return self.oos_start
        if self.oos_ratio <= 0:
            return self.end + timedelta(days=1)
        span = max((self.end - self.start).days, 1)
        return self.start + timedelta(days=int(round(span * (1.0 - self.oos_ratio))))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbols": self.symbols,
            "symbol_names": self.symbol_names,
            "symbol_lot_sizes": self.symbol_lot_sizes,
            "strategy": {
                "name": self.strategy_name,
                "params": self.strategy_params,
                "variants": self.variants,
            },
            "interval": self.interval,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "lookback_years": self.lookback_years,
            "capital": self.capital,
            "independent": self.independent,
            "max_trades_per_day": self.max_trades_per_day,
            "long_only": self.long_only,
            "signal_on": self.signal_on,
            "fill_on": self.fill_on,
            "position_mode": self.position_mode,
            "commission_rate": self.cost.commission_rate,
            "stamp_duty_rate": self.cost.stamp_duty_rate,
            "lot_size": self.cost.lot_size,
            "timezone": self.timezone,
            "benchmarks": self.benchmarks,
            "oos_start": self.resolved_oos_start().isoformat(),
            "oos_ratio": self.oos_ratio,
            "rolling_window_days": self.rolling_window_days,
            "extreme_days": self.extreme_days,
            "output_dir": self.output_dir,
            "config_path": self.config_path,
        }


def _copy_benchmarks() -> Dict[str, Dict[str, Any]]:
    return {
        key: {
            "name": spec["name"],
            "yahoo": spec["yahoo"],
            "fallbacks": list(spec.get("fallbacks") or []),
        }
        for key, spec in DEFAULT_BENCHMARKS.items()
    }


def _parse_date(value: Any, fallback: Optional[date] = None) -> Optional[date]:
    if value is None or value == "" or value == "today":
        return fallback
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def _normalize_strategy_block(strategy: Dict[str, Any]) -> Dict[str, Any]:
    """Support both new and legacy MA-cross YAML layouts."""
    block = dict(strategy or {})
    name = str(block.get("name") or "ma_cross").strip().lower()
    params = dict(block.get("params") or {})

    # Legacy: schemes / run_schemes at strategy root.
    if "schemes" in block and "schemes" not in params:
        params["schemes"] = block["schemes"]

    variants = block.get("variants")
    if variants is None:
        variants = block.get("run_schemes") or block.get("run_variants") or []

    return {
        "name": name,
        "params": params,
        "variants": list(variants) if variants else [],
        "long_only": block.get("long_only", True),
        "max_trades_per_day": block.get("max_trades_per_day", 2),
        "signal_on": block.get("signal_on", "close"),
        "fill_on": block.get("fill_on", "next_open"),
        "position_mode": block.get("position_mode", "all_in"),
    }


def load_backtest_yaml(path: Optional[str | Path] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"回测配置不存在: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"回测配置必须是 mapping: {cfg_path}")
    data["_config_path"] = str(cfg_path)
    return data


def backtest_config_from_mapping(
    data: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> BacktestConfig:
    universe = data.get("universe") or {}
    window = data.get("window") or {}
    capital = data.get("capital") or {}
    bars = data.get("bars") or {}
    strategy = _normalize_strategy_block(data.get("strategy") or {})
    costs = data.get("costs") or {}
    evaluation = data.get("evaluation") or {}
    output = data.get("output") or {}
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}

    raw_symbols = universe.get("symbols") or list(DEFAULT_SYMBOLS)
    symbols: List[str] = []
    names = dict(DEFAULT_SYMBOL_NAMES)
    lot_sizes = dict(DEFAULT_SYMBOL_LOT_SIZES)
    default_lot_size = int(overrides.get("lot_size", costs.get("lot_size", 1)))
    for item in raw_symbols:
        if isinstance(item, dict):
            code = str(item.get("code") or item.get("symbol") or "").strip()
            if not code:
                continue
            symbols.append(code)
            if item.get("name"):
                names[code] = str(item["name"])
            if item.get("lot_size") is not None:
                lot_sizes[code] = int(item["lot_size"])
            elif code not in lot_sizes:
                lot_sizes[code] = default_lot_size
        else:
            code = str(item).strip()
            symbols.append(code)
            if code not in lot_sizes:
                lot_sizes[code] = default_lot_size

    end = _parse_date(overrides.get("end", window.get("end")), date.today()) or date.today()
    lookback_years = float(overrides.get("lookback_years", window.get("lookback_years", 1)))
    start = _parse_date(overrides.get("start", window.get("start")))
    if start is None:
        start = end - timedelta(days=int(round(lookback_years * 365)))

    strategy_params = dict(strategy.get("params") or {})
    if overrides.get("strategy_params"):
        strategy_params.update(overrides["strategy_params"])

    if "variants" in overrides:
        variants = overrides["variants"]
    elif "schemes" in overrides:
        variants = overrides["schemes"]
    else:
        variants = strategy.get("variants")
    if variants:
        variants = list(variants)

    cfg = BacktestConfig(
        symbols=overrides.get("symbols", symbols),
        symbol_names=names,
        symbol_lot_sizes=lot_sizes,
        strategy_name=str(overrides.get("strategy", strategy.get("name", "ma_cross"))),
        strategy_params=strategy_params,
        variants=variants,
        interval=str(overrides.get("interval", bars.get("interval", "1d"))),
        start=start,
        end=end,
        lookback_years=lookback_years,
        capital=float(overrides.get("capital", capital.get("per_symbol", 300_000))),
        independent=bool(universe.get("independent", True)),
        max_trades_per_day=int(
            overrides.get("max_trades_per_day", strategy.get("max_trades_per_day", 2))
        ),
        long_only=bool(strategy.get("long_only", True)),
        signal_on=str(strategy.get("signal_on", "close")),
        fill_on=str(strategy.get("fill_on", "next_open")),
        position_mode=str(strategy.get("position_mode", "all_in")),
        cost=CostModel(
            commission_rate=float(overrides.get("commission", costs.get("commission_rate", 0.0003))),
            stamp_duty_rate=float(overrides.get("stamp", costs.get("stamp_duty_rate", 0.001))),
            lot_size=default_lot_size,
        ),
        timezone=str(window.get("timezone", "Asia/Hong_Kong")),
        benchmarks=data.get("benchmarks") or _copy_benchmarks(),
        oos_start=_parse_date(overrides.get("oos_start", evaluation.get("oos_start"))),
        oos_ratio=float(evaluation.get("oos_ratio", 0.3)),
        rolling_window_days=int(evaluation.get("rolling_window_days", 252)),
        extreme_days=int(evaluation.get("extreme_days", 5)),
        output_dir=str(overrides.get("output_dir", output.get("dir", "backtest_results"))),
        config_path=data.get("_config_path"),
    )
    return cfg


def load_backtest_config(
    path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> BacktestConfig:
    return backtest_config_from_mapping(load_backtest_yaml(path), overrides)
