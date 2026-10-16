"""趋势策略 YAML。只接受登记过的键，非法组合在加载时失败。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import yaml

from analysis.backtest.trend.markets import HK, US, ParsedSymbol, parse_symbol
from analysis.structure.config import enabled_methods
from analysis.structure.methods.phase import (
    ACCUMULATION,
    DISTRIBUTION,
    DOWN_BIRTH,
    DOWN_CONTINUE,
    DOWN_EXHAUSTION,
    DOWN_REBOUND,
    INSUFFICIENT,
    NEUTRAL,
    PENDING,
    UP_BIRTH,
    UP_CONTINUE,
    UP_EXHAUSTION,
    UP_PULLBACK,
)

SCHEMA_VERSION = "trend-strategy-v0"
DEFAULT_TREND_CONFIG_PATH = Path(__file__).resolve().parent.parent / "trend-strategy-v0.yaml"

_SCALES = ("short", "mid", "long")
_PRICE_AXES = ("log", "uniform")
_EXHAUSTION = ("reduce", "none")
_AUDITS = ("trades_and_cancels", "all_bars")
_CATEGORIES = frozenset({
    UP_BIRTH,
    UP_CONTINUE,
    UP_PULLBACK,
    UP_EXHAUSTION,
    DOWN_BIRTH,
    DOWN_CONTINUE,
    DOWN_REBOUND,
    DOWN_EXHAUSTION,
    ACCUMULATION,
    DISTRIBUTION,
    NEUTRAL,
    PENDING,
    INSUFFICIENT,
})


@dataclass(frozen=True)
class StructureSpec:
    param_version: str
    price_axis: str
    scale: str


@dataclass(frozen=True)
class PriceSpec:
    signal: str
    fill: str
    adjustment: str
    dividends: str


@dataclass(frozen=True)
class UnitSpec:
    kind: str
    span: int
    floor: float


@dataclass(frozen=True)
class HkMarket:
    currency: str
    timezone: str
    capital: float
    commission_rate: float
    min_commission: float
    platform_fee: float
    stamp_duty_rate: float
    exchange_fee_rate: float
    settlement_rate: float
    dividend_withholding: float
    benchmark: str


@dataclass(frozen=True)
class UsMarket:
    currency: str
    timezone: str
    capital: float
    lot_size: int
    commission_per_share: float
    min_commission: float
    max_commission_rate: float
    sec_fee_rate: float
    taf_per_share: float
    taf_max: float
    dividend_withholding: float
    benchmark: str


@dataclass(frozen=True)
class SymbolSpec:
    code: str
    name: str
    market: str
    lot_size: int
    lot_size_source: str
    dividend_withholding: float


@dataclass(frozen=True)
class WindowSpec:
    start: date
    end: date
    lookback_years: float
    warmup_bars: int
    interval: str


@dataclass(frozen=True)
class BreakoutEntry:
    lookback: int
    buffer: float
    stop_lookback: int
    stop_buffer: float


@dataclass(frozen=True)
class PullbackEntry:
    lookback: int
    break_lookback: int
    stop_buffer: float


@dataclass(frozen=True)
class TrendEntry:
    """策略 C。买入尺度在加载时必须与 structure.scale 相同。"""

    buy_scale: str
    sell_scale: str


@dataclass(frozen=True)
class D0Entry:
    """策略 D0。买卖用同一个尺度，加载时必须与 structure.scale 相同。"""

    scale: str


@dataclass(frozen=True)
class ScaleEntry:
    """策略 E。试仓尺度在加载时必须与 structure.scale 相同。"""

    scale: str


@dataclass(frozen=True)
class BreakEntry:
    """策略 F。交易尺度必须与 structure.scale 相同，参照尺度用来读中尺度突破。"""

    scale: str
    context_scale: str


@dataclass(frozen=True)
class BreakoutSpec:
    """向上突破和有效支撑各加一档风险。向下突破按尺度减仓或清仓。"""

    fresh_bars: int
    step: float
    range_break_risk: float
    short_down: int
    mid_down: int


@dataclass(frozen=True)
class OrderSpec:
    gap_cancel: float
    max_stop_distance: Optional[float]
    slippage_rate: float
    max_trades_per_day: int
    min_stop_distance: Optional[float] = None


@dataclass(frozen=True)
class RiskSpec:
    risk_per_trade: float
    one_lot_max_risk: float
    max_gross_exposure: float
    point_value: None
    sizing: str = "stop"
    max_open_risk: float = 0.0


@dataclass(frozen=True)
class ExitSpec:
    trail_buffer: float
    trail_states: Tuple[str, ...]
    reduce_fraction: float
    reduce_rounding: str
    exhaustion_action: str
    mode: str = "price"
    drawdown: Optional[float] = None
    bank_multiple: float = 0.0
    peel_on: Optional[str] = None
    flatten_categories: Tuple[str, ...] = ()
    extend_multiple: float = 0.0
    max_adds: int = 0
    add_floor: float = 0.0025
    second_category: str = ""


@dataclass(frozen=True)
class StrategySpec:
    name: str
    entry: BreakoutEntry | PullbackEntry | TrendEntry | D0Entry | ScaleEntry | BreakEntry
    order: OrderSpec
    risk: RiskSpec
    exit: ExitSpec
    breakout: Optional[BreakoutSpec] = None


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    strategy: str
    exhaustion_action: str


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    yahoo: str


@dataclass(frozen=True)
class EvaluationSpec:
    oos_ratio: float
    rolling_window_days: int
    extreme_days: int
    audit: str


def _plain_amount(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}"


@dataclass
class TrendStrategyConfig:
    version: str
    structure: StructureSpec
    prices: PriceSpec
    unit: UnitSpec
    markets: Dict[str, HkMarket | UsMarket]
    symbols: Tuple[SymbolSpec, ...]
    window: WindowSpec
    strategies: Dict[str, StrategySpec]
    benchmarks: Dict[str, BenchmarkSpec]
    evaluation: EvaluationSpec
    runs: Tuple[RunSpec, ...]
    output_dir: str
    config_path: Optional[str] = None

    def capital_text(self) -> str:
        hk = self.markets[HK]
        us = self.markets[US]
        return (
            f"港股 {_plain_amount(hk.capital)} {hk.currency}；"
            f"美股 {_plain_amount(us.capital)} {us.currency}"
        )

    def exit_for(self, run: RunSpec) -> ExitSpec:
        strategy = self.strategies[run.strategy]
        if run.exhaustion_action == strategy.exit.exhaustion_action:
            return strategy.exit
        return replace(strategy.exit, exhaustion_action=run.exhaustion_action)

    def to_dict(self) -> Dict[str, Any]:
        trade_caps = {self.strategies[run.strategy].order.max_trades_per_day for run in self.runs}
        trade_cap: Any = next(iter(trade_caps)) if len(trade_caps) == 1 else "按策略"
        hk = self.markets[HK]
        us = self.markets[US]
        return {
            "version": self.version,
            "config_path": self.config_path,
            "strategy": {
                "name": "trend_state",
                "params": {},
                "variants": [run.key for run in self.runs],
            },
            "interval": self.window.interval,
            "start": self.window.start.isoformat(),
            "end": self.window.end.isoformat(),
            "lookback_years": self.window.lookback_years,
            "timezone": f"{hk.timezone} / {us.timezone}",
            "capital": self.capital_text(),
            "max_trades_per_day": trade_cap,
            "signal_on": "close",
            "fill_on": "next_open",
            "position_mode": "分账户",
            "commission_rate": "按市场",
            "stamp_duty_rate": "按市场",
            "structure": {
                "param_version": self.structure.param_version,
                "price_axis": self.structure.price_axis,
                "scale": self.structure.scale,
            },
            "prices": {
                "signal": self.prices.signal,
                "fill": self.prices.fill,
                "adjustment": self.prices.adjustment,
                "dividends": self.prices.dividends,
            },
            "unit": {
                "kind": self.unit.kind,
                "span": self.unit.span,
                "floor": self.unit.floor,
            },
            "markets": {
                HK: asdict(hk),
                US: asdict(us),
            },
            "strategies": {
                name: {
                    "entry": asdict(spec.entry),
                    "order": asdict(spec.order),
                    "risk": asdict(spec.risk),
                    "exit": asdict(spec.exit),
                }
                for name, spec in self.strategies.items()
            },
            "symbols": [
                {
                    "code": item.code,
                    "name": item.name,
                    "market": item.market,
                    "lot_size": item.lot_size,
                    "lot_size_source": item.lot_size_source,
                    "dividend_withholding": item.dividend_withholding,
                    "currency": "HKD" if item.market == HK else "USD",
                }
                for item in self.symbols
            ],
            "symbol_names": {item.code: item.name for item in self.symbols},
            "benchmarks": {
                key: {"name": spec.name, "yahoo": spec.yahoo}
                for key, spec in self.benchmarks.items()
            },
            "evaluation": {
                "oos_ratio": self.evaluation.oos_ratio,
                "rolling_window_days": self.evaluation.rolling_window_days,
                "extreme_days": self.evaluation.extreme_days,
                "audit": self.evaluation.audit,
            },
            "runs": [
                {
                    "key": run.key,
                    "label": run.label,
                    "strategy": run.strategy,
                    "exhaustion_action": run.exhaustion_action,
                }
                for run in self.runs
            ],
            "output_dir": self.output_dir,
            "allow_short": False,
        }


def _exact(
    payload: Mapping[str, Any],
    required: Sequence[str],
    label: str,
    optional: Sequence[str] = (),
) -> None:
    allowed = set(required) | set(optional)
    extra = sorted(set(payload) - allowed)
    missing = [name for name in required if name not in payload]
    if "adds" in extra:
        raise ValueError(f"{label} 含有 adds。第一期不实现加仓")
    if extra:
        raise ValueError(f"{label} 有未登记的字段: {', '.join(extra)}")
    if missing:
        raise ValueError(f"{label} 缺少字段: {', '.join(missing)}")


def _mapping(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} 必须是 mapping")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} 必须是非空字符串")
    return value.strip()


def _int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} 必须是整数")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} 必须是数字")
    return float(value)


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} 必须是 true 或 false")
    return value


def _positive(value: float, label: str, *, allow_zero: bool = False) -> float:
    if allow_zero and value == 0:
        return value
    if value <= 0:
        raise ValueError(f"{label} 必须为正")
    return value


def _rate(value: float, label: str) -> float:
    if not 0 <= value <= 1:
        raise ValueError(f"{label} 必须在 [0, 1]")
    return value


def _date(value: Any, label: str) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if text.lower() in {"null", "none", "today"}:
        return None if text.lower() != "today" else date.today()
    try:
        return date.fromisoformat(text[:10])
    except ValueError as exc:
        raise ValueError(f"{label} 需要是 YYYY-MM-DD，收到 {value!r}") from exc


def _path_key(value: str, label: str) -> str:
    if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"{label} 不能作为目录名: {value!r}")
    return value


def load_trend_yaml(path: Optional[str | Path] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_TREND_CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"趋势回测配置不存在: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"趋势回测配置必须是 mapping: {cfg_path}")
    data["_config_path"] = str(cfg_path)
    return data


def load_trend_config(
    path: Optional[str | Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    *,
    today: Optional[date] = None,
    lot_lookup: Optional[Any] = None,
) -> TrendStrategyConfig:
    data = load_trend_yaml(path)
    source = str(data.pop("_config_path"))
    return trend_config_from_mapping(
        data, overrides=overrides, source=source, today=today, lot_lookup=lot_lookup,
    )


def trend_config_from_mapping(
    data: Mapping[str, Any],
    overrides: Optional[Mapping[str, Any]] = None,
    *,
    source: Optional[str] = None,
    today: Optional[date] = None,
    lot_lookup: Optional[Any] = None,
) -> TrendStrategyConfig:
    payload = dict(data)
    config_path = source or payload.pop("_config_path", None)
    payload.pop("_config_path", None)
    overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
    label = Path(config_path).name if config_path else "趋势回测配置"
    _exact(
        payload,
        (
            "version",
            "structure",
            "prices",
            "unit",
            "markets",
            "universe",
            "window",
            "strategies",
            "benchmarks",
            "evaluation",
            "runs",
            "output_dir",
        ),
        label,
    )
    version = _text(payload["version"], f"{label} 的 version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"{label} 的 version 必须是 {SCHEMA_VERSION}，收到 {version}")

    structure = _structure(payload["structure"])
    prices = _prices(payload["prices"])
    unit = _unit(payload["unit"])
    benchmarks = _benchmarks(payload["benchmarks"])
    markets = _markets(payload["markets"], benchmarks)
    symbols = _symbols(payload["universe"], markets)
    window = _window(payload["window"], today or date.today(), overrides or {})
    strategies = _strategies(payload["strategies"])
    _check_trend_scale(structure, strategies)
    _check_d0_scale(structure, strategies)
    _check_d1_scale(structure, strategies)
    _check_book_scale(structure, strategies)
    _check_break_scale(structure, strategies)
    runs = _runs(payload["runs"], strategies, overrides or {})
    symbols = _select_symbols(symbols, markets, overrides or {})
    symbols = _fill_missing_lots(symbols, lot_lookup)
    output_dir = _text(
        (overrides or {}).get("output_dir", payload["output_dir"]),
        "output_dir",
    )
    return TrendStrategyConfig(
        version=version,
        structure=structure,
        prices=prices,
        unit=unit,
        markets=markets,
        symbols=symbols,
        window=window,
        strategies=strategies,
        benchmarks=benchmarks,
        evaluation=_evaluation(payload["evaluation"]),
        runs=runs,
        output_dir=output_dir,
        config_path=str(config_path) if config_path else None,
    )


def _structure(value: Any) -> StructureSpec:
    row = _mapping(value, "structure")
    _exact(row, ("param_version", "price_axis", "scale"), "structure")
    version = _text(row["param_version"], "structure.param_version")
    methods = enabled_methods(version)
    if "phase" not in methods:
        raise ValueError(f"结构参数 {version} 没有启用 phase")
    axis = _text(row["price_axis"], "structure.price_axis").lower()
    if axis not in _PRICE_AXES:
        raise ValueError(f"structure.price_axis 必须是 log 或 uniform，收到 {axis}")
    scale = _text(row["scale"], "structure.scale").lower()
    if scale not in _SCALES:
        raise ValueError(f"structure.scale 必须是 short、mid、long 之一，收到 {scale}")
    return StructureSpec(version, axis, scale)


def _prices(value: Any) -> PriceSpec:
    row = _mapping(value, "prices")
    _exact(row, ("signal", "fill", "adjustment", "dividends"), "prices")
    signal = _text(row["signal"], "prices.signal")
    fill = _text(row["fill"], "prices.fill")
    adjustment = _text(row["adjustment"], "prices.adjustment")
    dividends = _text(row["dividends"], "prices.dividends")
    if (signal, fill, adjustment, dividends) != (
        "adjusted",
        "raw",
        "proportional_from_first_bar",
        "cash_on_ex_date",
    ):
        raise ValueError(
            "第一期 prices 只接受 signal: adjusted、fill: raw、"
            "adjustment: proportional_from_first_bar、dividends: cash_on_ex_date"
        )
    return PriceSpec(signal, fill, adjustment, dividends)


def _unit(value: Any) -> UnitSpec:
    row = _mapping(value, "unit")
    _exact(row, ("kind", "span", "floor"), "unit")
    kind = _text(row["kind"], "unit.kind")
    if kind != "log_true_range":
        raise ValueError(f"unit.kind 第一期只接受 log_true_range，收到 {kind}")
    span = _int(row["span"], "unit.span")
    if span < 1:
        raise ValueError("unit.span 必须 >= 1")
    floor = _positive(_number(row["floor"], "unit.floor"), "unit.floor")
    return UnitSpec(kind, span, floor)


def _benchmarks(value: Any) -> Dict[str, BenchmarkSpec]:
    row = _mapping(value, "benchmarks")
    if not row:
        raise ValueError("benchmarks 不能为空")
    specs: Dict[str, BenchmarkSpec] = {}
    for key, item in row.items():
        name = _text(str(key), "benchmarks 的键")
        block = _mapping(item, f"benchmarks.{name}")
        _exact(block, ("name", "yahoo"), f"benchmarks.{name}")
        specs[name] = BenchmarkSpec(
            _text(block["name"], f"benchmarks.{name}.name"),
            _text(block["yahoo"], f"benchmarks.{name}.yahoo"),
        )
    return specs


def _markets(value: Any, benchmarks: Mapping[str, BenchmarkSpec]) -> Dict[str, HkMarket | UsMarket]:
    row = _mapping(value, "markets")
    _exact(row, (HK, US), "markets")
    return {
        HK: _hk_market(row[HK], benchmarks),
        US: _us_market(row[US], benchmarks),
    }


def _hk_market(value: Any, benchmarks: Mapping[str, BenchmarkSpec]) -> HkMarket:
    row = _mapping(value, "markets.hk")
    _exact(
        row,
        (
            "currency",
            "timezone",
            "capital",
            "commission_rate",
            "min_commission",
            "platform_fee",
            "stamp_duty_rate",
            "exchange_fee_rate",
            "settlement_rate",
            "dividend_withholding",
            "benchmark",
        ),
        "markets.hk",
    )
    currency = _text(row["currency"], "markets.hk.currency")
    timezone = _text(row["timezone"], "markets.hk.timezone")
    if currency != "HKD":
        raise ValueError(f"markets.hk.currency 必须是 HKD，收到 {currency}")
    if timezone != "Asia/Hong_Kong":
        raise ValueError(f"markets.hk.timezone 必须是 Asia/Hong_Kong，收到 {timezone}")
    benchmark = _text(row["benchmark"], "markets.hk.benchmark")
    if benchmark not in benchmarks:
        raise ValueError(f"markets.hk.benchmark 未在 benchmarks 中登记: {benchmark}")
    return HkMarket(
        currency=currency,
        timezone=timezone,
        capital=_positive(_number(row["capital"], "markets.hk.capital"), "markets.hk.capital"),
        commission_rate=_rate(_number(row["commission_rate"], "markets.hk.commission_rate"), "markets.hk.commission_rate"),
        min_commission=_positive(_number(row["min_commission"], "markets.hk.min_commission"), "markets.hk.min_commission", allow_zero=True),
        platform_fee=_positive(_number(row["platform_fee"], "markets.hk.platform_fee"), "markets.hk.platform_fee", allow_zero=True),
        stamp_duty_rate=_rate(_number(row["stamp_duty_rate"], "markets.hk.stamp_duty_rate"), "markets.hk.stamp_duty_rate"),
        exchange_fee_rate=_rate(_number(row["exchange_fee_rate"], "markets.hk.exchange_fee_rate"), "markets.hk.exchange_fee_rate"),
        settlement_rate=_rate(_number(row["settlement_rate"], "markets.hk.settlement_rate"), "markets.hk.settlement_rate"),
        dividend_withholding=_rate(_number(row["dividend_withholding"], "markets.hk.dividend_withholding"), "markets.hk.dividend_withholding"),
        benchmark=benchmark,
    )


def _us_market(value: Any, benchmarks: Mapping[str, BenchmarkSpec]) -> UsMarket:
    row = _mapping(value, "markets.us")
    _exact(
        row,
        (
            "currency",
            "timezone",
            "capital",
            "lot_size",
            "commission_per_share",
            "min_commission",
            "max_commission_rate",
            "sec_fee_rate",
            "taf_per_share",
            "taf_max",
            "dividend_withholding",
            "benchmark",
        ),
        "markets.us",
    )
    currency = _text(row["currency"], "markets.us.currency")
    timezone = _text(row["timezone"], "markets.us.timezone")
    if currency != "USD":
        raise ValueError(f"markets.us.currency 必须是 USD，收到 {currency}")
    if timezone != "America/New_York":
        raise ValueError(f"markets.us.timezone 必须是 America/New_York，收到 {timezone}")
    lot_size = _int(row["lot_size"], "markets.us.lot_size")
    if lot_size != 1:
        raise ValueError(f"markets.us.lot_size 必须是 1，收到 {lot_size}")
    benchmark = _text(row["benchmark"], "markets.us.benchmark")
    if benchmark not in benchmarks:
        raise ValueError(f"markets.us.benchmark 未在 benchmarks 中登记: {benchmark}")
    return UsMarket(
        currency=currency,
        timezone=timezone,
        capital=_positive(_number(row["capital"], "markets.us.capital"), "markets.us.capital"),
        lot_size=lot_size,
        commission_per_share=_positive(
            _number(row["commission_per_share"], "markets.us.commission_per_share"),
            "markets.us.commission_per_share",
            allow_zero=True,
        ),
        min_commission=_positive(_number(row["min_commission"], "markets.us.min_commission"), "markets.us.min_commission", allow_zero=True),
        max_commission_rate=_rate(_number(row["max_commission_rate"], "markets.us.max_commission_rate"), "markets.us.max_commission_rate"),
        sec_fee_rate=_rate(_number(row["sec_fee_rate"], "markets.us.sec_fee_rate"), "markets.us.sec_fee_rate"),
        taf_per_share=_positive(_number(row["taf_per_share"], "markets.us.taf_per_share"), "markets.us.taf_per_share", allow_zero=True),
        taf_max=_positive(_number(row["taf_max"], "markets.us.taf_max"), "markets.us.taf_max", allow_zero=True),
        dividend_withholding=_rate(_number(row["dividend_withholding"], "markets.us.dividend_withholding"), "markets.us.dividend_withholding"),
        benchmark=benchmark,
    )


def _symbols(value: Any, markets: Mapping[str, HkMarket | UsMarket]) -> Tuple[SymbolSpec, ...]:
    row = _mapping(value, "universe")
    _exact(row, ("allow_short", "symbols"), "universe")
    if _bool(row["allow_short"], "universe.allow_short"):
        raise ValueError("universe.allow_short 必须为 false，第一期不做空")
    items = row["symbols"]
    if not isinstance(items, list) or not items:
        raise ValueError("universe.symbols 必须是非空列表")
    specs = []
    seen = set()
    for index, item in enumerate(items):
        specs.append(_symbol(item, index, markets))
        if specs[-1].code in seen:
            raise ValueError(f"股票代码重复: {specs[-1].code}")
        seen.add(specs[-1].code)
    return tuple(specs)


def _symbol(value: Any, index: int, markets: Mapping[str, HkMarket | UsMarket]) -> SymbolSpec:
    label = f"universe.symbols[{index}]"
    row = _mapping(value, label)
    _exact(row, ("code",), label, optional=("name", "lot_size", "dividend_withholding"))
    code_value = row["code"]
    if isinstance(code_value, (int, float, bool)):
        raise ValueError(f'{label} 的 code 必须是字符串。港股代码请加引号，例如 "00700"')
    parsed = parse_symbol(_text(code_value, f"{label}.code"))
    name = parsed.code
    if "name" in row and row["name"] is not None:
        name = _text(row["name"], f"{label}.name")
    market = markets[parsed.market]
    if "lot_size" in row and row["lot_size"] is not None:
        lot_size = _int(row["lot_size"], f"{label}.lot_size")
        if lot_size < 1:
            raise ValueError(f"{label}.lot_size 必须 >= 1")
        lot_source = "config"
    elif parsed.market == HK:
        lot_size = 0
        lot_source = "missing"
    else:
        if not isinstance(market, UsMarket):
            raise ValueError(f"{parsed.code} 缺少美股市场配置")
        lot_size = market.lot_size
        lot_source = "market_default"
    withholding = market.dividend_withholding
    if "dividend_withholding" in row and row["dividend_withholding"] is not None:
        withholding = _rate(
            _number(row["dividend_withholding"], f"{label}.dividend_withholding"),
            f"{label}.dividend_withholding",
        )
    return SymbolSpec(parsed.code, name, parsed.market, lot_size, lot_source, withholding)


def _window(value: Any, today: date, overrides: Mapping[str, Any]) -> WindowSpec:
    row = _mapping(value, "window")
    _exact(row, ("start", "end", "lookback_years", "warmup_bars", "interval"), "window")
    interval = _text(row["interval"], "window.interval").lower()
    if interval != "1d":
        raise ValueError(f"window.interval 第一期只接受 1d，收到 {interval}")
    lookback = _positive(_number(row["lookback_years"], "window.lookback_years"), "window.lookback_years")
    if "lookback_years" in overrides and overrides["lookback_years"] is not None:
        lookback = _positive(_number(overrides["lookback_years"], "--years"), "--years")
    warmup = _int(row["warmup_bars"], "window.warmup_bars")
    if warmup < 30:
        raise ValueError("window.warmup_bars 必须 >= 30")

    yaml_start = _date(row["start"], "window.start")
    cli_start = _date(overrides["start"], "--start") if "start" in overrides and overrides["start"] is not None else None
    years_overridden = "lookback_years" in overrides and overrides["lookback_years"] is not None
    if cli_start is not None and years_overridden:
        raise ValueError("起止日期和回看年数请分开使用")
    if yaml_start is not None and years_overridden and cli_start is None:
        raise ValueError("已写 start 时不要再用 --years")

    end = _date(overrides.get("end", row["end"]), "window.end") or today
    if cli_start is not None:
        start = cli_start
    elif years_overridden or yaml_start is None:
        start = end - timedelta(days=int(round(lookback * 365)))
    else:
        start = yaml_start
    if start is None:
        raise ValueError("window.start 无法确定")
    if end < start:
        raise ValueError("window.end 不能早于 window.start")
    return WindowSpec(start, end, lookback, warmup, interval)


def _strategies(value: Any) -> Dict[str, StrategySpec]:
    row = _mapping(value, "strategies")
    if set(row) == {"scale"}:
        return {"scale": _strategy("scale", row["scale"])}
    if set(row) == {"break"}:
        return {"break": _strategy("break", row["break"])}
    _exact(row, ("breakout", "pullback"), "strategies", optional=("trend", "d0", "d1", "scale", "break"))
    specs = {
        "breakout": _strategy("breakout", row["breakout"]),
        "pullback": _strategy("pullback", row["pullback"]),
    }
    if "trend" in row and row["trend"] is not None:
        specs["trend"] = _strategy("trend", row["trend"])
    if "d0" in row and row["d0"] is not None:
        specs["d0"] = _strategy("d0", row["d0"])
    if "d1" in row and row["d1"] is not None:
        specs["d1"] = _strategy("d1", row["d1"])
    if "scale" in row and row["scale"] is not None:
        specs["scale"] = _strategy("scale", row["scale"])
    if "break" in row and row["break"] is not None:
        specs["break"] = _strategy("break", row["break"])
    return specs


def _strategy(name: str, value: Any) -> StrategySpec:
    label = f"strategies.{name}"
    row = _mapping(value, label)
    if name == "scale":
        _exact(row, ("entry", "order", "risk", "exit", "add"), label)
    elif name == "break":
        _exact(row, ("entry", "order", "risk", "exit", "add", "breakout"), label)
    else:
        _exact(row, ("entry", "order", "risk", "exit"), label)
    if name == "breakout":
        entry: BreakoutEntry | PullbackEntry | TrendEntry | D0Entry | ScaleEntry | BreakEntry = _breakout_entry(row["entry"])
        order, risk, exit_spec = _order(row["order"], name), _risk(row["risk"], name), _exit(row["exit"], name)
    elif name == "pullback":
        entry = _pullback_entry(row["entry"])
        order, risk, exit_spec = _order(row["order"], name), _risk(row["risk"], name), _exit(row["exit"], name)
    elif name == "trend":
        entry = _trend_entry(row["entry"])
        order, risk, exit_spec = _exposure_order(row["order"], name), _exposure_risk(row["risk"], name), _scale_exit(row["exit"], name)
    elif name == "d0":
        entry = _d0_entry(row["entry"], name)
        order, risk, exit_spec = _exposure_order(row["order"], name), _exposure_risk(row["risk"], name), _scale_exit(row["exit"], name)
    elif name == "d1":
        entry = _d0_entry(row["entry"], name)
        order, risk, exit_spec = _exposure_order(row["order"], name), _exposure_risk(row["risk"], name), _scale_exit(row["exit"], name, drawdown=True)
    elif name == "scale":
        entry = _scale_entry(row["entry"])
        order, risk, exit_spec = _scale_order(row["order"]), _scale_risk(row["risk"]), _book_exit(row["exit"], row["add"])
        return StrategySpec(name, entry, order, risk, exit_spec)
    elif name == "break":
        entry = _break_entry(row["entry"])
        order = _scale_order(row["order"], name)
        risk = _scale_risk(row["risk"], name)
        spec = _breakout_spec(row["breakout"], risk)
        exit_spec = _book_exit(row["exit"], row["add"], name, add_floor=spec.step)
        _check_break_budget(risk, spec, name)
        return StrategySpec(name, entry, order, risk, exit_spec, spec)
    else:
        raise ValueError(f"未知策略 {name}")
    return StrategySpec(name, entry, order, risk, exit_spec)


def _check_trend_scale(structure: StructureSpec, strategies: Mapping[str, StrategySpec]) -> None:
    trend = strategies.get("trend")
    if trend is None:
        return
    entry = trend.entry
    if not isinstance(entry, TrendEntry):
        raise ValueError("strategies.trend 的入场参数类型不对")
    if entry.buy_scale != structure.scale:
        raise ValueError(
            "策略 C 的买入尺度必须与 structure.scale 相同，"
            f"收到 {entry.buy_scale} 和 {structure.scale}"
        )


def _check_d0_scale(structure: StructureSpec, strategies: Mapping[str, StrategySpec]) -> None:
    _check_same_scale(structure, strategies, "d0", "策略 D0")


def _check_d1_scale(structure: StructureSpec, strategies: Mapping[str, StrategySpec]) -> None:
    _check_same_scale(structure, strategies, "d1", "策略 D1")


def _check_break_scale(structure: StructureSpec, strategies: Mapping[str, StrategySpec]) -> None:
    strategy = strategies.get("break")
    if strategy is None:
        return
    entry = strategy.entry
    if not isinstance(entry, BreakEntry):
        raise ValueError("strategies.break 的入场参数类型不对")
    if entry.scale != structure.scale:
        raise ValueError(
            "策略 F 的尺度必须与 structure.scale 相同，"
            f"收到 {entry.scale} 和 {structure.scale}"
        )


def _check_book_scale(structure: StructureSpec, strategies: Mapping[str, StrategySpec]) -> None:
    strategy = strategies.get("scale")
    if strategy is None:
        return
    entry = strategy.entry
    if not isinstance(entry, ScaleEntry):
        raise ValueError("strategies.scale 的入场参数类型不对")
    if entry.scale != structure.scale:
        raise ValueError(
            "策略 E 的尺度必须与 structure.scale 相同，"
            f"收到 {entry.scale} 和 {structure.scale}"
        )


def _check_same_scale(
    structure: StructureSpec,
    strategies: Mapping[str, StrategySpec],
    key: str,
    title: str,
) -> None:
    strategy = strategies.get(key)
    if strategy is None:
        return
    entry = strategy.entry
    if not isinstance(entry, D0Entry):
        raise ValueError(f"strategies.{key} 的入场参数类型不对")
    if entry.scale != structure.scale:
        raise ValueError(
            f"{title} 的尺度必须与 structure.scale 相同，"
            f"收到 {entry.scale} 和 {structure.scale}"
        )


def _breakout_entry(value: Any) -> BreakoutEntry:
    row = _mapping(value, "strategies.breakout.entry")
    _exact(row, ("lookback", "buffer", "stop_lookback", "stop_buffer"), "strategies.breakout.entry")
    lookback = _int(row["lookback"], "strategies.breakout.entry.lookback")
    stop_lookback = _int(row["stop_lookback"], "strategies.breakout.entry.stop_lookback")
    if lookback < 1:
        raise ValueError("strategies.breakout.entry.lookback 必须 >= 1")
    if stop_lookback < 0:
        raise ValueError("strategies.breakout.entry.stop_lookback 必须 >= 0")
    return BreakoutEntry(
        lookback,
        _positive(_number(row["buffer"], "strategies.breakout.entry.buffer"), "strategies.breakout.entry.buffer", allow_zero=True),
        stop_lookback,
        _positive(_number(row["stop_buffer"], "strategies.breakout.entry.stop_buffer"), "strategies.breakout.entry.stop_buffer", allow_zero=True),
    )


def _pullback_entry(value: Any) -> PullbackEntry:
    row = _mapping(value, "strategies.pullback.entry")
    _exact(row, ("lookback", "break_lookback", "stop_buffer"), "strategies.pullback.entry")
    lookback = _int(row["lookback"], "strategies.pullback.entry.lookback")
    break_lookback = _int(row["break_lookback"], "strategies.pullback.entry.break_lookback")
    if lookback < 1 or break_lookback < 1:
        raise ValueError("strategies.pullback.entry 的 lookback 和 break_lookback 必须 >= 1")
    return PullbackEntry(
        lookback,
        break_lookback,
        _positive(_number(row["stop_buffer"], "strategies.pullback.entry.stop_buffer"), "strategies.pullback.entry.stop_buffer", allow_zero=True),
    )


def _trend_entry(value: Any) -> TrendEntry:
    label = "strategies.trend.entry"
    row = _mapping(value, label)
    _exact(row, ("buy_scale", "sell_scale"), label)
    buy = _scale(row["buy_scale"], f"{label}.buy_scale")
    sell = _scale(row["sell_scale"], f"{label}.sell_scale")
    if buy == sell:
        raise ValueError("策略 C 的买入尺度和卖出尺度不能相同")
    return TrendEntry(buy, sell)


def _scale(value: Any, label: str) -> str:
    scale = _text(value, label).lower()
    if scale not in _SCALES:
        raise ValueError(f"{label} 必须是 short、mid、long 之一，收到 {scale}")
    return scale


def _scale_entry(value: Any) -> ScaleEntry:
    label = "strategies.scale.entry"
    row = _mapping(value, label)
    _exact(row, ("scale",), label)
    return ScaleEntry(_scale(row["scale"], f"{label}.scale"))


def _break_entry(value: Any) -> BreakEntry:
    label = "strategies.break.entry"
    row = _mapping(value, label)
    _exact(row, ("scale", "context_scale"), label)
    scale = _scale(row["scale"], f"{label}.scale")
    context = _scale(row["context_scale"], f"{label}.context_scale")
    if context == scale:
        raise ValueError(f"{label}.context_scale 不能与 scale 相同")
    return BreakEntry(scale, context)


def _scale_order(value: Any, strategy: str = "scale") -> OrderSpec:
    label = f"strategies.{strategy}.order"
    row = _mapping(value, label)
    _exact(row, ("gap_cancel", "min_stop_distance", "slippage_rate", "max_trades_per_day"), label)
    trades = _int(row["max_trades_per_day"], f"{label}.max_trades_per_day")
    if trades < 1:
        raise ValueError(f"{label}.max_trades_per_day 必须 >= 1")
    return OrderSpec(
        _positive(_number(row["gap_cancel"], f"{label}.gap_cancel"), f"{label}.gap_cancel"),
        None,
        _positive(_number(row["slippage_rate"], f"{label}.slippage_rate"), f"{label}.slippage_rate", allow_zero=True),
        trades,
        _positive(_number(row["min_stop_distance"], f"{label}.min_stop_distance"), f"{label}.min_stop_distance"),
    )


def _scale_risk(value: Any, strategy: str = "scale") -> RiskSpec:
    label = f"strategies.{strategy}.risk"
    row = _mapping(value, label)
    _exact(
        row,
        ("risk_per_tranche", "max_open_risk", "max_gross_exposure", "one_lot_max_risk"),
        label,
    )
    tranche = _rate(_number(row["risk_per_tranche"], f"{label}.risk_per_tranche"), f"{label}.risk_per_tranche")
    cap = _rate(_number(row["max_open_risk"], f"{label}.max_open_risk"), f"{label}.max_open_risk")
    if tranche <= 0 or cap <= 0:
        raise ValueError(f"{label} 的风险必须为正")
    if cap < tranche:
        raise ValueError(f"{label}.max_open_risk 不得小于 risk_per_tranche")
    one_lot = _rate(_number(row["one_lot_max_risk"], f"{label}.one_lot_max_risk"), f"{label}.one_lot_max_risk")
    if one_lot != 0 and one_lot < tranche:
        raise ValueError(f"{label}.one_lot_max_risk 不为 0 时不得小于 risk_per_tranche")
    exposure = _number(row["max_gross_exposure"], f"{label}.max_gross_exposure")
    if not 0 < exposure <= 1:
        raise ValueError(f"{label}.max_gross_exposure 必须在 (0, 1]，第一期不使用杠杆")
    return RiskSpec(tranche, one_lot, exposure, None, sizing="stop", max_open_risk=cap)


def _breakout_spec(value: Any, risk: RiskSpec) -> BreakoutSpec:
    label = "strategies.break.breakout"
    row = _mapping(value, label)
    _exact(row, ("fresh_bars", "step", "range_break_risk", "short_down", "mid_down"), label)
    fresh = _int(row["fresh_bars"], f"{label}.fresh_bars")
    if fresh < 1:
        raise ValueError(f"{label}.fresh_bars 至少为 1，收到 {fresh}")
    step = _rate(_number(row["step"], f"{label}.step"), f"{label}.step")
    if step <= 0:
        raise ValueError(f"{label}.step 必须为正")
    range_risk = _rate(_number(row["range_break_risk"], f"{label}.range_break_risk"), f"{label}.range_break_risk")
    if range_risk <= 0:
        raise ValueError(f"{label}.range_break_risk 必须为正")
    if range_risk > risk.max_open_risk:
        raise ValueError(f"{label}.range_break_risk 不得超过 max_open_risk")
    short_down = _int(row["short_down"], f"{label}.short_down")
    mid_down = _int(row["mid_down"], f"{label}.mid_down")
    if short_down != 2:
        raise ValueError(f"{label}.short_down 只接受 2，收到 {short_down}")
    if mid_down != 3:
        raise ValueError(f"{label}.mid_down 只接受 3，收到 {mid_down}")
    return BreakoutSpec(fresh, step, range_risk, short_down, mid_down)


def _check_break_budget(risk: RiskSpec, spec: BreakoutSpec, strategy: str) -> None:
    full = risk.risk_per_trade + 2 * spec.step
    if full > risk.max_open_risk + 1e-12:
        raise ValueError(
            f"strategies.{strategy} 的试仓加上两次突破增量 {full} 超过 max_open_risk {risk.max_open_risk}"
        )


def _book_exit(exit_value: Any, add_value: Any, strategy: str = "scale", add_floor: float = 0.0025) -> ExitSpec:
    label = f"strategies.{strategy}.exit"
    row = _mapping(exit_value, label)
    _exact(row, ("trail_multiple", "trail_categories", "bank_multiple", "peel_on", "flatten_categories"), label)
    trail = _positive(_number(row["trail_multiple"], f"{label}.trail_multiple"), f"{label}.trail_multiple")
    categories = _category_list(row["trail_categories"], f"{label}.trail_categories")
    if UP_BIRTH in categories:
        raise ValueError(f"{label}.trail_categories 不得包含 up_birth")
    bank = _positive(_number(row["bank_multiple"], f"{label}.bank_multiple"), f"{label}.bank_multiple")
    peel = _text(row["peel_on"], f"{label}.peel_on")
    if peel not in _CATEGORIES:
        raise ValueError(f"{label}.peel_on 有未知主状态 {peel}")
    flatten = _category_list(row["flatten_categories"], f"{label}.flatten_categories")
    add_label = f"strategies.{strategy}.add"
    add = _mapping(add_value, add_label)
    _exact(add, ("max_adds", "first", "second_category", "extend_multiple"), add_label)
    adds = _int(add["max_adds"], f"{add_label}.max_adds")
    if adds != 2:
        raise ValueError(f"{add_label}.max_adds 这一版只接受 2，收到 {adds}")
    first = _text(add["first"], f"{add_label}.first")
    if first != "pullback_to_continue":
        raise ValueError(f"{add_label}.first 只接受 pullback_to_continue，收到 {first}")
    second = _text(add["second_category"], f"{add_label}.second_category")
    if second not in _CATEGORIES:
        raise ValueError(f"{add_label}.second_category 有未知主状态 {second}")
    extend = _positive(_number(add["extend_multiple"], f"{add_label}.extend_multiple"), f"{add_label}.extend_multiple")
    return ExitSpec(
        trail,
        tuple(categories),
        1.0,
        "floor_lot_else_flatten",
        "none",
        mode="break_book" if strategy == "break" else "scale_book",
        bank_multiple=bank,
        peel_on=peel,
        flatten_categories=tuple(flatten),
        extend_multiple=extend,
        max_adds=adds,
        add_floor=add_floor,
        second_category=second,
    )


def _category_list(value: Any, label: str) -> list:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} 必须是非空列表")
    cleaned = []
    for item in value:
        name = _text(item, label)
        if name not in _CATEGORIES:
            raise ValueError(f"{label} 有未知主状态 {name}")
        if name not in cleaned:
            cleaned.append(name)
    return cleaned


def _d0_entry(value: Any, strategy: str = "d0") -> D0Entry:
    label = f"strategies.{strategy}.entry"
    row = _mapping(value, label)
    _exact(row, ("scale",), label)
    return D0Entry(_scale(row["scale"], f"{label}.scale"))


def _exposure_order(value: Any, strategy: str) -> OrderSpec:
    label = f"strategies.{strategy}.order"
    row = _mapping(value, label)
    _exact(row, ("gap_cancel", "slippage_rate", "max_trades_per_day"), label)
    trades = _int(row["max_trades_per_day"], f"{label}.max_trades_per_day")
    if trades < 1:
        raise ValueError(f"{label}.max_trades_per_day 必须 >= 1")
    return OrderSpec(
        _positive(_number(row["gap_cancel"], f"{label}.gap_cancel"), f"{label}.gap_cancel"),
        None,
        _positive(_number(row["slippage_rate"], f"{label}.slippage_rate"), f"{label}.slippage_rate", allow_zero=True),
        trades,
    )


def _exposure_risk(value: Any, strategy: str) -> RiskSpec:
    label = f"strategies.{strategy}.risk"
    row = _mapping(value, label)
    _exact(row, ("sizing", "max_gross_exposure", "point_value"), label)
    if row["point_value"] is not None:
        raise ValueError(f"{label}.point_value 必须为空，第一期不支持期货每点价值")
    sizing = _text(row["sizing"], f"{label}.sizing")
    if sizing != "exposure":
        raise ValueError(f"{label}.sizing 只接受 exposure，收到 {sizing}")
    exposure = _number(row["max_gross_exposure"], f"{label}.max_gross_exposure")
    if not 0 < exposure <= 1:
        raise ValueError(f"{label}.max_gross_exposure 必须在 (0, 1]，第一期不使用杠杆")
    return RiskSpec(0.0, 0.0, exposure, None, sizing="exposure")


def _scale_exit(value: Any, strategy: str, *, drawdown: bool = False) -> ExitSpec:
    label = f"strategies.{strategy}.exit"
    row = _mapping(value, label)
    if drawdown:
        _exact(row, ("mode", "drawdown"), label)
    else:
        _exact(row, ("mode",), label)
    mode = _text(row["mode"], f"{label}.mode")
    if mode != "scale_downtrend":
        raise ValueError(f"{label}.mode 只接受 scale_downtrend，收到 {mode}")
    fraction = None
    if drawdown:
        fraction = _number(row["drawdown"], f"{label}.drawdown")
        if not 0 < fraction < 1:
            raise ValueError(f"{label}.drawdown 必须在 (0, 1)，收到 {fraction}")
    return ExitSpec(0.0, (), 0.5, "floor_lot_else_flatten", "none", mode="scale_downtrend", drawdown=fraction)


def _order(value: Any, strategy: str) -> OrderSpec:
    label = f"strategies.{strategy}.order"
    row = _mapping(value, label)
    _exact(row, ("gap_cancel", "max_stop_distance", "slippage_rate", "max_trades_per_day"), label)
    trades = _int(row["max_trades_per_day"], f"{label}.max_trades_per_day")
    if trades < 1:
        raise ValueError(f"{label}.max_trades_per_day 必须 >= 1")
    return OrderSpec(
        _positive(_number(row["gap_cancel"], f"{label}.gap_cancel"), f"{label}.gap_cancel"),
        _positive(_number(row["max_stop_distance"], f"{label}.max_stop_distance"), f"{label}.max_stop_distance"),
        _positive(_number(row["slippage_rate"], f"{label}.slippage_rate"), f"{label}.slippage_rate", allow_zero=True),
        trades,
    )


def _risk(value: Any, strategy: str) -> RiskSpec:
    label = f"strategies.{strategy}.risk"
    row = _mapping(value, label)
    _exact(row, ("risk_per_trade", "one_lot_max_risk", "max_gross_exposure", "point_value"), label)
    if row["point_value"] is not None:
        raise ValueError(f"{label}.point_value 必须为空，第一期不支持期货每点价值")
    risk = _rate(_number(row["risk_per_trade"], f"{label}.risk_per_trade"), f"{label}.risk_per_trade")
    if risk <= 0:
        raise ValueError(f"{label}.risk_per_trade 必须为正")
    one_lot = _rate(_number(row["one_lot_max_risk"], f"{label}.one_lot_max_risk"), f"{label}.one_lot_max_risk")
    if one_lot != 0 and one_lot < risk:
        raise ValueError(f"{label}.one_lot_max_risk 不为 0 时不得小于 risk_per_trade")
    exposure = _number(row["max_gross_exposure"], f"{label}.max_gross_exposure")
    if not 0 < exposure <= 1:
        raise ValueError(f"{label}.max_gross_exposure 必须在 (0, 1]，第一期不使用杠杆")
    return RiskSpec(risk, one_lot, exposure, None, sizing="stop")


def _exit(value: Any, strategy: str) -> ExitSpec:
    label = f"strategies.{strategy}.exit"
    row = _mapping(value, label)
    _exact(
        row,
        ("trail_buffer", "trail_states", "reduce_fraction", "reduce_rounding", "exhaustion_action"),
        label,
    )
    states = row["trail_states"]
    if not isinstance(states, list) or not states:
        raise ValueError(f"{label}.trail_states 必须是非空列表")
    cleaned = []
    for item in states:
        name = _text(item, f"{label}.trail_states")
        if name not in _CATEGORIES:
            raise ValueError(f"{label}.trail_states 有未知主状态 {name}")
        if name not in cleaned:
            cleaned.append(name)
    rounding = _text(row["reduce_rounding"], f"{label}.reduce_rounding")
    if rounding != "floor_lot_else_flatten":
        raise ValueError(f"{label}.reduce_rounding 只接受 floor_lot_else_flatten，收到 {rounding}")
    fraction = _number(row["reduce_fraction"], f"{label}.reduce_fraction")
    if not 0 < fraction <= 1:
        raise ValueError(f"{label}.reduce_fraction 必须在 (0, 1]")
    return ExitSpec(
        _positive(_number(row["trail_buffer"], f"{label}.trail_buffer"), f"{label}.trail_buffer"),
        tuple(cleaned),
        fraction,
        rounding,
        _exhaustion(row["exhaustion_action"], f"{label}.exhaustion_action"),
        mode="price",
    )


def _exhaustion(value: Any, label: str) -> str:
    action = _text(value, label)
    if action not in _EXHAUSTION:
        raise ValueError(f"{label} 只接受 reduce 或 none，收到 {action}")
    return action


def _runs(
    value: Any,
    strategies: Mapping[str, StrategySpec],
    overrides: Mapping[str, Any],
) -> Tuple[RunSpec, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("runs 必须是非空列表")
    specs = []
    seen = set()
    for index, item in enumerate(value):
        label = f"runs[{index}]"
        row = _mapping(item, label)
        extra = sorted(set(row) - {"key", "label", "strategy", "exhaustion_action"})
        if "adds" in extra:
            raise ValueError(f"{label} 含有 adds。第一期不实现加仓")
        if extra:
            raise ValueError(f"{label} 只能覆盖 exhaustion_action，收到 {', '.join(extra)}")
        _exact(row, ("key", "label", "strategy"), label, optional=("exhaustion_action",))
        key = _path_key(_text(row["key"], f"{label}.key"), f"{label}.key")
        if key in seen:
            raise ValueError(f"run 重复: {key}")
        seen.add(key)
        strategy_name = _text(row["strategy"], f"{label}.strategy")
        if strategy_name not in strategies:
            raise ValueError(f"run {key} 的策略 {strategy_name} 不在 strategies 里")
        action = strategies[strategy_name].exit.exhaustion_action
        if "exhaustion_action" in row and row["exhaustion_action"] is not None:
            action = _exhaustion(row["exhaustion_action"], f"runs.{key}.exhaustion_action")
        if strategies[strategy_name].exit.mode == "scale_downtrend" and action != "none":
            raise ValueError(f"run {key} 不设价格止损，不能减仓")
        specs.append(RunSpec(key, _text(row["label"], f"{label}.label"), strategy_name, action))

    selected = overrides.get("runs") if overrides else None
    if selected is None:
        return tuple(specs)
    if isinstance(selected, str):
        wanted = [part.strip() for part in selected.split(",") if part.strip()]
    else:
        wanted = [str(part).strip() for part in selected if str(part).strip()]
    if not wanted:
        raise ValueError("runs 不能为空")
    by_key = {item.key: item for item in specs}
    missing = [key for key in wanted if key not in by_key]
    if missing:
        raise ValueError(f"未知 run: {', '.join(missing)}")
    return tuple(by_key[key] for key in wanted)


def read_symbol_file(path: str | Path) -> Tuple[str, ...]:
    """读股票名单。一行一个代码，空行和 # 注释忽略，重复的只留第一次。"""
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"找不到股票列表: {file}")
    ordered = []
    seen = set()
    for line in file.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        for piece in text.replace("，", ",").replace("、", ",").split(","):
            code = piece.strip()
            if not code or code in seen:
                continue
            seen.add(code)
            ordered.append(code)
    if not ordered:
        raise ValueError(f"{file} 里没有股票代码")
    return tuple(ordered)


def _fill_missing_lots(symbols: Sequence[SymbolSpec], lookup: Optional[Any]) -> Tuple[SymbolSpec, ...]:
    missing = [item for item in symbols if item.lot_size_source == "missing"]
    if not missing:
        return tuple(symbols)
    found: Mapping[str, Any] = {}
    if lookup is not None:
        found = lookup([item.code for item in missing]) or {}
    filled = []
    absent = []
    for spec in symbols:
        if spec.lot_size_source != "missing":
            filled.append(spec)
            continue
        row = found.get(spec.code)
        lot, name = _lot_row(row)
        if lot is None:
            absent.append(spec.code)
            continue
        filled.append(replace(
            spec,
            name=name if name and spec.name == spec.code else spec.name,
            lot_size=lot,
            lot_size_source="longbridge",
        ))
    if absent:
        joined = ", ".join(absent)
        raise ValueError(f"这些港股没有 lot_size，Longbridge 也没有返回：{joined}")
    return tuple(filled)


def _lot_row(row: Any) -> Tuple[Optional[int], Optional[str]]:
    if not row:
        return None, None
    lot = row[0]
    name = row[1] if len(row) > 1 else None
    if isinstance(lot, bool) or not isinstance(lot, int) or lot < 1:
        return None, None
    text = str(name).strip() if name else None
    return lot, text or None


def _select_symbols(
    symbols: Sequence[SymbolSpec],
    markets: Mapping[str, HkMarket | UsMarket],
    overrides: Mapping[str, Any],
) -> Tuple[SymbolSpec, ...]:
    selected = overrides.get("symbols") if overrides else None
    if selected is None:
        return tuple(symbols)
    if isinstance(selected, str):
        raw_codes = [part.strip() for part in selected.split(",") if part.strip()]
    else:
        raw_codes = [str(part).strip() for part in selected if str(part).strip()]
    if not raw_codes:
        raise ValueError("symbols 不能为空")
    by_code = {item.code: item for item in symbols}
    chosen = []
    seen = set()
    for raw in raw_codes:
        parsed = parse_symbol(raw)
        if parsed.code in seen:
            continue
        seen.add(parsed.code)
        if parsed.code in by_code:
            chosen.append(by_code[parsed.code])
            continue
        if parsed.market == HK:
            chosen.append(SymbolSpec(
                parsed.code,
                parsed.code,
                HK,
                0,
                "missing",
                markets[HK].dividend_withholding,
            ))
            continue
        chosen.append(
            SymbolSpec(
                parsed.code,
                parsed.code,
                US,
                markets[US].lot_size,
                "market_default",
                markets[US].dividend_withholding,
            )
        )
    return tuple(chosen)


def _evaluation(value: Any) -> EvaluationSpec:
    row = _mapping(value, "evaluation")
    _exact(row, ("oos_ratio", "rolling_window_days", "extreme_days", "audit"), "evaluation")
    ratio = _number(row["oos_ratio"], "evaluation.oos_ratio")
    if not 0 <= ratio < 1:
        raise ValueError("evaluation.oos_ratio 必须在 [0, 1)")
    window = _int(row["rolling_window_days"], "evaluation.rolling_window_days")
    if window < 20:
        raise ValueError("evaluation.rolling_window_days 必须 >= 20")
    extreme = _int(row["extreme_days"], "evaluation.extreme_days")
    if extreme < 1:
        raise ValueError("evaluation.extreme_days 必须 >= 1")
    audit = _text(row["audit"], "evaluation.audit")
    if audit not in _AUDITS:
        raise ValueError(f"evaluation.audit 必须是 trades_and_cancels 或 all_bars，收到 {audit}")
    return EvaluationSpec(ratio, window, extreme, audit)
