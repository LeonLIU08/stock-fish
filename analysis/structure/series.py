"""标准序列：序号、收盘价、几何价格、间断和数据身份。

几何横轴是有效 K 线序号，不把停牌或缺失填成横盘。
v0 到 v3 的 `data_hash` 只覆盖时间戳和收盘价。启用停顿的版本在声明了
最高价、最低价和成交量时，把这三项一并算进哈希。启用阶段的版本若另外传入开盘价，
也把开盘价算进哈希；没传入时哈希不变。参数版本和价格轴放在输出目录里。
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import struct
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from analysis.structure.config import (
    DAILY_INTERVAL,
    INTERVAL_SECONDS,
    PARAM_VERSION,
    PRICE_BASIS_CLOSE,
    StructureParams,
    default_params,
    normalize_interval,
    normalize_price_axis,
    series_quote_fields,
    series_uses_open,
)

IDENTITY_FILENAME = "series_identity.json"


class NonPositivePriceError(ValueError):
    """收盘价不是有限正数。调用方应整段拒绝，不能删掉这根再往前填。"""

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = tuple(indices)
        self.count = len(self.indices)
        preview = ", ".join(str(index) for index in self.indices[:8])
        suffix = "" if self.count <= 8 else ", ..."
        super().__init__(f"拒绝非正收盘价 {self.count} 根，序号 {preview}{suffix}")


@dataclass(frozen=True)
class SeriesBar:
    index: int
    timestamp: datetime
    close: float
    geometric: float
    gap_before: bool
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None
    high_geometric: Optional[float] = None
    low_geometric: Optional[float] = None
    open: Optional[float] = None
    open_geometric: Optional[float] = None


@dataclass(frozen=True)
class StandardSeries:
    bars: Tuple[SeriesBar, ...]
    price_basis: str
    price_axis: str
    interval: str
    param_version: str
    data_hash: str
    params: StructureParams

    def __len__(self) -> int:
        return len(self.bars)

    def __post_init__(self) -> None:
        if self.price_basis != PRICE_BASIS_CLOSE:
            raise ValueError(f"price_basis 固定为 {PRICE_BASIS_CLOSE}")
        if self.price_axis != self.params.price_axis:
            raise ValueError("序列的 price_axis 与参数不一致")
        if self.param_version != self.params.version:
            raise ValueError("序列的参数版本与参数不一致")

    @property
    def gap_indices(self) -> Tuple[int, ...]:
        return tuple(bar.index for bar in self.bars if bar.gap_before)

    def prefix(self, bar_count: int) -> "StandardSeries":
        """前 bar_count 根。重新构建，使哈希和间断与单独喂入这段输入一致。"""
        count = len(self.bars)
        if bar_count == count:
            return self
        if bar_count < 1 or bar_count > count:
            raise ValueError(f"bar_count 必须在 1 和 {count} 之间，收到 {bar_count}")
        head = self.bars[:bar_count]
        quotes = {}
        if head and head[0].high is not None:
            quotes = {
                "highs": [bar.high for bar in head],
                "lows": [bar.low for bar in head],
                "volumes": [bar.volume for bar in head],
            }
        if head and head[0].open is not None:
            quotes["opens"] = [bar.open for bar in head]
        return build_series(
            [bar.timestamp for bar in head],
            [bar.close for bar in head],
            interval=self.interval,
            params=self.params,
            **quotes,
        )


def build_series(
    timestamps: Sequence[Any],
    closes: Sequence[Any],
    *,
    interval: str = DAILY_INTERVAL,
    price_axis: Optional[str] = None,
    params: Optional[StructureParams] = None,
    highs: Optional[Sequence[Any]] = None,
    lows: Optional[Sequence[Any]] = None,
    volumes: Optional[Sequence[Any]] = None,
    opens: Optional[Sequence[Any]] = None,
) -> StandardSeries:
    """把已完成 K 线收成标准序列。

    时间戳必须严格递增。非正收盘价整段拒绝。相邻两根的日历或时钟间隔
    超过门槛时只打间断标记，不插入新 K 线。
    """
    resolved_interval = normalize_interval(interval)
    if params is None:
        params = default_params(price_axis or "log")
    elif price_axis is not None and normalize_price_axis(price_axis) != params.price_axis:
        raise ValueError(
            f"price_axis {price_axis!r} 与参数中的 {params.price_axis!r} 不一致"
        )

    if len(timestamps) != len(closes):
        raise ValueError(
            f"时间戳与收盘价数量不一致: {len(timestamps)} 与 {len(closes)}"
        )
    if not timestamps:
        raise ValueError("收盘序列为空")

    parsed_times = [_parse_timestamp(value) for value in timestamps]
    parsed_closes, bad_indices = _parse_closes(closes)
    if bad_indices:
        raise NonPositivePriceError(bad_indices)
    _require_strictly_increasing(parsed_times)
    _require_uniform_awareness(parsed_times)

    gaps = _gap_flags(parsed_times, resolved_interval, params)
    quote_rows = _quote_rows(params, len(parsed_closes), highs, lows, volumes)
    parsed_opens = _open_rows(params, len(parsed_closes), opens)
    digest = _hash_timestamps_and_closes(
        parsed_times,
        parsed_closes,
        None if quote_rows is None else [row[0] for row in quote_rows],
        None if quote_rows is None else [row[1] for row in quote_rows],
        None if quote_rows is None else [row[2] for row in quote_rows],
        parsed_opens,
    )
    bars: List[SeriesBar] = []
    for index, (ts, close, gap_before) in enumerate(
        zip(parsed_times, parsed_closes, gaps)
    ):
        high = low = volume = high_geometric = low_geometric = None
        open_price = open_geometric = None
        if quote_rows is not None:
            high, low, volume = quote_rows[index]
            high_geometric = _finite_positive_geometric(high, params.price_axis)
            low_geometric = _finite_positive_geometric(low, params.price_axis)
        if parsed_opens is not None:
            open_price = parsed_opens[index]
            open_geometric = _finite_positive_geometric(open_price, params.price_axis)
        bars.append(
            SeriesBar(
                index=index,
                timestamp=ts,
                close=close,
                geometric=_geometric(close, params.price_axis),
                gap_before=gap_before,
                high=high,
                low=low,
                volume=volume,
                high_geometric=high_geometric,
                low_geometric=low_geometric,
                open=open_price,
                open_geometric=open_geometric,
            )
        )
    return StandardSeries(
        bars=tuple(bars),
        price_basis=PRICE_BASIS_CLOSE,
        price_axis=params.price_axis,
        interval=resolved_interval,
        param_version=params.version,
        data_hash=digest,
        params=params,
    )


def result_directory(
    output_root: Path | str,
    symbol: str,
    interval: str,
    price_axis: str,
    data_hash: str,
    param_version: str = PARAM_VERSION,
) -> Path:
    """`structure_results/{symbol}/{interval}/{param_version}/{price_axis}/{data_hash}`。

    同一身份重复运行时路径不变，写入同名文件就是覆盖。不同收盘序列的
    `data_hash` 不同，因此不会写进同一个目录。
    """
    parts = (
        _path_component(symbol, "symbol"),
        _path_component(normalize_interval(interval), "interval"),
        _path_component(param_version, "param_version"),
        _path_component(normalize_price_axis(price_axis), "price_axis"),
        _path_component(data_hash, "data_hash"),
    )
    return Path(output_root).joinpath(*parts)


def write_series_identity(output_root: Path | str, symbol: str, series: StandardSeries) -> Path:
    """把序列身份写进运行目录。重复调用覆盖 `series_identity.json`。"""
    directory = result_directory(
        output_root,
        symbol,
        series.interval,
        series.price_axis,
        series.data_hash,
        series.param_version,
    )
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "series_identity",
        "param_version": series.param_version,
        "price_basis": series.price_basis,
        "price_axis": series.price_axis,
        "interval": series.interval,
        "data_hash": series.data_hash,
        "bar_count": len(series),
        "gap_indices": list(series.gap_indices),
        "params": series.params.to_dict(),
    }
    path = directory / IDENTITY_FILENAME
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _geometric(close: float, price_axis: str) -> float:
    if price_axis == "log":
        return math.log(close)
    return close


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"无法解析时间戳 {value!r}") from exc
        return parsed
    to_py = getattr(value, "to_pydatetime", None)
    if callable(to_py):
        parsed = to_py()
        if isinstance(parsed, datetime):
            return parsed
    raise ValueError(f"无法解析时间戳 {value!r}")


def _parse_closes(closes: Sequence[Any]) -> Tuple[List[float], List[int]]:
    parsed: List[float] = []
    bad: List[int] = []
    for index, value in enumerate(closes):
        if isinstance(value, bool) or value is None:
            bad.append(index)
            parsed.append(float("nan"))
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            bad.append(index)
            parsed.append(float("nan"))
            continue
        if not math.isfinite(number) or number <= 0:
            bad.append(index)
        parsed.append(number)
    return parsed, bad


def _require_strictly_increasing(timestamps: Sequence[datetime]) -> None:
    for index in range(1, len(timestamps)):
        try:
            out_of_order = timestamps[index] <= timestamps[index - 1]
        except TypeError as exc:
            raise ValueError("同一序列里不能混用有时区和无时区的时间戳") from exc
        if out_of_order:
            raise ValueError(
                f"时间戳必须严格递增，序号 {index - 1} 和 {index} 不满足"
            )


def _require_uniform_awareness(timestamps: Sequence[datetime]) -> None:
    aware = timestamps[0].tzinfo is not None
    for index, ts in enumerate(timestamps[1:], start=1):
        if (ts.tzinfo is not None) != aware:
            raise ValueError(
                f"同一序列里不能混用有时区和无时区的时间戳，序号 {index}"
            )


def _trading_day_span(start: date, end: date) -> int:
    """从 start 的下一交易日数到 end（含）。交易日是周一至周五。

    周五到周一计 1。恰好 5 个交易日不算间断，第 6 个才算。
    交易所假日这一版仍算交易日。
    """
    if end <= start:
        return 0
    elapsed = (end - start).days
    weeks, extra = divmod(elapsed, 7)
    span = weeks * 5
    for offset in range(1, extra + 1):
        if (start + timedelta(days=offset)).weekday() < 5:
            span += 1
    return span


def _gap_flags(
    timestamps: Sequence[datetime],
    interval: str,
    params: StructureParams,
) -> List[bool]:
    flags = [False] * len(timestamps)
    if interval == DAILY_INTERVAL:
        for index in range(1, len(timestamps)):
            span = _trading_day_span(
                timestamps[index - 1].date(),
                timestamps[index].date(),
            )
            flags[index] = span > params.daily_gap_days
        return flags

    prior_seconds: List[float] = []
    nominal = timedelta(seconds=INTERVAL_SECONDS[interval])
    multiple = params.intraday_gap_median_multiple
    for index in range(1, len(timestamps)):
        delta = timestamps[index] - timestamps[index - 1]
        if prior_seconds:
            median_seconds = statistics.median(prior_seconds)
            threshold = timedelta(seconds=median_seconds * multiple)
        else:
            threshold = nominal * multiple
        flags[index] = delta > threshold
        prior_seconds.append(delta.total_seconds())
    return flags


def _quote_rows(
    params: StructureParams,
    count: int,
    highs: Optional[Sequence[Any]],
    lows: Optional[Sequence[Any]],
    volumes: Optional[Sequence[Any]],
) -> Optional[List[Tuple[float, float, float]]]:
    """启用停顿时，三项都给了才写入序列并进入哈希。缺任何一项就当没给。"""
    if not series_quote_fields(params.version):
        return None
    if highs is None or lows is None or volumes is None:
        return None
    if len(highs) != count or len(lows) != count or len(volumes) != count:
        raise ValueError(
            f"最高价、最低价、成交量必须与收盘价一一对应: "
            f"{len(highs)}、{len(lows)}、{len(volumes)} 与 {count}"
        )
    return list(
        zip(_parse_quote(highs), _parse_quote(lows), _parse_quote(volumes))
    )


def _open_rows(
    params: StructureParams,
    count: int,
    opens: Optional[Sequence[Any]],
) -> Optional[List[float]]:
    """启用阶段且调用方传入开盘价时才写入。其他版本即使传入也忽略。"""
    if opens is None or not series_uses_open(params.version):
        return None
    if len(opens) != count:
        raise ValueError(f"开盘价必须与收盘价一一对应: {len(opens)} 与 {count}")
    return _parse_quote(opens)


def _parse_quote(values: Sequence[Any]) -> List[float]:
    parsed: List[float] = []
    for value in values:
        if isinstance(value, bool) or value is None:
            parsed.append(float("nan"))
            continue
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            parsed.append(float("nan"))
    return parsed


def _finite_positive_geometric(value: float, price_axis: str) -> Optional[float]:
    if not math.isfinite(value) or value <= 0:
        return None
    return _geometric(value, price_axis)


def _hash_timestamps_and_closes(
    timestamps: Sequence[datetime],
    closes: Sequence[float],
    highs: Optional[Sequence[float]] = None,
    lows: Optional[Sequence[float]] = None,
    volumes: Optional[Sequence[float]] = None,
    opens: Optional[Sequence[float]] = None,
) -> str:
    digest = hashlib.sha256()
    use_quotes = highs is not None and lows is not None and volumes is not None
    for index, (ts, close) in enumerate(zip(timestamps, closes)):
        digest.update(_canonical_timestamp(ts))
        digest.update(struct.pack(">d", close))
        if use_quotes:
            digest.update(struct.pack(">d", float(highs[index])))
            digest.update(struct.pack(">d", float(lows[index])))
            digest.update(struct.pack(">d", float(volumes[index])))
        if opens is not None:
            digest.update(struct.pack(">d", float(opens[index])))
    return digest.hexdigest()[:32]


def _canonical_timestamp(ts: datetime) -> bytes:
    if ts.tzinfo is None:
        return b"naive:" + ts.isoformat().encode("ascii")
    utc = ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return b"utc:" + utc.encode("ascii")


def _path_component(value: str, label: str) -> str:
    text = str(value).strip()
    if (
        not text
        or text in {".", ".."}
        or "/" in text
        or "\\" in text
        or "\x00" in text
    ):
        raise ValueError(f"{label} 不能作为目录名: {value!r}")
    return text
