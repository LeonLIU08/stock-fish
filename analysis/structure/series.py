"""标准序列：序号、收盘价、几何价格、间断和数据身份。

引擎只读时间戳和收盘价。几何横轴是有效 K 线序号，不把停牌或缺失填成横盘。
`data_hash` 只覆盖时间戳和收盘价，参数版本和价格轴放在输出目录里。
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
        return build_series(
            [bar.timestamp for bar in head],
            [bar.close for bar in head],
            interval=self.interval,
            params=self.params,
        )


def build_series(
    timestamps: Sequence[Any],
    closes: Sequence[Any],
    *,
    interval: str = DAILY_INTERVAL,
    price_axis: Optional[str] = None,
    params: Optional[StructureParams] = None,
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
    digest = _hash_timestamps_and_closes(parsed_times, parsed_closes)
    bars: List[SeriesBar] = []
    for index, (ts, close, gap_before) in enumerate(
        zip(parsed_times, parsed_closes, gaps)
    ):
        bars.append(
            SeriesBar(
                index=index,
                timestamp=ts,
                close=close,
                geometric=_geometric(close, params.price_axis),
                gap_before=gap_before,
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


def _hash_timestamps_and_closes(
    timestamps: Sequence[datetime],
    closes: Sequence[float],
) -> str:
    digest = hashlib.sha256()
    for ts, close in zip(timestamps, closes):
        digest.update(_canonical_timestamp(ts))
        digest.update(struct.pack(">d", close))
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
