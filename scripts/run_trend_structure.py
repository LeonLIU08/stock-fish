#!/usr/bin/env python3
"""对一只股票的一段已完成 K 线做趋势结构拟合，并写出离线 HTML。

短、中、长的 k、门槛下限和最大跨度不随时间范围改变。
默认参数版本是 structure-params-v1。对照旧规则时加上
--param-version structure-params-v0。

示例:
    python scripts/run_trend_structure.py --symbol 00700 --interval 1d --years 1
    python scripts/run_trend_structure.py --symbol 00700 --start 2024-01-01 --end 2024-12-31
    python scripts/run_trend_structure.py --symbol 600519 --years 2 --price-axis uniform --scales short,mid,long
    python scripts/run_trend_structure.py --symbol 00700 --years 1 --as-of 2025-06-01
    python scripts/run_trend_structure.py --samples
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.structure.config import BAR_INTERVALS, PARAM_VERSIONS, SCALE_NAMES, default_params
from analysis.structure.html_report import write_result
from analysis.structure.lifecycle import build_lifecycle
from analysis.structure.series import NonPositivePriceError, build_series
from analysis.structure.snapshot import (
    SCALE_DEFINITION_NOTE,
    build_snapshot,
    resolve_requested_window,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="拟合短、中、长趋势线和趋势区间，写出可离线打开的 report.html",
        epilog="字段说明见 document/trend-structure-manual.md",
    )
    parser.add_argument("--symbol", help="股票代码，例如 00700、600519、AAPL")
    parser.add_argument("--interval", default="1d", choices=BAR_INTERVALS, help="K 线周期，默认 1d")
    parser.add_argument("--start", help="开始日期 YYYY-MM-DD。和 --years 分开使用")
    parser.add_argument("--end", help="结束日期 YYYY-MM-DD，默认今天")
    parser.add_argument("--years", type=float, help="回看年数。未写 --start 时默认 1 年")
    parser.add_argument("--price-axis", default="log", choices=("log", "uniform"), help="价格轴，默认 log")
    parser.add_argument(
        "--param-version",
        default="structure-params-v1",
        choices=PARAM_VERSIONS,
        help="参数版本。v1 放宽区间接触簇计数，v0 是原来的规则",
    )
    parser.add_argument(
        "--scales",
        default="mid",
        help="主图初次打开的尺度，逗号分隔：short,mid,long。三个尺度都会拟合",
    )
    parser.add_argument("--as-of", help="只显示该时刻已经可用的拐点、线和区间。YYYY-MM-DD 或更精确的时间")
    parser.add_argument("--output-dir", default="structure_results", help="输出根目录")
    parser.add_argument("--samples", action="store_true", help="写出四份合成检查页，不访问网络")
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.samples:
            return _write_samples(Path(args.output_dir), args.price_axis)
        if not args.symbol:
            raise ValueError("请提供 --symbol，或用 --samples 生成合成检查页")
        start, end = resolve_requested_window(
            start=_parse_date(args.start) if args.start else None,
            end=_parse_date(args.end) if args.end else None,
            years=args.years,
            today=date.today(),
        )
        params = default_params(args.price_axis, version=args.param_version)
        _print_scale_definition(start, end, params)
        timestamps, closes, source = _load_closes(args.symbol, args.interval, start, end)
        series = build_series(
            timestamps,
            closes,
            interval=args.interval,
            params=params,
        )
        snapshot = build_snapshot(
            series,
            symbol=args.symbol,
            requested_start=start,
            requested_end=end,
            as_of=args.as_of,
            visible_scales=args.scales,
            data_source=source,
        )
        written = write_result(args.output_dir, snapshot)
    except (ValueError, RuntimeError, NonPositivePriceError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"K 线: {snapshot['identity']['bar_count']} 根，来源 {snapshot['identity']['data_source']}")
    print(f"快照: {written['snapshot']}")
    print(f"报告: {written['report']}")
    return 0


def _print_scale_definition(start: date, end: date, params) -> None:
    pieces = [f"{scale.name} k={scale.k:g} 最大跨度 {scale.max_span}" for scale in params.scales]
    credit = "开启" if params.zone_credit_full_span else "关闭"
    print(f"请求区间: {start.isoformat()} 至 {end.isoformat()}（只选择 K 线）")
    print(f"参数版本: {params.version}；" + "；".join(pieces))
    print(
        f"区间接触: 有效区间内每侧至少 {params.zone_min_touch_clusters} 个；"
        f"边界全长折算 {credit}。趋势线仍要 {params.min_touch_clusters} 个"
    )
    print(SCALE_DEFINITION_NOTE)


def _load_closes(symbol: str, interval: str, start: date, end: date):
    """复用回测的行情加载，再裁回请求区间。只把时间戳和收盘价交给引擎。"""
    from analysis.backtest.bars import BarLoader, slice_eval_window
    from analysis.backtest.config import BacktestConfig

    config = BacktestConfig(symbols=[symbol], interval=interval, start=start, end=end)
    frame, source = BarLoader().load_symbol(symbol, config, warmup_bars=0)
    frame = slice_eval_window(frame, start, end)
    if frame is None or frame.empty:
        raise RuntimeError(f"{symbol} 在 {start.isoformat()} 至 {end.isoformat()} 没有已完成 K 线")
    timestamps = [_as_datetime(value) for value in frame["datetime"]]
    closes = [float(value) for value in frame["close"]]
    return timestamps, closes, source


def _write_samples(output_root: Path, price_axis: str) -> int:
    """四份肉眼检查页，外加一份均匀轴对照。形状固定，不作为阈值来源。"""
    cases = [
        ("SAMPLE-RISING", "单边上升后的支撑", "抬高的低点连成支撑。", _rising_support_levels(), "log"),
        ("SAMPLE-CHANNEL", "平行上升通道", "上下轨宽度近似稳定。", _rail_levels(), "log"),
        ("SAMPLE-BREAK", "通道末端跌破", "通道形成后，两根收盘落在下轨之外。", _channel_break_levels(), "log"),
        ("SAMPLE-DEEP", "深破位反例", "两个较浅低点之间有一个已确认的更深低点，穿过它的线不能已验证。", _broken_support_levels(), "log"),
        ("SAMPLE-UNIFORM", "均匀轴对照", "同一段上升支撑改在均匀价格轴上拟合。纵轴上等价差对应等距离。", _rising_support_levels(), "uniform"),
    ]
    if price_axis == "uniform":
        cases = [item for item in cases if item[0] != "SAMPLE-UNIFORM"]
        cases = [(symbol, title, blurb, levels, "uniform") for symbol, title, blurb, levels, _axis in cases]
    index_rows = []
    for symbol, title, blurb, levels, axis in cases:
        series = _series_from_levels(levels, price_axis=axis)
        snapshot = build_snapshot(
            series,
            symbol=symbol,
            requested_start=series.bars[0].timestamp.date(),
            requested_end=series.bars[-1].timestamp.date(),
            visible_scales=list(SCALE_NAMES),
            data_source="synthetic",
            engine_version="sample",
        )
        written = write_result(output_root, snapshot)
        index_rows.append((title, blurb, written["report"], axis))
        print(f"{title}: {written['report']}")
    index = output_root / "_samples" / "index.html"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(_sample_index(index_rows, index.parent), encoding="utf-8")
    print(f"索引: {index}")
    return 0


def _sample_index(rows, index_dir: Path) -> str:
    items = []
    for title, blurb, report, axis in rows:
        href = Path(os.path.relpath(report, index_dir)).as_posix()
        items.append(
            f"<li><a href=\"{href}\">{title}</a>"
            f"<span>（{'对数价格' if axis == 'log' else '均匀价格'}）</span>"
            f"<p>{blurb}</p></li>"
        )
    body = "\n".join(items)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>趋势结构合成检查页</title>
<style>
  body {{ margin: 2rem; background: #14120f; color: #f3efe4; font: 16px/1.6 "PingFang SC", sans-serif; }}
  a {{ color: #e0a45a; }}
  li {{ margin: 1rem 0; }}
  span, p {{ color: #a79f91; }}
</style>
</head>
<body>
<h1>合成检查页</h1>
<p>这几页用固定的收盘序列生成，用来看直线、色带和突破标记，不作为阈值的来源。三个尺度都会拟合；页面初始把它们都打开。</p>
<ol>
{body}
</ol>
</body>
</html>
"""


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"日期需要是 YYYY-MM-DD，收到 {value!r}") from exc


def _as_datetime(value):
    if isinstance(value, datetime):
        return value
    to_py = getattr(value, "to_pydatetime", None)
    if callable(to_py):
        parsed = to_py()
        if isinstance(parsed, datetime):
            return parsed
    return value


def _daily_times(count, start=date(2024, 1, 2)):
    cursor = datetime(start.year, start.month, start.day, 16, 0)
    times = []
    for index in range(count):
        if index:
            cursor = cursor + timedelta(days=1)
        times.append(cursor)
    return times


def _series_from_levels(levels, price_axis="log"):
    closes = [100.0 * math.exp(level) for level in levels]
    return build_series(_daily_times(len(levels)), closes, interval="1d", price_axis=price_axis)


def _warm(count=30):
    levels = [0.0, 0.001]
    levels.extend([0.001] * (count - 2))
    return levels


def _rising_support_levels():
    slope = 0.0002
    levels = _warm()
    for index in (37, 45, 53):
        levels.extend([0.03] * 4)
        levels.extend([slope * index] * 4)
    levels.extend([0.03] * 2)
    return levels


def _broken_support_levels():
    levels = _warm()
    for level in (0.02, 0.0, 0.02):
        levels.extend([0.03] * 4)
        levels.extend([level] * 4)
    levels.extend([0.03] * 2)
    return levels


def _rail_levels(a=0.001, width=0.08, pairs=4, step=6):
    warm = _warm()
    events = []
    cursor = len(warm)
    for _ in range(pairs):
        events.append(("H", cursor))
        cursor += step
        events.append(("L", cursor))
        cursor += step
    confirm = events[-1][1] + step
    levels = [None] * (confirm + 1)
    for index, value in enumerate(warm):
        levels[index] = value

    def low_at(index):
        return a * index

    def high_at(index):
        return a * index + width

    for kind, index in events:
        levels[index] = high_at(index) if kind == "H" else low_at(index)
    levels[confirm] = high_at(confirm)
    for index in range(len(warm), len(levels)):
        if levels[index] is None:
            levels[index] = (low_at(index) + high_at(index)) / 2.0
    return levels


def _channel_break_levels():
    levels = _rail_levels()
    series = _series_from_levels(levels, price_axis="log")
    lifecycle = build_lifecycle(series)
    zone = next(
        item
        for item in lifecycle.zones
        if item.scale == "short" and item.label == "channel" and item.primary and item.draw_on_main
    )
    return levels + _levels_from_line(zone, len(levels), 2, -0.8, below_zone=True)


def _levels_from_line(structure, start_index, count, volatility_multiple, below_zone=False):
    levels = []
    for step in range(count):
        index = start_index + step
        if below_zone:
            geometric = structure.lower_at(index) + volatility_multiple * structure.volatility
        else:
            geometric = structure.line_at(index) + volatility_multiple * structure.volatility
        levels.append(geometric - math.log(100.0))
    return levels


if __name__ == "__main__":
    raise SystemExit(main())
