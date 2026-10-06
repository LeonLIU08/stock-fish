#!/usr/bin/env python3
"""对一段已完成 K 线做趋势结构拟合，并写出离线 HTML。

一次命令可以跑多只股票。它们共用日期、周期、价格轴和参数版本，
每只仍写入自己的目录。每次运行都会刷新输出根目录的 index.html，
点一只股票就打开它的 report.html。

短、中、长的 k、门槛下限和最大跨度不随时间范围改变。
默认参数版本是 structure-params-v1。对照旧规则时加上
--param-version structure-params-v0。

示例:
    python scripts/run_trend_structure.py --symbol 00700 --interval 1d --years 1
    python scripts/run_trend_structure.py --symbol 00700,600519,AAPL --years 1 --param-version structure-params-v0
    python scripts/run_trend_structure.py --symbol 00700 --symbol 600519 --years 1
    python scripts/run_trend_structure.py --symbols-file symbols.txt --years 1
    python scripts/run_trend_structure.py --symbol 00700 --start 2024-01-01 --end 2024-12-31
    python scripts/run_trend_structure.py --symbol 600519 --years 2 --price-axis uniform --scales short,mid,long
    python scripts/run_trend_structure.py --symbol 00700 --years 1 --as-of 2025-06-01
    python scripts/run_trend_structure.py --samples
"""
from __future__ import annotations

import argparse
import json
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
    parser.add_argument(
        "--symbol",
        nargs="+",
        action="append",
        metavar="CODE",
        help="股票代码。可重复，也可用逗号、空格或顿号分隔：00700,600519 AAPL",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        action="append",
        metavar="CODE",
        help="同 --symbol",
    )
    parser.add_argument(
        "--symbols-file",
        help="股票列表文件。每行一个或多个代码，# 后为注释",
    )
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
        symbols = collect_symbols(args.symbol, args.symbols, args.symbols_file)
        if not symbols:
            raise ValueError("请提供 --symbol，或用 --samples 生成合成检查页")
        start, end = resolve_requested_window(
            start=_parse_date(args.start) if args.start else None,
            end=_parse_date(args.end) if args.end else None,
            years=args.years,
            today=date.today(),
        )
        params = default_params(args.price_axis, version=args.param_version)
        _print_scale_definition(start, end, params)
    except (ValueError, RuntimeError, NonPositivePriceError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    outcomes = []
    total = len(symbols)
    for index, symbol in enumerate(symbols, start=1):
        if total > 1:
            print(f"[{index}/{total}] {symbol}")
        try:
            outcome = _run_symbol(
                symbol,
                interval=args.interval,
                start=start,
                end=end,
                params=params,
                as_of=args.as_of,
                scales=args.scales,
                output_dir=args.output_dir,
            )
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            if not isinstance(exc, (ValueError, RuntimeError, NonPositivePriceError)):
                message = f"{exc.__class__.__name__}: {message}"
            print(f"{symbol}: {message}", file=sys.stderr)
            outcomes.append({"symbol": symbol, "error": message})
            continue
        outcomes.append(outcome)
        if outcome.get("history_note"):
            print(outcome["history_note"])
        print(f"K 线: {outcome['bar_count']} 根，来源 {outcome['source']}")
        print(f"快照: {outcome['snapshot']}")
        print(f"报告: {outcome['report']}")

    failed = [item for item in outcomes if item.get("error")]
    if total > 1:
        print(f"完成 {total - len(failed)} 只，失败 {len(failed)} 只")
    index_path = _write_catalog_index(Path(args.output_dir), failures=failed)
    if index_path is not None:
        print(f"索引: {index_path}")
    return 1 if failed else 0


def collect_symbols(symbol_groups, symbols_groups, symbols_file: str | None) -> list[str]:
    """按出现顺序收集代码，并去掉重复。"""
    tokens: list[str] = []
    for groups in (symbol_groups, symbols_groups):
        for group in groups or []:
            for item in group:
                tokens.extend(_split_symbol_text(item))
    if symbols_file:
        path = Path(symbols_file)
        if not path.is_file():
            raise ValueError(f"找不到股票列表: {path}")
        tokens.extend(_split_symbol_text(path.read_text(encoding="utf-8")))
    ordered: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _split_symbol_text(text: str) -> list[str]:
    normalized = text.replace("，", ",").replace("、", ",").replace(";", ",")
    chunks: list[str] = []
    for line in normalized.splitlines():
        comment = line.split("#", 1)[0]
        for piece in comment.split(","):
            chunks.extend(piece.split())
    return [chunk for chunk in chunks if chunk]


def _run_symbol(symbol, *, interval, start, end, params, as_of, scales, output_dir) -> dict:
    timestamps, closes, source = _load_closes(symbol, interval, start, end)
    series = build_series(timestamps, closes, interval=interval, params=params)
    snapshot = build_snapshot(
        series,
        symbol=symbol,
        requested_start=start,
        requested_end=end,
        as_of=as_of,
        visible_scales=scales,
        data_source=source,
    )
    written = write_result(output_dir, snapshot)
    identity = snapshot["identity"]
    return {
        "symbol": symbol,
        "bar_count": identity["bar_count"],
        "source": identity["data_source"],
        "snapshot": written["snapshot"],
        "report": written["report"],
        "history_note": identity.get("history_note") or "",
    }


def _write_catalog_index(output_root: Path, failures=None) -> Path | None:
    """扫描结果目录，写成点一次就打开 report.html 的入口。"""
    grouped = _catalog_groups(output_root)
    failure_rows = [item for item in (failures or []) if item.get("error")]
    samples = output_root / "_samples" / "index.html"
    if not grouped and not failure_rows and not samples.is_file():
        return None
    output_root.mkdir(parents=True, exist_ok=True)
    index = output_root / "index.html"
    cards = []
    for symbol, reports in grouped:
        primary = reports[0]
        alts = reports[1:]
        cards.append(_catalog_card(symbol, primary, alts, output_root))
    failed_html = ""
    if failure_rows:
        items = "\n".join(
            f"<li><strong>{_esc(item['symbol'])}</strong><p>{_esc(item['error'])}</p></li>"
            for item in failure_rows
        )
        failed_html = f"<section class=\"failed\"><h2>这次没有写出报告</h2><ul>{items}</ul></section>"
    sample_html = ""
    if samples.is_file():
        sample_html = '<p class="samples"><a href="_samples/index.html">合成检查页</a></p>'
    count = len(grouped)
    index.write_text(
        f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>趋势结构</title>
<style>
  body {{ margin: 0; background: #14120f; color: #f3efe4; font: 16px/1.5 "PingFang SC", sans-serif; }}
  header, main {{ margin: 0 auto; max-width: 1080px; padding: 1.5rem 1.25rem; }}
  h1 {{ margin: 0 0 0.35rem; font-size: 1.6rem; }}
  .lead, .meta, .note, .failed p {{ color: #a79f91; }}
  a {{ color: #e0a45a; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 0.9rem; margin-top: 1.2rem; }}
  a.card, article.card {{ display: block; background: #1e1b16; border-radius: 12px; padding: 1rem 1.05rem; text-decoration: none; color: inherit; }}
  a.card:hover, article.card:hover {{ background: #28241c; }}
  a.card strong, a.title {{ display: block; color: #e0a45a; font-size: 1.45rem; letter-spacing: 0.02em; }}
  .name {{ display: block; margin-top: 0.12rem; color: #f3efe4; font-size: 1rem; font-weight: 500; letter-spacing: 0; }}
  a.title {{ text-decoration: none; }}
  a.card:hover strong, a.title:hover {{ text-decoration: underline; }}
  .meta, .note {{ margin: 0.35rem 0 0; font-size: 0.92rem; }}
  .alts {{ display: flex; flex-wrap: wrap; gap: 0.45rem 0.8rem; margin-top: 0.7rem; }}
  .alts a {{ font-size: 0.88rem; }}
  .failed {{ margin-top: 1.6rem; }}
  .failed li {{ margin: 0.6rem 0; }}
  .samples {{ margin-top: 1.4rem; }}
</style>
</head>
<body>
<header>
  <h1>趋势结构</h1>
  <p class="lead">点一只股票，直接打开它的报告。共 {count} 只。</p>
</header>
<main>
  <div class="grid">
{"".join(cards)}
  </div>
  {failed_html}
  {sample_html}
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return index


def _catalog_groups(output_root: Path) -> list[tuple[str, list[dict]]]:
    """同一标的、周期、参数版本、价格轴只保留最新一份报告。"""
    latest: dict[tuple, dict] = {}
    if not output_root.is_dir():
        return []
    for report in output_root.rglob("report.html"):
        if "_samples" in report.parts:
            continue
        entry = _catalog_entry(output_root, report)
        if entry is None:
            continue
        key = (entry["symbol"], entry["interval"], entry["param_version"], entry["price_axis"])
        current = latest.get(key)
        if current is None or entry["mtime"] > current["mtime"]:
            latest[key] = entry
    by_symbol: dict[str, list[dict]] = {}
    for entry in latest.values():
        by_symbol.setdefault(entry["symbol"], []).append(entry)
    ordered = []
    for symbol in sorted(by_symbol, key=_symbol_sort_key):
        reports = sorted(by_symbol[symbol], key=_report_sort_key)
        ordered.append((symbol, reports))
    return ordered


def _catalog_entry(output_root: Path, report: Path) -> dict | None:
    snapshot_path = report.with_name("snapshot.json")
    identity = {}
    if snapshot_path.is_file():
        try:
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
            identity = payload.get("identity") or {}
        except (OSError, json.JSONDecodeError):
            identity = {}
    parts = report.relative_to(output_root).parts
    symbol = str(identity.get("symbol") or (parts[0] if parts else "")).strip()
    if not symbol or symbol.startswith("SAMPLE"):
        return None
    interval = str(identity.get("interval") or (parts[1] if len(parts) > 1 else ""))
    param_version = str(identity.get("param_version") or (parts[2] if len(parts) > 2 else ""))
    price_axis = str(identity.get("price_axis") or (parts[3] if len(parts) > 3 else ""))
    start = _day_text(identity.get("start_time") or identity.get("requested_start"))
    end = _day_text(identity.get("end_time") or identity.get("requested_end"))
    return {
        "symbol": symbol,
        "interval": interval,
        "param_version": param_version,
        "price_axis": price_axis,
        "price_axis_label": identity.get("price_axis_label") or ("对数价格" if price_axis == "log" else "均匀价格" if price_axis == "uniform" else price_axis),
        "bar_count": identity.get("bar_count"),
        "start": start,
        "end": end,
        "history_note": identity.get("history_note") or "",
        "report": report,
        "mtime": report.stat().st_mtime,
    }


def _catalog_card(symbol: str, primary: dict, alts: list[dict], output_root: Path) -> str:
    href = _report_href(primary["report"], output_root)
    title = _esc(symbol) + _name_markup(symbol)
    meta = _esc(_report_meta(primary))
    note = '<p class="note">可得区间短于请求</p>' if primary.get("history_note") else ""
    if not alts:
        return (
            f'<a class="card" href="{href}">'
            f"<strong>{title}</strong>"
            f'<p class="meta">{meta}</p>'
            f"{note}"
            "</a>\n"
        )
    links = "".join(
        f'<a href="{_report_href(item["report"], output_root)}">{_esc(_variant_label(item))}</a>'
        for item in alts
    )
    return (
        '<article class="card">'
        f'<a class="title" href="{href}">{title}</a>'
        f'<p class="meta">{meta}</p>'
        f"{note}"
        f'<div class="alts">{links}</div>'
        "</article>\n"
    )


def _name_markup(symbol: str) -> str:
    name = _stock_display_name(symbol)
    if not name:
        return ""
    return f'<span class="name">{_esc(name)}</span>'


def _stock_display_name(symbol: str) -> str:
    """港股数字代码和已有映射里的中文名。没有对应名称时留空。"""
    from market_data.stock_index.stock_mapping import STOCK_NAME_MAP, is_meaningful_stock_name

    code = str(symbol or "").strip().upper()
    if code.isdigit():
        code = code.zfill(5)
    name = STOCK_NAME_MAP.get(code, "")
    if not is_meaningful_stock_name(name, code):
        return ""
    return name


def _report_meta(entry: dict) -> str:
    window = ""
    if entry.get("start") and entry.get("end"):
        window = f"{entry['start']} 至 {entry['end']}"
    elif entry.get("start"):
        window = str(entry["start"])
    bars = f"{entry['bar_count']} 根" if entry.get("bar_count") not in (None, "") else ""
    pieces = [entry.get("price_axis_label") or "", entry.get("param_version") or "", window, bars]
    return " · ".join(str(piece) for piece in pieces if piece)


def _variant_label(entry: dict) -> str:
    return _report_meta(entry)


def _report_href(report: Path, output_root: Path) -> str:
    return _esc(Path(os.path.relpath(report, output_root)).as_posix())


def _day_text(value) -> str:
    text = str(value or "").strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    return ""


def _symbol_sort_key(symbol: str) -> tuple:
    if symbol[:1].isdigit():
        return (0, symbol)
    return (1, symbol)


def _report_sort_key(entry: dict) -> tuple:
    axis_rank = 0 if entry.get("price_axis") == "log" else 1
    version_rank = 0 if str(entry.get("param_version") or "").endswith("v0") else 1
    return (axis_rank, version_rank, entry.get("interval") or "", -float(entry.get("mtime") or 0))


def _esc(text) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


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
    """复用回测的行情加载，再裁回请求区间。只把时间戳和收盘价交给引擎。

    上市时间短于请求区间时，根数达到短、中尺度下限就用已有 K 线继续。
    """
    from analysis.backtest.bars import BarLoader, slice_eval_window
    from analysis.backtest.config import BacktestConfig
    from analysis.structure.config import partial_history_min_bars

    config = BacktestConfig(symbols=[symbol], interval=interval, start=start, end=end)
    frame, source = BarLoader().load_symbol(
        symbol,
        config,
        warmup_bars=0,
        allow_short_history=True,
        min_bars=partial_history_min_bars(),
    )
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
