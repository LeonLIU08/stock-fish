#!/usr/bin/env python3
"""趋势状态策略回测。

每只股票、每个 run 一个账户。信号用复权价，成交用不复权价。
策略 A、策略 B、策略 C、策略 D0 和策略 D1 分开记账。策略 C 在短尺度上升趋势出现后的下一根开盘买入，在中尺度下降趋势出现后的下一根开盘卖出。策略 D0 的买卖都用短尺度。策略 D1 在 D0 上再加持有以来最高价回撤 15% 的止损。

示例:
    /Users/leon/PyVenv/q_stock/bin/python scripts/run_trend_backtest.py
    /Users/leon/PyVenv/q_stock/bin/python scripts/run_trend_backtest.py \
        --symbols-file scripts/hkconnect_06Oct.txt --years 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.backtest.trend.config import (
    DEFAULT_TREND_CONFIG_PATH,
    load_trend_config,
    read_symbol_file,
)
from analysis.backtest.trend.lots import fetch_hk_lot_sizes
from analysis.backtest.trend.runner import run_trend_backtest
from analysis.structure.series import NonPositivePriceError


def _symbols(symbols: str | None, symbols_file: str | None) -> list | None:
    codes = list(read_symbol_file(symbols_file)) if symbols_file else []
    if symbols:
        seen = set(codes)
        for code in _csv(symbols):
            if code not in seen:
                seen.add(code)
                codes.append(code)
    return codes or None


def _csv(value: str) -> list:
    return [part.strip() for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="趋势状态策略回测（配置文件 + 可选命令行覆盖）")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_TREND_CONFIG_PATH),
        help=f"YAML 配置路径，默认 {DEFAULT_TREND_CONFIG_PATH}",
    )
    parser.add_argument("--symbols", help="覆盖标的，逗号分隔")
    parser.add_argument("--symbols-file", help="股票名单，一行一个代码。与 --symbols 同时写时合并，名单在前")
    parser.add_argument("--runs", help="覆盖要跑的 run，逗号分隔。不写则跑配置里的全部")
    parser.add_argument("--start", help="覆盖开始日期 YYYY-MM-DD")
    parser.add_argument("--end", help="覆盖结束日期 YYYY-MM-DD")
    parser.add_argument("--years", type=float, help="覆盖回看年数。已写开始日期时不要再用")
    parser.add_argument("--output-dir", help="覆盖报告根目录")
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    codes = _symbols(args.symbols, args.symbols_file)
    overrides = {
        "symbols": codes,
        "runs": _csv(args.runs) if args.runs else None,
        "start": args.start,
        "end": args.end,
        "lookback_years": args.years,
        "output_dir": args.output_dir,
    }
    try:
        config = load_trend_config(args.config, overrides, lot_lookup=fetch_hk_lot_sizes)
        payload = run_trend_backtest(config)
    except (ValueError, RuntimeError, FileNotFoundError, OSError, NonPositivePriceError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return _finish(payload)


def _finish(payload: dict) -> int:
    failures = list(payload.get("failures") or [])
    print(f"\n索引: {payload.get('index_html')}")
    print(f"报告: {payload.get('report_md')}")
    print(f"JSON: {Path(payload['output_dir']) / 'summary.json'}")
    if not failures:
        return 0
    print(f"跳过 {len(failures)} 只", file=sys.stderr)
    for item in failures:
        print(f"{item['symbol']}: {item['error']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
