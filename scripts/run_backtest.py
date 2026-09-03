#!/usr/bin/env python3
"""Run a strategy backtest from YAML config.

Defaults come from analysis/backtest/ma_cross.yaml.
CLI flags override the file when provided.

Examples:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --config analysis/backtest/ma_cross.yaml
    python scripts/run_backtest.py --strategy ma_cross --variants ma5_ma10 --interval 5m
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.backtest.config import (
    BAR_INTERVALS,
    DEFAULT_CONFIG_PATH,
    load_backtest_config,
)
from analysis.backtest.runner import run_backtest
from analysis.backtest.strategy import list_strategies


def _csv(value: str) -> list:
    return [part.strip() for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="策略回测（配置文件 + 可选命令行覆盖）")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"YAML 配置路径，默认 {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument("--strategy", help="覆盖策略名")
    parser.add_argument("--symbols", help="覆盖标的，逗号分隔")
    parser.add_argument("--variants", help="覆盖要跑的参数方案，逗号分隔")
    parser.add_argument("--schemes", help="同 --variants（兼容旧参数）")
    parser.add_argument("--interval", choices=BAR_INTERVALS, help="覆盖 K 线周期")
    parser.add_argument("--start", help="覆盖开始日期 YYYY-MM-DD")
    parser.add_argument("--end", help="覆盖结束日期 YYYY-MM-DD")
    parser.add_argument("--years", type=float, help="覆盖回看年数（仅当未指定 --start）")
    parser.add_argument("--capital", type=float, help="覆盖每股初始资金")
    parser.add_argument("--max-trades-per-day", type=int, help="覆盖每日最多成交次数")
    parser.add_argument("--commission", type=float, help="覆盖佣金费率")
    parser.add_argument("--stamp", type=float, help="覆盖印花税费率")
    parser.add_argument("--lot-size", type=int, help="覆盖默认最小买卖单位（未单独配置的标的）")
    parser.add_argument("--output-dir", help="覆盖报告目录")
    parser.add_argument("--list-strategies", action="store_true", help="列出已注册策略")
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_strategies:
        for name in list_strategies():
            print(name)
        return 0

    variants = _csv(args.variants) if args.variants else None
    if variants is None and args.schemes:
        variants = _csv(args.schemes)

    overrides = {
        "strategy": args.strategy,
        "symbols": _csv(args.symbols) if args.symbols else None,
        "variants": variants,
        "interval": args.interval,
        "start": args.start,
        "end": args.end,
        "lookback_years": args.years,
        "capital": args.capital,
        "max_trades_per_day": args.max_trades_per_day,
        "commission": args.commission,
        "stamp": args.stamp,
        "lot_size": args.lot_size,
        "output_dir": args.output_dir,
    }
    config = load_backtest_config(args.config, overrides)
    payload = run_backtest(config)
    print(f"\n索引: {payload.get('index_html')}")
    print(f"报告: {payload.get('report_md')}")
    print(f"JSON: {Path(payload['output_dir']) / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
