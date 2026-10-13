#!/usr/bin/env python3
"""趋势状态策略回测。

第一阶段对每只股票、每个 run 走取数、结构快照和空账户，并写出现有格式的报告。
策略 A 和策略 B 仍是不同账户，这一阶段都不下单。

示例:
    /Users/leon/PyVenv/q_stock/bin/python scripts/run_trend_backtest.py
    /Users/leon/PyVenv/q_stock/bin/python scripts/run_trend_backtest.py \
        --config analysis/backtest/trend-strategy-v0.yaml \
        --symbols 00700,09988,NVDA --runs breakout,pullback --years 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.backtest.trend.config import DEFAULT_TREND_CONFIG_PATH, load_trend_config
from analysis.backtest.trend.runner import run_trend_backtest
from analysis.structure.series import NonPositivePriceError


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
    parser.add_argument("--runs", help="覆盖要跑的 run，逗号分隔。不写则跑配置里的全部")
    parser.add_argument("--start", help="覆盖开始日期 YYYY-MM-DD")
    parser.add_argument("--end", help="覆盖结束日期 YYYY-MM-DD")
    parser.add_argument("--years", type=float, help="覆盖回看年数。已写开始日期时不要再用")
    parser.add_argument("--output-dir", help="覆盖报告根目录")
    return parser


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {
        "symbols": _csv(args.symbols) if args.symbols else None,
        "runs": _csv(args.runs) if args.runs else None,
        "start": args.start,
        "end": args.end,
        "lookback_years": args.years,
        "output_dir": args.output_dir,
    }
    try:
        config = load_trend_config(args.config, overrides)
        payload = run_trend_backtest(config)
    except (ValueError, RuntimeError, FileNotFoundError, OSError, NonPositivePriceError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"\n索引: {payload.get('index_html')}")
    print(f"报告: {payload.get('report_md')}")
    print(f"JSON: {Path(payload['output_dir']) / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
