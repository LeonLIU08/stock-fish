#!/usr/bin/env python3
"""同一段行情对照参数。每只股票的结构只算一次，不写个股报告。

baseline 沿用当前 YAML。其余每组只改一项：
trail_continue 只在上涨延续里抬跟踪止损，上涨回调里止损不动。
buffer0 把策略 A 的突破缓冲从 0.2 改为 0。
pullback2 把策略 B 的前高窗口从 3 根改为 2 根。
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.backtest.trend.config import load_trend_config
from analysis.backtest.trend.lots import fetch_hk_lot_sizes
from analysis.backtest.trend.runner import StrategyArm, run_trend_backtest

ARMS = (
    StrategyArm("baseline"),
    StrategyArm("trail_continue", trail_states=("up_continue",)),
    StrategyArm("buffer0", breakout_buffer=0.0),
    StrategyArm("pullback2", pullback_break_lookback=2),
)


def main() -> int:
    config = load_trend_config(
        overrides={
            "symbols": list(_read_list()),
            "runs": ["breakout", "pullback"],
            "lookback_years": 1,
            "output_dir": "backtest_results",
        },
        lot_lookup=fetch_hk_lot_sizes,
    )
    payload = run_trend_backtest(
        config,
        arms=ARMS,
        persist_cases=False,
    )
    out = Path("backtest_results") / "tuning_round2"
    out.mkdir(parents=True, exist_ok=True)
    compact = _compact(payload)
    (out / "summary.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    text = _markdown(compact)
    (out / "compare.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\n对照: {out / 'compare.md'}")
    return 1 if compact["failures"] else 0


def _read_list():
    path = PROJECT_ROOT / "scripts" / "hkconnect_06Oct.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        code = line.split("#", 1)[0].strip()
        if code:
            yield code


def _compact(payload: dict) -> dict:
    rows = []
    for row in payload["results"]:
        audit = row.get("audit") or {}
        trips = []
        for trip in audit.get("round_trips") or []:
            trips.append({
                "exit_reason": trip.get("exit_reason"),
                "r_multiple": trip.get("r_multiple"),
                "return_net": trip.get("return_net"),
                "pnl_net": trip.get("pnl_net"),
                "entry": trip.get("entry"),
                "exit": trip.get("exit"),
            })
        entries = [
            item.get("position_fraction")
            for item in row.get("trades") or []
            if item.get("action") == "入场"
        ]
        rows.append({
            "symbol": row["symbol"],
            "arm": row.get("arm"),
            "run": row["run"],
            "repaired_bars": row.get("repaired_bars") or 0,
            "state_days": row.get("state_days") or {},
            "total_return": row["metrics"]["total_return"],
            "buy_hold_return": row["metrics"]["buy_hold_return"],
            "dca_return": row["metrics"]["dca_return"],
            "n_trades": row["metrics"]["n_trades"],
            "n_round_trips": row["metrics"]["n_round_trips"],
            "win_rate": row["metrics"].get("win_rate"),
            "blocked": (audit.get("blocked") or {}).get("环境不允许", 0),
            "cancels": audit.get("cancels") or {},
            "entries": len(entries),
            "position_fractions": entries,
            "round_trips": trips,
        })
    failures = [
        {"symbol": item["symbol"], "name": item.get("name"), "error": item["error"]}
        for item in payload.get("failures") or []
    ]
    return {
        "generated_at": payload.get("generated_at"),
        "window": {
            "start": payload["config"]["start"],
            "end": payload["config"]["end"],
        },
        "rows": rows,
        "failures": failures,
    }


def _markdown(compact: dict) -> str:
    rows = compact["rows"]
    failures = compact["failures"]
    symbols = {row["symbol"] for row in rows}
    repaired = len({row["symbol"] for row in rows if row["repaired_bars"]})
    lines = [
        "# 调参对照",
        "",
        f"- 生成时间: {compact.get('generated_at')}",
        f"- 区间: {compact['window']['start']} ~ {compact['window']['end']}",
        f"- 完成 {len(symbols)} 只，跳过 {len(failures)} 只，其中 {repaired} 只修补过 K 线包络",
        "",
        "四组用的是同一批结构。baseline 是当前系数。trail_continue 只在上涨延续里抬止损。buffer0 只把策略 A 的突破缓冲改为 0。pullback2 只把策略 B 的前高窗口改为 2 根。",
        "",
    ]
    if failures:
        counted = Counter(_bucket(item["error"]) for item in failures)
        lines.append("## 跳过")
        lines.append("")
        for name, count in counted.most_common():
            lines.append(f"- {name}: {count}")
        lines.append("")
    lines.append("## 账户")
    lines.append("")
    lines.append("| 组别 | 策略 | 账户收益中位 | 来回中位 | 入场中位 | 环境不允许中位 | R 中位 | 胜率 | 持有天数中位 | 跟踪止损 | 初始止损 | 趋势失效 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in [item.key for item in ARMS]:
        for run, label in (("breakout", "策略 A"), ("pullback", "策略 B")):
            picked = [row for row in rows if row["arm"] == arm and row["run"] == run]
            if not picked:
                continue
            lines.append("| " + " | ".join([
                arm,
                label,
                _pct(_median(row["total_return"] for row in picked)),
                f"{_median(row['n_round_trips'] for row in picked):.0f}",
                f"{_median(row['entries'] for row in picked):.0f}",
                f"{_median(row['blocked'] for row in picked):.0f}",
                f"{_median(_trip_values(picked, 'r_multiple')):.2f}",
                _pct(_win(picked)),
                f"{_median(_hold_days(picked)):.0f}",
                _pct(_exit_share(picked, "跟踪止损")),
                _pct(_exit_share(picked, "止损")),
                _pct(_exit_share(picked, "趋势失效")),
            ]) + " |")
    lines.append("")
    lines.append("盈亏倍数、胜率、持有天数和退出占比按来回计算，其余按股票。")
    lines.append("")
    return "\n".join(lines)


def _bucket(message: str) -> str:
    if "包不住" in message:
        return "高低价仍包不住"
    if "必须为正" in message:
        return "价格非正"
    if "少于结构分析" in message:
        return "K 线太短"
    if "没有不复权" in message:
        return "没有不复权数据"
    if "不是整数" in message:
        return "拆股股数不是整数"
    return message.split("，")[0][:60]


def _median(values) -> float:
    data = [value for value in values if value is not None]
    if not data:
        return 0.0
    return float(statistics.median(data))


def _pct(value) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.2f}%"


def _trip_values(rows, key):
    for row in rows:
        for trip in row["round_trips"]:
            if trip.get(key) is not None:
                yield trip[key]


def _win(rows) -> float:
    trips = [trip for row in rows for trip in row["round_trips"]]
    if not trips:
        return 0.0
    wins = sum(1 for trip in trips if (trip.get("pnl_net") or 0) > 0)
    return wins / len(trips)


def _hold_days(rows):
    for row in rows:
        for trip in row["round_trips"]:
            start = trip.get("entry")
            end = trip.get("exit")
            if not start or not end:
                continue
            yield (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days


def _exit_share(rows, reason: str) -> float:
    trips = [trip for row in rows for trip in row["round_trips"]]
    if not trips:
        return 0.0
    return sum(1 for trip in trips if trip.get("exit_reason") == reason) / len(trips)


if __name__ == "__main__":
    raise SystemExit(main())
