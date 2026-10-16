#!/usr/bin/env bash
# 策略 F。突破和支撑改变仓位，每笔风险预算是策略 E 的 5 倍。
# 名单默认 scripts/hkconnect_06Oct.txt，回看年数沿用配置里的 3 年。
#
#   bash scripts/run_trend_break.sh
#   bash scripts/run_trend_break.sh --years 1
#   bash scripts/run_trend_break.sh --symbols 00700,00001

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec bash scripts/run_trend_backtest.sh \
  --config analysis/backtest/trend-strategy-break.yaml \
  "$@"
