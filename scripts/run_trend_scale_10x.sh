#!/usr/bin/env bash
# 策略 E，每笔风险预算 5%、未实现风险合计 10%、名义仓位上限 100%。
# 名单默认 scripts/hkconnect_06Oct.txt，回看年数沿用配置里的 3 年，和
# backtest_results/20261016-101129_1d_trend_state 那次相同。
#
#   bash scripts/run_trend_scale_10x.sh
#   bash scripts/run_trend_scale_10x.sh --years 1
#   bash scripts/run_trend_scale_10x.sh --symbols 00700,00001

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec bash scripts/run_trend_backtest.sh \
  --config analysis/backtest/trend-strategy-scale-10x.yaml \
  "$@"
