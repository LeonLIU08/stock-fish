#!/usr/bin/env bash
# 用一份股票名单跑趋势回测。
# 不传股票时读取 scripts/hkconnect_06Oct.txt。
# 港股没写每手股数时，从 Longbridge static_info 取当前值。
# 默认跑配置里的全部 run，包括策略 D0 和 D1。只要 D1 时加 --runs d1。回看年数沿用配置，默认 3 年。
#
# 用法:
#   bash scripts/run_trend_backtest.sh
#   bash scripts/run_trend_backtest.sh --years 1
#   bash scripts/run_trend_backtest.sh --symbols 00700,09988,NVDA --runs breakout
#   bash scripts/run_trend_backtest.sh --symbols-file other.txt --output-dir backtest_results

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="/usr/bin/python3"
if [[ ! -x "$PYTHON" ]]; then
  echo "找不到 Python: $PYTHON" >&2
  exit 1
fi

HOLDINGS_FILE="$ROOT/scripts/hkconnect_06Oct.txt"
rest=("$@")
has_symbols=0
for arg in "${rest[@]+"${rest[@]}"}"; do
  case "$arg" in
    --symbols|--symbols=*)
      has_symbols=1
      ;;
    --symbols-file|--symbols-file=*)
      has_symbols=1
      ;;
  esac
done

cmd=("$PYTHON" scripts/run_trend_backtest.py)
if [[ "$has_symbols" -eq 0 ]]; then
  cmd+=(--symbols-file "$HOLDINGS_FILE")
fi
if [[ ${#rest[@]} -gt 0 ]]; then
  cmd+=("${rest[@]}")
fi

exec "${cmd[@]}"
