#!/usr/bin/env bash
# 一次跑多只股票的趋势结构。
# 固定为日线、回看 1 年、参数版本 structure-params-v0。
#
# 不传参数时读取 scripts/holdingshares_06Oct.txt。
#
# 用法:
#   bash scripts/run_trend_structure.sh
#   bash scripts/run_trend_structure.sh 00700 09988 NVDA
#   bash scripts/run_trend_structure.sh --symbols-file symbols.txt
#   bash scripts/run_trend_structure.sh --price-axis uniform --symbol 00700 600519

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "找不到 python3 或 python" >&2
  exit 1
fi

HOLDINGS_FILE="$ROOT/scripts/holdingshares_06Oct.txt"

if [[ $# -eq 0 ]]; then
  exec "$PYTHON" scripts/run_trend_structure.py \
    --years 1 \
    --param-version structure-params-v0 \
    --symbols-file "$HOLDINGS_FILE"
fi

if [[ "$1" == -* ]]; then
  exec "$PYTHON" scripts/run_trend_structure.py \
    --years 1 \
    --param-version structure-params-v0 \
    "$@"
fi

exec "$PYTHON" scripts/run_trend_structure.py \
  --years 1 \
  --param-version structure-params-v0 \
  --symbol "$@"
