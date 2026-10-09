#!/usr/bin/env bash
# 一次跑多只股票的趋势结构。
# 固定为日线、回看 1 年、参数版本 structure-params-v4。
# 这一版一次算出方向、区间形状和停顿。反转倍数是短 1.4、中 2.8、长 5.6。
# 多只股票并行计算。默认同时算 10 只，传了 --workers 就用那个数。
#
# 不传股票时读取 scripts/hkconnect_06Oct.txt。
# 不传 --as-of 时，用运行当天的日期。只写日期时包含当天全部 K 线。
# 页面只保留该时刻已经可用的拐点、线和区间。
#
# 用法:
#   bash scripts/run_trend_structure.sh
#   bash scripts/run_trend_structure.sh --as-of 2026-10-06
#   bash scripts/run_trend_structure.sh 00700 09988 NVDA
#   bash scripts/run_trend_structure.sh --as-of 2026-10-06 00700 09988
#   bash scripts/run_trend_structure.sh --symbols-file symbols.txt
#   bash scripts/run_trend_structure.sh --workers 8 --symbol 00700 600519

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="/Users/leon/PyVenv/q_stock/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "找不到 Python: $PYTHON" >&2
  exit 1
fi

HOLDINGS_FILE="$ROOT/scripts/hkconnect_06Oct.txt"

workers_given=0
workers_value=""
cli_as_of=""
rest=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --as-of)
      if [[ $# -lt 2 || "$2" == -* ]]; then
        echo "缺少 --as-of 的日期，例如 --as-of 2026-10-06" >&2
        exit 1
      fi
      cli_as_of="$2"
      shift 2
      ;;
    --as-of=*)
      cli_as_of="${1#--as-of=}"
      if [[ -z "$cli_as_of" ]]; then
        echo "缺少 --as-of 的日期，例如 --as-of 2026-10-06" >&2
        exit 1
      fi
      shift
      ;;
    --workers)
      if [[ $# -lt 2 || "$2" == -* ]]; then
        echo "缺少 --workers 的数量" >&2
        exit 1
      fi
      workers_given=1
      workers_value="$2"
      shift 2
      ;;
    --workers=*)
      workers_value="${1#--workers=}"
      if [[ -z "$workers_value" ]]; then
        echo "缺少 --workers 的数量" >&2
        exit 1
      fi
      workers_given=1
      shift
      ;;
    *)
      rest+=("$1")
      shift
      ;;
  esac
done

cmd=(
  "$PYTHON" scripts/run_trend_structure.py
  --years 1
  --param-version structure-params-v4
)
if [[ "$workers_given" -eq 1 ]]; then
  cmd+=(--workers "$workers_value")
else
  cmd+=(--workers 10)
fi

if [[ -z "$cli_as_of" ]]; then
  cli_as_of="$(date +%Y-%m-%d)"
fi
cmd+=(--as-of "$cli_as_of")

if [[ ${#rest[@]} -eq 0 ]]; then
  cmd+=(--symbols-file "$HOLDINGS_FILE")
elif [[ "${rest[0]}" == -* ]]; then
  cmd+=("${rest[@]}")
else
  cmd+=(--symbol "${rest[@]}")
fi

exec "${cmd[@]}"
