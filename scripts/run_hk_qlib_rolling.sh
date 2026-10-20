#!/usr/bin/env bash
# Alpha158 扩展窗口滚动测试。步长见 baseline.yaml 的 rolling.step，当前为 1 个交易日。
#
#   bash scripts/run_hk_qlib_rolling.sh
#   bash scripts/run_hk_qlib_rolling.sh --dry-run
#   bash scripts/run_hk_qlib_rolling.sh --model xgboost

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

docker run --rm \
  --entrypoint python3 \
  -e PYTHONUNBUFFERED=1 \
  -v "${HOME}/.qlib:/root/.qlib" \
  -v "${ROOT}:/work" \
  -w /work \
  zhuhai123/qlib-rdagent:v1 \
  qlib-zh/hk/run_rolling.py "$@"
