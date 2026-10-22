#!/usr/bin/env bash
# A股 Alpha158 扩展窗口滚动回测。
# 模型参数与港股 baseline 相同。交易规则、费用和中证500名单在 qlib-zh/cn/baseline.yaml。
#
#   bash scripts/run_cn_qlib_rolling.sh
#   bash scripts/run_cn_qlib_rolling.sh --dry-run
#   bash scripts/run_cn_qlib_rolling.sh --model lightgbm
#   bash scripts/run_cn_qlib_rolling.sh --resume qlib-zh/cn/output/cn_data/20261021-1041-rolling --model lightgbm

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

docker run --rm \
  --entrypoint python3 \
  -e PYTHONUNBUFFERED=1 \
  -e OMP_NUM_THREADS=32 \
  -e MKL_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -v "${ROOT}:/work" \
  -w /work \
  zhuhai123/qlib-rdagent:v1 \
  qlib-zh/hk/run_rolling.py --config qlib-zh/cn/baseline.yaml "$@"
