#!/usr/bin/env bash
# 用已经下载的港股行情生成 qlib bin，并在 Docker 里跑基线。
# 默认只跑 lightgbm。已完成的下载不会重拉。
#
#   bash scripts/run_hk_qlib_baseline.sh
#   bash scripts/run_hk_qlib_baseline.sh --model all

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/Users/leon/PyVenv/q_stock/bin/python"
cd "$ROOT"

echo "把已下载行情写成 qlib bin ..."
"$PYTHON" qlib-zh/hk/fetch_data.py --dump-only

echo "补齐回测用的每手股数 ..."
"$PYTHON" qlib-zh/hk/lots.py

docker run --rm \
  --entrypoint python3 \
  -e PYTHONUNBUFFERED=1 \
  -v "${HOME}/.qlib:/root/.qlib" \
  -v "${ROOT}:/work" \
  -w /work \
  zhuhai123/qlib-rdagent:v1 \
  qlib-zh/hk/run_baseline.py "$@"
