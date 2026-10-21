#!/usr/bin/env bash
# A股基线按港股滚动的方式跑，不走一次性把全段特征留在内存里的 run_baseline.py。
# Alpha158 先在子进程里按月写到 qlib-zh/cn/cache，进程退出后内存交还系统。
# 之后每一折单独开一个进程训练，预测写入磁盘，进程退出再跑下一折。
# 时间切分和滚动步长读 qlib-zh/cn/baseline.yaml。行情在 qlib-zh/cn/data，不挂 ~/.qlib。
# 默认只跑 lightgbm。已完成的折会跳过。
#
#   bash scripts/run_cn_qlib_baseline.sh
#   bash scripts/run_cn_qlib_baseline.sh --dry-run
#   bash scripts/run_cn_qlib_baseline.sh --model all

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

docker run --rm \
  --entrypoint python3 \
  -e PYTHONUNBUFFERED=1 \
  -v "${ROOT}:/work" \
  -w /work \
  zhuhai123/qlib-rdagent:v1 \
  qlib-zh/hk/run_rolling.py --config qlib-zh/cn/baseline.yaml "$@"
