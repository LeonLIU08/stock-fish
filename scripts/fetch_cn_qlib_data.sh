#!/usr/bin/env bash
# 拉取中证500从 2010 年至今的日线，写成工程目录里的 qlib bin。
# Qlib 本身仍用 scripts/run_hk_qlib_baseline.sh 里的 Docker 镜像，不在这里安装。
# 行情不写 ~/.qlib。中断后直接再跑，已完成的股票会跳过。
#
#   bash scripts/fetch_cn_qlib_data.sh
#   bash scripts/fetch_cn_qlib_data.sh --symbols 600519,000001
#   bash scripts/fetch_cn_qlib_data.sh --dump-only

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/usr/bin/python3"
cd "$ROOT"

exec "$PYTHON" qlib-zh/cn/fetch_data.py "$@"
