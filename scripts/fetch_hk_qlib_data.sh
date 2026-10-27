#!/usr/bin/env bash
# 拉取港股日线，清洗后写成 qlib bin。中断后直接再跑，已完成的股票会跳过。
#
#   bash scripts/fetch_hk_qlib_data.sh --symbols 00700,02800
#   bash scripts/fetch_hk_qlib_data.sh --limit 20
#   bash scripts/fetch_hk_qlib_data.sh
#   bash scripts/fetch_hk_qlib_data.sh --dump-only
#
# 默认股票池是东财当前港股。输出在 ~/.qlib/qlib_data/hk_data_20261016/。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/usr/bin/python3"
cd "$ROOT"

exec "$PYTHON" qlib-zh/hk/fetch_data.py "$@"
