#!/usr/bin/env bash
# 用已经下载的港股行情生成 qlib bin，并在 Docker 里跑基线。
# 默认只跑 lightgbm。已完成的下载不会重拉。
#
#   bash scripts/run_hk_qlib_baseline.sh
#   bash scripts/run_hk_qlib_baseline.sh --model all
#   PORTFOLIO_POLICY_ENABLED=1 SCORE_GATE_MODE=disabled bash scripts/run_hk_qlib_baseline.sh --model xgboost,catboost

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/usr/bin/python3"
cd "$ROOT"

echo "把已下载行情写成 qlib bin ..."
"$PYTHON" qlib-zh/hk/fetch_data.py --dump-only

echo "补齐回测用的每手股数 ..."
"$PYTHON" qlib-zh/hk/lots.py

policy_env=()
for name in PORTFOLIO_POLICY_ENABLED SCORE_GATE_MODE SCORE_GATE_PROFILE PORTFOLIO_MAX_HOLDINGS ROLLING_MODE BASE_CASH_RATIO RISK_SCALE; do
  if [[ -n "${!name:-}" ]]; then
    policy_env+=(-e "${name}=${!name}")
  fi
done

docker run --rm \
  --entrypoint python3 \
  -e PYTHONUNBUFFERED=1 \
  -e CUDA_VISIBLE_DEVICES= \
  "${policy_env[@]}" \
  -v "${ROOT}:/work" \
  -w /work \
  zhuhai123/qlib-rdagent:v1 \
  qlib-zh/hk/run_baseline.py "$@"
