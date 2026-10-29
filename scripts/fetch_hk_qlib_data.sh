#!/usr/bin/env bash
# 在 qlib Docker 镜像里拉取港股日线，清洗后写成 qlib bin。
# 镜像里有 akshare，但没有 yfinance 和 fake-useragent。第一次运行会装到工程目录。
# 行情写在工程目录 qlib-zh/hk/data，不使用 ~/.qlib。
#
#   bash scripts/fetch_hk_qlib_data.sh --symbols 00700,02800
#   bash scripts/fetch_hk_qlib_data.sh --limit 20
#   bash scripts/fetch_hk_qlib_data.sh
#   bash scripts/fetch_hk_qlib_data.sh --dump-only
#
# 默认股票池是东财当前港股。输出在 qlib-zh/hk/data/hk_data_20261016/。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

env_file=()
if [[ -f "${ROOT}/.env" ]]; then
  cleaned="$(mktemp)"
  trap 'rm -f "${cleaned}"' EXIT
  python3 - "${ROOT}/.env" "${cleaned}" <<'PY'
import sys
from pathlib import Path

source, target = sys.argv[1:]
lines = []
for raw in Path(source).read_text(encoding="utf-8").splitlines():
    stripped = raw.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        continue
    key, value = stripped.split("=", 1)
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        lines.append(f"{key}={value[1:-1]}")
        continue
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    lines.append(f"{key}={value}")
Path(target).write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
  env_file=(--env-file "${cleaned}")
fi

docker run --rm \
  --entrypoint bash \
  -e PYTHONUNBUFFERED=1 \
  "${env_file[@]}" \
  -v "${ROOT}:/work" \
  -w /work \
  zhuhai123/qlib-rdagent:v1 \
  -c 'set -euo pipefail
deps=/work/qlib-zh/hk/data/.pydeps
export PYTHONPATH="${deps}${PYTHONPATH:+:$PYTHONPATH}"
if ! python3 -c "import yfinance,fake_useragent" >/dev/null 2>&1; then
  echo "安装 yfinance 和 fake-useragent 到 ${deps}"
  python3 -m pip install --target "${deps}" yfinance fake-useragent
fi
exec python3 qlib-zh/hk/fetch_data.py "$@"
' -- "$@"
