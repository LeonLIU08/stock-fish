#!/bin/bash
set -e

# ==========================================
# StockFish + MiroFish 一键部署脚本
# 完整链路: 分析 -> 种子文档 -> OASIS 模拟 -> 预测报告
#
# 模式:
#   1. Docker 部署（默认）: bash run.sh
#   2. 本地直接运行:       bash run.sh --local
#   3. Docker 跳过 MiroFish: bash run.sh --no-mirofish
#   4. 启动飞书 Bot:       bash run.sh --bot        (配合 --local 或 Docker)
#   5. Debug 模式:         bash run.sh --debug
# ==========================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
MODE="${1:-docker}"
OASIS_DEBUG="false"
START_BOT="false"

# 解析参数
for arg in "$@"; do
    case "$arg" in
        --debug)
            OASIS_DEBUG="true"
            ;;
        --bot)
            START_BOT="true"
            ;;
        --local)
            MODE="--local"
            ;;
        --no-mirofish)
            MODE="--no-mirofish"
            ;;
    esac
done

echo "========================================"
echo "  StockFish - A 股分析 + 股价推演"
[ "$OASIS_DEBUG" = "true" ] && echo "  🐛 Debug 模式: 2 Agent / 2轮"
echo "========================================"
echo ""

# ---- .env 检查 ----
if [ ! -f .env ]; then
    echo "[!] .env 文件不存在"
    echo "    请创建 .env 并填入 API Key（参考 .env.example）"
    exit 1
fi

# 导出环境变量供本地模式使用
set -a; source .env; set +a

# ==========================================
# 模式 A: 本地直接运行
# ==========================================
if [ "$MODE" = "--local" ]; then
    echo "[模式] 本地直接运行"
    echo ""

    # 清理旧进程
    lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null || true

    # 检查依赖
    echo "[1/3] 检查 Python 依赖..."
    python -c "import flask; import openai; import sxsc_tushare" 2>/dev/null || {
        echo "  安装依赖中..."
        pip install -r requirements.txt -q
    }
    echo "  ✓ 依赖就绪"

    echo ""
    echo "[2/3] 启动 StockFish（端口 8000）..."
    python app.py &
    STOCKFISH_PID=$!
    echo "  PID: $STOCKFISH_PID"

    # 等待启动 (Flask debug 会重启子进程, 需等待 API 可用)
    echo "  等待 Flask 就绪..."
    for i in $(seq 1 20); do
        if curl -s http://localhost:8000/api/config 2>/dev/null | grep -q backend; then
            echo "  服务就绪"
            break
        fi
        sleep 2
    done

    echo ""
    echo "[3/3] 验证..."
    curl -s http://localhost:8000/api/config | python -m json.tool

    echo ""
    echo "========================================"
    echo "  启动完成！"
    echo "  StockFish: http://localhost:8000"
    echo "  PID: $STOCKFISH_PID"

    # ── 飞书 Bot ──
    if [ "$START_BOT" = "true" ] && [ -n "$LARK_APP_ID" ]; then
        echo ""
        echo "[Bot] 启动飞书 Bot..."
        python integration/lark_bot.py &
        BOT_PID=$!
        echo "  Bot PID: $BOT_PID"
        echo "  飞书 Bot 已启动"
    elif [ "$START_BOT" = "true" ]; then
        echo "  ⚠️  未配置 LARK_APP_ID，跳过 Bot 启动"
    fi

    echo ""
    echo "  停止: kill $STOCKFISH_PID${BOT_PID:+ $BOT_PID}"
    echo "========================================"

    # 保持前台等待
    wait $STOCKFISH_PID
    exit 0
fi

# ==========================================
# 模式 B: Docker 部署
# ==========================================
echo "[1/3] 检查 Docker 环境..."
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker 未安装，请先安装 Docker"
    echo "  或使用本地模式: bash run.sh --local"
    exit 1
fi
echo "  ✓ Docker $(docker --version | cut -d' ' -f3 | tr -d ',')"

# 本地构建 StockFish / MiroFish；Qlib 计算镜像仓库内无 Dockerfile，仍拉取 v1
echo "  准备 Qlib 数据目录..."
mkdir -p "${HOME}/.qlib" "${HOME}/github/qlib-zh/mlruns"

echo "  拉取 Qlib 运行镜像 zhuhai123/qlib-rdagent:v1 ..."
if docker pull zhuhai123/qlib-rdagent:v1; then
    echo "  ✓ zhuhai123/qlib-rdagent:v1"
else
    echo "[ERROR] 无法拉取 Qlib 镜像 zhuhai123/qlib-rdagent:v1"
    echo "        训练/推理/微调都依赖该镜像，请检查网络后重试"
    exit 1
fi

echo ""
echo "[2/3] 本地构建 StockFish / MiroFish ..."
if [ "$MODE" = "--no-mirofish" ]; then
    docker compose build stockfish
else
    docker compose build stockfish mirofish
fi
echo "  ✓ 镜像构建完成"

echo ""
echo "[3/3] 启动服务..."
# 导出 OASIS_DEBUG 供 docker compose 使用
export OASIS_DEBUG
if [ "$MODE" = "--no-mirofish" ]; then
    OASIS_DEBUG="$OASIS_DEBUG" docker compose up -d stockfish
else
    OASIS_DEBUG="$OASIS_DEBUG" docker compose up -d
fi

# ── 飞书 Bot ──
if [ "$START_BOT" = "true" ] && [ -n "$LARK_APP_ID" ]; then
    echo "  启动飞书 Bot..."
    docker compose --profile bot up -d stockfish-bot
    echo "  飞书 Bot 已启动"
fi
echo ""

echo "  等待服务启动..."
for i in $(seq 1 15); do
    STOCKFISH_OK=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ 2>/dev/null || echo "000")
    MIROFISH_OK="000"
    if [ "$MODE" != "--no-mirofish" ]; then
        MIROFISH_OK=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5001/health 2>/dev/null || echo "000")
    fi
    if [ "$STOCKFISH_OK" != "000" ] && { [ "$MODE" = "--no-mirofish" ] || [ "$MIROFISH_OK" != "000" ]; }; then
        break
    fi
    sleep 2
done

echo ""
echo "========================================"
echo "  部署完成！"
echo "========================================"
echo ""
echo "  StockFish (分析+桥接): http://localhost:8000"
if [ "$MODE" != "--no-mirofish" ]; then
    echo "  MiroFish API:          http://localhost:5001"
fi
echo ""
echo "  使用:"
echo "    curl -X POST http://localhost:8000/api/analyze \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"symbol\": \"600519\"}'"
echo ""
echo "  日志: docker compose logs -f stockfish"
echo "  停止: docker compose down"
echo "========================================"
