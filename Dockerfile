FROM python:3.11-slim

WORKDIR /app

ARG TARGETARCH

# 基础依赖（gcc + Docker CLI 用于 qlib 推理容器编排）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 安装 Docker CLI（仅客户端，无需 daemon）。按构建架构选择静态包。
RUN case "${TARGETARCH}" in \
      amd64) DOCKER_ARCH=x86_64 ;; \
      arm64) DOCKER_ARCH=aarch64 ;; \
      *) DOCKER_ARCH=x86_64 ;; \
    esac \
    && curl -fsSL "https://download.docker.com/linux/static/stable/${DOCKER_ARCH}/docker-27.3.1.tgz" \
    | tar xzv -C /usr/local/bin --strip-components=1 docker/docker \
    && chmod +x /usr/local/bin/docker

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 从本地复制 sxsc_tushare（私有包，无法从 PyPI 安装）
COPY sxsc_tushare /app/sxsc_tushare/
RUN pip install /app/sxsc_tushare/ 2>/dev/null || echo "sxsc_tushare 安装跳过"

COPY . .

EXPOSE 8000

# 默认：仅启动 Flask API
# 飞书 Bot: 通过 docker-compose 的 command 覆盖 或添加 bot 服务
CMD ["python", "app.py"]
