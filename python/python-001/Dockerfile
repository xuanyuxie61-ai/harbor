FROM python:3.10-slim

LABEL maintainer="sci-swe"
LABEL version="1.0"
LABEL description="MOGOA Advanced Multi-Hole Benchmark"

# ---------------------------------------------------------------
# 1. 系统依赖安装
# ---------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------
# 2. Python 依赖安装
# ---------------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        numpy==2.2.6 \
        matplotlib==3.10.9

# ---------------------------------------------------------------
# 3. 工作目录设置
# ---------------------------------------------------------------
WORKDIR /app

RUN mkdir -p /app/workspace /logs/agent/ /logs/verifier
# ---------------------------------------------------------------
# 4. 代码复制
# ---------------------------------------------------------------
COPY workspace/. /app/workspace

# 确保日志和测试目录权限
RUN mkdir -p /logs/verifier && chmod +x /tests/test.sh

# ---------------------------------------------------------------
# 5. 默认命令
# ---------------------------------------------------------------
CMD ["/bin/bash"]