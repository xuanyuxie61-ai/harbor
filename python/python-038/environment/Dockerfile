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
        scipy==1.15.3 \
        matplotlib==3.10.9 \
        pandas==2.2.3 \
        scikit-learn==1.6.1 \
        scikit-image==0.25.2 \
        sympy==1.13.3 \
        networkx==3.4.2 \
        h5py==3.13.0 \
        pillow==11.1.0 \
        tqdm==4.67.1 \
        PyWavelets==1.8.0 \
        statsmodels==0.14.4

# ---------------------------------------------------------------
# 3. 工作目录设置
# ---------------------------------------------------------------
WORKDIR /app

RUN mkdir -p /app/workspace /logs/agent/ /logs/verifier
# ---------------------------------------------------------------
# 4. 代码复制
# ---------------------------------------------------------------
COPY environment/workspace/. /app/workspace
COPY tests/. /tests

# 确保日志和测试目录权限
RUN mkdir -p /logs/verifier && chmod +x /tests/test.sh

# ---------------------------------------------------------------
# 5. 预装 mini-swe-agent
# ---------------------------------------------------------------
RUN pip install --no-cache-dir mini-swe-agent==2.2.8

# ---------------------------------------------------------------
# 6. 默认命令
# ---------------------------------------------------------------
CMD ["/bin/bash"]