FROM python:3.10-slim

LABEL maintainer="sci-swe"
LABEL version="1.0"
LABEL description="Harbor ProgramBench-style Python 201 reconstruction task"

# ---------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------
# gcc/g++ are kept for Python packages or agent-generated native extensions.
# file/binutils/strace are useful for observing the provided reference binary.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    coreutils \
    file \
    binutils \
    strace \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------
# 2. Python dependencies
# ---------------------------------------------------------------
# The reference solution imports only numpy/scipy plus standard-library modules.
# pytest is used by the Harbor verifier.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        numpy==2.2.6 \
        scipy==1.15.3 \
        pytest==8.3.5

# ---------------------------------------------------------------
# 3. Workspace layout
# ---------------------------------------------------------------
WORKDIR /app

RUN mkdir -p /app/workspace /logs/agent /logs/verifier

# Agent-visible cleanroom files: README, COPYING, and reference executable.
COPY environment/workspace/. /app/workspace

# Hidden verifier tests.
COPY tests/. /tests

RUN chmod +x /app/workspace/executable /tests/test.sh

# ---------------------------------------------------------------
# 4. Agent runtime
# ---------------------------------------------------------------
RUN pip install --no-cache-dir mini-swe-agent==2.2.8

CMD ["/bin/bash"]

