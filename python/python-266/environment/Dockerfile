FROM python:3.10-slim

LABEL maintainer="sci-swe"
LABEL version="1.0"
LABEL description="Harbor ProgramBench-style Python 266 reconstruction task"

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash coreutils file binutils strace gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy scipy pytest

WORKDIR /app
RUN mkdir -p /app/workspace /logs/agent /logs/verifier

COPY environment/workspace/. /app/workspace
COPY tests/. /tests

RUN chmod +x /app/workspace/executable /tests/test.sh

RUN pip install --no-cache-dir mini-swe-agent==2.2.8

CMD ["/bin/bash"]
