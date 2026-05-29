#!/usr/bin/env python3
"""从 question_down.xlsx 生成 python-XXX 任务目录"""
import pandas as pd
from pathlib import Path

BASE = Path("/mnt/data/zpy/sci-swe/source code/harbor/python")
EXCEL = "/mnt/data/zpy/sci-swe/source code/benchmark/question_down.xlsx"

HEADER = """# 任务：修复挖空代码（Multi-Hole Benchmark）

## 目标
你面对的是一组Python 的科研代码。代码中有多处被"挖空"（函数体、关键逻辑、边界条件等被删除或替换为占位符），导致代码无法正确运行或输出错误结果。

## 工作目录
代码仓库位于 `/app` 目录下。

## 要求
"""

TOML = """version = "1.0"

[task]
name = "sci-swe/{name}"

[metadata]
author_name = "sci-swe"
author_email = "sci-swe@example.com"
difficulty = "hard"
category = "algorithm"
tags = ["multi-file", "optimization"]
expert_time_estimate_min = 180.0
junior_time_estimate_min = 360.0

[verifier]
timeout_sec = 3000.0

[agent]
timeout_sec = 3000.0

[environment]
build_timeout_sec = 300.0
docker_image = "agent-rl-acr-registry-vpc.cn-beijing.cr.aliyuncs.com/sci-swe-python/{name}:latest"
cpus = 2
memory_mb = 8192
storage_mb = 4096
"""

df = pd.read_excel(EXCEL)

for _, row in df.iterrows():
    sn = row["step_number"]
    if sn < 3:
        continue

    name = f"python-{sn:03d}"
    task_dir = BASE / name
    task_dir.mkdir(parents=True, exist_ok=True)

    prompt = str(row["step_description_prompt"]).strip()
    (task_dir / "instruction.md").write_text(HEADER + prompt + "\n", encoding="utf-8")
    (task_dir / "task.toml").write_text(TOML.format(name=name), encoding="utf-8")
    print(f"OK  {name}")