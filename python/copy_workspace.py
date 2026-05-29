#!/usr/bin/env python3
"""从 Synthesis-project-python 复制 workspace 到 harbor/python-XXX"""
import shutil
from pathlib import Path

SRC_ROOT = Path("/mnt/data/zpy/sci-swe/source code/Synthesis-project-python")
DST_ROOT = Path("/mnt/data/zpy/sci-swe/source code/harbor/python")

SKIP = {"__pycache__", "origin_code", "problem_description",
        "modification_record", "modified_code"}
SKIP_SUFFIX = (".md",)

for src_dir in sorted(SRC_ROOT.glob("*_synth_project")):
    # 提取数字编号
    num = src_dir.name.split("_")[0]  # "003"
    task_name = f"python-{int(num):03d}"
    dst_ws = DST_ROOT / task_name / "environment" / "workspace"

    # 找 multi_benchmark（不含 bug）子目录
    subs = [d for d in src_dir.iterdir()
            if d.is_dir() and "multi_benchmark" in d.name and "bug" not in d.name]

    if not subs:
        print(f"SKIP {task_name}: no multi_benchmark dir found")
        continue

    src_ws = subs[0]

    if not dst_ws.exists():
        continue

    # 清空目标
    if dst_ws.exists():
        shutil.rmtree(dst_ws)
    dst_ws.mkdir(parents=True)

    # 复制文件（跳过 __pycache__, origin_code 等）
    for item in src_ws.iterdir():
        if item.name in SKIP:
            continue
        if item.name.endswith(SKIP_SUFFIX):
            continue
        if item.is_dir():
            shutil.copytree(item, dst_ws / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst_ws / item.name)

    print(f"OK  {task_name}  ({len(list(dst_ws.iterdir()))} items)")