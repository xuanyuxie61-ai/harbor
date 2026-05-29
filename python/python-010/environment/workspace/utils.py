"""
utils.py
========
通用工具与文件 I/O 模块

融入 filum（文件字符统计、行计数、文件名解析）的核心算法，
为 N 体模拟提供日志记录、数据快照元数据管理与文本解析功能。

核心功能
--------
- 模拟输出文件的基本统计（行数、字符数）
- 快照文件名递增与扩展名管理
- 简单字符串工具（大小写转换、字符判断）
- 模拟参数与结果的文本格式化输出
"""

import os
from typing import Tuple


def file_line_count(filename: str) -> int:
    """
    统计文本文件的行数（融入 filum / file_line_count 核心算法）。
    """
    if not os.path.exists(filename):
        return 0
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def file_char_count(filename: str) -> int:
    """
    统计文本文件的字符数（融入 filum / file_char_count 核心算法）。
    """
    if not os.path.exists(filename):
        return 0
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return len(f.read())
    except Exception:
        return 0


def filename_ext_get(filename: str) -> str:
    """
    提取文件扩展名（融入 filum / filename_ext_get 核心算法）。
    """
    idx = filename.rfind(".")
    if idx == -1:
        return ""
    return filename[idx + 1 :]


def filename_ext_swap(filename: str, new_ext: str) -> str:
    """
    替换文件扩展名（融入 filum / filename_ext_swap 核心算法）。
    """
    idx = filename.rfind(".")
    if idx == -1:
        return filename + "." + new_ext
    return filename[: idx + 1] + new_ext


def ch_to_low(ch: str) -> str:
    """
    字符转小写（融入 filum / ch_low 核心算法）。
    """
    if len(ch) == 1 and "A" <= ch <= "Z":
        return chr(ord(ch) + 32)
    return ch


def ch_is_digit(ch: str) -> bool:
    """
    判断字符是否为数字（融入 filum / ch_is_digit 核心算法）。
    """
    return len(ch) == 1 and "0" <= ch <= "9"


def format_simulation_header(title: str, width: int = 70) -> str:
    """
    格式化模拟标题输出框。
    """
    line = "=" * width
    return f"{line}\n{title.center(width)}\n{line}"


def write_simulation_log(
    log_path: str,
    cosmology_params: dict,
    simulation_stats: dict,
) -> None:
    """
    将模拟参数与统计结果写入日志文件。
    """
    lines = []
    lines.append(format_simulation_header("N-body Simulation Log"))
    lines.append("")
    lines.append("[Cosmology Parameters]")
    for key, value in cosmology_params.items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("[Simulation Statistics]")
    for key, value in simulation_stats.items():
        lines.append(f"  {key}: {value}")
    lines.append("")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    # 自检
    print("ch_is_digit('5') =", ch_is_digit("5"))
    print("ch_to_low('A') =", ch_to_low("A"))
    print("filename_ext_get('data.txt') =", filename_ext_get("data.txt"))
    print("filename_ext_swap('data.txt', 'csv') =", filename_ext_swap("data.txt", "csv"))
