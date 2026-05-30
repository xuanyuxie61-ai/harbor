
import os
from typing import Tuple


def file_line_count(filename: str) -> int:
    if not os.path.exists(filename):
        return 0
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def file_char_count(filename: str) -> int:
    if not os.path.exists(filename):
        return 0
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return len(f.read())
    except Exception:
        return 0


def filename_ext_get(filename: str) -> str:
    idx = filename.rfind(".")
    if idx == -1:
        return ""
    return filename[idx + 1 :]


def filename_ext_swap(filename: str, new_ext: str) -> str:
    idx = filename.rfind(".")
    if idx == -1:
        return filename + "." + new_ext
    return filename[: idx + 1] + new_ext


def ch_to_low(ch: str) -> str:
    if len(ch) == 1 and "A" <= ch <= "Z":
        return chr(ord(ch) + 32)
    return ch


def ch_is_digit(ch: str) -> bool:
    return len(ch) == 1 and "0" <= ch <= "9"


def format_simulation_header(title: str, width: int = 70) -> str:
    line = "=" * width
    return f"{line}\n{title.center(width)}\n{line}"


def write_simulation_log(
    log_path: str,
    cosmology_params: dict,
    simulation_stats: dict,
) -> None:
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

    print("ch_is_digit('5') =", ch_is_digit("5"))
    print("ch_to_low('A') =", ch_to_low("A"))
    print("filename_ext_get('data.txt') =", filename_ext_get("data.txt"))
    print("filename_ext_swap('data.txt', 'csv') =", filename_ext_swap("data.txt", "csv"))
