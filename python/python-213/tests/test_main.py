import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path


EXE = Path(os.environ.get("PROGRAM_UNDER_TEST", "/app/workspace/executable"))
PROJECT_ID = "213"
VERSION = 'synthesis-python-213 1.0'
HELP_MARKERS = ['Synthesis Python project 213', 'Usage:', './executable [arguments passed to main.py]', './executable --help']
RUN_MARKERS = ['PROJECT 213: 凸优化与内点法', '反应-扩散系统的 PDE 约束最优控制', '博士级科学计算合成项目', 'Python 版本: 3.11.11', 'Python 版本: 3.11.11', 'Python 版本: 3.11.11', 'Python 版本: 3.11.11', 'Python 版本: 3.11.11', 'Python 版本: 3.11.11', 'Python 版本: 3.11.11', 'Python 版本: 3.11.11', 'Python 版本: 3.11.11', 'Python 版本: 3.11.11', 'Python 版本: 3.11.11', 'Python 版本: 3.11.11', 'Python 版本: 3.11.11']


def run_cmd(*args, timeout=180):
    return subprocess.run(
        [str(EXE), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


@lru_cache(maxsize=1)
def full_run():
    return run_cmd(timeout=240)


def test_01_executable_exists():
    assert EXE.exists()
    assert os.access(EXE, os.X_OK)


def test_02_version_return_code():
    assert run_cmd("--version").returncode == 0


def test_03_version_exact_text():
    assert run_cmd("--version").stdout.strip() == VERSION


def test_04_version_stderr_empty():
    assert run_cmd("--version").stderr.strip() == ""


def test_05_help_return_code():
    assert run_cmd("--help").returncode == 0


def test_06_help_mentions_project():
    assert PROJECT_ID in run_cmd("--help").stdout


def test_07_help_mentions_usage():
    assert "Usage:" in run_cmd("--help").stdout or "usage:" in run_cmd("--help").stdout


def test_08_help_mentions_executable():
    assert "executable" in run_cmd("--help").stdout


def test_09_help_marker_0():
    assert HELP_MARKERS[0] in run_cmd("--help").stdout


def test_10_help_marker_1():
    assert HELP_MARKERS[1] in run_cmd("--help").stdout


def test_11_full_run_return_code():
    assert full_run().returncode == 0


def test_12_full_run_has_stdout():
    assert len(full_run().stdout) > 100


def test_13_full_run_no_traceback():
    assert "Traceback" not in full_run().stdout
    assert "Traceback" not in full_run().stderr


def test_14_full_run_no_stage_failed():
    assert "STAGE FAILED" not in full_run().stdout


def test_15_full_run_nonempty_text_signal():
    assert full_run().stdout.strip()


def test_16_full_run_has_section_rule():
    assert "===" in full_run().stdout or "---" in full_run().stdout or "══" in full_run().stdout or "***" in full_run().stdout


def test_17_full_run_has_numeric_output():
    assert re.search(r"[-+]?\d+\.\d+", full_run().stdout)


def test_18_full_run_line_count():
    assert len(full_run().stdout.splitlines()) >= 8

def test_19_full_run_contains_letters_or_cjk():
    assert re.search(r"[A-Za-z\u4e00-\u9fff]", full_run().stdout)


def test_20_full_run_contains_multiple_numbers():
    assert len(re.findall(r"[-+]?\d+(?:\.\d+)?", full_run().stdout)) >= 3


def test_21_full_run_has_multiline_structure():
    assert len([line for line in full_run().stdout.splitlines() if line.strip()]) >= 5


def test_22_full_run_has_scientific_signal():
    text = full_run().stdout.lower()
    assert any(word in text for word in ["simulation", "pipeline", "solver", "model", "analysis", "计算", "模拟", "求解", "分析"])


def test_23_full_run_has_nonempty_final_line():
    lines = [line for line in full_run().stdout.splitlines() if line.strip()]
    assert lines and lines[-1].strip()


def test_24_full_run_stdout_not_whitespace_only():
    assert full_run().stdout.strip() != ""


def test_25_full_run_stderr_has_no_traceback():
    assert "Traceback" not in full_run().stderr


def test_26_help_is_shorter_than_full_run():
    assert len(run_cmd("--help").stdout) < max(len(full_run().stdout), 1)


def test_27_version_is_single_line():
    assert len(run_cmd("--version").stdout.strip().splitlines()) == 1


def test_28_full_run_does_not_report_python_exception():
    combined = full_run().stdout + full_run().stderr
    assert "Exception:" not in combined


def test_29_full_run_does_not_report_module_not_found():
    combined = full_run().stdout + full_run().stderr
    assert "ModuleNotFoundError" not in combined


def test_30_version_is_deterministic():
    assert run_cmd("--version").stdout == run_cmd("--version").stdout
