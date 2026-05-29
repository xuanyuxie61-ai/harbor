"""
topology_validator.py
=====================
分子拓扑校验模块（源自 seed 704_luhn 的校验和算法）

在粗粒化分子动力学（CG-MD）中，每个残基（residue）或原子珠（bead）
都携带一个拓扑标识符。本模块将 Luhn 模 10 校验算法改造为
**分子拓扑完整性校验协议（Molecular Topology Integrity Check, MTIC）**，
用于验证输入的分子拓扑文件是否在传输或编辑过程中出现数字错位。

核心公式（Luhn 算法改造）：
    给定 n 位拓扑码 d_1 d_2 ... d_n，校验和为

        S = sum_{i=n, n-2, ...} d_i
            + sum_{i=n-1, n-3, ...} f(d_i)

    其中 f(x) = sum_of_digits(2*x) = floor(2x/10) + (2x mod 10)。
    若 S mod 10 == 0，则拓扑码合法。
"""

from typing import List


def _sum_of_digits(val: int) -> int:
    """计算 val 的十进制各位数字之和。"""
    s = 0
    while val > 0:
        s += val % 10
        val //= 10
    return s


def topology_checksum(topo_string: str) -> int:
    """
    计算拓扑字符串的 MTIC 校验和。

    Parameters
    ----------
    topo_string : str
        可能包含数字、字母、空格、连字符的原始拓扑描述字符串。
        例如 "POPC-128-LIPID-042"（磷脂酰胆碱残基标识）。

    Returns
    -------
    checksum : int
        模 10 后的校验和；0 表示拓扑描述通过校验。
    """
    digits = [int(ch) for ch in topo_string if ch.isdigit()]
    if len(digits) == 0:
        return 1  # 无数字视为不通过
    n = len(digits)
    total = 0
    # 从右端开始，奇数位（1-indexed）直接相加
    for i in range(n - 1, -1, -2):
        total += digits[i]
    # 偶数位（1-indexed）先乘 2 再逐位求和
    for i in range(n - 2, -1, -2):
        doubled = digits[i] * 2
        total += _sum_of_digits(doubled)
    return total % 10


def topology_check_digit(topo_string: str) -> int:
    """
    计算应追加到 topo_string 末尾的校验位（0–9），使得整体校验和为 0。

    数学推导：设追加字符后总校验和为 0 (mod 10)，则
        check_digit = (10 - checksum(topo_string + '0') % 10) % 10
    """
    tmp = topo_string + "0"
    cs = topology_checksum(tmp)
    return (10 - cs) % 10


def validate_topology(topo_string: str) -> bool:
    """
    判断给定的拓扑字符串是否通过 MTIC 校验。
    """
    return topology_checksum(topo_string) == 0


def generate_topologies(bead_types: List[str], count: int) -> List[str]:
    """
    为粗粒化膜模型批量生成带校验位的拓扑标识符。

    Parameters
    ----------
    bead_types : List[str]
        珠子类型前缀列表，例如 ["PH", "GL", "EST1", "EST2", "C1A", "C2A", "C1B", "C2B"]
        对应典型的 Martini 力场磷脂 beads。
    count : int
        每个类型的残基数。

    Returns
    -------
    topologies : List[str]
        校验通过的拓扑字符串列表。
    """
    results = []
    for bt in bead_types:
        for i in range(count):
            base = f"{bt}-{i:04d}"
            cd = topology_check_digit(base)
            results.append(f"{base}{cd}")
    return results
