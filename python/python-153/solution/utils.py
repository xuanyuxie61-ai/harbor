"""
utils.py
通用工具函数与辅助计算模块。

核心数学工具:
1. 扩展欧几里得算法与模逆元
2. 快速幂算法
3. 数值稳定性检查
4. 格式化输出工具
"""

import numpy as np
from typing import Tuple, List


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """扩展欧几里得算法。返回 (g, x, y) 使得 a*x + b*y = g = gcd(a, b)。"""
    if b == 0:
        return (abs(a), 1 if a >= 0 else -1, 0)
    g, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return (g, x, y)


def mod_inverse(a: int, m: int) -> int:
    """计算 a 在模 m 下的乘法逆元。若不存在则引发 ValueError。"""
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        raise ValueError(f"Modular inverse does not exist: gcd({a}, {m}) = {g}")
    return x % m


def power_mod(a: int, n: int, m: int) -> int:
    """二进制快速幂: 计算 (a^n) mod m。"""
    if m <= 0:
        raise ValueError("Modulus must be positive")
    if n < 0:
        raise ValueError("Exponent must be non-negative")
    result = 1 % m
    base = a % m
    exp = n
    while exp > 0:
        if exp & 1:
            result = (result * base) % m
        base = (base * base) % m
        exp >>= 1
    return result


def is_power_of_two(n: int) -> bool:
    """检查 n 是否为 2 的幂。"""
    return n > 0 and (n & (n - 1)) == 0


def log2_int(n: int) -> int:
    """计算 floor(log2(n))。"""
    if n <= 0:
        raise ValueError("n must be positive")
    result = 0
    while n > 1:
        n >>= 1
        result += 1
    return result


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法，避免除以零。"""
    if abs(b) < 1e-15:
        return default
    return a / b


def clip_probability(p: float) -> float:
    """将概率值裁剪到 [0, 1] 区间。"""
    return max(0.0, min(1.0, p))


def normalize_vector(v: np.ndarray, ord: int = 2) -> np.ndarray:
    """归一化向量，若范数接近零则返回零向量。"""
    norm = np.linalg.norm(v, ord=ord)
    if norm < 1e-15:
        return np.zeros_like(v)
    return v / norm


def format_scientific(value: float, precision: int = 6) -> str:
    """科学计数法格式化。"""
    return f"{value:.{precision}e}"


def print_section(title: str, width: int = 70) -> None:
    """打印格式化分隔线。"""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subsection(title: str, width: int = 70) -> None:
    """打印子分隔线。"""
    print("\n" + "-" * width)
    print(f"  {title}")
    print("-" * width)


class QuantumGateLibrary:
    """标准单量子门与双量子门库。"""

    @staticmethod
    def I() -> np.ndarray:
        return np.eye(2, dtype=np.complex128)

    @staticmethod
    def X() -> np.ndarray:
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)

    @staticmethod
    def Y() -> np.ndarray:
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

    @staticmethod
    def Z() -> np.ndarray:
        return np.array([[1, 0], [0, -1]], dtype=np.complex128)

    @staticmethod
    def H() -> np.ndarray:
        return np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.0)

    @staticmethod
    def S() -> np.ndarray:
        return np.array([[1, 0], [0, 1j]], dtype=np.complex128)

    @staticmethod
    def T() -> np.ndarray:
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4.0)]], dtype=np.complex128)

    @staticmethod
    def Rx(theta: float) -> np.ndarray:
        return np.array([
            [np.cos(theta / 2.0), -1j * np.sin(theta / 2.0)],
            [-1j * np.sin(theta / 2.0), np.cos(theta / 2.0)]
        ], dtype=np.complex128)

    @staticmethod
    def Ry(theta: float) -> np.ndarray:
        return np.array([
            [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
            [np.sin(theta / 2.0), np.cos(theta / 2.0)]
        ], dtype=np.complex128)

    @staticmethod
    def Rz(theta: float) -> np.ndarray:
        return np.array([
            [np.exp(-1j * theta / 2.0), 0],
            [0, np.exp(1j * theta / 2.0)]
        ], dtype=np.complex128)

    @staticmethod
    def CNOT() -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=np.complex128)
