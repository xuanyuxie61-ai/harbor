# -*- coding: utf-8 -*-
"""
polynomial_basis_toolkit.py — 多项式基函数工具箱 (Legendre / Chebyshev / Lagrange)

本模块提供 BAO 分析中用到的全部正交多项式与插值基. 每个多项式族同时给出
(1) 显式求值, (2) 三点递推, (3) 导数, (4) 根 (节点).

种子项目映射
----------
- 990 r8poly : 该项目的 `r8poly_legendre`, `r8poly_chebyshev`,
  `r8poly_lagrange_value`, `r8poly_value_horner` 等被整体移植为 Python.
  与 r8poly 的"系数向量 + Horner 求值"不同, 本模块采用更稳定的三点递推.
- 198 collatz_polynomial : 该项目的"Collatz 迭代 + 模 2 约化"思想被用于
  构造"奇偶分离基" (odd/even basis), 用于红移 bin 的奇偶加权.
"""

from __future__ import annotations
from typing import Tuple, List
import math
import numpy as np


# =============================================================
# Legendre 多项式 P_ell(x)
# =============================================================
def legendre_P(ell: int, x: float) -> float:
    """
    返回 ell 阶 Legendre 多项式 P_ell(x), x ∈ [-1, 1].
    用三点递推:
      (ell+1) P_{ell+1}(x) = (2ell+1) x P_ell(x) - ell P_{ell-1}(x)
    """
    if ell < 0:
        raise ValueError("ell 必须 >= 0")
    if ell == 0:
        return 1.0
    if ell == 1:
        return float(x)
    P_prev = 1.0
    P_curr = float(x)
    for k in range(1, ell):
        P_next = ((2 * k + 1) * x * P_curr - k * P_prev) / (k + 1)
        P_prev = P_curr
        P_curr = P_next
    return P_curr


def legendre_P_array(ell: int, x: np.ndarray) -> np.ndarray:
    """向量化版本."""
    return np.array([legendre_P(ell, float(xi)) for xi in x])


def legendre_P_prime(ell: int, x: float) -> float:
    """
    P_ell'(x) 由公式:
      (1 - x^2) P_ell'(x) = ell (P_{ell-1}(x) - x P_ell(x))
    """
    if abs(abs(x) - 1.0) < 1.0e-12:
        # 边界: P_ell'(1) = ell(ell+1)/2, P_ell'(-1) = (-1)^{ell+1} ell(ell+1)/2
        val = ell * (ell + 1) / 2.0
        return val if x > 0 else ((-1) ** (ell + 1)) * val
    return ell * (legendre_P(ell - 1, x) - x * legendre_P(ell, x)) / (1.0 - x * x)


def legendre_roots(ell: int) -> np.ndarray:
    """返回 P_ell(x) 的 ell 个根 (Gauss-Legendre 节点)."""
    if ell < 1:
        return np.array([])
    nodes, _ = np.polynomial.legendre.leggauss(ell)
    return nodes


# =============================================================
# Chebyshev 多项式 T_n(x) 与 U_n(x)
# =============================================================
def chebyshev_T(n: int, x: float) -> float:
    """Chebyshev 第一类 T_n(x) = cos(n arccos x)."""
    if n == 0:
        return 1.0
    if n == 1:
        return float(x)
    T_prev, T_curr = 1.0, float(x)
    for k in range(1, n):
        T_next = 2.0 * x * T_curr - T_prev
        T_prev, T_curr = T_curr, T_next
    return T_curr


def chebyshev_U(n: int, x: float) -> float:
    """Chebyshev 第二类 U_n(x) = sin((n+1) arccos x) / sqrt(1-x^2)."""
    if n == 0:
        return 1.0
    if n == 1:
        return 2.0 * x
    U_prev, U_curr = 1.0, 2.0 * x
    for k in range(1, n):
        U_next = 2.0 * x * U_curr - U_prev
        U_prev, U_curr = U_curr, U_next
    return U_curr


def chebyshev_T_roots(n: int) -> np.ndarray:
    """T_n(x) 的 n 个根: x_k = cos((2k-1)π/(2n))."""
    k = np.arange(1, n + 1)
    return np.cos((2.0 * k - 1.0) * math.pi / (2.0 * n))


# =============================================================
# Lagrange 插值
# =============================================================
def lagrange_basis(j: int, x_nodes: np.ndarray, x: float) -> float:
    """第 j 个 Lagrange 基 L_j(x) = Π_{k≠j} (x - x_k) / (x_j - x_k)."""
    N = len(x_nodes)
    val = 1.0
    for k in range(N):
        if k == j:
            continue
        denom = x_nodes[j] - x_nodes[k]
        if abs(denom) < 1.0e-30:
            continue
        val *= (x - x_nodes[k]) / denom
    return val


def lagrange_interpolant(x_nodes: np.ndarray, y_values: np.ndarray, x: float
                         ) -> float:
    """Lagrange 插值: p(x) = Σ_j y_j L_j(x)."""
    N = len(x_nodes)
    if N != len(y_values):
        raise ValueError("节点与函数值长度不匹配")
    return sum(y_values[j] * lagrange_basis(j, x_nodes, x) for j in range(N))


def lagrange_derivative_matrix(x_nodes: np.ndarray) -> np.ndarray:
    """
    Lagrange 插值多项式的一阶导数矩阵 D:
      p'(x_i) = Σ_j D_{ij} p(x_j)
    D_{ij} = L_j'(x_i), 由 barycentric 公式给出.
    """
    N = len(x_nodes)
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i == j:
                # 对角: D_{ii} = Σ_{k≠i} 1/(x_i - x_k)
                s = 0.0
                for k in range(N):
                    if k != i:
                        s += 1.0 / (x_nodes[i] - x_nodes[k])
                D[i, i] = s
            else:
                # 非对角
                num = 1.0
                den = x_nodes[i] - x_nodes[j]
                for k in range(N):
                    if k != i and k != j:
                        num *= (x_nodes[i] - x_nodes[k]) / (x_nodes[j] - x_nodes[k])
                D[i, j] = num / den
    return D


# =============================================================
# 多项式 Collatz 迭代 (种子项目 198)
# =============================================================
def collatz_poly_step(coeffs: List[int]) -> List[int]:
    """
    多项式 Collatz 一步 (系数模 2):
      若 P(x) 能被 x 整除: P → P/x
      否则: P → P(x)·(x+1) + 1
    coeffs[0] 为常数项.
    """
    if not coeffs:
        return [1]
    # 去掉前导零
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs.pop()
    if coeffs[0] == 0:
        # 能被 x 整除
        return [(c % 2) for c in coeffs[1:]]
    else:
        # P(x) * (x+1) + 1
        new_len = len(coeffs) + 1
        new = [0] * new_len
        for i, c in enumerate(coeffs):
            new[i] = (new[i] + c) % 2      # × 1
            new[i + 1] = (new[i + 1] + c) % 2  # × x
        new[0] = (new[0] + 1) % 2           # + 1
        return new


def collatz_poly_sequence(coeffs: List[int], max_steps: int = 50
                          ) -> List[List[int]]:
    """生成 Collatz 多项式序列, 直到收敛或达到 max_steps."""
    seq = [list(coeffs)]
    current = list(coeffs)
    for _ in range(max_steps):
        nxt = collatz_poly_step(current)
        seq.append(nxt)
        if nxt == current or sum(nxt) == 0:
            break
        current = nxt
    return seq


# =============================================================
# 奇偶分离基 (红移 bin 奇偶加权)
# =============================================================
def odd_even_basis_decomposition(x: np.ndarray, y: np.ndarray
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    把 y(x) 分解为奇偶部分:
      y_even(x) = (y(x) + y(-x)) / 2
      y_odd(x)  = (y(x) - y(-x)) / 2
    用于红移 bin 的奇偶加权 (消除系统误差).
    """
    # 假设 x 关于 0 对称
    N = len(x)
    if N % 2 != 0:
        raise ValueError("需要偶数个对称点")
    half = N // 2
    y_even = np.zeros(half)
    y_odd = np.zeros(half)
    for i in range(half):
        j = N - 1 - i
        y_even[i] = 0.5 * (y[i] + y[j])
        y_odd[i] = 0.5 * (y[i] - y[j])
    return y_even, y_odd


# =============================================================
# Horner 求值 (种子项目 990 r8poly)
# =============================================================
def horner_eval(coeffs: np.ndarray, x: float) -> float:
    """
    Horner 法多项式求值. coeffs[0] 为最高次项, coeffs[-1] 为常数项.
    """
    val = 0.0
    for c in coeffs:
        val = val * x + c
    return val


# =============================================================
# 自检
# =============================================================
def _self_check() -> None:
    print("[polynomial_basis] Legendre P_3(0.5) =", legendre_P(3, 0.5))
    print("[polynomial_basis] Chebyshev T_4(0.5) =", chebyshev_T(4, 0.5))
    x_nodes = np.array([0.0, 0.5, 1.0, 1.5])
    y_vals = np.array([1.0, 1.25, 2.0, 3.25])
    print("[polynomial_basis] Lagrange interp at 0.7 =",
          lagrange_interpolant(x_nodes, y_vals, 0.7))
    seq = collatz_poly_sequence([1, 0, 1], max_steps=20)
    print(f"[polynomial_basis] Collatz 序列长度: {len(seq)}")


if __name__ == "__main__":
    _self_check()
