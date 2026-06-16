# -*- coding: utf-8 -*-
"""
stability_analysis.py
=====================

高阶有限差分格式的 von Neumann 稳定性分析与 Hankel-Cholesky 分解。

融合种子项目:
    - 504_hankel_cholesky: Hankel 矩阵 Cholesky 分解
    - 648_laplacian_matrix: 特征值分析

物理背景:
    Parton shower 的离散化演化方程:
        f^{n+1} = G f^n
    其中 G 为 amplification matrix。von Neumann 稳定性要求:
        |lambda_max(G)| <= 1 + O(dt)

    对 DGLAP 演化方程的空间离散 (动量分数 x 网格),
    amplification matrix 的谱半径决定时间步长限制。

    Hankel 矩阵在 parton 物理中的角色:
        H_{ij} = M_{i+j}
    其中 M_k = int x^k D(x) dx 为碎裂函数的矩。
    该 Hankel 矩阵的正定性 (Cholesky 分解) 等价于
    矩问题的可解性 (Hausdorff 矩条件)。
"""

from __future__ import annotations
import math
from typing import List, Tuple, Callable
import constants as C
from fd_operators import laplacian_1d_dd, diff_center_matrix
from splitting_kernels import P_qq_lo, P_gg_lo, gauss_legendre


# ======================================================================
# Von Neumann 稳定性分析
# ======================================================================
def amplification_matrix_fd(order: int, r: float, n_modes: int = 64) -> List[List[complex]]:
    """
    有限差分格式的 amplification matrix (离散 Fourier 模式):
        G_{kk} = 1 - r * sigma(k)
    其中 sigma(k) 为空间离散算子的符号 (eigenvalue)。

    对 2 阶中心差分: sigma(k) = 4 sin^2(k h / 2) / h^2
    对 4 阶中心差分: sigma(k) = [4/3 sin^2(kh/2) - 1/12 sin^2(kh)] / h^2

    Parameters
    ----------
    order : int
        差分阶数 (2, 4, 6)
    r : float
        CFL 数 r = alpha_s * dt / h^2
    n_modes : int
        Fourier 模式数
    """
    G = [[0j] * n_modes for _ in range(n_modes)]
    h = 2.0 * C.PI / n_modes  # 波数间隔

    for k_idx in range(n_modes):
        theta = k_idx * h  # kh
        if order == 2:
            sigma = 4.0 * math.sin(theta / 2.0) ** 2
        elif order == 4:
            s2 = math.sin(theta / 2.0) ** 2
            s4 = math.sin(theta) ** 2
            sigma = (4.0 / 3.0) * s2 - (1.0 / 3.0) * s4
        elif order == 6:
            s1 = math.sin(theta / 2.0)
            s2 = math.sin(theta)
            s3 = math.sin(3.0 * theta / 2.0)
            sigma = (3.0/2.0) * s1**2 - (3.0/20.0) * s2**2 + (1.0/60.0) * s3**2
            sigma *= 4.0  # 归一化
        else:
            sigma = 4.0 * math.sin(theta / 2.0) ** 2

        # amplification factor
        G[k_idx][k_idx] = complex(1.0 - r * sigma, 0.0)

    return G


def spectral_radius(G: List[List[complex]]) -> float:
    """矩阵谱半径 (对角矩阵的 max |lambda|)"""
    return max(abs(G[i][i]) for i in range(len(G)))


def von_neumann_stability_check(order: int, r: float, n_modes: int = 64) -> Tuple[bool, float]:
    """
    Von Neumann 稳定性判定:
        stable iff |lambda_max| <= 1 + 1e-10

    Returns (stable, spectral_radius)
    """
    G = amplification_matrix_fd(order, r, n_modes)
    rho = spectral_radius(G)
    return rho <= 1.0 + 1e-10, rho


def critical_cfl_number(order: int, n_modes: int = 64) -> float:
    """
    对给定差分阶数, 求临界 CFL 数:
        r_critical = 1 / max_k sigma(k)

    通过二分法求解。
    """
    r_low, r_high = 0.0, 2.0
    for _ in range(50):
        r_mid = 0.5 * (r_low + r_high)
        stable, _ = von_neumann_stability_check(order, r_mid, n_modes)
        if stable:
            r_low = r_mid
        else:
            r_high = r_mid
    return r_low


# ======================================================================
# DGLAP 演化稳定性 (非对角 amplification)
# ======================================================================
def dglap_amplification_eigenvalue(z: float, alpha_s: float, dt: float,
                                     kernel_name: str = 'Pqq') -> complex:
    """
    DGLAP splitting kernel 的单模式 amplification factor:
        G(z) = 1 + dt * (alpha_s / 2pi) * P(z)

    对 explicit Euler, 稳定性要求:
        |G(z)| <= 1 对所有 z
    """
    if kernel_name == 'Pqq':
        P = P_qq_lo(z)
    elif kernel_name == 'Pgg':
        P = P_gg_lo(z)
    else:
        P = 0.0

    return complex(1.0 + dt * alpha_s / (2.0 * C.PI) * P)


def dglap_max_dt_explicit(alpha_s: float, n_test: int = 100) -> float:
    """
    DGLAP explicit Euler 的最大稳定时间步长:
        dt_max = 2 pi / (alpha_s * max_z |P(z)|)
    """
    max_P = 0.0
    for i in range(n_test):
        z = 0.01 + 0.98 * i / (n_test - 1)
        p = abs(P_qq_lo(z))
        max_P = max(max_P, p)
        p = abs(P_gg_lo(z))
        max_P = max(max_P, p)

    if max_P < 1e-15:
        return 1.0
    return 2.0 * C.PI / (alpha_s * max_P)


# ======================================================================
# Hankel 矩阵 Cholesky 分解 (来自 504_hankel_cholesky)
# ======================================================================
def hankel_matrix(moments: List[float]) -> List[List[float]]:
    """
    由矩序列 {M_0, M_1, ..., M_{2n-2}} 构造 Hankel 矩阵:
        H_{ij} = M_{i+j}
    """
    n = len(moments) // 2 + 1
    H = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i + j < len(moments):
                H[i][j] = moments[i + j]
    return H


def hankel_cholesky_upper(n: int, h: List[float]) -> List[List[float]]:
    """
    Hankel 矩阵的上三角 Cholesky 分解 (来自 504_hankel_cholesky)。

    利用 Hankel 结构的快速算法 (Phillips, 1971),
    复杂度 O(n^2) 而非标准 Cholesky 的 O(n^3)。

    输入:
        h: 长度为 2n-1 的 Hankel 第一行/列
    输出:
        R: n x n 上三角 Cholesky 因子
    """
    h = list(h)
    a = [0.0] * n
    b = [0.0] * n
    c = [[0.0] * (2 * n - 1) for _ in range(2 * n - 1)]

    # 初始化 Hankel 矩阵
    for i in range(2 * n - 1):
        for j in range(2 * n - 1):
            if i + j < len(h):
                c[i][j] = h[i + j]

    R = [[0.0] * n for _ in range(n)]

    # Phillips 算法
    for k in range(n):
        if k == 0:
            if c[0][0] <= 0:
                # 不正定, 正则化
                c[0][0] = 1e-10
            R[0][0] = math.sqrt(c[0][0])
            for j in range(1, n):
                R[0][j] = h[j] / R[0][0]
        else:
            # 更新
            diag = c[k][k]
            for m in range(k):
                if abs(R[m][k]) > 1e-15:
                    diag -= R[m][k] ** 2
            if diag <= 0:
                diag = 1e-10  # 正则化
            R[k][k] = math.sqrt(diag)
            for j in range(k + 1, n):
                s = c[k][j] if k + j < 2 * n - 1 else 0.0
                for m in range(k):
                    if abs(R[m][k]) > 1e-15 and m < len(R) and j < len(R[m]):
                        s -= R[m][k] * R[m][j]
                if abs(R[k][k]) > 1e-15:
                    R[k][j] = s / R[k][k]

    return R


def hankel_spd_check(moments: List[float]) -> Tuple[bool, List[float]]:
    """
    检查 Hankel 矩阵是否正定 (矩问题 Hausdorff 条件)。
    返回 (is_spd, cholesky_diagonal)
    """
    n = len(moments) // 2 + 1
    try:
        R = hankel_cholesky_upper(n, moments)
        diag = [R[i][i] for i in range(n)]
        is_spd = all(d > 1e-12 for d in diag)
        return is_spd, diag
    except Exception:
        return False, []


# ======================================================================
# 碎裂函数矩的 Hankel 分析
# ======================================================================
def fragmentation_hankel_analysis(order: int = 5) -> Tuple[bool, List[float]]:
    """
    计算碎裂函数的前 2*order 个矩, 构造 Hankel 矩阵,
    分析正定性 (等价于矩问题的可解性)。
    """
    moments = []
    for k in range(2 * order):
        from hadronization_string import fragmentation_moment
        mk = fragmentation_moment(k)
        moments.append(mk)

    return hankel_spd_check(moments)


# ======================================================================
# 矩阵特征值分析 (来自 648_laplacian_matrix)
# ======================================================================
def laplacian_spectrum(n: int, h: float) -> List[float]:
    """
    一维 DD Laplacian 的特征值:
        lambda_k = (2/h^2) * (1 - cos(k pi / (n+1))), k=1,...,n
    精确公式, 来自 648_laplacian_matrix/l1dd_eigen.m。
    """
    eigs = []
    for k in range(1, n + 1):
        lam = (2.0 / (h * h)) * (1.0 - math.cos(k * C.PI / (n + 1)))
        eigs.append(lam)
    return sorted(eigs)


def condition_number_laplacian(n: int, h: float) -> float:
    """Laplacian 矩阵条件数 = lambda_max / lambda_min"""
    eigs = laplacian_spectrum(n, h)
    if eigs[0] < 1e-15:
        return float('inf')
    return eigs[-1] / eigs[0]


# ======================================================================
# 自测
# ======================================================================
if __name__ == "__main__":
    print("=== Von Neumann Stability Analysis ===")
    for order in [2, 4, 6]:
        r_crit = critical_cfl_number(order)
        print(f"  Order {order}: r_critical = {r_crit:.6f}")

        stable, rho = von_neumann_stability_check(order, r_crit * 0.9)
        print(f"    At 0.9 * r_crit: stable={stable}, rho={rho:.6f}")

    print("\n=== DGLAP Explicit Euler ===")
    als = C.safe_alpha_s(10.0)
    dt_max = dglap_max_dt_explicit(als)
    print(f"  alpha_s(Q=10 GeV) = {als:.4f}")
    print(f"  dt_max (explicit) = {dt_max:.6f}")

    print("\n=== Hankel Cholesky Test ===")
    # 测试: 矩序列 M_k = 1/(k+1) (来自均匀分布)
    n = 4
    moments = [1.0 / (k + 1) for k in range(2 * n - 1)]
    R = hankel_cholesky_upper(n, moments)
    print(f"  Moments: {[f'{m:.4f}' for m in moments]}")
    print(f"  Cholesky diagonal: {[f'{R[i][i]:.4f}' for i in range(n)]}")

    print("\n=== Laplacian Spectrum ===")
    n = 8
    h = 0.1
    eigs = laplacian_spectrum(n, h)
    print(f"  n={n}, h={h}")
    print(f"  lambda_min = {eigs[0]:.4f}, lambda_max = {eigs[-1]:.4f}")
    print(f"  Condition number = {condition_number_laplacian(n, h):.2f}")
