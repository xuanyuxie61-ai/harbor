"""
fd_stencils.py — 高阶有限差分模板生成与算子构造
==================================================

本模块为拓扑绝缘体 BHZ 哈密顿量提供高阶有限差分 (Finite Difference, FD)
模板系数的生成与验证。在将连续 k·p 模型离散化到实空间网格时,
有限差分精度直接决定边界态色散关系的数值误差和 von Neumann 稳定性区域。

核心公式
--------
**一阶导数 (2p 阶精度中心差分):**

    ∂f/∂x ≈ (1/h) Σ_{j=1}^{p} c_j^{(p)} [f(x+jh) - f(x-jh)]

其中系数 c_j^{(p)} 由 Taylor 展开匹配确定:

    Σ_{j=1}^{p} c_j^{(p)} j^{2m-1} = δ_{m,1} / 2,   m = 1, ..., p

精确解为:
    c_j^{(p)} = (-1)^{j+1} / (2j) × Π_{k=1,k≠j}^{p} k²/(k²-j²)

**二阶导数 (2p 阶精度中心差分):**

    ∂²f/∂x² ≈ (1/h²) {Σ_{j=1}^{p} d_j^{(p)} [f(x+jh) + f(x-jh)] - 2f(x) Σ_{j=1}^{p} d_j^{(p)}}

其中系数 d_j^{(p)} 满足:

    Σ_{j=1}^{p} d_j^{(p)} j^{2m} = δ_{m,1},   m = 1, ..., p
    (且 Σ d_j 给出对角元贡献)

**修正波数 (Modified Wavenumber):**

    k*_1(h) = (1/h) Σ_j c_j sin(jkh)     → 一阶导数的等效波数
    k*_2(h) = (1/h²) [2 Σ_j d_j (cos(jkh) - 1)]  → 二阶导数的等效波数

色散误差: ε(k) = |k* - k| / |k|

来源映射
--------
- 1435_zoomin: 利用多项式求根框架验证 FD 系数的多项式精确度
- 930_pyramid_exactness: 多项式精确度测试的思想用于验证 FD 模板
"""

import numpy as np
from numpy.polynomial import polynomial as P
from typing import Tuple, List, Dict


def first_derivative_coefficients(p: int) -> np.ndarray:
    """
    计算 2p 阶精度一阶导数中心差分系数。

    对于 p=1 (2阶精度): c = [1/2]
    对于 p=2 (4阶精度): c = [2/3, -1/12]
    对于 p=3 (6阶精度): c = [3/4, -3/20, 1/60]
    对于 p=4 (8阶精度): c = [4/5, -1/5, 4/105, -1/280]

    算法: 求解 Vandermonde 型线性方程组
        Σ_{j=1}^{p} c_j (2j-1)^{2m-1} = δ_{m,1} × (2m-1)! / 2

    Parameters
    ----------
    p : int
        半带宽, 精度阶数为 2p

    Returns
    -------
    coeffs : ndarray, shape (p,)
        差分系数 [c_1, c_2, ..., c_p]
    """
    if p < 1 or p > 10:
        raise ValueError(f"半带宽 p={p} 超出合理范围 [1, 10]")

    # 构建线性方程组: 对 j=1,...,p 的系数 c_j
    # 约束: Σ_j c_j * j^{2m-1} = delta_{m,1} * 1/2
    # m = 1, ..., p
    A_mat = np.zeros((p, p))
    rhs = np.zeros(p)
    for m in range(1, p + 1):
        for j_idx in range(p):
            j = j_idx + 1
            A_mat[m - 1, j_idx] = j ** (2 * m - 1)
        rhs[m - 1] = 0.5 if m == 1 else 0.0

    coeffs = np.linalg.solve(A_mat, rhs)

    # 数值精度验证: 检查系数是否满足反对称约束
    residual = A_mat @ coeffs - rhs
    if np.max(np.abs(residual)) > 1e-12:
        # 使用高精度求解器重新计算
        coeffs = np.linalg.lstsq(A_mat, rhs, rcond=None)[0]

    return coeffs


def second_derivative_coefficients(p: int) -> np.ndarray:
    """
    计算 2p 阶精度二阶导数中心差分系数。

    二阶导数离散化:
        ∂²f/∂x² ≈ (1/h²) [Σ_{j=1}^{p} d_j (f(x+jh) + f(x-jh)) - 2(Σ_j d_j) f(x)]

    其中 d_j 满足:
        Σ_{j=1}^{p} d_j j^{2m} = δ_{m,1},  m = 1, ..., p

    对角元 (f(x) 的系数): -2 Σ_{j=1}^{p} d_j

    Parameters
    ----------
    p : int
        半带宽

    Returns
    -------
    d_coeffs : ndarray, shape (p,)
        二阶差分非对角系数
    d_diag : float
        对角元系数 (负值)
    """
    if p < 1 or p > 10:
        raise ValueError(f"半带宽 p={p} 超出合理范围 [1, 10]")

    A_mat = np.zeros((p, p))
    rhs = np.zeros(p)
    for m in range(1, p + 1):
        for j_idx in range(p):
            j = j_idx + 1
            A_mat[m - 1, j_idx] = j ** (2 * m)
        rhs[m - 1] = 1.0 if m == 1 else 0.0

    d_coeffs = np.linalg.solve(A_mat, rhs)
    d_diag = -2.0 * np.sum(d_coeffs)

    return d_coeffs, d_diag


def modified_wavenumber_first(kh_array: np.ndarray, p: int) -> np.ndarray:
    """
    计算一阶导数的修正波数 k*₁(kh)。

    对于连续一阶导数: ∂/∂x → ik
    离散后: ∂/∂x → (i/h) Σ_{j=1}^{p} c_j sin(jkh) × 2

    修正波数: k*₁h = 2 Σ_{j=1}^{p} c_j sin(j·kh)

    色散误差度量: |k*₁h - kh| / |kh|

    Parameters
    ----------
    kh_array : ndarray
        无量纲波数 kh ∈ [0, π]
    p : int
        半带宽

    Returns
    -------
    k_star_h : ndarray
        修正波数 k*₁·h
    """
    coeffs = first_derivative_coefficients(p)
    k_star_h = np.zeros_like(kh_array)
    for j_idx in range(p):
        j = j_idx + 1
        k_star_h += 2.0 * coeffs[j_idx] * np.sin(j * kh_array)
    return k_star_h


def modified_wavenumber_second(kh_array: np.ndarray, p: int) -> np.ndarray:
    """
    计算二阶导数的修正波数 k*₂(kh)。

    对于连续二阶导数: ∂²/∂x² → -k²
    离散后: ∂²/∂x² → (1/h²) [2 Σ_j d_j (cos(jkh) - 1)]

    修正波数: (k*₂·h)² = -2 Σ_{j=1}^{p} d_j (cos(j·kh) - 1)
    (注意 (k*₂h)² 应为非负值, 对应 -∂² → k²)

    Parameters
    ----------
    kh_array : ndarray
        无量纲波数 kh ∈ [0, π]
    p : int
        半带宽

    Returns
    -------
    k2_star_h2 : ndarray
        修正波数的平方 (k*₂h)², 应为非负
    """
    d_coeffs, _ = second_derivative_coefficients(p)
    k2_star_h2 = np.zeros_like(kh_array)
    for j_idx in range(p):
        j = j_idx + 1
        k2_star_h2 += -2.0 * d_coeffs[j_idx] * (np.cos(j * kh_array) - 1.0)
    return k2_star_h2


def dispersion_error(kh_array: np.ndarray, p: int,
                     derivative_order: int = 1) -> np.ndarray:
    """
    计算有限差分算子的相对色散误差。

    对于一阶导数: ε₁(kh) = |k*₁h - kh| / |kh|
    对于二阶导数: ε₂(kh) = |(k*₂h)² - (kh)²| / (kh)²

    Parameters
    ----------
    kh_array : ndarray
        无量纲波数 (排除 0 附近以避免除零)
    p : int
        半带宽
    derivative_order : int
        1 或 2

    Returns
    -------
    error : ndarray
        相对色散误差
    """
    kh_safe = np.where(np.abs(kh_array) < 1e-14, 1e-14, kh_array)

    if derivative_order == 1:
        k_star = modified_wavenumber_first(kh_safe, p)
        error = np.abs(k_star - kh_safe) / np.abs(kh_safe)
    elif derivative_order == 2:
        k2_star = modified_wavenumber_second(kh_safe, p)
        exact_k2 = kh_safe ** 2
        error = np.abs(k2_star - exact_k2) / exact_k2
    else:
        raise ValueError(f"不支持的导数阶数: {derivative_order}")

    return error


def fd_stencil_matrix_1d(N: int, p: int, h: float,
                         derivative_order: int = 1,
                         bc_type: str = 'open') -> np.ndarray:
    """
    构造一维有限差分算子的稀疏矩阵表示 (密集格式, 用于小规模验证)。

    对于开边界条件 (Open BC): 在边界处截断模板
    对于周期边界条件 (Periodic BC): 使用环绕连接

    Parameters
    ----------
    N : int
        网格点数
    p : int
        半带宽
    h : float
        网格间距
    derivative_order : int
        1 (一阶导数) 或 2 (二阶导数)
    bc_type : str
        'open' 或 'periodic'

    Returns
    -------
    D : ndarray, shape (N, N)
        有限差分矩阵
    """
    if N < 2 * p + 1:
        raise ValueError(
            f"网格点 N={N} 不足以容纳半带宽 p={p} 的模板 "
            f"(需要 N >= {2*p+1})"
        )

    D = np.zeros((N, N), dtype=np.complex128)

    if derivative_order == 1:
        coeffs = first_derivative_coefficients(p)
        prefactor = 1.0 / h
        for i in range(N):
            for j_idx in range(p):
                j = j_idx + 1
                # 正方向
                i_plus = (i + j) % N if bc_type == 'periodic' else i + j
                # 负方向
                i_minus = (i - j) % N if bc_type == 'periodic' else i - j

                if 0 <= i_plus < N:
                    D[i, i_plus] += prefactor * coeffs[j_idx]
                if 0 <= i_minus < N:
                    D[i, i_minus] -= prefactor * coeffs[j_idx]

    elif derivative_order == 2:
        d_coeffs, d_diag = second_derivative_coefficients(p)
        prefactor = 1.0 / (h * h)
        for i in range(N):
            D[i, i] += prefactor * d_diag
            for j_idx in range(p):
                j = j_idx + 1
                i_plus = (i + j) % N if bc_type == 'periodic' else i + j
                i_minus = (i - j) % N if bc_type == 'periodic' else i - j

                if 0 <= i_plus < N:
                    D[i, i_plus] += prefactor * d_coeffs[j_idx]
                if 0 <= i_minus < N:
                    D[i, i_minus] += prefactor * d_coeffs[j_idx]
    else:
        raise ValueError(f"不支持的导数阶数: {derivative_order}")

    return D


def verify_polynomial_exactness(p: int, derivative_order: int = 1,
                                max_degree: int = None) -> Dict:
    """
    验证有限差分模板对多项式的精确度。

    对 f(x) = x^n, 检查 FD 算子是否精确给出:
        D^(1) x^n = n x^{n-1}     (一阶导数)
        D^(2) x^n = n(n-1) x^{n-2} (二阶导数)

    2p 阶精度模板应精确到 degree = 2p 的多项式。

    借鉴 930_pyramid_exactness 的多项式精确度测试框架。

    Parameters
    ----------
    p : int
        半带宽
    derivative_order : int
        1 或 2
    max_degree : int, optional
        测试的最大多项式次数 (默认 2p+2)

    Returns
    -------
    results : dict
        包含 'exact_up_to', 'errors' 等信息
    """
    if max_degree is None:
        max_degree = 2 * p + 2

    N_test = max(4 * p + 5, 20)
    h = 0.01
    x = np.linspace(-1, 1, N_test)
    # 重新计算 h 使其与 x 一致
    h = x[1] - x[0]

    D_mat = fd_stencil_matrix_1d(N_test, p, h, derivative_order, bc_type='open')

    errors = {}
    exact_up_to = -1

    for degree in range(max_degree + 1):
        f_vals = x ** degree
        fd_result = D_mat @ f_vals

        if derivative_order == 1:
            if degree >= 1:
                exact_result = degree * x ** (degree - 1)
            else:
                exact_result = np.zeros_like(x)
        else:
            if degree >= 2:
                exact_result = degree * (degree - 1) * x ** (degree - 2)
            else:
                exact_result = np.zeros_like(x)

        # 排除边界附近 (那里模板被截断)
        interior = slice(p + 1, N_test - p - 1)
        err = np.max(np.abs(fd_result[interior] - exact_result[interior]))
        errors[degree] = float(err)

        if err < 1e-8:
            exact_up_to = degree

    return {
        'half_bandwidth': p,
        'derivative_order': derivative_order,
        'expected_exact_up_to': 2 * p,
        'actual_exact_up_to': exact_up_to,
        'errors': errors,
        'passed': exact_up_to >= 2 * p
    }


def get_all_stencil_orders() -> List[int]:
    """返回所有可用的半带宽阶数列表。"""
    return [1, 2, 3, 4]


def stencil_summary() -> str:
    """生成所有阶数 FD 系数的文本摘要。"""
    lines = []
    lines.append("=" * 70)
    lines.append("有限差分模板系数汇总")
    lines.append("=" * 70)

    for p in get_all_stencil_orders():
        lines.append(f"\n--- 半带宽 p={p} (精度 {2*p} 阶) ---")

        c = first_derivative_coefficients(p)
        lines.append(f"  一阶导数系数 c_j:")
        for j_idx, cj in enumerate(c):
            lines.append(f"    c_{j_idx+1} = {cj:+.15e}")

        d, d0 = second_derivative_coefficients(p)
        lines.append(f"  二阶导数系数 d_j:")
        lines.append(f"    d_0 (对角) = {d0:+.15e}")
        for j_idx, dj in enumerate(d):
            lines.append(f"    d_{j_idx+1} = {dj:+.15e}")

    return "\n".join(lines)
