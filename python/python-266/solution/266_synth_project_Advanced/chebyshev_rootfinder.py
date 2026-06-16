"""
chebyshev_rootfinder.py — Chebyshev 代理求根法
=================================================

本模块实现 van Hove 奇点和带隙边缘的精确检测。
融合种子项目:
  - 225_cpr: Chebyshev Proxy Rootfinder (CPR)
  - 549_humps: 测试函数的导数零点检测
  - 302_disk01_rule: Jacobi 矩阵特征值 (用于伴随矩阵)

核心物理:
  van Hove 奇点: dε/dk = 0 的点
  带隙边缘: BZ 边界或能带极值点

  检测方法:
  1. 将 dε/dk 用 Chebyshev 多项式展开
  2. 构建伴随矩阵, 其特征值即 Chebyshev 展开的根
  3. 筛选实数根 (|Im| < τ)
  4. 残差检查: |f(root)| < tol

  Chebyshev 展开:
    f(x) ≈ Σ_{j=0}^{N} a_j T_j(x)
  其中 T_j 为 Chebyshev 多项式, 系数由 DCT 计算。

  伴随矩阵 (colleague matrix):
    C[i,j] 由 Chebyshev 递推关系导出,
    其特征值 = Chebyshev 展开的根。
"""

import numpy as np
from typing import Tuple, List, Optional, Callable
from physical_constants import PI


# ============================================================
# Chebyshev 系数计算
# ============================================================

def chebyshev_coefficients(f_values: np.ndarray) -> np.ndarray:
    """
    计算 Chebyshev 展开系数 (源自 225_cpr)。

    给定 f 在 Chebyshev 节点上的值:
      x_j = cos(π(j + 0.5) / N),  j = 0, ..., N-1

    Chebyshev 系数由 DCT-I 计算:
      a_k = (2/N) Σ_{j=0}^{N-1} f(x_j) cos(π k (j + 0.5) / N)

    其中 a_0 需除以 2。

    Parameters
    ----------
    f_values : np.ndarray, shape (N,)
        f 在 N 个 Chebyshev 节点上的值

    Returns
    -------
    coeffs : np.ndarray, shape (N,)
        Chebyshev 系数 a_0, a_1, ..., a_{N-1}
    """
    N = len(f_values)
    coeffs = np.zeros(N)

    for k in range(N):
        s = 0.0
        for j in range(N):
            s += f_values[j] * np.cos(PI * k * (j + 0.5) / N)
        coeffs[k] = 2.0 * s / N

    coeffs[0] /= 2.0
    return coeffs


def chebyshev_nodes(n_points: int) -> np.ndarray:
    """
    Chebyshev 节点 (源自 225_cpr)。

    x_j = cos(π(j + 0.5) / N),  j = 0, ..., N-1

    这些节点在 [-1, 1] 上非均匀分布,
    在端点附近更密集。

    Parameters
    ----------
    n_points : int

    Returns
    -------
    nodes : np.ndarray, shape (N,)
    """
    j = np.arange(n_points)
    return np.cos(PI * (j + 0.5) / n_points)


def chebyshev_evaluate(coeffs: np.ndarray, x: np.ndarray
                         ) -> np.ndarray:
    """
    Clenshaw 算法计算 Chebyshev 展开 (源自 225_cpr)。

    f(x) = Σ_{k=0}^{N-1} a_k T_k(x)

    使用 Clenshaw 递推:
      b_{N+1} = 0, b_N = 0
      b_k = 2x·b_{k+1} - b_{k+2} + a_k  for k = N-1, ..., 0
      f(x) = b_0 - x·b_1 = (a_0 + b_1·x + b_2·(-1))

    Parameters
    ----------
    coeffs : np.ndarray
        Chebyshev 系数
    x : np.ndarray
        计算点

    Returns
    -------
    f : np.ndarray
    """
    N = len(coeffs)
    if N == 0:
        return np.zeros_like(x)
    if N == 1:
        return np.ones_like(x) * coeffs[0]

    b_k_plus2 = np.zeros_like(x)
    b_k_plus1 = np.zeros_like(x)

    for k in range(N - 1, 0, -1):
        b_k = 2.0 * x * b_k_plus1 - b_k_plus2 + coeffs[k]
        b_k_plus2 = b_k_plus1
        b_k_plus1 = b_k

    return coeffs[0] + x * b_k_plus1 - b_k_plus2


# ============================================================
# 伴随矩阵与求根
# ============================================================

def companion_matrix_chebyshev(coeffs: np.ndarray) -> np.ndarray:
    """
    构建 Chebyshev 伴随矩阵 (colleague matrix, 源自 225_cpr)。

    对于 Chebyshev 展开 f(x) = Σ a_k T_k(x),
    伴随矩阵 C 的特征值即为 f 的根。

    C 的结构 (N-1 × N-1):
    C = J - (1/(2a_{N-1})) · e_{N-2} · a^T

    其中 J 为截断的 Jacobi 矩阵:
    J[j,j+1] = J[j+1,j] = 1/2  for j = 1,...,N-3
    J[0,1] = J[1,0] = 1/√2

    参数 a = [a_0, a_1, ..., a_{N-2}]^T

    Parameters
    ----------
    coeffs : np.ndarray, shape (N,)
        Chebyshev 系数 (a_{N-1} ≠ 0)

    Returns
    -------
    C : np.ndarray, shape (N-1, N-1)
        伴随矩阵
    """
    N = len(coeffs)
    if N <= 1:
        return np.zeros((0, 0))

    # 截断: 去掉最高阶系数
    n = N - 1  # 伴随矩阵大小

    # Jacobi 矩阵 (Chebyshev 递推)
    J = np.zeros((n, n))
    for i in range(n - 1):
        if i == 0:
            J[i, i + 1] = 1.0 / np.sqrt(2.0)
            J[i + 1, i] = 1.0 / np.sqrt(2.0)
        else:
            J[i, i + 1] = 0.5
            J[i + 1, i] = 0.5

    # 秩-1 修正
    a_lead = coeffs[-1]
    if abs(a_lead) < 1e-30:
        return J  # 退化情况

    # 修正向量
    correction = np.zeros(n)
    for i in range(n):
        if i == 0:
            correction[i] = coeffs[i] / (np.sqrt(2.0) * a_lead)
        else:
            correction[i] = coeffs[i] / (2.0 * a_lead)

    # 最后一行修正
    C = J.copy()
    C[-1, :] -= correction

    return C


def find_roots_chebyshev(func, a: float, b: float,
                           n_chebyshev: int = 64,
                           tau_imag: float = 1e-8,
                           sigma_real: float = 1e-6,
                           residual_tol: float = 1e-10
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chebyshev 代理求根 (CPR, 源自 225_cpr)。

    在 [a, b] 上找到 f(x) = 0 的所有根。

    算法:
    1. 在 Chebyshev 节点上采样 f
    2. 计算 Chebyshev 系数
    3. 截断小系数 (提高条件数)
    4. 构建伴随矩阵
    5. 求特征值 → 根
    6. 筛选: 实根 (|Im| < τ) 且在 [a,b] 内
    7. 残差检查: |f(root)| < tol

    Parameters
    ----------
    func : callable
        被求根函数
    a, b : float
        搜索区间
    n_chebyshev : int
        Chebyshev 展开阶数
    tau_imag : float
        虚部阈值 (筛选实根)
    sigma_real : float
        实部容限 (超出 [a,b] 的容差)
    residual_tol : float
        残差阈值

    Returns
    -------
    roots : np.ndarray
        找到的根
    residuals : np.ndarray
        对应残差 |f(root)|
    """
    # 映射 [a,b] → [-1,1]
    nodes_cheb = chebyshev_nodes(n_chebyshev)
    x_phys = 0.5 * (b - a) * nodes_cheb + 0.5 * (b + a)

    # 采样
    f_values = np.array([func(x) for x in x_phys])

    # Chebyshev 系数
    coeffs = chebyshev_coefficients(f_values)

    # 截断: 丢弃尾部小系数
    max_coeff = np.max(np.abs(coeffs))
    cutoff = 1e-13 * max_coeff
    n_effective = n_chebyshev
    for k in range(n_chebyshev - 1, 0, -1):
        if abs(coeffs[k]) > cutoff:
            n_effective = k + 1
            break

    coeffs_trunc = coeffs[:n_effective]

    if n_effective <= 1:
        return np.array([]), np.array([])

    # 伴随矩阵
    C = companion_matrix_chebyshev(coeffs_trunc)

    if C.size == 0:
        return np.array([]), np.array([])

    # 特征值
    eigenvalues = np.linalg.eigvals(C)

    # 筛选: 实根且在 [a,b] 内
    roots_raw = 0.5 * (b - a) * eigenvalues.real + 0.5 * (b + a)
    mask_real = np.abs(eigenvalues.imag) < tau_imag
    mask_interval = (eigenvalues.real >= -1.0 - sigma_real) & \
                    (eigenvalues.real <= 1.0 + sigma_real)

    mask = mask_real & mask_interval
    candidates = roots_raw[mask]

    # 残差检查
    roots = []
    residuals = []
    for root in candidates:
        if a - 0.01 * abs(b - a) <= root <= b + 0.01 * abs(b - a):
            res = abs(func(root))
            if res < residual_tol:
                roots.append(root)
                residuals.append(res)

    if len(roots) == 0:
        return np.array([]), np.array([])

    roots = np.array(roots)
    residuals = np.array(residuals)

    # 去重
    sorted_idx = np.argsort(roots)
    roots = roots[sorted_idx]
    residuals = residuals[sorted_idx]

    # 合并相近的根
    unique_roots = [roots[0]]
    unique_res = [residuals[0]]
    for i in range(1, len(roots)):
        if abs(roots[i] - unique_roots[-1]) > 1e-8 * abs(b - a):
            unique_roots.append(roots[i])
            unique_res.append(residuals[i])

    return np.array(unique_roots), np.array(unique_res)


# ============================================================
# van Hove 奇点检测
# ============================================================

def find_van_hove_singularities(band_energy_func: Callable,
                                  a: float,
                                  band_index: int = 0,
                                  n_chebyshev: int = 64
                                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    检测能带的 van Hove 奇点 (dε/dk = 0 的点)。

    使用 CPR 方法精确找到群速度为零的 k 点。

    Parameters
    ----------
    band_energy_func : callable
        k → ε_n(k) 的函数
    a : float
        晶格常数
    band_index : int
        能带指标
    n_chebyshev : int
        Chebyshev 展开阶数

    Returns
    -------
    k_vh : np.ndarray
        van Hove 奇点的 k 坐标
    energies : np.ndarray
        对应的能量值
    """
    # 导数函数 (数值差分)
    dk = 1e-7
    def band_derivative(k):
        return (band_energy_func(k + dk) -
                band_energy_func(k - dk)) / (2.0 * dk)

    # 在 BZ 上搜索
    k_min = 0.0
    k_max = PI / a

    k_roots, residuals = find_roots_chebyshev(
        band_derivative, k_min, k_max, n_chebyshev)

    energies = np.array([band_energy_func(k) for k in k_roots])

    return k_roots, energies
