"""
eigen_analysis.py
=================
交换矩阵特征值分析与自旋波激发谱模块。
融合来源：web_matrix（幂法求主导特征向量/特征值）。

物理背景：
    自旋波的色散关系由交换矩阵的本征值决定。
    对于海森堡铁磁体，色散关系 ω(k) = 2J S [1 - cos(ka)]。
    在阻挫系统中，最低本征值对应 Goldstone 模或软模，
    其特征向量指示自旋关联的主导模式。

    幂法用于求最大模特征值 λ_max；
    逆迭代用于求近零本征值（软模）。
"""

import numpy as np
from typing import Tuple
from utils import EPS_MACHINE, rms_norm


def power_method(
    A: np.ndarray,
    x0: np.ndarray = None,
    max_iter: int = 500,
    tol: float = 1e-12,
) -> Tuple[float, np.ndarray, int]:
    """
    幂法求矩阵的占优特征值与特征向量。
    融合来源：web_matrix/power_rank（转移矩阵的幂迭代）。

    算法：
        x_{k+1} = A x_k / ||A x_k||
        λ_k = x_k^T A x_k / (x_k^T x_k)

    收敛条件：|λ_{k} - λ_{k-1}| < tol 且 ||x_k - x_{k-1}|| < tol。

    参数
    ----
    A : np.ndarray, shape (N, N)
        目标矩阵（此处为交换耦合矩阵或其函数）。
    x0 : np.ndarray, optional
        初始向量，默认随机。
    max_iter : int
        最大迭代次数。
    tol : float
        收敛容差。

    返回
    ----
    lam : float
        占优特征值估计。
    v : np.ndarray
        对应特征向量。
    iters : int
        实际迭代次数。
    """
    N = A.shape[0]
    if x0 is None:
        x = np.random.randn(N)
    else:
        x = x0.copy()
    x = x / (np.linalg.norm(x) + EPS_MACHINE)
    lam_old = 0.0

    for it in range(max_iter):
        y = A @ x
        norm_y = np.linalg.norm(y)
        if norm_y < EPS_MACHINE:
            break
        x_new = y / norm_y
        lam = float(np.dot(x_new, A @ x_new))
        if abs(lam - lam_old) < tol and np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
        lam_old = lam

    # Rayleigh 商精化
    lam = float(np.dot(x, A @ x) / np.dot(x, x))
    return lam, x, it + 1


def inverse_iteration(
    A: np.ndarray,
    shift: float = 0.0,
    x0: np.ndarray = None,
    max_iter: int = 300,
    tol: float = 1e-12,
) -> Tuple[float, np.ndarray, int]:
    """
    带位移的逆迭代，用于求靠近 shift 的特征值。
    在阻挫磁学中特别用于识别接近零的软模（soft modes）。

    迭代格式：
        (A - σI) y = x_k
        x_{k+1} = y / ||y||

    参数
    ----
    A : np.ndarray
        目标矩阵。
    shift : float
        位移 σ。

    返回
    ----
    lam : float
        特征值估计（Rayleigh 商）。
    v : np.ndarray
        特征向量。
    iters : int
        迭代次数。
    """
    N = A.shape[0]
    if x0 is None:
        x = np.random.randn(N)
    else:
        x = x0.copy()
    x = x / (np.linalg.norm(x) + EPS_MACHINE)
    M = A - shift * np.eye(N)

    for it in range(max_iter):
        try:
            y = np.linalg.solve(M, x)
        except np.linalg.LinAlgError:
            # 若 M 接近奇异，加正则化
            M_reg = M + EPS_MACHINE * np.eye(N)
            y = np.linalg.solve(M_reg, x)
        norm_y = np.linalg.norm(y)
        if norm_y < EPS_MACHINE:
            break
        x_new = y / norm_y
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new

    lam = float(np.dot(x, A @ x) / np.dot(x, x))
    return lam, x, it + 1


def spectral_gap_and_soft_modes(
    J: np.ndarray, n_soft: int = 3, tol: float = 1e-10
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    计算交换矩阵的谱隙（spectral gap）与软模。

    对于对称矩阵 J，所有特征值 λ_1 <= λ_2 <= ... <= λ_N。
    谱隙 Δ = λ_2 - λ_1（若 λ_1 为基态）。
    软模对应最小的 n_soft 个特征值。

    返回
    ----
    lambda_min : float
        最小特征值。
    gap : float
        谱隙。
    eigenvalues : np.ndarray
        全部特征值（排序后）。
    eigenvectors : np.ndarray
        对应特征向量。
    """
    # 对于中小规模矩阵，直接稠密对角化更可靠
    w, v = np.linalg.eigh(J)
    lambda_min = float(w[0])
    if w.size > 1:
        gap = float(w[1] - w[0])
    else:
        gap = 0.0
    return lambda_min, gap, w, v


def spin_wave_dispersion_1d(J: float, S: float, a: float, k_points: np.ndarray) -> np.ndarray:
    """
    一维铁磁链的自旋波色散关系（Holstein-Primakoff 近似）。

    海森堡哈密顿量：
        H = -J Σ_{<i,j>} S_i · S_j   (J>0 铁磁)
    线性化后自旋波频率：
        ω(k) = 2 J S [1 - cos(ka)]

    参数
    ----
    J : float
        交换耦合常数。
    S : float
        自旋量子数（经典极限下为自旋模长）。
    a : float
        晶格常数。
    k_points : np.ndarray
        波矢数组。

    返回
    ----
    omega : np.ndarray
        对应频率。
    """
    omega = 2.0 * J * S * (1.0 - np.cos(k_points * a))
    return omega


def correlation_length_from_gap(gap: float, J: float, a: float = 1.0) -> float:
    """
    由谱隙估算自旋关联长度（平均场近似）：
        ξ ~ a / sqrt(Δ / J)
    当 Δ -> 0 时，ξ 发散，标志相变。
    """
    if gap <= EPS_MACHINE:
        return float("inf")
    return a / np.sqrt(gap / J)
