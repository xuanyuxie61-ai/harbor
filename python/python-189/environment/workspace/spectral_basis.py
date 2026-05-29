"""
spectral_basis.py

谱基函数与特征表示库

基于种子项目:
  - 326_eigenfaces: PCA / 主成分分析 (Turk-Pentland 技巧)
  - 664_legendre_product_polynomial: 多元 Legendre 乘积多项式

科学应用:
  1. PCA 用于高维观测降维:
     在策略梯度中, 原始观测 o_t ∈ R^D 往往维度过高 (如图像、传感器阵列).
     通过 PCA 投影到低维子空间:
         s_t = V^T (o_t - Ψ),  V ∈ R^{D×d},  d << D
     其中 V 为协方差矩阵前 d 个特征向量.

  2. Legendre 乘积多项式用于策略和值函数的谱逼近:
         f(x) ≈ Σ_{|α|≤p} c_α · P_α(x),  P_α(x) = ∏_{i=1}^m P_{α_i}(x_i)
     Legendre 多项式在 L^2([-1,1]) 上正交:
         ∫_{-1}^{1} P_n(x) P_m(x) dx = 2/(2n+1) δ_{nm}
     这保证了系数估计的数值稳定性.
"""

import numpy as np
from typing import Tuple, List


# ---------------------------------------------------------------------------
# PCA (主成分分析) —— Turk-Pentland 技巧
# ---------------------------------------------------------------------------

def pca_vectors(A: np.ndarray, numvecs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算数据矩阵 A 的前 numvecs 个主成分向量.

    数学推导:
        设 A ∈ R^{m×n} (m 个特征, n 个样本).
        1. 中心化:  A_c = A - Ψ·1^T,  Ψ = mean(A, axis=1)
        2. 小矩阵特征分解:  L = A_c^T A_c ∈ R^{n×n}
           L v = λ v
        3. 恢复大空间特征向量:  u = A_c v,  并归一化 ||u||=1
        4. 特征值缩放:  λ_i ← λ_i / (n-1)

    当 m >> n 时, Turk-Pentland 技巧将 O(m^3) 降为 O(n^3).

    参数:
        A: m×n 数据矩阵 (每列一个样本)
        numvecs: 需要的主成分数

    返回:
        (Vectors, Values, Psi)
        Vectors: m×numvecs 特征向量矩阵
        Values:  n 维全部特征值向量 (降序)
        Psi:     m 维均值向量
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("pca_vectors: A must be 2D")
    m, n = A.shape
    if numvecs > n:
        numvecs = n
    if numvecs < 1:
        raise ValueError("pca_vectors: numvecs must be positive")

    # 计算均值并中心化
    Psi = np.mean(A, axis=1)
    A_centered = A - Psi[:, np.newaxis]

    # 小矩阵特征分解
    L = A_centered.T @ A_centered
    eigvals, eigvecs = np.linalg.eigh(L)
    # 排序 (降序)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # 恢复大空间特征向量
    eigvals = eigvals / max(n - 1, 1)
    Vectors = A_centered @ eigvecs

    # 归一化
    for j in range(n):
        norm = np.linalg.norm(Vectors[:, j])
        if norm > 1.0e-12:
            Vectors[:, j] = Vectors[:, j] / norm
        else:
            Vectors[:, j] = 0.0
            eigvals[j] = 0.0

    # 筛选
    num_good = int(np.sum(eigvals > 1.0e-8))
    if num_good < numvecs:
        numvecs = num_good
    Vectors = Vectors[:, :numvecs]
    return Vectors, eigvals, Psi


def pca_transform(A: np.ndarray, V: np.ndarray, Psi: np.ndarray) -> np.ndarray:
    """将数据投影到 PCA 子空间."""
    A = np.asarray(A, dtype=float)
    if A.ndim == 1:
        return V.T @ (A - Psi)
    return V.T @ (A - Psi[:, np.newaxis])


def pca_reconstruct(z: np.ndarray, V: np.ndarray, Psi: np.ndarray) -> np.ndarray:
    """从 PCA 子空间重构原始数据."""
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        return V @ z + Psi
    return V @ z + Psi[:, np.newaxis]


# ---------------------------------------------------------------------------
# Legendre 多项式与乘积多项式
# ---------------------------------------------------------------------------

def legendre_polynomial_1d(n: int, x: np.ndarray) -> np.ndarray:
    """
    计算 n 次 Legendre 多项式 P_n(x) 在点 x 处的值.

    递推关系 (Bonnet 公式):
        (n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
        P_0(x) = 1,  P_1(x) = x

    正交性:
        ∫_{-1}^{1} P_n(x) P_m(x) dx = 2/(2n+1) δ_{nm}
    """
    x = np.asarray(x, dtype=float)
    if n < 0:
        raise ValueError("legendre_polynomial_1d: n must be non-negative")
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    P_prev2 = np.ones_like(x)
    P_prev1 = x.copy()
    for k in range(1, n):
        P_curr = ((2 * k + 1) * x * P_prev1 - k * P_prev2) / (k + 1)
        P_prev2 = P_prev1
        P_prev1 = P_curr
    return P_prev1


def legendre_product_polynomial(m: int, degrees: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    计算 m 元 Legendre 乘积多项式在多点处的值.

    定义:
        P_{α}(x_1, ..., x_m) = ∏_{i=1}^m P_{α_i}(x_i)

    参数:
        m:       空间维度
        degrees: m 维次数向量 α
        X:       m×N 点矩阵 (每列一个点)

    返回:
        N 维值向量
    """
    degrees = np.asarray(degrees, dtype=int)
    X = np.asarray(X, dtype=float)
    if degrees.ndim != 1 or len(degrees) != m:
        raise ValueError("legendre_product_polynomial: degrees shape mismatch")
    if X.ndim == 1:
        X = X.reshape(m, 1)
    n_pts = X.shape[1]
    v = np.ones(n_pts)
    for i in range(m):
        vi = legendre_polynomial_1d(degrees[i], X[i, :])
        v = v * vi
    return v


def build_legendre_basis(m: int, max_degree: int, X: np.ndarray) -> np.ndarray:
    """
    构造所有总次数 ≤ max_degree 的 m 元 Legendre 乘积多项式基函数矩阵.

    参数:
        m:          空间维度
        max_degree: 最大总次数
        X:          m×N 点矩阵

    返回:
        B: K×N 基函数矩阵, K 为基函数个数
           K = C(m+max_degree, max_degree)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(m, 1)
    # 生成所有 multi-indices α 满足 |α| ≤ max_degree
    indices = _generate_multi_indices(m, max_degree)
    K = len(indices)
    N = X.shape[1]
    B = np.zeros((K, N))
    for k, alpha in enumerate(indices):
        B[k, :] = legendre_product_polynomial(m, alpha, X)
    return B


def _generate_multi_indices(m: int, max_degree: int) -> List[np.ndarray]:
    """递归生成所有 m 元非负整数向量, 满足各分量之和 ≤ max_degree."""
    result = []
    def backtrack(pos: int, current: List[int], remaining: int):
        if pos == m - 1:
            current.append(remaining)
            result.append(np.array(current, dtype=int))
            current.pop()
            return
        for val in range(remaining + 1):
            current.append(val)
            backtrack(pos + 1, current, remaining - val)
            current.pop()
    for total in range(max_degree + 1):
        backtrack(0, [], total)
    return result


# ---------------------------------------------------------------------------
# 谱滤波器 (基于 Bessel 零点)
# ---------------------------------------------------------------------------

def bessel_spectral_filter(freqs: np.ndarray, n: float, k: int,
                           kind: int = 1, bandwidth: float = 1.0) -> np.ndarray:
    """
    基于 Bessel 零点的谱带通滤波器.

    物理意义:
        在圆柱/球形边界条件下, 系统共振频率为 ω_j = z_{nj} / a.
        该滤波器保留围绕这些共振峰的频段:
            H(ω) = ∏_{j=1}^k exp( -(ω - ω_j)^2 / (2 b^2) )

    参数:
        freqs:     频率数组
        n:         Bessel 阶数
        k:         零点数
        kind:      1 (J_n) 或 2 (Y_n)
        bandwidth: 滤波带宽

    返回:
        滤波器响应数组
    """
    from special_functions import bessel_zero
    response = np.ones_like(freqs)
    for j in range(1, k + 1):
        zj = bessel_zero(n, j, kind)
        response = response * np.exp(-(freqs - zj) ** 2 / (2.0 * bandwidth ** 2))
    return response
