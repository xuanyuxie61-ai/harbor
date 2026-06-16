"""
covariance_cholesky.py — 协方差算子构造与 Cholesky 分解

科学背景
========
对于定义在 [0,L] 上的平方指数协方差核:

    C(x,x') = σ² · exp( -|x-x'|² / (2·ℓ²) )

其对应的协方差矩阵 Σ ∈ ℝ^{N×N} 为对称正定 (SPD) 矩阵.
为生成具有给定相关结构的高斯随机向量, 需要计算:

    Σ = L_c · L_c^T     (Cholesky 分解)

则 z ~ N(0,I)  ⟹  L_c · z ~ N(0, Σ).

算法来源 (种子项目 026_asa007)
==============================
本模块移植自 Algorithm AS 6 (Healy, 1968):
  - Cholesky 分解:  Σ = U^T · U
  - 带秩亏检测 (nullty) 和错误标志 (efault)
  - 对称矩阵求逆:  Σ^{-1} = (U^{-1}) · (U^{-1})^T

改进之处
========
1. 自动正则化 (jitter):  当对角元 < jitter 时添加微小正值
2. 向量化的行-列扫描
3. 数值条件数估计

核心公式
========
1. Cholesky 递推:  L[j,j] = √(Σ[j,j] - Σ_{k<j} L[j,k]²)
                   L[i,j] = (Σ[i,j] - Σ_{k<j} L[i,k]·L[j,k]) / L[j,j]
2. 正则化:  Σ_reg = Σ + δ·I,  δ = jitter
3. 条件数上界:  cond(Σ) ≈ max(diag(Σ)) / min(diag(Σ))
"""

import numpy as np


class CovarianceKernel:
    """平方指数 (RBF / Gaussian) 协方差核.

    C(x, x') = σ² · exp( -‖x - x'‖² / (2·ℓ²) )

    参数
    ----
    sigma : float
        过程标准差 σ > 0
    length_scale : float
        相关长度 ℓ > 0
    """

    def __init__(self, sigma, length_scale):
        if sigma <= 0:
            raise ValueError(f"σ 必须为正, 得到 σ={sigma}")
        if length_scale <= 0:
            raise ValueError(f"ℓ 必须为正, 得到 ℓ={length_scale}")
        self.sigma = float(sigma)
        self.length_scale = float(length_scale)

    def evaluate(self, x1, x2=None):
        """计算协方差矩阵 Σ[i,j] = C(x1[i], x2[j]).

        若 x2 为 None, 则计算 Σ = C(x1, x1).

        公式:  C(r) = σ² · exp(-r² / (2ℓ²))
        """
        x1 = np.asarray(x1, dtype=float).ravel()
        if x2 is None:
            x2 = x1
        else:
            x2 = np.asarray(x2, dtype=float).ravel()

        # 高效计算平方距离矩阵
        sq_dist = (x1[:, None] - x2[None, :]) ** 2
        return (self.sigma ** 2) * np.exp(-sq_dist / (2.0 * self.length_scale ** 2))

    def evaluate_diagonal(self, x):
        """C(x, x) = σ²  (对角元)."""
        return np.full_like(np.asarray(x, dtype=float), self.sigma ** 2)

    def spectral_density(self, omega):
        """功率谱密度 (Wiener-Khinchin 定理).

        对于平方指数核:
            S(ω) = σ² · ℓ · √(2π) · exp(-ℓ²ω²/2)
        """
        omega = np.asarray(omega, dtype=float)
        return (self.sigma ** 2 * self.length_scale * np.sqrt(2.0 * np.pi)
                * np.exp(-0.5 * self.length_scale ** 2 * omega ** 2))


def cholesky_as006(sigma_matrix, jitter=1.0e-12):
    """Cholesky 分解 (移植自 Algorithm AS 6).

    计算 Σ = L · L^T, 其中 L 为下三角矩阵.

    参数
    ----
    sigma_matrix : ndarray, shape (n, n)
        对称正定矩阵 Σ
    jitter : float
        正则化参数, 添加到对角元以防止奇异性

    返回
    ----
    L : ndarray, shape (n, n)
        下三角 Cholesky 因子
    nullty : int
        秩亏数 (0 = 满秩)
    ifault : int
        错误标志: 0=正常, 1=n<1, 2=非正定, 3=非方阵

    算法
    ====
    for j = 1,...,n:
        s = Σ[j,j] - Σ_{k<j} L[j,k]²
        if s < jitter:
            s += jitter        # 正则化
            nullty += 1
        L[j,j] = √s
        for i = j+1,...,n:
            L[i,j] = (Σ[i,j] - Σ_{k<j} L[i,k]·L[j,k]) / L[j,j]
    """
    n = sigma_matrix.shape[0]
    if n != sigma_matrix.shape[1]:
        return None, 0, 3
    if n < 1:
        return None, 0, 1

    # 对称化 (消除浮点不对称性)
    A = 0.5 * (sigma_matrix + sigma_matrix.T)

    L = np.zeros((n, n))
    nullty = 0

    for j in range(n):
        # 对角元: L[j,j] = sqrt(A[j,j] - sum_{k<j} L[j,k]^2)
        s = A[j, j] - np.sum(L[j, :j] ** 2)
        if s < jitter:
            s = max(s + jitter, jitter)
            nullty += 1
        L[j, j] = np.sqrt(s)

        # 非对角元
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]

    return L, nullty, 0


def syminv_as007(sigma_matrix, jitter=1.0e-12):
    """对称正定矩阵求逆 (移植自 Algorithm AS 7, Healy 1968).

    计算 Σ^{-1} 利用 Cholesky 分解:
        Σ = L · L^T  ⟹  Σ^{-1} = L^{-T} · L^{-1}

    当矩阵接近奇异时, 使用增则正则化并回退到伪逆.

    参数
    ----
    sigma_matrix : ndarray, shape (n, n)
    jitter : float

    返回
    ----
    sigma_inv : ndarray, shape (n, n)
        Σ 的逆 (或广义逆)
    nullty : int
    ifault : int
    """
    n = sigma_matrix.shape[0]
    if n < 1:
        return None, 0, 1

    L, nullty, ifault = cholesky_as006(sigma_matrix, jitter)
    if L is None:
        return None, nullty, ifault

    # 如果秩亏严重, 使用 numpy 伪逆
    if nullty > n // 4:
        try:
            sigma_inv = np.linalg.pinv(sigma_matrix, rcond=1e-10)
            return sigma_inv, nullty, 0
        except np.linalg.LinAlgError:
            return np.eye(n) * 0.0, nullty, 2

    # 正常 Cholesky 求逆
    L_inv = np.zeros((n, n))
    for j in range(n):
        if abs(L[j, j]) < 1.0e-20:
            continue
        L_inv[j, j] = 1.0 / L[j, j]
        for i in range(j + 1, n):
            if abs(L[i, i]) < 1.0e-20:
                continue
            s = np.sum(L[i, j:i] * L_inv[j:i, j])
            L_inv[i, j] = -s / L[i, i]

    sigma_inv = L_inv.T @ L_inv

    # 检查 NaN/Inf
    if not np.all(np.isfinite(sigma_inv)):
        sigma_inv = np.linalg.pinv(sigma_matrix, rcond=1e-10)

    return sigma_inv, nullty, 0


def covariance_condition_estimate(sigma_matrix):
    """估计协方差矩阵的条件数.

    利用 Gershgorin 圆盘定理给出条件数的上界估计.

    参数
    ----
    sigma_matrix : ndarray, shape (n, n)

    返回
    ----
    cond_upper : float
        条件数上界
    diag_min : float
        最小对角元
    diag_max : float
        最大对角元
    """
    n = sigma_matrix.shape[0]
    diag = np.diag(sigma_matrix).copy()
    diag_min = np.min(diag)
    diag_max = np.max(diag)

    # Gershgorin 下界
    off_diag_sum = np.sum(np.abs(sigma_matrix), axis=1) - np.abs(diag)
    lambda_min_gb = np.min(diag - off_diag_sum)
    lambda_max_gb = np.max(diag + off_diag_sum)

    if lambda_min_gb <= 0:
        cond_upper = np.inf
    else:
        cond_upper = lambda_max_gb / lambda_min_gb

    return cond_upper, diag_min, diag_max


def build_precision_from_covariance(sigma_matrix, jitter=1.0e-12):
    """从协方差矩阵构造精度矩阵 (逆协方差).

    精度矩阵 Q = Σ^{-1} 在 Gaussian Markov 随机场中至关重要:
        p(x) ∝ exp(-0.5 · x^T · Q · x)

    返回
    ----
    precision : ndarray
    nullty : int
    ifault : int
    cond_est : float
    """
    precision, nullty, ifault = syminv_as007(sigma_matrix, jitter)
    cond_est, _, _ = covariance_condition_estimate(sigma_matrix)
    return precision, nullty, ifault, cond_est
