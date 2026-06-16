"""
random_field.py — 随机场建模与 Karhunen-Loève 展开
====================================================

本模块实现随机参数 PDE 中随机扩散系数 a(x,ω) 的建模。
核心方法:
  1. 指数型/Matérn 协方差函数
  2. Hankel-Cholesky 分解: 利用结构加速协方差矩阵分解
  3. Karhunen-Loève 展开: a(x,ω) = a₀(x) + Σₖ √λₖ φₖ(x) ξₖ(ω)
  4. 对数正态变换保证正定性: a(x,ω) = exp(Y(x,ω))

映射种子项目:
  - 504_hankel_cholesky: Hankel 矩阵 Cholesky 分解快速算法
  - 582_image_normalize: 随机场的标准化与归一化
"""

import numpy as np
from numpy.linalg import eigh, cholesky, LinAlgError


# ============================================================
# 第1部分: 协方差函数族
# ============================================================

def exponential_covariance(x, y, sigma2=1.0, length_scale=0.5):
    """
    指数型协方差 (Matérn ν=1/2):
        C(x,y) = σ² exp(-||x-y||₁/ℓ)
    """
    r = np.sum(np.abs(np.asarray(x) - np.asarray(y)))
    r = max(r, 1.0e-15)
    return sigma2 * np.exp(-r / length_scale)


def squared_exponential_covariance(x, y, sigma2=1.0, length_scale=0.5):
    """
    平方指数 (Gaussian) 协方差:
        C(x,y) = σ² exp(-||x-y||²₂/(2ℓ²))
    """
    diff = np.asarray(x) - np.asarray(y)
    r2 = np.dot(diff, diff)
    return sigma2 * np.exp(-r2 / (2.0 * length_scale ** 2))


def matern_covariance(x, y, sigma2=1.0, length_scale=0.5, nu=1.5):
    """
    Matérn 协方差 (通用 ν):
        C(x,y) = σ²·2^(1-ν)/Γ(ν)·(√(2ν)r/ℓ)^ν·K_ν(√(2ν)r/ℓ)
    特殊值: ν=1/2→指数; ν=3/2→(1+√3r/ℓ)exp(-√3r/ℓ); ν→∞→Gaussian
    """
    diff = np.asarray(x) - np.asarray(y)
    r = np.sqrt(np.dot(diff, diff))
    r = max(r, 1.0e-15)
    scaled = np.sqrt(2.0 * nu) * r / length_scale
    if abs(nu - 0.5) < 1e-12:
        return sigma2 * np.exp(-scaled)
    elif abs(nu - 1.5) < 1e-12:
        return sigma2 * (1.0 + scaled) * np.exp(-scaled)
    elif abs(nu - 2.5) < 1e-12:
        return sigma2 * (1.0 + scaled + scaled ** 2 / 3.0) * np.exp(-scaled)
    else:
        from scipy.special import kv, gamma
        coeff = sigma2 * 2.0 ** (1.0 - nu) / gamma(nu)
        return coeff * scaled ** nu * kv(nu, scaled)


# ============================================================
# 第2部分: Hankel-structured Cholesky 分解
# (映射自 504_hankel_cholesky: hankel_spd_cholesky_lower)
# ============================================================

def hankel_spd_cholesky_lower(n, diagonal_vals, subdiag_vals):
    """
    构造下三角 Cholesky 因子 L 使得 H = L·Lᵀ 为 Hankel SPD 矩阵。
    基于 antidiagonal 约束递推:
        α = Σ L(i-1,j+k)·L(j-1,j+k)  (antidiagonal 匹配行内积)
        β = Σ_{k<j} L(i,k)·L(j,k)    (已计算列内积)
        L(i,j) = (α - β) / L(j,j)
    """
    L = np.zeros((n, n))
    for i in range(n):
        L[i, i] = max(diagonal_vals[i], 1.0e-14)
    for i in range(1, n):
        L[i, i - 1] = subdiag_vals[i - 1]
    for j in range(n):
        for i in range(j + 2, n):
            alpha = 0.0
            for k in range(j + 1, n):
                if i - 1 < n and j + 1 < n and k < i:
                    alpha += L[i - 1, k] * L[j + 1, k]
            beta = 0.0
            for k in range(j):
                beta += L[i, k] * L[j, k]
            denom = L[j, j]
            if abs(denom) < 1.0e-15:
                L[i, j] = 0.0
            else:
                L[i, j] = (alpha - beta) / denom
    return L


def hankel_cholesky_upper(n, h_vec):
    """
    Hankel 矩阵上三角 Cholesky 因子。
    h_vec: 长度 2n-1, H(i,j) = h_vec[i+j]
    返回 R: (n,n) 上三角矩阵使得 H ≈ RᵀR
    """
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            idx = i + j
            if idx < len(h_vec):
                H[i, j] = h_vec[idx]
    H += np.eye(n) * 1.0e-10
    try:
        R = cholesky(H)
    except LinAlgError:
        eigvals = eigh(H, eigvals_only=True)
        shift = max(0.0, -np.min(eigvals) + 1.0e-8)
        H += np.eye(n) * shift
        R = cholesky(H)
    return R


def build_covariance_matrix(points, cov_func, **cov_kwargs):
    """构建协方差矩阵: C_ij = cov_func(points[i], points[j])"""
    N = len(points)
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            val = cov_func(points[i], points[j], **cov_kwargs)
            C[i, j] = val
            C[j, i] = val
    return C


# ============================================================
# 第3部分: Karhunen-Loève 展开
# ============================================================

def karhunen_loeve_decomposition(cov_matrix, n_modes=None, tolerance=1.0e-10):
    """
    离散 KL 展开: C = Σ λₖ vₖ vₖᵀ
    随机场: Y(x,ω) = Σ √λₖ vₖ(x) ξₖ(ω), ξₖ~N(0,1)
    能量截断: Σₖ₌₁ᴷ λₖ/Σⱼ λⱼ ≥ 1 - tolerance
    """
    eigenvalues, eigenvectors = eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total_energy = np.sum(eigenvalues)
    if total_energy < 1.0e-30:
        return np.array([0.0]), eigenvectors[:, :1], 0.0
    cumulative = np.cumsum(eigenvalues) / total_energy
    if n_modes is None:
        n_modes = int(np.searchsorted(cumulative, 1.0 - tolerance) + 1)
        n_modes = min(n_modes, len(eigenvalues))
    eigenvalues = eigenvalues[:n_modes]
    eigenvectors = eigenvectors[:, :n_modes]
    explained = cumulative[n_modes - 1] if n_modes > 0 else 0.0
    return eigenvalues, eigenvectors, explained


def sample_random_field(eigenvalues, eigenvectors, n_samples=1, seed=None):
    """
    KL 展开采样: Y⁽ˢ⁾(x) = Σ √λₖ vₖ(x) ξₖ⁽ˢ⁾
    返回 samples:(n_samples,N), xi:(n_samples,K)
    """
    rng = np.random.default_rng(seed)
    K = len(eigenvalues)
    xi = rng.standard_normal((n_samples, K))
    sqrt_lambda = np.sqrt(np.maximum(eigenvalues, 0.0))
    samples = (xi * sqrt_lambda[np.newaxis, :]) @ eigenvectors.T
    return samples, xi


def log_normal_transform(gaussian_field, mean_val=0.0, sigma_coeff=0.5):
    """对数正态变换: a(x,ω) = exp(mean + σ·Y(x,ω))"""
    return np.exp(mean_val + sigma_coeff * gaussian_field)


# ============================================================
# 第4部分: 随机场标准化
# (映射自 582_image_normalize)
# ============================================================

def normalize_random_field(field, method='zscore'):
    """
    随机场标准化:
      zscore:  z=(f-μ)/σ
      minmax:  z=(f-min)/(max-min)
      robust:  z=(f-median)/IQR
    """
    field = np.asarray(field, dtype=np.float64)
    if method == 'zscore':
        mu = np.mean(field, axis=-1, keepdims=True)
        sigma = np.maximum(np.std(field, axis=-1, keepdims=True), 1e-15)
        return (field - mu) / sigma, {'mean': mu.squeeze(), 'std': sigma.squeeze()}
    elif method == 'minmax':
        fmin = np.min(field, axis=-1, keepdims=True)
        fmax = np.max(field, axis=-1, keepdims=True)
        denom = np.maximum(fmax - fmin, 1e-15)
        return (field - fmin) / denom, {'min': fmin.squeeze(), 'max': fmax.squeeze()}
    elif method == 'robust':
        median = np.median(field, axis=-1, keepdims=True)
        q75 = np.percentile(field, 75, axis=-1, keepdims=True)
        q25 = np.percentile(field, 25, axis=-1, keepdims=True)
        iqr = np.maximum(q75 - q25, 1e-15)
        return (field - median) / iqr, {'median': median.squeeze(), 'iqr': iqr.squeeze()}
    else:
        raise ValueError(f"Unknown normalization: {method}")


def kl_energy_spectrum(eigenvalues):
    """KL 能量谱: E_k = λ_k/Σⱼ λⱼ, 累积能量 S_K = Σₖ₌₁ᴷ E_k"""
    total = np.sum(eigenvalues)
    if total < 1e-30:
        return np.zeros_like(eigenvalues), np.zeros_like(eigenvalues)
    energy = eigenvalues / total
    cumulative = np.cumsum(energy)
    return energy, cumulative
