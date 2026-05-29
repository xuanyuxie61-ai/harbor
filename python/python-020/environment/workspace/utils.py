# -*- coding: utf-8 -*-
"""
utils.py
公共工具模块：物理常数、数值稳定性处理、辅助函数
"""
import numpy as np

# ============================================================================
# 物理常数（自然单位制：ħ = e = 1，质量以有效质量 m* 表示）
# ============================================================================
H_BAR = 1.0          # 约化普朗克常数
E_CHARGE = 1.0       # 电子电荷量（绝对值）
MU_B = 0.5           # 玻尔磁子 (eħ/2m* 在自然单位下)

# ============================================================================
# 数值稳定性工具
# ============================================================================
def safe_exp(x, max_val=700.0):
    """
    安全指数函数，防止溢出。
    当 x > max_val 时，返回 exp(max_val) 并发出警告。
    """
    x = np.asarray(x, dtype=float)
    if np.any(x > max_val):
        # 截断处理：对于溢出区域使用线性外推
        result = np.empty_like(x)
        mask = x > max_val
        result[~mask] = np.exp(x[~mask])
        result[mask] = np.exp(max_val) * (1.0 + (x[mask] - max_val) / max_val)
        return result
    return np.exp(x)


def safe_log(x, eps=1e-300):
    """
    安全对数函数，防止对零或负数取对数。
    """
    x = np.asarray(x, dtype=float)
    x_safe = np.where(x > eps, x, eps)
    return np.log(x_safe)


def normalize_vector(v):
    """
    安全归一化向量，处理零向量边界情况。
    """
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        return np.zeros_like(v)
    return v / norm


def condition_number(A):
    """
    计算矩阵条件数，处理奇异矩阵边界情况。
    """
    A = np.asarray(A, dtype=float)
    s = np.linalg.svd(A, compute_uv=False)
    if len(s) == 0 or s[-1] < 1e-15:
        return np.inf
    return s[0] / s[-1]


def gram_schmidt_qr(V, tol=1e-12):
    """
    改进的Gram-Schmidt正交化（含数值稳定性处理）。
    输入: V (N, k) 列向量组
    输出: Q (N, k) 标准正交基, R (k, k) 上三角矩阵
    """
    V = np.asarray(V, dtype=complex)
    N, k = V.shape
    Q = np.zeros((N, k), dtype=complex)
    R = np.zeros((k, k), dtype=complex)
    for j in range(k):
        v = V[:, j].copy()
        for i in range(j):
            R[i, j] = np.vdot(Q[:, i], v)
            v = v - R[i, j] * Q[:, i]
        norm_v = np.linalg.norm(v)
        if norm_v < tol:
            # 数值线性相关：用随机扰动恢复
            v = np.random.randn(N) + 1j * np.random.randn(N)
            for i in range(j):
                v = v - np.vdot(Q[:, i], v) * Q[:, i]
            norm_v = np.linalg.norm(v)
        R[j, j] = norm_v
        Q[:, j] = v / norm_v
    return Q, R


def fermi_dirac(E, mu, T, eps=1e-12):
    """
    Fermi-Dirac分布函数：
        f(E) = 1 / [exp((E - μ)/(k_B T)) + 1]
    在 T → 0 时退化为阶跃函数。
    """
    T = max(T, eps)
    beta = 1.0 / T
    arg = beta * (E - mu)
    # 防止溢出
    arg = np.clip(arg, -700.0, 700.0)
    return 1.0 / (np.exp(arg) + 1.0)


def magnetic_length(B, m_star=1.0):
    """
    磁长度：
        l_B = sqrt(ħ / (m* ω_c)) = sqrt(ħ / (e B))
    其中 ω_c = eB/m* 为回旋频率。
    """
    if B <= 0:
        raise ValueError("磁场强度 B 必须为正")
    return np.sqrt(H_BAR / (E_CHARGE * B))


def cyclotron_frequency(B, m_star=1.0):
    """
    回旋频率：
        ω_c = eB / m*
    """
    if B <= 0:
        raise ValueError("磁场强度 B 必须为正")
    return E_CHARGE * B / m_star


def landau_level_energy(n, B, m_star=1.0):
    """
    Landau能级能量：
        E_n = ħ ω_c (n + 1/2),  n = 0, 1, 2, ...
    """
    omega_c = cyclotron_frequency(B, m_star)
    return H_BAR * omega_c * (n + 0.5)


def filling_factor(N_e, B, A, m_star=1.0):
    """
    填充因子：
        ν = N_e / N_Φ = N_e * (2π ħ) / (e B A)
    其中 N_Φ = BA / Φ_0 为磁通量子数，Φ_0 = h/e = 2πħ/e。
    """
    if B <= 0 or A <= 0 or N_e < 0:
        raise ValueError("参数必须满足 B>0, A>0, N_e>=0")
    flux_quantum = 2.0 * np.pi * H_BAR / E_CHARGE
    N_phi = B * A / flux_quantum
    if N_phi < 1e-15:
        return np.inf
    return N_e / N_phi


def gaussian_2d(x, y, x0, y0, sigma):
    """
    二维高斯函数：
        G(x,y) = (1/(2πσ²)) exp(-[(x-x₀)² + (y-y₀)²]/(2σ²))
    """
    dx = x - x0
    dy = y - y0
    r2 = dx * dx + dy * dy
    return np.exp(-r2 / (2.0 * sigma * sigma)) / (2.0 * np.pi * sigma * sigma)
