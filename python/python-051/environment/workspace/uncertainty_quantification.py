"""
uncertainty_quantification.py
=============================
海洋模拟的不确定性量化与收敛性分析模块。

融合算法
--------
1. L∞ 范数估计（源自 814_norm_loo）：
   对模拟输出场 f(x,z) 在离散网格上估计最大绝对值：
       ‖f‖_∞ = max_{i,j} |f_{ij}|
   用于监测数值稳定性与生物量爆发现象。

2. Legendre 求积精确性检验（源自 659_legendre_exactness）：
   验证高维稀疏网格对生态积分（如总初级生产力）的代数精度。

3. 蒙特卡洛采样与方差缩减：
   结合稀疏网格作为控制变量，估计参数敏感性指标 Sobol' 一阶指数。

数学公式
--------
Sobol' 一阶敏感性指数：
    S_i = V_{X_i}[E_{X_~i}(Y | X_i)] / V(Y)

总方差分解：
    V(Y) = Σ_i V_i + Σ_{i<j} V_{ij} + ... + V_{12...d}

此处采用稀疏网格多项式混沌展开（PCE）近似计算：
    Y ≈ Σ_{α∈A} c_α Ψ_α(X)
    S_i ≈ (1/V(Y)) Σ_{α: α_i>0} c_α²
"""

import numpy as np
from sparse_grid_cubature import legendre_monomial_integral


# ---------------------------------------------------------------------------
# L∞ 范数（源自 814_norm_loo）
# ---------------------------------------------------------------------------

def norm_loo(field):
    """
    计算离散场的 L∞ 范数：max |f_{ij}|。
    """
    return np.max(np.abs(field))


def norm_loo_location(field, x_coords, z_coords):
    """
    返回 L∞ 范数及其发生位置。
    """
    idx = np.unravel_index(np.argmax(np.abs(field)), field.shape)
    return np.abs(field[idx]), x_coords[idx[0]], z_coords[idx[1]]


# ---------------------------------------------------------------------------
# Legendre 精确性检验
# ---------------------------------------------------------------------------

def test_legendre_exactness_1d(points, weights, degree_max=11):
    """
    一维求积规则精确性检验。
    """
    tol = 1e-12
    results = []
    max_exact = -1
    for degree in range(degree_max + 1):
        exact = legendre_monomial_integral(degree)
        quad = np.sum(weights * (points ** degree))
        if exact == 0.0:
            err = abs(quad)
        else:
            err = abs((quad - exact) / exact)
        results.append((degree, err))
        if err < tol:
            max_exact = degree
    return max_exact, results


# ---------------------------------------------------------------------------
# 方差与敏感性分析
# ---------------------------------------------------------------------------

def compute_mean_variance(samples):
    """
    样本均值与方差。
    """
    mean = np.mean(samples)
    var = np.var(samples, ddof=1)
    return mean, var


def first_order_sobol_pce(coeffs, multi_indices, total_var, dim):
    """
    由多项式混沌展开系数计算一阶 Sobol' 指数。

    参数
    ----
    coeffs : ndarray (n_terms,)
        PCE 系数
    multi_indices : ndarray (n_terms, dim)
        多指标 α
    total_var : float
        总方差 V(Y)
    dim : int
        输入维度

    返回
    ----
    S1 : ndarray (dim,)
        一阶 Sobol' 指数
    """
    if total_var <= 1e-30:
        return np.zeros(dim)

    S1 = np.zeros(dim)
    for d in range(dim):
        # 选取仅第 d 维非零的项
        mask = (multi_indices[:, d] > 0) & (np.sum(multi_indices, axis=1) == multi_indices[:, d])
        S1[d] = np.sum(coeffs[mask] ** 2) / total_var

    return S1


def total_order_sobol_pce(coeffs, multi_indices, total_var, dim):
    """
    总阶 Sobol' 指数：包含所有涉及第 d 维的交互项。
    """
    if total_var <= 1e-30:
        return np.zeros(dim)

    ST = np.zeros(dim)
    for d in range(dim):
        mask = multi_indices[:, d] > 0
        ST[d] = np.sum(coeffs[mask] ** 2) / total_var

    return ST


# ---------------------------------------------------------------------------
# 收敛性诊断
# ---------------------------------------------------------------------------

def gci_refinement_estimator(fine, medium, coarse, r=2.0):
    """
    Grid Convergence Index (GCI) 用于估计数值解的离散误差。

    Roache (1994) 公式：
        p = ln|(f_coarse - f_medium)/(f_medium - f_fine)| / ln(r)
        GCI = F_s |ε| / (r^p - 1)
    其中 ε = (f_fine - f_medium)/f_fine，F_s 为安全系数（通常取 1.25）。
    """
    if abs(fine) < 1e-30:
        fine = 1e-30
    eps = (fine - medium) / fine
    denom = medium - fine
    if abs(denom) < 1e-30:
        denom = 1e-30
    p = np.log(abs((coarse - medium) / denom)) / np.log(max(r, 1.001))
    if abs(p) < 1e-6:
        p = 1e-6
    F_s = 1.25
    rp = r ** p
    rp_diff = rp - 1.0
    if abs(rp_diff) < 1e-30:
        rp_diff = 1e-30 if rp_diff >= 0 else -1e-30
    gci = F_s * abs(eps) / rp_diff
    return p, gci


def convergence_rate(errors, resolutions):
    """
    由多分辨率误差序列拟合收敛率：error ∝ h^p。
    """
    log_h = np.log(resolutions)
    log_e = np.log(errors)
    # 线性回归
    A = np.vstack([log_h, np.ones_like(log_h)]).T
    p, c = np.linalg.lstsq(A, log_e, rcond=None)[0]
    return p, np.exp(c)
