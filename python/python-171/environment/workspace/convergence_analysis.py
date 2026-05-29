# -*- coding: utf-8 -*-
"""
convergence_analysis.py
=======================
迭代法收敛性分析与谱分布估计。

融合种子项目：
- 497_halton : 准随机序列用于随机 SVD 与迹估计
- 149_cg     : 随机正交矩阵、随机 SPD 矩阵
- 1270_toms443 : Lambert W 函数用于收敛速率理论估计
"""

import numpy as np
import math
from random_tools import randomized_svd_approx, hutchinson_trace_estimator, random_probe_vector
from special_functions import lambert_w_fast
from utils import condition_number_estimate


# ---------------------------------------------------------------------------
# 谱分布估计（随机 SVD + 直方图）
# ---------------------------------------------------------------------------

def estimate_spectral_density(matvec, n, n_samples=100, n_bins=40, seed=None):
    """
    用随机探测向量估计矩阵谱密度（Smoothed Approximation）。
    算法：
        1) 生成若干随机高斯向量 v_j
        2) 对每步 Lanczos 迭代，计算 Ritz 值
        3) 用高斯核平滑 Ritz 值分布
    简化实现：用随机 SVD 估计前 k 个特征值，再做核密度估计。
    """
    rank = min(n_samples, n)
    U, lam = randomized_svd_approx(matvec, n, rank, power_iterations=2, seed=seed)

    # 直方图
    hist, bin_edges = np.histogram(lam, bins=n_bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return lam, bin_centers, hist


def estimate_condition_number_randomized(matvec, n, seed=None):
    """
    用随机 SVD 估计条件数。
    """
    rank = min(30, n)
    _, lam = randomized_svd_approx(matvec, n, rank, power_iterations=3, seed=seed)
    if len(lam) < 2:
        return 1.0, 1.0, 1.0
    lam_max = lam[0]
    lam_min = max(lam[-1], 1e-15)
    return lam_max / lam_min, lam_max, lam_min


# ---------------------------------------------------------------------------
# 收敛速率理论分析
# ---------------------------------------------------------------------------

def theoretical_cg_error_bound(kappa, k):
    """
    CG 第 k 步后的 A-范数误差理论上界：
        ||e_k||_A / ||e_0||_A <= 2 * ρ^k
    其中 ρ = (sqrt(κ) - 1) / (sqrt(κ) + 1)。
    """
    if kappa <= 1.0 or k < 0:
        return 0.0
    rho = (math.sqrt(kappa) - 1.0) / (math.sqrt(kappa) + 1.0)
    return 2.0 * (rho ** k)


def theoretical_cg_iteration_count(kappa, epsilon=1e-10):
    """
    达到精度 ε 所需的理论迭代次数：
        k ≈ 0.5 * sqrt(κ) * log(2/ε)
    """
    if kappa <= 1.0:
        return 1
    return int(math.ceil(0.5 * math.sqrt(kappa) * math.log(2.0 / epsilon)))


def lambert_w_refined_convergence_bound(kappa, k):
    """
    利用 Lambert W 函数精细化 CG 误差上界。
    对极大条件数，rho ≈ 1 - 2/sqrt(κ)，利用 W(-2/sqrt(κ)) 修正对数项。
    """
    if kappa <= 1.0 or k < 0:
        return 0.0
    rho = (math.sqrt(kappa) - 1.0) / (math.sqrt(kappa) + 1.0)
    arg = -2.0 / math.sqrt(kappa)
    if arg >= -1.0 / math.e:
        w_val, _ = lambert_w_fast(arg)
        correction = 1.0 + w_val / (2.0 * math.sqrt(kappa))
    else:
        correction = 1.0
    return 2.0 * (rho ** (k * correction))


# ---------------------------------------------------------------------------
# 预处理效果评估
# ---------------------------------------------------------------------------

def preconditioner_quality(A, precond_apply, seed=None):
    """
    评估预处理子质量：
        计算 M^{-1} A 的条件数估计（通过随机探测）。
    对理想预处理子，cond(M^{-1} A) ≈ 1。
    """
    n = A.shape[0]

    def matvec_MA(v):
        return precond_apply(A @ v)

    kappa, lam_max, lam_min = estimate_condition_number_randomized(matvec_MA, n, seed=seed)
    return kappa, lam_max, lam_min


def eigenvalue_clustering_measure(eigenvalues, threshold=0.1):
    """
    衡量特征值聚类程度：
        计算相邻特征值相对差距小于 threshold 的比例。
    比例越高，说明谱聚类越好，CG 收敛越快。
    """
    ev = np.sort(np.asarray(eigenvalues, dtype=float))
    if len(ev) < 2:
        return 0.0
    gaps = np.diff(ev) / (np.abs(ev[:-1]) + 1e-30)
    clustered = np.sum(gaps < threshold)
    return clustered / len(gaps)


# ---------------------------------------------------------------------------
# 迭代历史对比分析
# ---------------------------------------------------------------------------

def compare_solvers(results_dict):
    """
    对比多种求解器的收敛历史。
    results_dict: {name: info_dict}
    返回文本报告。
    """
    lines = []
    lines.append("=" * 70)
    lines.append("  Solver Comparison Report")
    lines.append("=" * 70)
    for name, info in results_dict.items():
        it = info.get('iterations', -1)
        res = info.get('final_residual', -1)
        conv = "YES" if info.get('converged', False) else "NO"
        lines.append(f"  {name:20s} | Iter: {it:4d} | Final Res: {res:.3e} | Converged: {conv}")
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A-范数误差估计（利用迭代系数）
# ---------------------------------------------------------------------------

def estimate_a_norm_error(info, A_matvec, x_iter, x_exact=None):
    """
    估计 CG 迭代解的 A-范数误差 ||x - x*||_A。
    若提供精确解 x_exact，直接计算；否则利用递推公式估计。
    """
    if x_exact is not None:
        e = x_iter - x_exact
        return math.sqrt(float(e @ A_matvec(e)))

    # Gauss 求积估计：利用 CG 的 Lanczos 连接
    # 简化：用残差历史做近似
    res_hist = info.get('residual_history', [])
    if len(res_hist) < 2:
        return 1.0
    # 粗略估计：||e||_A ≈ ||r||_2 / sqrt(lambda_min)
    # 这里取最后几个残差的几何平均作为代理
    return math.sqrt(res_hist[-1] * res_hist[-2]) if len(res_hist) >= 2 else res_hist[-1]
