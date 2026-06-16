"""
error_analysis.py — 误差分析与验证指标
========================================

本模块实现随机 PDE 求解的全面误差分析:
  1. 时空误差度量 (映射自 1164_jenzenho_flame-ai-2024-reproduce)
  2. 统计矩误差: 均值/方差/高阶矩
  3. 收敛性分析: h-p 收敛率

数学框架:
  确定性误差: e_h = ||u - u_h|| ≤ C·h^p
  统计误差:   ε_MC = σ/√N, ε_QMC = O((log N)^d/N)
  PCE 截断:   ε_PCE = O(p^{-s}) (s 为解的正则性)

映射种子项目:
  - 1164_jenzenho_flame-ai-2024-reproduce: MSE, SSIM, Jaccard, 火线的类比指标
"""

import numpy as np


# ============================================================
# 第1部分: 场误差度量
# (映射自 1164_jenzenho_flame-ai: calculate_metrics)
# ============================================================

def mean_squared_error(field_true, field_pred):
    """MSE = (1/N) Σ (u_true - u_pred)²"""
    return np.mean((field_true - field_pred) ** 2)


def relative_l2_error(field_true, field_pred):
    """相对 L2 误差: ||e||₂/||u||₂"""
    num = np.sqrt(np.sum((field_true - field_pred) ** 2))
    denom = np.sqrt(np.sum(field_true ** 2))
    return num / max(denom, 1e-15)


def linf_error(field_true, field_pred):
    """L∞ 误差: max|u_true - u_pred|"""
    return np.max(np.abs(field_true - field_pred))


def structural_similarity_index(field_true, field_pred, window_size=7,
                                 k1=0.01, k2=0.03, L=1.0):
    """
    结构相似性指数 (SSIM):
        SSIM(x,y) = (2μ_xμ_y + C1)(2σ_xy + C2) /
                     ((μ_x²+μ_y²+C1)(σ_x²+σ_y²+C2))

    映射自 flame-ai 的 SSIM 计算。
    简化版: 全局 SSIM (非局部窗口)。
    """
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    mu_x = np.mean(field_true)
    mu_y = np.mean(field_pred)
    sigma_x2 = np.var(field_true)
    sigma_y2 = np.var(field_pred)
    sigma_xy = np.mean((field_true - mu_x) * (field_pred - mu_y))
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return numerator / max(denominator, 1e-30)


def jaccard_index(field_true, field_pred, threshold=0.5):
    """
    Jaccard 指数 (IoU):
        J = |A ∩ B| / |A ∪ B|
    用于二值化场的重叠度量。
    """
    A = field_true > threshold
    B = field_pred > threshold
    intersection = np.sum(A & B)
    union = np.sum(A | B)
    return intersection / max(union, 1)


def field_centroid(field, threshold=0.1):
    """
    场的质心 (映射自 flame-ai: calculate_centroid):
        x_c = Σ x·|u(x)| / Σ |u(x)|
    """
    abs_field = np.abs(field)
    mask = abs_field > threshold
    if not np.any(mask):
        return np.zeros(field.ndim)
    total = np.sum(abs_field[mask])
    if field.ndim == 1:
        x = np.arange(len(field))
        return np.array([np.sum(x[mask] * abs_field[mask]) / max(total, 1e-15)])
    elif field.ndim == 2:
        rows, cols = np.where(mask)
        weights = abs_field[mask]
        return np.array([
            np.sum(rows * weights) / max(total, 1e-15),
            np.sum(cols * weights) / max(total, 1e-15)
        ])
    return np.zeros(field.ndim)


def field_spread_rate(field_sequence):
    """
    场扩展速率 (映射自 flame-ai: spread rates):
        v(t) = d/dt ||u(t)||₂
    近似为差分: v(t) ≈ (||u(t+1)|| - ||u(t)||) / Δt
    """
    norms = [np.linalg.norm(f) for f in field_sequence]
    if len(norms) < 2:
        return np.array([0.0])
    rates = np.diff(norms)
    return rates


# ============================================================
# 第2部分: 统计矩误差
# ============================================================

def statistical_moment_errors(samples_true, samples_pred, max_moment=4):
    """
    统计矩误差分析:
        E_k[true] vs E_k[pred], k=1,...,max_moment

    返回:
        moments_true: 各阶矩
        moments_pred: 各阶矩
        moment_errors: 矩的绝对误差
    """
    moments_true = []
    moments_pred = []
    for k in range(1, max_moment + 1):
        mk_true = np.mean(samples_true ** k)
        mk_pred = np.mean(samples_pred ** k)
        moments_true.append(mk_true)
        moments_pred.append(mk_pred)
    moments_true = np.array(moments_true)
    moments_pred = np.array(moments_pred)
    errors = np.abs(moments_true - moments_pred)
    return moments_true, moments_pred, errors


def kolmogorov_smirnov_distance(samples_true, samples_pred):
    """
    KS 距离: sup_x |F_true(x) - F_pred(x)|
    用于比较两个样本集的累积分布函数差异。
    """
    combined = np.sort(np.concatenate([samples_true, samples_pred]))
    n1, n2 = len(samples_true), len(samples_pred)
    ecdf1 = np.searchsorted(np.sort(samples_true), combined, side='right') / n1
    ecdf2 = np.searchsorted(np.sort(samples_pred), combined, side='right') / n2
    return np.max(np.abs(ecdf1 - ecdf2))


# ============================================================
# 第3部分: 收敛性分析
# ============================================================

def convergence_rate(errors, mesh_sizes):
    """
    计算收敛阶:
        e = C·h^p → log(e) = log(C) + p·log(h)
        p = d(log e) / d(log h)

    返回:
        rates: 相邻点的局部收敛阶
        avg_rate: 平均收敛阶
    """
    errors = np.asarray(errors)
    mesh_sizes = np.asarray(mesh_sizes)
    # 过滤无效值
    valid = (errors > 1e-15) & (mesh_sizes > 1e-15)
    if np.sum(valid) < 2:
        return np.array([0.0]), 0.0
    log_e = np.log(errors[valid])
    log_h = np.log(mesh_sizes[valid])
    rates = np.diff(log_e) / np.diff(log_h)
    avg_rate = np.polyfit(log_h, log_e, 1)[0]
    return rates, avg_rate


def pce_convergence_analysis(pce_errors, polynomial_orders):
    """
    PCE 收敛性: 误差 vs 多项式阶数
        指数收敛: e ~ exp(-b·p) (解析解)
        代数收敛: e ~ p^{-s} (有限正则性)

    返回:
        is_exponential: 是否指数收敛
        estimated_rate: 收敛率
    """
    pce_errors = np.asarray(pce_errors)
    orders = np.asarray(polynomial_orders)
    valid = (pce_errors > 1e-15) & (orders > 0)
    if np.sum(valid) < 2:
        return False, 0.0
    log_e = np.log(pce_errors[valid])
    p = orders[valid]
    # 指数拟合: log(e) = a - b*p
    if len(p) >= 2:
        coeffs = np.polyfit(p, log_e, 1)
        rate = -coeffs[0]
        is_exp = rate > 0.1
    else:
        rate = 0.0
        is_exp = False
    return is_exp, rate


# ============================================================
# 第4部分: 综合误差报告
# ============================================================

def generate_error_report(reference_solution, computed_solution,
                          reference_samples=None, computed_samples=None,
                          mesh_size=None, pce_order=None):
    """
    生成综合误差报告:
      - 确定性场误差 (MSE, L2, L∞, SSIM, Jaccard)
      - 统计矩误差
      - 收敛指标

    返回:
        report: 字典形式的误差报告
    """
    report = {}
    # 场误差
    if reference_solution is not None and computed_solution is not None:
        ref = np.asarray(reference_solution)
        comp = np.asarray(computed_solution)
        if ref.shape == comp.shape:
            report['mse'] = mean_squared_error(ref, comp)
            report['relative_l2'] = relative_l2_error(ref, comp)
            report['linf'] = linf_error(ref, comp)
            report['ssim'] = structural_similarity_index(ref, comp)
            report['jaccard'] = jaccard_index(ref, comp)
            report['centroid_error'] = np.linalg.norm(
                field_centroid(ref) - field_centroid(comp)
            )

    # 统计矩误差
    if reference_samples is not None and computed_samples is not None:
        mt, mp, me = statistical_moment_errors(
            np.asarray(reference_samples), np.asarray(computed_samples)
        )
        report['moment_errors'] = me.tolist()
        report['ks_distance'] = kolmogorov_smirnov_distance(
            np.asarray(reference_samples), np.asarray(computed_samples)
        )

    # 收敛指标
    if mesh_size is not None:
        report['mesh_size'] = mesh_size
    if pce_order is not None:
        report['pce_order'] = pce_order

    return report
