"""
chi_square.py — χ² 拟合与统计推断
====================================

融合种子项目:
    [165_chebyshev1_rule]   : Gauss-Chebyshev 求积 → χ² 积分
    [694_local_min]         : Brent 最小化 → 参数优化

核心统计框架:

[χ² 定义]
    χ² = Σ_{i=1}^{N} r_i^T · V_i^{-1} · r_i

    其中:
        r_i = m_i - h(x_i)  为残差向量 (测量值 - 预测值)
        V_i = V_meas_i + H_i · C_i · H_i^T  为残差协方差
        H_i = ∂h/∂x|_{x_i}  为设计矩阵 (Jacobian)

[Kalman χ² 递推]
    χ²_n = χ²_{n-1} + r_n^T · G_n^{-1} · r_n
    G_n = V_n + H_n · C_n · H_n^T  (创新协方差)

[参数估计]
    p̂ = argmin_p χ²(p)
    δp = C · H^T · G^{-1} · r  (参数更新)

[Goodness-of-fit]
    χ²/dof ~ 1 表示良好拟合
    p-value = 1 - F(χ²; ndof)  其中 F 为 χ² CDF
"""

import math
import numpy as np
from typing import List, Tuple, Optional


# ============================================================
# 矩阵运算辅助
# ============================================================
def safe_invert_2x2(matrix):
    """
    安全求逆 2×2 矩阵

    对于 2D 测量 (r, z):
        V = [[σ²_r, ρσ_rσ_z],
             [ρσ_rσ_z, σ²_z]]

    det(V) = σ²_r · σ²_z · (1 - ρ²)
    数值安全: 限制 det > eps

    Parameters
    ----------
    matrix : list of list, shape (2, 2)

    Returns
    -------
    list of list : 逆矩阵
    """
    a, b = matrix[0][0], matrix[0][1]
    c, d = matrix[1][0], matrix[1][1]

    det = a * d - b * c

    # 数值安全
    if abs(det) < 1e-30:
        # 正则化
        reg = 1e-10 * max(abs(a), abs(d), 1e-10)
        a += reg
        d += reg
        det = a * d - b * c

    inv_det = 1.0 / det
    return [[d * inv_det, -b * inv_det],
            [-c * inv_det, a * inv_det]]


def safe_invert_matrix(matrix):
    """
    安全求逆一般矩阵 (通过 numpy)

    添加 Tikhonov 正则化保证数值稳定:
        A_inv = (A + εI)^{-1}
    """
    A = np.array(matrix, dtype=float)
    n = A.shape[0]

    try:
        # 对称化
        A = 0.5 * (A + A.T)
        # 检查条件数
        eigenvalues = np.linalg.eigvalsh(A)
        min_eig = min(eigenvalues)

        if min_eig < 1e-14:
            # Tikhonov 正则化
            reg = max(1e-10, -min_eig + 1e-14)
            A += reg * np.eye(n)

        return np.linalg.inv(A).tolist()
    except np.linalg.LinAlgError:
        # 回退: 伪逆
        return np.linalg.pinv(A).tolist()


# ============================================================
# χ² 计算
# ============================================================
def compute_chi2_single(residual, cov_inv):
    """
    单次测量 χ² 贡献

    χ²_i = r_i^T · V_i^{-1} · r_i

    Parameters
    ----------
    residual : array_like
        残差向量
    cov_inv : array_like
        逆协方差矩阵

    Returns
    -------
    float : χ² 贡献
    """
    r = np.array(residual, dtype=float)
    vi = np.array(cov_inv, dtype=float)
    chi2 = float(r.T @ vi @ r)
    return max(chi2, 0.0)  # 确保非负


def compute_chi2_track(measurements, predictions, covariances):
    """
    完整径迹 χ² 计算

    χ²_total = Σ_i r_i^T · V_i^{-1} · r_i

    Parameters
    ----------
    measurements : list of ndarray
        测量向量列表
    predictions : list of ndarray
        预测向量列表
    covariances : list of ndarray
        测量协方差列表

    Returns
    -------
    dict : {
        'chi2_total': float,
        'chi2_per_hit': list,
        'ndof': int,
        'chi2_ndof': float,
        'residuals': list,
    }
    """
    n_hits = len(measurements)
    chi2_per_hit = []
    residuals = []

    for i in range(n_hits):
        r = np.array(measurements[i]) - np.array(predictions[i])
        V = np.array(covariances[i])

        try:
            V_inv = np.linalg.inv(V)
        except np.linalg.LinAlgError:
            V_inv = np.linalg.pinv(V)

        chi2_i = float(r.T @ V_inv @ r)
        chi2_per_hit.append(max(chi2_i, 0.0))
        residuals.append(r.tolist())

    chi2_total = sum(chi2_per_hit)

    # 自由度: ndof = 2*N_hits - 5 (每击中 2 个测量，5 个 track params)
    ndof = max(2 * n_hits - 5, 1)
    chi2_ndof = chi2_total / ndof

    return {
        'chi2_total': chi2_total,
        'chi2_per_hit': chi2_per_hit,
        'ndof': ndof,
        'chi2_ndof': chi2_ndof,
        'residuals': residuals,
        'n_hits': n_hits,
    }


# ============================================================
# [165_chebyshev1_rule] Chebyshev 求积 → χ² 似然积分
# ============================================================
def chi2_likelihood_integral(chi2_func, param_range, n_quad=16):
    """
    基于 [165_chebyshev1_rule] 的 Gauss 求积计算 χ² 似然积分

    原算法通过 Golub-Welsch 方法计算 Gauss-Chebyshev 节点和权重

    这里用于计算参数后验分布的归一化积分:
        Z = ∫ exp(-χ²(p)/2) dp

    以及参数期望值:
        <p> = (1/Z) ∫ p · exp(-χ²(p)/2) dp

    使用 Gauss-Legendre 求积:
        ∫_a^b f(x)dx ≈ Σ w_i · f(x_i)

    Parameters
    ----------
    chi2_func : callable
        χ²(p) 函数
    param_range : tuple
        (p_min, p_max) 参数范围
    n_quad : int
        求积阶数

    Returns
    -------
    dict : {
        'log_evidence': float,   # ln(Z)
        'param_mean': float,     # <p>
        'param_std': float,      # σ_p
        'nodes': list,           # 求积节点
        'weights': list,         # 求积权重
    }
    """
    # 计算 Gauss-Legendre 节点和权重
    nodes, weights = _gauss_legendre_rule(n_quad)

    # 区间变换
    a, b = param_range
    half = (b - a) / 2.0
    mid = (b + a) / 2.0

    # 计算被积函数值
    log_weights = []
    param_values = []

    for i in range(n_quad):
        p_i = half * nodes[i] + mid
        chi2_i = chi2_func(p_i)
        param_values.append(p_i)
        log_weights.append(-0.5 * chi2_i + math.log(max(weights[i], 1e-300)))

    # Log-sum-exp 技巧 (数值稳定)
    max_log_w = max(log_weights)
    exp_weights = [math.exp(lw - max_log_w) for lw in log_weights]
    Z_raw = sum(exp_weights) * half
    log_Z = max_log_w + math.log(max(Z_raw, 1e-300))

    # 归一化权重
    if Z_raw > 1e-300:
        norm_weights = [w / Z_raw for w in exp_weights]
    else:
        norm_weights = [1.0 / n_quad] * n_quad

    # 参数期望值
    param_mean = sum(nw * pv for nw, pv in zip(norm_weights, param_values))
    param_var = sum(nw * (pv - param_mean)**2
                    for nw, pv in zip(norm_weights, param_values))
    param_std = math.sqrt(max(param_var, 0.0))

    return {
        'log_evidence': log_Z,
        'param_mean': param_mean,
        'param_std': param_std,
        'nodes': param_values,
        'weights': norm_weights,
    }


def _gauss_legendre_rule(n):
    """
    Gauss-Legendre 求积规则 (简化 Golub-Welsch)

    返回 [-1, 1] 上的节点和权重
    """
    if n <= 0:
        return [0.0], [2.0]
    if n == 1:
        return [0.0], [2.0]

    # 使用 numpy 的 Gauss 求积 (可靠)
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return nodes.tolist(), weights.tolist()


# ============================================================
# [694_local_min] Brent 最小化 → χ² 参数优化
# ============================================================
def brent_chi2_minimize(chi2_func, a, b, tol=1e-5, max_iter=100):
    """
    基于 [694_local_min] 的 Brent 方法最小化 χ²

    原算法结合:
    1. 黄金分割搜索 (保证收敛)
    2. 抛物线插值 (加速收敛)

    停止条件: |x - m| ≤ 2·tol - 0.5·(b-a)
    其中 m = (a+b)/2 为区间中点

    Parameters
    ----------
    chi2_func : callable
        χ²(p) 目标函数
    a, b : float
        搜索区间 [a, b]
    tol : float
        收敛容差
    max_iter : int
        最大迭代次数

    Returns
    -------
    dict : {
        'p_opt': float,        # 最优参数
        'chi2_min': float,     # 最小 χ²
        'n_evals': int,        # 函数求值次数
        'converged': bool,
    }
    """
    # 黄金分割常数 (与 [694] 一致)
    c = 0.5 * (3.0 - math.sqrt(5.0))  # ≈ 0.38197

    sa, sb = float(a), float(b)
    tol_blend = tol

    # 初始化三个点
    x = sa + c * (sb - sa)
    v = x
    w = x
    fx = chi2_func(x)
    fv = fx
    fw = fx

    e = 0.0  # 上一步的步长
    d = 0.0  # 当前步长
    n_evals = 1

    converged = False

    for _ in range(max_iter):
        m = 0.5 * (sa + sb)
        tol1 = tol_blend * abs(x) + 1e-10
        tol2 = 2.0 * tol1

        # 收敛检查 (与 [694] 一致)
        if abs(x - m) <= tol2 - 0.5 * (sb - sa):
            converged = True
            break

        # 尝试抛物线插值
        if abs(e) > tol1:
            # 拟合抛物线
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)

            if q > 0:
                p = -p
            else:
                q = -q

            r = e
            e = d

            # 检查抛物线步是否安全
            if (abs(p) < abs(0.5 * q * r) and
                    p > q * (sa - x) and p < q * (sb - x)):
                # 抛物线步
                d = p / q
                u = x + d
                # 防止过于接近边界
                if (u - sa) < tol2 or (sb - u) < tol2:
                    d = tol1 if x < m else -tol1
            else:
                # 黄金分割步
                e = sb - x if x < m else sa - x
                d = c * e
        else:
            # 黄金分割步
            e = sb - x if x < m else sa - x
            d = c * e

        # 计算新点
        u = x + (d if abs(d) >= tol1 else (tol1 if d > 0 else -tol1))
        fu = chi2_func(u)
        n_evals += 1

        # 更新括号
        if fu <= fx:
            if u < x:
                sb = x
            else:
                sa = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                sa = u
            else:
                sb = u

            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu

    return {
        'p_opt': x,
        'chi2_min': fx,
        'n_evals': n_evals,
        'converged': converged,
    }


# ============================================================
# p-value 计算
# ============================================================
def chi2_pvalue(chi2, ndof):
    """
    计算 χ² 检验的 p-value

    p = 1 - F(χ²; ndof) = Γ(ndof/2, χ²/2) / Γ(ndof/2)

    使用不完全 Gamma 函数的近似

    Parameters
    ----------
    chi2 : float
        χ² 值
    ndof : int
        自由度

    Returns
    -------
    float : p-value ∈ [0, 1]
    """
    if ndof <= 0:
        return 0.0
    if chi2 <= 0:
        return 1.0

    # 使用 Wilson-Hilferty 近似 (正态近似)
    # z = [(χ²/k)^{1/3} - (1 - 2/(9k))] / √(2/(9k))
    k = float(ndof)
    ratio = chi2 / k

    if ratio < 1e-10:
        return 1.0

    term1 = ratio ** (1.0 / 3.0)
    term2 = 1.0 - 2.0 / (9.0 * k)
    term3 = math.sqrt(2.0 / (9.0 * k))

    if term3 < 1e-15:
        return 0.5

    z = (term1 - term2) / term3

    # 标准正态 CDF (误差函数)
    p_value = 0.5 * math.erfc(z / math.sqrt(2.0))

    return max(0.0, min(1.0, p_value))


def pull_distribution(residuals, residual_errors):
    """
    计算 pull 分布

    pull_i = r_i / σ_i

    良好的拟合应有 pull ~ N(0, 1)

    Parameters
    ----------
    residuals : list of float
    residual_errors : list of float

    Returns
    -------
    dict : {
        'pulls': list,
        'mean': float,
        'std': float,
        'chi2_pull': float,
    }
    """
    pulls = []
    for r, e in zip(residuals, residual_errors):
        if abs(e) > 1e-15:
            pulls.append(r / e)
        else:
            pulls.append(0.0)

    if not pulls:
        return {'pulls': [], 'mean': 0.0, 'std': 0.0, 'chi2_pull': 0.0}

    mean_pull = sum(pulls) / len(pulls)
    var_pull = sum((p - mean_pull)**2 for p in pulls) / max(len(pulls) - 1, 1)
    std_pull = math.sqrt(max(var_pull, 0.0))
    chi2_pull = sum(p**2 for p in pulls)

    return {
        'pulls': pulls,
        'mean': mean_pull,
        'std': std_pull,
        'chi2_pull': chi2_pull,
        'n_pulls': len(pulls),
    }
