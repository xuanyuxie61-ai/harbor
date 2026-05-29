"""
error_analysis.py
误差分析与统计验证模块

融合种子项目:
- 024_asa005 (正态分布累积密度)
- 1026_risk_matrix (转移矩阵用于马尔可夫链收敛分析)

科学背景:
FMM的误差来源包括:
1. 多极展开截断误差 (truncation error)
2. M2L转换近似误差
3. 数值舍入误差
4. 树结构划分带来的几何误差

核心公式:
    - 相对L2误差:
        e_L2 = ||phi_fmm - phi_direct||_2 / ||phi_direct||_2
        
    - 相对L_inf误差:
        e_inf = max_i |phi_fmm(i) - phi_direct(i)| / max_i |phi_direct(i)|
        
    - 收敛阶数估计:
        p = log2(e(h) / e(h/2))
        其中 h 为离散化参数 (如展开阶数倒数)
        
    - 马尔可夫链稳态收敛 (融合1026_risk_matrix):
        对于转移矩阵 T, 稳态分布 pi 满足: pi = pi * T
        收敛速度由第二特征值 |lambda_2| 决定:
            ||pi^{(k)} - pi|| <= C * |lambda_2|^k
        
    - 正态性检验 (Shapiro-Wilk近似):
        用于检验误差分布是否近似正态分布
        
    - Kolmogorov-Smirnov检验统计量:
        D_n = sup_x |F_n(x) - F(x)|
        其中 F_n 为经验分布, F 为理论正态分布
"""

import numpy as np
from monte_carlo_sampler import alnorm


def relative_l2_error(approx, exact):
    """
    计算相对L2误差
    
    公式:
        e = sqrt( sum (a_i - e_i)^2 ) / sqrt( sum e_i^2 )
    """
    approx = np.asarray(approx, dtype=float)
    exact = np.asarray(exact, dtype=float)
    denom = np.linalg.norm(exact)
    if denom < 1e-15:
        denom = 1.0
    return np.linalg.norm(approx - exact) / denom


def relative_inf_error(approx, exact):
    """
    计算相对L_inf误差
    
    公式:
        e = max |a_i - e_i| / max |e_i|
    """
    approx = np.asarray(approx, dtype=float)
    exact = np.asarray(exact, dtype=float)
    denom = np.max(np.abs(exact))
    if denom < 1e-15:
        denom = 1.0
    return np.max(np.abs(approx - exact)) / denom


def convergence_order(errors, parameters):
    """
    估计收敛阶数
    
    假设误差满足 e(h) = C * h^p
    则 p = log(e_i / e_{i+1}) / log(h_i / h_{i+1})
    
    参数:
        errors: ndarray (K,), 各参数下的误差
        parameters: ndarray (K,), 对应的离散化参数
    
    返回:
        ndarray (K-1,), 局部收敛阶数
    """
    errors = np.asarray(errors)
    parameters = np.asarray(parameters)
    if len(errors) != len(parameters):
        raise ValueError("长度不匹配")
    p = []
    for i in range(len(errors) - 1):
        if errors[i+1] < 1e-15 or errors[i] < 1e-15:
            p.append(0.0)
            continue
        ratio_e = errors[i] / errors[i+1]
        ratio_h = parameters[i] / parameters[i+1]
        if ratio_h <= 0:
            p.append(0.0)
        else:
            p.append(np.log(ratio_e) / np.log(ratio_h))
    return np.array(p)


def estimate_truncation_order(fmm_error, direct_error, expansion_orders):
    """
    估计FMM展开截断阶数对误差的影响
    
    理论预期: 误差 ~ exp(-c * L) (指数收敛)
    取对数: log(error) ~ -c * L
    通过线性回归估计 c
    
    参数:
        fmm_error: ndarray (K,), 各阶数下的FMM误差
        direct_error: float, 直接求和的基准误差 (舍入误差)
        expansion_orders: ndarray (K,)
    
    返回:
        dict: 包含rate, predicted_errors
    """
    fmm_error = np.asarray(fmm_error)
    expansion_orders = np.asarray(expansion_orders)
    # 减去直接求和误差 (视为下限)
    adjusted = np.maximum(fmm_error - direct_error, 1e-16)
    log_err = np.log(adjusted)
    # 线性回归: log_err = a + b * L
    L = expansion_orders.astype(float)
    n = len(L)
    if n < 2:
        return {"rate": 0.0, "predicted_errors": fmm_error}
    A = np.vstack([np.ones(n), L]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, log_err, rcond=None)
    b = coeffs[1]
    a = coeffs[0]
    predicted = np.exp(a + b * L) + direct_error
    return {
        "rate": float(-b),
        "intercept": float(a),
        "predicted_errors": predicted,
        "adjusted_errors": adjusted
    }


def kolmogorov_smirnov_statistic(samples, mu=0.0, sigma=1.0):
    """
    计算Kolmogorov-Smirnov统计量 (与标准正态分布比较)
    
    公式:
        D_n = sup_x |F_n(x) - Phi((x-mu)/sigma)|
    
    参数:
        samples: ndarray (n,)
        mu, sigma: 理论正态分布参数
    
    返回:
        float, D_n 统计量
    """
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    sorted_samples = np.sort(samples)
    empirical = np.arange(1, n + 1) / n
    theoretical = np.array([alnorm((x - mu) / sigma, upper=False) for x in sorted_samples])
    diff1 = np.abs(empirical - theoretical)
    diff2 = np.abs(np.arange(0, n) / n - theoretical)
    return float(np.max(np.concatenate([diff1, diff2])))


def markov_chain_steady_state(transition_matrix, tol=1e-10, max_iter=10000):
    """
    计算马尔可夫链稳态分布 (融合1026_risk_matrix)
    
    公式:
        pi^{(k+1)} = pi^{(k)} * T
        收敛条件: ||pi^{(k+1)} - pi^{(k)}||_1 < tol
    
    参数:
        transition_matrix: ndarray (n, n), 行随机矩阵
        tol: float, 收敛容差
        max_iter: int
    
    返回:
        ndarray (n,), 稳态分布
    """
    T = np.asarray(transition_matrix, dtype=float)
    n = T.shape[0]
    if T.shape != (n, n):
        raise ValueError("必须是方阵")
    # 验证行随机
    row_sums = np.sum(T, axis=1)
    if np.any(np.abs(row_sums - 1.0) > 1e-6):
        raise ValueError("转移矩阵必须是行随机矩阵")

    pi = np.ones(n) / n
    for _ in range(max_iter):
        pi_new = pi @ T
        if np.linalg.norm(pi_new - pi, ord=1) < tol:
            return pi_new
        pi = pi_new
    return pi


def second_eigenvalue_rate(transition_matrix):
    """
    计算马尔可夫链收敛率 (由第二特征值决定)
    
    公式:
        rate = |lambda_2|
        混合时间上界: t_mix(epsilon) <= log(1/(epsilon*pi_min)) / (1 - |lambda_2|)
    
    参数:
        transition_matrix: ndarray (n, n)
    
    返回:
        float, |lambda_2|
    """
    T = np.asarray(transition_matrix, dtype=float)
    eigenvalues = np.linalg.eigvals(T)
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
    if len(eigenvalues) < 2:
        return 0.0
    return float(eigenvalues[1])


def fmm_error_budget(n_particles, expansion_order, separation_param=2.0, machine_eps=2.2e-16):
    """
    FMM误差预算分析
    
    误差组成:
        1. 截断误差: e_trunc ~ C1 * exp(-c * L)
        2. 转换误差: e_trans ~ C2 * (1/s)^{L+1}
        3. 舍入误差: e_round ~ C3 * N * machine_eps
        4. 总误差: e_total = sqrt(e_trunc^2 + e_trans^2 + e_round^2)
    
    参数:
        n_particles: int
        expansion_order: int
        separation_param: float
        machine_eps: float
    
    返回:
        dict, 各误差分量估计
    """
    L = expansion_order
    s = separation_param
    # 经验常数
    C1 = 1.0
    C2 = 0.5
    C3 = 1.0
    e_trunc = C1 * np.exp(-0.5 * L)
    e_trans = C2 * (1.0 / s) ** (L + 1)
    e_round = C3 * n_particles * machine_eps
    e_total = np.sqrt(e_trunc**2 + e_trans**2 + e_round**2)
    return {
        "truncation_error": float(e_trunc),
        "translation_error": float(e_trans),
        "roundoff_error": float(e_round),
        "total_error_estimate": float(e_total)
    }
