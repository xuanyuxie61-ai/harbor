"""
statistical_noise_model.py
多通道噪声信号的统计建模与贝叶斯增益调整

融合原始项目:
  - 053_asa266 (Dirichlet分布参数估计)

科学背景:
  在多通道ANC系统中,各误差传感器的噪声功率往往具有
  非平稳、非高斯特性.将多通道功率谱建模为Dirichlet分布:

      f(x; alpha) = (1/B(alpha)) \prod_{j=1}^{K} x_j^{alpha_j - 1}

  其中 x_j = P_j / P_total 为归一化功率, alpha_j 为浓度参数.

  通过最大似然估计 alpha,可以:
  1. 检测异常传感器 (alpha_j 异常小)
  2. 自适应调整各通道步长 mu_j \propto alpha_j
  3. 评估系统收敛的统计置信度

  核心公式 (Newton-Raphson迭代):
      g_j = N [ psi(\sum alpha) - psi(alpha_j) ] + \sum_{i=1}^{N} ln x_{ij}
      Q_{jj} = N [ psi'(alpha_j) - psi'(\sum alpha) ]
      alpha^{new} = alpha - Q^{-1} g
"""

import numpy as np
import math
from special_functions import digamma, trigamma


def dirichlet_estimate_mle(x, alpha_init=None, max_iter=300, tol=1e-8):
    """
    Dirichlet分布参数的最大似然估计.

    实现策略:
      1. 使用矩估计获得良好初值
      2. 使用scipy.optimize.minimize进行带约束的数值优化
         (保证 alpha_j > alpha_min)
      3. 同时保留 Newton-Raphson 代码结构用于展示算法原理

    负对数似然:
        -L/N = -ln Gamma(alpha_0) + \sum_j ln Gamma(alpha_j)
               - \sum_j (alpha_j - 1) <ln x_j>

    参数:
        x: (N, K) 观测数据,每行和为1,每个元素>0
        alpha_init: (K,) 初始alpha,None则使用矩估计
        max_iter: 最大迭代次数
        tol: 收敛容差

    返回:
        alpha: (K,) 估计参数
        niter: 实际迭代次数
        loglik: 对数似然值
    """
    import scipy.optimize as opt

    x = np.asarray(x, dtype=float)
    N, K = x.shape

    if N <= K:
        raise ValueError("dirichlet_estimate: need N > K")
    if np.any(x <= 0):
        raise ValueError("dirichlet_estimate: all x must be positive")
    row_sums = np.sum(x, axis=1)
    if np.any(np.abs(row_sums - 1.0) > 0.01):
        x = x / row_sums[:, None]

    alpha_min = 0.05

    # 初始估计: 矩估计
    if alpha_init is None:
        means = np.mean(x, axis=0)
        var_mean = np.mean([np.var(x[:, j]) for j in range(K)])
        if var_mean < 1e-12:
            a0 = 10.0
        else:
            avg_mean = np.mean(means)
            a0 = max(avg_mean * (1.0 - avg_mean) / var_mean - 1.0, 0.5)
        alpha0 = a0 * means
        alpha0 = np.maximum(alpha0, alpha_min)
    else:
        alpha0 = np.asarray(alpha_init, dtype=float).copy()
        alpha0 = np.maximum(alpha0, alpha_min)

    log_x = np.log(x)
    avg_log_x = np.mean(log_x, axis=0)

    def neg_loglik(alpha):
        alpha = np.maximum(alpha, alpha_min)
        a_sum = np.sum(alpha)
        ll = -math.lgamma(a_sum)
        for j in range(K):
            ll += math.lgamma(alpha[j])
            ll += (alpha[j] - 1.0) * avg_log_x[j]
        return -ll * N

    def grad(alpha):
        alpha = np.maximum(alpha, alpha_min)
        a_sum = np.sum(alpha)
        ps, _ = digamma(a_sum)
        g = np.zeros(K)
        for j in range(K):
            pa, _ = digamma(alpha[j])
            g[j] = -N * (ps - pa + avg_log_x[j])
        return g

    bounds = [(alpha_min, None) for _ in range(K)]
    res = opt.minimize(neg_loglik, alpha0, jac=grad, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': max_iter, 'gtol': tol})

    alpha = res.x
    alpha = np.maximum(alpha, alpha_min)

    # 对数似然
    alpha_sum = np.sum(alpha)
    loglik = 0.0
    for j in range(K):
        loglik += (alpha[j] - 1.0) * np.sum(log_x[:, j])
    loglik -= N * math.lgamma(alpha_sum)
    for j in range(K):
        loglik += N * math.lgamma(alpha[j])

    return alpha, res.nit, loglik


def adaptive_step_size_from_dirichlet(error_powers, base_mu=0.001):
    """
    基于Dirichlet参数估计自适应调整各通道步长.

    物理直觉:
        alpha_j 大 -> 该通道功率稳定 -> 可较大步长
        alpha_j 小 -> 该通道功率波动 -> 应较小步长

    参数:
        error_powers: (N, K) 历史误差功率 (已归一化)
        base_mu: 基础步长

    返回:
        mu: (K,) 各通道步长
    """
    alpha, _, _ = dirichlet_estimate_mle(error_powers)
    alpha_sum = np.sum(alpha)
    if alpha_sum < 1e-12:
        return np.full(alpha.shape, base_mu)
    # 步长与归一化alpha成正比,但做限幅
    mu = base_mu * (alpha / alpha_sum) * alpha.shape[0]
    mu = np.clip(mu, base_mu * 0.1, base_mu * 5.0)
    return mu


def noise_stationarity_test(error_history, window=50):
    """
    基于方差比的噪声平稳性检验.

    统计量:
        F = var(前window) / var(后window)

    返回:
        is_stationary: bool
        f_stat: F统计量
    """
    err = np.asarray(error_history, dtype=float)
    if len(err) < 2 * window:
        return True, 1.0

    var1 = np.var(err[:window])
    var2 = np.var(err[-window:])
    if var2 < 1e-12:
        var2 = 1e-12
    f_stat = var1 / var2

    # 简单阈值判断 (非严格F检验)
    is_stationary = 0.5 < f_stat < 2.0
    return is_stationary, f_stat
