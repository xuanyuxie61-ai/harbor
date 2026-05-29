"""
bayesian_inference.py
贝叶斯参数推断 — Dirichlet/Gamma 先验与冰盖流变参数后验估计

基于种子项目 053_asa266 的 Dirichlet 分布最大似然估计与 Gamma 采样算法，
实现冰盖模型参数的贝叶斯后验推断。

核心数学:
  1. Gamma 分布先验 (率因子 A_0 的正则化):
       p(A_0) = \text{Gamma}(A_0; \alpha, \beta)
              = \frac{\beta^\alpha}{\Gamma(\alpha)} A_0^{\alpha-1} e^{-\beta A_0}

  2. Dirichlet 分布先验 (多源冰流贡献比例):
       p(\pi) = \text{Dir}(\pi; \alpha_1, ..., \alpha_K)
              = \frac{1}{B(\boldsymbol{\alpha})} \prod_{k=1}^K \pi_k^{\alpha_k - 1}

  3. 观测似然 (高斯近似):
       p(\mathbf{d} | \theta) \propto \exp\left( -\frac{1}{2} (\mathbf{d} - \mathbf{f}(\theta))^T C_d^{-1} (\mathbf{d} - \mathbf{f}(\theta)) \right)

  4. 后验分布:
       p(\theta | \mathbf{d}) \propto p(\mathbf{d} | \theta) \, p(\theta)

  5. 最大后验 (MAP) 估计:
       \theta_{MAP} = \arg\max_\theta \left[ \log p(\mathbf{d} | \theta) + \log p(\theta) \right]

数值方法:
  - Newton-Raphson 迭代用于 Gamma 形状参数 ML 估计
  - Digamma / Trigamma 函数通过递推与渐近展开计算
  - 后验采样采用 Metropolis-Hastings MCMC
"""

import numpy as np
from typing import Callable, Tuple, Optional


def digamma_asymptotic(x: np.ndarray) -> np.ndarray:
    """
    计算 Digamma 函数 \psi(x) = d/dx \ln \Gamma(x)。

    递推关系:
        \psi(x+1) = \psi(x) + 1/x

    渐近展开 (x -> +\infty):
        \psi(x) \sim \ln x - \frac{1}{2x} - \frac{1}{12x^2} + \frac{1}{120x^4} - \frac{1}{252x^6}

    参数:
        x: 输入 (> 0)

    返回:
        psi: Digamma 值
    """
    x = np.asarray(x, dtype=np.float64)
    if np.any(x <= 0):
        raise ValueError("digamma requires positive arguments.")

    psi = np.zeros_like(x)

    # 对 x < 6 使用递推提升到足够大
    for _ in range(20):
        mask = x < 6.0
        if not np.any(mask):
            break
        psi[mask] -= 1.0 / x[mask]
        x[mask] += 1.0

    # 渐近展开
    inv_x = 1.0 / x
    inv_x2 = inv_x ** 2
    psi += np.log(x) - 0.5 * inv_x - inv_x2 / 12.0 + inv_x2 ** 2 / 120.0 - inv_x2 ** 3 / 252.0

    return psi


def trigamma_asymptotic(x: np.ndarray) -> np.ndarray:
    """
    计算 Trigamma 函数 \psi_1(x) = d^2/dx^2 \ln \Gamma(x)。

    递推:
        \psi_1(x+1) = \psi_1(x) - 1/x^2

    渐近展开:
        \psi_1(x) \sim 1/x + 1/(2x^2) + 1/(6x^3) - 1/(30x^5) + 1/(42x^7)
    """
    x = np.asarray(x, dtype=np.float64)
    if np.any(x <= 0):
        raise ValueError("trigamma requires positive arguments.")

    psi1 = np.zeros_like(x)

    for _ in range(20):
        mask = x < 6.0
        if not np.any(mask):
            break
        psi1[mask] += 1.0 / (x[mask] ** 2)
        x[mask] += 1.0

    inv_x = 1.0 / x
    inv_x2 = inv_x ** 2
    psi1 += inv_x + 0.5 * inv_x2 + inv_x * inv_x2 / 6.0 - inv_x2 ** 2 * inv_x / 30.0 + inv_x2 ** 3 * inv_x / 42.0

    return psi1


def gamma_log_likelihood(data: np.ndarray,
                         shape: float,
                         scale: float) -> float:
    """
    Gamma 分布的对数似然。

        \ln p(x | \alpha, \beta) = \alpha \ln \beta - \ln \Gamma(\alpha)
                                   + (\alpha - 1) \sum \ln x - \beta \sum x
    """
    alpha = float(shape)
    beta = float(scale)
    x = np.asarray(data, dtype=np.float64)
    x = x[x > 0]
    if len(x) == 0:
        return -1e20

    n = len(x)
    from math import lgamma, log
    logL = n * (alpha * log(beta) - lgamma(alpha))
    logL += (alpha - 1.0) * np.sum(np.log(x))
    logL -= beta * np.sum(x)
    return float(logL)


def gamma_mle_newton_raphson(data: np.ndarray,
                              alpha_init: float = 1.0,
                              tol: float = 1e-8,
                              max_iter: int = 100) -> Tuple[float, float]:
    """
    用 Newton-Raphson 方法估计 Gamma 分布的 MLE。

    对数似然方程:
        \ln \alpha - \psi(\alpha) = \ln \bar{x} - \overline{\ln x}

    Newton 迭代:
        \alpha_{new} = \alpha - \frac{\ln \alpha - \psi(\alpha) - s}{1/\alpha - \psi_1(\alpha)}

    其中 s = \ln \bar{x} - \overline{\ln x}。

    参数:
        data: 观测数据 (>0)
        alpha_init: 初始形状参数

    返回:
        alpha, beta: 估计的 Gamma 参数
    """
    x = np.asarray(data, dtype=np.float64)
    x = x[x > 0]
    if len(x) == 0:
        raise ValueError("No positive data for Gamma MLE.")

    log_x_bar = np.mean(np.log(x))
    x_bar = np.mean(x)
    s = np.log(x_bar) - log_x_bar

    alpha = float(alpha_init)
    for _ in range(max_iter):
        psi_a = float(digamma_asymptotic(np.array([alpha]))[0])
        psi1_a = float(trigamma_asymptotic(np.array([alpha]))[0])

        f = np.log(alpha) - psi_a - s
        fp = 1.0 / alpha - psi1_a

        if abs(fp) < 1e-15:
            break

        alpha_new = alpha - f / fp
        alpha_new = max(alpha_new, 1e-3)

        if abs(alpha_new - alpha) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new

    beta = alpha / x_bar
    return float(alpha), float(beta)


def dirichlet_log_likelihood(data: np.ndarray,
                              alpha: np.ndarray) -> float:
    """
    Dirichlet 分布的对数似然。

        \ln p(X | \boldsymbol{\alpha}) = N \ln \Gamma(\alpha_0) - N \sum_k \ln \Gamma(\alpha_k)
                                       + \sum_k (\alpha_k - 1) \sum_n \ln X_{nk}

    其中 \alpha_0 = \sum_k \alpha_k，数据 X_{nk} 满足 \sum_k X_{nk} = 1。
    """
    X = np.asarray(data, dtype=np.float64)
    alpha_vec = np.asarray(alpha, dtype=np.float64)

    if np.any(alpha_vec <= 0):
        return -1e20

    # 归一化数据行
    row_sums = np.sum(X, axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-15)
    X = X / row_sums

    n, k = X.shape
    alpha0 = np.sum(alpha_vec)

    from math import lgamma
    logL = n * lgamma(alpha0) - n * np.sum([lgamma(a) for a in alpha_vec])
    logL += np.sum((alpha_vec - 1.0) * np.sum(np.log(np.maximum(X, 1e-15)), axis=0))

    return float(logL)


def dirichlet_mle_newton(data: np.ndarray,
                          alpha_init: Optional[np.ndarray] = None,
                          tol: float = 1e-6,
                          max_iter: int = 100) -> np.ndarray:
    """
    Dirichlet 分布参数的 Newton-Raphson MLE 估计。

    梯度:
        g_k = N [\psi(\alpha_0) - \psi(\alpha_k)] + \sum_n \ln X_{nk}

    Fisher 信息:
        I_{kk} = N [\psi_1(\alpha_0) - \psi_1(\alpha_k)]
        I_{kj} = N \psi_1(\alpha_0)  (k \neq j)

    迭代:
        \boldsymbol{\alpha}^{new} = \boldsymbol{\alpha} - I^{-1} g

    参数:
        data: (N, K) 成分数据
        alpha_init: 初始参数

    返回:
        alpha: 估计参数
    """
    X = np.asarray(data, dtype=np.float64)
    row_sums = np.sum(X, axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-15)
    X = X / row_sums

    n, k = X.shape
    if alpha_init is None:
        alpha = np.ones(k, dtype=np.float64) * 2.0
    else:
        alpha = np.asarray(alpha_init, dtype=np.float64).copy()

    log_data = np.sum(np.log(np.maximum(X, 1e-15)), axis=0)

    for _ in range(max_iter):
        alpha0 = np.sum(alpha)
        psi0 = float(digamma_asymptotic(np.array([alpha0]))[0])
        psi_alpha = digamma_asymptotic(alpha)
        psi1_0 = float(trigamma_asymptotic(np.array([alpha0]))[0])
        psi1_alpha = trigamma_asymptotic(alpha)

        g = n * (psi0 - psi_alpha) + log_data

        # Fisher 信息矩阵
        I = np.full((k, k), n * psi1_0, dtype=np.float64)
        I[np.arange(k), np.arange(k)] += n * (psi1_alpha - psi1_0)

        # 解线性系统
        try:
            delta = np.linalg.solve(I, g)
        except np.linalg.LinAlgError:
            break

        # 阻尼 Newton 步
        step_size = 1.0
        alpha_new = alpha - step_size * delta
        alpha_new = np.maximum(alpha_new, 1e-3)

        # 确保目标函数不上升 (简单回溯)
        for _ in range(10):
            new_ll = dirichlet_log_likelihood(X, alpha_new)
            old_ll = dirichlet_log_likelihood(X, alpha)
            if new_ll >= old_ll or step_size < 0.01:
                break
            step_size *= 0.5
            alpha_new = np.maximum(alpha - step_size * delta, 1e-3)

        if np.linalg.norm(alpha_new - alpha) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new

    return alpha


def metropolis_hastings_posterior(log_posterior: Callable[[np.ndarray], float],
                                  theta_init: np.ndarray,
                                  proposal_std: np.ndarray,
                                  n_samples: int = 10000,
                                  burn_in: int = 2000) -> np.ndarray:
    """
    Metropolis-Hastings MCMC 采样后验分布。

    参数:
        log_posterior: 对数后验函数 (必须返回标量，可为负无穷)
        theta_init: 初始参数
        proposal_std: 提议分布标准差
        n_samples: 总采样数
        burn_in: 预烧期

    返回:
        samples: (n_samples - burn_in, dim) 后验样本
    """
    theta = np.asarray(theta_init, dtype=np.float64)
    proposal_std = np.asarray(proposal_std, dtype=np.float64)
    dim = len(theta)

    samples = []
    current_log_p = log_posterior(theta)
    if not np.isfinite(current_log_p):
        current_log_p = -1e20

    n_accepted = 0
    rng = np.random.default_rng(42)

    for i in range(n_samples):
        proposal = theta + proposal_std * rng.standard_normal(dim)
        proposal_log_p = log_posterior(proposal)
        if not np.isfinite(proposal_log_p):
            proposal_log_p = -1e20

        log_alpha = proposal_log_p - current_log_p
        if np.log(rng.random()) < log_alpha:
            theta = proposal
            current_log_p = proposal_log_p
            n_accepted += 1

        if i >= burn_in:
            samples.append(theta.copy())

    acceptance_rate = n_accepted / n_samples
    # print(f"MCMC acceptance rate: {acceptance_rate:.3f}")
    return np.array(samples, dtype=np.float64)


def gamma_sample(alpha: float, beta: float, size: int = 1, seed: int = 42) -> np.ndarray:
    """
    从 Gamma(\alpha, \beta) 分布采样 (尺度参数化: mean = \alpha/\beta)。

    使用 numpy 的默认实现。
    """
    rng = np.random.default_rng(seed)
    return rng.gamma(shape=alpha, scale=1.0 / beta, size=size)
