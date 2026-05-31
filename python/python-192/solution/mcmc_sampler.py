"""
================================================================================
马尔可夫链蒙特卡洛(MCMC)统计采样模块 (mcmc_sampler.py)
================================================================================
融合项目:
  - 1095_snakes_probability: 马尔可夫链转移矩阵与概率分布

在湍流CFD的不确定性量化(UQ)中，MCMC用于采样模型参数的后验分布。
本模块提供：
  1. Metropolis-Hastings算法（参数后验采样）
  2. 马尔可夫链转移矩阵构建
  3. 稳态分布与收敛诊断

数学基础:
    对参数θ，后验分布由Bayes定理给出：

        p(θ|D) ∝ p(D|θ) · p(θ)

    Metropolis-Hastings算法通过提议分布 q(θ*|θ_t) 生成候选点，
    以接受概率 α = min(1, p(θ*)q(θ_t|θ*) / p(θ_t)q(θ*|θ_t)) 决定是否转移。
================================================================================
"""

import numpy as np
from utils_numerical import safe_divide


def build_markov_transition_matrix(n_states: int, move_range: int = 1,
                                   boundary_reflect: bool = True) -> np.ndarray:
    """
    构建一维随机游走的马尔可夫转移矩阵

    状态空间 S = {0, 1, ..., N-1}，每次以相等概率移动到相邻状态。

    转移概率:
        P(i → i±1) = 1/2,   0 < i < N-1
        P(0 → 1) = 1,       P(N-1 → N-2) = 1  (反射边界)

    稳态分布: π = [1/N, 1/N, ..., 1/N]（均匀分布）

    参数:
        n_states: 状态数
        move_range: 每次移动最大步数
        boundary_reflect: 边界是否反射

    返回:
        P: 转移矩阵 (n_states x n_states)
    """
    P = np.zeros((n_states, n_states))

    for i in range(n_states):
        neighbors = []
        weights = []

        for step in range(1, move_range + 1):
            if i - step >= 0:
                neighbors.append(i - step)
                weights.append(1.0)
            elif boundary_reflect:
                neighbors.append(step - i - 1)
                weights.append(1.0)

            if i + step < n_states:
                neighbors.append(i + step)
                weights.append(1.0)
            elif boundary_reflect:
                neighbors.append(2 * n_states - i - step - 1)
                weights.append(1.0)

        if len(neighbors) == 0:
            P[i, i] = 1.0
        else:
            total = sum(weights)
            for j, w in zip(neighbors, weights):
                P[i, j] += w / total

    return P


def metropolis_hastings_sampler(log_posterior, x0: np.ndarray, n_samples: int = 5000,
                                proposal_cov: np.ndarray = None,
                                burn_in: int = 1000, thin: int = 5) -> dict:
    """
    Metropolis-Hastings MCMC采样器

    算法流程:
        1. 初始化 θ₀ = x0
        2. 对 t = 1,...,T:
           a. 从提议分布采样 θ* ~ N(θ_{t-1}, Σ)
           b. 计算接受概率 α = min(1, exp(log_posterior(θ*) - log_posterior(θ_{t-1})))
           c. 以概率α接受 θ_t = θ*，否则 θ_t = θ_{t-1}
        3. 丢弃burn-in期，按thin间隔取样

    参数:
        log_posterior: 对数后验密度函数
        x0: 初始参数向量
        n_samples: 目标样本数
        proposal_cov: 提议分布协方差矩阵
        burn_in: 预烧期长度
        thin: 稀释间隔

    返回:
        dict 包含样本链、接受率、统计量
    """
    dim = len(x0)
    if proposal_cov is None:
        proposal_cov = np.eye(dim) * 0.01

    total_iterations = burn_in + n_samples * thin
    chain = np.zeros((total_iterations, dim))
    chain[0, :] = x0

    log_p_current = log_posterior(x0)
    accepts = 0

    for t in range(1, total_iterations):
        # 提议
        proposal = np.random.multivariate_normal(chain[t - 1, :], proposal_cov)

        log_p_proposal = log_posterior(proposal)
        log_alpha = log_p_proposal - log_p_current

        if np.random.rand() < np.exp(min(log_alpha, 0.0)):
            chain[t, :] = proposal
            log_p_current = log_p_proposal
            accepts += 1
        else:
            chain[t, :] = chain[t - 1, :]

    # 丢弃burn-in并稀释
    samples = chain[burn_in::thin, :]

    # 统计量
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    acceptance_rate = accepts / total_iterations

    # 收敛诊断：Gelman-Rubin R-hat（单链用分割法近似）
    n_eff = len(samples)
    if n_eff > 100:
        mid = n_eff // 2
        var_first = np.var(samples[:mid, :], axis=0, ddof=1)
        var_second = np.var(samples[mid:, :], axis=0, ddof=1)
        W = 0.5 * (var_first + var_second)
        B = np.var([np.mean(samples[:mid, :], axis=0), np.mean(samples[mid:, :], axis=0)], axis=0, ddof=1)
        V_hat = (mid - 1) / mid * W + B
        r_hat = np.sqrt(V_hat / (W + 1e-14))
    else:
        r_hat = np.ones(dim)

    return {
        'samples': samples,
        'chain': chain,
        'mean': mean,
        'std': std,
        'acceptance_rate': acceptance_rate,
        'r_hat': r_hat,
        'n_samples': len(samples)
    }


def sample_turbulence_parameters(u_data: np.ndarray, v_data: np.ndarray,
                                 n_samples: int = 2000) -> dict:
    """
    对湍流模型参数进行MCMC后验采样

    假设观测数据服从正态分布，对以下参数采样：
      - C_μ: 涡粘性系数
      - σ_k: 湍动能Prandtl数
      - σ_ε: 耗散率Prandtl数

    似然函数:
        p(D|θ) = ∏_i N(u_i | u_model(θ), σ²)

    先验分布:
        C_μ ~ Uniform(0.05, 0.15)
        σ_k ~ Uniform(0.5, 2.0)
        σ_ε ~ Uniform(0.5, 2.0)
    """
    # 简化模型：k-epsilon模型参数
    def log_posterior(theta):
        C_mu, sigma_k, sigma_eps = theta

        # 先验约束
        if not (0.05 <= C_mu <= 0.15 and 0.5 <= sigma_k <= 2.0 and 0.5 <= sigma_eps <= 2.0):
            return -np.inf

        # 简化模型预测（用观测数据的统计量近似）
        k_obs = 0.5 * (np.var(u_data) + np.var(v_data))
        epsilon_obs = k_obs ** 1.5 / 0.1  # 简化耗散率估计

        # 模型预测
        nu_t = C_mu * k_obs ** 2 / (epsilon_obs + 1e-14)
        k_pred = nu_t / (C_mu + 1e-14)

        # 似然（高斯）
        sigma_noise = 0.1 * k_obs
        log_likelihood = -0.5 * ((k_obs - k_pred) / sigma_noise) ** 2

        # 对数先验（均匀先验的对数为常数，在约束外为-inf）
        log_prior = 0.0

        return log_likelihood + log_prior

    x0 = np.array([0.09, 1.0, 1.3])
    proposal_cov = np.diag([0.001, 0.05, 0.05])

    result = metropolis_hastings_sampler(
        log_posterior, x0, n_samples=n_samples,
        proposal_cov=proposal_cov, burn_in=500, thin=3
    )

    return result


def compute_markov_chain_stationary(P: np.ndarray, max_iter: int = 500, tol: float = 1e-12) -> np.ndarray:
    """
    通过幂迭代计算马尔可夫链的稳态分布

        π_{k+1} = π_k · P

    当 P 不可约且非周期时，π_k 收敛到唯一稳态分布 π*。

    收敛速率由第二大特征值决定：
        ||π_k - π*|| ≤ C · |λ₂|^k
    """
    n = P.shape[0]
    pi = np.ones(n) / n

    for _ in range(max_iter):
        pi_new = pi @ P
        if np.linalg.norm(pi_new - pi, 1) < tol:
            break
        pi = pi_new

    return pi
