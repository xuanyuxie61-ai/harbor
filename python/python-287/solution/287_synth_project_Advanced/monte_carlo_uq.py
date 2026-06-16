# -*- coding: utf-8 -*-
"""
monte_carlo_uq.py
=================

基于 MCMC 的不确定性量化 (UQ) 模块.

来源种子项目:
  - 1099_amsontag_mcpse -> MCMC 抽样与 ODE 参数扫描
  - 778_monopoly_matrix -> Markov 转移矩阵 (用于模式耦合)

物理背景:
  撕裂模增长率的解析理论 (FKR, Copson) 依赖于多个等离子体参数:
    - Lundquist 数 S
    - 波数 kL
    - 平衡磁场梯度

  在真实实验中, 这些参数存在不确定性.
  我们使用 MCMC 方法传播这些不确定性, 得到增长率的后验分布.

  Markov 链:
    模式之间的耦合可以建模为 Markov 过程:
      P(m_{t+1} = j | m_t = i) = T_{ij}
    转移矩阵 T 由物理约束确定:
      - 能量守恒: sum_j T_{ij} = 1
      - 细致平衡: π_i T_{ij} = π_j T_{ji}

  Metropolis-Hastings 算法:
    1. 从当前状态 θ 提议新状态 θ'
    2. 计算接受率: α = min(1, p(θ')/p(θ) * q(θ)/q(θ'))
    3. 以概率 α 接受新状态
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Tuple, Dict


# -------------------------------------------------------------------
#  Metropolis-Hastings MCMC
# -------------------------------------------------------------------
class MetropolisHastings:
    """
    Metropolis-Hastings MCMC 采样器.

    用于等离子体参数不确定性传播.
    """

    def __init__(
        self,
        log_posterior: Callable[[np.ndarray], float],
        proposal_std: np.ndarray,
        seed: int = None,
    ):
        """
        参数:
            log_posterior: 对数后验概率函数
            proposal_std: 提议分布的标准差
            seed: 随机种子
        """
        self.log_posterior = log_posterior
        self.proposal_std = np.asarray(proposal_std)
        self.rng = np.random.default_rng(seed)
        self.current_state = None
        self.current_logp = -np.inf
        self.n_accept = 0
        self.n_total = 0

    def initialize(self, x0: np.ndarray):
        """初始化链."""
        self.current_state = np.asarray(x0, dtype=float).copy()
        self.current_logp = self.log_posterior(self.current_state)
        self.n_accept = 0
        self.n_total = 0

    def step(self) -> np.ndarray:
        """
        单步 Metropolis-Hastings.

        提议分布: 高斯随机游走
            θ' = θ + N(0, proposal_std^2)

        接受率:
            α = min(1, exp(logp(θ') - logp(θ)))
        """
        if self.current_state is None:
            raise RuntimeError("链未初始化, 请先调用 initialize()")

        # 提议
        proposal = self.current_state + self.rng.normal(
            size=self.current_state.size
        ) * self.proposal_std

        # 计算对数后验
        prop_logp = self.log_posterior(proposal)

        # 接受/拒绝
        log_alpha = prop_logp - self.current_logp
        self.n_total += 1

        if np.log(self.rng.uniform()) < log_alpha:
            self.current_state = proposal
            self.current_logp = prop_logp
            self.n_accept += 1

        return self.current_state.copy()

    def sample(
        self,
        x0: np.ndarray,
        n_samples: int,
        burn_in: int = 0,
        thin: int = 1,
    ) -> Tuple[np.ndarray, float]:
        """
        采样 MCMC 链.

        参数:
            x0: 初始状态
            n_samples: 样本数
            burn_in: 燃烧期步数
            thin: 稀释间隔

        返回:
            (chain, acceptance_rate)
        """
        self.initialize(x0)

        total_steps = burn_in + n_samples * thin
        chain = np.zeros((n_samples, x0.size))
        sample_idx = 0

        for step in range(total_steps):
            state = self.step()
            if step >= burn_in and (step - burn_in) % thin == 0:
                if sample_idx < n_samples:
                    chain[sample_idx] = state
                    sample_idx += 1

        acceptance_rate = self.n_accept / max(1, self.n_total)
        return chain, acceptance_rate


# -------------------------------------------------------------------
#  撕裂模增长率的不确定性量化
# -------------------------------------------------------------------
def tearing_growth_rate_model(params: np.ndarray) -> float:
    """
    撕裂模增长率模型 (输入: 等离子体参数).

    参数:
        params: [log10(S), kL, delta_prime]

    返回:
        gamma * tau_A (归一化增长率)
    """
    log_S, kL, dp = params
    S = 10.0 ** log_S

    if kL <= 0.0 or dp <= 0.0:
        return 0.0

    # FKR 标度: gamma * tau_R ~ 0.6 * (kL)^{2/5} * S^{3/5} * dp^{4/5}
    # gamma * tau_A = gamma * tau_R / S
    gamma_tau_R = 0.6 * (kL ** (2.0 / 3.0)) * (S ** 0.6) * (dp ** 0.8)
    return gamma_tau_R / S


def log_posterior_tearing(
    params: np.ndarray,
    data_mean: float,
    data_std: float,
    prior_std: np.ndarray,
) -> float:
    """
    撕裂模参数的对数后验.

    模型: gamma_model(params) ~ N(data_mean, data_std^2)
    先验: params ~ N(0, prior_std^2)
    """
    # 先验
    log_prior = -0.5 * np.sum((params / prior_std) ** 2)

    # 似然
    gamma_model = tearing_growth_rate_model(params)
    log_likelihood = -0.5 * ((gamma_model - data_mean) / data_std) ** 2

    return log_prior + log_likelihood


# -------------------------------------------------------------------
#  Markov 转移矩阵 (来自 778_monopoly_matrix)
# -------------------------------------------------------------------
class ModeCouplingMarkov:
    """
    模式耦合的 Markov 链模型.

    状态: 各撕裂模模式 (不同 m 数)
    转移: 非线性耦合导致模式间的能量转移

    转移矩阵 T 满足:
      - T_{ij} >= 0 (非负)
      - sum_j T_{ij} = 1 (行随机)
    """

    def __init__(self, n_modes: int, seed: int = None):
        self.n_modes = n_modes
        self.rng = np.random.default_rng(seed)
        self.transition_matrix = None

    def build_transition_matrix(
        self,
        coupling_strength: np.ndarray,
        decay_rates: np.ndarray,
    ) -> np.ndarray:
        """
        构造转移矩阵.

        参数:
            coupling_strength: (n_modes, n_modes) 耦合系数
            decay_rates: (n_modes,) 各模式衰减速率

        返回:
            T: 行随机转移矩阵
        """
        n = self.n_modes
        T = np.zeros((n, n))

        for i in range(n):
            # 对角元: 模式自身存留
            T[i, i] = np.exp(-decay_rates[i])
            # 非对角元: 转移到其他模式
            for j in range(n):
                if i != j:
                    T[i, j] = coupling_strength[i, j] * (1.0 - T[i, i])

            # 行归一化
            row_sum = np.sum(T[i])
            if row_sum > 0:
                T[i] /= row_sum

        self.transition_matrix = T
        return T

    def stationary_distribution(self) -> np.ndarray:
        """
        计算平稳分布 π: π T = π, sum(π) = 1.
        """
        if self.transition_matrix is None:
            raise RuntimeError("转移矩阵未构造")

        T = self.transition_matrix
        # 求解 (T^T - I) π = 0, sum(π) = 1
        n = T.shape[0]
        A = np.vstack([T.T - np.eye(n), np.ones(n)])
        b = np.zeros(n + 1)
        b[-1] = 1.0
        pi = np.linalg.lstsq(A, b, rcond=None)[0]
        pi = np.maximum(pi, 0)
        pi /= np.sum(pi)
        return pi

    def simulate_chain(
        self,
        n_steps: int,
        initial_state: int = 0,
    ) -> np.ndarray:
        """
        模拟 Markov 链轨迹.
        """
        if self.transition_matrix is None:
            raise RuntimeError("转移矩阵未构造")

        chain = np.zeros(n_steps, dtype=int)
        chain[0] = initial_state

        for t in range(1, n_steps):
            probs = self.transition_matrix[chain[t - 1]]
            chain[t] = self.rng.choice(self.n_modes, p=probs)

        return chain


# -------------------------------------------------------------------
#  汇总统计
# -------------------------------------------------------------------
def summarize_chain(
    chain: np.ndarray,
    param_names: list = None,
) -> Dict[str, np.ndarray]:
    """
    汇总 MCMC 链的统计量.

    返回:
        dict: "mean", "std", "median", "q05", "q95"
    """
    n_params = chain.shape[1]
    if param_names is None:
        param_names = [f"p{i}" for i in range(n_params)]

    results = {}
    for i, name in enumerate(param_names):
        results[f"{name}_mean"] = np.mean(chain[:, i])
        results[f"{name}_std"] = np.std(chain[:, i])
        results[f"{name}_median"] = np.median(chain[:, i])
        results[f"{name}_q05"] = np.percentile(chain[:, i], 5)
        results[f"{name}_q95"] = np.percentile(chain[:, i], 95)

    return results
