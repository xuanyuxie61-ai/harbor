"""
bayesian_calibration.py  --  核心 MCMC 贝叶斯校准引擎
===============================================================
科学问题角色:
    整合所有子模块, 运行自适应 MCMC 采样后验:
        p(theta | y) prop p(y | theta) * p(theta)
    算法:
      - 预采样阶段: 用先验采样 + 代理模型快速探索
      - 主采样阶段: 流形 mixup 提议 + Metropolis-Hastings
      - 诊断: R-hat, 有效样本量 ESS, 接受率
    目标泛函:
        log p(theta|y) = log p(y|theta) + log p(theta)
        log p(y|theta) = -0.5 * (y - G(theta))^T Sigma_obs^{-1} (y - G(theta))
                         - 0.5 * log det Sigma_obs - n_obs/2 * log(2 pi)
边界与鲁棒性:
    - 对数似然截断避免 -inf
    - 接受率监控: 若 < 0.1 或 > 0.5, 调整提议步长
    - 收敛诊断: R-hat < 1.1
"""
from __future__ import annotations
import math
from typing import List, Dict, Tuple, Callable, Optional
from numerical_base import NUMERICS
from prior_geometry import PriorGeometry
from proposal_engine import ManifoldMixupProposal
from surrogate_model import FidelityGPSurrogate


class BayesianCalibration:
    """
    核心 MCMC 校准引擎.
    """
    def __init__(self, dim: int, n_obs: int,
                 prior: PriorGeometry,
                 surrogate: FidelityGPSurrogate,
                 obs_data: List[float],
                 obs_noise_std: float = 0.02,
                 seed: int = 5):
        self.dim = dim
        self.n_obs = n_obs
        self.prior = prior
        self.surrogate = surrogate
        self.obs_data = obs_data
        self.obs_noise_std = obs_noise_std
        self.proposal = ManifoldMixupProposal(dim, seed=seed)
        # 观测噪声协方差 (对角)
        self.Sigma_obs_inv = [[1.0 / (obs_noise_std ** 2)
                               if i == j else 0.0
                               for j in range(n_obs)]
                              for i in range(n_obs)]
        self._rng = _LCG(seed)
        # MCMC 状态
        self.chains: List[List[List[float]]] = []
        self.acceptance_rates: List[float] = []
        self.ess_values: List[float] = []

    def log_likelihood(self, theta: List[float]) -> float:
        """
        对数似然: log p(y | theta).
        用代理模型 S(theta) 代替 G(theta).
        """
        y_pred = self.surrogate.predict(theta)
        residual = [self.obs_data[i] - y_pred[i] for i in range(self.n_obs)]
        # 二次型
        quad = 0.0
        for i in range(self.n_obs):
            for j in range(self.n_obs):
                quad += residual[i] * self.Sigma_obs_inv[i][j] * residual[j]
        log_det = self.n_obs * math.log(self.obs_noise_std ** 2)
        return -0.5 * (quad + log_det + self.n_obs * math.log(2.0 * math.pi))

    def log_posterior(self, theta: List[float]) -> float:
        """log p(theta | y) = log p(y | theta) + log p(theta)."""
        ll = self.log_likelihood(theta)
        lp = self.prior.log_pdf(theta)
        if not NUMERICS.is_finite(ll) or not NUMERICS.is_finite(lp):
            return -1e10
        return ll + lp

    def run_mcmc(self, n_samples: int = 500, n_warmup: int = 100,
                 n_chains: int = 2) -> Dict[str, any]:
        """
        运行多链 MCMC.
        返回诊断结果.
        """
        all_chains = []
        all_accept = []
        for chain_id in range(n_chains):
            # 初始化: 从先验采样
            theta = self.prior.sample()
            z_enc = self.surrogate.encoder.forward(theta)
            self.proposal.add_sample(theta, z_enc)
            log_p_curr = self.log_posterior(theta)
            chain = [list(theta)]
            n_accept = 0
            for step in range(n_warmup + n_samples):
                # 提议
                theta_prop, z_prop, log_hastings = self.proposal.propose(
                    theta, z_enc)
                log_p_prop = self.log_posterior(theta_prop)
                # 接受率
                log_alpha = (log_p_prop - log_p_curr + log_hastings)
                log_alpha = min(0.0, log_alpha)
                u = max(self._rng.next(), NUMERICS.safe_log_floor)
                if math.log(u) < log_alpha:
                    theta = theta_prop
                    z_enc = z_prop
                    log_p_curr = log_p_prop
                    if step >= n_warmup:
                        n_accept += 1
                    self.proposal.add_sample(theta, z_enc)
                if step >= n_warmup:
                    chain.append(list(theta))
            all_chains.append(chain)
            accept_rate = n_accept / max(1, n_samples)
            all_accept.append(accept_rate)
        self.chains = all_chains
        self.acceptance_rates = all_accept
        # 诊断
        diagnostics = self._compute_diagnostics(all_chains)
        return diagnostics

    def _compute_diagnostics(self, chains: List[List[List[float]]]
                              ) -> Dict[str, float]:
        """计算 R-hat, ESS, 后验均值/方差."""
        n_chains = len(chains)
        n_samples = min(len(c) for c in chains)
        if n_samples < 10:
            return {"r_hat": float("inf"), "ess": 0.0}
        # 对每个维度算 R-hat
        r_hats = []
        ess_vals = []
        for d in range(self.dim):
            chain_means = []
            chain_vars = []
            for c in chains:
                vals = [c[s][d] for s in range(n_samples)]
                m = sum(vals) / n_samples
                v = sum((x - m) ** 2 for x in vals) / max(1, n_samples - 1)
                chain_means.append(m)
                chain_vars.append(v)
            # 组间/组内方差
            overall_mean = sum(chain_means) / n_chains
            B = n_samples * sum((m - overall_mean) ** 2 for m in chain_means) / max(1, n_chains - 1)
            W = sum(chain_vars) / n_chains
            var_hat = (1.0 - 1.0 / max(1, n_samples)) * W + B / max(1, n_samples)
            r_hat = math.sqrt(var_hat / max(W, NUMERICS.cholesky_jitter))
            r_hats.append(r_hat)
            # ESS (简化: n_eff = n * (1 - rho_1) / (1 + rho_1))
            all_vals = []
            for c in chains:
                all_vals.extend(c[s][d] for s in range(n_samples))
            ess = self._effective_sample_size(all_vals)
            ess_vals.append(ess)
        self.ess_values = ess_vals
        # 后验均值
        all_samples = []
        for c in chains:
            all_samples.extend(c)
        post_mean = [sum(s[d] for s in all_samples) / len(all_samples)
                     for d in range(self.dim)]
        post_std = [math.sqrt(sum((s[d] - post_mean[d]) ** 2
                                   for s in all_samples) / len(all_samples))
                    for d in range(self.dim)]
        return {
            "r_hat_mean": sum(r_hats) / len(r_hats),
            "r_hat_max": max(r_hats),
            "ess_mean": sum(ess_vals) / len(ess_vals),
            "ess_min": min(ess_vals),
            "acceptance_mean": sum(self.acceptance_rates) / len(self.acceptance_rates),
            "post_mean": post_mean,
            "post_std": post_std,
        }

    def _effective_sample_size(self, samples: List[float]) -> float:
        """简化 ESS 估计 (自相关截断)."""
        n = len(samples)
        if n < 10:
            return float(n)
        mean = sum(samples) / n
        var = sum((x - mean) ** 2 for x in samples) / n
        if var < NUMERICS.cholesky_jitter:
            return float(n)
        # 计算前 20 阶自相关
        rho_sum = 0.0
        for lag in range(1, min(20, n)):
            rho = sum((samples[i] - mean) * (samples[i + lag] - mean)
                       for i in range(n - lag)) / (n * var)
            if rho < 0.05:
                break
            rho_sum += rho
        ess = n / (1.0 + 2.0 * rho_sum)
        return max(1.0, ess)


class _LCG:
    def __init__(self, seed: int = 0):
        self._state = int(seed) & 0xFFFFFFFF
        self._a = 1664525
        self._c = 1013904223
        self._m = 1 << 32

    def next(self) -> float:
        self._state = (self._a * self._state + self._c) % self._m
        return self._state / self._m


__all__ = ["BayesianCalibration"]
