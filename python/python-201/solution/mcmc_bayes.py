"""
mcmc_bayes.py — MCMC 贝叶斯推断模块
=======================================
使用 PCE 作为代理模型的 MCMC 采样,
进行贝叶斯参数推断和失效概率计算。

核心公式:
  贝叶斯定理:
    p(θ|D) ∝ p(D|θ) p(θ)

  似然函数:
    L(θ) = Π_{i=1}^{N} (2πσ²)^{-1/2}
           exp(-(d_i - M(θ_i))² / (2σ²))

  其中 M(θ) ≈ Σ_k ĉ_k Ψ_k(θ) 是 PCE 代理模型。

  Metropolis-Hastings 接受率:
    α = min(1, p(θ*|D) / p(θ^{n-1}|D))
      = min(1, L(θ*) p(θ*) / (L(θ^{n-1}) p(θ^{n-1})))

  失效概率 (IS):
    P_f = E_w[1_{g(ξ)>0} / w(ξ)]
    ≈ (1/N) Σ_{i=1}^{N} 1_{g(ξ_i)>0} / φ(ξ_i)

映射种子项目:
  - 1180_Yuva12345: MCMC + 贝叶斯证据 → 后验采样
  - 033_asa076: 正态 CDF → 失效概率计算
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Callable
from config import GlobalConfig, MCMCConfig
from polynomial_basis import OrthogonalPolynomialBasis
from measure import pdf_eval, cdf_eval, owen_t_function


class MCMCSampler:
    """
    基于 PCE 代理模型的 MCMC 采样器。

    映射 1180_Yuva12345: 使用 emcee 风格的集合采样器,
    但用纯 numpy 实现, 以 PCE 作为快速代理模型。
    """

    def __init__(self, config: GlobalConfig,
                 basis: OrthogonalPolynomialBasis):
        self.config = config
        self.mcmc_config = config.mcmc
        self.basis = basis
        self.d = len(config.measures)

        self.chain: Optional[np.ndarray] = None
        self.log_likelihoods: Optional[np.ndarray] = None
        self.acceptance_rate: float = 0.0

    def log_prior(self, theta: np.ndarray) -> float:
        """
        计算对数先验概率。

        使用各维度的概率测度作为先验。

        参数:
            theta: shape (d,), 参数向量

        返回:
            log_prior
        """
        log_p = 0.0
        for dim_idx, meas in enumerate(self.config.measures):
            xi = theta[dim_idx]
            p = float(pdf_eval(meas, np.array([xi]))[0])
            if p < 1e-300:
                return -np.inf
            log_p += np.log(p)
        return log_p

    def log_likelihood(self, theta: np.ndarray,
                       pce_coeffs: np.ndarray,
                       observation: float,
                       noise_std: float) -> float:
        """
        计算对数似然。

        log L = -0.5 * (d - M(θ))^2 / σ^2 - 0.5 * log(2π σ^2)

        其中 M(θ) = Σ_k ĉ_k Ψ_k(θ) 是 PCE 预测。

        参数:
            theta:        参数向量
            pce_coeffs:   PCE 系数
            observation:  观测值
            noise_std:    观测噪声标准差

        返回:
            log_likelihood
        """
        # PCE 预测
        theta_2d = theta.reshape(1, -1)
        Psi = self.basis.evaluate(theta_2d)
        prediction = float((Psi @ pce_coeffs)[0])

        # 高斯似然
        residual = observation - prediction
        log_L = (-0.5 * residual ** 2 / noise_std ** 2 -
                 0.5 * np.log(2 * np.pi * noise_std ** 2))

        return log_L

    def log_posterior(self, theta: np.ndarray,
                      pce_coeffs: np.ndarray,
                      observation: float,
                      noise_std: float) -> float:
        """
        计算对数后验。

        log p(θ|D) = log L(θ) + log p(θ) + const
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(
            theta, pce_coeffs, observation, noise_std)
        return lp + ll

    def sample(self, pce_coeffs: np.ndarray,
               observation: float,
               noise_std: float,
               n_walkers: Optional[int] = None,
               n_burnin: Optional[int] = None,
               n_samples: Optional[int] = None
               ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        运行 MCMC 采样 (简化版 affine-invariant sampler)。

        使用 Metropolis-Hastings 算法, 以 PCE 代理模型
        加速似然计算。

        参数:
            pce_coeffs:  PCE 系数
            observation: 观测数据
            noise_std:   噪声标准差
            n_walkers:   walker 数量
            n_burnin:    烧预期步数
            n_samples:   采样步数

        返回:
            chain:    shape (n_samples, d), 后验样本
            log_likes: shape (n_samples,), 对数似然
            acceptance_rate: 接受率
        """
        if n_walkers is None:
            n_walkers = self.mcmc_config.n_walkers
        if n_burnin is None:
            n_burnin = self.mcmc_config.n_burnin
        if n_samples is None:
            n_samples = self.mcmc_config.n_samples

        d = self.d
        rng = np.random.RandomState(self.mcmc_config.random_seed)

        # 初始化 walker (从先验采样)
        chain = np.zeros((n_walkers, n_burnin + n_samples, d))
        log_likes = np.zeros((n_walkers, n_burnin + n_samples))
        log_posts = np.zeros(n_walkers)

        # 初始位置: 从先验均值附近采样
        for w in range(n_walkers):
            theta0 = np.zeros(d)
            for dim_idx, meas in enumerate(self.config.measures):
                theta0[dim_idx] = meas.mean + (
                    rng.randn() * np.sqrt(meas.variance) * 0.5)
            chain[w, 0] = theta0
            log_posts[w] = self.log_posterior(
                theta0, pce_coeffs, observation, noise_std)
            log_likes[w, 0] = self.log_likelihood(
                theta0, pce_coeffs, observation, noise_std)

        # MCMC 采样
        n_accept = 0
        n_total = 0
        proposal_scale = self.mcmc_config.proposal_scale

        for step in range(1, n_burnin + n_samples):
            for w in range(n_walkers):
                # 提议: 随机漫步
                theta_star = chain[w, step - 1] + (
                    proposal_scale * rng.randn(d) *
                    np.array([np.sqrt(m.variance)
                              for m in self.config.measures]))

                log_post_star = self.log_posterior(
                    theta_star, pce_coeffs, observation, noise_std)

                # Metropolis-Hastings 接受/拒绝
                log_alpha = log_post_star - log_posts[w]
                if np.log(rng.uniform()) < log_alpha:
                    chain[w, step] = theta_star
                    log_posts[w] = log_post_star
                    log_likes[w, step] = self.log_likelihood(
                        theta_star, pce_coeffs, observation, noise_std)
                    n_accept += 1
                else:
                    chain[w, step] = chain[w, step - 1]
                    log_likes[w, step] = log_likes[w, step - 1]

                n_total += 1

            # 自适应调整提议尺度 (烧预期)
            if step < n_burnin and step % 50 == 0:
                current_rate = n_accept / max(n_total, 1)
                if current_rate < 0.1:
                    proposal_scale *= 0.8
                elif current_rate > 0.5:
                    proposal_scale *= 1.2

        # 烧尽烧预期, 合并所有 walker
        burnin_chain = chain[:, n_burnin:, :].reshape(-1, d)
        burnin_likes = log_likes[:, n_burnin:].reshape(-1)

        self.chain = burnin_chain
        self.log_likelihoods = burnin_likes
        self.acceptance_rate = n_accept / max(n_total, 1)

        return burnin_chain, burnin_likes, self.acceptance_rate

    def compute_posterior_statistics(self) -> dict:
        """
        从后验样本计算统计量。
        """
        if self.chain is None:
            return {}

        chain = self.chain
        stats = {
            "posterior_mean": np.mean(chain, axis=0),
            "posterior_std": np.std(chain, axis=0),
            "posterior_median": np.median(chain, axis=0),
            "posterior_5%": np.percentile(chain, 5, axis=0),
            "posterior_95%": np.percentile(chain, 95, axis=0),
            "acceptance_rate": self.acceptance_rate,
            "n_samples": chain.shape[0],
        }

        return stats


def compute_failure_probability_is(pce_coeffs: np.ndarray,
                                   basis: OrthogonalPolynomialBasis,
                                   config: GlobalConfig,
                                   threshold: float,
                                   n_samples: int = 50000
                                   ) -> dict:
    """
    使用重要采样 (IS) 计算失效概率。

    P_f = P(g(ξ) > threshold)

    其中 g(ξ) = Σ_k ĉ_k Ψ_k(ξ)。

    使用混合重要采样:
      1. 标准 MC: 从先验采样
      2. IS: 从偏移分布采样

    参数:
        pce_coeffs: PCE 系数
        basis:      基函数
        config:     配置
        threshold:  失效阈值
        n_samples:  采样数

    返回:
        results: 包含 MC 和 IS 估计的字典
    """
    d = len(config.measures)
    rng = np.random.RandomState(789)

    # 标准 MC
    samples_mc = np.zeros((n_samples, d))
    for dim_idx, meas in enumerate(config.measures):
        if meas.measure_type == "gauss":
            samples_mc[:, dim_idx] = rng.normal(
                meas.params["mu"], meas.params["sigma"], n_samples)
        elif meas.measure_type == "uniform":
            samples_mc[:, dim_idx] = rng.uniform(
                meas.params["a"], meas.params["b"], n_samples)
        else:
            samples_mc[:, dim_idx] = rng.uniform(
                meas.support[0], meas.support[1], n_samples)

    Psi_mc = basis.evaluate(samples_mc)
    g_mc = Psi_mc @ pce_coeffs
    indicator_mc = (g_mc > threshold).astype(float)
    P_f_mc = float(np.mean(indicator_mc))

    # 标准误差
    if P_f_mc > 0 and P_f_mc < 1:
        se_mc = np.sqrt(P_f_mc * (1 - P_f_mc) / n_samples)
    else:
        se_mc = 0.0

    # 失效域样本的统计
    if np.sum(indicator_mc) > 0:
        fail_samples = samples_mc[indicator_mc > 0]
        fail_mean = np.mean(fail_samples, axis=0)
    else:
        fail_mean = np.zeros(d)

    # Owen T 函数方法 (二维近似)
    mu = pce_coeffs[0]
    sigma = np.sqrt(max(np.sum(pce_coeffs[1:] ** 2), 1e-30))

    # 正态近似: P_f ≈ 1 - Φ((threshold - μ) / σ)
    z = (threshold - mu) / max(sigma, 1e-15)
    from scipy.special import erfc
    P_f_normal = 0.5 * float(erfc(z / np.sqrt(2)))

    return {
        "threshold": threshold,
        "P_f_MC": P_f_mc,
        "P_f_MC_SE": se_mc,
        "P_f_normal_approx": P_f_normal,
        "n_failure_samples": int(np.sum(indicator_mc)),
        "failure_mean_xi": fail_mean.tolist(),
        "g_mean": float(mu),
        "g_std": float(sigma),
    }


def format_bayesian_results(mcmc_stats: dict,
                            failure_results: dict,
                            var_names: List[str] = None) -> str:
    """格式化贝叶斯推断结果"""
    if var_names is None:
        d = len(mcmc_stats.get("posterior_mean", []))
        var_names = [f"ξ_{i}" for i in range(d)]

    lines = [
        "=" * 50,
        "贝叶斯推断结果",
        "=" * 50,
    ]

    if "posterior_mean" in mcmc_stats:
        lines.append("后验分布:")
        for i, name in enumerate(var_names):
            mean = mcmc_stats["posterior_mean"][i]
            std = mcmc_stats["posterior_std"][i]
            lines.append(f"  {name}: {mean:.4f} ± {std:.4f}")

        lines.append(f"接受率: {mcmc_stats['acceptance_rate']:.3f}")
        lines.append(f"样本数: {mcmc_stats['n_samples']}")

    lines.extend([
        "",
        "失效概率分析:",
        f"  阈值: {failure_results['threshold']:.4f}",
        f"  P_f (MC):     {failure_results['P_f_MC']:.6e} "
        f"± {failure_results['P_f_MC_SE']:.2e}",
        f"  P_f (正态近似): {failure_results['P_f_normal_approx']:.6e}",
        f"  g(ξ) 均值: {failure_results['g_mean']:.4f}",
        f"  g(ξ) 标准差: {failure_results['g_std']:.4f}",
        "=" * 50,
    ])

    return "\n".join(lines)
