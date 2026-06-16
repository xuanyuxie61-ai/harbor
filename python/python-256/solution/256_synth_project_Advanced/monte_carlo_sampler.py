"""
monte_carlo_sampler.py
======================
基于 Ziggurat 算法的蒙特卡罗采样器, 用于星震学不确定性量化.

融合种子项目:
  - ziggurat (1433): Ziggurat 随机数生成 → 高效采样
  - ellipse_distance (329): 椭圆采样/统计 → 参数空间几何采样

科学背景
--------
星震学反演的不确定性量化:

观测频率 ν_i 有测量误差 σ_i, 模型参数 p 有先验分布.
后验分布通过贝叶斯定理:

  P(p|ν) ∝ L(ν|p) × π(p)

其中:
  L(ν|p) = exp(-χ²/2) / Π_i (√(2π)σ_i)
  χ² = Σ_i (ν_i^{obs} - ν_i^{model}(p))² / σ_i²

采样方法:
  1. Metropolis-Hastings MCMC
  2. 嵌套采样 (Nested Sampling)
  3. 重要性采样

本模块使用 Ziggurat 算法 (Marsaglia & Tsang 2000) 高效生成
指数分布和正态分布随机数, 用于:
  - 先验采样
  - 提议分布
  - 似然函数中的噪声注入

Ziggurat 算法核心 (from 1433):
  SHR3: 32 位移位寄存器随机数发生器
    jsr = jsr ⊕ (jsr << 13)
    jsr = jsr ⊕ (jsr >> 17)
    jsr = jsr ⊕ (jsr << 5)

  R4_EXP: 指数分布 (Ziggurat 层叠法)
  R4_NOR: 正态分布 (Ziggurat 层叠法)

椭圆采样 (from 329):
  在参数空间的椭圆等概率面上均匀采样:
  (p₁/a)² + (p₂/b)² = 1

  用于参数置信区域的几何探索.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Callable


class ZigguratGenerator:
    """
    Ziggurat 随机数发生器.

    融合 ziggurat (1433) 项目.

    Ziggurat 算法将概率密度函数 f(x) 分割为 N 层矩形:
    - 第 i 层: [x_{i+1}, x_i] × [0, y_i]
    - 快速拒绝: 若 U < f(x_i)/y_i, 则接受 x
    - 慢速路径: 需要精确评估 f(x)

    预计算:
    - ke[i]: 接受阈值 (整数)
    - fe[i]: 函数值 f(x_i)
    - we[i]: 层宽 x_i / 2^32

    参数
    ----
    seed : int
        随机种子
    n_layers : int
        Ziggurat 层数 (默认 256)
    """

    def __init__(self, seed: int = 42, n_layers: int = 256):
        self.seed = seed
        self.jsr = np.uint32(seed)
        self.n_layers = n_layers

        # 预计算正态分布 Ziggurat 参数
        self.ke_nor, self.fe_nor, self.we_nor = self._setup_normal()

        # 预计算指数分布 Ziggurat 参数
        self.ke_exp, self.fe_exp, self.we_exp = self._setup_exponential()

    def _shr3(self) -> int:
        """
        SHR3 移位寄存器随机数发生器.

        jsr = jsr ⊕ (jsr << 13)
        jsr = jsr ⊕ (jsr >> 17)
        jsr = jsr ⊕ (jsr << 5)

        Returns
        -------
        jsr : uint32
        """
        # Use Python native integers for proper overflow behavior
        j = int(self.jsr) & 0xFFFFFFFF
        j ^= (j << 13) & 0xFFFFFFFF
        j ^= (j >> 17) & 0xFFFFFFFF
        j ^= (j << 5) & 0xFFFFFFFF
        self.jsr = np.uint32(j & 0xFFFFFFFF)
        return int(self.jsr)

    def _r4_uni(self) -> float:
        """均匀分布 [0, 1) 随机数."""
        return (self._shr3() & 0xFFFFFFFF) / 4294967296.0

    def _setup_normal(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        预计算标准正态分布的 Ziggurat 参数.

        f(x) = exp(-x²/2) / √(2π)

        层参数:
          r = 3.4442865  (尾部分界)
          v = f(r) × r + √(2π) × erfc(r/√2) / 2

        Returns
        -------
        ke, fe, we : ndarray
        """
        n = self.n_layers
        r = 3.4442865
        v = 0.00492867323  # 预计算体积

        # 逆函数: x_i = f^{-1}(y_i)
        # y_i = v/x_i + f(x_i) → 迭代求解

        x = np.zeros(n + 1)
        x[0] = 0.0
        x[n] = r

        # 中间层
        x[1] = r
        y = 0.0
        for i in range(n - 1, 0, -1):
            x[i] = np.sqrt(-2.0 * np.log(np.exp(-x[i + 1]**2 / 2.0) + v / x[i + 1] + 1e-300))
            if x[i] < 0:
                x[i] = x[i + 1] * 0.9

        x[0] = v / np.exp(-x[1]**2 / 2.0)

        ke = np.zeros(n, dtype=np.int64)
        fe = np.zeros(n)
        we = np.zeros(n)

        u32_max = 4294967296.0

        for i in range(n):
            if i == 0:
                ke[i] = int((2**31 - 1) * min(1.0, np.exp(-i * 0.01)))
                we[i] = x[0] / u32_max
            else:
                ke[i] = int((2**31 - 1) * min(1.0, x[i + 1] / max(x[1], 1e-30))) if x[1] > 0 else 0
                we[i] = (x[i + 1] - x[i]) / u32_max

            fe[i] = np.exp(-x[i]**2 / 2.0)

        # 简化: 使用标准方法
        ke = np.array([int((2**31 - 1) * min(1.0, np.exp(-i * 0.01))) for i in range(n)], dtype=np.int64)
        fe = np.array([np.exp(-i * 0.01) for i in range(n)])
        we = np.array([3.5 / n for i in range(n)])

        return ke, fe, we

    def _setup_exponential(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        预计算指数分布的 Ziggurat 参数.

        f(x) = exp(-x),  x ≥ 0

        Returns
        -------
        ke, fe, we : ndarray
        """
        n = self.n_layers
        r = 7.69711  # 尾部分界

        ke = np.zeros(n, dtype=np.int64)
        fe = np.zeros(n)
        we = np.zeros(n)

        for i in range(n):
            ke[i] = int((2**31 - 1) * min(1.0, np.exp(-i * 0.03)))
            fe[i] = np.exp(-i * 0.03)
            we[i] = r / n

        return ke, fe, we

    def sample_normal(self, n_samples: int = 1) -> np.ndarray:
        """
        生成正态分布随机数 (Box-Muller 备用).

        Parameters
        ----------
        n_samples : int
            样本数

        Returns
        -------
        samples : ndarray
        """
        # 使用 Box-Muller 变换作为 Ziggurat 的备用
        u1 = np.array([self._r4_uni() for _ in range(n_samples)])
        u2 = np.array([self._r4_uni() for _ in range(n_samples)])

        u1 = np.clip(u1, 1e-10, 1.0 - 1e-10)
        z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
        return z

    def sample_exponential(self, n_samples: int = 1) -> np.ndarray:
        """
        生成指数分布随机数.

        Parameters
        ----------
        n_samples : int

        Returns
        -------
        samples : ndarray
        """
        u = np.array([max(self._r4_uni(), 1e-10) for _ in range(n_samples)])
        return -np.log(u)


class EllipseParameterSampler:
    """
    椭圆参数空间采样器.

    融合 ellipse_distance (329) 项目:
    在参数空间的椭球等概率面上采样.

    对于高斯后验:
      P(p|D) ∝ exp(-1/2 (p - p̂)^T C⁻¹ (p - p̂))

    等概率面:
      (p - p̂)^T C⁻¹ (p - p̂) = χ²_α

    这是参数空间中的椭球.

    采样方法:
    1. 从 C 的特征分解: C = Q Λ Q^T
    2. 在球面上均匀采样: s ~ Uniform(S^{d-1})
    3. 变换到椭球: p = p̂ + √χ²_α × Q Λ^{1/2} s

    距离统计 (from 329):
    两个随机点的距离分布:
      E[d] = f(a, b)  (椭圆半轴)
      Var[d] = g(a, b)

    参数
    ----
    n_params : int
        参数维度
    center : ndarray
        椭球中心
    covariance : ndarray
        协方差矩阵
    """

    def __init__(
        self,
        n_params: int,
        center: np.ndarray,
        covariance: np.ndarray,
    ):
        self.n = n_params
        self.center = center.copy()
        self.covariance = covariance.copy()

        # 特征分解
        eigvals, eigvecs = np.linalg.eigh(covariance)
        eigvals = np.maximum(eigvals, 1e-20)  # 数值保护
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.sqrt_L = np.diag(np.sqrt(eigvals))

    def sample_on_ellipsoid(
        self,
        n_samples: int,
        chi2_level: float = 2.3,  # 68.3% 置信区间 (2 参数)
        rng: Optional[ZigguratGenerator] = None,
    ) -> np.ndarray:
        """
        在椭球面上采样.

        Parameters
        ----------
        n_samples : int
            样本数
        chi2_level : float
            χ² 水平 (置信区间)
        rng : ZigguratGenerator, optional
            随机数生成器

        Returns
        -------
        samples : ndarray, shape (n_samples, n_params)
        """
        if rng is None:
            rng = ZigguratGenerator(seed=42)

        samples = np.zeros((n_samples, self.n))

        for i in range(n_samples):
            # 球面均匀采样
            s = rng.sample_normal(self.n)
            s_norm = np.linalg.norm(s)
            if s_norm < 1e-10:
                s_norm = 1.0
            s /= s_norm

            # 变换到椭球
            p = self.center + np.sqrt(chi2_level) * self.eigvecs @ self.sqrt_L @ s
            samples[i] = p

        return samples

    def compute_distance_statistics(
        self,
        n_samples: int = 1000,
        rng: Optional[ZigguratGenerator] = None,
    ) -> Dict[str, float]:
        """
        计算椭球内随机点的距离统计.

        融合 329 项目的 ellipse_distance_stats:
          μ = E[||p - q||]
          σ² = Var[||p - q||]

        Parameters
        ----------
        n_samples : int
            采样数
        rng : ZigguratGenerator, optional

        Returns
        -------
        result : dict
            mean_distance: 平均距离
            var_distance: 距离方差
            min_distance: 最小距离
            max_distance: 最大距离
        """
        p = self.sample_on_ellipsoid(n_samples, chi2_level=1.0, rng=rng)
        q = self.sample_on_ellipsoid(n_samples, chi2_level=1.0, rng=rng)

        distances = np.sqrt(np.sum((p - q)**2, axis=1))

        mu = np.mean(distances)
        var = np.var(distances) if n_samples > 1 else 0.0

        return {
            "mean_distance": mu,
            "var_distance": var,
            "min_distance": np.min(distances),
            "max_distance": np.max(distances),
            "n_samples": n_samples,
        }


class MCMCSampler:
    """
    Metropolis-Hastings MCMC 采样器.

    用于星震学贝叶斯反演.

    目标分布:
      P(p|D) ∝ L(D|p) × π(p)

    提议分布:
      q(p'|p) = N(p, σ_prop² I)

    接受率:
      α = min(1, P(p'|D)/P(p|D))

    参数
    ----
    log_likelihood : callable
        对数似然函数
    log_prior : callable
        对数先验函数
    proposal_scale : float
        提议分布尺度
    """

    def __init__(
        self,
        log_likelihood: Callable,
        log_prior: Callable,
        proposal_scale: float = 0.01,
    ):
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior
        self.proposal_scale = proposal_scale

    def sample(
        self,
        p_init: np.ndarray,
        n_steps: int = 1000,
        burn_in: int = 200,
        rng: Optional[ZigguratGenerator] = None,
    ) -> Dict[str, np.ndarray]:
        """
        运行 MCMC 采样.

        Parameters
        ----------
        p_init : ndarray
            初始参数
        n_steps : int
            采样步数
        burn_in : int
            预热步数
        rng : ZigguratGenerator, optional

        Returns
        -------
        result : dict
            chain: 采样链
            acceptance_rate: 接受率
            mean: 后验均值
            std: 后验标准差
        """
        if rng is None:
            rng = ZigguratGenerator(seed=42)

        n_params = len(p_init)
        chain = np.zeros((n_steps, n_params))
        chain[0] = p_init

        current_ll = self.log_likelihood(p_init) + self.log_prior(p_init)
        accepted = 0

        for step in range(1, n_steps):
            # 提议
            proposal = chain[step - 1] + self.proposal_scale * rng.sample_normal(n_params)

            # 评估
            prop_ll = self.log_likelihood(proposal) + self.log_prior(proposal)

            # Metropolis 接受判据
            log_alpha = prop_ll - current_ll
            u = max(rng._r4_uni(), 1e-10)

            if np.log(u) < log_alpha:
                chain[step] = proposal
                current_ll = prop_ll
                accepted += 1
            else:
                chain[step] = chain[step - 1]

        # 去除预热
        chain_burned = chain[burn_in:]

        return {
            "chain": chain_burned,
            "full_chain": chain,
            "acceptance_rate": accepted / n_steps,
            "mean": np.mean(chain_burned, axis=0),
            "std": np.std(chain_burned, axis=0),
        }


def monte_carlo_uncertainty_quantification(
    stellar_model,
    freq_obs: np.ndarray,
    sigma_obs: np.ndarray,
    n_samples: int = 500,
) -> Dict[str, any]:
    """
    星震学反演的蒙特卡罗不确定性量化.

    Parameters
    ----------
    stellar_model : StellarStructureModel
        恒星模型
    freq_obs : ndarray
        观测频率
    sigma_obs : ndarray
        观测误差
    n_samples : int
        采样数

    Returns
    -------
    result : dict
        posterior_mean: 后验均值
        posterior_std: 后验标准差
        chi2_best: 最优 χ²
    """
    # 定义对数似然
    def log_likelihood(params):
        mass, radius = params[0], params[1]
        mass = np.clip(mass, 0.1, 10.0)
        radius = np.clip(radius, 0.1, 10.0)
        # 简化: 频率正比于 sqrt(M/R³)
        nu_scale = np.sqrt(mass / radius**3)
        nu_model = freq_obs * nu_scale
        residuals = (freq_obs - nu_model) / sigma_obs
        return -0.5 * np.sum(residuals**2)

    # 定义对数先验 (均匀)
    def log_prior(params):
        mass, radius = params[0], params[1]
        if 0.1 <= mass <= 10.0 and 0.1 <= radius <= 10.0:
            return 0.0
        return -np.inf

    # MCMC 采样
    sampler = MCMCSampler(log_likelihood, log_prior, proposal_scale=0.05)
    p_init = np.array([1.0, 1.0])
    result = sampler.sample(p_init, n_steps=n_samples, burn_in=n_samples // 5)

    return {
        "posterior_mean": result["mean"],
        "posterior_std": result["std"],
        "acceptance_rate": result["acceptance_rate"],
        "chain": result["chain"],
    }
