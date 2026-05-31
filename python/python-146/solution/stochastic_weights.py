"""
stochastic_weights.py
突触权重随机模型模块

融合 log_normal (对数正态分布采样与统计)。

核心科学模型：
  突触权重分布建模：
    实验观测表明，皮层突触权重常呈对数正态分布:
      w ~ LogNormal(mu, sigma^2)

    概率密度函数 (PDF):
      f(w) = 1 / (w * sigma * sqrt(2*pi)) * exp( -(ln w - mu)^2 / (2 sigma^2) ),  w > 0

    累积分布函数 (CDF):
      F(w) = Phi( (ln w - mu) / sigma )
      其中 Phi 为标准正态 CDF。

    均值与方差：
      E[w] = exp( mu + sigma^2 / 2 )
      Var[w] = [exp(sigma^2) - 1] * exp(2*mu + sigma^2)

  突触可塑性的随机微分方程 (SDE)：
    考虑噪声的权重演化:
      dw/dt = eta(w) + sigma_w(w) * xi(t)
    其中 xi(t) 为白噪声，<xi(t) xi(t')> = delta(t-t')。

    对数坐标下的 Ornstein-Uhlenbeck 近似:
      d(ln w) = -theta (ln w - mu) dt + sigma dW_t
    稳态分布即为对数正态分布。

  权重归一化与稳定性：
    总突触权重约束 (synaptic homeostasis):
      sum_j w_{ij} = W_target
    通过乘法归一化:
      w_{ij} <- w_{ij} * (W_target / sum_j w_{ij})
"""

import numpy as np


class LogNormalSynapse:
    """
    对数正态突触权重模型，融合 log_normal_sample / log_normal_pdf 等思想。
    """

    def __init__(self, mu, sigma):
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.mu = mu
        self.sigma = sigma

    def pdf(self, w):
        """
        对数正态 PDF。
        w: 权重值，必须为正数。
        """
        w = np.atleast_1d(w)
        pdf_vals = np.zeros_like(w, dtype=float)
        mask = w > 0
        pdf_vals[mask] = (
            1.0 / (w[mask] * self.sigma * np.sqrt(2.0 * np.pi))
            * np.exp(-((np.log(w[mask]) - self.mu) ** 2) / (2.0 * self.sigma ** 2))
        )
        return pdf_vals

    def cdf(self, w):
        """
        对数正态 CDF，使用标准正态 CDF 近似。
        """
        w = np.atleast_1d(w)
        z = (np.log(w) - self.mu) / self.sigma
        # 标准正态 CDF 近似 (误差函数)
        cdf_vals = 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
        cdf_vals = np.where(w > 0, cdf_vals, 0.0)
        return cdf_vals

    def inv_cdf(self, p):
        """
        对数正态逆 CDF (分位函数)。
        p in (0, 1)
        """
        p = np.atleast_1d(p)
        if np.any((p <= 0) | (p >= 1)):
            raise ValueError("p must be in (0, 1).")
        # 标准正态逆 CDF
        z = np.sqrt(2.0) * np.erfcinv(2.0 * (1.0 - p))
        w = np.exp(self.mu + self.sigma * z)
        return w

    def sample(self, size=None):
        """
        采样对数正态分布。
        """
        # 先生成正态分布样本，再取指数
        z = np.random.normal(loc=self.mu, scale=self.sigma, size=size)
        return np.exp(z)

    def mean(self):
        """理论均值。"""
        return np.exp(self.mu + 0.5 * self.sigma ** 2)

    def variance(self):
        """理论方差。"""
        return (np.exp(self.sigma ** 2) - 1.0) * np.exp(2.0 * self.mu + self.sigma ** 2)

    def sample_mean_variance(self, n_samples=10000):
        """
        蒙特卡洛估计样本均值和方差。
        融合 log_normal_sample + r8vec_mean / r8vec_variance 思想。
        """
        samples = self.sample(size=n_samples)
        return np.mean(samples), np.var(samples, ddof=1)


class SynapticWeightSDE:
    """
    突触权重的随机微分方程演化。
    d(ln w) = -theta (ln w - mu) dt + sigma dW_t
    """

    def __init__(self, mu=0.0, sigma=0.5, theta=0.1, dt=0.01):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt

    def euler_maruyama_step(self, w_current):
        """
        Euler-Maruyama 单步推进。
        """
        if w_current <= 0:
            w_current = 1e-6
        ln_w = np.log(w_current)
        dW = np.random.normal(0.0, np.sqrt(self.dt))
        ln_w_new = ln_w - self.theta * (ln_w - self.mu) * self.dt + self.sigma * dW
        w_new = np.exp(ln_w_new)
        return max(w_new, 1e-6)

    def simulate_trajectory(self, w0, T_total):
        """模拟权重轨迹。"""
        n_steps = int(np.ceil(T_total / self.dt))
        trajectory = np.zeros(n_steps)
        w = max(w0, 1e-6)
        for k in range(n_steps):
            w = self.euler_maruyama_step(w)
            trajectory[k] = w
        return trajectory

    def steady_state_check(self, samples):
        """
        检验样本是否符合对数正态稳态分布。
        返回 Kolmogorov-Smirnov 统计量的简化版本。
        """
        ln_samples = np.log(samples[samples > 0])
        emp_mean = np.mean(ln_samples)
        emp_std = np.std(ln_samples)
        # 理论参数 (近似)
        return abs(emp_mean - self.mu), abs(emp_std - self.sigma)


def normalize_weights_multiplicative(weights, target_sum=1.0):
    """
    乘法归一化突触权重 (稳态可塑性)。
    w_i <- w_i * (target_sum / sum_j w_j)
    """
    weights = np.asarray(weights, dtype=float)
    current_sum = np.sum(weights)
    if current_sum <= 0:
        return weights
    return weights * (target_sum / current_sum)


def demo_log_normal_weights():
    """对数正态权重 demo。"""
    model = LogNormalSynapse(mu=-0.5, sigma=0.8)
    samples = model.sample(size=5000)
    theoretical_mean = model.mean()
    theoretical_var = model.variance()
    empirical_mean, empirical_var = model.sample_mean_variance(n_samples=5000)
    return {
        'theoretical_mean': theoretical_mean,
        'theoretical_var': theoretical_var,
        'empirical_mean': empirical_mean,
        'empirical_var': empirical_var,
        'samples': samples
    }


def demo_weight_sde():
    """权重 SDE 演化 demo。"""
    sde = SynapticWeightSDE(mu=0.0, sigma=0.3, theta=0.05, dt=0.01)
    traj = sde.simulate_trajectory(w0=1.0, T_total=50.0)
    return traj
