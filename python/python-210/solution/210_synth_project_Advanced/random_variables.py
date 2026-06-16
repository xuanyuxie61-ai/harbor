"""
random_variables.py - 概率分布引擎：可靠性分析中的随机变量建模

本模块实现不确定性量化中常用的边缘概率分布，为结构可靠性分析提供
随机输入变量的完整描述。支持从物理空间到标准正态空间的等概率变换
(X-space → U-space)，这是FORM/SORM方法的数学基础。

核心公式：
  等概率变换:  u = Phi^{-1}(F_X(x))
  逆变换:      x = F_X^{-1}(Phi(u))
  其中 Phi 为标准正态CDF, F_X 为随机变量X的边缘CDF

种子项目映射:
  855_pdflib, 918_prob → 本模块（PDF/CDF/采样/变换）
"""

import numpy as np
from scipy.special import gammaln, erfc, erfcinv
from scipy.stats import norm as sp_norm


def std_normal_pdf(x):
    """标准正态概率密度函数 (Standard Normal PDF)

    phi(x) = (1/sqrt(2*pi)) * exp(-x^2 / 2)
    """
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / np.sqrt(2.0 * np.pi)


def std_normal_cdf(x):
    """标准正态累积分布函数 (Standard Normal CDF)

    Phi(x) = 0.5 * erfc(-x / sqrt(2))
    """
    return 0.5 * erfc(-np.asarray(x, dtype=float) / np.sqrt(2.0))


def std_normal_ppf(p):
    """标准正态逆CDF (分位函数, Quantile Function)

    Phi^{-1}(p) = -sqrt(2) * erfcinv(2p)
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1.0 - 1e-15)
    return -np.sqrt(2.0) * erfcinv(2.0 * p)


def _safe_std(u):
    """将输入安全转换为 float 数组，标量→1D."""
    return np.atleast_1d(np.asarray(u, dtype=float))


class NormalRV:
    """正态分布 N(mu, sigma)

    PDF:  f(x) = (1/(sigma*sqrt(2*pi))) * exp(-(x-mu)^2 / (2*sigma^2))
    CDF:  F(x) = Phi((x - mu) / sigma)
    采样: X = mu + sigma * Z,  Z ~ N(0,1)
    """

    def __init__(self, mu=0.0, sigma=1.0, name='X'):
        if sigma <= 0:
            raise ValueError(f"sigma 必须为正, 输入 sigma={sigma}")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.name = name
        self._dist = sp_norm(loc=mu, scale=sigma)

    def pdf(self, x):
        return self._dist.pdf(x)

    def cdf(self, x):
        return self._dist.cdf(x)

    def ppf(self, p):
        return self._dist.ppf(np.clip(p, 1e-15, 1.0 - 1e-15))

    def sample(self, n, rng=None):
        rng = rng or np.random.default_rng()
        return rng.normal(self.mu, self.sigma, size=n)

    def to_std_normal(self, x):
        """物理空间 → 标准正态空间:  u = (x - mu) / sigma"""
        return (np.asarray(x) - self.mu) / self.sigma

    def from_std_normal(self, u):
        """标准正态空间 → 物理空间:  x = mu + sigma * u"""
        return self.mu + self.sigma * np.asarray(u)

    def mean(self):
        return self.mu

    def std(self):
        return self.sigma


class LognormalRV:
    """对数正态分布 X = exp(Y), Y ~ N(lambda_zeta, zeta)

    PDF:  f(x) = (1/(x*zeta*sqrt(2*pi))) * exp(-(ln(x)-lambda_zeta)^2 / (2*zeta^2)),  x > 0
    CDF:  F(x) = Phi((ln(x) - lambda_zeta) / zeta)

    参数化:  mu_x = exp(lambda_zeta + zeta^2/2)
             sigma_x^2 = mu_x^2 * (exp(zeta^2) - 1)
    矩→参数: zeta^2 = ln(1 + (sigma_x/mu_x)^2)
             lambda_zeta = ln(mu_x) - zeta^2/2
    """

    def __init__(self, mu_x=1.0, sigma_x=0.3, name='X'):
        if mu_x <= 0:
            raise ValueError(f"mu_x 必须为正, 输入 mu_x={mu_x}")
        if sigma_x <= 0:
            raise ValueError(f"sigma_x 必须为正, 输入 sigma_x={sigma_x}")
        self.mu_x = float(mu_x)
        self.sigma_x = float(sigma_x)
        self.name = name
        cov = sigma_x / mu_x
        self.zeta = np.sqrt(np.log(1.0 + cov ** 2))
        self.lambda_zeta = np.log(mu_x) - 0.5 * self.zeta ** 2

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        mask = x > 0
        if np.any(mask):
            xm = x[mask] if x.ndim > 0 else np.array([x])
            lx = np.log(xm)
            z = (lx - self.lambda_zeta) / self.zeta
            out[mask] = std_normal_pdf(z) / (xm * self.zeta)
        return out

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        mask = x > 0
        if np.any(mask):
            xm = x[mask] if x.ndim > 0 else np.array([x])
            z = (np.log(xm) - self.lambda_zeta) / self.zeta
            out[mask] = std_normal_cdf(z)
        return out

    def ppf(self, p):
        p = np.clip(np.asarray(p, dtype=float), 1e-15, 1.0 - 1e-15)
        return np.exp(self.lambda_zeta + self.zeta * std_normal_ppf(p))

    def sample(self, n, rng=None):
        rng = rng or np.random.default_rng()
        return np.exp(rng.normal(self.lambda_zeta, self.zeta, size=n))

    def to_std_normal(self, x):
        """等概率变换:  u = Phi^{-1}(F_X(x))"""
        x = np.asarray(x, dtype=float)
        return std_normal_ppf(np.clip(self.cdf(x), 1e-15, 1.0 - 1e-15))

    def from_std_normal(self, u):
        """逆变换:  x = F_X^{-1}(Phi(u))"""
        return self.ppf(std_normal_cdf(np.asarray(u, dtype=float)))

    def mean(self):
        return self.mu_x

    def std(self):
        return self.sigma_x


class GammaRV:
    """Gamma分布 Gamma(alpha, beta)

    PDF:  f(x) = beta^alpha / Gamma(alpha) * x^(alpha-1) * exp(-beta*x),  x > 0
    均值: E[X] = alpha / beta
    方差: Var[X] = alpha / beta^2
    """

    def __init__(self, alpha=2.0, beta=1.0, name='X'):
        if alpha <= 0:
            raise ValueError(f"alpha 必须为正, 输入 alpha={alpha}")
        if beta <= 0:
            raise ValueError(f"beta 必须为正, 输入 beta={beta}")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.name = name
        self._mu = alpha / beta
        self._sigma = np.sqrt(alpha) / beta

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        mask = x > 0
        if np.any(mask):
            xm = x[mask] if x.ndim > 0 else np.array([x])
            log_p = (self.alpha * np.log(self.beta)
                     + (self.alpha - 1.0) * np.log(xm)
                     - self.beta * xm
                     - gammaln(self.alpha))
            out[mask] = np.exp(log_p)
        return out

    def cdf(self, x):
        from scipy.stats import gamma as sp_gamma
        return sp_gamma.cdf(np.maximum(x, 0), a=self.alpha, scale=1.0 / self.beta)

    def ppf(self, p):
        from scipy.stats import gamma as sp_gamma
        p = np.clip(np.asarray(p, dtype=float), 1e-15, 1.0 - 1e-15)
        return sp_gamma.ppf(p, a=self.alpha, scale=1.0 / self.beta)

    def sample(self, n, rng=None):
        rng = rng or np.random.default_rng()
        return rng.gamma(self.alpha, 1.0 / self.beta, size=n)

    def to_std_normal(self, x):
        x = np.asarray(x, dtype=float)
        return std_normal_ppf(np.clip(self.cdf(x), 1e-15, 1.0 - 1e-15))

    def from_std_normal(self, u):
        return self.ppf(std_normal_cdf(np.asarray(u, dtype=float)))

    def mean(self):
        return self._mu

    def std(self):
        return self._sigma


class WeibullRV:
    """Weibull分布 (结构可靠性疲劳寿命常用)

    PDF:  f(x) = (k/lambda) * (x/lambda)^(k-1) * exp(-(x/lambda)^k),  x >= 0
    CDF:  F(x) = 1 - exp(-(x/lambda)^k)
    均值: E[X] = lambda * Gamma(1 + 1/k)
    """

    def __init__(self, k=2.0, lam=1.0, name='X'):
        if k <= 0:
            raise ValueError(f"k 必须为正, 输入 k={k}")
        if lam <= 0:
            raise ValueError(f"lambda 必须为正, 输入 lam={lam}")
        self.k = float(k)
        self.lam = float(lam)
        self.name = name
        self._mu = lam * np.exp(gammaln(1.0 + 1.0 / k))
        self._sigma = lam * np.sqrt(
            np.exp(gammaln(1.0 + 2.0 / k)) - np.exp(gammaln(1.0 + 1.0 / k)) ** 2
        )

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        mask = x > 0
        if np.any(mask):
            xm = x[mask] if x.ndim > 0 else np.array([x])
            z = xm / self.lam
            log_p = (np.log(self.k / self.lam)
                     + (self.k - 1.0) * np.log(z)
                     - z ** self.k)
            out[mask] = np.exp(log_p)
        return out

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        mask = x > 0
        if np.any(mask):
            xm = x[mask] if x.ndim > 0 else np.array([x])
            out[mask] = 1.0 - np.exp(-(xm / self.lam) ** self.k)
        return out

    def ppf(self, p):
        p = np.clip(np.asarray(p, dtype=float), 1e-15, 1.0 - 1e-15)
        return self.lam * (-np.log(1.0 - p)) ** (1.0 / self.k)

    def sample(self, n, rng=None):
        rng = rng or np.random.default_rng()
        return self.lam * (-np.log(rng.uniform(size=n))) ** (1.0 / self.k)

    def to_std_normal(self, x):
        x = np.asarray(x, dtype=float)
        return std_normal_ppf(np.clip(self.cdf(x), 1e-15, 1.0 - 1e-15))

    def from_std_normal(self, u):
        return self.ppf(std_normal_cdf(np.asarray(u, dtype=float)))

    def mean(self):
        return self._mu

    def std(self):
        return self._sigma
