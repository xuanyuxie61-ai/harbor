"""
calibration.py - 数据驱动可靠性模型校准

本模块实现从观测数据校准可靠性模型中的未知参数:

1. 最大似然估计 (MLE):
   给定观测数据 {y_i}, 求参数 theta 使得:
   L(theta) = prod_i f(y_i | theta)  最大

2. 贝叶斯信息准则 (BIC):
   BIC = -2*log(L_max) + k*log(n)
   用于模型选择 (分布拟合优度比较)。

3. 矩匹配:
   通过样本均值和方差匹配理论矩,
   获得分布参数的初始估计。

4. 核密度估计 (KDE):
   非参数密度估计, 用于无先验分布假设时的可靠性建模:
   f_hat(x) = (1/(n*h)) sum K((x - x_i)/h)

种子项目映射:
  1165_bioinfo_GEMS → 数据驱动的模型校准与优化
  855_pdflib        → 分布族的参数校验
  345_exm           → 统计估计的数值方法
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.special import gammaln, digamma, polygamma
from random_variables import std_normal_pdf, std_normal_cdf


class DistributionFitter:
    """分布拟合: 从数据估计概率分布参数

    支持:
    - Normal 分布
    - Lognormal 分布
    - Gamma 分布
    - Weibull 分布

    方法: MLE (最大似然估计) + BIC (模型选择)
    """

    @staticmethod
    def fit_gamma_mle(observations):
        """Gamma 分布 MLE

        对数似然: L = alpha*log(beta) - gammaln(alpha)
                    + (alpha-1)*sum(log(x_i)) - beta*sum(x_i)

        形状参数 alpha 的 MLE 方程:
        log(alpha) - digamma(alpha) = log(mean(x)) - mean(log(x))

        尺度参数: beta = alpha / mean(x)
        """
        obs = np.asarray(observations, dtype=float)
        obs = obs[obs > 0]
        n = len(obs)
        if n < 2:
            return 1.0, 1.0

        log_mean = np.mean(np.log(obs))
        mean_log = np.log(np.mean(obs))
        s = mean_log - log_mean

        if s < 1e-10:
            return 1.0, 1.0 / np.mean(obs)

        # 近似解 (Wald 近似)
        alpha_init = (3.0 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)

        # Newton 迭代: log(alpha) - digamma(alpha) = s
        alpha = alpha_init
        for _ in range(50):
            psi = digamma(alpha)
            psi1 = polygamma(1, alpha)  # trigamma
            f = np.log(alpha) - psi - s
            fp = 1.0 / alpha - psi1
            if abs(fp) < 1e-30:
                break
            alpha_new = alpha - f / fp
            if alpha_new <= 0:
                alpha_new = alpha * 0.5
            if abs(alpha_new - alpha) < 1e-10:
                alpha = alpha_new
                break
            alpha = alpha_new

        alpha = max(alpha, 0.01)
        beta = alpha / np.mean(obs)
        return float(alpha), float(beta)

    @staticmethod
    def fit_normal_mle(observations):
        """正态分布 MLE: mu_hat = mean, sigma_hat = std"""
        obs = np.asarray(observations, dtype=float)
        return float(np.mean(obs)), float(max(np.std(obs, ddof=0), 1e-10))

    @staticmethod
    def fit_lognormal_mle(observations):
        """对数正态分布 MLE

        令 y_i = log(x_i), 则:
        lambda_hat = mean(y_i)
        zeta_hat^2 = var(y_i)
        """
        obs = np.asarray(observations, dtype=float)
        obs = obs[obs > 0]
        if len(obs) < 2:
            return 0.0, 1.0
        log_obs = np.log(obs)
        lambda_hat = float(np.mean(log_obs))
        zeta_hat = float(max(np.std(log_obs, ddof=0), 1e-10))
        return lambda_hat, zeta_hat

    @staticmethod
    def fit_weibull_mle(observations, max_iter=100):
        """Weibull 分布 MLE (Profile Likelihood)

        对数似然: L = n*log(k) - n*k*log(lam)
                  + (k-1)*sum(log(x_i)) - sum((x_i/lam)^k)

        对 lam 的 profile: lam = (mean(x^k)/1)^{1/k}
        对 k 的方程: 1/k + mean(log(x)) - sum(x^k * log(x)) / sum(x^k) = 0
        """
        obs = np.asarray(observations, dtype=float)
        obs = obs[obs > 0]
        n = len(obs)
        if n < 2:
            return 1.0, 1.0

        log_obs = np.log(obs)
        mean_log = np.mean(log_obs)

        # Newton 迭代求 k
        k = 1.5  # 初始值
        for _ in range(max_iter):
            xk = obs ** k
            sum_xk = np.sum(xk)
            if sum_xk < 1e-30:
                break
            sum_xk_log = np.sum(xk * log_obs)

            g = 1.0 / k + mean_log - sum_xk_log / sum_xk
            # 数值微分求 g'
            dk = 1e-6
            xk2 = obs ** (k + dk)
            sum_xk2 = np.sum(xk2)
            if sum_xk2 < 1e-30:
                break
            sum_xk2_log = np.sum(xk2 * log_obs)
            g2 = 1.0 / (k + dk) + mean_log - sum_xk2_log / sum_xk2
            gp = (g2 - g) / dk

            if abs(gp) < 1e-30:
                break
            k_new = k - g / gp
            if k_new <= 0:
                k_new = k * 0.5
            if abs(k_new - k) < 1e-8:
                k = k_new
                break
            k = k_new

        k = max(k, 0.01)
        lam = (np.mean(obs ** k)) ** (1.0 / k)
        return float(k), float(max(lam, 1e-10))

    @staticmethod
    def log_likelihood_normal(obs, mu, sigma):
        n = len(obs)
        return (-n * np.log(sigma) - 0.5 * n * np.log(2 * np.pi)
                - np.sum((obs - mu) ** 2) / (2 * sigma ** 2))

    @staticmethod
    def log_likelihood_gamma(obs, alpha, beta):
        obs = obs[obs > 0]
        n = len(obs)
        return (n * alpha * np.log(beta) - n * gammaln(alpha)
                + (alpha - 1) * np.sum(np.log(obs)) - beta * np.sum(obs))

    @staticmethod
    def log_likelihood_weibull(obs, k, lam):
        obs = obs[obs > 0]
        n = len(obs)
        if k <= 0 or lam <= 0:
            return -np.inf
        return (n * np.log(k) - n * k * np.log(lam)
                + (k - 1) * np.sum(np.log(obs))
                - np.sum((obs / lam) ** k))


class ModelSelection:
    """基于 BIC 的分布模型选择

    BIC = -2 * log_L_max + k * log(n)
    选择 BIC 最小的模型。

    k = 模型参数个数, n = 样本量。
    """

    @staticmethod
    def compare_distributions(observations):
        """比较 Normal / Lognormal / Gamma / Weibull 的拟合优度"""
        obs = np.asarray(observations, dtype=float)
        obs_pos = obs[obs > 0]
        n = len(obs)
        n_pos = len(obs_pos)

        results = {}

        # Normal
        mu, sigma = DistributionFitter.fit_normal_mle(obs)
        ll = DistributionFitter.log_likelihood_normal(obs, mu, sigma)
        bic = -2 * ll + 2 * np.log(max(n, 1))
        results['Normal'] = {'params': (mu, sigma), 'll': ll, 'bic': bic, 'k': 2}

        # Lognormal (仅对正观测)
        if n_pos >= 3:
            lam, zeta = DistributionFitter.fit_lognormal_mle(obs_pos)
            # 近似 LL (在 log 空间)
            log_obs = np.log(obs_pos)
            ll = (-n_pos * np.log(zeta) - 0.5 * n_pos * np.log(2 * np.pi)
                  - np.sum((log_obs - lam) ** 2) / (2 * zeta ** 2)
                  - np.sum(log_obs))  # Jacobian
            bic = -2 * ll + 2 * np.log(n_pos)
            results['Lognormal'] = {'params': (lam, zeta), 'll': ll, 'bic': bic, 'k': 2}

        # Gamma
        if n_pos >= 3:
            alpha, beta = DistributionFitter.fit_gamma_mle(obs_pos)
            ll = DistributionFitter.log_likelihood_gamma(obs_pos, alpha, beta)
            bic = -2 * ll + 2 * np.log(n_pos)
            results['Gamma'] = {'params': (alpha, beta), 'll': ll, 'bic': bic, 'k': 2}

        # Weibull
        if n_pos >= 3:
            k_w, lam_w = DistributionFitter.fit_weibull_mle(obs_pos)
            ll = DistributionFitter.log_likelihood_weibull(obs_pos, k_w, lam_w)
            bic = -2 * ll + 2 * np.log(n_pos)
            results['Weibull'] = {'params': (k_w, lam_w), 'll': ll, 'bic': bic, 'k': 2}

        # 选择最佳
        best_name = min(results, key=lambda k: results[k]['bic'])
        return results, best_name


class KernelDensityReliability:
    """核密度估计用于非参数可靠性建模

    当没有先验分布假设时, 使用 KDE 估计随机变量的密度:
    f_hat(x) = (1/(n*h)) sum K((x - x_i) / h)

    其中 K 为 Gaussian 核, h 为带宽 (Silverman 规则):
    h = 0.9 * min(sigma, IQR/1.34) * n^{-1/5}
    """

    def __init__(self, data, bandwidth=None):
        self.data = np.asarray(data, dtype=float)
        self.n = len(self.data)
        if bandwidth is None:
            sigma = np.std(self.data, ddof=1)
            iqr = np.percentile(self.data, 75) - np.percentile(self.data, 25)
            scale = min(sigma, iqr / 1.34) if iqr > 0 else sigma
            self.h = 0.9 * max(scale, 1e-10) * self.n ** (-0.2)
        else:
            self.h = float(bandwidth)

    def pdf(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        result = np.zeros_like(x)
        for xi in self.data:
            result += std_normal_pdf((x - xi) / self.h) / self.h
        return result / self.n

    def cdf(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        result = np.zeros_like(x)
        for xi in self.data:
            result += std_normal_cdf((x - xi) / self.h)
        return result / self.n

    def sample(self, n, rng=None):
        rng = rng or np.random.default_rng()
        indices = rng.integers(0, self.n, size=n)
        return self.data[indices] + self.h * rng.standard_normal(n)
