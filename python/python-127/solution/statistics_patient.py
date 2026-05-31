"""
statistics_patient.py
=====================
患者变异性统计建模模块

基于种子项目:
  - 055_asa310: 非中心 Beta 分布 CDF 计算

科学背景:
  人工耳蜗植入效果存在显著的个体差异。
  组织电导率、神经存活率、电极位置等参数服从特定统计分布。

  非中心 Beta 分布用于建模有偏的生理参数比例:
      X ~ Beta(α, β, λ)

  其中 λ 为非中心参数，反映某种系统性偏移（如疾病导致的神经退化）。

  概率密度:
      f(x; α, β, λ) = Σ_{j=0}^∞ exp(-λ/2) (λ/2)^j / j!
                        * x^{α+j-1} (1-x)^{β-1} / B(α+j, β)

  累积分布函数:
      F(x; α, β, λ) = Σ_{j=0}^∞ exp(-λ/2) (λ/2)^j / j! I_x(α+j, β)

  其中 I_x(a,b) 为正则化不完全 Beta 函数。
"""

import numpy as np
from scipy.special import betainc, gammaln, factorial
from scipy.stats import beta as beta_dist


def noncentral_beta_cdf(x, a, b, lam, errmax=1e-10, max_iter=1000):
    """
    计算非中心 Beta 累积分布函数。

    基于种子 055_asa310 的级数展开算法。

    Parameters
    ----------
    x : float
        0 < x < 1
    a, b : float
        形状参数，必须 > 0
    lam : float
        非中心参数 λ >= 0
    errmax : float
        精度容差
    max_iter : int
        最大迭代次数

    Returns
    -------
    cdf : float
        累积概率
    ifault : int
        错误码 (0=正常)
    """
    if a <= 0 or b <= 0:
        return 0.0, 3
    if lam < 0:
        return 0.0, 3
    if x <= 0.0:
        return 0.0, 0
    if x >= 1.0:
        return 1.0, 0

    # 当 λ 较小时，直接用 Poisson 加权和
    if lam < 54.0:
        c = 0.5 * lam
        cdf = 0.0
        j = 0
        while j < max_iter:
            pois_prob = np.exp(-c + j * np.log(c) - gammaln(j + 1.0))
            if pois_prob < errmax and j > c:
                break
            beta_cdf = betainc(a + j, b, x)
            cdf += pois_prob * beta_cdf
            j += 1
        return cdf, 0

    # λ 较大时，使用中心近似 + 修正
    m = int(np.floor(0.5 * lam + 0.5))
    c = 0.5 * lam
    iterlo = max(0, m - int(5.0 * np.sqrt(m)))
    iterhi = m + int(5.0 * np.sqrt(m))

    t = -c + m * np.log(c) - gammaln(m + 1.0)
    q = np.exp(t)
    r = q
    psum = q

    beta_ln = gammaln(a + m) + gammaln(b) - gammaln(a + m + b)
    s1 = (a + m) * np.log(x) + b * np.log(1.0 - x) - np.log(a + m) - beta_ln
    gx = np.exp(s1)
    fx = gx
    ftemp = betainc(a + m, b, x)
    sum_val = q * ftemp

    # 向下迭代
    iter1 = m
    while iter1 > iterlo and q > errmax:
        q = q * iter1 / c
        gx = (a + iter1) / (x * (a + b + iter1 - 1.0)) * gx
        iter1 -= 1
        temp = ftemp + gx
        psum += q
        sum_val += q * temp

    # 向上迭代
    q = r
    temp = ftemp
    gx = fx
    iter2 = m
    while iter2 < iterhi:
        ebd = (1.0 - psum) * temp
        if ebd < errmax:
            break
        iter2 += 1
        q = q * c / iter2
        psum += q
        temp = temp - gx
        gx = x * (a + b + iter2 - 1.0) / (a + iter2) * gx
        sum_val += q * temp

    return sum_val, 0


class PatientVariabilityModel:
    """
    患者生理参数变异性模型。
    """

    def __init__(self, sigma_mean=0.3, sigma_std=0.08,
                 survival_alpha=5.0, survival_beta=2.0, survival_lambda=2.0):
        """
        Parameters
        ----------
        sigma_mean : float
            电导率均值 (S/m)
        sigma_std : float
            电导率标准差
        survival_alpha, survival_beta : float
            神经存活率 Beta 分布参数
        survival_lambda : float
            神经存活率非中心参数
        """
        self.sigma_mean = float(sigma_mean)
        self.sigma_std = float(sigma_std)
        self.survival_alpha = float(survival_alpha)
        self.survival_beta = float(survival_beta)
        self.survival_lambda = float(survival_lambda)

    def sample_conductivity(self, n_samples=1):
        """
        采样组织电导率。

        使用截断正态分布保证正值:
            σ ~ N(μ, σ_0²)_{σ>0}
        """
        samples = np.random.randn(n_samples) * self.sigma_std + self.sigma_mean
        samples = np.maximum(samples, 0.05)
        return samples

    def sample_neural_survival_rate(self, n_samples=1):
        """
        采样神经存活率。

        使用非中心 Beta 分布反映疾病导致的系统性退化:
            p_survival ~ Beta(α, β, λ)
        """
        # 使用近似: 非中心 Beta = 中心 Beta(α + Poisson(λ/2)*2, β) 的混合
        # 这里简化为带有偏移的 Beta
        c = 0.5 * self.survival_lambda
        n = np.random.poisson(c, n_samples)
        alpha_eff = self.survival_alpha + n
        samples = beta_dist.rvs(alpha_eff, self.survival_beta)
        return np.clip(samples, 0.0, 1.0)

    def sample_electrode_offset(self, n_samples=1):
        """
        采样电极-蜗轴距离偏移。

        临床中 perimodiolar 电极距离约 0.3-0.8 mm，
        使用 Gamma 分布:
            offset ~ Gamma(k=4, θ=0.15)
        """
        k, theta = 4.0, 0.15
        samples = np.random.gamma(k, theta, n_samples)
        return np.clip(samples, 0.1, 2.0)

    def generate_patient_cohort(self, n_patients=100):
        """
        生成患者队列参数。

        Returns
        -------
        cohort : dict
            {'conductivity': array, 'survival_rate': array, 'offset': array}
        """
        return {
            'conductivity': self.sample_conductivity(n_patients),
            'survival_rate': self.sample_neural_survival_rate(n_patients),
            'offset': self.sample_electrode_offset(n_patients),
        }

    def probability_threshold_hearing(self, survival_rate, threshold=0.5):
        """
        计算患者达到特定听力阈值的概率。

        使用非中心 Beta CDF 评估:
            P(p > threshold) = 1 - F(threshold; α, β, λ)

        Parameters
        ----------
        survival_rate : float
            当前患者神经存活率
        threshold : float
            临床有效阈值

        Returns
        -------
        prob : float
        """
        cdf, ifault = noncentral_beta_cdf(
            survival_rate, self.survival_alpha, self.survival_beta,
            self.survival_lambda
        )
        if ifault != 0:
            return 0.0
        return 1.0 - cdf


def clinical_outcome_probability(survival_rate, stimulation_level,
                                  alpha=5.0, beta=2.0, lambda_nc=2.0):
    """
    计算临床预后概率。

    综合考虑神经存活率和刺激水平:
        P_success = P(survival > threshold | stimulation)

    Parameters
    ----------
    survival_rate : float
        实测或估计的神经存活率
    stimulation_level : float
        刺激水平 (归一化 0-1)
    alpha, beta, lambda_nc : float
        非中心 Beta 参数

    Returns
    -------
    prob : float
    """
    # 有效存活率随刺激水平提升
    effective_survival = survival_rate * (0.5 + 0.5 * stimulation_level)
    effective_survival = min(effective_survival, 0.999)

    cdf, ifault = noncentral_beta_cdf(
        effective_survival, alpha, beta, lambda_nc
    )
    if ifault != 0:
        return 0.0
    return cdf
