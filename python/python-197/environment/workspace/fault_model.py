"""
fault_model.py
================================================================================
高性能计算检查点容错：故障到达统计模型与预测模块

融合原项目：
  - 035_asa091 (Gamma / Chi2 / Normal 分布函数)
  - 036_asa103 (Digamma 函数)

科学角色：
  1) 使用 Gamma 分布建模并行节点的故障间隔时间:
         f(t; alpha, beta) = beta^alpha / Gamma(alpha) * t^{alpha-1} * e^{-beta*t}
  2) 利用 Digamma 函数计算故障分布的熵与统计矩；
  3) 基于正态/Chi2 假设检验判断系统是否进入高故障率模式。
================================================================================
"""

import math
import numpy as np
from special_functions import gammad, ppchi2, ppnd, alnorm, digamma


class GammaFaultModel:
    """Gamma 分布故障到达模型。"""

    def __init__(self, alpha: float = 2.0, beta: float = 0.001):
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError("alpha and beta must be positive")
        self.alpha = alpha
        self.beta = beta

    def pdf(self, t: float) -> float:
        """概率密度函数。"""
        if t <= 0.0:
            return 0.0
        return (self.beta ** self.alpha / math.gamma(self.alpha)
                * t ** (self.alpha - 1.0) * math.exp(-self.beta * t))

    def cdf(self, t: float) -> float:
        """累积分布函数 P(T <= t)。"""
        if t <= 0.0:
            return 0.0
        x = self.beta * t
        val, _ = gammad(x, self.alpha)
        return val

    def survival(self, t: float) -> float:
        """生存函数 R(t) = P(T > t) = 1 - CDF(t)。"""
        return 1.0 - self.cdf(t)

    def hazard(self, t: float) -> float:
        """风险率函数 h(t) = f(t) / R(t)。"""
        s = self.survival(t)
        if s < 1.0e-14:
            return 0.0
        return self.pdf(t) / s

    def mean(self) -> float:
        """期望 E[T] = alpha / beta。"""
        return self.alpha / self.beta

    def variance(self) -> float:
        """方差 Var(T) = alpha / beta^2。"""
        return self.alpha / (self.beta ** 2)

    def entropy(self) -> float:
        """
        微分熵 H(T) = alpha - ln(beta) + ln(Gamma(alpha))
                     + (1-alpha) * psi(alpha)
        其中 psi 为 Digamma 函数。
        """
        psi_val, _ = digamma(self.alpha)
        return (self.alpha - math.log(self.beta) + math.lgamma(self.alpha)
                + (1.0 - self.alpha) * psi_val)

    def quantile(self, p: float) -> float:
        """逆 CDF: 求 t 使得 P(T <= t) = p。"""
        if p <= 0.0:
            return 0.0
        if p >= 1.0:
            return 1.0e10
        g = math.lgamma(self.alpha)
        chi2_val, _ = ppchi2(p, 2.0 * self.alpha, g)
        return chi2_val / (2.0 * self.beta)

    def sample(self, size: int = 1, seed: int = None) -> np.ndarray:
        """使用 numpy 生成 Gamma 随机样本。"""
        rng = np.random.default_rng(seed)
        return rng.gamma(shape=self.alpha, scale=1.0 / self.beta, size=size)


class FaultPredictor:
    """
    基于历史故障间隔的预测器，使用正态近似与 Chi2 方差检验
    判断系统是否进入高故障率状态。
    """

    def __init__(self, significance: float = 0.05):
        self.significance = significance
        self.history = []

    def observe(self, inter_arrival: float):
        """记录一次故障间隔时间。"""
        if inter_arrival > 0.0:
            self.history.append(inter_arrival)

    def test_increase(self) -> tuple:
        """
        对最近 k 个样本进行方差膨胀检验：
        若样本方差显著大于基准方差，则认为故障率上升。
        返回 (is_increased, p_value)。
        """
        if len(self.history) < 3:
            return False, 1.0
        arr = np.array(self.history[-20:])
        n = len(arr)
        mean_est = np.mean(arr)
        var_est = np.var(arr, ddof=1)
        if var_est <= 0.0:
            return False, 1.0
        # 构造检验统计量: (n-1)*s^2 / sigma0^2 ~ Chi2(n-1)
        # 取 sigma0 = mean_est (指数分布假设)
        sigma0_sq = mean_est ** 2
        if sigma0_sq <= 0.0:
            return False, 1.0
        chi2_stat = (n - 1) * var_est / sigma0_sq
        # p-value: P(Chi2 > stat)
        from special_functions import gammad
        cdf, _ = gammad(chi2_stat * 0.5, (n - 1) * 0.5)
        p_value = 1.0 - cdf
        is_increased = p_value < self.significance
        return is_increased, p_value

    def recommended_checkpoint_interval(self, safety_factor: float = 0.8) -> float:
        """
        基于故障历史推荐检查点间隔:
            T_ckpt = safety_factor * quantile(0.5)
        """
        if len(self.history) < 2:
            return 100.0
        arr = np.array(self.history)
        mean_est = np.mean(arr)
        var_est = np.var(arr, ddof=1)
        if var_est <= 0.0:
            return safety_factor * mean_est
        # 矩估计匹配 Gamma(alpha, beta)
        alpha_est = mean_est ** 2 / var_est
        beta_est = mean_est / var_est
        model = GammaFaultModel(alpha=max(alpha_est, 0.1), beta=max(beta_est, 1.0e-6))
        q = model.quantile(0.5)
        return safety_factor * q
