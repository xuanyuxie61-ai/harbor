"""
uncertainty_quantification.py
风电场性能不确定性量化

融合源项目：
- 581_image_noise: 图像噪声模型（风速测量不确定性传播）
- 1147_square_integrals: 单位正方形积分（尾流亏损在扫掠面上的积分平均）
- 032_asa066: 标准正态CDF（统计假设检验）
"""

import numpy as np
from typing import Tuple, List, Callable, Optional
from numerical_utils import alnorm, alnorm_array, weibull_pdf


class UncertaintyQuantification:
    """
    风电场性能的不确定性量化分析。

    物理模型：
    -----------
    年发电量 AEP 的不确定性来源：
        1. 风速测量误差：σ_u,meas
        2. 功率曲线不确定性：σ_P(u)
        3. 尾流模型参数不确定性：σ_k, σ_Ct
        4. 风向不确定性：σ_θ

    采用 Monte Carlo 方法传播不确定性：

        AEP^{(i)} = Σ_t Σ_j P_j(u_eff^{(i)}(t)) · Δt

    其中上标 (i) 表示第 i 次 Monte Carlo 采样。

    同时采用 Taylor 展开的一阶近似：

        σ_AEP² ≈ Σ_k (∂AEP/∂p_k)² · σ_{p_k}²

    其中 p_k 为不确定性参数。
    """

    def __init__(self, n_mc_samples: int = 1000, seed: Optional[int] = 42):
        """
        Parameters
        ----------
        n_mc_samples : int
            Monte Carlo 采样次数。
        seed : Optional[int]
            随机种子。
        """
        self.n_mc_samples = n_mc_samples
        self.rng = np.random.default_rng(seed)

    def propagate_speed_uncertainty(self, u_nominal: float,
                                     sigma_u: float,
                                     power_curve: Callable[[float], float]) -> Tuple[float, float]:
        """
        传播风速测量不确定性到功率输出。

        模型：
            u^{(i)} = u_nominal + ε^{(i)},  ε ~ N(0, σ_u²)

        输出：
            μ_P = E[P(u)]
            σ_P = √Var[P(u)]

        Parameters
        ----------
        u_nominal : float
            名义风速 [m/s]。
        sigma_u : float
            风速标准差 [m/s]。
        power_curve : Callable[[float], float]
            功率曲线函数 P(u) [MW]。

        Returns
        -------
        mean_power : float
            平均功率 [MW]。
        std_power : float
            功率标准差 [MW]。
        """
        samples = self.rng.normal(u_nominal, sigma_u, self.n_mc_samples)
        samples = np.maximum(samples, 0.0)  # 风速非负
        powers = np.array([power_curve(float(u)) for u in samples])
        return float(np.mean(powers)), float(np.std(powers))

    def aep_monte_carlo(self, wind_speed_distribution: Callable[[int], np.ndarray],
                        power_curve: Callable[[float], float],
                        wake_model_func: Callable[[np.ndarray], np.ndarray],
                        sigma_params: dict) -> Tuple[float, float, np.ndarray]:
        """
        Monte Carlo 估算年发电量的不确定性。

        Parameters
        ----------
        wind_speed_distribution : Callable[[int], np.ndarray]
            风速分布采样函数。
        power_curve : Callable[[float], float]
            功率曲线。
        wake_model_func : Callable[[np.ndarray], np.ndarray]
            尾流模型函数，输入风速数组，输出有效风速数组。
        sigma_params : dict
            不确定性参数字典，如 {'k_wake': 0.01, 'Ct': 0.05}。

        Returns
        -------
        mean_aep : float
            平均 AEP [MWh]。
        std_aep : float
            AEP 标准差 [MWh]。
        aep_samples : np.ndarray
            AEP 样本数组。
        """
        aep_samples = np.zeros(self.n_mc_samples)

        for i in range(self.n_mc_samples):
            # 采样风速
            u_samples = wind_speed_distribution(100)
            # 应用尾流
            u_eff = wake_model_func(u_samples)
            # 计算功率
            powers = np.array([power_curve(float(u)) for u in u_eff])
            # 简化的 AEP 估算
            hours = 8760.0
            aep_samples[i] = np.mean(powers) * hours

        return float(np.mean(aep_samples)), float(np.std(aep_samples)), aep_samples

    def confidence_interval(self, samples: np.ndarray,
                            confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        计算样本的置信区间。

        对于正态假设：
            CI = [μ - z_{1-α/2}·σ, μ + z_{1-α/2}·σ]

        其中 z_{1-α/2} 为标准正态分位数。

        Parameters
        ----------
        samples : np.ndarray
            样本数组。
        confidence : float
            置信水平。

        Returns
        -------
        mean : float
            样本均值。
        lower : float
            下限。
        upper : float
            上限。
        """
        alpha = 1.0 - confidence
        mean = float(np.mean(samples))
        std = float(np.std(samples, ddof=1))
        n = len(samples)

        # 使用正态近似（源自 032_asa066 的 alnorm）
        # z_{1-α/2}
        z = self._inverse_normal_cdf(1.0 - alpha / 2.0)
        margin = z * std / np.sqrt(n)
        return mean, mean - margin, mean + margin

    def _inverse_normal_cdf(self, p: float, tol: float = 1e-8) -> float:
        """
        用二分法求标准正态分位数 Φ^{-1}(p)。

        利用 alnorm 计算 CDF 值。
        """
        if p <= 0:
            return -10.0
        if p >= 1:
            return 10.0

        lo, hi = -10.0, 10.0
        while hi - lo > tol:
            mid = (lo + hi) / 2.0
            cdf_mid = 1.0 - alnorm(mid, upper=True)
            if cdf_mid < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def ks_test_normality(self, samples: np.ndarray,
                          n_bins: int = 20) -> Tuple[float, bool]:
        """
        Kolmogorov-Smirnov 正态性检验。

        检验统计量：
            D_n = sup_x |F_n(x) - Φ((x-μ)/σ)|

        其中 F_n 为经验 CDF，Φ 为标准正态 CDF（源自 032_asa066）。

        Parameters
        ----------
        samples : np.ndarray
            待检验样本。
        n_bins : int
            分箱数。

        Returns
        -------
        d_statistic : float
            KS 统计量。
        reject : bool
            是否拒绝正态性假设（α = 0.05）。
        """
        samples = np.asarray(samples, dtype=float)
        n = len(samples)
        if n < 3:
            return 0.0, False

        mu = np.mean(samples)
        sigma = np.std(samples, ddof=1)
        if sigma < 1e-14:
            return 0.0, False

        sorted_samples = np.sort(samples)
        empirical_cdf = np.arange(1, n + 1) / n
        z_scores = (sorted_samples - mu) / sigma
        theoretical_cdf = 1.0 - alnorm_array(z_scores, upper=True)

        d_statistic = float(np.max(np.abs(empirical_cdf - theoretical_cdf.ravel())))

        # 近似临界值（α = 0.05）：1.36 / sqrt(n)
        critical_value = 1.36 / np.sqrt(n)
        reject = d_statistic > critical_value
        return d_statistic, reject

    def sensitivity_analysis(self, base_aep: float,
                              param_names: List[str],
                              param_base: List[float],
                              param_sigma: List[float],
                              aep_func: Callable[[List[float]], float],
                              delta: float = 0.01) -> dict:
        """
        一阶灵敏度分析。

        计算各参数对 AEP 的偏导数：

            S_k = (∂AEP/∂p_k) · (σ_{p_k} / AEP)

        Parameters
        ----------
        base_aep : float
            基准 AEP。
        param_names : List[str]
            参数名称。
        param_base : List[float]
            参数基准值。
        param_sigma : List[float]
            参数不确定性。
        aep_func : Callable[[List[float]], float]
            AEP 计算函数。
        delta : float
            数值微分步长比例。

        Returns
        -------
        dict
            灵敏度分析结果。
        """
        results = {}
        n_params = len(param_names)

        for k in range(n_params):
            name = param_names[k]
            p_base = param_base[k]
            p_sigma = param_sigma[k]

            # 数值微分
            p_plus = param_base.copy()
            p_minus = param_base.copy()
            step = max(abs(p_base) * delta, 1e-6)
            p_plus[k] += step
            p_minus[k] -= step

            aep_plus = aep_func(p_plus)
            aep_minus = aep_func(p_minus)

            derivative = (aep_plus - aep_minus) / (2.0 * step)
            sensitivity = derivative * p_sigma / base_aep if base_aep != 0 else 0.0

            results[name] = {
                'derivative': derivative,
                'sensitivity_index': sensitivity,
                'uncertainty_contribution': abs(derivative * p_sigma)
            }

        return results

    def integral_deficit_uncertainty(self, deficit_func: Callable[[float], float],
                                      x_range: Tuple[float, float],
                                      sigma_x: float,
                                      n_points: int = 100) -> Tuple[float, float]:
        """
        尾流亏损积分的空间不确定性。

        融合 1147_square_integrals 的积分思想，计算尾流亏损在下游距离上的
        积分不确定性：

            I = ∫_{x1}^{x2} δ(x) dx

        当 x 存在不确定性 σ_x 时，通过 Monte Carlo 传播。

        Parameters
        ----------
        deficit_func : Callable[[float], float]
            亏损函数 δ(x)。
        x_range : Tuple[float, float]
            积分区间。
        sigma_x : float
            距离不确定性 [m]。
        n_points : int
            积分点数。

        Returns
        -------
        mean_integral : float
            平均积分值。
        std_integral : float
            积分标准差。
        """
        x1, x2 = x_range
        integral_samples = np.zeros(self.n_mc_samples)

        for i in range(self.n_mc_samples):
            # 扰动积分区间
            dx = self.rng.normal(0.0, sigma_x)
            x1_p = max(0.0, x1 + dx)
            x2_p = max(x1_p + 1.0, x2 + dx)

            x = np.linspace(x1_p, x2_p, n_points)
            dx = x[1] - x[0]
            deficits = np.array([deficit_func(float(xi)) for xi in x])
            # Simpson 积分
            integral_samples[i] = dx / 3.0 * (
                deficits[0] + deficits[-1] +
                4.0 * np.sum(deficits[1:-1:2]) +
                2.0 * np.sum(deficits[2:-1:2])
            )

        return float(np.mean(integral_samples)), float(np.std(integral_samples))
