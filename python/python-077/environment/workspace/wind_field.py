"""
wind_field.py
风资源评估与风场模型

融合源项目：
- 581_image_noise: 图像噪声（风速测量数据的不确定性量化，模拟传感器噪声）
- 032_asa066: 正态CDF（Weibull分布与正态分布的统计检验）
"""

import numpy as np
from typing import Tuple, Optional
from numerical_utils import weibull_pdf, weibull_cdf, alnorm_array


class WindField:
    """
    风资源场模型。

    物理模型：
    -----------
    风速服从 Weibull 分布：

        f(u) = (k/A)·(u/A)^{k-1}·exp(-(u/A)^k),  u ≥ 0

    其中 A 为尺度参数 [m/s]，k 为形状参数（通常 1.5~3.0）。

    平均风速：
        ū = A · Γ(1 + 1/k)

    风向服从 von Mises 分布（周期性）：

        g(θ) = exp(κ·cos(θ - μ_θ)) / (2π·I_0(κ))

    其中 I_0(κ) 为修正贝塞尔函数，κ 为浓度参数，μ_θ 为平均风向。

    湍流强度：
        I = σ_u / ū

    其中 σ_u 为风速标准差。
    """

    def __init__(self, A: float = 10.0, k: float = 2.0,
                 mu_theta: float = 270.0, kappa: float = 2.0,
                 turbulence_intensity: float = 0.12):
        """
        Parameters
        ----------
        A : float
            Weibull 尺度参数 [m/s]。
        k : float
            Weibull 形状参数。
        mu_theta : float
            平均风向 [度]。
        kappa : float
            von Mises 浓度参数。
        turbulence_intensity : float
            湍流强度，典型值 0.08~0.20。
        """
        if A <= 0:
            raise ValueError("Weibull 尺度参数 A 必须为正")
        if k <= 0:
            raise ValueError("Weibull 形状参数 k 必须为正")
        if turbulence_intensity < 0:
            raise ValueError("湍流强度不能为负")

        self.A = A
        self.k = k
        self.mu_theta = mu_theta % 360.0
        self.kappa = kappa
        self.I = turbulence_intensity

    def mean_wind_speed(self) -> float:
        """
        计算 Weibull 分布的均值：

            ū = A · Γ(1 + 1/k)
        """
        from math import gamma
        return self.A * gamma(1.0 + 1.0 / self.k)

    def std_wind_speed(self) -> float:
        """
        计算 Weibull 分布的标准差：

            σ = A · √[Γ(1 + 2/k) - Γ²(1 + 1/k)]
        """
        from math import gamma
        var = self.A**2 * (gamma(1.0 + 2.0 / self.k) - gamma(1.0 + 1.0 / self.k)**2)
        return np.sqrt(max(0.0, var))

    def sample_speed(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """
        从 Weibull 分布中采样 n 个风速值。

        逆变换采样：
            u = A · (-ln(1 - ξ))^{1/k},  ξ ~ U(0,1)
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        xi = rng.random(n)
        return self.A * (-np.log(1.0 - xi)) ** (1.0 / self.k)

    def sample_direction(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """
        从 von Mises 分布中采样 n 个风向值 [度]。

        采用拒绝采样法。
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        mu_rad = np.radians(self.mu_theta)
        samples = []
        batch = n * 10
        while len(samples) < n:
            theta_prop = rng.uniform(0, 2 * np.pi, batch)
            u = rng.uniform(0, 1, batch)
            # 提案分布为均匀分布，接受概率正比于 exp(κ·cos(θ-μ))
            accept_prob = np.exp(self.kappa * (np.cos(theta_prop - mu_rad) - 1.0))
            accepted = theta_prop[u < accept_prob]
            samples.extend(accepted)
        directions = np.degrees(np.array(samples[:n])) % 360.0
        return directions

    def add_measurement_noise(self, u: np.ndarray, noise_level: float = 0.05,
                               noise_type: str = 'uniform',
                               seed: Optional[int] = None) -> np.ndarray:
        """
        给风速测量数据添加噪声，模拟传感器测量误差。

        融合 581_image_noise 的思想，将噪声模型应用于风速数据：

        均匀噪声：
            u_noisy = u + ε,  ε ~ U(-level·ū, +level·ū)

        高斯噪声：
            u_noisy = u + ε,  ε ~ N(0, (level·ū)²)

        Parameters
        ----------
        u : np.ndarray
            原始风速数据 [m/s]。
        noise_level : float
            噪声水平，相对于平均风速的比例。
        noise_type : str
            'uniform' 或 'gaussian'。
        seed : Optional[int]
            随机种子。

        Returns
        -------
        np.ndarray
            加噪后的风速数据。
        """
        u = np.asarray(u, dtype=float)
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        u_mean = self.mean_wind_speed()
        scale = noise_level * u_mean

        if noise_type == 'uniform':
            noise = rng.uniform(-scale, scale, size=u.shape)
        elif noise_type == 'gaussian':
            noise = rng.normal(0.0, scale, size=u.shape)
        else:
            raise ValueError("noise_type 必须是 'uniform' 或 'gaussian'")

        u_noisy = u + noise
        # 边界处理：风速不能为负
        u_noisy = np.maximum(u_noisy, 0.0)
        return u_noisy

    def weibull_to_normal_test(self, n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Weibull 风速分布的正态近似检验。

        将 Weibull 分布分箱后，用标准正态 CDF (asa066) 检验累积概率。

        采用 z-score 转换：
            z = (u - ū) / σ

        然后比较 Weibull CDF 与标准正态 CDF 的差异。

        Returns
        -------
        u_bins : np.ndarray
            分箱中心风速。
        weibull_cdf_vals : np.ndarray
            Weibull CDF 值。
        max_diff : float
            最大差异（Kolmogorov-Smirnov 统计量）。
        """
        u_max = self.A * 3.0  # 取 3 倍尺度参数
        u_bins = np.linspace(0, u_max, n_bins)
        weibull_cdf_vals = weibull_cdf(u_bins, self.A, self.k)

        u_mean = self.mean_wind_speed()
        u_std = self.std_wind_speed()
        if u_std < 1e-10:
            return u_bins, weibull_cdf_vals, 0.0

        z_scores = (u_bins - u_mean) / u_std
        normal_cdf_vals = 1.0 - alnorm_array(z_scores, upper=True)

        diff = np.abs(weibull_cdf_vals - normal_cdf_vals)
        max_diff = float(np.max(diff))
        return u_bins, weibull_cdf_vals, max_diff

    def annual_energy_density(self, u_cut_in: float = 3.0,
                               u_rated: float = 12.0,
                               u_cut_out: float = 25.0) -> float:
        """
        计算单位面积的年风能密度 [kWh/m²]。

        风功率密度：
            P_wind(u) = 0.5 · ρ · u³   [W/m²]

        年风能密度：
            E = 8760 · ∫_{u_cut_in}^{u_cut_out} P_wind(u)·f(u) du

        Parameters
        ----------
        u_cut_in, u_rated, u_cut_out : float
            切入、额定、切出风速 [m/s]。

        Returns
        -------
        float
            年风能密度 [kWh/m²]。
        """
        rho = 1.225  # 空气密度 [kg/m³]
        hours_per_year = 8760.0

        # 数值积分（Simpson 法则）
        n = 1000
        u = np.linspace(max(0.0, u_cut_in), u_cut_out, n)
        f = weibull_pdf(u, self.A, self.k)
        p = 0.5 * rho * u**3 / 1000.0  # [kW/m²]

        integrand = p * f
        # Simpson 积分
        du = u[1] - u[0]
        integral = du / 3.0 * (integrand[0] + integrand[-1] +
                               4.0 * np.sum(integrand[1:-1:2]) +
                               2.0 * np.sum(integrand[2:-1:2]))
        return hours_per_year * integral
