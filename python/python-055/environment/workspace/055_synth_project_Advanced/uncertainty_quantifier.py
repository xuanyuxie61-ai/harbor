"""
uncertainty_quantifier.py
基于种子项目 805_nintlib（多维蒙特卡洛积分）与
1361_truncated_normal_rule（截断正态分布求积），
构建海底地形反演的不确定性量化与误差传播分析模块。

科学背景：声纳测深受多种误差源影响：
    - 声速剖面不确定性 Δc(z)
    - 波束指向角误差 Δθ
    - 传播时间测量噪声 Δt
    - 船舶姿态（横摇、纵摇、升沉）扰动

根据误差传播定律（Gauss 误差传播公式），
若深度反演模型为 z = f(c, θ, t)，则深度方差为：

    σ_z² ≈ (∂f/∂c)² σ_c² + (∂f/∂θ)² σ_θ² + (∂f/∂t)² σ_t²
          + 2 (∂f/∂c)(∂f/∂θ) ρ_{cθ} σ_c σ_θ + ...

对于非线性模型，采用蒙特卡洛方法直接估计输出分布：
    1. 从输入误差分布中随机采样 N 组 (c_i, θ_i, t_i)；
    2. 计算对应的深度 z_i = f(c_i, θ_i, t_i)；
    3. 统计 {z_i} 的均值、方差、置信区间。

截断正态分布用于描述有界物理量的测量误差，例如：
    - 声速测量误差限制在仪器标称精度范围内；
    - 角度误差受波束宽度物理约束。
"""

import numpy as np
import math
from scipy.special import erf


class TruncatedNormalDistribution:
    """
    截断正态分布（源自 truncated_normal_rule.m 的分布函数）。

    概率密度函数:
        f(x; μ, σ, a, b) = φ((x-μ)/σ) / [σ · (Φ((b-μ)/σ) - Φ((a-μ)/σ))]
    其中 φ 为标准正态 PDF，Φ 为 CDF。
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0, a: float = -np.inf, b: float = np.inf):
        if sigma <= 0.0:
            raise ValueError("sigma 必须为正")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.a = float(a)
        self.b = float(b)

        # 计算归一化常数
        self._alpha = (self.a - self.mu) / self.sigma
        self._beta = (self.b - self.mu) / self.sigma
        self._Z = 0.5 * (erf(self._beta / np.sqrt(2.0)) - erf(self._alpha / np.sqrt(2.0)))
        if self._Z < 1e-15:
            self._Z = 1e-15

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """计算概率密度。"""
        x = np.asarray(x, dtype=np.float64)
        z = (x - self.mu) / self.sigma
        pdf_vals = np.exp(-0.5 * z ** 2) / (self.sigma * np.sqrt(2.0 * np.pi))
        # 在截断区间外为 0
        pdf_vals = np.where((x < self.a) | (x > self.b), 0.0, pdf_vals / self._Z)
        return pdf_vals

    def sample(self, size: int = 1, seed: int = None) -> np.ndarray:
        """
        采用拒绝采样从截断正态分布生成随机样本。
        """
        rng = np.random.default_rng(seed)
        samples = []
        batch = size * 10
        while len(samples) < size:
            z = rng.standard_normal(batch)
            x = self.mu + self.sigma * z
            accepted = x[(x >= self.a) & (x <= self.b)]
            samples.extend(accepted[:max(0, size - len(samples))])
        return np.array(samples[:size], dtype=np.float64)

    def moment(self, order: int) -> float:
        """
        计算截断正态分布的矩（源自 moments_truncated_normal_ab）。

        使用递推公式:
            I_0 = 1
            I_1 = - (φ(β) - φ(α)) / (Φ(β) - Φ(α))
            I_r = (r-1) · I_{r-2} - (β^{r-1}·φ(β) - α^{r-1}·φ(α)) / (Φ(β) - Φ(α))
        """
        if order < 0:
            raise ValueError("order 必须 >= 0")

        alpha = self._alpha
        beta = self._beta
        phi_a = np.exp(-0.5 * alpha ** 2) / np.sqrt(2.0 * np.pi)
        phi_b = np.exp(-0.5 * beta ** 2) / np.sqrt(2.0 * np.pi)

        irm2 = 0.0
        irm1 = 0.0
        moment_val = 0.0

        for r in range(order + 1):
            if r == 0:
                ir = 1.0
            elif r == 1:
                ir = -(phi_b - phi_a) / self._Z
            else:
                ir = (r - 1) * irm2 - (
                    beta ** (r - 1) * phi_b - alpha ** (r - 1) * phi_a
                ) / self._Z

            moment_val += math.comb(order, r) * (self.mu ** (order - r)) * (self.sigma ** r) * ir
            irm2 = irm1
            irm1 = ir

        return float(moment_val)


class MonteCarloUncertaintyQuantifier:
    """
    蒙特卡洛不确定性量化器（源自 monte_carlo_nd.m）。
    """

    def __init__(self, dim: int = 3):
        self.dim = dim

    @staticmethod
    def monte_carlo_nd(func, dim_num: int, a: np.ndarray, b: np.ndarray, eval_num: int, seed: int = 55) -> float:
        """
        多维蒙特卡洛积分。

        公式:
            I ≈ V · (1/N) · Σ_{i=1}^{N} f(x_i)
        其中 V = Π (b_j - a_j) 为超体积，x_i 为均匀随机采样点。
        """
        rng = np.random.default_rng(seed)
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        volume = np.prod(b - a)

        total = 0.0
        for _ in range(eval_num):
            x = rng.random(dim_num)
            x = a + (b - a) * x
            total += func(x)

        result = total * volume / eval_num
        return float(result)

    def propagate_depth_uncertainty(
        self,
        base_sound_speed: float,
        base_angle_deg: float,
        base_ttw: float,
        sigma_c: float,
        sigma_theta_deg: float,
        sigma_t: float,
        n_samples: int = 5000,
        seed: int = 55
    ) -> dict:
        """
        通过蒙特卡洛模拟传播深度反演的不确定性。

        反演模型（简化几何）：
            z = (c · t_tw · cos(θ)) / 2
        其中 c 为声速，t_tw 为双程时间，θ 为掠射角。

        参数:
            base_sound_speed: 基准声速 (m/s)
            base_angle_deg:   基准掠射角 (度)
            base_ttw:         基准双程时间 (s)
            sigma_c:          声速标准差 (m/s)
            sigma_theta_deg:  角度标准差 (度)
            sigma_t:          时间测量标准差 (s)
            n_samples:        蒙特卡洛样本数
        返回:
            统计字典
        """
        rng = np.random.default_rng(seed)

        # 截断正态采样（物理边界保护）
        c_dist = TruncatedNormalDistribution(
            mu=base_sound_speed, sigma=sigma_c,
            a=base_sound_speed - 3 * sigma_c,
            b=base_sound_speed + 3 * sigma_c
        )
        theta_dist = TruncatedNormalDistribution(
            mu=base_angle_deg, sigma=sigma_theta_deg,
            a=max(-89.0, base_angle_deg - 3 * sigma_theta_deg),
            b=min(89.0, base_angle_deg + 3 * sigma_theta_deg)
        )
        t_dist = TruncatedNormalDistribution(
            mu=base_ttw, sigma=sigma_t,
            a=max(0.0, base_ttw - 3 * sigma_t),
            b=base_ttw + 3 * sigma_t
        )

        c_samples = c_dist.sample(n_samples)
        theta_samples = theta_dist.sample(n_samples)
        t_samples = t_dist.sample(n_samples)

        # 深度计算
        depths = c_samples * t_samples * np.cos(np.radians(theta_samples)) / 2.0

        mean_depth = float(np.mean(depths))
        std_depth = float(np.std(depths, ddof=1))
        var_depth = float(np.var(depths, ddof=1))

        # 95% 置信区间
        ci_lower = float(np.percentile(depths, 2.5))
        ci_upper = float(np.percentile(depths, 97.5))

        # 解析误差传播（对比）
        theta_rad = np.radians(base_angle_deg)
        dz_dc = base_ttw * np.cos(theta_rad) / 2.0
        dz_dtheta = -base_sound_speed * base_ttw * np.sin(theta_rad) / 2.0 * (np.pi / 180.0)
        dz_dt = base_sound_speed * np.cos(theta_rad) / 2.0
        analytic_var = (dz_dc ** 2) * (sigma_c ** 2) + \
                       (dz_dtheta ** 2) * (sigma_theta_deg ** 2) + \
                       (dz_dt ** 2) * (sigma_t ** 2)

        return {
            'mean_depth': mean_depth,
            'std_depth': std_depth,
            'var_depth': var_depth,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'analytic_variance': float(analytic_var),
            'mc_samples': depths,
        }

    def integrate_error_pdf_over_depth_range(
        self,
        depth_samples: np.ndarray,
        z_min: float,
        z_max: float
    ) -> float:
        """
        估计深度落在特定区间内的概率（蒙特卡洛积分）。

        公式:
            P(z_min ≤ Z ≤ z_max) = ∫_{z_min}^{z_max} f_Z(z) dz
                                 ≈ (1/N) · Σ_i 1_{[z_min, z_max]}(z_i)
        """
        depth_samples = np.asarray(depth_samples)
        count = np.sum((depth_samples >= z_min) & (depth_samples <= z_max))
        prob = count / len(depth_samples)
        return float(prob)
