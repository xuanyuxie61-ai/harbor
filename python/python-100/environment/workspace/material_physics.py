"""
material_physics.py
================================================================================
电机电磁场分析中的材料物理模型与不确定性量化

融合原项目:
  - 698_log_normal : 对数正态分布的PDF、CDF、采样、统计矩计算

核心科学内容:
  1. 非线性磁材料 B-H 曲线建模（基于反正切函数的非线性磁阻率模型）
  2. 永磁体（NdFeB）的退磁曲线与剩磁模型
  3. 硅钢片磁导率的对数正态不确定性量化
  4. 电导率与磁导率的温度依赖关系
================================================================================
"""

import numpy as np
from scipy.special import erf, erfcinv


class NonlinearMagneticMaterial:
    """
    非线性磁性材料模型，基于反正切函数的 B-H 曲线逼近。

    磁通密度 B 与磁场强度 H 的关系由以下经验公式描述:

        B(H) = μ_0 * H + (2 B_s / π) * arctan( π (μ_r - 1) μ_0 H / (2 B_s) )

    其中:
        μ_0 = 4π × 10^{-7} H/m   为真空磁导率
        μ_r                      为初始相对磁导率
        B_s                      为饱和磁通密度 (T)

    微分磁导率（增量磁导率）:

        μ_diff(H) = dB/dH = μ_0 + (μ_r - 1) μ_0 / (1 + (π (μ_r - 1) μ_0 H / (2 B_s))^2 )

    磁阻率（reluctivity）定义为:

        ν(H) = H / B(H) = 1 / μ_apparent(H)

    该模型满足以下边界条件:
        - 当 H → 0 时,  μ_diff → μ_r μ_0
        - 当 H → ∞ 时,  B → B_s,  μ_diff → μ_0
    """

    MU0 = 4.0 * np.pi * 1.0e-7  # 真空磁导率 [H/m]

    def __init__(self, mu_r_init: float, B_sat: float, name: str = "SiSteel"):
        if mu_r_init <= 1.0:
            raise ValueError(f"初始相对磁导率必须大于1, 得到 {mu_r_init}")
        if B_sat <= 0.0:
            raise ValueError(f"饱和磁通密度必须为正, 得到 {B_sat}")
        self.mu_r_init = float(mu_r_init)
        self.B_sat = float(B_sat)
        self.name = name
        self._mu0 = self.MU0
        self._prefactor = 2.0 * self.B_sat / np.pi
        self._slope_arg = np.pi * (self.mu_r_init - 1.0) * self._mu0 / (2.0 * self.B_sat)

    def b_field(self, H: np.ndarray) -> np.ndarray:
        """由磁场强度 H (A/m) 计算磁通密度 B (T)."""
        H = np.asarray(H, dtype=float)
        # 边界保护: 防止极端H值导致数值溢出
        H = np.clip(H, -1.0e7, 1.0e7)
        return self._mu0 * H + self._prefactor * np.arctan(self._slope_arg * H)

    def h_field(self, B: np.ndarray, tol: float = 1.0e-12, max_iter: int = 100) -> np.ndarray:
        """由磁通密度 B (T) 反解磁场强度 H (A/m), 使用牛顿迭代法."""
        B = np.asarray(B, dtype=float)
        B = np.clip(B, -self.B_sat * 0.999999, self.B_sat * 0.999999)
        # 初始猜测: 线性近似
        H = B / (self._mu0 * self.mu_r_init)
        for _ in range(max_iter):
            B_pred = self.b_field(H)
            dB_dH = self.differential_permeability(H)
            delta = (B_pred - B) / dB_dH
            H_new = H - delta
            if np.all(np.abs(delta) < tol):
                return H_new
            H = H_new
        return H

    def differential_permeability(self, H: np.ndarray) -> np.ndarray:
        """计算微分磁导率 μ_diff = dB/dH."""
        H = np.asarray(H, dtype=float)
        arg = self._slope_arg * H
        return self._mu0 + (self.mu_r_init - 1.0) * self._mu0 / (1.0 + arg * arg)

    def reluctivity(self, B: np.ndarray) -> np.ndarray:
        """计算磁阻率 ν = H/B = 1/μ_apparent, 用于有限元刚度矩阵组装."""
        # TODO: 请实现非线性磁阻率计算
        raise NotImplementedError("Hole_1: 需实现磁阻率 ν = H/B 的计算逻辑")

    def differential_reluctivity(self, B: np.ndarray) -> np.ndarray:
        """
        计算磁阻率对 B 的导数 dν/dB, 用于牛顿-拉夫逊非线性求解.

        由 ν = H/B, 得:
            dν/dB = (dH/dB * B - H) / B^2
                  = (B / μ_diff - H) / B^2
                  = (1/μ_diff - ν) / B
        """
        B = np.asarray(B, dtype=float)
        H = self.h_field(B)
        mu_diff = self.differential_permeability(H)
        safe_B = np.where(np.abs(B) < 1.0e-14, np.sign(B + 1.0e-20) * 1.0e-14, B)
        nu = H / safe_B
        return (1.0 / mu_diff - nu) / safe_B


class PermanentMagnet:
    """
    永磁体（NdFeB）模型，采用线性退磁曲线近似.

    永磁体的工作点由退磁曲线描述:

        B = μ_0 * μ_rec * H + B_r

    其中:
        μ_rec 为回复磁导率（通常 ≈ 1.05）
        B_r   为剩磁磁通密度 [T]
        H_c   为矫顽力 [A/m], 满足 B_r = μ_0 * μ_rec * H_c

    等效面电流密度（磁化电流模型）:

        J_m = ∇ × M,   M = B_r / μ_0   (A/m)

    在二维轴对称模型中, 永磁体作为源项贡献:

        J_s,eq = (1/μ) (∂M_y/∂x - ∂M_x/∂y)
    """

    MU0 = 4.0 * np.pi * 1.0e-7

    def __init__(self, B_r: float, mu_rec: float = 1.05):
        if B_r <= 0.0:
            raise ValueError(f"剩磁必须为正, 得到 {B_r}")
        if mu_rec < 1.0:
            raise ValueError(f"回复磁导率必须 ≥ 1, 得到 {mu_rec}")
        self.B_r = float(B_r)
        self.mu_rec = float(mu_rec)
        self.H_c = self.B_r / (self.MU0 * self.mu_rec)
        self.M = self.B_r / self.MU0  # 磁化强度 [A/m]

    def b_field(self, H: float) -> float:
        """退磁曲线 B(H)."""
        return self.MU0 * self.mu_rec * H + self.B_r

    def reluctivity(self) -> float:
        """永磁体区域的恒定磁阻率."""
        return 1.0 / (self.MU0 * self.mu_rec)


class LogNormalUncertainty:
    """
    对数正态不确定性量化模型，用于电机材料参数的随机建模.

    若随机变量 X 服从对数正态分布，则 ln(X) ~ N(μ_ln, σ_ln^2),
    其概率密度函数为:

        f(x; μ_ln, σ_ln) = 1/(x σ_ln √(2π)) exp( - (ln x - μ_ln)^2 / (2 σ_ln^2) ),  x > 0

    均值与方差:
        E[X] = exp(μ_ln + σ_ln^2 / 2)
        Var[X] = (exp(σ_ln^2) - 1) exp(2 μ_ln + σ_ln^2)

    本类提供:
        - PDF / CDF 计算
        - 参数估计（由均值和方差反解 μ_ln, σ_ln）
        - 随机采样
        - 统计矩计算

    融合原项目 698_log_normal 的核心算法.
    """

    def __init__(self, mu_ln: float = 0.0, sigma_ln: float = 1.0):
        if sigma_ln <= 0.0:
            raise ValueError(f"对数标准差必须为正, 得到 {sigma_ln}")
        self.mu_ln = float(mu_ln)
        self.sigma_ln = float(sigma_ln)
        self._sqrt2 = np.sqrt(2.0)

    @classmethod
    def from_mean_variance(cls, mean: float, variance: float):
        """由均值和方差构造对数正态分布."""
        if mean <= 0.0:
            raise ValueError(f"均值必须为正, 得到 {mean}")
        if variance <= 0.0:
            raise ValueError(f"方差必须为正, 得到 {variance}")
        sigma2_ln = np.log(1.0 + variance / (mean * mean))
        sigma_ln = np.sqrt(sigma2_ln)
        mu_ln = np.log(mean) - 0.5 * sigma2_ln
        return cls(mu_ln, sigma_ln)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """计算对数正态 PDF."""
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x)
        mask = x > 0.0
        if np.any(mask):
            xm = x[mask]
            z = (np.log(xm) - self.mu_ln) / self.sigma_ln
            result[mask] = np.exp(-0.5 * z * z) / (self.sigma_ln * xm * np.sqrt(2.0 * np.pi))
        return result

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """计算对数正态 CDF."""
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x)
        mask = x > 0.0
        if np.any(mask):
            z = (np.log(x[mask]) - self.mu_ln) / (self.sigma_ln * self._sqrt2)
            result[mask] = 0.5 * (1.0 + erf(z))
        return result

    def cdf_inv(self, p: np.ndarray) -> np.ndarray:
        """计算对数正态 CDF 的反函数（分位数函数）."""
        p = np.asarray(p, dtype=float)
        p = np.clip(p, 1.0e-14, 1.0 - 1.0e-14)
        z = self._sqrt2 * erfcinv(2.0 * (1.0 - p))
        return np.exp(self.mu_ln + self.sigma_ln * z)

    def sample(self, size: int = 1, rng: np.random.Generator = None) -> np.ndarray:
        """从对数正态分布中采样."""
        if rng is None:
            rng = np.random.default_rng()
        return np.exp(rng.normal(self.mu_ln, self.sigma_ln, size=size))

    def mean(self) -> float:
        """计算均值 E[X]."""
        return np.exp(self.mu_ln + 0.5 * self.sigma_ln * self.sigma_ln)

    def variance(self) -> float:
        """计算方差 Var[X]."""
        return (np.exp(self.sigma_ln * self.sigma_ln) - 1.0) * np.exp(
            2.0 * self.mu_ln + self.sigma_ln * self.sigma_ln
        )

    def sample_mean_variance(self, samples: np.ndarray) -> tuple:
        """由样本估计对数正态参数."""
        samples = np.asarray(samples, dtype=float)
        samples = samples[samples > 0.0]
        if len(samples) < 2:
            raise ValueError("至少需要2个正样本")
        log_samples = np.log(samples)
        mu_est = float(np.mean(log_samples))
        sigma_est = float(np.std(log_samples, ddof=1))
        return mu_est, sigma_est


def temperature_dependent_conductivity(sigma_20: float, T: float, alpha_T: float = 0.00393) -> float:
    """
    铜/铝电导率的温度依赖关系（IEC标准近似）.

        σ(T) = σ_20 / (1 + α_T (T - 20°C))

    参数:
        sigma_20 : 20°C 时的电导率 [S/m]
        T        : 温度 [°C]
        alpha_T  : 电阻温度系数 [1/°C], 铜约 0.00393
    """
    if T < -273.15:
        raise ValueError("温度低于绝对零度")
    denom = 1.0 + alpha_T * (T - 20.0)
    if np.abs(denom) < 1.0e-14:
        denom = np.sign(denom + 1.0e-20) * 1.0e-14
    return sigma_20 / denom


def build_motor_material_library():
    """构建典型电机材料库，返回字典."""
    materials = {
        "stator_core": NonlinearMagneticMaterial(mu_r_init=5000.0, B_sat=2.0, name="M19_GrainOriented"),
        "rotor_core": NonlinearMagneticMaterial(mu_r_init=3000.0, B_sat=2.1, name="M15_NonOriented"),
        "permanent_magnet": PermanentMagnet(B_r=1.2, mu_rec=1.05),
        "copper_winding": {"sigma_20": 5.8e7, "alpha_T": 0.00393, "mu_r": 0.999991},
        "air_gap": {"mu_r": 1.0, "sigma": 0.0},
    }
    return materials
