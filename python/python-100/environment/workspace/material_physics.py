
import numpy as np
from scipy.special import erf, erfcinv


class NonlinearMagneticMaterial:

    MU0 = 4.0 * np.pi * 1.0e-7

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
        H = np.asarray(H, dtype=float)

        H = np.clip(H, -1.0e7, 1.0e7)
        return self._mu0 * H + self._prefactor * np.arctan(self._slope_arg * H)

    def h_field(self, B: np.ndarray, tol: float = 1.0e-12, max_iter: int = 100) -> np.ndarray:
        B = np.asarray(B, dtype=float)
        B = np.clip(B, -self.B_sat * 0.999999, self.B_sat * 0.999999)

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
        H = np.asarray(H, dtype=float)
        arg = self._slope_arg * H
        return self._mu0 + (self.mu_r_init - 1.0) * self._mu0 / (1.0 + arg * arg)

    def reluctivity(self, B: np.ndarray) -> np.ndarray:

        raise NotImplementedError("Hole_1: 需实现磁阻率 ν = H/B 的计算逻辑")

    def differential_reluctivity(self, B: np.ndarray) -> np.ndarray:
        B = np.asarray(B, dtype=float)
        H = self.h_field(B)
        mu_diff = self.differential_permeability(H)
        safe_B = np.where(np.abs(B) < 1.0e-14, np.sign(B + 1.0e-20) * 1.0e-14, B)
        nu = H / safe_B
        return (1.0 / mu_diff - nu) / safe_B


class PermanentMagnet:

    MU0 = 4.0 * np.pi * 1.0e-7

    def __init__(self, B_r: float, mu_rec: float = 1.05):
        if B_r <= 0.0:
            raise ValueError(f"剩磁必须为正, 得到 {B_r}")
        if mu_rec < 1.0:
            raise ValueError(f"回复磁导率必须 ≥ 1, 得到 {mu_rec}")
        self.B_r = float(B_r)
        self.mu_rec = float(mu_rec)
        self.H_c = self.B_r / (self.MU0 * self.mu_rec)
        self.M = self.B_r / self.MU0

    def b_field(self, H: float) -> float:
        return self.MU0 * self.mu_rec * H + self.B_r

    def reluctivity(self) -> float:
        return 1.0 / (self.MU0 * self.mu_rec)


class LogNormalUncertainty:

    def __init__(self, mu_ln: float = 0.0, sigma_ln: float = 1.0):
        if sigma_ln <= 0.0:
            raise ValueError(f"对数标准差必须为正, 得到 {sigma_ln}")
        self.mu_ln = float(mu_ln)
        self.sigma_ln = float(sigma_ln)
        self._sqrt2 = np.sqrt(2.0)

    @classmethod
    def from_mean_variance(cls, mean: float, variance: float):
        if mean <= 0.0:
            raise ValueError(f"均值必须为正, 得到 {mean}")
        if variance <= 0.0:
            raise ValueError(f"方差必须为正, 得到 {variance}")
        sigma2_ln = np.log(1.0 + variance / (mean * mean))
        sigma_ln = np.sqrt(sigma2_ln)
        mu_ln = np.log(mean) - 0.5 * sigma2_ln
        return cls(mu_ln, sigma_ln)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x)
        mask = x > 0.0
        if np.any(mask):
            xm = x[mask]
            z = (np.log(xm) - self.mu_ln) / self.sigma_ln
            result[mask] = np.exp(-0.5 * z * z) / (self.sigma_ln * xm * np.sqrt(2.0 * np.pi))
        return result

    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x)
        mask = x > 0.0
        if np.any(mask):
            z = (np.log(x[mask]) - self.mu_ln) / (self.sigma_ln * self._sqrt2)
            result[mask] = 0.5 * (1.0 + erf(z))
        return result

    def cdf_inv(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        p = np.clip(p, 1.0e-14, 1.0 - 1.0e-14)
        z = self._sqrt2 * erfcinv(2.0 * (1.0 - p))
        return np.exp(self.mu_ln + self.sigma_ln * z)

    def sample(self, size: int = 1, rng: np.random.Generator = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return np.exp(rng.normal(self.mu_ln, self.sigma_ln, size=size))

    def mean(self) -> float:
        return np.exp(self.mu_ln + 0.5 * self.sigma_ln * self.sigma_ln)

    def variance(self) -> float:
        return (np.exp(self.sigma_ln * self.sigma_ln) - 1.0) * np.exp(
            2.0 * self.mu_ln + self.sigma_ln * self.sigma_ln
        )

    def sample_mean_variance(self, samples: np.ndarray) -> tuple:
        samples = np.asarray(samples, dtype=float)
        samples = samples[samples > 0.0]
        if len(samples) < 2:
            raise ValueError("至少需要2个正样本")
        log_samples = np.log(samples)
        mu_est = float(np.mean(log_samples))
        sigma_est = float(np.std(log_samples, ddof=1))
        return mu_est, sigma_est


def temperature_dependent_conductivity(sigma_20: float, T: float, alpha_T: float = 0.00393) -> float:
    if T < -273.15:
        raise ValueError("温度低于绝对零度")
    denom = 1.0 + alpha_T * (T - 20.0)
    if np.abs(denom) < 1.0e-14:
        denom = np.sign(denom + 1.0e-20) * 1.0e-14
    return sigma_20 / denom


def build_motor_material_library():
    materials = {
        "stator_core": NonlinearMagneticMaterial(mu_r_init=5000.0, B_sat=2.0, name="M19_GrainOriented"),
        "rotor_core": NonlinearMagneticMaterial(mu_r_init=3000.0, B_sat=2.1, name="M15_NonOriented"),
        "permanent_magnet": PermanentMagnet(B_r=1.2, mu_rec=1.05),
        "copper_winding": {"sigma_20": 5.8e7, "alpha_T": 0.00393, "mu_r": 0.999991},
        "air_gap": {"mu_r": 1.0, "sigma": 0.0},
    }
    return materials
