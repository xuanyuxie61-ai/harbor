
import numpy as np
import math
from scipy.special import erf


class TruncatedNormalDistribution:

    def __init__(self, mu: float = 0.0, sigma: float = 1.0, a: float = -np.inf, b: float = np.inf):
        if sigma <= 0.0:
            raise ValueError("sigma 必须为正")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.a = float(a)
        self.b = float(b)


        self._alpha = (self.a - self.mu) / self.sigma
        self._beta = (self.b - self.mu) / self.sigma
        self._Z = 0.5 * (erf(self._beta / np.sqrt(2.0)) - erf(self._alpha / np.sqrt(2.0)))
        if self._Z < 1e-15:
            self._Z = 1e-15

    def pdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        z = (x - self.mu) / self.sigma
        pdf_vals = np.exp(-0.5 * z ** 2) / (self.sigma * np.sqrt(2.0 * np.pi))

        pdf_vals = np.where((x < self.a) | (x > self.b), 0.0, pdf_vals / self._Z)
        return pdf_vals

    def sample(self, size: int = 1, seed: int = None) -> np.ndarray:
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

    def __init__(self, dim: int = 3):
        self.dim = dim

    @staticmethod
    def monte_carlo_nd(func, dim_num: int, a: np.ndarray, b: np.ndarray, eval_num: int, seed: int = 55) -> float:
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
        rng = np.random.default_rng(seed)


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


        depths = c_samples * t_samples * np.cos(np.radians(theta_samples)) / 2.0

        mean_depth = float(np.mean(depths))
        std_depth = float(np.std(depths, ddof=1))
        var_depth = float(np.var(depths, ddof=1))


        ci_lower = float(np.percentile(depths, 2.5))
        ci_upper = float(np.percentile(depths, 97.5))


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
        depth_samples = np.asarray(depth_samples)
        count = np.sum((depth_samples >= z_min) & (depth_samples <= z_max))
        prob = count / len(depth_samples)
        return float(prob)
