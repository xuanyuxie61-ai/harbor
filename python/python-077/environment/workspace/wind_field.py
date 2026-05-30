
import numpy as np
from typing import Tuple, Optional
from numerical_utils import weibull_pdf, weibull_cdf, alnorm_array


class WindField:

    def __init__(self, A: float = 10.0, k: float = 2.0,
                 mu_theta: float = 270.0, kappa: float = 2.0,
                 turbulence_intensity: float = 0.12):
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
        from math import gamma
        return self.A * gamma(1.0 + 1.0 / self.k)

    def std_wind_speed(self) -> float:
        from math import gamma
        var = self.A**2 * (gamma(1.0 + 2.0 / self.k) - gamma(1.0 + 1.0 / self.k)**2)
        return np.sqrt(max(0.0, var))

    def sample_speed(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        xi = rng.random(n)
        return self.A * (-np.log(1.0 - xi)) ** (1.0 / self.k)

    def sample_direction(self, n: int, seed: Optional[int] = None) -> np.ndarray:
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

            accept_prob = np.exp(self.kappa * (np.cos(theta_prop - mu_rad) - 1.0))
            accepted = theta_prop[u < accept_prob]
            samples.extend(accepted)
        directions = np.degrees(np.array(samples[:n])) % 360.0
        return directions

    def add_measurement_noise(self, u: np.ndarray, noise_level: float = 0.05,
                               noise_type: str = 'uniform',
                               seed: Optional[int] = None) -> np.ndarray:
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

        u_noisy = np.maximum(u_noisy, 0.0)
        return u_noisy

    def weibull_to_normal_test(self, n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray, float]:
        u_max = self.A * 3.0
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
        rho = 1.225
        hours_per_year = 8760.0


        n = 1000
        u = np.linspace(max(0.0, u_cut_in), u_cut_out, n)
        f = weibull_pdf(u, self.A, self.k)
        p = 0.5 * rho * u**3 / 1000.0

        integrand = p * f

        du = u[1] - u[0]
        integral = du / 3.0 * (integrand[0] + integrand[-1] +
                               4.0 * np.sum(integrand[1:-1:2]) +
                               2.0 * np.sum(integrand[2:-1:2]))
        return hours_per_year * integral
