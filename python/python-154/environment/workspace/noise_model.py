
import numpy as np
from math import sqrt, pi, exp, erf


def _normal_01_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _normal_01_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _normal_01_cdf_inv(p: float) -> float:
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")

    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00
    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01
    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00
    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00
    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = sqrt(-2.0 * np.log(p))
        x = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / (
            (((d1 * q + d2) * q + d3) * q + d4) * q + 1.0
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / (
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        )
    else:
        q = sqrt(-2.0 * np.log(1.0 - p))
        x = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / (
            (((d1 * q + d2) * q + d3) * q + d4) * q + 1.0
        )

    e = _normal_01_cdf(x) - p
    u = e * sqrt(2.0 * pi) * exp(0.5 * x * x)
    x = x - u
    return float(x)


def truncated_normal_ab_sample(mu: float, sigma: float, a: float, b: float,
                                rng: np.random.Generator) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if a >= b:
        raise ValueError("must have a < b")
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    alpha_cdf = _normal_01_cdf(alpha)
    beta_cdf = _normal_01_cdf(beta)
    u = rng.random()
    xi_cdf = alpha_cdf + u * (beta_cdf - alpha_cdf)
    xi = _normal_01_cdf_inv(xi_cdf)
    return float(mu + sigma * xi)


def truncated_normal_ab_mean(mu: float, sigma: float, a: float, b: float) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    alpha_pdf = _normal_01_pdf(alpha)
    beta_pdf = _normal_01_pdf(beta)
    alpha_cdf = _normal_01_cdf(alpha)
    beta_cdf = _normal_01_cdf(beta)
    denom = beta_cdf - alpha_cdf
    if denom < 1e-300:
        return float(mu)
    return float(mu + sigma * (alpha_pdf - beta_pdf) / denom)


def truncated_normal_ab_variance(mu: float, sigma: float, a: float, b: float) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    alpha_pdf = _normal_01_pdf(alpha)
    beta_pdf = _normal_01_pdf(beta)
    alpha_cdf = _normal_01_cdf(alpha)
    beta_cdf = _normal_01_cdf(beta)
    denom = beta_cdf - alpha_cdf
    if denom < 1e-300:
        return float(sigma ** 2)
    term = (alpha_pdf - beta_pdf) / denom
    var = sigma * sigma * (1.0 + (alpha * alpha_pdf - beta * beta_pdf) / denom - term * term)
    return float(max(var, 0.0))


class QuantumAnnealingNoiseModel:

    def __init__(self, n_spins: int, T_bath: float = 0.015,
                 h_noise_sigma: float = 0.05, j_noise_sigma: float = 0.02,
                 gamma_jitter: float = 0.01, seed: int = 154):
        if n_spins <= 0:
            raise ValueError("n_spins must be positive")
        if T_bath < 0:
            raise ValueError("T_bath must be non-negative")
        self.n_spins = n_spins
        self.T_bath = float(T_bath)
        self.h_noise_sigma = float(h_noise_sigma)
        self.j_noise_sigma = float(j_noise_sigma)
        self.gamma_jitter = float(gamma_jitter)
        self.rng = np.random.default_rng(seed)

    def disordered_h(self, h_nominal: np.ndarray) -> np.ndarray:
        h = np.array(h_nominal, dtype=float)
        if h.size != self.n_spins:
            raise ValueError("h_nominal size mismatch")
        a = -3.0 * self.h_noise_sigma
        b = 3.0 * self.h_noise_sigma
        noise = np.array([
            truncated_normal_ab_sample(0.0, self.h_noise_sigma, a, b, self.rng)
            for _ in range(self.n_spins)
        ])
        return h + noise

    def disordered_J(self, J_nominal: np.ndarray) -> np.ndarray:
        J = np.array(J_nominal, dtype=float)
        if J.shape != (self.n_spins, self.n_spins):
            raise ValueError("J_nominal shape mismatch")
        a = -3.0 * self.j_noise_sigma
        b = 3.0 * self.j_noise_sigma
        noise = np.zeros_like(J)
        for i in range(self.n_spins):
            for j in range(i + 1, self.n_spins):
                d = truncated_normal_ab_sample(0.0, self.j_noise_sigma, a, b, self.rng)
                noise[i, j] = d
                noise[j, i] = d
        return J + noise

    def fluctuating_gamma(self, gamma_nominal: float) -> float:
        if gamma_nominal < 0:
            raise ValueError("gamma_nominal must be non-negative")
        a = -0.1
        b = 0.1
        eps = truncated_normal_ab_sample(0.0, self.gamma_jitter, a, b, self.rng)
        return float(max(gamma_nominal * (1.0 + eps), 0.0))

    def thermal_excitation_probability(self, delta_e: float) -> float:
        if self.T_bath < 1e-15:
            return 0.0 if delta_e > 0 else 1.0
        arg = -delta_e / self.T_bath

        if arg < -700:
            return 0.0
        return float(min(1.0, exp(arg)))

    def sample_thermal_state(self, spin_config: np.ndarray,
                              energy_func, n_sweeps: int = 100) -> np.ndarray:
        s = np.array(spin_config, dtype=int)
        if s.size != self.n_spins:
            raise ValueError("spin_config size mismatch")
        e_curr = energy_func(s)
        for _ in range(n_sweeps):
            for i in range(self.n_spins):
                s[i] *= -1
                e_new = energy_func(s)
                delta = e_new - e_curr
                if delta <= 0 or self.rng.random() < self.thermal_excitation_probability(delta):
                    e_curr = e_new
                else:
                    s[i] *= -1
        return s
