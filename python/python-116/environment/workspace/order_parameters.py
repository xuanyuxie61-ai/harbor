
import numpy as np
from scipy.special import gamma as scipy_gamma


def jacobi_polynomial(n, alpha, beta, x):
    if alpha <= -1.0 or beta <= -1.0:
        raise ValueError("alpha, beta 必须大于 -1。")
    if n < 0:
        return np.zeros((len(np.atleast_1d(x)), 0))

    x = np.asarray(x, dtype=float)
    m = x.size
    P = np.zeros((m, n + 1))
    P[:, 0] = 1.0

    if n == 0:
        return P

    P[:, 1] = ((alpha + beta + 2.0) * x + (alpha - beta)) / 2.0

    for k in range(1, n):
        ak = 2.0 * (k + 1.0) * (k + alpha + beta + 1.0) * (2.0 * k + alpha + beta)
        bk = (2.0 * k + alpha + beta + 1.0) * (alpha**2 - beta**2)
        ck = (2.0 * k + alpha + beta + 1.0) * (2.0 * k + alpha + beta) * (2.0 * k + alpha + beta + 2.0)
        dk = 2.0 * (k + alpha) * (k + beta) * (2.0 * k + alpha + beta + 2.0)

        if abs(ak) < 1e-15:
            raise RuntimeError("Jacobi 递推分母 ak 接近零。")

        P[:, k + 1] = ((bk + ck * x) * P[:, k] - dk * P[:, k - 1]) / ak

    return P


def jacobi_norm_constant(n, alpha, beta):
    if n < 0:
        return 0.0
    num = (2.0 ** (alpha + beta + 1.0)) * scipy_gamma(n + alpha + 1.0) * scipy_gamma(n + beta + 1.0)
    den = (2.0 * n + alpha + beta + 1.0) * scipy_gamma(n + 1.0) * scipy_gamma(n + alpha + beta + 1.0)
    if den == 0 or not np.isfinite(den):
        return 0.0
    return float(num / den)


def jacobi_zeros_guess(n, alpha, beta):
    if n <= 0:
        return np.array([])
    k = np.arange(1, n + 1)
    denom = n + 0.5 * (alpha + beta + 1.0)
    if denom <= 0:
        denom = 1.0
    return np.cos((k - 0.25) * np.pi / denom)


class OrientationalOrderAnalysis:

    def __init__(self, n_max=12, alpha=0.0, beta=2.0):
        if n_max < 0:
            raise ValueError("n_max 必须非负。")
        if alpha <= -1.0 or beta <= -1.0:
            raise ValueError("alpha, beta 必须大于 -1。")
        self.n_max = n_max
        self.alpha = alpha
        self.beta = beta

    def expand_odf(self, cos_theta_samples):
        cos_theta_samples = np.asarray(cos_theta_samples)
        cos_theta_samples = np.clip(cos_theta_samples, -1.0, 1.0)

        coeffs = np.zeros(self.n_max + 1)
        P_vals = jacobi_polynomial(self.n_max, self.alpha, self.beta,
                                   cos_theta_samples)
        w = ((1.0 - cos_theta_samples) ** self.alpha *
             (1.0 + cos_theta_samples) ** self.beta)
        w = np.where(w < 0, 0.0, w)

        for n in range(self.n_max + 1):
            h_n = jacobi_norm_constant(n, self.alpha, self.beta)
            if h_n <= 0:
                coeffs[n] = 0.0
                continue
            integrand = w * P_vals[:, n]
            coeffs[n] = np.mean(integrand) / h_n

        return coeffs

    def order_parameters_from_coeffs(self, coeffs):
        coeffs = np.asarray(coeffs)
        if coeffs[0] == 0:
            return np.zeros_like(coeffs)
        s_params = coeffs / coeffs[0]
        return s_params

    def reconstruct_odf(self, x_grid, coeffs):
        x_grid = np.asarray(x_grid)
        x_grid = np.clip(x_grid, -1.0, 1.0)
        P_vals = jacobi_polynomial(self.n_max, self.alpha, self.beta, x_grid)
        return P_vals @ coeffs

    def compute_entropy(self, coeffs):

        n_quad = max(2 * self.n_max + 1, 20)
        x_nodes = np.linspace(-0.999, 0.999, n_quad)
        dx = x_nodes[1] - x_nodes[0]
        f = self.reconstruct_odf(x_nodes, coeffs)
        f = np.clip(f, 1e-12, None)
        entropy = -np.sum(f * np.log(f)) * dx
        return float(entropy)


def spherical_harmonic_y20_approx(cos_theta):
    coeff = np.sqrt(5.0 / (16.0 * np.pi))
    return coeff * (3.0 * cos_theta ** 2 - 1.0)


def debye_waller_factor(order_param, temperature, moment_inertia=1.0):
    if moment_inertia <= 0:
        raise ValueError("转动惯量必须为正。")
    kb = 1.380649e-23
    u2 = kb * temperature / moment_inertia * (1.0 - order_param)
    B = (8.0 * np.pi ** 2 / 3.0) * u2
    return float(B)
