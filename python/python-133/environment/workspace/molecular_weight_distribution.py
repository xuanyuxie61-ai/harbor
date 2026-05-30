
import numpy as np
from typing import Tuple, Optional, Callable
from scipy.special import erf, gammaln, factorial


def flory_schulz_distribution(n: np.ndarray, p: float) -> np.ndarray:
    n = np.asarray(n)
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")
    n = np.maximum(n, 1.0)
    pmf = (1.0 - p) * (p ** (n - 1.0))

    pmf = np.where(n < 1.0, 0.0, pmf)
    return pmf


def flory_schulz_moments(p: float, max_moment: int = 4) -> np.ndarray:
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")
    moments = np.zeros(max_moment + 1)
    moments[0] = 1.0
    if max_moment >= 1:
        moments[1] = 1.0 / (1.0 - p)
    if max_moment >= 2:
        moments[2] = (1.0 + p) / (1.0 - p) ** 2
    if max_moment >= 3:
        moments[3] = (1.0 + 4.0 * p + p ** 2) / (1.0 - p) ** 3
    if max_moment >= 4:
        moments[4] = (1.0 + 11.0 * p + 11.0 * p ** 2 + p ** 3) / (1.0 - p) ** 4
    return moments


def normal_01_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def normal_01_cdf_inv(p: float) -> float:
    from scipy.special import erfinv
    p = np.clip(p, 1.0e-10, 1.0 - 1.0e-10)
    return np.sqrt(2.0) * erfinv(2.0 * p - 1.0)


def truncated_normal_pdf(n: np.ndarray,
                         mu: float,
                         sigma: float,
                         a: float,
                         b: float) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    sigma = max(sigma, 1.0e-12)
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma

    denom = sigma * (normal_01_cdf(beta) - normal_01_cdf(alpha))
    denom = max(denom, 1.0e-15)

    z = (n - mu) / sigma
    pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z ** 2) / denom
    pdf = np.where((n >= a) & (n <= b), pdf, 0.0)
    return pdf


def truncated_normal_sample(mu: float,
                            sigma: float,
                            a: float,
                            b: float,
                            size: int = 1,
                            rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(seed=42)
    sigma = max(sigma, 1.0e-12)
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    alpha_cdf = normal_01_cdf(alpha)
    beta_cdf = normal_01_cdf(beta)

    u = rng.random(size=size)
    xi_cdf = alpha_cdf + u * (beta_cdf - alpha_cdf)
    xi = normal_01_cdf_inv(xi_cdf)
    x = mu + sigma * xi
    x = np.clip(x, a, b)
    return x


def lognormal_mwd_pdf(n: np.ndarray,
                      mu_log: float,
                      sigma_log: float,
                      a: float = 1.0,
                      b: float = 1.0e6) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    n = np.maximum(n, 1.0e-3)
    sigma_log = max(sigma_log, 1.0e-12)

    alpha = (np.log(a) - mu_log) / sigma_log
    beta = (np.log(b) - mu_log) / sigma_log
    Z = normal_01_cdf(beta) - normal_01_cdf(alpha)
    Z = max(Z, 1.0e-15)

    z = (np.log(n) - mu_log) / sigma_log
    pdf = (1.0 / (n * sigma_log * np.sqrt(2.0 * np.pi))
           * np.exp(-0.5 * z ** 2) / Z)
    pdf = np.where((n >= a) & (n <= b), pdf, 0.0)
    return pdf


def polygon_moment(n_vertices: int,
                   x: np.ndarray,
                   y: np.ndarray,
                   p: int,
                   q: int) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size != n_vertices or y.size != n_vertices:
        raise ValueError("x and y must have length n_vertices")

    nu_pq = 0.0
    xj = x[-1]
    yj = y[-1]

    for i in range(n_vertices):
        xi = x[i]
        yi = y[i]
        s_pq = 0.0
        for k in range(p + 1):
            for l in range(q + 1):

                ck = factorial(k + l) / (factorial(k) * factorial(l))
                ckl = factorial(p + q - k - l) / (factorial(q - l) * factorial(p - k))
                s_pq += ck * ckl * (xi ** k) * (xj ** (p - k)) * (yi ** l) * (yj ** (q - l))
        nu_pq += (xj * yi - xi * yj) * s_pq
        xj = xi
        yj = yi

    denom = (p + q + 2) * (p + q + 1) * factorial(p + q) / (factorial(p) * factorial(q))
    nu_pq = nu_pq / denom
    return float(nu_pq)


def mwd_from_moments(moments: np.ndarray,
                     n_grid: np.ndarray,
                     method: str = 'maxent') -> np.ndarray:
    n_grid = np.asarray(n_grid, dtype=float)
    n_grid = np.maximum(n_grid, 1.0e-3)

    if method == 'gamma':

        mu1 = moments[1] if len(moments) > 1 else 100.0
        mu2 = moments[2] if len(moments) > 2 else mu1 ** 2 * 2.0
        var = max(mu2 - mu1 ** 2, 1.0e-6)
        alpha = mu1 ** 2 / var
        beta = var / mu1
        from scipy.stats import gamma
        pdf = gamma.pdf(n_grid, alpha, scale=beta)
    elif method == 'lognormal':
        mu1 = moments[1] if len(moments) > 1 else 100.0
        mu2 = moments[2] if len(moments) > 2 else mu1 ** 2 * 2.0
        var = max(mu2 - mu1 ** 2, 1.0e-6)
        sigma2 = np.log(var / mu1 ** 2 + 1.0)
        mu_log = np.log(mu1) - 0.5 * sigma2
        sigma_log = np.sqrt(sigma2)
        pdf = lognormal_mwd_pdf(n_grid, mu_log, sigma_log)
    else:

        mu1 = moments[1] if len(moments) > 1 else 100.0
        lam = 1.0 / mu1
        pdf = lam * np.exp(-lam * n_grid)

    pdf = np.maximum(pdf, 0.0)

    integral = np.trapezoid(pdf, n_grid)
    if integral > 1.0e-15:
        pdf /= integral
    return pdf


def compute_pdi_from_moments(moments: np.ndarray) -> Tuple[float, float, float]:
    mu0 = moments[0] if len(moments) > 0 else 1.0
    mu1 = moments[1] if len(moments) > 1 else 1.0
    mu2 = moments[2] if len(moments) > 2 else 1.0






    raise NotImplementedError("Hole 2: 请实现由矩计算 DP_n、DP_w、PDI 的公式")


def local_mwd_broadening(moments_local: np.ndarray,
                         velocity_gradient: float,
                         diffusion_coeff: float,
                         reaction_rate: float) -> np.ndarray:
    chi = 0.05
    denominator = max(diffusion_coeff + reaction_rate, 1.0e-12)
    correction = 1.0 + chi * velocity_gradient / denominator
    correction = min(correction, 2.0)
    return moments_local * correction
