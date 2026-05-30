
import numpy as np
from scipy.special import erf


SQRT2 = np.sqrt(2.0)
SQRT2PI = np.sqrt(2.0 * np.pi)


def normal_01_pdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x ** 2) / SQRT2PI


def normal_01_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + erf(x / SQRT2))


def truncated_normal_ab_pdf(x, mu, sigma, a, b):
    x = np.asarray(x, dtype=float)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if b <= a:
        raise ValueError("truncation limits must satisfy a < b")
    
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    denom = normal_01_cdf(beta) - normal_01_cdf(alpha)
    
    if denom <= 1e-15:
        raise ValueError("truncation interval too narrow relative to sigma")
    
    xi = (x - mu) / sigma
    pdf = normal_01_pdf(xi) / (denom * sigma)
    pdf = np.where((x > a) & (x < b), pdf, 0.0)
    return pdf


def truncated_normal_ab_mean(mu, sigma, a, b):
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    denom = normal_01_cdf(beta) - normal_01_cdf(alpha)
    if denom <= 1e-15:
        return mu
    return mu + sigma * (normal_01_pdf(alpha) - normal_01_pdf(beta)) / denom


def truncated_normal_ab_variance(mu, sigma, a, b):
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    denom = normal_01_cdf(beta) - normal_01_cdf(alpha)
    if denom <= 1e-15:
        return sigma ** 2
    
    pdf_a = normal_01_pdf(alpha)
    pdf_b = normal_01_pdf(beta)
    
    term1 = (alpha * pdf_a - beta * pdf_b) / denom
    term2 = ((pdf_a - pdf_b) / denom) ** 2
    var = sigma ** 2 * (1.0 + term1 - term2)
    return max(var, 0.0)


def truncated_normal_ab_sample(mu, sigma, a, b, size=None):
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    cdf_a = normal_01_cdf(alpha)
    cdf_b = normal_01_cdf(beta)
    
    if cdf_b - cdf_a < 1e-15:
        return np.full(size if size is not None else (), mu)
    
    u = np.random.uniform(0.0, 1.0, size=size)

    from scipy.special import erfinv
    z = SQRT2 * erfinv(2.0 * (cdf_a + u * (cdf_b - cdf_a)) - 1.0)
    return mu + sigma * z


def generate_kl_coefficients(n_modes, correlation_length, domain_size=1.0):
    if n_modes <= 0:
        return np.ones(1)
    if correlation_length <= 0:
        raise ValueError("correlation_length must be positive")
    
    k = np.arange(1, n_modes + 1)

    eigenvalues = (correlation_length ** 2) / (correlation_length ** 2 + (k * np.pi / domain_size) ** 2)
    eigenvalues = np.sqrt(eigenvalues)
    

    coeffs = truncated_normal_ab_sample(
        mu=0.0, sigma=1.0, a=-2.0, b=2.0, size=n_modes
    ) * eigenvalues
    return coeffs
