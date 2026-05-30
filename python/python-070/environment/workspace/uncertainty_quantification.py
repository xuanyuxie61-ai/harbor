
import numpy as np
from utils import NumericalConfig



_PRIMES = np.array([
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
    233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
    283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
    353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
    419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
    467, 479, 487, 491, 499, 503, 509, 521, 523, 541
], dtype=int)


def van_der_corput(i, base):
    if i < 0:
        raise ValueError("i must be non-negative")
    if base < 2:
        raise ValueError("base must be >= 2")

    r = 0.0
    inv_base = 1.0 / base
    factor = inv_base
    while i > 0:
        digit = i % base
        r += digit * factor
        factor *= inv_base
        i //= base
    return r


def hammersley_value(i, m, n):
    if m < 1 or m > 100:
        raise ValueError("m must be between 1 and 100")
    if n < 1:
        raise ValueError("n must be >= 1")

    r = np.zeros(m, dtype=float)
    r[0] = (i % (n + 1)) / n if n > 0 else 0.0

    for j in range(1, m):
        base = _PRIMES[j - 1]
        r[j] = van_der_corput(i, base)

    return r


def hammersley_sequence(i1, i2, m, n):
    if i1 <= i2:
        step = 1
    else:
        step = -1

    l = abs(i2 - i1) + 1
    points = np.zeros((l, m), dtype=float)
    idx = 0
    for i in range(i1, i2 + step, step):
        points[idx, :] = hammersley_value(i, m, n)
        idx += 1

    return points


def monte_carlo_integral_1d(func, a, b, n_samples, method='mc'):
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    if method == 'mc':
        x = np.random.uniform(0.0, 1.0, size=n_samples)
    elif method == 'qmc':
        pts = hammersley_sequence(0, n_samples - 1, m=1, n=n_samples)
        x = pts[:, 0]
    else:
        raise ValueError("method must be 'mc' or 'qmc'")


    x_mapped = a + (b - a) * x
    fx = func(x_mapped)
    return (b - a) * np.mean(fx)


def fishery_risk_assessment(r_dist, K_dist, q_dist, E_fixed, p, c, delta,
                            n_samples=5000, method='qmc'):
    r_mean, r_std = r_dist
    K_mean, K_std = K_dist
    q_mean, q_std = q_dist


    if method == 'qmc':
        samples = hammersley_sequence(0, n_samples - 1, m=3, n=n_samples)
    else:
        samples = np.random.uniform(0.0, 1.0, size=(n_samples, 3))



    def lognormal_params(mu, sigma):
        sigma_ln = np.sqrt(np.log(1.0 + (sigma / mu) ** 2))
        mu_ln = np.log(mu) - 0.5 * sigma_ln ** 2
        return mu_ln, sigma_ln

    mu_r, sig_r = lognormal_params(r_mean, r_std)
    mu_K, sig_K = lognormal_params(K_mean, K_std)
    mu_q, sig_q = lognormal_params(q_mean, q_std)




    def approx_norm_ppf(u):
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
        d1 = 7.784695709041460e-03
        d2 = 3.224671290700398e-01
        d3 = 2.445134137142996e+00
        d4 = 3.754408661907416e+00
        p_low = 0.02425
        p_high = 1.0 - p_low

        u = np.asarray(u, dtype=float)

        u = np.clip(u, NumericalConfig.EPS, 1.0 - NumericalConfig.EPS)
        z = np.zeros_like(u)

        mask1 = u < p_low
        q_val = np.sqrt(-2.0 * np.log(u[mask1]))
        z[mask1] = (((((c1 * q_val + c2) * q_val + c3) * q_val + c4) * q_val + c5) * q_val + c6) / \
                   ((((d1 * q_val + d2) * q_val + d3) * q_val + d4) * q_val + 1.0)

        mask2 = (u >= p_low) & (u <= p_high)
        q_val = u[mask2] - 0.5
        r_val = q_val * q_val
        z[mask2] = (((((a1 * r_val + a2) * r_val + a3) * r_val + a4) * r_val + a5) * r_val + a6) * q_val / \
                   (((((b1 * r_val + b2) * r_val + b3) * r_val + b4) * r_val + b5) * r_val + 1.0)

        mask3 = u > p_high
        q_val = np.sqrt(-2.0 * np.log(1.0 - u[mask3]))
        z[mask3] = -(((((c1 * q_val + c2) * q_val + c3) * q_val + c4) * q_val + c5) * q_val + c6) / \
                    ((((d1 * q_val + d2) * q_val + d3) * q_val + d4) * q_val + 1.0)

        return z

    z_r = approx_norm_ppf(samples[:, 0])
    z_K = approx_norm_ppf(samples[:, 1])
    z_q = approx_norm_ppf(samples[:, 2])

    r_samples = np.exp(mu_r + sig_r * z_r)
    K_samples = np.exp(mu_K + sig_K * z_K)
    q_samples = np.exp(mu_q + sig_q * z_q)


    profits = np.zeros(n_samples, dtype=float)
    biomasses = np.zeros(n_samples, dtype=float)
    B_lim = 0.3 * K_mean







    for i in range(n_samples):
        pass

    results = {
        'expected_profit': np.mean(profits),
        'std_profit': np.std(profits),
        'profit_cv': np.std(profits) / abs(np.mean(profits)) if abs(np.mean(profits)) > NumericalConfig.EPS else np.inf,
        'prob_biomass_below_limit': np.mean(biomasses < B_lim),
        'prob_negative_profit': np.mean(profits < 0),
        'expected_biomass': np.mean(biomasses),
        'biomass_percentile_5': np.percentile(biomasses, 5),
        'biomass_percentile_95': np.percentile(biomasses, 95),
    }

    return results
