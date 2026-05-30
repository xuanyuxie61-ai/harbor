
import numpy as np
from typing import Tuple, List, Optional
from scipy.special import factorial






def hermite_polynomial(n: int, x: float) -> float:
    if n < 0:
        return 0.0
    if n == 0:
        return 1.0
    if n == 1:
        return x
    H_prev2 = 1.0
    H_prev1 = x
    for k in range(1, n):
        H_curr = x * H_prev1 - k * H_prev2
        H_prev2, H_prev1 = H_prev1, H_curr
    return H_prev1


def hermite_basis_vector(x: float, degree: int) -> np.ndarray:
    vals = np.zeros(degree + 1, dtype=np.float64)
    vals[0] = 1.0
    if degree >= 1:
        vals[1] = x
    for n in range(1, degree):
        vals[n + 1] = x * vals[n] - n * vals[n - 1]
    return vals


def hermite_double_product(i: int, j: int) -> float:
    if i == j:
        return float(factorial(i))
    return 0.0


def hermite_triple_product(i: int, j: int, k: int) -> float:
    total = i + j + k
    if total % 2 != 0:
        return 0.0
    s = total // 2
    if s < i or s < j or s < k:
        return 0.0
    num = float(factorial(i) * factorial(j) * factorial(k))
    den = float(factorial(s) * factorial(s - i) * factorial(s - j) * factorial(s - k))
    return num / den






def standard_normal_cdf(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def standard_normal_cdf_inv(p: float) -> float:
    if p <= 0:
        return -1e12
    if p >= 1:
        return 1e12
    from math import log, sqrt

    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        num = (((((((2.5090809287301226727e+3 * r +
                     3.3430575583588128105e+4) * r +
                     6.7265770927008700853e+4) * r +
                     4.5921953931549871457e+4) * r +
                     1.3731693765509461125e+4) * r +
                     1.9715909503065514427e+3) * r +
                     1.3314136788658342429e+2) * r +
                     3.3871328727963666080e+0) * q
        den = (((((((5.2264952788528545610e+3 * r +
                     2.8729085735721942674e+4) * r +
                     3.9307895800092710610e+4) * r +
                     2.1213794301586595867e+4) * r +
                     5.3941960214247511077e+3) * r +
                     6.8718700749205790830e+2) * r +
                     4.2313330701600911252e+1) * r +
                     1.0)
        return num / den
    else:
        r = p if q <= 0 else 1.0 - p
        r = -log(r)
        if r <= 5.0:
            r = r - 1.6
            num = (((((((7.7454501427834140764e-4 * r +
                         2.27238449892691845833e-2) * r +
                         2.41780725177450611770e-1) * r +
                         1.27045825245236838258e+0) * r +
                         3.64784832476320460504e+0) * r +
                         5.76949722146069140550e+0) * r +
                         4.63033784615654529590e+0) * r +
                         1.42343711074968357734e+0)
            den = (((((((1.05075007164441684324e-9 * r +
                         5.47593808499534494600e-4) * r +
                         1.51986665636164571966e-2) * r +
                         1.48103976427480074590e-1) * r +
                         6.89767334985100004550e-1) * r +
                         1.67638483018380384940e+0) * r +
                         2.05319162663775882187e+0) * r +
                         1.0)
        else:
            r = sqrt(r) - 3.0
            num = (((((((2.01033439929228813265e-7 * r +
                         2.71155556874348757815e-5) * r +
                         1.24266094738807843860e-3) * r +
                         2.65321895265761230930e-2) * r +
                         2.96560571828504891230e-1) * r +
                         1.78482653991729133580e+0) * r +
                         5.46378491116411436990e+0) * r +
                         6.65790464350110377720e+0)
            den = (((((((2.04426310338993978564e-15 * r +
                         1.42151175831644588870e-7) * r +
                         1.84631831751005468180e-5) * r +
                         7.86869131145613259100e-4) * r +
                         1.48753612908506148525e-2) * r +
                         1.36929880922735805310e-1) * r +
                         5.99832206555887937690e-1) * r +
                         1.0)
        x = num / den
        return -x if q < 0 else x


def truncated_normal_sample(mu_param: float, sigma_param: float,
                             a: float, b: float,
                             n_samples: int = 1,
                             rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if sigma_param <= 0:
        raise ValueError("标准差必须为正")
    if rng is None:
        rng = np.random.default_rng(seed=42)
    alpha = (a - mu_param) / sigma_param
    beta = (b - mu_param) / sigma_param
    Phi_alpha = standard_normal_cdf(alpha)
    Phi_beta = standard_normal_cdf(beta)
    U = rng.random(n_samples)
    Z = Phi_alpha + U * (Phi_beta - Phi_alpha)

    Z = np.clip(Z, 1e-12, 1.0 - 1e-12)
    samples = mu_param + sigma_param * np.array([standard_normal_cdf_inv(z) for z in Z])

    samples = np.clip(samples, a, b)
    return samples






def pce_coefficients_from_samples(samples: np.ndarray, degree: int = 3) -> np.ndarray:
    coeffs = np.zeros(degree + 1, dtype=np.float64)
    for k in range(degree + 1):
        basis_vals = np.array([hermite_polynomial(k, xi) for xi in samples])

        numerator = np.mean(samples * basis_vals)
        denominator = hermite_double_product(k, k)
        coeffs[k] = numerator / denominator
    return coeffs


def pce_mean(coeffs: np.ndarray) -> float:
    return float(coeffs[0])


def pce_variance(coeffs: np.ndarray) -> float:
    var = 0.0
    for k in range(1, len(coeffs)):
        var += coeffs[k] ** 2 * float(factorial(k))
    return var


def pce_standard_deviation(coeffs: np.ndarray) -> float:
    return np.sqrt(pce_variance(coeffs))


def generate_hermite_quadrature_points(n_points: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from numpy.polynomial.hermite_e import hermegauss
        xi, w = hermegauss(n_points)
        return xi.astype(np.float64), w.astype(np.float64)
    except Exception:

        if n_points == 3:
            xi = np.array([-1.7320508075688772, 0.0, 1.7320508075688772])
            w = np.array([0.16666666666666666, 0.6666666666666666, 0.16666666666666666])
        elif n_points == 5:
            xi = np.array([-2.8569700138728, -1.3556261799743, 0.0,
                           1.3556261799743, 2.8569700138728])
            w = np.array([0.011257411327721, 0.11723990766176, 0.24300470558030,
                          0.11723990766176, 0.011257411327721])
        else:
            xi = np.array([0.0])
            w = np.array([1.0])
        return xi, w
