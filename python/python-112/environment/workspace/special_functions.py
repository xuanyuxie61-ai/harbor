
import numpy as np
from scipy.special import gamma, gammaln, psi as digamma, hyp2f1
from scipy.special import iv as bessel_i
import warnings




_EULER_MASCHERONI = 0.5772156649015328606065120900824024310421
_MAX_HYP2F1_ITER = 250
_EPS_SMALL = 2.05e-9
_EPS_LARGE = 1.0e-15


def r8_hyper_2f1(a: float, b: float, c: float, x: float) -> float:
    if c <= 0.0 and abs(c - round(c)) < 1.0e-14:
        raise ValueError("r8_hyper_2f1: c must not be a non-positive integer.")
    if x >= 1.0:
        raise ValueError("r8_hyper_2f1: x must be < 1.0 for convergence.")


    if x < -1.0:


        return float(hyp2f1(a, b, c, x))

    result = float(hyp2f1(a, b, c, x))
    if not np.isfinite(result):

        result = _hyp2f1_series(a, b, c, x)
    return result


def _hyp2f1_series(a: float, b: float, c: float, x: float) -> float:
    if abs(x) < 1.0e-15 or a == 0.0 or b == 0.0:
        return 1.0

    hf = 1.0
    r = 1.0
    for k in range(1, _MAX_HYP2F1_ITER + 1):
        r *= (a + k - 1.0) * (b + k - 1.0) / (k * (c + k - 1.0)) * x
        hf += r
        if abs(r) < _EPS_LARGE * abs(hf):
            break
    else:
        warnings.warn("_hyp2f1_series: reached max iterations without convergence.")
    return float(hf)


def r8_psi(x: float) -> float:
    if not np.isfinite(x):
        raise ValueError("r8_psi: x must be finite.")
    return float(digamma(x))


def gegenbauer_integral(expon: int, alpha: float) -> float:
    if alpha <= -1.0:
        raise ValueError("gegenbauer_integral: alpha must be > -1.0.")
    if expon < 0:
        raise ValueError("gegenbauer_integral: expon must be non-negative.")

    if expon % 2 == 1:
        return 0.0

    c = float(expon)
    val1 = r8_hyper_2f1(-alpha, 1.0 + c, 2.0 + alpha + c, -1.0)
    value = 2.0 * gamma(1.0 + c) * gamma(1.0 + alpha) * val1 / gamma(2.0 + alpha + c)
    return float(value)


def gegenbauer_exactness_monomial(expon: int, alpha: float, order: int,
                                  w: np.ndarray, x: np.ndarray) -> float:
    if order < 1:
        raise ValueError("gegenbauer_exactness_monomial: order >= 1 required.")
    if w.shape[0] != order or x.shape[0] != order:
        raise ValueError("gegenbauer_exactness_monomial: w and x must have length == order.")

    exact = gegenbauer_integral(expon, alpha)
    quad_val = float(np.dot(w, x ** expon))

    if exact == 0.0:
        err = abs(quad_val)
    else:
        err = abs((quad_val - exact) / exact)
    return err


def membrane_vibration_bessel(r: np.ndarray, t: float,
                              mu_n: np.ndarray, nu: float = 2.0 / 3.0) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    if np.any(r < 0.0):
        raise ValueError("membrane_vibration_bessel: r must be non-negative.")
    if t < 0.0:
        raise ValueError("membrane_vibration_bessel: t must be non-negative.")

    U = np.zeros_like(r)
    for k, mu in enumerate(mu_n, start=1):

        from scipy.special import jv as bessel_j
        term = (1.0 / np.sqrt(k)) * np.sin(mu * t) * bessel_j(nu, mu * r)
        U += term

    return U


def screened_coulomb_green(r: float, kappa: float, epsilon: float = 1.0) -> float:
    if r < 0.0:
        raise ValueError("screened_coulomb_green: r must be >= 0.")
    if kappa < 0.0:
        raise ValueError("screened_coulomb_green: kappa must be >= 0.")
    if epsilon <= 0.0:
        raise ValueError("screened_coulomb_green: epsilon must be > 0.")

    if r < 1.0e-12:
        return kappa / (4.0 * np.pi * epsilon)

    return np.exp(-kappa * r) / (4.0 * np.pi * epsilon * r)
