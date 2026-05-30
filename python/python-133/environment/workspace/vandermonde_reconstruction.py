
import numpy as np
from typing import Tuple, Optional


def vandermonde_matrix_1d(x: np.ndarray, n: int, scale: Optional[float] = None) -> np.ndarray:
    x = np.asarray(x).flatten()
    if scale is None:
        scale = np.max(np.abs(x))
        if scale < 1.0e-12:
            scale = 1.0

    V = np.zeros((x.size, n))
    for j in range(n):
        V[:, j] = (x / scale) ** j
    return V, scale


def vandermonde_interp_coef(x: np.ndarray, y: np.ndarray,
                            use_chebyshev: bool = False,
                            a: float = 0.0, b: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n = x.size

    if use_chebyshev:

        k = np.arange(1, n + 1)
        x_cheb = np.cos((2.0 * k - 1.0) * np.pi / (2.0 * n))

        x_use = 0.5 * (a + b) + 0.5 * (b - a) * x_cheb

        y_use = np.interp(x_use, np.sort(x), y[np.argsort(x)])
    else:
        x_use = x.copy()
        y_use = y.copy()

    V, scale = vandermonde_matrix_1d(x_use, n)


    cond_v = np.linalg.cond(V)
    if cond_v > 1.0e12 or n > 20:

        n_reduced = min(n, 15)
        V_red, scale = vandermonde_matrix_1d(x_use, n_reduced)
        c, residuals, rank, s = np.linalg.lstsq(V_red, y_use, rcond=None)

        c = np.pad(c, (0, n - n_reduced), mode='constant')
    else:
        c = np.linalg.solve(V, y_use)

    return c, x_use, scale


def polyval_horner(c: np.ndarray, x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    x = np.asarray(x).flatten()
    z = x / scale
    n = c.size
    p = c[-1]
    for i in range(n - 2, -1, -1):
        p = p * z + c[i]
    return p


def reconstruct_mwd_curve(molecular_weights: np.ndarray,
                          mass_fractions: np.ndarray,
                          n_interp: int = 200,
                          log_scale: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mw = np.asarray(molecular_weights, dtype=float)
    wf = np.asarray(mass_fractions, dtype=float)


    idx = np.argsort(mw)
    mw = mw[idx]
    wf = wf[idx]


    wf = np.maximum(wf, 0.0)
    sum_wf = np.sum(wf)
    if sum_wf > 1.0e-15:
        wf /= sum_wf

    if log_scale:
        x_data = np.log10(mw)
    else:
        x_data = mw.copy()


    x_min, x_max = x_data[0], x_data[-1]


    c, _, scale = vandermonde_interp_coef(x_data, wf, use_chebyshev=True,
                                          a=x_min, b=x_max)


    x_interp = np.linspace(x_min, x_max, n_interp)
    w_interp = polyval_horner(c, x_interp, scale=scale)


    w_interp = np.maximum(w_interp, 0.0)
    integral = np.trapezoid(w_interp, x_interp)
    if integral > 1.0e-15:
        w_interp /= integral

    if log_scale:
        M_interp = 10.0 ** x_interp
    else:
        M_interp = x_interp

    return M_interp, w_interp, c


def derivative_mwd_curve(c: np.ndarray,
                         molecular_weights: np.ndarray,
                         scale: float = 1.0,
                         log_scale: bool = True) -> np.ndarray:
    x = np.log10(molecular_weights) if log_scale else molecular_weights
    x = np.asarray(x)

    n = c.size
    if n <= 1:
        return np.zeros_like(x)

    c_deriv = np.zeros(n - 1)
    for k in range(1, n):
        c_deriv[k - 1] = k * c[k] / scale

    dw_dx = polyval_horner(c_deriv, x, scale=1.0)

    if log_scale:
        M = np.asarray(molecular_weights)
        M = np.maximum(M, 1.0e-12)
        dw_dM = dw_dx / (M * np.log(10.0))
    else:
        dw_dM = dw_dx

    return dw_dM


def monomial_moments_from_coeffs(c: np.ndarray,
                                 scale: float,
                                 max_moment: int = 3,
                                 log_scale: bool = True) -> np.ndarray:

    n = c.size
    x_grid = np.linspace(-2.0, 6.0, 500)
    w_grid = polyval_horner(c, x_grid, scale=scale)
    w_grid = np.maximum(w_grid, 0.0)

    moments = np.zeros(max_moment + 1)
    for m in range(max_moment + 1):
        M_grid = 10.0 ** x_grid
        integrand = (M_grid ** m) * w_grid * M_grid * np.log(10.0)
        moments[m] = np.trapezoid(integrand, x_grid)


    if moments[0] > 1.0e-15:
        moments /= moments[0]

        moments[0] = 1.0

    return moments
