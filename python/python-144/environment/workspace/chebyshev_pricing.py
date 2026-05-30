
import numpy as np
from scipy.special import gamma as scipy_gamma


def chebyshev_grid(n: int) -> np.ndarray:
    if n < 0:
        raise ValueError("chebyshev_grid: n 必须为非负整数。")
    if n == 0:
        return np.array([1.0])
    return np.cos(np.pi * np.arange(n + 1) / n)


def chebyshev_diff_matrix(n: int) -> np.ndarray:
    if n < 0:
        raise ValueError("chebyshev_diff_matrix: n 必须为非负整数。")
    if n == 0:
        return np.zeros((1, 1))
    x = chebyshev_grid(n)
    c = np.ones(n + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** np.arange(n + 1))
    X = np.tile(x[:, np.newaxis], (1, n + 1))
    dX = X - X.T

    D = (c[:, np.newaxis] / c[np.newaxis, :]) / (dX + np.eye(n + 1))

    D = D - np.diag(np.sum(D, axis=1))
    return D


def chebyshev_barycentric_interpolate(x_grid: np.ndarray, v: np.ndarray,
                                       x_query: np.ndarray) -> np.ndarray:
    n = len(x_grid) - 1
    w = np.ones(n + 1) * ((-1.0) ** np.arange(n + 1))
    w[0] = 0.5
    w[-1] = 0.5 * ((-1.0) ** n)


    x_query = np.asarray(x_query).reshape(-1)
    result = np.zeros_like(x_query, dtype=float)
    for i, xq in enumerate(x_query):
        exact = np.isclose(xq, x_grid)
        if np.any(exact):
            result[i] = v[np.argmax(exact)]
            continue
        weights = w / (xq - x_grid)
        result[i] = np.dot(weights, v) / np.sum(weights)
    return result


def spectral_var_cvar(returns: np.ndarray, alpha: float = 0.05,
                       n_cheb: int = 64) -> dict:
    if not (0.0 < alpha < 1.0):
        raise ValueError("spectral_var_cvar: alpha 必须在 (0, 1) 区间内。")
    if returns.size < 10:
        raise ValueError("spectral_var_cvar: 样本量不足（至少10个）。")

    r_min = np.min(returns)
    r_max = np.max(returns)

    margin = 0.1 * max(abs(r_max), abs(r_min), 1e-6)
    r_lo = r_min - margin
    r_hi = r_max + margin


    def to_std(r):
        return (2.0 * r - (r_hi + r_lo)) / (r_hi - r_lo)

    def from_std(x):
        return 0.5 * ((r_hi - r_lo) * x + (r_hi + r_lo))


    x_nodes = chebyshev_grid(n_cheb)
    r_nodes = from_std(x_nodes)


    cdf_nodes = np.array([np.mean(returns <= r) for r in r_nodes])

    cdf_nodes = np.maximum.accumulate(np.minimum.accumulate(cdf_nodes))
    cdf_nodes = np.clip(cdf_nodes, 0.0, 1.0)


    D = chebyshev_diff_matrix(n_cheb)

    pdf_nodes = D @ cdf_nodes * (2.0 / (r_hi - r_lo))
    pdf_nodes = np.maximum(pdf_nodes, 0.0)



    x_left, x_right = -1.0, 1.0
    for _ in range(60):
        x_mid = 0.5 * (x_left + x_right)
        f_mid = chebyshev_barycentric_interpolate(x_nodes, cdf_nodes,
                                                   np.array([x_mid]))[0]
        if f_mid < alpha:
            x_left = x_mid
        else:
            x_right = x_mid
        if x_right - x_left < 1e-14:
            break
    x_var = 0.5 * (x_left + x_right)
    var_val = from_std(x_var)



    tail_mask = r_nodes <= var_val
    cvar_val = None
    if not np.any(tail_mask):
        cvar_val = var_val
    else:

        x_tail = x_nodes[tail_mask]
        r_tail = r_nodes[tail_mask]
        pdf_tail = pdf_nodes[tail_mask]

        order = np.argsort(x_tail)
        x_tail = x_tail[order]
        r_tail = r_tail[order]
        pdf_tail = pdf_tail[order]

        integrand = r_tail * pdf_tail * ((r_hi - r_lo) / 2.0)
        integral = np.trapezoid(integrand, x_tail)

        cdf_at_var = chebyshev_barycentric_interpolate(
            x_nodes, cdf_nodes, np.array([x_var]))[0]
        if cdf_at_var < 1e-12:
            cvar_val = var_val
        else:
            cvar_val = integral / cdf_at_var


    tail_returns = returns[returns <= var_val]
    if len(tail_returns) == 0:
        empirical_cvar = var_val
    else:
        empirical_cvar = np.mean(tail_returns)
    if (cvar_val is None or not np.isfinite(cvar_val)
            or abs(cvar_val) > 10 * max(abs(var_val), 1e-6)
            or abs(cvar_val) < 1e-12):
        cvar_val = empirical_cvar

    return {
        "VaR": float(var_val),
        "CVaR": float(cvar_val),
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "alpha": alpha,
        "spectral_nodes": r_nodes,
        "cdf_values": cdf_nodes,
        "pdf_values": pdf_nodes,
    }


def circle01_monomial_integral(e: np.ndarray) -> float:
    if np.any(e < 0):
        raise ValueError("circle01_monomial_integral: 指数必须为非负整数。")
    if np.any(e % 2 == 1):
        return 0.0
    val = 2.0
    for i in range(2):
        val *= scipy_gamma(0.5 * (e[i] + 1))
    val /= scipy_gamma(0.5 * np.sum(e + 1))
    return float(val)
