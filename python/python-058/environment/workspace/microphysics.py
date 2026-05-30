
import numpy as np
from typing import Tuple, List


def laguerre_polynomials(order: int, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    shape = x.shape
    x = x.flatten()
    npts = len(x)
    L = np.zeros((order + 1, npts))
    L[0, :] = 1.0
    if order >= 1:
        L[1, :] = 1.0 - x
    for n in range(1, order):
        L[n+1, :] = ((2.0 * n + 1.0 - x) * L[n, :] - n * L[n-1, :]) / (n + 1.0)
    return L.reshape((order + 1,) + shape)


def gauss_laguerre_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n >= 1 required")
    if n == 1:
        return np.array([1.0]), np.array([1.0])


    k = np.arange(1, n + 1)
    x_init = np.pi**2 * (k - 0.25)**2 / (4.0 * n)
    x = x_init.copy()


    for _ in range(50):
        L = laguerre_polynomials(n, x)

        dL = -np.sum(L[:-1, :], axis=0)
        dx = L[n, :] / (dL + 1e-30)
        x -= dx
        if np.max(np.abs(dx)) < 1e-14:
            break


    Ln_1 = laguerre_polynomials(n - 1, x)[n - 1, :]
    w = 1.0 / (n * Ln_1**2 + 1e-30)

    x = np.where(x > 0, x, 1e-10)
    w = np.where(w > 0, w, 0.0)
    return x, w


def laguerre_exponential_product(order: int, b: float, n_quad: int = 32) -> np.ndarray:
    x, w = gauss_laguerre_nodes_weights(n_quad)
    L = laguerre_polynomials(order, x)
    T = np.zeros((order + 1, order + 1))
    for i in range(order + 1):
        for j in range(i, order + 1):
            val = np.sum(w * np.exp(b * x) * L[i, :] * L[j, :])
            T[i, j] = val
            T[j, i] = val
    return T


def log_gamma_for_microphysics(x: float) -> float:
    if x <= 0.0:
        return -np.inf
    if x < 7.0:
        f = 1.0
        y = x
        while y < 7.0:
            f *= y
            y += 1.0
        return log_gamma_for_microphysics(y) - np.log(f)
    z = 1.0 / x**2
    s = (1.0 / 12.0 - z * (1.0 / 360.0 - z * (1.0 / 1260.0 - z * (1.0 / 1680.0)))) / x
    return (x - 0.5) * np.log(x) - x + 0.5 * np.log(2.0 * np.pi) + s


def gamma_distribution_pdf(D: np.ndarray, N0: float, mu: float, lam: float) -> np.ndarray:
    D = np.asarray(D)
    D = np.where(D > 0, D, 1e-10)
    return N0 * (D**mu) * np.exp(-lam * D)


def gamma_moment(k: int, N0: float, mu: float, lam: float) -> float:
    if lam <= 0.0:
        return 0.0
    log_moment = np.log(N0 + 1e-30) + log_gamma_for_microphysics(k + mu + 1.0) - (k + mu + 1.0) * np.log(lam)
    return np.exp(log_moment)


class StochasticMicrophysics:

    def __init__(self, chaos_order: int = 4, n_quad: int = 16):
        self.order = chaos_order
        self.n_quad = n_quad

        self.xi_nodes, self.xi_weights = gauss_laguerre_nodes_weights(n_quad)

        self.L_poly = laguerre_polynomials(chaos_order, self.xi_nodes)

    def project_to_chaos(self, f_values: np.ndarray) -> np.ndarray:
        coeffs = np.zeros(self.order + 1)
        for i in range(self.order + 1):
            coeffs[i] = np.sum(self.xi_weights * f_values * self.L_poly[i, :])
        return coeffs

    def evaluate_from_chaos(self, coeffs: np.ndarray, xi: float) -> float:
        L = laguerre_polynomials(self.order, np.array([xi]))[:, 0]
        return float(np.dot(coeffs, L))

    def condensate_rate_ensemble(self, qv: float, qvs: float,
                                  tau_mean: float = 300.0,
                                  tau_std: float = 60.0) -> np.ndarray:


        tau_vals = tau_mean * np.exp(-self.xi_nodes * tau_std / tau_mean)
        tau_vals = np.where(tau_vals > 1.0, tau_vals, 1.0)

        if qv > qvs:
            rates = (qv - qvs) / tau_vals
        else:
            rates = np.zeros_like(tau_vals)

        return self.project_to_chaos(rates)

    def precipitation_rate(self, ql: float, N0_mean: float = 8e6,
                           mu: float = 2.0, lam: float = 3.0) -> float:









        raise NotImplementedError("HOLE 2: precipitation_rate 的 Z-R 关系尚未实现")


    def stochastic_precipitation(self, ql_field: np.ndarray,
                                  xi_sample: float = 0.0) -> np.ndarray:
        result = np.zeros_like(ql_field)
        for idx in np.ndindex(ql_field.shape):

            pert = 1.0 + 0.1 * xi_sample * np.sin(np.pi * idx[0] / max(1, ql_field.shape[0] - 1))
            result[idx] = self.precipitation_rate(ql_field[idx] * pert)
        return result
