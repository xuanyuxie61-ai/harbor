
import numpy as np
import math
from utils import ensure_positive, clip_with_warning, safe_divide






def jacobi_polynomial(m, n, alpha, beta, x):
    if alpha <= -1.0 or beta <= -1.0:
        raise ValueError("jacobi_polynomial: alpha and beta must be > -1")
    if n < 0:
        return np.empty((m, 0))

    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != m:
        m = x.size

    v = np.ones((m, n + 1), dtype=float)
    if n == 0:
        return v

    v[:, 1] = (1.0 + 0.5 * (alpha + beta)) * x + 0.5 * (alpha - beta)

    for i in range(2, n + 1):
        c1 = 2.0 * i * (i + alpha + beta) * (2.0 * i - 2.0 + alpha + beta)
        c2 = (2.0 * i - 1.0 + alpha + beta) * (2.0 * i + alpha + beta) * (2.0 * i - 2.0 + alpha + beta)
        c3 = (2.0 * i - 1.0 + alpha + beta) * (alpha + beta) * (alpha - beta)
        c4 = -2.0 * (i - 1.0 + alpha) * (i - 1.0 + beta) * (2.0 * i + alpha + beta)
        v[:, i] = ((c3 + c2 * x) * v[:, i - 1] + c4 * v[:, i - 2]) / c1

    return v






def laguerre_root(x, norder, alpha, b, c):
    eps = 2.220446049250313e-16
    max_iter = 100
    for _ in range(max_iter):
        p1 = 1.0
        p2 = 0.0
        for j in range(1, norder + 1):
            p3 = p2
            p2 = p1
            p1 = (x - b[j - 1]) * p2 - c[j - 1] * p3
        dp1 = norder * (p1 - p2) / x if abs(x) > eps else norder * p1
        dx = p1 / dp1
        x = x - dx
        if abs(dx) < eps:
            break
    return x, dp1, p1


def laguerre_compute(norder, alpha=0.0):
    if norder < 1:
        raise ValueError("laguerre_compute: norder must be >= 1")
    if alpha < 0.0:
        alpha = 0.0

    b = np.empty(norder, dtype=float)
    c = np.empty(norder, dtype=float)
    for i in range(1, norder + 1):
        b[i - 1] = alpha + 2.0 * i - 1.0
        c[i - 1] = (i - 1.0) * (alpha + i - 1.0)

    cc = math.gamma(alpha + 1.0) * np.prod(c[1:]) if norder > 1 else math.gamma(alpha + 1.0)

    xtab = np.empty(norder, dtype=float)
    weight = np.empty(norder, dtype=float)

    for i in range(1, norder + 1):
        if i == 1:
            x = (1.0 + alpha) * (3.0 + 0.92 * alpha) / (1.0 + 2.4 * norder + 1.8 * alpha)
        elif i == 2:
            x = xtab[0] + (15.0 + 6.25 * alpha) / (1.0 + 0.9 * alpha + 2.5 * norder)
        else:
            r1 = (1.0 + 2.55 * (i - 2)) / (1.9 * (i - 2))
            r2 = 1.26 * (i - 2) * alpha / (1.0 + 3.5 * (i - 2))
            ratio = (r1 + r2) / (1.0 + 0.3 * alpha)
            x = xtab[i - 2] + ratio * (xtab[i - 2] - xtab[i - 3])

        x, dp2, p1 = laguerre_root(x, norder, alpha, b, c)
        xtab[i - 1] = x
        weight[i - 1] = cc / dp2 / p1

    return xtab, weight


def laguerre_quadrature_integrate(f, norder=16, alpha=0.0, transform=None):
    xtab, weight = laguerre_compute(norder, alpha)
    if transform is not None:
        vals = f(transform(xtab))
    else:
        vals = f(xtab)
    vals = np.asarray(vals, dtype=float)
    return float(np.sum(weight * vals))






def antoine_vapor_pressure(T, A, B, C):
    T = float(T)
    P_mmHg = 10.0 ** (A - B / (T + C))
    P_pa = P_mmHg * 133.322
    return P_pa






def wilson_parameters(V, Lambda_ij, T):
    nc = len(V)
    T = ensure_positive(T, name="T")
    V = ensure_positive(V, name="V")
    Lambda = np.zeros((nc, nc), dtype=float)
    R = 8.314
    for i in range(nc):
        for j in range(nc):
            if i == j:
                Lambda[i, j] = 1.0
            else:
                Lambda[i, j] = (V[j] / V[i]) * np.exp(-Lambda_ij[i, j] / T)
    return Lambda


def wilson_activity_coefficient(x, V, Lambda_ij, T):
    x = np.asarray(x, dtype=float)
    x = x / np.sum(x)
    nc = len(x)
    Lambda = wilson_parameters(V, Lambda_ij, T)

    gamma = np.zeros(nc, dtype=float)
    for i in range(nc):
        sum1 = np.sum(x * Lambda[i, :])
        sum1 = ensure_positive(sum1, name="sum1_wilson")
        term1 = -np.log(sum1)
        term2 = 0.0
        for k in range(nc):
            sum2 = np.sum(x * Lambda[k, :])
            sum2 = ensure_positive(sum2, name="sum2_wilson")
            term2 += x[k] * Lambda[k, i] / sum2
        gamma[i] = np.exp(1.0 + term1 - term2)

    return gamma






def vle_flash_calculation(x, P_total, T, A_ant, B_ant, C_ant, V, Lambda_ij):
    x = np.asarray(x, dtype=float)
    x = x / np.sum(x)
    nc = len(x)
    T_celsius = T - 273.15

    P_sat = np.array([antoine_vapor_pressure(T_celsius, A_ant[i], B_ant[i], C_ant[i]) for i in range(nc)])
    gamma = wilson_activity_coefficient(x, V, Lambda_ij, T)









    K = None
    y = None

    return y, K, gamma


def vle_relative_volatility(K):
    K = np.asarray(K, dtype=float)
    K_ref = np.max(K)
    K_ref = max(K_ref, 1e-15)
    return K / K_ref






def activity_coefficient_spectral_expansion(x_range, nc, alpha_jac=0.0, beta_jac=0.0, n_modes=8):
    x_range = np.asarray(x_range, dtype=float)
    xi = 2.0 * x_range - 1.0
    xi = clip_with_warning(xi, -1.0, 1.0, "xi")

    V_jac = jacobi_polynomial(len(xi), n_modes, alpha_jac, beta_jac, xi)
    return V_jac
