
import numpy as np


def r8vec_bracket5(nxd, xd, xq):
    xd = np.asarray(xd, dtype=float)
    if xq < xd[0] or xq > xd[-1]:
        return -1
    i = int(np.searchsorted(xd, xq, side='right')) - 1
    i = max(0, min(i, nxd - 2))
    return i


def pwl_interp_2d_scalar(nxd, nyd, xd, yd, zd, xi, yi):
    i = r8vec_bracket5(nxd, xd, xi)
    if i == -1:
        return np.inf
    j = r8vec_bracket5(nyd, yd, yi)
    if j == -1:
        return np.inf


    y_diag = yd[j + 1] + (yd[j] - yd[j + 1]) * (xi - xd[i]) / (xd[i + 1] - xd[i])

    if yi < y_diag:

        dxa = xd[i + 1] - xd[i]
        dya = yd[j] - yd[j]
        dxb = xd[i] - xd[i]
        dyb = yd[j + 1] - yd[j]
        dxi = xi - xd[i]
        dyi = yi - yd[j]

        det = dxa * dyb - dya * dxb
        if abs(det) < 1e-15:
            return np.inf

        alpha = (dxi * dyb - dyi * dxb) / det
        beta = (dxa * dyi - dya * dxi) / det
        gamma = 1.0 - alpha - beta

        zi = alpha * zd[i + 1, j] + beta * zd[i, j + 1] + gamma * zd[i, j]
    else:

        dxa = xd[i] - xd[i + 1]
        dya = yd[j + 1] - yd[j + 1]
        dxb = xd[i + 1] - xd[i + 1]
        dyb = yd[j] - yd[j + 1]
        dxi = xi - xd[i + 1]
        dyi = yi - yd[j + 1]

        det = dxa * dyb - dya * dxb
        if abs(det) < 1e-15:
            return np.inf

        alpha = (dxi * dyb - dyi * dxb) / det
        beta = (dxa * dyi - dya * dxi) / det
        gamma = 1.0 - alpha - beta

        zi = alpha * zd[i, j + 1] + beta * zd[i + 1, j] + gamma * zd[i + 1, j + 1]

    return zi


def pwl_interp_2d(nxd, nyd, xd, yd, zd, ni, xi, yi):
    zi = np.full(ni, np.inf, dtype=float)
    for k in range(ni):
        zi[k] = pwl_interp_2d_scalar(nxd, nyd, xd, yd, zd, xi[k], yi[k])
    return zi


def build_opacity_table(n_rho=16, n_T=16):
    log_rho = np.linspace(-20, -10, n_rho)
    log_T = np.linspace(4, 9, n_T)

    sigma_T = 6.6524587158e-25
    m_p = 1.6726219e-24
    kappa_es = sigma_T / m_p

    rho = 10.0 ** log_rho.reshape(-1, 1)
    T = 10.0 ** log_T.reshape(1, -1)


    kappa_ff = 0.64e23 * rho * T ** (-3.5)
    kappa_ff = np.clip(kappa_ff, 0.0, 1e4)

    kappa = kappa_es + kappa_ff
    kappa = np.clip(kappa, 1e-4, 1e4)
    return log_rho, log_T, kappa


def interpolate_opacity(rho_query, T_query, log_rho, log_T, kappa_table):
    log_rho_q = np.log10(np.clip(rho_query, 1e-30, None))
    log_T_q = np.log10(np.clip(T_query, 1.0, None))

    n = log_rho_q.size
    kappa = pwl_interp_2d(log_rho.size, log_T.size,
                          log_rho, log_T, kappa_table,
                          n, log_rho_q, log_T_q)


    kappa = np.where(np.isinf(kappa), kappa_table.mean(), kappa)
    return kappa
