#!/usr/bin/env python3

import numpy as np


def proton_potential_exact(x, y, params):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)


    num = 2.0 * (1.0 + y)
    den = (3.0 + x) ** 2 + (1.0 + y) ** 2
    U = num / den


    dnum_dx = 0.0
    dnum_dy = 2.0
    dden_dx = 2.0 * (3.0 + x)
    dden_dy = 2.0 * (1.0 + y)

    Ux = (dnum_dx * den - num * dden_dx) / (den ** 2)
    Uy = (dnum_dy * den - num * dden_dy) / (den ** 2)


    Uxx = (-num * 2.0 * den - (dnum_dx * den - num * dden_dx) * 2.0 * dden_dx) / (den ** 3)
    Uyy = (2.0 * den - num * 2.0 - dnum_dy * 2.0 * (1.0 + y)) / (den ** 2)

    Uxy = (-dnum_dy * dden_dx * den - num * 2.0 * (1.0 + y) * dden_dx) / (den ** 3)


    U = np.where(den > 1e-12, U, 0.0)
    Ux = np.where(den > 1e-12, Ux, 0.0)
    Uy = np.where(den > 1e-12, Uy, 0.0)
    Uxx = np.where(den > 1e-12, Uxx, 0.0)
    Uxy = np.where(den > 1e-12, Uxy, 0.0)
    Uyy = np.where(den > 1e-12, Uyy, 0.0)

    return U, Ux, Uy, Uxx, Uxy, Uyy


def membrane_conductivity(lambda_w, T):
    sigma_0 = 1.0
    sigma = sigma_0 * np.exp(1268.0 * (1.0 / 303.15 - 1.0 / T))
    sigma *= np.clip(0.005139 * lambda_w - 0.00326, 1e-6, 10.0)
    return sigma


def solve_proton_potential(params, lambda_field=None):
    Nx = params['Nx']
    Ny = max(5, Nx // 2)
    Lx, Ly = 1.0, 1.0
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    x_grid = np.linspace(0.0, Lx, Nx)
    y_grid = np.linspace(0.0, Ly, Ny)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')


    if lambda_field is None:
        lambda_field = np.full((Nx, Ny), params['lambda_eq'])
    else:
        lambda_field = np.clip(lambda_field, 1.0, 22.0)


    sigma = membrane_conductivity(lambda_field, params['T'])


    S_p = np.zeros((Nx, Ny))
    S_p[:, -1] = -5000.0
    S_p[:, 0] = 5000.0


    n_unknowns = Nx * Ny
    A = np.zeros((n_unknowns, n_unknowns), dtype=float)
    b = np.zeros(n_unknowns, dtype=float)

    def idx(i, j):
        return i * Ny + j

    for i in range(Nx):
        for j in range(Ny):
            k = idx(i, j)
            if i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1:

                A[k, k] = 1.0

                b[k] = proton_potential_exact(X[i, j], Y[i, j], params)[0]
            else:

                s_c = sigma[i, j]
                s_e = 0.5 * (sigma[i, j] + sigma[i + 1, j])
                s_w = 0.5 * (sigma[i, j] + sigma[i - 1, j])
                s_n = 0.5 * (sigma[i, j] + sigma[i, j + 1])
                s_s = 0.5 * (sigma[i, j] + sigma[i, j - 1])

                A[k, k] = (s_e + s_w) / dx ** 2 + (s_n + s_s) / dy ** 2
                if i + 1 < Nx:
                    A[k, idx(i + 1, j)] = -s_e / dx ** 2
                if i - 1 >= 0:
                    A[k, idx(i - 1, j)] = -s_w / dx ** 2
                if j + 1 < Ny:
                    A[k, idx(i, j + 1)] = -s_n / dy ** 2
                if j - 1 >= 0:
                    A[k, idx(i, j - 1)] = -s_s / dy ** 2
                b[k] = S_p[i, j]


    phi_vec = np.linalg.solve(A, b)
    phi = phi_vec.reshape((Nx, Ny))

    return phi, x_grid, y_grid






def hermite_cubic_value(x1, x2, f1, d1, f2, d2, n_eval, x_eval=None):
    h = x2 - x1
    if abs(h) < 1e-14:
        return np.full(n_eval, f1), np.zeros(n_eval), np.zeros(n_eval), np.zeros(n_eval)

    if x_eval is None:
        x_eval = np.linspace(x1, x2, n_eval)
    else:
        x_eval = np.clip(np.asarray(x_eval, dtype=float), x1, x2)

    t = (x_eval - x1) / h
    t2 = t * t
    t3 = t2 * t


    H1 = 2.0 * t3 - 3.0 * t2 + 1.0
    H2 = t3 - 2.0 * t2 + t
    H3 = -2.0 * t3 + 3.0 * t2
    H4 = t3 - t2

    p = f1 * H1 + d1 * h * H2 + f2 * H3 + d2 * h * H4


    dH1 = 6.0 * t2 - 6.0 * t
    dH2 = 3.0 * t2 - 4.0 * t + 1.0
    dH3 = -6.0 * t2 + 6.0 * t
    dH4 = 3.0 * t2 - 2.0 * t

    dp = (f1 * dH1 + d1 * h * dH2 + f2 * dH3 + d2 * h * dH4) / h
    d2p = (f1 * (12.0 * t - 6.0) + d1 * h * (6.0 * t - 4.0) +
           f2 * (-12.0 * t + 6.0) + d2 * h * (6.0 * t - 2.0)) / (h ** 2)
    d3p = (f1 * 12.0 + d1 * h * 6.0 + f2 * (-12.0) + d2 * h * 6.0) / (h ** 3)

    return p, dp, d2p, d3p


def hermite_cubic_spline(x_nodes, f_nodes, d_nodes, x_query):
    x_query = np.asarray(x_query, dtype=float)
    x_nodes = np.asarray(x_nodes, dtype=float)
    f_nodes = np.asarray(f_nodes, dtype=float)
    d_nodes = np.asarray(d_nodes, dtype=float)

    y_out = np.zeros_like(x_query)
    for k in range(x_query.size):
        xq = x_query[k]
        if xq <= x_nodes[0]:
            y_out[k] = f_nodes[0]
            continue
        if xq >= x_nodes[-1]:
            y_out[k] = f_nodes[-1]
            continue

        idx = np.searchsorted(x_nodes, xq) - 1
        idx = np.clip(idx, 0, len(x_nodes) - 2)
        p, _, _, _ = hermite_cubic_value(
            x_nodes[idx], x_nodes[idx + 1],
            f_nodes[idx], d_nodes[idx],
            f_nodes[idx + 1], d_nodes[idx + 1],
            1, x_eval=[xq]
        )
        y_out[k] = p[0]
    return y_out


def interpolate_proton_potential_hermite(phi, x_grid, y_grid, xq, yq):

    Ny = y_grid.size
    phi_at_y = np.zeros(Ny)
    for j in range(Ny):

        dphi_dx = np.gradient(phi[:, j], x_grid)
        phi_at_y[j] = hermite_cubic_spline(x_grid, phi[:, j], dphi_dx, [xq])[0]


    dphi_dy = np.gradient(phi_at_y, y_grid)
    phi_q = hermite_cubic_spline(y_grid, phi_at_y, dphi_dy, [yq])[0]
    return phi_q


if __name__ == '__main__':
    p = {'T': 353.15, 'lambda_eq': 14.0, 'Nx': 41, 'sigma_m_ref': 10.0}
    phi, xg, yg = solve_proton_potential(p)
    print("phi range:", phi.min(), phi.max())
