#!/usr/bin/env python3

import numpy as np






def porous_medium_exact(x, t, m, params):
    x = np.asarray(x, dtype=float)
    t = float(t)
    if t <= 0:
        return np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)

    alpha = 1.0 / (m + 1.0)
    beta = 1.0 / (2.0 * (m + 1.0))
    gamma = (m - 1.0) / (2.0 * m * (m + 1.0))
    C = 1.0

    xi = x * t ** (-beta)
    factor = C - gamma * xi ** 2

    u = np.where(factor > 0, t ** (-alpha) * factor ** (1.0 / (m - 1.0)), 0.0)
    ut = np.zeros_like(x)
    ux = np.zeros_like(x)
    uxx = np.zeros_like(x)

    mask = factor > 1e-12
    if np.any(mask):
        fm = factor[mask]
        u_m = u[mask]
        ut[mask] = (-alpha * u_m / t +
                    t ** (-alpha) * (1.0 / (m - 1.0)) * fm ** (1.0 / (m - 1.0) - 1.0)
                    * (-gamma) * 2.0 * xi[mask] * (-beta) * xi[mask] / t)
        ux[mask] = (t ** (-alpha) * (1.0 / (m - 1.0)) * fm ** (1.0 / (m - 1.0) - 1.0)
                    * (-2.0 * gamma * xi[mask]) * t ** (-beta))

        uxx[mask] = t ** (-alpha - 2.0 * beta) * (
            (1.0 / (m - 1.0)) * (1.0 / (m - 1.0) - 1.0) * fm ** (1.0 / (m - 1.0) - 2.0)
            * (2.0 * gamma * xi[mask]) ** 2
            + (1.0 / (m - 1.0)) * fm ** (1.0 / (m - 1.0) - 1.0) * (-2.0 * gamma)
        )

    return u, ut, ux, uxx


def porous_medium_residual(x, t, m, params):
    u, ut, ux, uxx = porous_medium_exact(x, t, m, params)

    um = u ** m
    umx = m * u ** (m - 1) * ux
    umxx = m * (m - 1) * u ** (m - 2) * ux ** 2 + m * u ** (m - 1) * uxx
    r = ut - umxx
    return r






def capillary_diffusivity(s, params):
    s = np.clip(np.asarray(s, dtype=float), 1e-4, 1.0 - 1e-4)


    k_rl = s ** 3.0


    u = 1.0 - s
    dJ_ds = -1.417 + 4.240 * u - 3.789 * u ** 2


    K_abs = 1.0e-12
    mu_l = 3.5e-4
    sigma = 0.062
    theta = 110.0 * np.pi / 180.0
    eps = params['epsilon_gdl']


    dPc_ds = sigma * np.cos(theta) * np.sqrt(eps / K_abs) * dJ_ds

    D_cap = (K_abs * k_rl / mu_l) * np.abs(dPc_ds)
    return np.clip(D_cap, 1e-15, 1.0)


def solve_gdl_saturation(params, s_init=None):
    Nz = max(21, params['Nx'] // 4)
    L_gdl = params['t_gdl']

    dz = L_gdl / (Nz - 1)
    z = np.linspace(0.0, L_gdl, Nz)

    if s_init is None:
        s = np.full(Nz, 0.1)
    else:
        s = np.clip(s_init, 0.0, 1.0)


    D_max_est = 1.0e-6
    dt_stable = 0.4 * dz ** 2 / D_max_est
    t_final = params['t_final']
    Nt = max(int(t_final / dt_stable) + 1, 500)
    dt = t_final / Nt


    for n in range(Nt):
        s_old = s.copy()
        D = capillary_diffusivity(s_old, params)

        s_new = s_old.copy()
        for i in range(1, Nz - 1):
            D_e = 0.5 * (D[i] + D[i + 1])
            D_w = 0.5 * (D[i] + D[i - 1])

            flux_e = D_e * (s_old[i + 1] - s_old[i]) / dz
            flux_w = D_w * (s_old[i] - s_old[i - 1]) / dz


            S_w = 0.5 * (z[i] / L_gdl) ** 2 * 1e-4

            s_new[i] = s_old[i] + dt * ((flux_e - flux_w) / dz + S_w)
            s_new[i] = np.clip(s_new[i], 0.0, 1.0)


        s_new[0] = 0.05
        s_new[-1] = 0.6

        s = s_new

    return s, z


def gdl_saturation_profile_barenblatt(z, t, params):
    m = params['m_porous']
    u, ut, ux, uxx = porous_medium_exact(z, t, m, params)

    u_max = np.max(u) if np.max(u) > 1e-12 else 1.0
    s = np.clip(u / u_max, 0.0, 1.0)
    return s


if __name__ == '__main__':
    p = {'epsilon_gdl': 0.4, 't_gdl': 200e-6, 'Nx': 81, 't_final': 2.0, 'm_porous': 2.5}
    s, z = solve_gdl_saturation(p)
    print("s range:", s.min(), s.max())
