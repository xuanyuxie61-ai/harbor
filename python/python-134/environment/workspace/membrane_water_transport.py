#!/usr/bin/env python3

import numpy as np






def water_content_wave_exact(z, t, params):
    L_m = params['t_membrane']
    c_w = 1.0e-3

    z = np.asarray(z, dtype=float)
    t = float(t)
    u = np.sin(np.pi * z / L_m) * np.cos(c_w * np.pi * t / L_m)
    ut = -np.sin(np.pi * z / L_m) * (c_w * np.pi / L_m) * np.sin(c_w * np.pi * t / L_m)
    utt = -np.sin(np.pi * z / L_m) * (c_w * np.pi / L_m) ** 2 * np.cos(c_w * np.pi * t / L_m)
    uz = (np.pi / L_m) * np.cos(np.pi * z / L_m) * np.cos(c_w * np.pi * t / L_m)
    uzz = -(np.pi / L_m) ** 2 * np.sin(np.pi * z / L_m) * np.cos(c_w * np.pi * t / L_m)


    z = np.clip(z, 0.0, L_m)
    return u, ut, utt, uz, uzz


def water_transport_residual(z, t, params):
    u, ut, utt, uz, uzz = water_content_wave_exact(z, t, params)
    c_w = 1.0e-3
    r = utt - c_w ** 2 * uzz
    return r






def water_content_stiff_ode(t, lambda0, params):
    Lambda = 50.0
    mu = lambda0 - Lambda ** 2 / (Lambda ** 2 + 1.0)

    t = np.asarray(t, dtype=float)
    lam = (Lambda * np.sin(t) / (Lambda ** 2 + 1.0) +
           Lambda ** 2 * np.cos(t) / (Lambda ** 2 + 1.0) +
           mu * np.exp(-Lambda * t))

    lam = np.clip(lam, 0.0, 22.0)
    return lam


def water_content_stiff_deriv(t, lam, params):
    Lambda = 50.0
    dlam = Lambda * (np.cos(t) - lam)
    return dlam






def water_diffusivity(lambda_w, T):



    pass


def electro_osmotic_drag_coeff(lambda_w):
    return 2.5 * np.clip(lambda_w, 0.0, 22.0) / 22.0


def vapor_source(lambda_w, T, params):
    lambda_eq = params['lambda_eq']
    k_vap = 1.0e-2
    return k_vap * (lambda_w - lambda_eq)


def solve_membrane_water_transport(params, j_profile=None):
    Nz = max(21, params['Nx'] // 4)
    Nt = params['Nt']
    t_final = params['t_final']
    T = params['T']
    F = params['F']
    L_m = params['t_membrane']

    dz = L_m / (Nz - 1)
    dt = t_final / Nt

    z = np.linspace(0.0, L_m, Nz)
    t = np.linspace(0.0, t_final, Nt + 1)


    lambda_w = np.full(Nz, params['lambda_eq'])


    if j_profile is None:
        j_profile = np.linspace(5000.0, 10000.0, Nz)
    j_profile = np.clip(j_profile, 0.0, 50000.0)


    lambda_history = np.zeros((Nt + 1, Nz))
    lambda_history[0, :] = lambda_w


    for n in range(Nt):
        lam_old = lambda_history[n, :].copy()


        lam_new = lam_old.copy()
        for _ in range(5):

            D = water_diffusivity(lam_new, T)
            n_d = electro_osmotic_drag_coeff(lam_new)


            a = np.zeros(Nz)
            b = np.zeros(Nz)
            c = np.zeros(Nz)
            rhs = np.zeros(Nz)

            for i in range(Nz):
                if i == 0:

                    b[i] = 1.0
                    rhs[i] = 3.0
                elif i == Nz - 1:

                    b[i] = 1.0
                    rhs[i] = 14.0
                else:
                    D_e = 0.5 * (D[i] + D[i + 1])
                    D_w = 0.5 * (D[i] + D[i - 1])

                    a[i] = -dt * D_w / dz ** 2
                    c[i] = -dt * D_e / dz ** 2
                    b[i] = 1.0 + dt * (D_w + D_e) / dz ** 2 + dt * 0.01
                    rhs[i] = lam_old[i] + dt * (
                        n_d[i] * j_profile[i] / F / 1e4 - vapor_source(lam_new[i], T, params)
                    )


            lam_new = thomas_algorithm(a, b, c, rhs)
            lam_new = np.clip(lam_new, 0.0, 22.0)

        lambda_history[n + 1, :] = lam_new


    return lambda_history[-1, :], t


def thomas_algorithm(a, b, c, d):
    n = d.size
    cp = c.copy()
    dp = d.copy()
    bp = b.copy()


    for i in range(1, n):
        w = a[i] / bp[i - 1]
        bp[i] -= w * cp[i - 1]
        dp[i] -= w * dp[i - 1]


    x = np.zeros(n)
    x[-1] = dp[-1] / bp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (dp[i] - cp[i] * x[i + 1]) / bp[i]
    return x


if __name__ == '__main__':
    p = {'t_membrane': 50e-6, 'T': 353.15, 'F': 96485.0,
         'lambda_eq': 14.0, 'Nx': 81, 'Nt': 200, 't_final': 5.0}
    lam, t = solve_membrane_water_transport(p)
    print("lambda range:", lam.min(), lam.max())
