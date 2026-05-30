#!/usr/bin/env python3

import numpy as np
from porous_gdl_transport import capillary_diffusivity


def compute_residuals(phi_m, lambda_profile, s_gdl, params):
    res = {}


    if phi_m.ndim == 2:
        Nx, Ny = phi_m.shape
        dx = 1.0 / (Nx - 1)
        dy = 1.0 / (Ny - 1)
        laplacian_phi = np.zeros_like(phi_m)
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                laplacian_phi[i, j] = ((phi_m[i + 1, j] - 2 * phi_m[i, j] + phi_m[i - 1, j]) / dx ** 2 +
                                        (phi_m[i, j + 1] - 2 * phi_m[i, j] + phi_m[i, j - 1]) / dy ** 2)

        S_p = np.zeros_like(phi_m)
        residual_p = laplacian_phi - S_p
        norm_phi = max(np.linalg.norm(phi_m), 1.0)
        res['proton'] = float(np.linalg.norm(residual_p) / norm_phi)
    else:
        res['proton'] = 0.0


    if lambda_profile.ndim == 1:
        Nz = len(lambda_profile)
        dz = params['t_membrane'] / (Nz - 1)




        pass
    else:
        res['water'] = 0.0


    if s_gdl.ndim == 1:
        Nz = len(s_gdl)
        dz = params['t_gdl'] / (Nz - 1)
        z = np.linspace(0.0, params['t_gdl'], Nz)

        D = capillary_diffusivity(s_gdl, params)
        flux = np.zeros(Nz)
        for i in range(Nz - 1):
            D_face = 0.5 * (D[i] + D[i + 1])
            flux[i] = D_face * (s_gdl[i + 1] - s_gdl[i]) / dz
        div_flux = np.zeros(Nz)
        for i in range(1, Nz - 1):
            div_flux[i] = (flux[i] - flux[i - 1]) / dz
        S_porous = np.zeros(Nz)
        for i in range(1, Nz - 1):
            S_porous[i] = 0.5 * (z[i] / params['t_gdl']) ** 2 * 1e-4

        denom = np.abs(div_flux) + np.abs(S_porous) + 1e-15
        rel_residual = np.abs(div_flux + S_porous) / denom
        res['porous'] = float(np.mean(rel_residual))
    else:
        res['porous'] = 0.0

    return res


def convergence_study(solver_func, params, n_grids=[21, 41, 81]):
    errors = []
    for N in n_grids:
        p_local = params.copy()
        p_local['Nx'] = N
        try:
            result = solver_func(p_local)

            errors.append(np.mean(np.abs(result)))
        except Exception:
            errors.append(np.nan)

    p_order = []
    for i in range(len(errors) - 1):
        if errors[i] > 1e-14 and errors[i + 1] > 1e-14:
            ratio = errors[i] / errors[i + 1]
            if ratio > 0:
                p_order.append(np.log(ratio) / np.log(2.0))

    return {'n_grids': n_grids, 'errors': errors, 'p_order': p_order}


def compute_mass_balance_error(lambda_profile, j_profile, params):
    Nz = len(lambda_profile)
    dz = params['t_membrane'] / (Nz - 1)
    lambda_total = np.trapezoid(lambda_profile, dx=dz)


    F = params['F']
    n_drag = 2.5
    j_avg = np.mean(j_profile)
    flux_eod = n_drag * j_avg / F


    flux_diff = -(lambda_profile[-1] - lambda_profile[0]) / params['t_membrane'] * 1e-10


    mass_balance = abs(flux_eod + flux_diff)
    return mass_balance, lambda_total


if __name__ == '__main__':
    p = {'t_membrane': 50e-6, 't_gdl': 200e-6, 'Nx': 41}
    phi = np.random.rand(41, 21)
    lam = np.linspace(3.0, 14.0, 21)
    s = np.linspace(0.05, 0.6, 21)
    res = compute_residuals(phi, lam, s, p)
    print("Residuals:", res)
