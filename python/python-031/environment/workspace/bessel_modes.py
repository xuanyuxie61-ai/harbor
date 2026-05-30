# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import jv, jvp


def besselj_zero(n, nt):
    rj0 = np.zeros(nt)

    if n <= 20:
        x = 2.82141 + 1.15859 * n
    else:
        x = n + 1.85576 * (n ** 0.33333) + 1.03315 / (n ** 0.33333)

    l = 0
    max_iter = 1000
    for _ in range(max_iter):
        x0 = x
        bjn = jv(n, x)
        djn = jvp(n, x, 1)

        if abs(djn) < 1e-15:
            x = x + 0.1
            continue

        x = x - bjn / djn

        if abs(x - x0) > 1.0e-9:
            continue

        if x > 0:
            rj0[l] = x
            l += 1

            if l < nt:

                x = x + np.pi + (0.0972 + 0.0679 * n - 0.000354 * n**2) / l
        else:
            x = x0 + np.pi

        if l >= nt:
            break

    return rj0


def jyndd(n, x):
    from scipy.special import yv, yvp
    bjn = jv(n, x)
    djn = jvp(n, x, 1)
    byn = yv(n, x)
    dyn = yvp(n, x, 1)
    return bjn, djn, byn, dyn


def cylinder_coulomb_potential(r, R_cyl, rho_p, n_modes=20):
    r = np.asarray(r)
    alpha = besselj_zero(0, n_modes)

    Phi = np.zeros_like(r, dtype=float)
    for n in range(n_modes):
        al = alpha[n]
        if al <= 0.0:
            continue
        J1_al = jv(1, al)
        if abs(J1_al) < 1e-15:
            continue
        coeff = 1.0 / (al**3 * J1_al)
        arg = al * r / R_cyl

        mask = r <= R_cyl
        Phi[mask] += coeff * jv(0, arg[mask])

    Phi = Phi * 4.0 * np.pi * 1.43996448 * rho_p * R_cyl**2
    return Phi


def cylinder_vibration_frequencies(R_cyl, surface_tension, mass_density, n_modes=10):
    if R_cyl <= 0.0 or surface_tension <= 0.0 or mass_density <= 0.0:
        return np.array([])

    freqs = []
    for m in range(n_modes):
        alpha = besselj_zero(m, n_modes)
        for n in range(len(alpha)):
            al = alpha[n]
            if al <= m:
                continue
            omega_sq = (surface_tension / (mass_density * R_cyl**3)) * al * (al**2 - m**2)
            if omega_sq > 0:
                freqs.append(np.sqrt(omega_sq))

    return np.array(sorted(freqs))


def spherical_coulomb_potential(r, R_sphere, rho_p):
    r = np.asarray(r)
    e2 = 1.43996448
    Phi = np.zeros_like(r, dtype=float)

    mask_in = r <= R_sphere
    mask_out = r > R_sphere

    Phi[mask_in] = 2.0 * np.pi * e2 * rho_p * (R_sphere**2 - r[mask_in]**2 / 3.0)
    Phi[mask_out] = (4.0 * np.pi * e2 * rho_p * R_sphere**3) / (3.0 * r[mask_out])

    return Phi


def sheet_coulomb_potential(z, t_sheet, rho_p):
    z = np.asarray(z)
    e2 = 1.43996448
    Phi = np.zeros_like(z, dtype=float)

    mask_in = np.abs(z) <= t_sheet / 2.0
    mask_out = np.abs(z) > t_sheet / 2.0

    Phi[mask_in] = 2.0 * np.pi * e2 * rho_p * (t_sheet**2 / 4.0 - z[mask_in]**2)
    Phi[mask_out] = np.pi * e2 * rho_p * t_sheet * (t_sheet / 2.0 - np.abs(z[mask_out]))

    return Phi


def pasta_deformation_energy(phase_id, R, amplitude, mode_m, surface_tension):
    if R <= 0.0 or surface_tension <= 0.0:
        return 0.0

    eps_sq = amplitude**2
    m = mode_m

    if phase_id == 1:

        deltaE = 4.0 * np.pi * surface_tension * R**2 * eps_sq * (m - 1) * (m + 2) / 2.0
    elif phase_id == 2:

        deltaE = np.pi * surface_tension * R**2 * eps_sq * (m**2 + m - 2.0)
    else:
        deltaE = 0.0

    return deltaE


if __name__ == '__main__':

    zeros = besselj_zero(0, 5)
    print(f"J_0 zeros: {zeros}")
    Phi = cylinder_coulomb_potential(np.array([0.0, 0.5, 1.0]), 1.0, 0.01)
    print(f"Cylinder Coulomb potential: {Phi}")
