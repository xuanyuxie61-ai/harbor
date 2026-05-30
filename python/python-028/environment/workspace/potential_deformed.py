
import numpy as np
from math import exp, sqrt, pi, sin, cos



HBARC = 197.3269804
M_NUCLEON = 939.0


def woods_saxon_potential(r, V0, R, a):
    r = np.asarray(r, dtype=float)

    arg = (r - R) / a


    V = np.zeros_like(r)
    mask_pos = arg > 700
    mask_neg = arg < -700
    mask_mid = ~(mask_pos | mask_neg)
    V[mask_pos] = 0.0
    V[mask_neg] = V0
    V[mask_mid] = V0 / (1.0 + np.exp(arg[mask_mid]))
    return V


def woods_saxon_derivative(r, V0, R, a):
    r = np.asarray(r, dtype=float)
    arg = (r - R) / a
    dV = np.zeros_like(r)
    mask_pos = arg > 700
    mask_neg = arg < -700
    mask_mid = ~(mask_pos | mask_neg)
    dV[mask_pos] = 0.0
    dV[mask_neg] = 0.0
    earg = np.exp(arg[mask_mid])
    dV[mask_mid] = -(V0 / a) * earg / ((1.0 + earg) ** 2)
    return dV


def spin_orbit_potential(r, Vso0, Rso, aso, l, s, kappa=-0.5):
    if r < 1e-6:
        return 0.0
    dVdr = woods_saxon_derivative(r, 1.0, Rso, aso)


    j_plus = l + s
    j_minus = l - s

    ls_coupling = 0.5 * (j_plus * (j_plus + 1) - l * (l + 1) - s * (s + 1))

    hbar_over_mc_sq = 2.0
    return -Vso0 * hbar_over_mc_sq * (1.0 / r) * dVdr * ls_coupling


def deformed_woods_saxon(r, theta, phi, V0, R0, a, beta2, gamma,
                         beta3=0.0, beta4=0.0):


    Y20 = sqrt(5.0 / (16.0 * pi)) * (3.0 * cos(theta) ** 2 - 1.0)

    Y22_real = sqrt(15.0 / (32.0 * pi)) * sin(theta) ** 2 * cos(2.0 * phi)

    Y30 = sqrt(7.0 / (16.0 * pi)) * (5.0 * cos(theta) ** 3 - 3.0 * cos(theta))

    Y40 = sqrt(9.0 / (256.0 * pi)) * (35.0 * cos(theta) ** 4
                                        - 30.0 * cos(theta) ** 2 + 3.0)

    R_def = R0 * (1.0 + beta2 * (cos(gamma) * Y20 + sin(gamma) * Y22_real)
                  + beta3 * Y30 + beta4 * Y40)

    return woods_saxon_potential(r, V0, R_def, a)


def bilinear_interpolate_2d(x, y, x_grid, y_grid, Z):

    if x < x_grid[0] or x > x_grid[-1] or y < y_grid[0] or y > y_grid[-1]:
        raise ValueError("插值点超出网格范围")

    ix = np.searchsorted(x_grid, x) - 1
    iy = np.searchsorted(y_grid, y) - 1
    ix = max(0, min(ix, len(x_grid) - 2))
    iy = max(0, min(iy, len(y_grid) - 2))

    x0, x1 = x_grid[ix], x_grid[ix + 1]
    y0, y1 = y_grid[iy], y_grid[iy + 1]

    dx = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
    dy = (y - y0) / (y1 - y0) if y1 != y0 else 0.0

    z00 = Z[iy, ix]
    z10 = Z[iy, ix + 1]
    z01 = Z[iy + 1, ix]
    z11 = Z[iy + 1, ix + 1]

    return (z00 * (1 - dx) * (1 - dy) +
            z10 * dx * (1 - dy) +
            z01 * (1 - dx) * dy +
            z11 * dx * dy)


def build_potential_energy_surface(n_beta, n_gamma, V0, R0, a, l_max=6):
    beta_grid = np.linspace(-0.3, 0.5, n_beta)
    gamma_grid = np.linspace(0.0, pi / 3.0, n_gamma)
    energy_surface = np.zeros((n_gamma, n_beta))

    for i, gamma in enumerate(gamma_grid):
        for j, beta2 in enumerate(beta_grid):



            hbar_omega = 41.0 * (A_eff := 100) ** (-1.0 / 3.0)
            deform_correction = -beta2 ** 2 / (4.0 * pi) * hbar_omega

            avg_centrifugal = 0.0
            for l in range(l_max + 1):
                avg_centrifugal += HBARC ** 2 * l * (l + 1) / (2.0 * M_NUCLEON * R0 ** 2)
            avg_centrifugal /= (l_max + 1)

            energy_surface[i, j] = V0 + deform_correction + avg_centrifugal

    return beta_grid, gamma_grid, energy_surface


def total_single_particle_potential(r, theta, phi, l, j, params):
    V0 = params.get('V0', -50.0)
    R0 = params.get('R0', 5.0)
    a = params.get('a', 0.65)
    Vso0 = params.get('Vso0', 12.0)
    Rso = params.get('Rso', R0)
    aso = params.get('aso', a)
    beta2 = params.get('beta2', 0.0)
    gamma = params.get('gamma', 0.0)
    s = 0.5

    V_central = deformed_woods_saxon(r, theta, phi, V0, R0, a, beta2, gamma)
    V_so = spin_orbit_potential(r, Vso0, Rso, aso, l, s)



    V_cent = 0.0

    return V_central + V_so + V_cent
