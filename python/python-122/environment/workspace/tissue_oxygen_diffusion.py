
import numpy as np


def oxygen_diffusion_ftcs_1d(C0, nx, nt, t_max, D, lam, k_met, C_max,
                              bc_left_type='dirichlet', bc_left_val=1.0,
                              bc_right_type='neumann', bc_right_val=0.0):
    xmin = 0.0
    xmax = 1.0
    x = np.linspace(xmin, xmax, nx)
    dx = (xmax - xmin) / (nx - 1)
    dt = t_max / nt


    if D * dt / dx ** 2 > 0.5:

        dt = 0.45 * dx ** 2 / D
        nt = int(np.ceil(t_max / dt))
        dt = t_max / nt

    C = np.zeros((nt + 1, nx))
    c = C0(x).astype(float)
    C[0, :] = c


    Im1 = np.array([0] + list(range(nx - 2)) + [nx - 2])
    I = np.arange(nx)
    Ip1 = np.array([1] + list(range(1, nx - 1)) + [nx - 1])

    for j in range(1, nt + 1):
        d2c_dx2 = (c[Ip1] - 2.0 * c[I] + c[Im1]) / dx ** 2
        reaction = lam * c * (1.0 - c / C_max) - k_met * c
        c_new = c + dt * (D * d2c_dx2 + reaction)


        if bc_left_type == 'dirichlet':
            c_new[0] = bc_left_val
        elif bc_left_type == 'neumann':
            c_new[0] = c_new[1] - bc_left_val * dx

        if bc_right_type == 'dirichlet':
            c_new[-1] = bc_right_val
        elif bc_right_type == 'neumann':
            c_new[-1] = c_new[-2] + bc_right_val * dx

        c = np.clip(c_new, 0.0, C_max)
        C[j, :] = c

    t = np.linspace(0, t_max, nt + 1)
    return C, x, t


def oxygen_diffusion_2d_radial(C0, nr, nt, t_max, D, lam, k_met, C_max,
                                R_tissue=0.05, R_cap=0.003):
    r = np.linspace(R_cap, R_tissue, nr)
    dr = (R_tissue - R_cap) / (nr - 1)
    dt = t_max / nt

    if D * dt / dr ** 2 > 0.25:
        dt = 0.2 * dr ** 2 / D
        nt = int(np.ceil(t_max / dt))
        dt = t_max / nt

    C = np.zeros((nt + 1, nr))
    c = C0(r).astype(float)
    C[0, :] = c

    for j in range(1, nt + 1):
        c_new = np.zeros_like(c)
        for i in range(1, nr - 1):
            r_i = max(r[i], 1e-10)
            laplacian_r = (c[i + 1] - 2.0 * c[i] + c[i - 1]) / dr ** 2 + \
                          (c[i + 1] - c[i - 1]) / (2.0 * dr * r_i)
            reaction = lam * c[i] * (1.0 - c[i] / C_max) - k_met * c[i]
            c_new[i] = c[i] + dt * (D * laplacian_r + reaction)


        c_new[0] = C_max * 0.95

        c_new[-1] = c_new[-2]

        c = np.clip(c_new, 0.0, C_max)
        C[j, :] = c

    t = np.linspace(0, t_max, nt + 1)
    return C, r, t


def michaelis_menten_oxygen_consumption(C, V_max, K_m):
    C = np.asarray(C, dtype=float)
    C_safe = np.where(C < 0, 0.0, C)
    return V_max * C_safe / (K_m + C_safe + 1e-14)


def krogh_oxygen_tension(r, R_t, R_c, P_c, P_tissue, D_t, M0):
    r = np.asarray(r, dtype=float)
    r_safe = np.where(r < R_c, R_c, r)
    term1 = (M0 / (4.0 * D_t)) * (r_safe ** 2 - R_c ** 2)
    term2 = (M0 * R_t ** 2 / (2.0 * D_t)) * np.log(r_safe / R_c)
    P = P_c - term1 + term2
    return P
