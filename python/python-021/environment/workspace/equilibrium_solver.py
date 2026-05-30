
import numpy as np
from parameters import (
    MU0, R0, a_minor, B0, KAPPA, DELTA,
    NR_EQUIL, NTHETA_EQUIL
)


def miller_boundary(theta, R0=R0, a=a_minor, kappa=KAPPA, delta=DELTA):
    theta = np.asarray(theta, dtype=float)
    R = R0 + a * np.cos(theta + delta * np.sin(theta))
    Z = kappa * a * np.sin(theta)
    return R, Z


def pressure_profile(psi_norm, p0=1.0e5, alpha_p=2.0, beta_p=1.5):
    psi_norm = np.clip(np.asarray(psi_norm, dtype=float), 0.0, 1.0)
    val = 1.0 - np.power(psi_norm, alpha_p)
    val = np.maximum(val, 0.0)
    return p0 * np.power(val, beta_p)


def f_profile(psi_norm, R0=R0, B0=B0, epsilon=a_minor / R0,
              beta_pol=0.5, alpha_I=2.0, beta_I=1.5):




    raise NotImplementedError("此处需补全 f_profile 的科学公式实现")


def gs_operator(psi, R_grid, Z_grid):
    nr, nz = psi.shape
    if nr < 3 or nz < 3:
        raise ValueError("网格维度必须至少为 3×3")
    dR = R_grid[1] - R_grid[0]
    dZ = Z_grid[1] - Z_grid[0]
    if dR <= 0 or dZ <= 0:
        raise ValueError("网格间距必须为正")

    residual = np.zeros_like(psi)
    R = R_grid[:, np.newaxis]


    for i in range(1, nr - 1):
        for j in range(1, nz - 1):
            dpsi_dR = (psi[i + 1, j] - psi[i - 1, j]) / (2.0 * dR)
            d2psi_dR2 = (psi[i + 1, j] - 2.0 * psi[i, j] + psi[i - 1, j]) / (dR ** 2)
            d2psi_dZ2 = (psi[i, j + 1] - 2.0 * psi[i, j] + psi[i, j - 1]) / (dZ ** 2)


            residual[i, j] = d2psi_dR2 - (1.0 / R[i, 0]) * dpsi_dR + d2psi_dZ2

    return residual


def solve_grad_shafranov(max_iter=500, tol=1e-8, relaxation=0.3,
                         nr=NR_EQUIL, nz=NTHETA_EQUIL):

    R_min = R0 - 1.2 * a_minor
    R_max = R0 + 1.2 * a_minor
    Z_min = -1.2 * KAPPA * a_minor
    Z_max = 1.2 * KAPPA * a_minor

    R_grid = np.linspace(R_min, R_max, nr)
    Z_grid = np.linspace(Z_min, Z_max, nz)
    R, Z = np.meshgrid(R_grid, Z_grid, indexing='ij')


    psi = np.zeros((nr, nz), dtype=float)
    R_axis = R0
    Z_axis = 0.0
    for i in range(nr):
        for j in range(nz):
            dist = np.sqrt(((R[i, j] - R_axis) / a_minor) ** 2 +
                           ((Z[i, j] - Z_axis) / (KAPPA * a_minor)) ** 2)
            psi[i, j] = max(0.0, 1.0 - dist ** 2)


    psi[0, :] = 0.0
    psi[-1, :] = 0.0
    psi[:, 0] = 0.0
    psi[:, -1] = 0.0

    dR = R_grid[1] - R_grid[0]
    dZ = Z_grid[1] - Z_grid[0]

    for it in range(max_iter):
        psi_old = psi.copy()


        psi_min = psi.min()
        psi_max = psi.max()
        if psi_max - psi_min < 1e-14:
            raise RuntimeError("磁通范围过小，数值发散")
        psi_norm = (psi - psi_min) / (psi_max - psi_min)
        psi_norm = np.clip(psi_norm, 0.0, 1.0)


        p = pressure_profile(psi_norm)
        F = f_profile(psi_norm)



        dp_dpsi = np.zeros_like(p)
        dF2_dpsi = np.zeros_like(F)
        for i in range(1, nr - 1):
            for j in range(1, nz - 1):
                dp_dpsi[i, j] = (p[i + 1, j] - p[i - 1, j]) / (psi_old[i + 1, j] - psi_old[i - 1, j] + 1e-20)
                dF2_dpsi[i, j] = (F[i + 1, j] ** 2 - F[i - 1, j] ** 2) / (psi_old[i + 1, j] - psi_old[i - 1, j] + 1e-20)

        source = -MU0 * R ** 2 * dp_dpsi - 0.5 * dF2_dpsi


        psi_new = psi_old.copy()
        for i in range(1, nr - 1):
            for j in range(1, nz - 1):

                term = (
                    (psi_old[i + 1, j] + psi_new[i - 1, j]) / (dR ** 2)
                    + (psi_old[i, j + 1] + psi_new[i, j - 1]) / (dZ ** 2)
                    - source[i, j]
                )


                denom = 2.0 / (dR ** 2) + 2.0 / (dZ ** 2)
                psi_new[i, j] = term / denom


        psi = relaxation * psi_new + (1.0 - relaxation) * psi_old


        psi[0, :] = 0.0
        psi[-1, :] = 0.0
        psi[:, 0] = 0.0
        psi[:, -1] = 0.0

        err = np.max(np.abs(psi - psi_old))
        if err < tol:
            break
    else:

        pass




    dpsi_dr = np.zeros(nr)
    r_mid = (nr // 2)
    for i in range(1, nr - 1):
        dpsi_dr[i] = (psi[i, nz // 2] - psi[i - 1, nz // 2]) / dR
    q_profile = np.zeros(nr)
    for i in range(1, nr - 1):
        R_loc = R_grid[i]
        B_theta = np.abs(dpsi_dr[i]) / (R_loc + 1e-10)
        q_profile[i] = (R_loc * B0 / (R0 + 1e-10)) / (B_theta + 1e-10)

    info = {
        "iterations": it + 1,
        "final_error": err,
        "R_grid": R_grid,
        "Z_grid": Z_grid,
        "q_profile": q_profile,
        "psi_axis": psi_max,
        "psi_edge": psi_min,
    }
    return psi, R_grid, Z_grid, info


def compute_magnetic_field(psi, R_grid, Z_grid):
    nr, nz = psi.shape
    dR = R_grid[1] - R_grid[0]
    dZ = Z_grid[1] - Z_grid[0]
    R, Z = np.meshgrid(R_grid, Z_grid, indexing='ij')

    B_R = np.zeros_like(psi)
    B_Z = np.zeros_like(psi)
    psi_norm = (psi - psi.min()) / (psi.max() - psi.min() + 1e-20)
    F = f_profile(psi_norm)
    B_phi = F / (R + 1e-20)

    for i in range(1, nr - 1):
        for j in range(1, nz - 1):
            dpsi_dR = (psi[i + 1, j] - psi[i - 1, j]) / (2.0 * dR)
            dpsi_dZ = (psi[i, j + 1] - psi[i, j - 1]) / (2.0 * dZ)
            B_R[i, j] = -dpsi_dZ / (R[i, j] + 1e-20)
            B_Z[i, j] = dpsi_dR / (R[i, j] + 1e-20)

    return B_R, B_Z, B_phi
