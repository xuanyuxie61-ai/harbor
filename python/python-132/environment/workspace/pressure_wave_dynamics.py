
import numpy as np
from utils import clip_with_warning


def fd1d_wave_solve(z_num, z1, z2, t_num, t1, t2, c, P_x1, P_x2, P_t1, Pt_t1):
    if z_num < 1:
        z_num = 1
    if t_num < 1:
        t_num = 1

    t_delta = (t2 - t1) / t_num
    z_delta = (z2 - z1) / z_num
    alpha = c * t_delta / z_delta

    if abs(alpha) > 1.0:
        print(f"[WARN] fd1d_wave: CFL condition |alpha|={abs(alpha):.4f} > 1 violated.")

        t_delta = z_delta / abs(c) * 0.95
        t_num = int(np.ceil((t2 - t1) / t_delta))
        t_delta = (t2 - t1) / t_num
        alpha = c * t_delta / z_delta
        print(f"       Adjusted to t_num={t_num}, alpha={abs(alpha):.4f}")

    P = np.zeros((t_num + 1, z_num + 1), dtype=float)


    for n in range(t_num + 1):
        t = t1 + n * t_delta
        P[n, 0] = P_x1(t)
        P[n, z_num] = P_x2(t)


    z_grid = np.linspace(z1, z2, z_num + 1)
    P[0, :] = P_t1(z_grid)
    Pt0 = Pt_t1(z_grid)


    for j in range(1, z_num):
        P[1, j] = (
            0.5 * alpha ** 2 * P[0, j + 1]
            + (1.0 - alpha ** 2) * P[0, j]
            + 0.5 * alpha ** 2 * P[0, j - 1]
            + t_delta * Pt0[j]
        )


    for n in range(1, t_num):
        for j in range(1, z_num):
            P[n + 1, j] = (
                2.0 * (1.0 - alpha ** 2) * P[n, j]
                + alpha ** 2 * (P[n, j + 1] + P[n, j - 1])
                - P[n - 1, j]
            )

    return P, alpha


def pressure_wave_in_column(column_height, c_sound, P_bottom, P_top, P_initial,
                            disturbance_z, disturbance_amp, t_end, nz=50, nt=200):

    def P_x1(t):
        return P_bottom

    def P_x2(t):
        return P_top


    def P_t1(z):
        z = np.asarray(z, dtype=float)
        sigma = column_height * 0.05
        dist = np.exp(-0.5 * ((z - disturbance_z) / sigma) ** 2)
        return P_initial + disturbance_amp * dist

    def Pt_t1(z):
        z = np.asarray(z, dtype=float)
        return np.zeros_like(z)

    P_field, alpha = fd1d_wave_solve(
        nz, 0.0, column_height, nt, 0.0, t_end, c_sound,
        P_x1, P_x2, P_t1, Pt_t1
    )

    z_grid = np.linspace(0.0, column_height, nz + 1)
    t_grid = np.linspace(0.0, t_end, nt + 1)

    return P_field, z_grid, t_grid, alpha


def pressure_stability_index(P_field):
    nt = P_field.shape[0]
    var_t = np.var(P_field, axis=1)
    if nt > 1:
        dvar = np.abs(var_t[-1] - var_t[0]) / (nt - 1)
    else:
        dvar = 0.0
    return float(dvar)
