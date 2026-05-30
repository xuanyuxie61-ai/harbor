
import numpy as np


def grid_2d(x_num, x_lo, x_hi, y_num, y_lo, y_hi):
    x = np.zeros((x_num, y_num))
    y = np.zeros((x_num, y_num))

    if x_num == 1:
        x[:, :] = (x_lo + x_hi) / 2.0
    else:
        for i in range(x_num):
            xi = ((x_num - 1 - i) * x_lo + i * x_hi) / (x_num - 1)
            x[i, :] = xi

    if y_num == 1:
        y[:, :] = (y_lo + y_hi) / 2.0
    else:
        for j in range(y_num):
            yi = ((y_num - 1 - j) * y_lo + j * y_hi) / (y_num - 1)
            y[:, j] = yi

    return x, y


def phi_stream(z, c):
    return (1.0 - np.cos(c * np.pi * z)) * (1.0 - z) ** 2


def dphi_stream(z, c):
    term1 = c * np.pi * np.sin(c * np.pi * z) * (1.0 - z) ** 2
    term2 = (1.0 - np.cos(c * np.pi * z)) * 2.0 * (1.0 - z)
    return term1 - term2


def uv_spiral(n, x, y, c):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    phi_x = phi_stream(x, c)
    phi_y = phi_stream(y, c)
    dphi_x = dphi_stream(x, c)
    dphi_y = dphi_stream(y, c)

    u = 10.0 * phi_x * dphi_y
    v = -10.0 * dphi_x * phi_y
    return u, v


def lorentz_factor(vx, vy, vz, c_light=2.99792458e10):
    v2 = vx ** 2 + vy ** 2 + vz ** 2

    beta2 = np.clip(v2 / c_light ** 2, 0.0, 1.0 - 1e-12)
    return 1.0 / np.sqrt(1.0 - beta2)


def relativistic_continuity_residual(rho, vr, vz, r, z, Gamma):
    flux_r = r * Gamma * rho * vr
    flux_z = Gamma * rho * vz

    dr = r[1, 0] - r[0, 0] if r.shape[0] > 1 else 1.0
    dz = z[0, 1] - z[0, 0] if z.shape[1] > 1 else 1.0

    dfr_dr = np.zeros_like(flux_r)
    dfz_dz = np.zeros_like(flux_z)


    if flux_r.shape[0] > 2:
        dfr_dr[1:-1, :] = (flux_r[2:, :] - flux_r[:-2, :]) / (2.0 * dr)
        dfr_dr[0, :] = (flux_r[1, :] - flux_r[0, :]) / dr
        dfr_dr[-1, :] = (flux_r[-1, :] - flux_r[-2, :]) / dr

    if flux_z.shape[1] > 2:
        dfz_dz[:, 1:-1] = (flux_z[:, 2:] - flux_z[:, :-2]) / (2.0 * dz)
        dfz_dz[:, 0] = (flux_z[:, 1] - flux_z[:, 0]) / dz
        dfz_dz[:, -1] = (flux_z[:, -1] - flux_z[:, -2]) / dz


    r_safe = np.where(r > 1e-12, r, 1e-12)
    residual = dfr_dr / r_safe + dfz_dz
    return residual


def compute_jet_profiles(n_r=32, n_z=64, r_max=1e13, z_max=1e15,
                         c_param=1.0, rho_0=1e-24, Gamma_0=300.0):
    r, z = grid_2d(n_r, 0.0, r_max, n_z, 0.0, z_max)


    x_norm = r / r_max
    y_norm = z / z_max


    n_pts = n_r * n_z
    x_flat = x_norm.reshape(n_pts)
    y_flat = y_norm.reshape(n_pts)
    u_flat, v_flat = uv_spiral(n_pts, x_flat, y_flat, c_param)

    u = u_flat.reshape(n_r, n_z)
    v = v_flat.reshape(n_r, n_z)



    vr = u * 1e9
    vz = 0.9 * 2.99792458e10 + v * 1e7


    c_light = 2.99792458e10
    vz = np.clip(vz, 0.0, 0.99 * c_light)

    Gamma = lorentz_factor(vr, np.zeros_like(vr), vz)


    rho = rho_0 * (1.0 + 10.0 * np.exp(-(r / (0.1 * r_max)) ** 2)) * (z_max / (z + 1e10))
    rho = np.clip(rho, 1e-30, 1e-18)

    residual = relativistic_continuity_residual(rho, vr, vz, r, z, Gamma)

    return {
        "r": r,
        "z": z,
        "rho": rho,
        "vr": vr,
        "vz": vz,
        "Gamma": Gamma,
        "residual": residual,
    }
