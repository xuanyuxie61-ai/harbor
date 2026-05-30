# -*- coding: utf-8 -*-

import numpy as np


def strain_rate_tensor(u, v, w, dx, dy, dz):
    def ddx(f):
        result = np.zeros_like(f)
        result[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2.0 * dx)
        return result

    def ddy(f):
        result = np.zeros_like(f)
        result[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2.0 * dy)
        return result

    def ddz(f):
        result = np.zeros_like(f)
        result[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2.0 * dz)
        return result

    dudx = ddx(u)
    dudy = ddy(u)
    dudz = ddz(u)
    dvdx = ddx(v)
    dvdy = ddy(v)
    dvdz = ddz(v)
    dwdx = ddx(w)
    dwdy = ddy(w)
    dwdz = ddz(w)

    S11 = dudx
    S22 = dvdy
    S33 = dwdz
    S12 = 0.5 * (dudy + dvdx)
    S13 = 0.5 * (dudz + dwdx)
    S23 = 0.5 * (dvdz + dwdy)

    return S11, S12, S13, S22, S23, S33


def smagorinsky_model(u, v, w, dx, dy, dz, Cs=0.18):










    raise NotImplementedError("Hole 1: Smagorinsky model core formula not implemented")


def dynamic_smagorinsky(u, v, w, dx, dy, dz, Cs_test=0.18):

    def test_filter(f):
        result = np.zeros_like(f)
        result[1:-1, 1:-1, 1:-1] = 0.125 * (
            f[1:-1, 1:-1, 1:-1] + f[2:, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1]
            + f[1:-1, 2:, 1:-1] + f[1:-1, :-2, 1:-1]
            + f[1:-1, 1:-1, 2:] + f[1:-1, 1:-1, :-2]
        )
        return result


    S11, S12, S13, S22, S23, S33 = strain_rate_tensor(u, v, w, dx, dy, dz)
    Delta = (dx * dy * dz) ** (1.0 / 3.0)
    S_mag = np.sqrt(2.0 * (S11 ** 2 + S22 ** 2 + S33 ** 2
                           + 2.0 * S12 ** 2 + 2.0 * S13 ** 2 + 2.0 * S23 ** 2))
    S_mag = np.clip(S_mag, 1e-10, 1e6)


    u_test = test_filter(u)
    v_test = test_filter(v)
    w_test = test_filter(w)


    S11_t, S12_t, S13_t, S22_t, S23_t, S33_t = strain_rate_tensor(
        u_test, v_test, w_test, dx, dy, dz)
    S_mag_t = np.sqrt(2.0 * (S11_t ** 2 + S22_t ** 2 + S33_t ** 2
                             + 2.0 * S12_t ** 2 + 2.0 * S13_t ** 2 + 2.0 * S23_t ** 2))
    S_mag_t = np.clip(S_mag_t, 1e-10, 1e6)


    uu = u * u
    uv = u * v
    uw = u * w
    vv = v * v
    vw = v * w
    ww = w * w

    L11 = test_filter(uu) - u_test * u_test
    L12 = test_filter(uv) - u_test * v_test
    L13 = test_filter(uw) - u_test * w_test
    L22 = test_filter(vv) - v_test * v_test
    L23 = test_filter(vw) - v_test * w_test
    L33 = test_filter(ww) - w_test * w_test


    M11 = 4.0 * S_mag_t * S11_t - test_filter(S_mag * S11)
    M12 = 4.0 * S_mag_t * S12_t - test_filter(S_mag * S12)
    M13 = 4.0 * S_mag_t * S13_t - test_filter(S_mag * S13)
    M22 = 4.0 * S_mag_t * S22_t - test_filter(S_mag * S22)
    M23 = 4.0 * S_mag_t * S23_t - test_filter(S_mag * S23)
    M33 = 4.0 * S_mag_t * S33_t - test_filter(S_mag * S33)


    LM = (L11 * M11 + L12 * M12 + L13 * M13
          + L12 * M12 + L22 * M22 + L23 * M23
          + L13 * M13 + L23 * M23 + L33 * M33)
    MM = (M11 ** 2 + M12 ** 2 + M13 ** 2
          + M12 ** 2 + M22 ** 2 + M23 ** 2
          + M13 ** 2 + M23 ** 2 + M33 ** 2)


    MM = np.where(MM < 1e-15, 1e-15, MM)
    C_dynamic = -0.5 * LM / MM


    C_dynamic = np.clip(C_dynamic, -0.5, 0.5)

    nu_sgs = C_dynamic * Delta ** 2 * S_mag
    nu_sgs = np.clip(nu_sgs, 0.0, 10.0 * nu_sgs.max())

    return nu_sgs, C_dynamic


def ifs_turbulence_generator(n_points=5000, n_iter=1000, seed=42):
    rng = np.random.default_rng(seed)


    transforms = [

        {
            'A': np.array([[0.05, 0.0, 0.0],
                           [0.0, 0.05, 0.0],
                           [0.0, 0.0, 0.05]]),
            'b': np.array([0.5, 0.0, 0.0]),
            'scale': 1.0
        },

        {
            'A': np.array([[0.42, -0.42, 0.0],
                           [0.42, 0.42, 0.0],
                           [0.0, 0.0, 0.1]]),
            'b': np.array([0.29, -0.01, 0.0]),
            'scale': 0.5
        },

        {
            'A': np.array([[0.42, 0.42, 0.0],
                           [-0.42, 0.42, 0.0],
                           [0.0, 0.0, 0.1]]),
            'b': np.array([0.29, 0.41, 0.0]),
            'scale': 0.5
        },

        {
            'A': np.array([[0.1, 0.0, 0.0],
                           [0.0, 0.1, 0.0],
                           [0.0, 0.0, 0.1]]),
            'b': np.array([0.45, 0.15, 0.0]),
            'scale': 0.1
        }
    ]


    x = rng.random(3)
    points = []
    energies = []


    for _ in range(100):
        j = rng.integers(0, 4)
        x = transforms[j]['A'] @ x + transforms[j]['b']


    for _ in range(n_iter):
        j = rng.integers(0, 4)
        x = transforms[j]['A'] @ x + transforms[j]['b']
        points.append(x.copy())
        energies.append(transforms[j]['scale'])

    points = np.array(points)
    energies = np.array(energies)


    if len(points) > n_points:
        idx = rng.choice(len(points), size=n_points, replace=False)
        points = points[idx]
        energies = energies[idx]

    return points, energies


def structure_function_model(u, v, w, dx, dy, dz, order=2):

    du_x = u[1:, :, :] - u[:-1, :, :]
    dv_y = v[:, 1:, :] - v[:, :-1, :]
    dw_z = w[:, :, 1:] - w[:, :, :-1]


    D_ll = np.zeros_like(u)
    D_ll[1:-1, 1:-1, 1:-1] = 0.333 * (
        (du_x[1:, 1:-1, 1:-1] ** 2 + du_x[:-1, 1:-1, 1:-1] ** 2)
        + (dv_y[1:-1, 1:, 1:-1] ** 2 + dv_y[1:-1, :-1, 1:-1] ** 2)
        + (dw_z[1:-1, 1:-1, 1:] ** 2 + dw_z[1:-1, 1:-1, :-1] ** 2)
    )

    Delta = (dx * dy * dz) ** (1.0 / 3.0)
    C_SF = 1.4

    D_ll = np.clip(D_ll, 0.0, 1e6)
    nu_sgs = C_SF * Delta * np.sqrt(D_ll)

    return nu_sgs
