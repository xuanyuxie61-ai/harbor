
import numpy as np


def magic3():
    return np.array([
        [8, 1, 6],
        [3, 5, 7],
        [4, 9, 2]
    ], dtype=float)


def magicsquare(n):
    if n % 2 == 1:
        M = np.zeros((n, n), dtype=int)
        i, j = 0, n // 2
        for k in range(1, n * n + 1):
            M[i, j] = k
            ni = (i - 1) % n
            nj = (j + 1) % n
            if M[ni, nj] != 0:
                i = (i + 1) % n
            else:
                i, j = ni, nj
        return M.astype(float)
    elif n % 4 == 0:
        M = np.arange(1, n * n + 1).reshape(n, n)
        I = np.arange(n)
        J = np.arange(n)
        mask = ((I[:, None] % 4 == J[None, :] % 4)
                | ((I[:, None] + J[None, :]) % 4 == 3))
        M[mask] = n * n + 1 - M[mask]
        return M.astype(float)
    else:

        return magicsquare(n - 1)


def anisotropic_diffusion_tensor(D_perp, D_para, pitch_angle):
    b = np.array([0.0, np.sin(pitch_angle), np.cos(pitch_angle)], dtype=float)
    delta = np.eye(3, dtype=float)
    D = D_perp * delta + (D_para - D_perp) * np.outer(b, b)
    return D


def magic_anisotropic_field(n_r=16, D_perp_base=1e20, D_para_base=1e24):
    r = np.linspace(0.0, 1e13, n_r)
    M = magicsquare(3)
    M_norm = M / np.sum(M)

    D_tensors = np.zeros((n_r, 3, 3), dtype=float)
    for i in range(n_r):

        modulation = 1.0 + 0.5 * np.sin(2 * np.pi * r[i] / 1e13)
        D_perp = D_perp_base * modulation
        D_para = D_para_base * modulation * (1.0 + M_norm[i % 3, 0])
        psi = np.arctan2(r[i], 1e13)
        D_tensors[i] = anisotropic_diffusion_tensor(D_perp, D_para, psi)

    return r, D_tensors
