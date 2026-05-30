
import numpy as np
import math


def sphere_fibonacci_grid_points(ng, radius=1.0):
    if ng <= 0:
        raise ValueError("ng must be positive")





    raise NotImplementedError("Hole 1: sphere_fibonacci_grid_points 待实现")


def spherical_harmonic_transform_matrix(points, L_max):
    from scipy.special import sph_harm
    N = points.shape[0]
    n_coeffs = (L_max + 1) ** 2


    r = np.linalg.norm(points, axis=1)
    theta = np.arccos(np.clip(points[:, 2] / (r + 1e-12), -1.0, 1.0))
    phi = np.arctan2(points[:, 1], points[:, 0])

    Y = np.zeros((N, n_coeffs), dtype=complex)
    idx = 0
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            Y[:, idx] = sph_harm(m, l, phi, theta)
            idx += 1


    return np.real(Y)


def spherical_array_directivity(weights, points, theta_grid, phi_grid, k, radius=1.0):
    N = points.shape[0]
    weights = np.asarray(weights, dtype=complex)

    B = np.zeros((len(theta_grid), len(phi_grid)), dtype=complex)
    for ti, th in enumerate(theta_grid):
        for pi_, ph in enumerate(phi_grid):
            r_hat = np.array([np.sin(th) * np.cos(ph),
                              np.sin(th) * np.sin(ph),
                              np.cos(th)])
            phase = k * np.dot(points, r_hat)
            B[ti, pi_] = np.sum(weights * np.exp(1j * phase))

    return np.abs(B)
