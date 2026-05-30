
import numpy as np
from math import factorial


def zernike_radial(n, m, rho):
    if n < 0 or m < 0 or (n - m) % 2 != 0:
        raise ValueError("Invalid (n,m): must satisfy n>=0, m>=0, n-m even.")
    if np.any(rho < 0) or np.any(rho > 1):
        rho = np.clip(rho, 0.0, 1.0)
    R = np.zeros_like(rho, dtype=np.float64)
    for k in range((n - m) // 2 + 1):
        coeff = ((-1) ** k) * factorial(n - k) / (
            factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)
        )
        R += coeff * (rho ** (n - 2 * k))
    return R


def zernike_polynomial(n, m, rho, theta):
    R = zernike_radial(n, abs(m), rho)
    if m > 0:
        return R * np.cos(m * theta)
    elif m < 0:
        return R * np.sin(abs(m) * theta)
    else:
        return R


def noll_to_nm(j):
    if j < 1:
        raise ValueError("Noll index j must be >= 1.")
    n = int(np.ceil((-3 + np.sqrt(1 + 8 * j)) / 2))
    m = 2 * j - n * (n + 2)
    if m > 0 and (n % 4 in [0, 1]):
        m = -m
    elif m < 0 and (n % 4 in [2, 3]):
        m = -m
    return n, m


def compute_zernike_basis(grid_size, max_noll):




    raise NotImplementedError("Hole 1: 请实现 compute_zernike_basis 函数体.")


def zernike_decompose(phase, mask, basis_flat):
    if phase.shape != mask.shape:
        raise ValueError("phase and mask must have the same shape.")
    phase_vec = phase[mask]
    Z = basis_flat[mask.ravel(), :]
    if Z.shape[0] < Z.shape[1]:
        raise ValueError("Insufficient data points for Zernike decomposition.")
    coeffs, _, _, _ = np.linalg.lstsq(Z, phase_vec, rcond=None)
    return coeffs


def zernike_reconstruct(coeffs, basis_flat, mask):
    phase_flat = basis_flat @ coeffs
    phase = np.zeros(mask.shape, dtype=np.float64)
    phase[mask] = phase_flat[mask.ravel()]
    return phase


def kolmogorov_zernike_covariance(max_noll, D_r0):
    num_modes = max_noll
    cov = np.zeros((num_modes, num_modes), dtype=np.float64)
    for j in range(1, num_modes + 1):
        n, m = noll_to_nm(j)
        var_j = (n + 1.0) ** (-1.0) * (D_r0 ** (5.0 / 3.0))
        if n == 0:
            var_j *= 1.0299
        elif n == 1:
            var_j *= 0.582
        elif n == 2:
            var_j *= 0.134
        else:
            var_j *= 0.134 * (2.0 / (n + 1.0)) ** (5.0 / 3.0)
        cov[j - 1, j - 1] = max(var_j, 1e-20)
    return cov


def simplex_lattice_enum(N, T):
    if N < 1:
        raise ValueError("N must be >= 1.")
    if T < 0:
        raise ValueError("T must be >= 0.")

    results = []
    x = np.zeros(N, dtype=int)
    x[0] = T
    results.append(x.copy())

    while x[-1] < T:
        j = N - 2
        while j >= 0 and x[j] == 0:
            j -= 1
        if j < 0:
            break
        x[j] -= 1
        s = np.sum(x[:j + 1])
        x[j + 1] = T - s
        for k in range(j + 2, N):
            x[k] = 0
        results.append(x.copy())

    return np.array(results, dtype=int)


def zernike_coefficient_simplex_search(phase_target, basis_flat, mask, T_max=5):
    if T_max < 0:
        raise ValueError("T_max must be non-negative.")
    phase_vec = phase_target[mask]
    Z = basis_flat[mask.ravel(), :]
    norm_bound = T_max * np.max(np.abs(phase_vec))
    N = Z.shape[1]

    lattice = simplex_lattice_enum(N, T_max)
    best_err = np.inf
    best_c = None

    for pt in lattice:
        c = pt.astype(np.float64) / max(T_max, 1) * norm_bound
        pred = Z @ c
        err = np.linalg.norm(pred - phase_vec)
        if err < best_err:
            best_err = err
            best_c = c.copy()

    if best_c is None:
        best_c = np.zeros(N, dtype=np.float64)

    return best_c, best_err
