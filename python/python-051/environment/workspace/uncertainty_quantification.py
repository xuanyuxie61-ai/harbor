
import numpy as np
from sparse_grid_cubature import legendre_monomial_integral






def norm_loo(field):
    return np.max(np.abs(field))


def norm_loo_location(field, x_coords, z_coords):
    idx = np.unravel_index(np.argmax(np.abs(field)), field.shape)
    return np.abs(field[idx]), x_coords[idx[0]], z_coords[idx[1]]






def test_legendre_exactness_1d(points, weights, degree_max=11):
    tol = 1e-12
    results = []
    max_exact = -1
    for degree in range(degree_max + 1):
        exact = legendre_monomial_integral(degree)
        quad = np.sum(weights * (points ** degree))
        if exact == 0.0:
            err = abs(quad)
        else:
            err = abs((quad - exact) / exact)
        results.append((degree, err))
        if err < tol:
            max_exact = degree
    return max_exact, results






def compute_mean_variance(samples):
    mean = np.mean(samples)
    var = np.var(samples, ddof=1)
    return mean, var


def first_order_sobol_pce(coeffs, multi_indices, total_var, dim):
    if total_var <= 1e-30:
        return np.zeros(dim)

    S1 = np.zeros(dim)
    for d in range(dim):

        mask = (multi_indices[:, d] > 0) & (np.sum(multi_indices, axis=1) == multi_indices[:, d])
        S1[d] = np.sum(coeffs[mask] ** 2) / total_var

    return S1


def total_order_sobol_pce(coeffs, multi_indices, total_var, dim):
    if total_var <= 1e-30:
        return np.zeros(dim)

    ST = np.zeros(dim)
    for d in range(dim):
        mask = multi_indices[:, d] > 0
        ST[d] = np.sum(coeffs[mask] ** 2) / total_var

    return ST






def gci_refinement_estimator(fine, medium, coarse, r=2.0):
    if abs(fine) < 1e-30:
        fine = 1e-30
    eps = (fine - medium) / fine
    denom = medium - fine
    if abs(denom) < 1e-30:
        denom = 1e-30
    p = np.log(abs((coarse - medium) / denom)) / np.log(max(r, 1.001))
    if abs(p) < 1e-6:
        p = 1e-6
    F_s = 1.25
    rp = r ** p
    rp_diff = rp - 1.0
    if abs(rp_diff) < 1e-30:
        rp_diff = 1e-30 if rp_diff >= 0 else -1e-30
    gci = F_s * abs(eps) / rp_diff
    return p, gci


def convergence_rate(errors, resolutions):
    log_h = np.log(resolutions)
    log_e = np.log(errors)

    A = np.vstack([log_h, np.ones_like(log_h)]).T
    p, c = np.linalg.lstsq(A, log_e, rcond=None)[0]
    return p, np.exp(c)
