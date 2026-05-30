# -*- coding: utf-8 -*-

import numpy as np
import math
from random_tools import randomized_svd_approx, hutchinson_trace_estimator, random_probe_vector
from special_functions import lambert_w_fast
from utils import condition_number_estimate






def estimate_spectral_density(matvec, n, n_samples=100, n_bins=40, seed=None):
    rank = min(n_samples, n)
    U, lam = randomized_svd_approx(matvec, n, rank, power_iterations=2, seed=seed)


    hist, bin_edges = np.histogram(lam, bins=n_bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return lam, bin_centers, hist


def estimate_condition_number_randomized(matvec, n, seed=None):
    rank = min(30, n)
    _, lam = randomized_svd_approx(matvec, n, rank, power_iterations=3, seed=seed)
    if len(lam) < 2:
        return 1.0, 1.0, 1.0
    lam_max = lam[0]
    lam_min = max(lam[-1], 1e-15)
    return lam_max / lam_min, lam_max, lam_min






def theoretical_cg_error_bound(kappa, k):
    if kappa <= 1.0 or k < 0:
        return 0.0
    rho = (math.sqrt(kappa) - 1.0) / (math.sqrt(kappa) + 1.0)
    return 2.0 * (rho ** k)


def theoretical_cg_iteration_count(kappa, epsilon=1e-10):
    if kappa <= 1.0:
        return 1
    return int(math.ceil(0.5 * math.sqrt(kappa) * math.log(2.0 / epsilon)))


def lambert_w_refined_convergence_bound(kappa, k):
    if kappa <= 1.0 or k < 0:
        return 0.0
    rho = (math.sqrt(kappa) - 1.0) / (math.sqrt(kappa) + 1.0)
    arg = -2.0 / math.sqrt(kappa)
    if arg >= -1.0 / math.e:
        w_val, _ = lambert_w_fast(arg)
        correction = 1.0 + w_val / (2.0 * math.sqrt(kappa))
    else:
        correction = 1.0
    return 2.0 * (rho ** (k * correction))






def preconditioner_quality(A, precond_apply, seed=None):
    n = A.shape[0]

    def matvec_MA(v):
        return precond_apply(A @ v)

    kappa, lam_max, lam_min = estimate_condition_number_randomized(matvec_MA, n, seed=seed)
    return kappa, lam_max, lam_min


def eigenvalue_clustering_measure(eigenvalues, threshold=0.1):
    ev = np.sort(np.asarray(eigenvalues, dtype=float))
    if len(ev) < 2:
        return 0.0
    gaps = np.diff(ev) / (np.abs(ev[:-1]) + 1e-30)
    clustered = np.sum(gaps < threshold)
    return clustered / len(gaps)






def compare_solvers(results_dict):
    lines = []
    lines.append("=" * 70)
    lines.append("  Solver Comparison Report")
    lines.append("=" * 70)
    for name, info in results_dict.items():
        it = info.get('iterations', -1)
        res = info.get('final_residual', -1)
        conv = "YES" if info.get('converged', False) else "NO"
        lines.append(f"  {name:20s} | Iter: {it:4d} | Final Res: {res:.3e} | Converged: {conv}")
    lines.append("=" * 70)
    return "\n".join(lines)






def estimate_a_norm_error(info, A_matvec, x_iter, x_exact=None):
    if x_exact is not None:
        e = x_iter - x_exact
        return math.sqrt(float(e @ A_matvec(e)))



    res_hist = info.get('residual_history', [])
    if len(res_hist) < 2:
        return 1.0


    return math.sqrt(res_hist[-1] * res_hist[-2]) if len(res_hist) >= 2 else res_hist[-1]
