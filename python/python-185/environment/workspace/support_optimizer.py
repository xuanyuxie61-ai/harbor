
import numpy as np
from typing import List, Tuple, Optional, Callable


def subset_sum_backtrack(s: float, values: np.ndarray,
                         more: bool = False,
                         u: Optional[np.ndarray] = None,
                         t: int = 0) -> Tuple[bool, np.ndarray, int]:
    values = np.asarray(values, dtype=float)
    n = len(values)

    if not more:
        t = 0
        u = np.zeros(n, dtype=int)
    else:
        more = False
        if t > 0:
            u[t - 1] = 0

        told = t
        t = -1
        for i in range(told - 1, 0, -1):
            if u[i - 1] == 1:
                t = i
                break

        if t < 1:
            return False, u, 0

        u[t - 1] = 0
        t = t + 1
        u[t - 1] = 1

    while True:
        su = float(np.dot(u, values))

        if su < s and t < n:
            t = t + 1
            u[t - 1] = 1
        elif abs(su - s) < 1e-10 * max(1.0, abs(s)):
            more = True
            return more, u, t
        else:
            u[t - 1] = 0
            told = t
            t = -1
            for i in range(told - 1, 0, -1):
                if u[i - 1] == 1:
                    t = i
                    break
            if t < 1:
                return False, u, 0
            u[t - 1] = 0
            t = t + 1
            u[t - 1] = 1


def regula_falsi(f: Callable, a: float, b: float,
                 tol: float = 1e-8, max_iter: int = 100) -> Tuple[float, int]:
    fa = f(a)
    fb = f(b)

    if np.sign(fa) == np.sign(fb):
        raise ValueError(f"f(a)={fa:.3e} 和 f(b)={fb:.3e} 同号，假位法要求异号")

    it = 0
    while abs(b - a) > tol and it < max_iter:

        if abs(fb - fa) < 1e-20:
            break
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        it += 1

        if np.sign(fc) == np.sign(fa):
            a = c
            fa = fc
        else:
            b = c
            fb = fc

    return 0.5 * (a + b), it


def optimize_threshold_for_sparsity(coefficients: np.ndarray,
                                    target_sparsity: int,
                                    lambda_min: float = 1e-6,
                                    lambda_max: float = 1.0) -> float:
    coeffs = np.asarray(coefficients, dtype=float)

    def g(lam: float) -> float:
        thresholded = np.sign(coeffs) * np.maximum(np.abs(coeffs) - lam, 0.0)
        support_size = np.count_nonzero(np.abs(thresholded) > 1e-10)
        return float(support_size - target_sparsity)


    g_min = g(lambda_min)
    g_max = g(lambda_max)


    while np.sign(g_min) == np.sign(g_max) and lambda_max < 1e6:
        lambda_max *= 2.0
        g_max = g(lambda_max)

    if np.sign(g_min) == np.sign(g_max):

        if abs(g_min) < abs(g_max):
            return lambda_min
        else:
            return lambda_max

    try:
        lam_opt, _ = regula_falsi(g, lambda_min, lambda_max, tol=1e-4, max_iter=50)
    except ValueError:

        lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), 100)
        best_lam = lambda_min
        best_err = abs(g(lambda_min))
        for lam in lambdas:
            err = abs(g(lam))
            if err < best_err:
                best_err = err
                best_lam = lam
        lam_opt = best_lam

    return max(lambda_min, min(lambda_max, lam_opt))


def backtracking_support_recovery(correlations: np.ndarray,
                                  target_energy: float,
                                  max_support_size: int = 100) -> np.ndarray:
    correlations = np.asarray(correlations, dtype=float)
    n = len(correlations)

    if n == 0:
        return np.array([], dtype=int)


    sorted_idx = np.argsort(-correlations)
    sorted_vals = correlations[sorted_idx]


    cumulative = np.cumsum(sorted_vals)
    cutoff = np.searchsorted(cumulative, target_energy, side='right') + 1
    cutoff = min(cutoff, n, max_support_size)

    support = sorted_idx[:cutoff]


    current_energy = np.sum(correlations[support])
    while current_energy > target_energy * 1.05 and len(support) > 1:

        min_idx_in_support = np.argmin(correlations[support])
        support = np.delete(support, min_idx_in_support)
        current_energy = np.sum(correlations[support])

    return np.sort(support)


def refined_support_reconstruction(A: np.ndarray, y: np.ndarray,
                                   target_sparsity: int,
                                   Psi: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    from cs_detector import orthogonal_matching_pursuit

    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()


    x_omp, support_omp = orthogonal_matching_pursuit(A, y, target_sparsity)


    residual = y - A @ x_omp
    correlations = np.abs(A.T @ residual)


    lambda_opt = optimize_threshold_for_sparsity(x_omp, target_sparsity)


    x_thresholded = np.sign(x_omp) * np.maximum(np.abs(x_omp) - lambda_opt, 0.0)
    support = np.where(np.abs(x_thresholded) > 1e-10)[0]


    if len(support) > 0:
        A_support = A[:, support]
        try:
            x_support, _, _, _ = np.linalg.lstsq(A_support, y, rcond=None)
        except np.linalg.LinAlgError:
            x_support = np.zeros(len(support))
        x_recon = np.zeros(A.shape[1], dtype=float)
        x_recon[support] = x_support
    else:
        x_recon = x_thresholded

    return x_recon, support
