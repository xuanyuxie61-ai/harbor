
import numpy as np
from monte_carlo_sampler import alnorm


def relative_l2_error(approx, exact):
    approx = np.asarray(approx, dtype=float)
    exact = np.asarray(exact, dtype=float)
    denom = np.linalg.norm(exact)
    if denom < 1e-15:
        denom = 1.0
    return np.linalg.norm(approx - exact) / denom


def relative_inf_error(approx, exact):
    approx = np.asarray(approx, dtype=float)
    exact = np.asarray(exact, dtype=float)
    denom = np.max(np.abs(exact))
    if denom < 1e-15:
        denom = 1.0
    return np.max(np.abs(approx - exact)) / denom


def convergence_order(errors, parameters):
    errors = np.asarray(errors)
    parameters = np.asarray(parameters)
    if len(errors) != len(parameters):
        raise ValueError("长度不匹配")
    p = []
    for i in range(len(errors) - 1):
        if errors[i+1] < 1e-15 or errors[i] < 1e-15:
            p.append(0.0)
            continue
        ratio_e = errors[i] / errors[i+1]
        ratio_h = parameters[i] / parameters[i+1]
        if ratio_h <= 0:
            p.append(0.0)
        else:
            p.append(np.log(ratio_e) / np.log(ratio_h))
    return np.array(p)


def estimate_truncation_order(fmm_error, direct_error, expansion_orders):
    fmm_error = np.asarray(fmm_error)
    expansion_orders = np.asarray(expansion_orders)

    adjusted = np.maximum(fmm_error - direct_error, 1e-16)
    log_err = np.log(adjusted)

    L = expansion_orders.astype(float)
    n = len(L)
    if n < 2:
        return {"rate": 0.0, "predicted_errors": fmm_error}
    A = np.vstack([np.ones(n), L]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, log_err, rcond=None)
    b = coeffs[1]
    a = coeffs[0]
    predicted = np.exp(a + b * L) + direct_error
    return {
        "rate": float(-b),
        "intercept": float(a),
        "predicted_errors": predicted,
        "adjusted_errors": adjusted
    }


def kolmogorov_smirnov_statistic(samples, mu=0.0, sigma=1.0):
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    sorted_samples = np.sort(samples)
    empirical = np.arange(1, n + 1) / n
    theoretical = np.array([alnorm((x - mu) / sigma, upper=False) for x in sorted_samples])
    diff1 = np.abs(empirical - theoretical)
    diff2 = np.abs(np.arange(0, n) / n - theoretical)
    return float(np.max(np.concatenate([diff1, diff2])))


def markov_chain_steady_state(transition_matrix, tol=1e-10, max_iter=10000):
    T = np.asarray(transition_matrix, dtype=float)
    n = T.shape[0]
    if T.shape != (n, n):
        raise ValueError("必须是方阵")

    row_sums = np.sum(T, axis=1)
    if np.any(np.abs(row_sums - 1.0) > 1e-6):
        raise ValueError("转移矩阵必须是行随机矩阵")

    pi = np.ones(n) / n
    for _ in range(max_iter):
        pi_new = pi @ T
        if np.linalg.norm(pi_new - pi, ord=1) < tol:
            return pi_new
        pi = pi_new
    return pi


def second_eigenvalue_rate(transition_matrix):
    T = np.asarray(transition_matrix, dtype=float)
    eigenvalues = np.linalg.eigvals(T)
    eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
    if len(eigenvalues) < 2:
        return 0.0
    return float(eigenvalues[1])


def fmm_error_budget(n_particles, expansion_order, separation_param=2.0, machine_eps=2.2e-16):
    L = expansion_order
    s = separation_param

    C1 = 1.0
    C2 = 0.5
    C3 = 1.0
    e_trunc = C1 * np.exp(-0.5 * L)
    e_trans = C2 * (1.0 / s) ** (L + 1)
    e_round = C3 * n_particles * machine_eps
    e_total = np.sqrt(e_trunc**2 + e_trans**2 + e_round**2)
    return {
        "truncation_error": float(e_trunc),
        "translation_error": float(e_trans),
        "roundoff_error": float(e_round),
        "total_error_estimate": float(e_total)
    }
