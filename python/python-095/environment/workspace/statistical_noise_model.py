
import numpy as np
import math
from special_functions import digamma, trigamma


def dirichlet_estimate_mle(x, alpha_init=None, max_iter=300, tol=1e-8):
    import scipy.optimize as opt

    x = np.asarray(x, dtype=float)
    N, K = x.shape

    if N <= K:
        raise ValueError("dirichlet_estimate: need N > K")
    if np.any(x <= 0):
        raise ValueError("dirichlet_estimate: all x must be positive")
    row_sums = np.sum(x, axis=1)
    if np.any(np.abs(row_sums - 1.0) > 0.01):
        x = x / row_sums[:, None]

    alpha_min = 0.05


    if alpha_init is None:
        means = np.mean(x, axis=0)
        var_mean = np.mean([np.var(x[:, j]) for j in range(K)])
        if var_mean < 1e-12:
            a0 = 10.0
        else:
            avg_mean = np.mean(means)
            a0 = max(avg_mean * (1.0 - avg_mean) / var_mean - 1.0, 0.5)
        alpha0 = a0 * means
        alpha0 = np.maximum(alpha0, alpha_min)
    else:
        alpha0 = np.asarray(alpha_init, dtype=float).copy()
        alpha0 = np.maximum(alpha0, alpha_min)

    log_x = np.log(x)
    avg_log_x = np.mean(log_x, axis=0)

    def neg_loglik(alpha):
        alpha = np.maximum(alpha, alpha_min)
        a_sum = np.sum(alpha)
        ll = -math.lgamma(a_sum)
        for j in range(K):
            ll += math.lgamma(alpha[j])
            ll += (alpha[j] - 1.0) * avg_log_x[j]
        return -ll * N

    def grad(alpha):
        alpha = np.maximum(alpha, alpha_min)
        a_sum = np.sum(alpha)
        ps, _ = digamma(a_sum)
        g = np.zeros(K)
        for j in range(K):
            pa, _ = digamma(alpha[j])
            g[j] = -N * (ps - pa + avg_log_x[j])
        return g

    bounds = [(alpha_min, None) for _ in range(K)]
    res = opt.minimize(neg_loglik, alpha0, jac=grad, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': max_iter, 'gtol': tol})

    alpha = res.x
    alpha = np.maximum(alpha, alpha_min)


    alpha_sum = np.sum(alpha)
    loglik = 0.0
    for j in range(K):
        loglik += (alpha[j] - 1.0) * np.sum(log_x[:, j])
    loglik -= N * math.lgamma(alpha_sum)
    for j in range(K):
        loglik += N * math.lgamma(alpha[j])

    return alpha, res.nit, loglik


def adaptive_step_size_from_dirichlet(error_powers, base_mu=0.001):
    alpha, _, _ = dirichlet_estimate_mle(error_powers)
    alpha_sum = np.sum(alpha)
    if alpha_sum < 1e-12:
        return np.full(alpha.shape, base_mu)

    mu = base_mu * (alpha / alpha_sum) * alpha.shape[0]
    mu = np.clip(mu, base_mu * 0.1, base_mu * 5.0)
    return mu


def noise_stationarity_test(error_history, window=50):
    err = np.asarray(error_history, dtype=float)
    if len(err) < 2 * window:
        return True, 1.0

    var1 = np.var(err[:window])
    var2 = np.var(err[-window:])
    if var2 < 1e-12:
        var2 = 1e-12
    f_stat = var1 / var2


    is_stationary = 0.5 < f_stat < 2.0
    return is_stationary, f_stat
