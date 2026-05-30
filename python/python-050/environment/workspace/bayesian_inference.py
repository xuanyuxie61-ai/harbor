
import numpy as np
from typing import Callable, Tuple, Optional


def digamma_asymptotic(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if np.any(x <= 0):
        raise ValueError("digamma requires positive arguments.")

    psi = np.zeros_like(x)


    for _ in range(20):
        mask = x < 6.0
        if not np.any(mask):
            break
        psi[mask] -= 1.0 / x[mask]
        x[mask] += 1.0


    inv_x = 1.0 / x
    inv_x2 = inv_x ** 2
    psi += np.log(x) - 0.5 * inv_x - inv_x2 / 12.0 + inv_x2 ** 2 / 120.0 - inv_x2 ** 3 / 252.0

    return psi


def trigamma_asymptotic(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if np.any(x <= 0):
        raise ValueError("trigamma requires positive arguments.")

    psi1 = np.zeros_like(x)

    for _ in range(20):
        mask = x < 6.0
        if not np.any(mask):
            break
        psi1[mask] += 1.0 / (x[mask] ** 2)
        x[mask] += 1.0

    inv_x = 1.0 / x
    inv_x2 = inv_x ** 2
    psi1 += inv_x + 0.5 * inv_x2 + inv_x * inv_x2 / 6.0 - inv_x2 ** 2 * inv_x / 30.0 + inv_x2 ** 3 * inv_x / 42.0

    return psi1


def gamma_log_likelihood(data: np.ndarray,
                         shape: float,
                         scale: float) -> float:
    alpha = float(shape)
    beta = float(scale)
    x = np.asarray(data, dtype=np.float64)
    x = x[x > 0]
    if len(x) == 0:
        return -1e20

    n = len(x)
    from math import lgamma, log
    logL = n * (alpha * log(beta) - lgamma(alpha))
    logL += (alpha - 1.0) * np.sum(np.log(x))
    logL -= beta * np.sum(x)
    return float(logL)


def gamma_mle_newton_raphson(data: np.ndarray,
                              alpha_init: float = 1.0,
                              tol: float = 1e-8,
                              max_iter: int = 100) -> Tuple[float, float]:
    x = np.asarray(data, dtype=np.float64)
    x = x[x > 0]
    if len(x) == 0:
        raise ValueError("No positive data for Gamma MLE.")

    log_x_bar = np.mean(np.log(x))
    x_bar = np.mean(x)
    s = np.log(x_bar) - log_x_bar

    alpha = float(alpha_init)
    for _ in range(max_iter):
        psi_a = float(digamma_asymptotic(np.array([alpha]))[0])
        psi1_a = float(trigamma_asymptotic(np.array([alpha]))[0])

        f = np.log(alpha) - psi_a - s
        fp = 1.0 / alpha - psi1_a

        if abs(fp) < 1e-15:
            break

        alpha_new = alpha - f / fp
        alpha_new = max(alpha_new, 1e-3)

        if abs(alpha_new - alpha) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new

    beta = alpha / x_bar
    return float(alpha), float(beta)


def dirichlet_log_likelihood(data: np.ndarray,
                              alpha: np.ndarray) -> float:
    X = np.asarray(data, dtype=np.float64)
    alpha_vec = np.asarray(alpha, dtype=np.float64)

    if np.any(alpha_vec <= 0):
        return -1e20


    row_sums = np.sum(X, axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-15)
    X = X / row_sums

    n, k = X.shape
    alpha0 = np.sum(alpha_vec)

    from math import lgamma
    logL = n * lgamma(alpha0) - n * np.sum([lgamma(a) for a in alpha_vec])
    logL += np.sum((alpha_vec - 1.0) * np.sum(np.log(np.maximum(X, 1e-15)), axis=0))

    return float(logL)


def dirichlet_mle_newton(data: np.ndarray,
                          alpha_init: Optional[np.ndarray] = None,
                          tol: float = 1e-6,
                          max_iter: int = 100) -> np.ndarray:
    X = np.asarray(data, dtype=np.float64)
    row_sums = np.sum(X, axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-15)
    X = X / row_sums

    n, k = X.shape
    if alpha_init is None:
        alpha = np.ones(k, dtype=np.float64) * 2.0
    else:
        alpha = np.asarray(alpha_init, dtype=np.float64).copy()

    log_data = np.sum(np.log(np.maximum(X, 1e-15)), axis=0)

    for _ in range(max_iter):
        alpha0 = np.sum(alpha)
        psi0 = float(digamma_asymptotic(np.array([alpha0]))[0])
        psi_alpha = digamma_asymptotic(alpha)
        psi1_0 = float(trigamma_asymptotic(np.array([alpha0]))[0])
        psi1_alpha = trigamma_asymptotic(alpha)

        g = n * (psi0 - psi_alpha) + log_data


        I = np.full((k, k), n * psi1_0, dtype=np.float64)
        I[np.arange(k), np.arange(k)] += n * (psi1_alpha - psi1_0)


        try:
            delta = np.linalg.solve(I, g)
        except np.linalg.LinAlgError:
            break


        step_size = 1.0
        alpha_new = alpha - step_size * delta
        alpha_new = np.maximum(alpha_new, 1e-3)


        for _ in range(10):
            new_ll = dirichlet_log_likelihood(X, alpha_new)
            old_ll = dirichlet_log_likelihood(X, alpha)
            if new_ll >= old_ll or step_size < 0.01:
                break
            step_size *= 0.5
            alpha_new = np.maximum(alpha - step_size * delta, 1e-3)

        if np.linalg.norm(alpha_new - alpha) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new

    return alpha


def metropolis_hastings_posterior(log_posterior: Callable[[np.ndarray], float],
                                  theta_init: np.ndarray,
                                  proposal_std: np.ndarray,
                                  n_samples: int = 10000,
                                  burn_in: int = 2000) -> np.ndarray:
    theta = np.asarray(theta_init, dtype=np.float64)
    proposal_std = np.asarray(proposal_std, dtype=np.float64)
    dim = len(theta)

    samples = []
    current_log_p = log_posterior(theta)
    if not np.isfinite(current_log_p):
        current_log_p = -1e20

    n_accepted = 0
    rng = np.random.default_rng(42)

    for i in range(n_samples):
        proposal = theta + proposal_std * rng.standard_normal(dim)
        proposal_log_p = log_posterior(proposal)
        if not np.isfinite(proposal_log_p):
            proposal_log_p = -1e20

        log_alpha = proposal_log_p - current_log_p
        if np.log(rng.random()) < log_alpha:
            theta = proposal
            current_log_p = proposal_log_p
            n_accepted += 1

        if i >= burn_in:
            samples.append(theta.copy())

    acceptance_rate = n_accepted / n_samples

    return np.array(samples, dtype=np.float64)


def gamma_sample(alpha: float, beta: float, size: int = 1, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.gamma(shape=alpha, scale=1.0 / beta, size=size)
