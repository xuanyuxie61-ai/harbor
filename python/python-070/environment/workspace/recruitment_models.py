
import numpy as np
from utils import NumericalConfig, safe_divide


def sigmoid(x):
    x = np.asarray(x, dtype=float)

    x_clip = np.clip(x, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def sigmoid_derivative_coef(n):
    if n < 0:
        raise ValueError("Derivative order n must be non-negative")

    coef = np.zeros(n + 2, dtype=float)

    for k in range(n + 1):
        cnk = 0.0
        mop = -1.0
        for j in range(k + 1):
            mop = -mop
            cnk += mop * ((j + 1) ** n) * comb(k, j)
        coef[k + 1] = cnk

    return coef


def comb(n, k):
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    k = min(k, n - k)
    result = 1.0
    for i in range(1, k + 1):
        result = result * (n - k + i) / i
    return result


def sigmoid_derivative(n, x):
    coef = sigmoid_derivative_coef(n)
    s = sigmoid(x)

    d = np.zeros_like(np.asarray(x), dtype=float)
    for j in range(1, n + 2):
        d += coef[j] * (s ** j)
    return d


def beverton_holt(S, alpha, beta):
    S = np.asarray(S, dtype=float)
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")


    S = np.maximum(S, 0.0)
    denom = 1.0 + beta * S
    return safe_divide(alpha * S, denom, 0.0)


def ricker_recruitment(S, alpha, beta):
    S = np.asarray(S, dtype=float)
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")

    S = np.maximum(S, 0.0)
    return alpha * S * np.exp(-beta * S)


def sigmoid_allee_recruitment(S, alpha, beta, S_crit, steepness=10.0):
    S = np.asarray(S, dtype=float)
    if S_crit < 0:
        raise ValueError("S_crit must be non-negative")
    if steepness <= 0:
        raise ValueError("steepness must be positive")

    R_bh = beverton_holt(S, alpha, beta)
    allee_factor = sigmoid(steepness * (S - S_crit))
    return R_bh * allee_factor


def recruitment_derivative(S, alpha, beta, S_crit, steepness, model_type='allee'):
    S = float(S)
    if S < 0:
        S = 0.0

    if model_type == 'bh':

        return alpha / ((1.0 + beta * S) ** 2)

    elif model_type == 'ricker':

        return alpha * np.exp(-beta * S) * (1.0 - beta * S)

    elif model_type == 'allee':
        R_bh = beverton_holt(S, alpha, beta)
        z = steepness * (S - S_crit)
        sigma_z = sigmoid(z)

        dsigma_dz = sigma_z * (1.0 - sigma_z)
        dRbh_dS = alpha / ((1.0 + beta * S) ** 2)
        return dRbh_dS * sigma_z + R_bh * dsigma_dz * steepness

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
