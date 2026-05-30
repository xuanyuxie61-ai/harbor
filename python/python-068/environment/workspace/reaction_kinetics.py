
import numpy as np


def selkov_infection_rate(
    S,
    I,
    a: float = 0.08,
    b: float = 0.6,
    c: float = 1.0
):
    S = np.clip(np.asarray(S, dtype=float), 0.0, 1e4)
    I = np.clip(np.asarray(I, dtype=float), 0.0, 1e4)

    SI2 = np.clip(S * I ** 2, 0.0, 1e8)
    numerator = a * I + SI2
    denominator = 1.0 + S ** 2
    return np.clip(c * numerator / denominator, 0.0, 1e6)


def selkov_growth_rate(
    N: float,
    K: float,
    a: float = 0.5,
    b: float = 0.1
) -> float:
    if K <= 0:
        return 0.0
    n = N / K
    if n < 0:
        return 0.0

    allee = max(n - a, -1.0)
    denom = max(1.0 - b * n, 0.1)
    return n * (1.0 - n) * allee / denom


def compute_reaction_terms(
    state: np.ndarray,
    K_field: np.ndarray,
    r_field: np.ndarray,
    params: dict
) -> np.ndarray:
    S1, I1, R1, S2, I2, R2 = state
    N1 = S1 + I1 + R1
    N2 = S2 + I2 + R2

    D_s1 = params.get('D_s1', 0.01)
    D_s2 = params.get('D_s2', 0.01)
    beta11 = params.get('beta11', 0.3)
    beta12 = params.get('beta12', 0.1)
    beta21 = params.get('beta21', 0.1)
    beta22 = params.get('beta22', 0.3)
    gamma1 = params.get('gamma1', 0.1)
    gamma2 = params.get('gamma2', 0.1)
    mu1 = params.get('mu1', 0.02)
    mu2 = params.get('mu2', 0.02)
    alpha12 = params.get('alpha12', 0.5)
    alpha21 = params.get('alpha21', 0.5)










    raise NotImplementedError("HOLE 1: 请实现 compute_reaction_terms 的核心反应项计算")



def equilibrium_analysis(params: dict) -> dict:
    beta = params.get('beta11', 0.3)
    gamma = params.get('gamma1', 0.1)
    mu = params.get('mu1', 0.02)
    r = params.get('r_mean', 1.0)
    K = params.get('K_mean', 100.0)

    if beta > 0:
        S_star = (gamma + mu) / beta
        I_star = max(0.0, r * S_star * (1.0 - S_star / K) / (gamma + mu))
        R_star = gamma * I_star / max(mu, 1e-10)
    else:
        S_star, I_star, R_star = K, 0.0, 0.0


    R0 = beta * K / (gamma + mu)

    return {
        'S_star': S_star,
        'I_star': I_star,
        'R_star': R_star,
        'R0': R0,
        'disease_free_stable': R0 < 1.0,
    }
