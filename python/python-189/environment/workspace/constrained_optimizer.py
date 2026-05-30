
import numpy as np
from typing import Tuple
from scipy.optimize import linprog
from special_functions import incomplete_beta






def lp_action_projection(action_raw: np.ndarray,
                          C: np.ndarray = None,
                          d: np.ndarray = None,
                          bounds: Tuple[float, float] = (-2.0, 2.0)) -> np.ndarray:
    action_raw = np.asarray(action_raw, dtype=float)
    n = len(action_raw)


    if C is None or d is None or len(C) == 0:
        return np.clip(action_raw, bounds[0], bounds[1])





    c = np.concatenate([np.zeros(n), np.ones(n)])





    A_ub = []
    b_ub = []
    for i in range(n):
        row1 = np.zeros(2 * n)
        row1[i] = 1.0
        row1[n + i] = -1.0
        A_ub.append(row1)
        b_ub.append(action_raw[i])

        row2 = np.zeros(2 * n)
        row2[i] = -1.0
        row2[n + i] = -1.0
        A_ub.append(row2)
        b_ub.append(-action_raw[i])

    if C is not None and d is not None:
        C = np.atleast_2d(C)
        d = np.asarray(d, dtype=float)
        for j in range(C.shape[0]):
            row = np.zeros(2 * n)
            row[:n] = C[j, :]
            A_ub.append(row)
            b_ub.append(d[j])


    bounds_lp = [(bounds[0], bounds[1]) for _ in range(n)] + [(0, None) for _ in range(n)]

    try:
        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      bounds=bounds_lp, method='highs')
        if res.success:
            return np.clip(res.x[:n], bounds[0], bounds[1])
    except Exception:
        pass


    return np.clip(action_raw, bounds[0], bounds[1])






def trust_region_probability(delta: float, param_dim: int,
                              sample_size: int, sigma: float = 1.0) -> float:
    if delta <= 0 or param_dim <= 0 or sample_size <= param_dim:
        return 0.0

    p = param_dim / 2.0
    q = (sample_size - param_dim) / 2.0
    if q <= 0:
        q = 1.0
    x = delta / (delta + sigma ** 2)
    prob, ier = incomplete_beta(x, p, q)
    if ier != 0:
        return 0.0
    return prob


def check_trust_region(kl_value: float, max_kl: float,
                        param_dim: int, sample_size: int) -> bool:
    if kl_value <= max_kl:
        return True
    prob = trust_region_probability(kl_value, param_dim, sample_size)

    return prob > 0.95






class CosineAnnealingScheduler:

    def __init__(self, alpha_max: float = 0.01, alpha_min: float = 1.0e-5,
                 T_period: int = 100):
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.T_period = T_period
        self.t = 0

    def step(self) -> float:
        ratio = self.t / self.T_period
        alpha = self.alpha_min + 0.5 * (self.alpha_max - self.alpha_min) \
                * (1.0 + np.cos(np.pi * ratio))
        self.t += 1
        return alpha

    def reset(self):
        self.t = 0
