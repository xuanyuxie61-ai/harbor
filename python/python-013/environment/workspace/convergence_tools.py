
import numpy as np
from typing import List, Callable, Optional, Tuple






def collatz_sequence(start: int) -> List[int]:
    if start <= 0:
        return []
    seq = [start]
    t = start
    max_steps = 10000
    steps = 0
    while t != 1 and steps < max_steps:
        if t % 2 == 0:
            t = t // 2
        else:
            t = 3 * t + 1
        seq.append(t)
        steps += 1
    return seq


def collatz_stopping_time(start: int) -> int:
    return len(collatz_sequence(start)) - 1


def iteration_complexity_index(residuals: List[float]) -> float:
    if len(residuals) < 2:
        return 0.0
    C = 0.0
    for i in range(len(residuals) - 1):
        r = abs(residuals[i])
        if r < 1e-14:
            r = 1e-14
        C += abs(residuals[i + 1] - residuals[i]) / r
    return C






def simple_mixing(old: np.ndarray, new: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha 必须在 (0, 1] 内")
    return alpha * new + (1.0 - alpha) * old


def pulay_mixing(history: List[np.ndarray], residuals: List[np.ndarray], n_keep: int = 5) -> np.ndarray:
    m = min(len(history), len(residuals), n_keep)
    if m < 2:
        return history[-1]

    A = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            idx_i = -(m - i)
            idx_j = -(m - j)
            A[i, j] = np.vdot(residuals[idx_i], residuals[idx_j]).real

    A += 1e-10 * np.eye(m)

    rhs = np.zeros(m + 1)
    rhs[m] = 1.0
    M = np.zeros((m + 1, m + 1))
    M[:m, :m] = A
    M[m, :m] = 1.0
    M[:m, m] = 1.0
    try:
        c = np.linalg.solve(M, rhs)[:m]
    except np.linalg.LinAlgError:

        return history[-1]

    x_mix = np.zeros_like(history[-1])
    for i in range(m):
        x_mix += c[i] * history[-(m - i)]
    return x_mix


def self_consistent_iteration(update_func: Callable, x0: np.ndarray,
                               tol: float = 1e-6, max_iter: int = 100,
                               mixing: str = "simple", alpha: float = 0.5,
                               n_keep: int = 5) -> Tuple[np.ndarray, int, List[float]]:
    if tol <= 0:
        raise ValueError("tol > 0")
    if max_iter < 1:
        raise ValueError("max_iter >= 1")
    x = x0.copy()
    residuals = []
    history = []
    res_history = []
    for it in range(max_iter):
        x_new = update_func(x)
        r = x_new - x
        res_norm = float(np.linalg.norm(r))
        residuals.append(res_norm)
        res_history.append(res_norm)
        history.append(x.copy())
        if res_norm < tol:
            return x_new, it + 1, res_history
        if mixing == "pulay" and len(history) >= 2:

            res_vecs = [history[i + 1] - history[i] for i in range(len(history) - 1)]
            res_vecs.append(r)
            x = pulay_mixing(history, res_vecs, n_keep)
        else:
            x = simple_mixing(x, x_new, alpha)
    return x, max_iter, res_history






def interpolation_stability_analysis(x_nodes: np.ndarray, x_test: np.ndarray) -> dict:
    from matsubara_green import lebesgue_constant_estimate
    n = len(x_nodes)
    lmax = lebesgue_constant_estimate(n, x_nodes, x_test)

    dx = np.diff(np.sort(x_nodes))
    min_spacing = float(np.min(dx)) if len(dx) > 0 else 0.0
    max_spacing = float(np.max(dx)) if len(dx) > 0 else 0.0
    return {
        "lebesgue_constant": lmax,
        "min_spacing": min_spacing,
        "max_spacing": max_spacing,
        "node_count": n,
    }






def adaptive_damping(residuals: List[float], alpha_min: float = 0.05,
                      alpha_max: float = 0.8) -> float:
    if len(residuals) < 3:
        return alpha_max

    dr1 = residuals[-2] - residuals[-3]
    dr2 = residuals[-1] - residuals[-2]
    if dr1 < 0 and dr2 < 0:

        return min(alpha_max, alpha_max * 1.1)
    elif dr1 * dr2 < 0:

        return max(alpha_min, alpha_max * 0.5)
    else:
        return alpha_max * 0.7


if __name__ == "__main__":
    seq = collatz_sequence(27)
    print(f"Collatz(27) stopping time = {len(seq)-1}")
    x0 = np.array([1.0, 2.0])
    def f(x):
        return np.array([0.5 * x[0] + 1.0, 0.3 * x[1] + 0.5])
    x, it, res = self_consistent_iteration(f, x0, tol=1e-8, max_iter=50)
    print(f"Converged in {it} iterations, residual={res[-1]:.2e}")
