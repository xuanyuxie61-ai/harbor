
import numpy as np
from typing import Callable, Optional, Tuple


def evaluate_policy_cost(
    theta: np.ndarray,
    policy_fn: Callable[[np.ndarray, np.ndarray], float],
    sde_integrator: Callable,
    cost_fn: Callable,
    n_mc: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng(seed=42)

    costs = []
    for _ in range(n_mc):
        t, y = sde_integrator(rng=rng)
        c = cost_fn(t, y, policy_fn, theta)
        costs.append(c)

    costs_arr = np.array(costs)

    q_low, q_high = np.percentile(costs_arr, [5, 95])
    mask = (costs_arr >= q_low) & (costs_arr <= q_high)
    if np.sum(mask) > 0:
        return float(np.mean(costs_arr[mask]))
    return float(np.mean(costs_arr))


def nelder_mead_optimize(
    obj_fn: Callable[[np.ndarray], float],
    x0: np.ndarray,
    rho: float = 1.0,
    xi: float = 2.0,
    gam: float = 0.5,
    sig: float = 0.5,
    tolerance: float = 1e-6,
    max_feval: int = 500,
) -> Tuple[np.ndarray, int]:
    m = len(x0)

    simplex = np.zeros((m + 1, m))
    simplex[0, :] = x0

    scale = np.maximum(np.abs(x0), 0.1)
    for i in range(m):
        simplex[i + 1, :] = x0.copy()
        simplex[i + 1, i] += 0.05 * scale[i]


    f_vals = np.zeros(m + 1)
    for i in range(m + 1):
        f_vals[i] = obj_fn(simplex[i, :])

    n_feval = m + 1

    converged = False
    diverged = False

    while not converged and not diverged:

        order = np.argsort(f_vals)
        simplex = simplex[order, :]
        f_vals = f_vals[order]


        f_best = f_vals[0]
        f_worst = f_vals[-1]
        f_second_worst = f_vals[-2]


        x_bar = np.mean(simplex[:m, :], axis=0)


        x_r = x_bar + rho * (x_bar - simplex[-1, :])
        f_r = obj_fn(x_r)
        n_feval += 1

        if f_best <= f_r < f_second_worst:
            simplex[-1, :] = x_r
            f_vals[-1] = f_r
        elif f_r < f_best:

            x_e = x_bar + xi * (x_bar - simplex[-1, :])
            f_e = obj_fn(x_e)
            n_feval += 1
            if f_e < f_r:
                simplex[-1, :] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1, :] = x_r
                f_vals[-1] = f_r
        elif f_second_worst <= f_r < f_worst:

            x_c = x_bar + gam * (x_bar - simplex[-1, :])
            f_c = obj_fn(x_c)
            n_feval += 1
            if f_c <= f_r:
                simplex[-1, :] = x_c
                f_vals[-1] = f_c
            else:

                for i in range(1, m + 1):
                    simplex[i, :] = simplex[0, :] + sig * (simplex[i, :] - simplex[0, :])
                    f_vals[i] = obj_fn(simplex[i, :])
                    n_feval += 1
        else:

            x_ci = x_bar - gam * (x_bar - simplex[-1, :])
            f_ci = obj_fn(x_ci)
            n_feval += 1
            if f_ci < f_worst:
                simplex[-1, :] = x_ci
                f_vals[-1] = f_ci
            else:

                for i in range(1, m + 1):
                    simplex[i, :] = simplex[0, :] + sig * (simplex[i, :] - simplex[0, :])
                    f_vals[i] = obj_fn(simplex[i, :])
                    n_feval += 1


        converged = abs(f_vals[-1] - f_vals[0]) < tolerance
        diverged = n_feval > max_feval


        simplex_spread = np.max(np.std(simplex, axis=0))
        if simplex_spread < 1e-12 and not converged:
            for i in range(1, m + 1):
                noise = np.random.randn(m) * 1e-4
                simplex[i, :] += noise
                f_vals[i] = obj_fn(simplex[i, :])
                n_feval += 1

    x_opt = simplex[0, :]
    return x_opt, n_feval


def linear_feedback_policy(
    theta: np.ndarray,
    x: np.ndarray,
    x_eq: Optional[np.ndarray] = None,
) -> float:
    if x_eq is None:
        x_eq = np.zeros_like(x)
    dx = x - x_eq
    K = theta[: len(dx)]
    u = -np.dot(K, dx)

    return float(np.clip(u, -5.0, 5.0))


def quadratic_policy(
    theta: np.ndarray,
    x: np.ndarray,
    x_eq: Optional[np.ndarray] = None,
) -> float:
    dim = len(x)
    if x_eq is None:
        x_eq = np.zeros(dim)
    dx = x - x_eq

    K1 = theta[:dim]

    k2_vals = theta[dim:]
    K2 = np.zeros((dim, dim))
    idx = 0
    for i in range(dim):
        for j in range(i, dim):
            if idx < len(k2_vals):
                K2[i, j] = k2_vals[idx]
                K2[j, i] = k2_vals[idx]
                idx += 1

    u = -np.dot(K1, dx) - float(dx @ K2 @ dx)
    return float(np.clip(u, -5.0, 5.0))
