
import numpy as np
from typing import Callable, Tuple, Optional


def _minfit(n: int, tol: float, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        U, d, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        d = np.ones(n, dtype=float)
        Vt = np.eye(n, dtype=float)

    idx = np.argsort(-d)
    d = d[idx]
    V = Vt.T[:, idx]
    return d, V


def _line_search(f: Callable, x: np.ndarray, d: np.ndarray,
                 f0: float, h0: float = 1.0, tol: float = 1e-8) -> Tuple[float, np.ndarray]:
    d = np.asarray(d, dtype=float)
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-14:
        return f0, x.copy()
    d = d / d_norm

    x1 = x + h0 * d
    x2 = x + 2.0 * h0 * d
    f1 = f(x1)
    f2 = f(x2)





    denom = 2.0 * h0 * h0
    a = (f2 - 2.0 * f1 + f0) / denom
    b = (4.0 * f1 - f2 - 3.0 * f0) / (2.0 * h0)
    if a > 1e-14:
        s_star = -b / (2.0 * a)
        s_star = np.clip(s_star, -h0 * 3.0, h0 * 3.0)
    elif b < 0:
        s_star = h0 * 3.0
    else:
        s_star = 0.0
    x_best = x + s_star * d
    f_best = f(x_best)
    if f_best > f0:
        return f0, x.copy()
    return f_best, x_best


class PraxisOptimizer:

    def __init__(self, tol: float = 1e-6, max_iter: int = 500,
                 h0: float = 0.1, scbd: float = 10.0):
        self.tol = tol
        self.max_iter = max_iter
        self.h0 = h0
        self.scbd = scbd

    def minimize(self, f: Callable, x0: np.ndarray) -> Tuple[np.ndarray, float]:
        x = np.asarray(x0, dtype=float).copy()
        n = x.size
        fval = float(f(x))

        V = np.eye(n, dtype=float)
        D_est = np.ones(n, dtype=float)

        for iteration in range(self.max_iter):
            x_old = x.copy()
            f_old = fval

            for i in range(n):
                d = V[:, i]
                f_new, x_new = _line_search(f, x, d, fval, h0=self.h0, tol=self.tol)
                if f_new < fval:
                    D_est[i] = max(D_est[i] * 0.5, 1e-8)
                    fval = f_new
                    x = x_new
                else:
                    D_est[i] = min(D_est[i] * 2.0, 1e6)

            if np.linalg.norm(x - x_old) < self.tol and abs(fval - f_old) < self.tol:
                break

            if (iteration + 1) % max(n, 3) == 0:


                H_approx = V @ np.diag(D_est) @ V.T
                d_new, V_new = _minfit(n, self.tol, H_approx)

                for i in range(n):
                    if d_new[i] > 1e-14:
                        V[:, i] = V_new[:, i] * (1.0 / np.sqrt(d_new[i]))
                    else:
                        V[:, i] = V_new[:, i]

                for i in range(n):
                    norm = np.linalg.norm(V[:, i])
                    if norm > 1e-14:
                        V[:, i] = V[:, i] / norm

            if abs(fval - f_old) < self.tol * 0.01:
                perturb = self.h0 * (np.random.default_rng().random(n) - 0.5) * self.scbd
                x_try = x + perturb
                f_try = f(x_try)
                if f_try < fval:
                    fval = f_try
                    x = x_try

            if not np.isfinite(fval):
                x = x_old
                fval = f_old
                break
        return x, fval


def trajectory_cost_function(q_nodes_flat: np.ndarray,
                              kin_func,
                              obstacle_checker,
                              q_start: np.ndarray,
                              q_goal: np.ndarray,
                              dt: float = 0.1) -> float:
    n_dof = q_start.size
    n_nodes = q_nodes_flat.size // n_dof
    if n_nodes * n_dof != q_nodes_flat.size:
        return 1e10
    q_nodes = q_nodes_flat.reshape(n_nodes, n_dof)

    q_nodes[0] = q_start
    q_nodes[-1] = q_goal

    cost = 0.0
    w = [1.0, 500.0, 50.0, 10.0, 100.0]

    for i in range(n_nodes - 1):
        cost += w[0] * np.linalg.norm(q_nodes[i + 1] - q_nodes[i])

    for i in range(n_nodes):
        q = q_nodes[i]

        try:
            T_ee = kin_func(q)
            p_ee = T_ee[:3, 3]
        except Exception:
            cost += 1e6
            continue
        dist_penalty = obstacle_checker(p_ee)
        if dist_penalty < 0:
            cost += w[1] * abs(dist_penalty)
        elif dist_penalty < 0.2:
            cost += w[2] * (0.2 - dist_penalty)


        for j in range(n_dof):
            margin = min(abs(q[j] + np.pi), abs(q[j] - np.pi))
            if margin < 0.1:
                cost += w[4] / (margin + 1e-3)

    for i in range(1, n_nodes - 1):
        ddq = q_nodes[i + 1] - 2 * q_nodes[i] + q_nodes[i - 1]
        cost += w[3] * np.linalg.norm(ddq)

    if not np.isfinite(cost):
        cost = 1e10
    return float(cost)
