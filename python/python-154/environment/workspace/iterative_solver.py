
import numpy as np
from typing import Tuple, Optional, Callable


def jacobi_iteration_step(A: np.ndarray, b: np.ndarray,
                          x: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be square")
    if b.shape != (n,) or x.shape != (n,):
        raise ValueError("b and x must be 1D arrays of length n")
    x_new = np.zeros(n, dtype=float)
    for i in range(n):
        diag = A[i, i]
        if abs(diag) < 1e-15:
            raise RuntimeError(f"Zero diagonal element at index {i}")
        s = b[i]
        for j in range(n):
            if j != i:
                s -= A[i, j] * x[j]
        x_new[i] = s / diag
    return x_new


def jacobi_solve(A: np.ndarray, b: np.ndarray,
                 x0: Optional[np.ndarray] = None,
                 max_iter: int = 10000, tol: float = 1e-10,
                 omega: float = 1.0) -> Tuple[np.ndarray, int, float]:
    n = A.shape[0]
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.array(x0, dtype=float)
    if not (0.0 < omega <= 2.0):
        raise ValueError("omega must be in (0, 2]")

    residual_history = []
    for it in range(max_iter):
        x_new = jacobi_iteration_step(A, b, x)
        x = (1.0 - omega) * x + omega * x_new
        r = b - A @ x
        res_norm = float(np.linalg.norm(r, ord=np.inf))
        residual_history.append(res_norm)
        if res_norm < tol:
            return x, it + 1, res_norm

        if it > 100 and res_norm > 1e12 * max(residual_history[0], 1.0):
            raise RuntimeError("Jacobi iteration appears to diverge")
    return x, max_iter, float(np.linalg.norm(b - A @ x, ord=np.inf))


def self_consistent_mean_field(J: np.ndarray, h: np.ndarray,
                                beta: float = 1.0,
                                max_iter: int = 5000,
                                tol: float = 1e-10,
                                damping: float = 0.5) -> Tuple[np.ndarray, int]:



    raise NotImplementedError("Hole 2: 请补全平均场自洽场方程的阻尼迭代求解逻辑")


def variational_ground_state_energy(J: np.ndarray, h: np.ndarray,
                                     m: np.ndarray) -> float:
    n = h.size
    m = np.clip(m, -1.0 + 1e-12, 1.0 - 1e-12)
    e_int = -0.5 * float(m @ J @ m) - float(h @ m)

    p_plus = (1.0 + m) / 2.0
    p_minus = (1.0 - m) / 2.0
    entropy = np.sum(p_plus * np.log(p_plus) + p_minus * np.log(p_minus))
    return float(e_int + entropy)


def power_iteration_eigenvalue(A: np.ndarray, max_iter: int = 1000,
                                tol: float = 1e-10) -> Tuple[float, np.ndarray]:
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    lam = 0.0
    for it in range(max_iter):
        Av = A @ v
        v_new = Av / np.linalg.norm(Av)
        lam_new = float(v_new @ A @ v_new)
        if abs(lam_new - lam) < tol:
            return lam_new, v_new
        lam = lam_new
        v = v_new
    return lam, v


def chebyshev_accelerated_jacobi(A: np.ndarray, b: np.ndarray,
                                  x0: Optional[np.ndarray] = None,
                                  max_iter: int = 500,
                                  tol: float = 1e-10) -> Tuple[np.ndarray, int, float]:
    n = A.shape[0]
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.array(x0, dtype=float)
    Dinv = np.diag(1.0 / np.diag(A))

    mu_est = []
    for i in range(n):
        row_sum = np.sum(np.abs(A[i, :])) / abs(A[i, i]) - 1.0
        mu_est.append(row_sum)
    mu_max = min(max(mu_est), 0.99)
    mu_min = -mu_max
    if mu_max <= 0:
        mu_max = 0.5
        mu_min = -0.5


    sigma = (mu_max - mu_min) / 2.0
    delta = (mu_max + mu_min) / 2.0
    alpha = 2.0 / (2.0 - mu_max - mu_min)

    x_prev = x.copy()
    for it in range(max_iter):
        r = b - A @ x
        if it == 0:
            x_new = x + alpha * (Dinv @ r)
            rho = 1.0 / (1.0 - 0.5 * sigma * sigma)
        else:
            rho_prev = rho
            rho = 1.0 / (1.0 - 0.25 * sigma * sigma * rho_prev)
            x_new = x + rho * alpha * (Dinv @ r) + (1.0 - rho) * (x - x_prev)
        x_prev, x = x, x_new
        res_norm = float(np.linalg.norm(b - A @ x, ord=np.inf))
        if res_norm < tol:
            return x, it + 1, res_norm
    return x, max_iter, float(np.linalg.norm(b - A @ x, ord=np.inf))
