"""
control_optimizer.py - Convergence optimization for freeze-out conditions.

Adapted from 215_control_bio (optimization convergence).  Uses gradient
descent and Newton methods to find the freeze-out temperature T_f and
density rho_f that minimize the residual of the abundance equations.

Objective: minimize  || dY/dt ||^2  subject to conservation laws.
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Tuple

def gradient_descent(grad_func: Callable, Y0: np.ndarray,
                     lr: float = 0.01, tol: float = 1e-6,
                     max_iter: int = 1000) -> Tuple[np.ndarray, int, float]:
    """
    Gradient descent: Y_{k+1} = Y_k - lr * grad(Y_k).
    Returns (Y_opt, n_iter, final_residual).
    """
    Y = Y0.copy()
    for k in range(max_iter):
        g = grad_func(Y)
        res = np.linalg.norm(g)
        if res < tol:
            return Y, k, res
        Y = Y - lr * g
    return Y, max_iter, np.linalg.norm(grad_func(Y))


def conjugate_gradient(A: np.ndarray, b: np.ndarray,
                       tol: float = 1e-8, max_iter: int = 100) -> np.ndarray:
    """
    Conjugate gradient for  A x = b  with A symmetric positive-definite.
    """
    n = len(b)
    x = np.zeros(n)
    r = b - A @ x
    p = r.copy()
    rs_old = r @ r
    for k in range(max_iter):
        if np.sqrt(rs_old) < tol:
            break
        Ap = A @ p
        alpha = rs_old / (p @ Ap + 1e-300)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        beta = rs_new / (rs_old + 1e-300)
        p = r + beta * p
        rs_old = rs_new
    return x


def newton_optimizer(F: Callable, J: Callable, Y0: np.ndarray,
                     tol: float = 1e-8, max_iter: int = 50) -> Tuple[np.ndarray, int, float]:
    """
    Newton optimization: find Y such that F(Y) = 0.
    """
    Y = Y0.copy()
    for k in range(max_iter):
        Fv = F(Y)
        res = np.linalg.norm(Fv)
        if res < tol:
            return Y, k, res
        Jv = J(Y)
        try:
            delta = np.linalg.solve(Jv, Fv)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(Jv, Fv, rcond=None)[0]
        Y = Y - delta
    return Y, max_iter, np.linalg.norm(F(Y))


def find_freeze_out_conditions(T9_func: Callable, rho_func: Callable,
                                Y_eq_func: Callable,
                                T9_range: Tuple[float, float] = (0.5, 10.0),
                                tol: float = 1e-6) -> Tuple[float, float, float]:
    """
    Find freeze-out T9_f, rho_f, and residual.
    Freeze-out occurs when  || dY/dt || < tol.
    """
    T9_grid = np.linspace(T9_range[0], T9_range[1], 50)
    best_T9 = T9_grid[0]
    best_res = 1e300
    for T9 in T9_grid:
        rho = rho_func(T9)
        Y_eq = Y_eq_func(T9, rho)
        res = np.linalg.norm(Y_eq)
        if res < best_res:
            best_res = res
            best_T9 = T9
    return float(best_T9), float(rho_func(best_T9)), float(best_res)


__all__ = [
    "gradient_descent", "conjugate_gradient", "newton_optimizer",
    "find_freeze_out_conditions",
]
