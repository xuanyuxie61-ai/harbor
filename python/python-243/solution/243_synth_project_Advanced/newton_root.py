"""
newton_root.py - Newton-Maehly root finding for equilibrium abundances.

Adapted from 801_newton_maehly.  Solves  f(Y) = 0  where Y is the abundance
vector, using deflation to find multiple equilibria (NSE, QSE, freeze-out).

  Newton step:  Y_{k+1} = Y_k - J^{-1} F(Y_k)
  Maehly deflation:  after root r found, solve  g(x) = f(x) / (x - r) = 0
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Tuple

def newton_step(F: Callable, J: Callable, Y: np.ndarray) -> np.ndarray:
    """One Newton step: Y_new = Y - J^{-1} F(Y)."""
    Fv = F(Y)
    Jv = J(Y)
    try:
        delta = np.linalg.solve(Jv, Fv)
    except np.linalg.LinAlgError:
        # Fallback: pseudo-inverse
        delta = np.linalg.lstsq(Jv, Fv, rcond=None)[0]
    return Y - delta


def newton_maehly(F: Callable, J: Callable, Y0: np.ndarray,
                  tol: float = 1e-8, max_iter: int = 100) -> Tuple[np.ndarray, int, float]:
    """
    Newton-Maehly iteration.  Returns (Y_root, n_iter, residual_norm).
    """
    Y = Y0.copy()
    roots_found = []
    for k in range(max_iter):
        Fv = F(Y)
        res = np.linalg.norm(Fv)
        if res < tol:
            return Y, k, res
        # Maehly deflation: modify F to avoid already-found roots
        F_deflated = lambda y, _F=F, _roots=roots_found: _deflate(_F, y, _roots)
        J_deflated = lambda y, _J=J: _J(y)
        Y = newton_step(F_deflated, J_deflated, Y)
        # Check if converged to a new root
        if res < tol * 10:
            roots_found.append(Y.copy())
    return Y, max_iter, np.linalg.norm(F(Y))


def _deflate(F: Callable, Y: np.ndarray, roots: list) -> np.ndarray:
    """Maehly deflation: F_deflated(Y) = F(Y) / prod(Y - r_i)."""
    Fv = F(Y)
    for r in roots:
        diff = Y - r
        diff = np.where(np.abs(diff) < 1e-12, 1e-12, diff)
        Fv = Fv / diff
    return Fv


def find_equilibrium_abundances(Y_init: np.ndarray,
                                 F_rhs: Callable,
                                 J_rhs: Callable,
                                 tol: float = 1e-8,
                                 max_iter: int = 100) -> Tuple[np.ndarray, int, float]:
    """
    Find equilibrium abundances Y_eq such that F(Y_eq) = 0.
    F_rhs is the net rate of change  dY/dt.
    """
    return newton_maehly(F_rhs, J_rhs, Y_init, tol, max_iter)


__all__ = [
    "newton_step", "newton_maehly", "find_equilibrium_abundances",
]
