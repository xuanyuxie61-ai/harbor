# -*- coding: utf-8 -*-
"""
Utility Functions for Numerical Robustness
===========================================
Boundary condition enforcement, numerical stability checks, and helper
functions for the spectral PDE solver.
"""

import numpy as np


def enforce_dirichlet(u, val_left, val_right):
    """
    Enforce Dirichlet boundary conditions on a 1D solution vector.
    Assumes u[0] corresponds to right boundary (x=+1) and u[-1] to left (x=-1)
    for descending CGL ordering.

    Parameters
    ----------
    u : ndarray
        Solution vector.
    val_left : float
        Value at x = -1.
    val_right : float
        Value at x = +1.

    Returns
    -------
    u : ndarray
        Modified vector.
    """
    u = np.asarray(u, dtype=np.float64).copy()
    u[-1] = val_left
    u[0] = val_right
    return u


def enforce_neumann(u, D, val_left=0.0, val_right=0.0):
    """
    Enforce Neumann boundary conditions: du/dx = g at boundaries.
    Uses the spectral differentiation matrix D to set the derivative.
    For interior solution, replaces the boundary equations with:
        D[0,:] @ u = val_right
        D[-1,:] @ u = val_left

    Parameters
    ----------
    u : ndarray
        Solution vector (will be used in system assembly, not modified directly).
    D : ndarray
        Spectral differentiation matrix.
    val_left, val_right : float
        Neumann values.

    Returns
    -------
    A_bc, b_bc : ndarrays
        Modified system row and RHS for boundary conditions.
    """
    n = len(u)
    A_bc = np.zeros((2, n), dtype=np.float64)
    b_bc = np.array([val_right, val_left], dtype=np.float64)
    A_bc[0, :] = D[0, :]
    A_bc[1, :] = D[-1, :]
    return A_bc, b_bc


def check_solution_stability(u, max_val=1e6, min_val=-1e6):
    """
    Check if solution values are within reasonable bounds.

    Parameters
    ----------
    u : ndarray
        Solution vector.
    max_val, min_val : float
        Bounds.

    Returns
    -------
    stable : bool
    """
    u = np.asarray(u)
    if np.any(np.isnan(u)) or np.any(np.isinf(u)):
        return False
    if np.any(u > max_val) or np.any(u < min_val):
        return False
    return True


def smooth_initial_condition(x, case="gaussian"):
    """
    Generate smooth initial conditions for PDE tests.

    Parameters
    ----------
    x : ndarray
        Spatial grid.
    case : str
        "gaussian", "sine", or "poly".

    Returns
    -------
    u0 : ndarray
        Initial condition.
    """
    x = np.asarray(x, dtype=np.float64)
    if case == "gaussian":
        return np.exp(-10.0 * x ** 2)
    elif case == "sine":
        return np.sin(np.pi * (x + 1.0) / 2.0)
    elif case == "poly":
        return (1.0 - x ** 2) ** 2
    else:
        return np.exp(-10.0 * x ** 2)


def map_domain(x, a, b):
    """
    Map points from [-1, 1] to [a, b].

    Parameters
    ----------
    x : ndarray
        Points in [-1, 1].
    a, b : float
        Target domain.

    Returns
    -------
    y : ndarray
        Mapped points.
    """
    return 0.5 * (b - a) * x + 0.5 * (a + b)


def relative_l2_error(u_num, u_ref, weights=None):
    """
    Compute relative L2 error between numerical and reference solutions.

    Parameters
    ----------
    u_num, u_ref : ndarray
    weights : ndarray, optional
        Quadrature weights.

    Returns
    -------
    err : float
    """
    u_num = np.asarray(u_num, dtype=np.float64)
    u_ref = np.asarray(u_ref, dtype=np.float64)
    diff = u_num - u_ref
    if weights is None:
        num = np.sqrt(np.mean(diff ** 2))
        den = np.sqrt(np.mean(u_ref ** 2))
    else:
        num = np.sqrt(np.sum(weights * diff ** 2))
        den = np.sqrt(np.sum(weights * u_ref ** 2))
    if den < 1e-15:
        return num
    return num / den


def print_banner(title, width=70):
    """
    Print a formatted banner.
    """
    print("=" * width)
    print(title.center(width))
    print("=" * width)
