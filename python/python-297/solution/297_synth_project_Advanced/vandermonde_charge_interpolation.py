"""
vandermonde_charge_interpolation.py
=====================================
Vandermonde-based polynomial reconstruction for dust grain charge
distribution interpolation on non-uniform grids.

Physical motivation:
    In dusty plasma simulations, dust grains may have non-uniform
    spatial distributions. The grain charge Z_d(r) depends on local
    plasma parameters (n_e, T_e) which vary in space. We use
    Vandermonde matrix techniques to:
        1. Interpolate charge distributions on non-uniform grids
        2. Solve polynomial fitting problems for charge profiles
        3. Use the Bjorck-Pereyra algorithm for stable solutions

    The Vandermonde system:
        V * c = f,  where V[i,j] = x[j]^i
    arises naturally when fitting polynomials to discrete charge
    measurements at grain locations.

References:
    - Bjorck & Pereyra, "Solution of Vandermonde systems of equations",
      Math. Comp. 24, 893-903 (1970) (from 1381_vandermonde)
    - Gaul, Reinsch, "The Bjorck-Pereyra algorithm: new error analysis
      and improved implementation", Computing 87, 1-17 (2010)
"""

import numpy as np
from typing import Tuple, Optional


def vandermonde_matrix(x: np.ndarray, degree: Optional[int] = None) -> np.ndarray:
    """
    Construct the Vandermonde matrix V[i,j] = x[i]^j.

    For dusty plasma charge interpolation, x represents normalized
    radial positions of dust grains and we fit a polynomial of
    specified degree.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Interpolation nodes (grain positions).
    degree : int, optional
        Polynomial degree. Default: N-1.

    Returns
    -------
    V : np.ndarray, shape (N, degree+1)
        Vandermonde matrix.
    """
    if degree is None:
        degree = len(x) - 1
    N = len(x)
    V = np.zeros((N, degree + 1), dtype=np.float64)
    V[:, 0] = 1.0
    for j in range(1, degree + 1):
        V[:, j] = V[:, j - 1] * x
    return V


def vandermonde_transpose_matrix(x: np.ndarray, degree: Optional[int] = None) -> np.ndarray:
    """
    Construct V^T where V is the Vandermonde matrix.

    V^T * c = f corresponds to the dual problem: given coefficients c,
    evaluate the polynomial at nodes x.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Nodes.
    degree : int, optional
        Polynomial degree.

    Returns
    -------
    VT : np.ndarray, shape (degree+1, N)
        Transpose of Vandermonde matrix.
    """
    return vandermonde_matrix(x, degree).T


def bjorck_pereyra_solve(
    x: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Solve V*c = b using the Bjorck-Pereyra algorithm.

    The BP algorithm exploits the structure of the Vandermonde matrix
    to solve the system in O(N^2) operations (vs O(N^3) for generic
    Gaussian elimination). More importantly, it is forward-stable
    when x nodes are in [0,1] or similarly well-scaled.

    Algorithm (from 1381_vandermonde vand1.m):
        for j = 1, ..., N-1:
            for i = N, N-1, ..., j+1:
                b[i] = (b[i] - b[i-1]) / (x[i] - x[i-j])

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Interpolation nodes (must be distinct).
    b : np.ndarray, shape (N,)
        Right-hand side (function values at nodes).

    Returns
    -------
    c : np.ndarray, shape (N,)
        Polynomial coefficients such that V*c ≈ b.

    Raises
    ------
    ValueError
        If nodes are not distinct or system is singular.
    """
    N = len(x)
    b_work = b.copy().astype(np.float64)

    # Check node distinctness
    for i in range(N):
        for j in range(i + 1, N):
            if abs(x[i] - x[j]) < 1e-14 * max(abs(x[i]), abs(x[j]), 1.0):
                raise ValueError(
                    f"Vandermonde system is singular: nodes {i} and {j} "
                    f"coincide (x[{i}]={x[i]}, x[{j}]={x[j]})"
                )

    # Bjorck-Pereyra divided difference scheme
    for j in range(1, N):
        for i in range(N - 1, j - 1, -1):
            denom = x[i] - x[i - j]
            if abs(denom) < 1e-30:
                raise ValueError(
                    f"Near-zero denominator in BP algorithm at i={i}, j={j}"
                )
            b_work[i] = (b_work[i] - b_work[i - 1]) / denom

    return b_work


def bjorck_pereyra_transpose_solve(
    x: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Solve V^T * c = b using the transposed Bjorck-Pereyra algorithm.

    This is the dual problem, used for computing polynomial values
    or moment-based reconstructions.

    Algorithm (from 1381_vandermonde dvand.m):
        for j = 1, ..., N-1:
            for i = N-1, N-2, ..., j:
                b[i] = b[i] - x[j] * b[i+1]  (upward sweep)
            for i = j, j+1, ..., N-1:
                b[i] = b[i] / (x[i] - x[i-j])

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Nodes.
    b : np.ndarray, shape (N,)
        Right-hand side.

    Returns
    -------
    c : np.ndarray, shape (N,)
        Solution of V^T * c = b.
    """
    N = len(x)
    b_work = b.copy().astype(np.float64)

    for j in range(N - 1):
        # Downward sweep
        for i in range(N - 2, j - 1, -1):
            b_work[i] = b_work[i] - x[j] * b_work[i + 1]

        # Upward normalization
        for i in range(j, N - 1):
            denom = x[i + 1] - x[i + 1 - (j + 1)]
            if abs(denom) < 1e-30:
                denom = 1e-30 if denom >= 0 else -1e-30
            b_work[i] = b_work[i] / denom

    # Final normalization
    if abs(x[-1]) > 1e-30:
        b_work[N - 1] = b_work[N - 1] / x[N - 1] if N > 1 else b_work[N - 1]

    return b_work


def bivariate_vandermonde(
    x: np.ndarray,
    y: np.ndarray,
    degree_x: int,
    degree_y: int,
) -> np.ndarray:
    """
    Construct a bivariate Vandermonde matrix for 2D interpolation.

    For dusty plasma, this is used to reconstruct the charge
    distribution Z_d(x,y) on the crystal plane.

    V[i, (j,k)] = x[i]^j * y[i]^k
    where j ranges over [0, degree_x] and k over [0, degree_y].

    Parameters
    ----------
    x, y : np.ndarray, shape (N,)
        2D node coordinates.
    degree_x, degree_y : int
        Maximum polynomial degrees in x and y.

    Returns
    -------
    V : np.ndarray, shape (N, (degree_x+1)*(degree_y+1))
        Bivariate Vandermonde matrix.
    """
    N = len(x)
    ncols = (degree_x + 1) * (degree_y + 1)
    V = np.zeros((N, ncols), dtype=np.float64)

    col = 0
    for jx in range(degree_x + 1):
        for jy in range(degree_y + 1):
            V[:, col] = (x ** jx) * (y ** jy)
            col += 1

    return V


def interpolate_dust_charge(
    positions: np.ndarray,
    charges: np.ndarray,
    eval_positions: np.ndarray,
    degree: Optional[int] = None,
) -> np.ndarray:
    """
    Interpolate dust grain charges to new positions using Vandermonde.

    Parameters
    ----------
    positions : np.ndarray, shape (N,)
        Known grain positions (normalized).
    charges : np.ndarray, shape (N,)
        Known grain charges Z_d at each position.
    eval_positions : np.ndarray, shape (M,)
        Positions where we want to evaluate the interpolated charge.
    degree : int, optional
        Polynomial degree for fit. Default: N-1.

    Returns
    -------
    interp_charges : np.ndarray, shape (M,)
        Interpolated charge values at eval_positions.
    """
    if degree is None:
        degree = min(len(positions) - 1, 8)  # Cap at degree 8 for stability

    # Fit polynomial to known charges
    V = vandermonde_matrix(positions, degree)
    # Use least-squares if overdetermined
    if len(positions) > degree + 1:
        coeffs, _, _, _ = np.linalg.lstsq(V, charges, rcond=None)
    else:
        coeffs = bjorck_pereyra_solve(positions[:degree + 1], charges[:degree + 1])

    # Evaluate at new positions
    V_eval = vandermonde_matrix(eval_positions, degree)
    return V_eval @ coeffs


def vandermonde_condition_number(x: np.ndarray) -> float:
    """
    Estimate the condition number of the Vandermonde matrix.

    For dusty plasma simulations, we need to check if the grain
    positions lead to a well-conditioned interpolation problem.
    Vandermonde matrices are notoriously ill-conditioned for
    high degrees and equally-spaced nodes.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Interpolation nodes.

    Returns
    -------
    cond : float
        Condition number of V.
    """
    V = vandermonde_matrix(x)
    return float(np.linalg.cond(V))
