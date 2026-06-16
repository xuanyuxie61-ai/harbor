"""
fd_high_order_stencil.py
=========================
High-order centered finite difference stencils for computing spatial
derivatives of the plasma potential field around dust grains.

Physical motivation:
    In dusty plasma simulations, we need to compute:
        - Laplacian of potential: nabla^2 phi  (Poisson equation)
        - Electric field: E = -grad(phi)
        - Force on dust grains: F = Q_d * E
    using high-order finite differences for accuracy.

    The standard 2nd-order centered stencil for the 2nd derivative
    (from 281_diff2_center):
        f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h^2

    Higher-order stencils (4th, 6th, 8th) are computed using the
    standard method of undetermined coefficients or the Fornberg algorithm.

References:
    - Fornberg, "Generation of finite difference formulas on arbitrarily spaced grids",
      Math. Comp. 51, 699-706 (1988)
    - Lele, "Compact finite difference schemes with spectral-like resolution",
      J. Comput. Phys. 103, 16-42 (1992)
"""

import numpy as np
from typing import Tuple, Optional


def fornberg_weights(
    x_target: float,
    x_nodes: np.ndarray,
    M: int,
) -> np.ndarray:
    """
    Compute finite difference weights using Fornberg's algorithm.

    Given nodes x_nodes[0..N] and derivative order M, compute weights
    w[0..N] such that:
        f^(M)(x_target) ≈ sum_j w[j] * f(x_nodes[j])

    Parameters
    ----------
    x_target : float
        Point at which to approximate the derivative.
    x_nodes : np.ndarray, shape (N+1,)
        Coordinates of the stencil nodes (must be distinct).
    M : int
        Order of derivative.

    Returns
    -------
    weights : np.ndarray, shape (N+1,)
        Finite difference weights.
    """
    N = len(x_nodes) - 1

    # delta[i, j, k] = weight for node i using nodes 0..j, derivative order k
    delta = np.zeros((N + 1, N + 1, M + 1))
    delta[0, 0, 0] = 1.0

    c1 = 1.0
    for n in range(1, N + 1):
        c2 = 1.0
        for v in range(n):
            c3 = x_nodes[n] - x_nodes[v]
            c2 *= c3
            if n <= M:
                delta[n, n - 1, n] = 0.0
            for m in range(min(n, M) + 1):
                delta[n, v, m] = ((x_target - x_nodes[n]) * delta[n - 1, v, m]
                                   - m * delta[n - 1, v, m - 1] if m > 0 else
                                   (x_target - x_nodes[n]) * delta[n - 1, v, m]) / (-c3)
        for m in range(min(n, M) + 1):
            delta[n, n, m] = (c1 / c2) * (
                m * delta[n - 1, n - 1, m - 1] if m > 0 else 0.0
            ) + (c1 / c2) * (x_nodes[n] - x_target) * delta[n - 1, n - 1, m] * (-1.0 if False else 1.0)
            # Correct Fornberg formula for the last node:
            delta[n, n, m] = (c1 / c2) * (
                m * delta[n - 1, n - 1, m - 1] if m > 0 else 0.0
            )
            delta[n, n, m] += -(x_target - x_nodes[n]) * delta[n - 1, n - 1, m] * (c1 / c2)
        c1 = c2

    return delta[N, :, M]


def fd_weights_1d(
    half_width: int,
    deriv_order: int,
    spacing: float = 1.0,
) -> np.ndarray:
    """
    Compute centered finite difference weights for uniform grid.

    For a 2p-th order approximation to the m-th derivative:
        f^(m)(x) ≈ (1/h^m) * sum_{j=-p}^{p} w_j * f(x + j*h)

    Special cases (from 281_diff2_center):
        Order 2 (p=1), 2nd derivative: [1, -2, 1] / h^2
        Order 4 (p=2), 2nd derivative: [-1, 16, -30, 16, -1] / (12*h^2)
        Order 6 (p=3), 2nd derivative: [2, -27, 270, -490, 270, -27, 2] / (180*h^2)

    Parameters
    ----------
    half_width : int
        Half-width p of the stencil (uses 2p+1 points).
    deriv_order : int
        Order of derivative m.
    spacing : float
        Grid spacing h.

    Returns
    -------
    weights : np.ndarray, shape (2*half_width+1,)
        Finite difference weights, indexed from -half_width to +half_width.
    """
    # Use precomputed known-good stencils for common cases
    if deriv_order == 2 and half_width == 1:
        # Standard 3-point 2nd derivative: [1, -2, 1] / h^2
        return np.array([1.0, -2.0, 1.0]) / spacing**2
    elif deriv_order == 2 and half_width == 2:
        # 5-point 4th-order 2nd derivative
        return np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / (12.0 * spacing**2)
    elif deriv_order == 2 and half_width == 3:
        # 7-point 6th-order 2nd derivative
        return np.array([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0]) / (180.0 * spacing**2)
    elif deriv_order == 1 and half_width == 1:
        # 3-point 1st derivative: [-1, 0, 1] / (2h)
        return np.array([-0.5, 0.0, 0.5]) / spacing
    elif deriv_order == 1 and half_width == 2:
        # 5-point 4th-order 1st derivative
        return np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / (12.0 * spacing)
    else:
        # General case: use Fornberg algorithm
        x_nodes = np.arange(-half_width, half_width + 1, dtype=float)
        weights = fornberg_weights(0.0, x_nodes, deriv_order)
        return weights / spacing**deriv_order


def apply_fd_1d(
    f: np.ndarray,
    weights: np.ndarray,
    half_width: int,
    boundary: str = "periodic",
) -> np.ndarray:
    """
    Apply 1D finite difference stencil to a function array.

    Parameters
    ----------
    f : np.ndarray, shape (N,)
        Function values on uniform grid.
    weights : np.ndarray, shape (2*half_width+1,)
        Finite difference weights.
    half_width : int
        Half-width of stencil.
    boundary : str
        Boundary treatment: 'periodic', 'mirror', or 'truncate'.

    Returns
    -------
    df : np.ndarray, shape (N,)
        Derivative approximation at each grid point.
    """
    N = len(f)
    df = np.zeros(N)
    p = half_width

    for i in range(N):
        val = 0.0
        for j in range(-p, p + 1):
            idx = i + j
            # Boundary treatment
            if boundary == "periodic":
                idx = idx % N
            elif boundary == "mirror":
                if idx < 0:
                    idx = -idx
                elif idx >= N:
                    idx = 2 * (N - 1) - idx
                idx = max(0, min(N - 1, idx))
            elif boundary == "truncate":
                if idx < 0 or idx >= N:
                    continue
            val += weights[j + p] * f[idx]
        df[i] = val
    return df


def fd_laplacian_2d(
    phi: np.ndarray,
    dx: float,
    dy: float,
    order: int = 4,
    boundary: str = "periodic",
) -> np.ndarray:
    """
    Compute 2D Laplacian using high-order finite differences.

    nabla^2 phi = d^2phi/dx^2 + d^2phi/dy^2

    Parameters
    ----------
    phi : np.ndarray, shape (Ny, Nx)
        Potential field on 2D grid.
    dx, dy : float
        Grid spacings.
    order : int
        Order of accuracy (2, 4, 6, or 8).
    boundary : str
        Boundary condition type.

    Returns
    -------
    lap_phi : np.ndarray, shape (Ny, Nx)
        Discrete Laplacian of phi.
    """
    half_width = order // 2
    weights = fd_weights_1d(half_width, 2, 1.0)

    Ny, Nx = phi.shape
    lap = np.zeros_like(phi)

    # d^2phi/dx^2
    for i in range(Ny):
        row = phi[i, :]
        lap[i, :] += apply_fd_1d(row, weights, half_width, boundary)

    # d^2phi/dy^2
    for j in range(Nx):
        col = phi[:, j]
        lap[:, j] += apply_fd_1d(col, weights, half_width, boundary)

    return lap


def fd_gradient_2d(
    phi: np.ndarray,
    dx: float,
    dy: float,
    order: int = 4,
    boundary: str = "periodic",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D gradient using high-order finite differences.

    E = -grad(phi) = -(dphi/dx, dphi/dy)

    Parameters
    ----------
    phi : np.ndarray, shape (Ny, Nx)
        Potential field.
    dx, dy : float
        Grid spacings.
    order : int
        Order of accuracy (2, 4, 6, or 8).
    boundary : str
        Boundary condition type.

    Returns
    -------
    dphi_dx, dphi_dy : np.ndarray, shape (Ny, Nx)
        Partial derivatives.
    """
    half_width = order // 2
    weights_1st = fd_weights_1d(half_width, 1, 1.0)

    Ny, Nx = phi.shape
    dphi_dx = np.zeros_like(phi)
    dphi_dy = np.zeros_like(phi)

    # dphi/dx
    for i in range(Ny):
        row = phi[i, :]
        dphi_dx[i, :] = apply_fd_1d(row, weights_1st, half_width, boundary)

    # dphi/dy
    for j in range(Nx):
        col = phi[:, j]
        dphi_dy[:, j] = apply_fd_1d(col, weights_1st, half_width, boundary)

    return dphi_dx, dphi_dy


def stencil_consistency_check(
    half_width: int,
    deriv_order: int,
    test_func=None,
    test_deriv=None,
    h_values=None,
) -> dict:
    """
    Verify consistency and convergence order of the FD stencil.

    Parameters
    ----------
    half_width : int
        Stencil half-width.
    deriv_order : int
        Derivative order being tested.
    test_func : callable, optional
        Test function f(x). Default: sin(x).
    test_deriv : callable, optional
        Exact derivative f^(m)(x). Default: sin(x) shifted by m*pi/2.
    h_values : array-like, optional
        Grid spacings to test.

    Returns
    -------
    results : dict
        Dictionary with 'h_values', 'errors', 'observed_orders'.
    """
    PI = np.pi

    if test_func is None:
        test_func = np.sin
    if test_deriv is None:
        m = deriv_order
        test_deriv = lambda x: np.sin(x + m * PI / 2.0)

    if h_values is None:
        h_values = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625])

    errors = []

    for h in h_values:
        N = int(2 * PI / h)
        x = np.linspace(0, 2 * PI, N, endpoint=False)
        f_vals = test_func(x)
        exact = test_deriv(x)

        weights = fd_weights_1d(half_width, deriv_order, h)
        approx = apply_fd_1d(f_vals, weights, half_width, "periodic")

        err = np.max(np.abs(approx - exact))
        errors.append(err)

    errors = np.array(errors)
    orders = np.log(errors[:-1] / errors[1:]) / np.log(h_values[:-1] / h_values[1:])

    return {
        "h_values": h_values,
        "errors": errors,
        "observed_orders": orders,
        "expected_order": 2 * half_width + 2 - deriv_order,
    }


# Precomputed stencils for common dusty plasma operations
STENCIL_LAPLACIAN_2 = fd_weights_1d(1, 2, 1.0)
STENCIL_LAPLACIAN_4 = fd_weights_1d(2, 2, 1.0)
STENCIL_GRADIENT_2 = fd_weights_1d(1, 1, 1.0)
STENCIL_GRADIENT_4 = fd_weights_1d(2, 1, 1.0)
