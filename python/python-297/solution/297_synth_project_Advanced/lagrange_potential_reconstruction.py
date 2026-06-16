"""
lagrange_potential_reconstruction.py
=======================================
Lagrange polynomial reconstruction for the plasma sheath potential
from discrete grain charge measurements.

Physical motivation:
    In dusty plasma experiments, the plasma sheath potential phi(z)
    varies across the discharge gap. We can reconstruct the potential
    from discrete measurements at grain locations using Lagrange
    interpolation.

    The Lagrange basis polynomials are:
        L_i(x) = prod_{j != i} (x - x_j) / (x_i - x_j)

    And the interpolant is:
        p(x) = sum_i f_i * L_i(x)

    For dusty plasma, we also need:
        1. The derivative p'(x) = -E(x) (electric field)
        2. The second derivative p''(x) ~ -rho(x)/eps0 (charge density)
    from the reconstructed potential.

    We check for Runge's phenomenon and use optimal node placement
    (Chebyshev nodes) when necessary.

References:
    - From 634_lagrange_basis_display: Lagrange basis evaluation
    - Trefethen, "Is Gauss Quadrature Better than Clenshaw-Curtis?",
      SIAM Rev. 50, 67-87 (2008)
"""

import numpy as np
from typing import Tuple, Optional, Callable


def lagrange_basis(
    x_nodes: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    """
    Evaluate all Lagrange basis polynomials at evaluation points.

    L_i(x) = prod_{j != i} (x - x_j) / (x_i - x_j)

    For dusty plasma, the nodes are grain positions and the basis
    functions are used to reconstruct the sheath potential.

    Parameters
    ----------
    x_nodes : np.ndarray, shape (M+1,)
        Interpolation nodes.
    x_eval : np.ndarray, shape (K,)
        Evaluation points.

    Returns
    -------
    L : np.ndarray, shape (K, M+1)
        L[k, i] = L_i(x_eval[k]).
    """
    M = len(x_nodes) - 1
    K = len(x_eval)
    L = np.zeros((K, M + 1), dtype=np.float64)

    for i in range(M + 1):
        L[:, i] = 1.0
        for j in range(M + 1):
            if j != i:
                denom = x_nodes[i] - x_nodes[j]
                if abs(denom) < 1e-30:
                    raise ValueError(
                        f"Coincident nodes at i={i}, j={j}: "
                        f"x_nodes[{i}]={x_nodes[i]}, x_nodes[{j}]={x_nodes[j]}"
                    )
                L[:, i] *= (x_eval - x_nodes[j]) / denom

    return L


def lagrange_interpolation(
    x_nodes: np.ndarray,
    f_values: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    """
    Lagrange polynomial interpolation.

    p(x) = sum_i f_i * L_i(x)

    Parameters
    ----------
    x_nodes : np.ndarray, shape (M+1,)
        Interpolation nodes.
    f_values : np.ndarray, shape (M+1,)
        Function values at nodes.
    x_eval : np.ndarray, shape (K,)
        Evaluation points.

    Returns
    -------
    f_interp : np.ndarray, shape (K,)
        Interpolated values.
    """
    L = lagrange_basis(x_nodes, x_eval)
    return L @ f_values


def lagrange_derivative(
    x_nodes: np.ndarray,
    f_values: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    """
    Derivative of the Lagrange interpolant.

    p'(x) = sum_i f_i * L_i'(x)
    where L_i'(x) = sum_{k != i} prod_{j != i, j != k} (x - x_j) /
                    prod_{j != i} (x_i - x_j)

    For dusty plasma: E(x) = -p'(x) is the electric field.

    Parameters
    ----------
    x_nodes : np.ndarray, shape (M+1,)
        Interpolation nodes.
    f_values : np.ndarray, shape (M+1,)
        Function values at nodes.
    x_eval : np.ndarray, shape (K,)
        Evaluation points.

    Returns
    -------
    df : np.ndarray, shape (K,)
        Derivative of interpolant at evaluation points.
    """
    M = len(x_nodes) - 1
    K = len(x_eval)
    dL = np.zeros((K, M + 1), dtype=np.float64)

    for i in range(M + 1):
        # Compute denominator prod_{j != i} (x_i - x_j)
        denom = 1.0
        for j in range(M + 1):
            if j != i:
                denom *= (x_nodes[i] - x_nodes[j])

        # Compute derivative of numerator
        # d/dx prod_{j != i} (x - x_j) = sum_{k != i} prod_{j != i, j != k} (x - x_j)
        for k_idx in range(K):
            x = x_eval[k_idx]
            deriv_sum = 0.0
            for k in range(M + 1):
                if k == i:
                    continue
                prod = 1.0
                for j in range(M + 1):
                    if j != i and j != k:
                        prod *= (x - x_nodes[j])
                deriv_sum += prod
            dL[k_idx, i] = deriv_sum / denom

    return dL @ f_values


def chebyshev_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """
    Compute Chebyshev nodes on [a, b].

    x_k = (a+b)/2 + (b-a)/2 * cos((2k+1)*pi/(2n)) for k=0,...,n-1

    Chebyshev nodes minimize the Lebesgue constant and avoid
    Runge's phenomenon for polynomial interpolation.

    For dusty plasma, these are optimal positions for placing
    diagnostic grains to measure the sheath potential.

    Parameters
    ----------
    n : int
        Number of nodes.
    a, b : float
        Interval endpoints.

    Returns
    -------
    nodes : np.ndarray, shape (n,)
        Chebyshev nodes.
    """
    k = np.arange(n)
    nodes_ref = np.cos((2 * k + 1) * np.pi / (2 * n))
    return 0.5 * (a + b) + 0.5 * (b - a) * nodes_ref


def reconstruct_sheath_potential(
    z_grains: np.ndarray,
    phi_grains: np.ndarray,
    z_eval: np.ndarray,
    use_chebyshev: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct the plasma sheath potential from grain measurements.

    In a RF discharge, the sheath potential is approximately:
        phi(z) = phi_0 * (z/d)^alpha
    where d is the sheath thickness and alpha ~ 5/3 for a collisionless
    Child-Langmuir sheath.

    Parameters
    ----------
    z_grains : np.ndarray, shape (M,)
        Vertical positions of dust grains [m].
    phi_grains : np.ndarray, shape (M,)
        Measured potential at grain positions [V].
    z_eval : np.ndarray, shape (K,)
        Positions where we want the potential.
    use_chebyshev : bool
        If True, use Chebyshev nodes from the grain positions.

    Returns
    -------
    phi_interp : np.ndarray, shape (K,)
        Interpolated potential.
    E_field : np.ndarray, shape (K,)
        Electric field E = -dphi/dz.
    charge_density : np.ndarray, shape (K,)
        Estimated charge density rho ~ eps0 * d^2phi/dz^2.
    """
    if use_chebyshev and len(z_grains) >= 3:
        # Select Chebyshev-distributed subset of grains
        n_use = min(len(z_grains), 8)
        z_cheb = chebyshev_nodes(n_use, z_grains.min(), z_grains.max())
        # Find nearest actual grain positions
        indices = []
        for zc in z_cheb:
            idx = np.argmin(np.abs(z_grains - zc))
            indices.append(idx)
        indices = list(set(indices))  # Remove duplicates
        z_use = z_grains[indices]
        f_use = phi_grains[indices]
    else:
        z_use = z_grains
        f_use = phi_grains

    phi_interp = lagrange_interpolation(z_use, f_use, z_eval)
    E_field = -lagrange_derivative(z_use, f_use, z_eval)

    # Second derivative for charge density (via finite differences on dphi/dz)
    dz = np.gradient(z_eval)
    dz = np.where(np.abs(dz) < 1e-30, 1e-30, dz)
    dphi_dz = -E_field
    d2phi_dz2 = np.gradient(dphi_dz, dz)

    from dusty_plasma_physics_constants import VACUUM_PERMITTIVITY
    charge_density = -VACUUM_PERMITTIVITY * d2phi_dz2

    return phi_interp, E_field, charge_density


def lebesgue_constant(x_nodes: np.ndarray, n_eval: int = 1000) -> float:
    """
    Compute the Lebesgue constant for a set of interpolation nodes.

    Lambda = max_x sum_i |L_i(x)|

    The Lebesgue constant bounds the interpolation error:
        ||f - p||_inf <= (1 + Lambda) * ||f - p*||_inf
    where p* is the best polynomial approximation.

    For equally-spaced nodes, Lambda ~ 2^n/(n*log(n)) (exponential growth).
    For Chebyshev nodes, Lambda ~ (2/pi)*log(n) (logarithmic growth).

    Parameters
    ----------
    x_nodes : np.ndarray, shape (M+1,)
        Interpolation nodes.
    n_eval : int
        Number of evaluation points for estimating the max.

    Returns
    -------
    Lambda : float
        Lebesgue constant.
    """
    x_min, x_max = x_nodes.min(), x_nodes.max()
    margin = 0.01 * (x_max - x_min)
    x_eval = np.linspace(x_min - margin, x_max + margin, n_eval)

    L = lagrange_basis(x_nodes, x_eval)
    lambda_func = np.sum(np.abs(L), axis=1)
    return float(np.max(lambda_func))


def lagrange_basis_summary(x_nodes: np.ndarray) -> dict:
    """
    Summary of Lagrange basis properties for given nodes (from 634_lagrange_basis_display).

    Parameters
    ----------
    x_nodes : np.ndarray
        Interpolation nodes.

    Returns
    -------
    summary : dict
        'n_nodes': number of nodes,
        'lebesgue': Lebesgue constant,
        'spacing': node spacings,
        'min_spacing': minimum spacing,
        'condition_number': Vandermonde condition number.
    """
    from vandermonde_charge_interpolation import vandermonde_condition_number

    spacings = np.diff(np.sort(x_nodes))
    return {
        "n_nodes": len(x_nodes),
        "lebesgue": lebesgue_constant(x_nodes),
        "spacing": spacings,
        "min_spacing": float(np.min(spacings)) if len(spacings) > 0 else 0.0,
        "condition_number": vandermonde_condition_number(x_nodes),
    }
