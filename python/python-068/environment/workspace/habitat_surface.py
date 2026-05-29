"""
habitat_surface.py
Bicubic Bezier surface evaluation for spatially heterogeneous habitat suitability.

Adapted from:
  - 083_bezier_surface: Bicubic Bezier patch evaluation and topology

Role in synthesis:
  Models spatially varying carrying capacity K(x,y) and intrinsic growth rate r(x,y)
  as smooth bicubic Bezier surfaces, representing heterogeneous landscapes.
"""

import numpy as np


def bernstein_basis(u: float) -> np.ndarray:
    """
    Cubic Bernstein basis vector at parameter u:
    B(u) = [(1-u)^3, 3u(1-u)^2, 3u^2(1-u), u^3]
    """
    u = np.clip(u, 0.0, 1.0)
    return np.array([
        (1.0 - u) ** 3,
        3.0 * u * (1.0 - u) ** 2,
        3.0 * u ** 2 * (1.0 - u),
        u ** 3
    ])


def bezier_patch_evaluate(control_points: np.ndarray, u: float, v: float) -> float:
    """
    Evaluate a scalar bicubic Bezier patch at parameter (u, v).

    Parameters
    ----------
    control_points : ndarray, shape (4, 4)
        Control point matrix.
    u, v : float
        Parameters in [0, 1].

    Returns
    -------
    value : float
        Evaluated patch value.
    """
    uvec = bernstein_basis(u)
    vvec = bernstein_basis(v)
    return float(uvec @ control_points @ vvec)


def bezier_surface_grid(control_points: np.ndarray, nu: int = 64, nv: int = 64) -> np.ndarray:
    """
    Evaluate Bezier surface on a regular grid.

    Returns
    -------
    Z : ndarray, shape (nu, nv)
    """
    u = np.linspace(0.0, 1.0, nu)
    v = np.linspace(0.0, 1.0, nv)
    Z = np.zeros((nu, nv))
    for i in range(nu):
        uvec = bernstein_basis(u[i])
        for j in range(nv):
            vvec = bernstein_basis(v[j])
            Z[i, j] = uvec @ control_points @ vvec
    return Z


def create_habitat_carrying_capacity(
    nx: int = 64,
    ny: int = 64,
    K_base: float = 100.0,
    K_peak: float = 200.0
) -> np.ndarray:
    """
    Create a spatially varying carrying capacity map using a bicubic Bezier surface.
    Represents a landscape with a central high-quality habitat patch.

    The control points are designed to produce a smooth dome-shaped habitat:
    K(x,y) = K_base + (K_peak - K_base) * B(u,v)
    """
    # Control points for a dome-shaped surface (center elevated)
    cp = np.array([
        [0.0, 0.1, 0.1, 0.0],
        [0.1, 0.5, 0.5, 0.1],
        [0.1, 0.5, 0.5, 0.1],
        [0.0, 0.1, 0.1, 0.0]
    ], dtype=float)
    B = bezier_surface_grid(cp, nx, ny)
    K = K_base + (K_peak - K_base) * B
    return K


def create_growth_rate_map(
    nx: int = 64,
    ny: int = 64,
    r_base: float = 0.5,
    r_peak: float = 1.5
) -> np.ndarray:
    """
    Create spatially varying intrinsic growth rate map.
    Correlated with carrying capacity (richer habitats support faster growth).
    """
    cp = np.array([
        [0.0, 0.05, 0.05, 0.0],
        [0.05, 0.4, 0.4, 0.05],
        [0.05, 0.4, 0.4, 0.05],
        [0.0, 0.05, 0.05, 0.0]
    ], dtype=float)
    B = bezier_surface_grid(cp, nx, ny)
    r = r_base + (r_peak - r_base) * B
    return r


def bezier_surface_gradient(control_points: np.ndarray, u: float, v: float) -> tuple[float, float]:
    """
    Compute gradient of Bezier surface w.r.t. parameters (u, v).
    dB/du = 3 * sum_{i=0}^{2} [B_i^2(u) * (P_{i+1,j} - P_{i,j})]
    """
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)

    # Derivative of Bernstein basis
    dBu = np.array([
        -3.0 * (1.0 - u) ** 2,
        3.0 * (1.0 - u) ** 2 - 6.0 * u * (1.0 - u),
        6.0 * u * (1.0 - u) - 3.0 * u ** 2,
        3.0 * u ** 2
    ])
    dBv = np.array([
        -3.0 * (1.0 - v) ** 2,
        3.0 * (1.0 - v) ** 2 - 6.0 * v * (1.0 - v),
        6.0 * v * (1.0 - v) - 3.0 * v ** 2,
        3.0 * v ** 2
    ])

    du = float(dBu @ control_points @ bernstein_basis(v))
    dv = float(bernstein_basis(u) @ control_points @ dBv)
    return du, dv
