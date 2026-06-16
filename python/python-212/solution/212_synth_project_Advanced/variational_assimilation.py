"""
variational_assimilation.py
===========================
Data assimilation framework for generating noisy observations, building
covariance matrices, and constructing the target state  y_d.

This module adapts the 4D-Var variational data assimilation approach
(from project 1124_rspence821505_Variational-Data-Consistent-Assimilation
and 1247_vcasasmo_BayRad3D) to our steady-state PDE-constrained
optimisation setting.

In 4D-Var the cost function is

    J(z0) = 0.5 (z0 - z_b)^T B^{-1} (z0 - z_b)
          + 0.5 sum_k (H M^k z0 - y_obs_k)^T R^{-1} (H M^k z0 - y_obs_k)

For our steady-state problem, the "time propagation" M^k is replaced
by the PDE solution operator  A^{-1}, and the "background state" z_b
becomes the background control u_b.

KKT role
--------
The observation operator H_obs and the covariance matrices B, R enter
the KKT system through the cost functional and the observation term
in the right-hand side.
"""

from __future__ import annotations
import numpy as np

from physics_models import PhysicalParameters
from matrix_kernels import (
    build_background_covariance,
    build_observation_covariance,
    build_observation_operator,
)


# ---------------------------------------------------------------------------
# Target state generation  (inspired by double_c_data, project 314)
# ---------------------------------------------------------------------------

def generate_double_c_target(
    XX: np.ndarray,
    YY: np.ndarray,
    n1: int = 80,
    n2: int = 80,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a target state inspired by the 'double-C' data geometry.

    The double-C data consists of two nested C-shaped curves that do not
    intersect but are closely intertwined.  We use this geometry to create
    a challenging target state with sharp gradients and non-trivial topology.

    Specifically, we define

        y_d(x, y) = phi_C1(x, y) - phi_C2(x, y)

    where phi_Ci is a smoothed indicator of the i-th C-curve:

        C1:  centre (1, 0), radius in [2, 5], angle in [0.5 pi, 1.5 pi]
        C2:  centre (0, 3.5), radius in [2, 5], angle in [1.5 pi, 2.5 pi]

    We rescale to the unit domain [0,1]^2.

    Returns
    -------
    y_d_clean : (N^2,) clean target state
    y_class   : (N^2,) class labels (1 or 2) for each grid point
    """
    rng = np.random.default_rng(seed)
    x = XX.ravel()
    y = YY.ravel()

    # Scale to the double-C domain
    x_sc = x * 5.0
    y_sc = y * 5.0

    # C1: centred at (1, 0), angles in [0.5 pi, 1.5 pi]
    dx1 = x_sc - 1.0
    dy1 = y_sc - 0.0
    r1 = np.sqrt(dx1 ** 2 + dy1 ** 2)
    theta1 = np.arctan2(dy1, dx1)
    mask1 = (r1 >= 2.0) & (r1 <= 5.0) & (theta1 >= 0.5 * np.pi) & (theta1 <= 1.5 * np.pi)

    # C2: centred at (0, 3.5), angles in [1.5 pi, 2.5 pi]
    dx2 = x_sc - 0.0
    dy2 = y_sc - 3.5
    r2 = np.sqrt(dx2 ** 2 + dy2 ** 2)
    theta2 = np.arctan2(dy2, dx2)
    mask2 = (r2 >= 2.0) & (r2 <= 5.0) & (theta2 >= 1.5 * np.pi) & (theta2 <= 2.5 * np.pi)

    # Smooth indicator:  phi = exp(-dist^2 / sigma^2)
    sigma = 0.3
    dist1 = np.minimum(np.abs(r1 - 2.0), np.abs(r1 - 5.0))
    dist2 = np.minimum(np.abs(r2 - 2.0), np.abs(r2 - 5.0))
    phi1 = np.exp(-dist1 ** 2 / sigma ** 2) * mask1.astype(float)
    phi2 = np.exp(-dist2 ** 2 / sigma ** 2) * mask2.astype(float)

    y_d_clean = phi1 - phi2
    y_class = np.where(mask1, 1, np.where(mask2, 2, 0))

    return y_d_clean, y_class


# ---------------------------------------------------------------------------
# Observation generation
# ---------------------------------------------------------------------------

def generate_noisy_observations(
    y_true: np.ndarray,
    obs_indices: np.ndarray,
    obs_var: float = 1.0e-2,
    seed: int = 123,
) -> np.ndarray:
    """Generate noisy observations  y_obs = H y_true + epsilon,  epsilon ~ N(0, R).

    Parameters
    ----------
    y_true      : (N^2,) true state
    obs_indices : indices of observed components
    obs_var     : observation error variance
    seed        : random seed

    Returns
    -------
    y_obs : (n_obs,) noisy observations
    """
    rng = np.random.default_rng(seed)
    y_obs = y_true[obs_indices] + rng.normal(0.0, np.sqrt(obs_var), size=obs_indices.size)
    return y_obs


# ---------------------------------------------------------------------------
# Build the target state from observations  (analysis step)
# ---------------------------------------------------------------------------

def build_target_from_observations(
    y_obs: np.ndarray,
    obs_indices: np.ndarray,
    N: int,
    obs_var: float = 1.0e-2,
    background_var: float = 1.0,
    correlation_length: float = 0.2,
) -> np.ndarray:
    """Construct the target state  y_d  by interpolating noisy observations
    onto the full grid using a simple distance-weighted average.

    This is a simplified analysis step; the full 4D-Var analysis would
    solve  (H^T R^{-1} H + B^{-1}) y_d = H^T R^{-1} y_obs + B^{-1} y_b.

    Here we use a Gaussian-kernel interpolation:

        y_d(x_i) = sum_k  w_k y_obs_k  /  sum_k w_k

    where  w_k = exp(-|x_i - x_k|^2 / (2 L^2))  and L is the correlation
    length.
    """
    from pde_operator import create_grid

    _, _, _, XX, YY = create_grid(N)
    x_flat = XX.ravel()
    y_flat = YY.ravel()

    # Observation locations
    obs_x = x_flat[obs_indices]
    obs_y = y_flat[obs_indices]

    n2 = N * N
    y_d = np.zeros(n2, dtype=np.float64)
    L2 = 2.0 * correlation_length ** 2

    for i in range(n2):
        dist2 = (x_flat[i] - obs_x) ** 2 + (y_flat[i] - obs_y) ** 2
        weights = np.exp(-dist2 / L2)
        w_sum = np.sum(weights)
        if w_sum > 1.0e-30:
            y_d[i] = np.dot(weights, y_obs) / w_sum
        else:
            y_d[i] = 0.0
    return y_d


# ---------------------------------------------------------------------------
# Full 4D-Var-style cost function evaluation
# ---------------------------------------------------------------------------

def fourdvar_cost(
    y: np.ndarray,
    y_b: np.ndarray,
    y_obs: np.ndarray,
    H_obs: np.ndarray,
    B_inv: np.ndarray,
    R_inv: np.ndarray,
) -> float:
    """Evaluate the 4D-Var cost function:

        J = 0.5 (y - y_b)^T B^{-1} (y - y_b)
          + 0.5 (H y - y_obs)^T R^{-1} (H y - y_obs)

    This is used as a diagnostic for the data-assimilation quality.
    """
    diff_b = np.asarray(y).ravel() - np.asarray(y_b).ravel()
    J_b = 0.5 * float(np.dot(diff_b, B_inv @ diff_b))

    Hy = H_obs @ np.asarray(y).ravel()
    diff_o = Hy - np.asarray(y_obs).ravel()
    J_o = 0.5 * float(np.dot(diff_o, R_inv @ diff_o))

    return J_b + J_o


# ---------------------------------------------------------------------------
# Background state for control
# ---------------------------------------------------------------------------

def build_background_control(N: int, params: PhysicalParameters) -> np.ndarray:
    """Build a smooth background control  u_b  that satisfies the box
    constraints  u_lower <= u_b <= u_upper.

    We use a low-frequency sinusoidal pattern:

        u_b(x, y) = u_mid + A * sin(2 pi x) * sin(2 pi y)

    where u_mid = (u_lower + u_upper) / 2 and A is chosen to stay within bounds.
    """
    from pde_operator import create_grid

    _, _, _, XX, YY = create_grid(N)
    u_mid = 0.5 * (params.u_lower + params.u_upper)
    A = 0.3 * (params.u_upper - params.u_lower)
    u_b = u_mid + A * np.sin(2.0 * np.pi * XX) * np.sin(2.0 * np.pi * YY)
    # Clamp to box
    u_b = np.clip(u_b, params.u_lower, params.u_upper)
    return u_b.ravel()
