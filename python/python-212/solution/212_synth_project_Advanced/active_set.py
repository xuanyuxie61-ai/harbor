"""
active_set.py
=============
Primal-dual active-set strategy for identifying and enforcing the
inequality constraints in the KKT system.

Mathematical background
-----------------------
The box constraints  u_a <= u(x) <= u_b  and the integral constraint
c^T u <= E_max  define a closed convex feasible set

    K = { u in R^{N^2} : u_a <= u_i <= u_b,  c^T u <= E_max }.

The KKT complementarity conditions for the box constraints are:

    For each i:
        u_i = u_a  =>  (nabla_u L)_i >= 0   (lower active)
        u_a < u_i < u_b  =>  (nabla_u L)_i = 0   (inactive)
        u_i = u_b  =>  (nabla_u L)_i <= 0   (upper active)

The primal-dual active-set method (a.k.a. semismooth Newton method)
iteratively:
  1. Identifies the active sets  A_low, A_up  from the current iterate.
  2. Solves the reduced KKT system on the inactive set  I = complement.
  3. Updates the active sets based on the sign of the Lagrange multipliers.

Convergence is typically finite (the active set stabilises after a few
iterations) and locally superlinear.

KKT role
--------
This module is the core mechanism for enforcing inequality constraints
in the KKT framework.  Without it, we could only handle equality-
constrained problems.
"""

from __future__ import annotations
import numpy as np

from physics_models import PhysicalParameters


# ---------------------------------------------------------------------------
# Active-set identification
# ---------------------------------------------------------------------------

def identify_active_sets(
    u: np.ndarray,
    grad_u: np.ndarray,
    params: PhysicalParameters,
    tol: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify the active and inactive index sets for box constraints.

    Parameters
    ----------
    u       : (N^2,) current control iterate
    grad_u  : (N^2,) gradient of the Lagrangian w.r.t. u  (excluding box multipliers)
    params  : physical parameters (provides u_lower, u_upper)
    tol     : tolerance for active-set decision

    Returns
    -------
    active_lower : boolean mask, True where u_i = u_lower  (lower bound active)
    active_upper : boolean mask, True where u_i = u_upper  (upper bound active)
    inactive     : boolean mask, True where u_a < u_i < u_b  (inactive)

    The decision rule is:
        i in A_low  if  u_i <= u_lower + tol  OR  (u_i - u_lower < tol and grad_u_i > 0)
        i in A_up   if  u_i >= u_upper - tol  OR  (u_upper - u_i < tol and grad_u_i < 0)
        i in I      otherwise
    """
    u = np.asarray(u, dtype=np.float64).ravel()
    grad_u = np.asarray(grad_u, dtype=np.float64).ravel()
    n = u.size

    at_lower = u <= params.u_lower + tol
    at_upper = u >= params.u_upper - tol
    push_lower = (u - params.u_lower < tol) & (grad_u > 0.0)
    push_upper = (params.u_upper - u < tol) & (grad_u < 0.0)

    active_lower = at_lower | push_lower
    active_upper = at_upper | push_upper
    # Ensure mutual exclusion (a point cannot be in both active sets)
    both = active_lower & active_upper
    if np.any(both):
        # Resolve: keep the one with larger |grad|
        mask_fix = both & (np.abs(grad_u) > 0.0)
        active_lower[mask_fix] = grad_u[mask_fix] > 0.0
        active_upper[mask_fix] = ~active_lower[mask_fix]
        # If grad == 0, keep lower
        active_upper[both & ~mask_fix] = False
        active_lower[both & ~mask_fix] = True

    inactive = ~(active_lower | active_upper)
    return active_lower, active_upper, inactive


# ---------------------------------------------------------------------------
# Active-set update
# ---------------------------------------------------------------------------

def update_active_sets(
    u: np.ndarray,
    grad_u: np.ndarray,
    active_lower_prev: np.ndarray,
    active_upper_prev: np.ndarray,
    params: PhysicalParameters,
    tol: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Update the active sets and check for convergence.

    Returns
    -------
    active_lower, active_upper, inactive : updated masks
    changed : True if the active set changed from the previous iteration
    """
    active_lower, active_upper, inactive = identify_active_sets(
        u, grad_u, params, tol
    )
    changed = not (
        np.array_equal(active_lower, active_lower_prev)
        and np.array_equal(active_upper, active_upper_prev)
    )
    return active_lower, active_upper, inactive, changed


# ---------------------------------------------------------------------------
# Project onto box constraints
# ---------------------------------------------------------------------------

def project_box(u: np.ndarray, params: PhysicalParameters) -> np.ndarray:
    """Project u onto the box  [u_lower, u_upper].

        u_proj = clip(u, u_lower, u_upper)
    """
    return np.clip(u, params.u_lower, params.u_upper)


# ---------------------------------------------------------------------------
# Project onto integral constraint
# ---------------------------------------------------------------------------

def project_integral(
    u: np.ndarray,
    c_vec: np.ndarray,
    E_max: float,
) -> tuple[np.ndarray, float]:
    """Project u onto the half-space  c^T u <= E_max.

    If c^T u > E_max, the projection is

        u_proj = u - ((c^T u - E_max) / ||c||^2) * c

    Returns
    -------
    u_proj : projected control
    lam    : Lagrange multiplier for the integral constraint  (>= 0)
    """
    ctu = float(np.dot(c_vec, u))
    if ctu <= E_max:
        return u.copy(), 0.0
    c_norm2 = float(np.dot(c_vec, c_vec))
    if c_norm2 < 1.0e-30:
        return u.copy(), 0.0
    lam = (ctu - E_max) / c_norm2
    u_proj = u - lam * c_vec
    return u_proj, lam


# ---------------------------------------------------------------------------
# Compute box constraint multipliers
# ---------------------------------------------------------------------------

def compute_box_multipliers(
    grad_u: np.ndarray,
    active_lower: np.ndarray,
    active_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Lagrange multipliers for the box constraints.

    On the active lower set:  eta_i = max(0, grad_u_i)  (should be >= 0)
    On the active upper set:  theta_i = max(0, -grad_u_i)  (should be >= 0)
    On the inactive set:  eta_i = theta_i = 0.

    Returns
    -------
    eta   : (N^2,) multipliers for lower bound
    theta : (N^2,) multipliers for upper bound
    """
    eta = np.zeros_like(grad_u)
    theta = np.zeros_like(grad_u)
    eta[active_lower] = np.maximum(0.0, grad_u[active_lower])
    theta[active_upper] = np.maximum(0.0, -grad_u[active_upper])
    return eta, theta


# ---------------------------------------------------------------------------
# Strict complementarity check
# ---------------------------------------------------------------------------

def check_strict_complementarity(
    u: np.ndarray,
    eta: np.ndarray,
    theta: np.ndarray,
    params: PhysicalParameters,
    tol: float = 1.0e-6,
) -> dict:
    """Check the KKT strict complementarity conditions.

    Strict complementarity holds if for each i:
        u_i = u_lower  =>  eta_i > 0   (not just >= 0)
        u_i = u_upper  =>  theta_i > 0
        u_lower < u_i < u_upper  =>  eta_i = theta_i = 0

    Returns a dict with:
        'strict' : bool, True if strict complementarity holds
        'violations' : number of near-degenerate constraints
        'min_multiplier' : smallest non-zero multiplier
    """
    u = np.asarray(u).ravel()
    eta = np.asarray(eta).ravel()
    theta = np.asarray(theta).ravel()
    n = u.size

    violations = 0
    min_mult = float("inf")
    for i in range(n):
        if abs(u[i] - params.u_lower) < tol:
            if eta[i] < tol:
                violations += 1
            elif eta[i] < min_mult:
                min_mult = eta[i]
        elif abs(u[i] - params.u_upper) < tol:
            if theta[i] < tol:
                violations += 1
            elif theta[i] < min_mult:
                min_mult = theta[i]

    if min_mult == float("inf"):
        min_mult = 0.0
    return {
        "strict": violations == 0,
        "violations": violations,
        "min_multiplier": min_mult,
    }
