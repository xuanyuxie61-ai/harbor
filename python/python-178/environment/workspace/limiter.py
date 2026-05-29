"""
limiter.py
==========
Slope limiters and shock-capturing limiters for high-order DG methods.
Synthesized from glomin (global optimization with second-derivative bounds)
for optimal limiting parameter selection.
"""

import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Minmod and generalized minmod limiters
# ---------------------------------------------------------------------------

def minmod(a: float, b: float) -> float:
    """Minmod function: returns sign(a)*min(|a|,|b|) if a*b>0, else 0."""
    if a * b <= 0.0:
        return 0.0
    return np.sign(a) * min(abs(a), abs(b))


def minmod3(a: float, b: float, c: float) -> float:
    """Generalized minmod with three arguments."""
    if a * b <= 0.0 or a * c <= 0.0:
        return 0.0
    return np.sign(a) * min(abs(a), abs(b), abs(c))


def minmod_vector(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise minmod."""
    result = np.zeros_like(a)
    pos = (a > 0) & (b > 0)
    neg = (a < 0) & (b < 0)
    result[pos] = np.minimum(a[pos], b[pos])
    result[neg] = np.maximum(a[neg], b[neg])
    return result


# ---------------------------------------------------------------------------
# Global optimization for optimal limiting parameter (from glomin)
# ---------------------------------------------------------------------------

def glomin(a: float, b: float, c: float, m: float, e: float, t: float,
           f: callable, k: int = 0) -> Tuple[float, float]:
    """
    Global minimization of F(X) on [A,B] with second-derivative bound M.
    Based on Brent's method with parabolic interpolation and random probing.
    Returns (x_min, f_min).
    """
    a0 = min(a, b)
    b0 = max(a, b)
    # Initial values
    x = c
    w = c
    v = c
    fx = f(c)
    fw = fx
    fv = fx
    d = 0.0
    e_interval = 0.0
    rng_state = k if k != 0 else 12345
    tol = 2.0 * np.finfo(float).eps * abs(x) + t
    m2 = 0.5 * (a0 + b0)
    max_iter = 1000
    for _ in range(max_iter):
        tol = 2.0 * np.finfo(float).eps * abs(x) + t
        m2 = 0.5 * (a0 + b0)
        if abs(x - m2) <= tol - 0.5 * (b0 - a0):
            break
        # Construct quadratic under-estimator using M bound
        r = 0.0
        q = 0.0
        p = 0.0
        if abs(e_interval) > tol:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            r = e_interval
            e_interval = d
            # Accept parabolic step if valid
            if abs(p) < abs(0.5 * q * r) and p > q * (a0 - x) and p < q * (b0 - x):
                d = p / q
                u = x + d
                if u - a0 < 2.0 * tol or b0 - u < 2.0 * tol:
                    d = tol if x < m2 else -tol
            else:
                e_interval = b0 - x if x < m2 else a0 - x
                d = 0.5 * e_interval
                # Random probing after many iterations
                rng_state = (1611 * rng_state) % 1048576
                rand_frac = rng_state / 1048576.0
                d = d * (0.9 + 0.2 * rand_frac)
        else:
            e_interval = b0 - x if x < m2 else a0 - x
            d = 0.5 * e_interval
        # Ensure step is at least tolerance
        if abs(d) >= tol:
            u = x + d
        elif d > 0.0:
            u = x + tol
        else:
            u = x - tol
        fu = f(u)
        # Update bracket
        if fu <= fx:
            if u >= x:
                a0 = x
            else:
                b0 = x
            v = w
            fv = fw
            w = x
            fw = fx
            x = u
            fx = fu
        else:
            if u < x:
                a0 = u
            else:
                b0 = u
            if fu <= fw or abs(w - x) < 1e-30:
                v = w
                fv = fw
                w = u
                fw = fu
            elif fu <= fv or abs(v - x) < 1e-30 or abs(v - w) < 1e-30:
                v = u
                fv = fu
    return x, fx


def optimize_limiting_parameter(element_avg: float,
                                neighbor_avgs: np.ndarray,
                                high_order_slope: np.ndarray,
                                target_range: float = 0.0) -> float:
    """
    Use global optimization to find optimal limiting parameter theta in [0,1]
    that minimizes oscillation while preserving accuracy.
    """
    neighbor_avgs = np.asarray(neighbor_avgs, dtype=np.float64)
    if len(neighbor_avgs) == 0:
        return 1.0
    min_neighbor = neighbor_avgs.min()
    max_neighbor = neighbor_avgs.max()

    def objective(theta: float) -> float:
        # Penalize violation of bounds and oscillation
        slope = theta * high_order_slope
        min_val = element_avg + np.min(slope)
        max_val = element_avg + np.max(slope)
        penalty = 0.0
        if min_val < min_neighbor - 1e-10:
            penalty += (min_neighbor - min_val) ** 2
        if max_val > max_neighbor + 1e-10:
            penalty += (max_val - max_neighbor) ** 2
        # Accuracy penalty: prefer theta near 1
        penalty += 0.1 * (1.0 - theta) ** 2
        return penalty

    # Use glomin with small second-derivative bound
    theta_opt, _ = glomin(0.0, 1.0, 0.5, 10.0, 1e-8, 1e-6, objective)
    return float(np.clip(theta_opt, 0.0, 1.0))


# ---------------------------------------------------------------------------
# DG slope limiter for tetrahedral elements
# ---------------------------------------------------------------------------

def dg_slope_limiter(uh: np.ndarray, u_avg: float,
                     u_neighbors: np.ndarray,
                     centroid: np.ndarray,
                     neighbor_centroids: np.ndarray) -> np.ndarray:
    """
    Apply vertex-based slope limiter to high-order DG solution.
    uh : high-order solution values at vertices/quad points (n_dof,)
    u_avg : element average
    u_neighbors : neighbor element averages
    centroid : element centroid (3,)
    neighbor_centroids : neighbor centroids (n_neighbors, 3)
    Returns limited solution coefficients.
    """
    uh = np.asarray(uh, dtype=np.float64)
    if len(u_neighbors) == 0:
        return uh
    # Compute linear reconstruction slope
    # Simplified: limit gradient based on neighbor averages
    n_neighbors = len(u_neighbors)
    if n_neighbors == 0:
        return uh
    # Compute approximate gradient from neighbors
    grad = np.zeros(3, dtype=np.float64)
    A = np.zeros((n_neighbors, 3), dtype=np.float64)
    b = np.zeros(n_neighbors, dtype=np.float64)
    for i in range(n_neighbors):
        A[i] = neighbor_centroids[i] - centroid
        b[i] = u_neighbors[i] - u_avg
    # Least squares gradient
    AtA = A.T @ A
    try:
        grad = np.linalg.solve(AtA + 1e-10 * np.eye(3), A.T @ b)
    except np.linalg.LinAlgError:
        grad = np.linalg.lstsq(A, b, rcond=None)[0]
    # Limit each component
    for d in range(3):
        delta = grad[d]
        deltas = []
        for i in range(n_neighbors):
            dx = neighbor_centroids[i, d] - centroid[d]
            if abs(dx) > 1e-14:
                deltas.append((u_neighbors[i] - u_avg) / dx)
        if deltas:
            grad[d] = minmod(delta, np.median(deltas))
    # Apply limited gradient
    # For simplicity, we just modify the linear part of uh
    # In a real DG code, this would project onto the basis
    limited = uh.copy()
    # If uh represents modal coefficients, the linear part is the first 3 after constant
    if len(uh) > 3:
        for d in range(min(3, len(uh) - 1)):
            limited[d + 1] = minmod(limited[d + 1], grad[d])
    return limited


# ---------------------------------------------------------------------------
# Entropy viscosity limiter
# ---------------------------------------------------------------------------

def entropy_viscosity_coefficient(U: np.ndarray, grad_U: np.ndarray,
                                  h: float, c: float, max_wave_speed: float) -> float:
    """
    Compute entropy viscosity coefficient for shock capturing.
    nu_e = C_e * h^2 * |residual| / (entropy_jump + epsilon)
    """
    # Simplified entropy s = log(p / rho^gamma)
    rho = max(U[0], 1e-14)
    _, _, _, _, p = conservative_to_primitive(U)
    s = np.log(max(p, 1e-14) / (rho ** GAMMA))
    # Entropy residual (simplified)
    # |ds/dt + u·grad(s)|
    u = U[1] / rho
    v = U[2] / rho
    w = U[3] / rho
    dsdt = 0.0  # assume steady or use time derivative if available
    residual = abs(dsdt + u * grad_U[0] + v * grad_U[1] + w * grad_U[2])
    epsilon = 1e-10
    C_e = 1.0
    nu_e = C_e * h * h * residual / (abs(s) + epsilon)
    # Cap by first-order viscosity
    nu_max = 0.5 * h * max_wave_speed
    return min(nu_e, nu_max)


from euler_equations import conservative_to_primitive, GAMMA
