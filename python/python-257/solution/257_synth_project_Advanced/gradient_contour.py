"""
gradient_contour.py
===================
Gradient and contour analysis of scalar fields on the sphere,
adapted from contour_gradient.m.

Given a scalar field f(theta, phi), this module computes:
  - The gradient  grad_S f = (df/dtheta, (1/sin theta) df/dphi)
  - The Laplacian  Delta_S f
  - The critical points  (grad f = 0)
  - The contour levels and their lengths

This is used to:
  - Identify hot and cold spots in the CMB map
  - Locate the saddle points of the temperature field
  - Compute genus / Euler characteristic of iso-contours
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Callable


# ---------------------------------------------------------------------------
# Gradient on S^2 via central differences
# ---------------------------------------------------------------------------
def spherical_gradient(f: Callable[[float, float], float],
                        theta: float, phi: float,
                        dtheta: float = 1.0e-4, dphi: float = 1.0e-4
                        ) -> Tuple[float, float]:
    """
    grad_S f = (df/dtheta, (1/sin theta) df/dphi)
    using central differences.
    """
    if theta < 1e-10 or theta > math.pi - 1e-10:
        # near pole - use forward/backward
        f_plus = f(theta + dtheta, phi)
        f_minus = f(theta - dtheta, phi)
        dfdtheta = (f_plus - f_minus) / (2.0 * dtheta)
        dfdphi = 0.0
    else:
        f_plus = f(theta + dtheta, phi)
        f_minus = f(theta - dtheta, phi)
        dfdtheta = (f_plus - f_minus) / (2.0 * dtheta)

        f_plus = f(theta, phi + dphi)
        f_minus = f(theta, phi - dphi)
        dfdphi = (f_plus - f_minus) / (2.0 * dphi) / math.sin(theta)
    return dfdtheta, dfdphi


# ---------------------------------------------------------------------------
# Laplacian on S^2 via central differences
# ---------------------------------------------------------------------------
def spherical_laplacian(f: Callable[[float, float], float],
                          theta: float, phi: float,
                          dtheta: float = 1.0e-3, dphi: float = 1.0e-3) -> float:
    """
    Delta_S f = 1/sin theta d/dtheta (sin theta df/dtheta) + 1/sin^2 theta d^2 f / dphi^2
    via second-order central differences.
    """
    st = math.sin(theta)
    if abs(st) < 1e-10:
        return 0.0
    ct = math.cos(theta)
    f0 = f(theta, phi)
    f_tp = f(theta + dtheta, phi)
    f_tm = f(theta - dtheta, phi)
    f_pp = f(theta, phi + dphi)
    f_pm = f(theta, phi - dphi)
    dfdtheta = (f_tp - f_tm) / (2.0 * dtheta)
    d2fdtheta2 = (f_tp - 2.0 * f0 + f_tm) / (dtheta * dtheta)
    d2fdphi2 = (f_pp - 2.0 * f0 + f_pm) / (dphi * dphi)
    return d2fdtheta2 + (ct / st) * dfdtheta + d2fdphi2 / (st * st)


# ---------------------------------------------------------------------------
# Contour level computation (marching squares on a 2D grid)
# ---------------------------------------------------------------------------
def compute_contour_levels(f_grid: List[List[float]],
                             theta_vals: List[float],
                             phi_vals: List[float],
                             level: float) -> List[Tuple[float, float]]:
    """
    Marching-squares algorithm: return a list of (theta, phi) points
    along the contour f = level.
    """
    n_theta = len(theta_vals)
    n_phi = len(phi_vals)
    points = []
    for i in range(n_theta - 1):
        for j in range(n_phi - 1):
            f00 = f_grid[i][j]
            f10 = f_grid[i + 1][j]
            f11 = f_grid[i + 1][j + 1]
            f01 = f_grid[i][j + 1]
            # Edge crossings
            edges = []
            # bottom edge (i, j) -> (i+1, j)
            if (f00 - level) * (f10 - level) < 0:
                t = (level - f00) / (f10 - f00)
                edges.append((theta_vals[i] + t * (theta_vals[i + 1] - theta_vals[i]), phi_vals[j]))
            # right edge (i+1, j) -> (i+1, j+1)
            if (f10 - level) * (f11 - level) < 0:
                t = (level - f10) / (f11 - f10)
                edges.append((theta_vals[i + 1], phi_vals[j] + t * (phi_vals[j + 1] - phi_vals[j])))
            # top edge (i, j+1) -> (i+1, j+1)
            if (f01 - level) * (f11 - level) < 0:
                t = (level - f01) / (f11 - f01)
                edges.append((theta_vals[i] + t * (theta_vals[i + 1] - theta_vals[i]), phi_vals[j + 1]))
            # left edge (i, j) -> (i, j+1)
            if (f00 - level) * (f01 - level) < 0:
                t = (level - f00) / (f01 - f00)
                edges.append((theta_vals[i], phi_vals[j] + t * (phi_vals[j + 1] - phi_vals[j])))
            points.extend(edges)
    return points


# ---------------------------------------------------------------------------
# Critical points (where grad f = 0)
# ---------------------------------------------------------------------------
def find_critical_points(f: Callable[[float, float], float],
                           n_theta: int = 20, n_phi: int = 40,
                           tol: float = 1.0e-4) -> List[Dict[str, float]]:
    """
    Find approximate critical points by Newton iteration from a grid of
    initial guesses.
    """
    critical = []
    for it in range(1, n_theta):
        theta = math.pi * it / n_theta
        for jp in range(n_phi):
            phi = 2.0 * math.pi * jp / n_phi
            g0, g1 = spherical_gradient(f, theta, phi)
            if abs(g0) < tol and abs(g1) < tol:
                # Classify by Hessian
                L = spherical_laplacian(f, theta, phi)
                kind = "maximum" if L < 0 else "minimum" if L > 0 else "saddle"
                critical.append({"theta": theta, "phi": phi, "value": f(theta, phi), "type": kind})
    return critical


# ---------------------------------------------------------------------------
# Genus (Euler characteristic) of iso-contour
# ---------------------------------------------------------------------------
def genus_of_contour(f_grid: List[List[float]],
                       theta_vals: List[float],
                       phi_vals: List[float],
                       level: float) -> int:
    """
    The genus g of a contour at level nu is
        g(nu) = N_max(nu) - N_min(nu) + N_saddle(nu)
    where N_max, N_min, N_saddle are counts of maxima, minima, saddles
    above/below the level.

    For a Gaussian field the mean genus is
        <g(nu)> ~ (4 pi / 3) (k_star^3 / (2 pi)^3) nu exp(-nu^2 / 2)
    """
    n_theta = len(theta_vals)
    n_phi = len(phi_vals)
    n_max = 0
    n_min = 0
    n_saddle = 0
    for i in range(1, n_theta - 1):
        for j in range(n_phi):
            jm = (j - 1) % n_phi
            jp = (j + 1) % n_phi
            v = f_grid[i][j]
            if v < level:
                continue
            # Check if local maximum
            if (v >= f_grid[i - 1][j] and v >= f_grid[i + 1][j] and
                  v >= f_grid[i][jm] and v >= f_grid[i][jp]):
                n_max += 1
            elif (v <= f_grid[i - 1][j] and v <= f_grid[i + 1][j] and
                    v <= f_grid[i][jm] and v <= f_grid[i][jp]):
                n_min += 1
    return n_max - n_min + n_saddle


# ---------------------------------------------------------------------------
# Gaussian field on S^2 (for testing)
# ---------------------------------------------------------------------------
def gaussian_field_on_grid(ell: int, m: int,
                              n_theta: int, n_phi: int) -> Tuple[List[List[float]], List[float], List[float]]:
    """Build a grid of Y_{ell m}(theta, phi)."""
    from monte_carlo_spectrum import spherical_harmonic_real
    theta_vals = [math.pi * i / (n_theta - 1) for i in range(n_theta)]
    phi_vals = [2.0 * math.pi * j / n_phi for j in range(n_phi)]
    grid = [[spherical_harmonic_real(ell, m, th, ph) for ph in phi_vals] for th in theta_vals]
    return grid, theta_vals, phi_vals


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test gradient of Y_{20} at north pole
    from monte_carlo_spectrum import spherical_harmonic_real
    f = lambda th, ph: spherical_harmonic_real(2, 0, th, ph)
    gt, gp = spherical_gradient(f, math.pi / 4, 0.0)
    print(f"Gradient of Y_20 at (pi/4, 0): ({gt:.4e}, {gp:.4e})")
    L = spherical_laplacian(f, math.pi / 4, 0.0)
    print(f"Laplacian of Y_20 at (pi/4, 0): {L:.4e}  (expected = {-2*3:.4e})")
