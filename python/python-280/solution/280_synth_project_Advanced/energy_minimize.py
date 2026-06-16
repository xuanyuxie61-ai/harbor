"""
energy_minimize.py — Energy minimization for crack propagation direction.

Seeds: 694_local_min (Brent's method for 1-D minimization),
       940_quad_gauss (Gauss-Legendre quadrature for integration).

Core idea
=========
The crack propagation direction is determined by minimizing the total
potential energy of the system, which is equivalent to maximizing the
energy release rate G(θ) as a function of propagation angle θ.

The total potential energy is:
    Π(u, D) = ∫_Ω ½ (1-D) ε(u) : C : ε(u) dA
              - ∫_{Γ_t} t̄ · u ds
              - ∫_Ω b · u dA

The energy release rate for a virtual crack extension da in direction θ:
    G(θ) = -dΠ/da|_θ = ∫_Ω ½ Y(θ) dA

where Y is the energy release rate density.

We find θ* = argmax_θ G(θ) using:
  1. Brent's method for 1-D minimization of -G(θ) over θ ∈ [-π, π]
  2. Gauss-Legendre quadrature for evaluating the J-integral along
     a contour around the crack tip

The J-integral (path-independent for elastic materials):
    J = ∫_Γ (W dy - t · ∂u/∂x ds)

where W = ½ σ:ε is the strain energy density, Γ is a contour around
the crack tip, and t = σ·n is the traction vector.

For the numerical evaluation, we use N-point Gauss-Legendre quadrature
on each segment of the contour:
    J ≈ Σ_segments Σ_quadrature_points (W dy - t_x ∂u_x/∂x - t_y ∂u_y/∂x) w_i
"""

import math
import numpy as np
from typing import Dict, Tuple, Callable, Optional
from config import SimulationConfig, MaterialParams, CrackConfig


# ===================================================================
# Gauss-Legendre quadrature  (seed 940_quad_gauss)
# ===================================================================

def gauss_legendre_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute n-point Gauss-Legendre quadrature nodes and weights on [-1,1].

    Uses the Golub-Welsch algorithm: construct the symmetric tridiagonal
    Jacobi matrix for Legendre polynomials and find its eigenvalues (= nodes)
    and eigenvectors (→ weights).

    For Legendre polynomials on [-1,1]:
        Jacobi matrix J has:
            diagonal:  a_k = 0  for all k
            sub/super-diagonal:  b_k = k / sqrt(4k² - 1)

    The eigenvalues of J are the quadrature nodes x_i.
    The weights are:  w_i = 2 * (v_i[0])²
    where v_i is the i-th normalized eigenvector.
    """
    if n <= 0:
        return np.array([]), np.array([])

    if n == 1:
        return np.array([0.0]), np.array([2.0])

    # Construct Jacobi matrix
    k = np.arange(1, n, dtype=float)
    b = k / np.sqrt(4.0 * k ** 2 - 1.0)

    J = np.zeros((n, n))
    for i in range(n - 1):
        J[i, i + 1] = b[i]
        J[i + 1, i] = b[i]

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(J)

    # Nodes = eigenvalues
    nodes = eigenvalues

    # Weights = 2 * (first component of eigenvector)²
    weights = 2.0 * eigenvectors[0, :] ** 2

    # Sort by node position
    sort_idx = np.argsort(nodes)
    nodes = nodes[sort_idx]
    weights = weights[sort_idx]

    return nodes, weights


def gauss_legendre_1d(f_values: np.ndarray,
                      a: float, b: float,
                      n_points: int = 5) -> float:
    """Integrate a function on [a,b] using n-point Gauss-Legendre quadrature.

    Mapping from [-1,1] to [a,b]:
        x = (b-a)/2 * ξ + (a+b)/2
        dx = (b-a)/2 * dξ

    ∫_a^b f(x) dx = (b-a)/2 * Σ_i w_i f((b-a)/2 * x_i + (a+b)/2)

    For tabulated function values, we interpolate at the quadrature nodes.
    """
    nodes, weights = gauss_legendre_nodes_weights(n_points)

    # Map nodes to [a, b]
    x_mapped = 0.5 * (b - a) * nodes + 0.5 * (a + b)

    # Interpolate function values at quadrature nodes
    # Assume f_values are on a uniform grid over [a, b]
    n_grid = len(f_values)
    x_grid = np.linspace(a, b, n_grid)
    f_interp = np.interp(x_mapped, x_grid, f_values)

    # Quadrature sum
    integral = 0.5 * (b - a) * np.sum(weights * f_interp)

    return float(integral)


def gauss_legendre_2d(integrand: Callable[[np.ndarray, np.ndarray], np.ndarray],
                      x_range: Tuple[float, float],
                      y_range: Tuple[float, float],
                      n_x: int = 5, n_y: int = 5) -> float:
    """2D Gauss-Legendre quadrature over a rectangle.

    ∫∫ f(x,y) dx dy ≈ Σ_i Σ_j w_i w_j f(x_i, y_j) * (Δx/2)(Δy/2)
    """
    nx_nodes, nx_weights = gauss_legendre_nodes_weights(n_x)
    ny_nodes, ny_weights = gauss_legendre_nodes_weights(n_y)

    x0, x1 = x_range
    y0, y1 = y_range

    # Map to [x0, x1] × [y0, y1]
    xi = 0.5 * (x1 - x0) * nx_nodes + 0.5 * (x0 + x1)
    eta = 0.5 * (y1 - y0) * ny_nodes + 0.5 * (y0 + y1)

    XI, ETA = np.meshgrid(xi, eta)
    f_vals = integrand(XI, ETA)

    # Tensor product quadrature
    integral = 0.0
    for i in range(n_y):
        for j in range(n_x):
            integral += nx_weights[j] * ny_weights[i] * f_vals[i, j]

    integral *= 0.25 * (x1 - x0) * (y1 - y0)

    return float(integral)


# ===================================================================
# J-integral evaluation
# ===================================================================

def evaluate_j_integral(u_field: np.ndarray,
                        v_field: np.ndarray,
                        x: np.ndarray, y: np.ndarray,
                        damage: np.ndarray,
                        crack: CrackConfig,
                        material: MaterialParams,
                        contour_radius: float = 0.02,
                        n_quadrature: int = 8,
                        n_contour_points: int = 32) -> float:
    """Evaluate the J-integral along a circular contour around the crack tip.

    J = ∫_Γ (W dy - t · ∂u/∂x ds)

    For a circular contour of radius R centered at the crack tip:
        x(θ) = x_tip + R cos(θ)
        y(θ) = y_tip + R sin(θ)
        dx = -R sin(θ) dθ
        dy = R cos(θ) dθ
        ds = R dθ

    So:
        J = ∫_0^{2π} [W(θ) R cos(θ) - t_x(θ) ∂u_x/∂x R
                       - t_y(θ) ∂u_y/∂x R] dθ

    We evaluate this using n_contour_points Gauss-Legendre quadrature
    on each segment of the contour.

    For a damaged material, the J-integral is modified:
        J_damaged = (1 - D_avg) * J_undamaged

    where D_avg is the average damage along the contour.
    """
    R = contour_radius
    tip_x, tip_y = crack.tip_x, crack.tip_y
    ny_grid, nx_grid = x.shape
    dx_grid = x[0, 1] - x[0, 0] if nx_grid > 1 else 1.0
    dy_grid = y[1, 0] - y[0, 0] if ny_grid > 1 else 1.0

    lam = material.lame_lambda
    mu = material.lame_mu

    # Gauss-Legendre nodes on [0, 2π]
    nodes_ref, weights_ref = gauss_legendre_nodes_weights(n_quadrature)

    # Divide contour into segments
    n_segments = max(n_contour_points // n_quadrature, 1)
    segment_size = 2.0 * math.pi / n_segments

    J_total = 0.0
    D_total = 0.0
    n_points_counted = 0

    for seg in range(n_segments):
        theta_start = seg * segment_size
        theta_end = (seg + 1) * segment_size

        # Map GL nodes to this segment
        theta_nodes = 0.5 * (theta_end - theta_start) * nodes_ref \
                      + 0.5 * (theta_start + theta_end)
        theta_weights = 0.5 * (theta_end - theta_start) * weights_ref

        for k in range(len(theta_nodes)):
            theta = theta_nodes[k]
            w = theta_weights[k]

            # Point on contour
            px = tip_x + R * math.cos(theta)
            py = tip_y + R * math.sin(theta)

            # Find nearest grid point
            i_grid = int(round((py - y[0, 0]) / dy_grid))
            j_grid = int(round((px - x[0, 0]) / dx_grid))
            i_grid = max(0, min(i_grid, ny_grid - 1))
            j_grid = max(0, min(j_grid, nx_grid - 1))

            # Interpolate displacement gradients at this point
            # (using nearest grid point for simplicity)
            jm = max(j_grid - 1, 0)
            jp = min(j_grid + 1, nx_grid - 1)
            im = max(i_grid - 1, 0)
            ip = min(i_grid + 1, ny_grid - 1)

            ux = (u_field[i_grid, jp] - u_field[i_grid, jm]) / max(x[0, jp] - x[0, jm], 1.0e-15)
            uy = (u_field[ip, j_grid] - u_field[im, j_grid]) / max(y[ip, 0] - y[im, 0], 1.0e-15)
            vx = (v_field[i_grid, jp] - v_field[i_grid, jm]) / max(x[0, jp] - x[0, jm], 1.0e-15)
            vy = (v_field[ip, j_grid] - v_field[im, j_grid]) / max(y[ip, 0] - y[im, 0], 1.0e-15)

            # Stress
            sxx = (lam + 2.0 * mu) * ux + lam * vy
            syy = lam * ux + (lam + 2.0 * mu) * vy
            sxy = mu * (uy + vx)

            # Strain energy density
            W = 0.5 * (sxx * ux + syy * vy + sxy * (uy + vx))

            # Traction on contour (outward normal = (cos θ, sin θ))
            n_x = math.cos(theta)
            n_y = math.sin(theta)
            t_x = sxx * n_x + sxy * n_y
            t_y = sxy * n_x + syy * n_y

            # J integrand: W dy/dθ - t · ∂u/∂x * ds/dθ
            # dy/dθ = R cos(θ),  ∂u_x/∂x needs special treatment
            # Simplified: J = W * R * cos(θ) - (t_x * ux + t_y * vx) * R
            # Actually the correct form:
            # J = ∫ (W n_1 - t_i ∂u_i/∂x_1) ds
            # For circular contour: n_1 = cos(θ), ds = R dθ
            dudx1 = ux * n_x + uy * n_y  # ∂u/∂n approximation
            dvdx1 = vx * n_x + vy * n_y
            integrand_val = W * n_x - (t_x * dudx1 + t_y * dvdx1)

            J_total += w * integrand_val * R

            # Damage along contour
            D_total += damage[i_grid, j_grid]
            n_points_counted += 1

    # Average damage correction
    D_avg = D_total / max(n_points_counted, 1)
    damage_factor = max(1.0 - D_avg, 1.0e-8)

    J_total *= damage_factor

    return float(J_total)


# ===================================================================
# Brent's method for crack direction optimization  (seed 694_local_min)
# ===================================================================

def brent_minimize(f: Callable[[float], float],
                   a: float, b: float,
                   tol: float = 1.0e-6,
                   max_iter: int = 100) -> Tuple[float, float, int]:
    """Find the minimum of f(x) on [a,b] using Brent's method.

    Combines golden-section search with parabolic interpolation.
    The golden ratio constant: c = 0.5*(3 - √5) ≈ 0.38197

    Returns (x_min, f_min, n_iterations).

    This is directly adapted from seed 694_local_min.
    """
    golden_c = 0.5 * (3.0 - math.sqrt(5.0))

    x = a + golden_c * (b - a)
    w = x
    v = x
    fx = f(x)
    fw = fx
    fv = fx

    e = 0.0
    d_step = 0.0

    for iteration in range(max_iter):
        xm = 0.5 * (a + b)
        tol1 = tol * abs(x) + 1.0e-12
        tol2 = 2.0 * tol1

        # Convergence check
        if abs(x - xm) <= tol2 - 0.5 * (b - a):
            break

        # Try parabolic interpolation
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0:
                p = -p
            else:
                q = abs(q)
            e_old = e
            e = d_step

            if (abs(p) < abs(0.5 * q * e_old)
                    and p > q * (a - x)
                    and p < q * (b - x)):
                # Parabolic interpolation step
                d_step = p / q
                u = x + d_step
                if (u - a) < tol2 or (b - u) < tol2:
                    d_step = tol1 if x < xm else -tol1
            else:
                # Golden section step
                e = (a - x) if x < xm else (b - x)
                d_step = golden_c * e
        else:
            # Golden section step
            e = (a - x) if x < xm else (b - x)
            d_step = golden_c * e

        # Compute f at new point
        u = x + (d_step if abs(d_step) >= tol1 else
                 (tol1 if d_step > 0 else -tol1))
        fu = f(u)

        # Update bracketing
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu

    return x, fx, iteration


def find_optimal_crack_direction(u_field: np.ndarray,
                                 v_field: np.ndarray,
                                 x: np.ndarray, y: np.ndarray,
                                 damage: np.ndarray,
                                 crack: CrackConfig,
                                 material: MaterialParams,
                                 contour_radius: float = 0.02
                                 ) -> Dict:
    """Find the optimal crack propagation direction via energy minimization.

    Minimizes -G(θ) over θ ∈ [-π/2, π/2] using Brent's method.
    G(θ) is evaluated via the J-integral with the crack virtually
    extended in direction θ.

    The maximum energy release rate criterion predicts:
        θ* = 0 for pure mode-I
        θ* ≠ 0 for mixed-mode loading
    """
    ny, nx = x.shape
    dx = x[0, 1] - x[0, 0] if nx > 1 else 1.0
    dy = y[1, 0] - y[0, 0] if ny > 1 else 1.0

    def neg_G(theta):
        """Negative energy release rate as function of propagation angle."""
        # Create virtual crack extension
        da = contour_radius * 0.1
        extended_crack = CrackConfig(
            tip_x=crack.tip_x + da * math.cos(crack.angle + theta),
            tip_y=crack.tip_y + da * math.sin(crack.angle + theta),
            length=crack.length + da,
            angle=crack.angle + theta,
            tip_radius=crack.tip_radius,
        )
        J_val = evaluate_j_integral(
            u_field, v_field, x, y, damage,
            extended_crack, material, contour_radius
        )
        E_prime = material.young_modulus / (1.0 - material.poisson_ratio ** 2)
        G_val = J_val / max(E_prime, 1.0)
        return -G_val  # minimize negative G = maximize G

    # Brent's method on [-π/2, π/2]
    theta_opt, neg_G_opt, n_iter = brent_minimize(
        neg_G, -math.pi / 2.0, math.pi / 2.0, tol=1.0e-4
    )

    return {
        "optimal_angle_rad": float(theta_opt),
        "optimal_angle_deg": float(math.degrees(theta_opt)),
        "max_energy_release_rate": float(-neg_G_opt),
        "n_brent_iterations": int(n_iter),
    }
