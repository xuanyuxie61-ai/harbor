"""
fd_operators.py — High-order finite-difference operators for the
damage-modified elastodynamic equations.

Implements:
  - 4th-order central differences for interior points
  - 2nd-order one-sided differences at boundaries
  - Variable-coefficient operators (damage-degraded stiffness)
  - Compact (Padé) finite-difference stencils
  - Spectral radius estimation for CFL analysis

Governing equation (2-D plane strain, damage-coupled):

    ρ ∂²u_i/∂t² = ∂/∂x_j [ (1-D) C_ijkl ε_kl ] + f_i

where D(x,t) ∈ [0,1] is the scalar damage field and C_ijkl is the
undamaged isotropic stiffness tensor.  After expansion:

    ρ ü = (1-D)[(λ+2μ) u_{xx} + μ u_{yy} + (λ+μ) v_{xy}]
          - D_x [(λ+2μ) u_x + λ v_y]
          - D_y [μ u_y + μ v_x]
          + f_x

and similarly for the v-equation.  The spatial discretisation uses
4th-order central differences for all second and first derivatives.
"""

import math
import numpy as np
from typing import Dict, Tuple, Optional
from config import SimulationConfig, MaterialParams


# ===================================================================
# 1-D finite-difference coefficient tables
# ===================================================================

def fd_coefficients_1d(order: int = 4) -> Dict[str, np.ndarray]:
    """Compute FD coefficients for 1st and 2nd derivatives.

    For interior points (central differences):
      order 2:  f' ≈ (-f_{i-1} + f_{i+1}) / (2h)
      order 4:  f' ≈ (f_{i-2} - 8f_{i-1} + 8f_{i+1} - f_{i+2}) / (12h)
      order 2:  f'' ≈ (f_{i-1} - 2f_i + f_{i+1}) / h²
      order 4:  f'' ≈ (-f_{i-2}+16f_{i-1}-30f_i+16f_{i+1}-f_{i+2}) / (12h²)
    """
    coeffs = {}
    if order == 2:
        coeffs["first_deriv_center"] = np.array([-1.0, 0.0, 1.0]) / 2.0
        coeffs["second_deriv_center"] = np.array([1.0, -2.0, 1.0])
        coeffs["first_deriv_left"] = np.array([-3.0, 4.0, -1.0]) / 2.0
        coeffs["first_deriv_right"] = np.array([1.0, -4.0, 3.0]) / 2.0
    elif order == 4:
        coeffs["first_deriv_center"] = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0
        coeffs["second_deriv_center"] = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / 12.0
        # 4th-order one-sided for boundaries
        coeffs["first_deriv_left"] = np.array([-25.0, 48.0, -36.0, 16.0, -3.0]) / 12.0
        coeffs["first_deriv_right"] = np.array([3.0, -16.0, 36.0, -48.0, 25.0]) / 12.0
        coeffs["second_deriv_left"] = np.array([35.0, -104.0, 114.0, -56.0, 11.0]) / 12.0
        coeffs["second_deriv_right"] = np.array([11.0, -56.0, 114.0, -104.0, 35.0]) / 12.0
    elif order == 6:
        # 6th-order central differences
        coeffs["first_deriv_center"] = np.array(
            [-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0]) / 60.0
        coeffs["second_deriv_center"] = np.array(
            [2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0]) / 180.0
        coeffs["first_deriv_left"] = np.array(
            [-137.0, 300.0, -300.0, 200.0, -75.0, 12.0]) / 60.0
        coeffs["first_deriv_right"] = np.array(
            [-12.0, 75.0, -200.0, 300.0, -300.0, 137.0]) / 60.0
    else:
        raise ValueError(f"Unsupported FD order: {order}")

    return coeffs


# ===================================================================
# 2-D derivative operators on uniform grid
# ===================================================================

def apply_d_dx(f: np.ndarray, dx: float, order: int = 4) -> np.ndarray:
    """Compute ∂f/∂x on a 2D array using finite differences.

    Uses central differences in the interior and one-sided at boundaries.
    f has shape (ny, nx).
    """
    ny, nx = f.shape
    result = np.zeros_like(f)
    half_w = order // 2  # half-stencil width

    # Interior: central differences
    coeffs = fd_coefficients_1d(order)
    c1 = coeffs["first_deriv_center"]
    for k, coeff in enumerate(c1):
        offset = k - half_w
        if offset == 0:
            continue
        # Shifted array (interior columns only)
        src_slice = slice(max(0, -offset), min(nx, nx - offset))
        dst_slice = slice(max(0, offset), min(nx, nx + offset))
        # Only apply to interior columns
        if half_w < nx:
            result[:, half_w:nx - half_w] += coeff * f[:, half_w + offset:nx - half_w + offset] / dx

    # Left boundary (one-sided)
    c_left = coeffs.get("first_deriv_left", coeffs["first_deriv_center"])
    n_bc = min(len(c_left), nx)
    for i in range(min(half_w, nx)):
        for k in range(n_bc):
            j = min(i + k, nx - 1)
            result[:, i] += c_left[k] * f[:, j] / dx

    # Right boundary
    c_right = coeffs.get("first_deriv_right", coeffs["first_deriv_center"])
    for i in range(max(0, nx - half_w), nx):
        for k in range(n_bc):
            j = max(i - (n_bc - 1 - k), 0)
            result[:, i] += c_right[k] * f[:, j] / dx

    return result


def apply_d_dy(f: np.ndarray, dy: float, order: int = 4) -> np.ndarray:
    """Compute ∂f/∂y on a 2D array.  Analogous to apply_d_dx."""
    ny, nx = f.shape
    result = np.zeros_like(f)
    half_w = order // 2

    coeffs = fd_coefficients_1d(order)
    c1 = coeffs["first_deriv_center"]
    for k, coeff in enumerate(c1):
        offset = k - half_w
        if offset == 0:
            continue
        result[half_w:ny - half_w, :] += coeff * f[half_w + offset:ny - half_w + offset, :] / dy

    c_left = coeffs.get("first_deriv_left", coeffs["first_deriv_center"])
    n_bc = min(len(c_left), ny)
    for i in range(min(half_w, ny)):
        for k in range(n_bc):
            j = min(i + k, ny - 1)
            result[i, :] += c_left[k] * f[j, :] / dy

    c_right = coeffs.get("first_deriv_right", coeffs["first_deriv_center"])
    for i in range(max(0, ny - half_w), ny):
        for k in range(n_bc):
            j = max(i - (n_bc - 1 - k), 0)
            result[i, :] += c_right[k] * f[j, :] / dy

    return result


def apply_d2_dx2(f: np.ndarray, dx: float, order: int = 4) -> np.ndarray:
    """Compute ∂²f/∂x² using central differences."""
    ny, nx = f.shape
    result = np.zeros_like(f)
    half_w = order // 2

    coeffs = fd_coefficients_1d(order)
    c2 = coeffs["second_deriv_center"]
    for k, coeff in enumerate(c2):
        offset = k - half_w
        result[:, half_w:nx - half_w] += coeff * f[:, half_w + offset:nx - half_w + offset] / (dx ** 2)

    # Boundary: fall back to 2nd-order for second derivative
    for i in range(min(half_w, nx)):
        if i == 0 and nx > 2:
            result[:, 0] = (f[:, 0] - 2.0 * f[:, 0] + f[:, min(1, nx - 1)]) / (dx ** 2)
            result[:, 0] = (2.0 * f[:, 0] - 5.0 * f[:, 1] + 4.0 * f[:, min(2, nx - 1)]
                            - f[:, min(3, nx - 1)]) / (dx ** 2)
        elif i > 0 and i < nx:
            jm = max(i - 1, 0)
            jp = min(i + 1, nx - 1)
            result[:, i] = (f[:, jm] - 2.0 * f[:, i] + f[:, jp]) / (dx ** 2)

    for i in range(max(0, nx - half_w), nx):
        jm = max(i - 1, 0)
        jp = min(i + 1, nx - 1)
        result[:, i] = (f[:, jm] - 2.0 * f[:, i] + f[:, jp]) / (dx ** 2)

    return result


def apply_d2_dy2(f: np.ndarray, dy: float, order: int = 4) -> np.ndarray:
    """Compute ∂²f/∂y² using central differences."""
    ny, nx = f.shape
    result = np.zeros_like(f)
    half_w = order // 2

    coeffs = fd_coefficients_1d(order)
    c2 = coeffs["second_deriv_center"]
    for k, coeff in enumerate(c2):
        offset = k - half_w
        result[half_w:ny - half_w, :] += coeff * f[half_w + offset:ny - half_w + offset, :] / (dy ** 2)

    for i in range(min(half_w, ny)):
        jm = max(i - 1, 0)
        jp = min(i + 1, ny - 1)
        result[i, :] = (f[jm, :] - 2.0 * f[i, :] + f[jp, :]) / (dy ** 2)

    for i in range(max(0, ny - half_w), ny):
        jm = max(i - 1, 0)
        jp = min(i + 1, ny - 1)
        result[i, :] = (f[jm, :] - 2.0 * f[i, :] + f[jp, :]) / (dy ** 2)

    return result


def apply_d2_dxdy(f: np.ndarray, dx: float, dy: float, order: int = 4) -> np.ndarray:
    """Compute ∂²f/(∂x∂y) = ∂/∂x(∂f/∂y) via successive application."""
    df_dy = apply_d_dy(f, dy, order)
    return apply_d_dx(df_dy, dx, order)


# ===================================================================
# Damage-modified divergence of stress
# ===================================================================

def compute_damage_elastic_rhs(u: np.ndarray, v: np.ndarray,
                               damage: np.ndarray,
                               cfg: SimulationConfig,
                               body_force_x: Optional[np.ndarray] = None,
                               body_force_y: Optional[np.ndarray] = None,
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the RHS of the damage-modified momentum equation.

    RHS_x = (1-D)[(λ+2μ) u_{xx} + μ u_{yy} + (λ+μ) v_{xy}]
            - D_x [(λ+2μ) u_x + λ v_y]
            - D_y [μ u_y + μ v_x]
            + b_x

    Uses the product rule to handle the variable coefficient (1-D).
    All spatial derivatives use 4th-order FD by default.

    Returns (rhs_x, rhs_y) each of shape (ny, nx).
    """
    dx = cfg.dx()
    dy = cfg.dy()
    order = cfg.numerical.fd_order
    lam = cfg.material.lame_lambda
    mu = cfg.material.lame_mu

    # Displacement gradients
    ux = apply_d_dx(u, dx, order)
    uy = apply_d_dy(u, dy, order)
    vx = apply_d_dx(v, dx, order)
    vy = apply_d_dy(v, dy, order)

    # Second derivatives
    uxx = apply_d2_dx2(u, dx, order)
    uyy = apply_d2_dy2(u, dy, order)
    vxx = apply_d2_dx2(v, dx, order)
    vyy = apply_d2_dy2(v, dy, order)
    uxy = apply_d2_dxdy(u, dx, dy, order)
    vxy = apply_d2_dxdy(v, dx, dy, order)

    # Damage gradients
    Dx = apply_d_dx(damage, dx, order)
    Dy = apply_d_dy(damage, dy, order)

    # Undamaged stress divergence terms
    # σ_xx = (λ+2μ) ε_xx + λ ε_yy  →  ∂σ_xx/∂x = (λ+2μ) u_{xx} + λ v_{xy}
    # σ_xy = μ (u_y + v_x)          →  ∂σ_xy/∂y = μ (u_{yy} + v_{xy})
    # Total x: (λ+2μ) u_{xx} + μ u_{yy} + (λ+μ) v_{xy}

    undamaged_x = ((lam + 2.0 * mu) * uxx + mu * uyy
                   + (lam + mu) * vxy)
    undamaged_y = ((lam + 2.0 * mu) * vyy + mu * vxx
                   + (lam + mu) * uxy)

    # Damage gradient coupling
    damage_coupling_x = Dx * ((lam + 2.0 * mu) * ux + lam * vy)
    damage_coupling_y = Dy * ((lam + 2.0 * mu) * vy + lam * ux)

    # Shear coupling
    shear_coupling_x = Dy * mu * (uy + vx)
    shear_coupling_y = Dx * mu * (uy + vx)

    # Assemble
    stiffness_factor = np.clip(1.0 - damage, 1.0e-8, 1.0)

    rhs_x = (stiffness_factor * undamaged_x
             - damage_coupling_x - shear_coupling_x)
    rhs_y = (stiffness_factor * undamaged_y
             - damage_coupling_y - shear_coupling_y)

    # Add body forces
    if body_force_x is not None:
        rhs_x += body_force_x
    if body_force_y is not None:
        rhs_y += body_force_y

    # Clamp boundary artifacts
    rhs_x[0, :] = 0.0
    rhs_x[-1, :] = 0.0
    rhs_x[:, 0] = 0.0
    rhs_x[:, -1] = 0.0
    rhs_y[0, :] = 0.0
    rhs_y[-1, :] = 0.0
    rhs_y[:, 0] = 0.0
    rhs_y[:, -1] = 0.0

    return rhs_x, rhs_y


# ===================================================================
# Compact (Padé) finite differences
# ===================================================================

def compact_first_derivative_1d(f: np.ndarray, dx: float,
                                alpha: float = 1.0 / 3.0) -> np.ndarray:
    """4th-order compact (Padé) scheme for f'(x).

    Implicit relation:
        α f'_{i-1} + f'_i + α f'_{i+1}
            = a (f_{i+1} - f_{i-1}) / (2h)
              + b (f_{i+2} - f_{i-2}) / (4h)

    For α = 1/3, a = 4/3, b = -1/3  (4th-order accurate).

    Solved via Thomas algorithm (tridiagonal solver).
    """
    n = len(f)
    a_coeff = 4.0 / 3.0
    b_coeff = -1.0 / 3.0

    # RHS
    rhs = np.zeros(n)
    for i in range(2, n - 2):
        rhs[i] = (a_coeff * (f[i + 1] - f[i - 1]) / (2.0 * dx)
                  + b_coeff * (f[i + 2] - f[i - 2]) / (4.0 * dx))

    # Boundary: 2nd-order one-sided
    if n > 2:
        rhs[0] = (-3.0 * f[0] + 4.0 * f[1] - f[min(2, n - 1)]) / (2.0 * dx)
        rhs[-1] = (3.0 * f[-1] - 4.0 * f[-2] + f[max(0, n - 3)]) / (2.0 * dx)
    if n > 4:
        rhs[1] = (-3.0 * f[0] + 4.0 * f[1] - f[2]) / (2.0 * dx) * 0.5 \
                 + (f[2] - f[0]) / (2.0 * dx) * 0.5
        rhs[-2] = (3.0 * f[-1] - 4.0 * f[-2] + f[-3]) / (2.0 * dx) * 0.5 \
                  + (f[-1] - f[-3]) / (2.0 * dx) * 0.5

    # Thomas algorithm for tridiagonal system:
    #  α * x_{i-1} + x_i + α * x_{i+1} = rhs_i
    lower = np.full(n, alpha)
    diag = np.ones(n)
    upper = np.full(n, alpha)

    # Forward sweep
    for i in range(1, n):
        w = lower[i] / diag[i - 1]
        diag[i] -= w * upper[i - 1]
        rhs[i] -= w * rhs[i - 1]

    # Back substitution
    result = np.zeros(n)
    result[-1] = rhs[-1] / diag[-1]
    for i in range(n - 2, -1, -1):
        result[i] = (rhs[i] - upper[i] * result[i + 1]) / diag[i]

    return result


# ===================================================================
# Spectral radius estimation
# ===================================================================

def estimate_spectral_radius(dx: float, dy: float,
                             material: MaterialParams,
                             damage_max: float = 0.0) -> float:
    """Estimate the spectral radius of the FD spatial operator.

    For the 4th-order central scheme applied to the wave equation
    with wave speed c, the spectral radius is approximately:

        ρ(A) ≈ c² * (27/(4h²))  for 4th-order in 2D

    More precisely, for the 1-D Laplacian with 4th-order FD:
        eigenvalues ∈ [-c² * (1/h²) * (4/3 sin²(ξh/2) - 1/12 sin²(ξh)), 0]

    The most negative eigenvalue (stiffest mode) occurs at ξ = π/h:
        λ_max ≈ c² / h² * (4/3 + 1/12) * 4 = c² / h² * 57/12

    With damage, c → c * sqrt(1 - D), reducing the spectral radius.
    """
    c_p = material.p_wave_speed
    c_s = material.s_wave_speed
    c_max = max(c_p, c_s)

    # Damaged wave speed
    damage_factor = max(1.0 - damage_max, 1.0e-8)
    c_eff = c_max * math.sqrt(damage_factor)

    h_min = min(dx, dy)

    # Spectral radius of the 4th-order discrete Laplacian
    # For stencil [-1, 16, -30, 16, -1] / (12 h²), max eigenvalue magnitude:
    #   at frequency ξ = π/h:  |λ| = (1 + 16 + 30 + 16 + 1) / (12 h²) = 64/(12 h²) = 16/(3 h²)
    rho_spatial = c_eff ** 2 * 16.0 / (3.0 * h_min ** 2)

    # For 2D, both directions contribute
    rho_2d = rho_spatial * (1.0 / dx ** 2 + 1.0 / dy ** 2) / (2.0 / h_min ** 2)

    return float(rho_2d)
