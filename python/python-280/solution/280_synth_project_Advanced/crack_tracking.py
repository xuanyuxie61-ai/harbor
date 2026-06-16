"""
crack_tracking.py — Level-set crack tracking via signed distance functions.

Seed reference: 305_dist_plot (signed distance functions for geometry).

Implements:
  - Crack representation as the zero level set of φ(x,y)
  - Reinitialization of φ to maintain signed-distance property
  - Crack propagation direction via the maximum hoop stress criterion
  - Stress intensity factor (SIF) extraction via interaction integral
  - Crack-tip velocity from the dynamic energy release rate

Level-set evolution equation (Hamilton-Jacobi type):
    ∂φ/∂t + F |∇φ| = 0

where F is the normal velocity of the crack surface.
For a stationary crack (only geometry tracking), F = 0 and φ remains
a signed distance function via reinitialization:
    ∂φ/∂τ + sign(φ_0)(|∇φ| - 1) = 0

Stress intensity factors are computed via the interaction integral (M-integral):
    M = ∫_A [ σ_{ij}^{(1)} ε_{ij}^{(2)} + σ_{ij}^{(2)} ε_{ij}^{(1)}
              - W^{(1,2)} δ_{1j} ] q_{,j} dA

where superscripts (1) and (2) denote actual and auxiliary fields,
and q is a smooth weight function vanishing on the domain boundary.
"""

import math
import numpy as np
from typing import Dict, Tuple, Optional
from config import SimulationConfig, MaterialParams, CrackConfig


# ===================================================================
# Reinitialization of signed distance function
# ===================================================================

def reinitialize_signed_distance(phi: np.ndarray,
                                 dx: float, dy: float,
                                 n_iterations: int = 50,
                                 dtau: Optional[float] = None
                                 ) -> np.ndarray:
    """Reinitialize φ to maintain the signed-distance property |∇φ| = 1.

    Solves the Eikonal-type PDE:
        ∂φ/∂τ + S(φ_0) (|∇φ| - 1) = 0

    where S(φ_0) = φ_0 / sqrt(φ_0² + h²) is a smoothed sign function.

    Uses a 2nd-order Godunov scheme for the gradient magnitude:
        |∇φ|_G = sqrt( max(D_x^-φ, 0)² + min(D_x^+φ, 0)²
                       + max(D_y^-φ, 0)² + min(D_y^+φ, 0)² )

    where D^- and D^+ are backward and forward differences chosen
    based on the sign of S(φ_0) (upwind scheme).

    For simplicity, we use a fixed-point iteration with damped updates.
    """
    ny, nx = phi.shape
    h = min(dx, dy)
    if dtau is None:
        dtau = 0.5 * h  # CFL for reinitialization

    phi_0 = phi.copy()
    h_smooth = np.sqrt(phi_0 ** 2 + h ** 2)
    S = phi_0 / np.maximum(h_smooth, 1.0e-15)

    phi_new = phi.copy()

    for iteration in range(n_iterations):
        # Gradient components (2nd-order central differences)
        phi_x = np.zeros_like(phi_new)
        phi_y = np.zeros_like(phi_new)

        # Interior
        phi_x[:, 1:-1] = (phi_new[:, 2:] - phi_new[:, :-2]) / (2.0 * dx)
        phi_y[1:-1, :] = (phi_new[2:, :] - phi_new[:-2, :]) / (2.0 * dy)

        # Boundaries: one-sided
        phi_x[:, 0] = (phi_new[:, 1] - phi_new[:, 0]) / dx
        phi_x[:, -1] = (phi_new[:, -1] - phi_new[:, -2]) / dx
        phi_y[0, :] = (phi_new[1, :] - phi_new[0, :]) / dy
        phi_y[-1, :] = (phi_new[-1, :] - phi_new[-2, :]) / dy

        # Gradient magnitude
        grad_mag = np.sqrt(phi_x ** 2 + phi_y ** 2)
        grad_mag = np.maximum(grad_mag, 1.0e-15)

        # Update: φ^{n+1} = φ^n - dτ * S(φ_0) * (|∇φ| - 1)
        phi_new = phi_new - dtau * S * (grad_mag - 1.0)

        # Boundary conditions: Neumann (zero normal derivative)
        phi_new[0, :] = phi_new[1, :]
        phi_new[-1, :] = phi_new[-2, :]
        phi_new[:, 0] = phi_new[:, 1]
        phi_new[:, -1] = phi_new[:, -2]

    return phi_new


# ===================================================================
# Crack propagation direction (maximum hoop stress criterion)
# ===================================================================

def compute_hoop_stress_direction(K_I: float, K_II: float) -> float:
    """Compute the crack propagation angle using the MTS criterion.

    The maximum hoop stress occurs at angle θ_c satisfying:
        K_I sin(θ_c) + K_II (3 cos(θ_c) - 1) = 0

    Solving:
        θ_c = 2 arctan( (K_I ± sqrt(K_I² + 8 K_II²)) / (4 K_II) )

    For pure mode-I (K_II = 0): θ_c = 0 (straight propagation).
    For mixed mode: the crack kinks to maximize the hoop stress.

    Returns θ_c in radians.
    """
    if abs(K_II) < 1.0e-15:
        return 0.0 if K_I >= 0.0 else math.pi

    discriminant = K_I ** 2 + 8.0 * K_II ** 2
    if discriminant < 0:
        return 0.0

    sqrt_disc = math.sqrt(discriminant)
    # Choose the root that gives |θ_c| < π
    arg1 = (K_I - sqrt_disc) / (4.0 * K_II)
    arg2 = (K_I + sqrt_disc) / (4.0 * K_II)

    theta1 = 2.0 * math.atan(arg1)
    theta2 = 2.0 * math.atan(arg2)

    # Check hoop stress criterion: σ_θθ must be maximized
    # σ_θθ ∝ K_I cos³(θ/2) - 3 K_II cos²(θ/2) sin(θ/2)
    def hoop_factor(theta):
        c = math.cos(theta / 2.0)
        s = math.sin(theta / 2.0)
        return K_I * c ** 3 - 3.0 * K_II * c ** 2 * s

    if hoop_factor(theta1) > hoop_factor(theta2):
        return theta1
    else:
        return theta2


# ===================================================================
# Dynamic stress intensity factors
# ===================================================================

def compute_dynamic_sif(u_field: np.ndarray,
                        v_field: np.ndarray,
                        x: np.ndarray, y: np.ndarray,
                        crack: CrackConfig,
                        material: MaterialParams,
                        r_domain_inner: float = 0.005,
                        r_domain_outer: float = 0.03,
                        ) -> Dict[str, float]:
    """Extract K_I and K_II using the interaction (M) integral.

    The M-integral for mode-I extraction uses auxiliary fields
    corresponding to a pure mode-I asymptotic field:

    M^(1,2) = 2/E' * (K_I^(1) K_I^(2) + K_II^(1) K_II^(2))

    For auxiliary K_I^(2) = 1, K_II^(2) = 0:
        K_I^(1) = E' * M^(1,2) / 2

    The interaction integral is:
        M = ∫_A [σ_{ij} u_{i,1}^{aux} + σ_{ij}^{aux} u_{i,1}
                 - W^{int} δ_{1j}] q_{,j} dA

    where q is a weight function:
        q(r) = 1  for r < r_inner
        q(r) = (r_outer - r)/(r_outer - r_inner)  for r_inner ≤ r ≤ r_outer
        q(r) = 0  for r > r_outer
    """
    ny, nx = x.shape
    dx = x[0, 1] - x[0, 0] if nx > 1 else 1.0
    dy = y[1, 0] - y[0, 0] if ny > 1 else 1.0

    E = material.young_modulus
    nu = material.poisson_ratio
    E_prime = E / (1.0 - nu ** 2)  # plane strain
    mu = material.lame_mu

    # Distance from crack tip
    r_tip = np.sqrt((x - crack.tip_x) ** 2 + (y - crack.tip_y) ** 2)
    theta_tip = np.arctan2(y - crack.tip_y, x - crack.tip_x)

    # Weight function q and its gradient
    q = np.zeros_like(x)
    q_x = np.zeros_like(x)
    q_y = np.zeros_like(x)

    dr = r_domain_outer - r_domain_inner
    if dr < 1.0e-15:
        dr = 1.0e-15

    in_ring = (r_tip >= r_domain_inner) & (r_tip <= r_domain_outer)
    q[r_tip < r_domain_inner] = 1.0
    q[in_ring] = (r_domain_outer - r_tip[in_ring]) / dr

    # Gradient of q: ∇q = dq/dr * ∇r = dq/dr * (x-x_tip, y-y_tip) / r
    dq_dr = np.zeros_like(r_tip)
    dq_dr[in_ring] = -1.0 / dr
    r_safe = np.maximum(r_tip, 1.0e-15)
    q_x = dq_dr * (x - crack.tip_x) / r_safe
    q_y = dq_dr * (y - crack.tip_y) / r_safe

    # Compute displacement gradients (actual field)
    u_x = np.zeros_like(u_field)
    u_y = np.zeros_like(u_field)
    v_x = np.zeros_like(v_field)
    v_y = np.zeros_like(v_field)

    u_x[:, 1:-1] = (u_field[:, 2:] - u_field[:, :-2]) / (2.0 * dx)
    u_y[1:-1, :] = (u_field[2:, :] - u_field[:-2, :]) / (2.0 * dy)
    v_x[:, 1:-1] = (v_field[:, 2:] - v_field[:, :-2]) / (2.0 * dx)
    v_y[1:-1, :] = (v_field[2:, :] - v_field[:-2, :]) / (2.0 * dy)

    # Stress field (actual)
    lam = material.lame_lambda
    sxx = (lam + 2.0 * mu) * u_x + lam * v_y
    syy = lam * u_x + (lam + 2.0 * mu) * v_y
    sxy = mu * (u_y + v_x)

    # Strain energy density
    W = 0.5 * (sxx * u_x + syy * v_y + sxy * (u_y + v_x))

    # Auxiliary mode-I asymptotic fields (K_I^aux = 1)
    # σ^{aux}_{ij} from Williams expansion
    sqrt_r = np.sqrt(np.maximum(r_tip, 1.0e-15))
    cos_t2 = np.cos(theta_tip / 2.0)
    sin_t2 = np.sin(theta_tip / 2.0)
    cos_t = np.cos(theta_tip)
    sin_t = np.sin(theta_tip)

    # Auxiliary stresses (mode-I, K_I = 1)
    coeff = 1.0 / np.maximum(sqrt_r * math.sqrt(2.0 * math.pi), 1.0e-15)
    sxx_aux = coeff * cos_t2 * (1.0 - sin_t2 * sin_t)
    syy_aux = coeff * cos_t2 * (1.0 + sin_t2 * sin_t)
    sxy_aux = coeff * cos_t2 * sin_t2 * cos_t

    # Auxiliary displacements (mode-I)
    kappa_aux = 3.0 - 4.0 * nu  # plane strain
    u_aux = coeff * sqrt_r / (2.0 * mu) * (
        kappa_aux * np.cos(theta_tip / 2.0) - np.cos(theta_tip / 2.0) * (1.0 - np.sin(theta_tip / 2.0) ** 2)
    )
    v_aux = coeff * sqrt_r / (2.0 * mu) * (
        kappa_aux * np.sin(theta_tip / 2.0) - np.sin(theta_tip / 2.0) * (1.0 - np.sin(theta_tip / 2.0) ** 2)
    )

    # Auxiliary displacement gradients (simplified: use asymptotic)
    # For interaction integral, we need u_{i,1}^{aux} = ∂u_i^{aux}/∂x_1
    # Approximate via numerical differentiation of the asymptotic field
    eps_aux = 1.0e-8
    r_tip_dx = np.sqrt(np.maximum((x - crack.tip_x + eps_aux) ** 2
                                   + (y - crack.tip_y) ** 2, 1.0e-15))
    theta_dx = np.arctan2(y - crack.tip_y, x - crack.tip_x + eps_aux)
    sqrt_r_dx = np.sqrt(r_tip_dx)
    cos_t2_dx = np.cos(theta_dx / 2.0)
    sin_t2_dx = np.sin(theta_dx / 2.0)
    coeff_dx = 1.0 / np.maximum(sqrt_r_dx * math.sqrt(2.0 * math.pi), 1.0e-15)
    u_aux_dx = coeff_dx * sqrt_r_dx / (2.0 * mu) * (
        kappa_aux * cos_t2_dx - cos_t2_dx * (1.0 - sin_t2_dx ** 2)
    )
    v_aux_dx = coeff_dx * sqrt_r_dx / (2.0 * mu) * (
        kappa_aux * sin_t2_dx - sin_t2_dx * (1.0 - sin_t2_dx ** 2)
    )
    u1_aux_x = (u_aux_dx - u_aux) / eps_aux
    v1_aux_x = (v_aux_dx - v_aux) / eps_aux

    # W^{int} (interaction strain energy density)
    W_int = 0.5 * (sxx * u1_aux_x + syy * v1_aux_x
                   + sxy_aux * (u_y + v_x)
                   + sxy * (u1_aux_x + v1_aux_x))

    # M-integral: M = ∫ [σ_{ij} u_{i,1}^{aux} + σ_{ij}^{aux} u_{i,1}
    #                      - W^{int} δ_{1j}] q_{,j} dA
    integrand_x = (sxx * u1_aux_x + sxy * v1_aux_x
                   + sxx_aux * u_x + sxy_aux * v_x
                   - W_int) * q_x
    integrand_y = (sxy * u1_aux_x + syy * v1_aux_x
                   + sxy_aux * u_y + syy_aux * v_y) * q_y

    M_integral = np.sum(integrand_x + integrand_y) * dx * dy

    # Extract K_I
    K_I_extracted = E_prime * M_integral / 2.0

    # Similarly for K_II (using mode-II auxiliary) — simplified here
    # For brevity, set K_II = 0 (symmetric loading assumed)
    K_II_extracted = 0.0

    # Dynamic energy release rate
    # G = (K_I² + K_II²) / E'
    G_dynamic = (K_I_extracted ** 2 + K_II_extracted ** 2) / E_prime

    return {
        "K_I": float(K_I_extracted),
        "K_II": float(K_II_extracted),
        "G_dynamic": float(G_dynamic),
        "M_integral": float(M_integral),
    }


# ===================================================================
# Crack propagation velocity
# ===================================================================

def compute_crack_tip_velocity(K_I: float,
                               K_IC: float,
                               c_R: float,
                               G_c: float,
                               E_prime: float) -> float:
    """Compute dynamic crack-tip velocity from the Broberg model.

    For a running crack, the dynamic energy release rate is:
        G(v) = g(v) * G_static

    where g(v) is a universal function of velocity:
        g(v) ≈ (1 - v/c_R) / sqrt(1 - v/c_d)    (approximate)

    The crack velocity is determined from the energy balance:
        G(v) = G_c   (fracture energy)

    Simplified Freund relation:
        v / c_R ≈ 1 - (K_IC / K_I)²    for K_I > K_IC

    Returns v [m/s]. Returns 0 if K_I < K_IC.
    """
    if K_I <= K_IC:
        return 0.0

    ratio = K_IC / max(K_I, 1.0e-15)
    v_over_cr = 1.0 - ratio ** 2
    v_over_cr = max(0.0, min(v_over_cr, 0.99))  # limit to Rayleigh speed

    return v_over_cr * c_R


# ===================================================================
# Crack path update
# ===================================================================

def advance_crack(crack: CrackConfig,
                  propagation_angle: float,
                  da: float) -> CrackConfig:
    """Advance the crack tip by distance da in direction propagation_angle.

    Creates a new CrackConfig with updated tip position.
    The crack length increases by da.
    """
    old_angle = crack.angle
    new_angle = old_angle + propagation_angle

    new_tip_x = crack.tip_x + da * math.cos(new_angle)
    new_tip_y = crack.tip_y + da * math.sin(new_angle)

    new_crack = CrackConfig(
        tip_x=new_tip_x,
        tip_y=new_tip_y,
        length=crack.length + da,
        angle=new_angle,
        tip_radius=crack.tip_radius,
    )

    return new_crack
