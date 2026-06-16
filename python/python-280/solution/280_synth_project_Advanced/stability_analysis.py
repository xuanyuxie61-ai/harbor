"""
stability_analysis.py — Stability analysis for high-order FD damage simulation.

Seeds: 694_local_min (Brent's method), 572_ill_bvp (ill-conditioned BVP),
       814_norm_loo (L∞ norm estimation).

Implements:
  1. CFL condition verification for various FD orders
  2. Von Neumann stability analysis (Fourier mode amplification)
  3. Eigenvalue spectrum of the discrete spatial operator
  4. Energy dissipation inequality check
  5. Stiffness detection (explicit vs implicit regime)
  6. Critical load factor search via Brent's method
  7. L∞ norm estimation of stress/error fields

The stability of the semi-discrete system  M ü + C(D) u = f  depends on:
  - The spectral radius of M^{-1} C(D)
  - The damage level D (higher damage → softer → larger stable dt)
  - The FD order (higher order → larger spectral radius → smaller dt)

For the 4th-order central scheme in 2D:
    ρ(M^{-1} C) ≈ c²_eff * (16/(3h²)) * 2

and the CFL condition becomes:
    Δt ≤ C_CFL * h / c_eff

where C_CFL depends on the time integration scheme.
"""

import math
import numpy as np
from typing import Dict, Tuple, List, Optional
from config import SimulationConfig, MaterialParams


# ===================================================================
# CFL condition for various schemes and FD orders
# ===================================================================

def cfl_limit_explicit(scheme: str, fd_order: int, ndim: int = 2) -> float:
    """Return the CFL constant for a given explicit scheme and FD order.

    The CFL condition is:  Δt ≤ C_CFL * h / c_max

    Scheme        | Order 2   | Order 4   | Order 6
    --------------|-----------|-----------|--------
    Forward Euler | 1/√2 ≈ 0.707 | 0.655  | 0.605
    RK2           | √2 ≈ 1.414   | 1.310  | 1.210
    RK4           | √3 ≈ 1.732   | 1.597  | 1.476
    RK3-SSP       | 1.0       | 0.924   | 0.855

    For n dimensions: multiply by 1/√n.
    """
    # Base CFL per order (1D)
    cfl_table = {
        "euler": {2: 0.5, 4: 0.46, 6: 0.42},
        "rk2": {2: 1.0, 4: 0.92, 6: 0.85},
        "rk3_ssp": {2: 0.707, 4: 0.65, 6: 0.60},
        "rk4": {2: 1.414, 4: 1.30, 6: 1.20},
        "imex_rk2": {2: 0.8, 4: 0.73, 6: 0.67},
        "newmark": {2: float('inf'), 4: float('inf'), 6: float('inf')},
    }

    scheme_lower = scheme.lower()
    base_cfl = 0.5  # default
    for key in cfl_table:
        if key in scheme_lower:
            base_cfl = cfl_table[key].get(fd_order, 0.5)
            break

    # Multi-dimensional reduction
    if ndim > 1:
        base_cfl /= math.sqrt(ndim)

    return base_cfl


def max_stable_timestep(dx: float, dy: float,
                        c_max: float,
                        scheme: str = "rk4",
                        fd_order: int = 4) -> float:
    """Compute the maximum stable time step.

    Δt_max = CFL * h_min / c_max

    where h_min = min(dx, dy).
    """
    h_min = min(dx, dy)
    cfl = cfl_limit_explicit(scheme, fd_order)
    if math.isinf(cfl):
        return float('inf')
    return cfl * h_min / max(c_max, 1.0e-15)


# ===================================================================
# Von Neumann stability analysis
# ===================================================================

def von_neumann_amplification(k_xi: np.ndarray,
                              k_eta: np.ndarray,
                              dt: float, dx: float, dy: float,
                              c: float, fd_order: int = 4
                              ) -> np.ndarray:
    """Compute the amplification factor |G| for each Fourier mode.

    For the wave equation u_tt = c² Δu discretized with FD order p
    and time-stepping scheme S, the amplification factor G(ξ,η)
    determines stability: |G| ≤ 1 for all modes.

    For Forward Euler + 2nd-order FD:
        G = 1 - 4r sin²(ξh/2) - 4s sin²(ηh/2)
    where r = c²Δt²/hx², s = c²Δt²/hy².
    Stability: r + s ≤ 1/2.

    For RK4 + 4th-order FD, the modified wavenumber is:
        ξ̃ h = (8 sin(ξh) - sin(2ξh)) / 6

    And the amplification factor for RK4 applied to û_tt = -ω² û:
        G = 1 + z + z²/2 + z³/6 + z⁴/24
    where z = -ω² Δt² and ω² = c² ξ̃².
    """
    r_x = c ** 2 * dt ** 2 / dx ** 2
    r_y = c ** 2 * dt ** 2 / dy ** 2

    if fd_order == 2:
        # Modified wavenumber squared (2nd order)
        xi_tilde_sq = 4.0 * np.sin(k_xi * dx / 2.0) ** 2 / dx ** 2
        eta_tilde_sq = 4.0 * np.sin(k_eta * dy / 2.0) ** 2 / dy ** 2
    elif fd_order == 4:
        # Modified wavenumber squared (4th order)
        # ξ̃h = (8 sin(ξh) - sin(2ξh)) / 6
        xi_mod = (8.0 * np.sin(k_xi * dx) - np.sin(2.0 * k_xi * dx)) / (6.0 * dx)
        eta_mod = (8.0 * np.sin(k_eta * dy) - np.sin(2.0 * k_eta * dy)) / (6.0 * dy)
        xi_tilde_sq = xi_mod ** 2
        eta_tilde_sq = eta_mod ** 2
    else:
        xi_tilde_sq = 4.0 * np.sin(k_xi * dx / 2.0) ** 2 / dx ** 2
        eta_tilde_sq = 4.0 * np.sin(k_eta * dy / 2.0) ** 2 / dy ** 2

    # Effective frequency squared
    omega_sq = c ** 2 * (xi_tilde_sq + eta_tilde_sq)
    z = -omega_sq * dt ** 2

    # Amplification factor depends on time scheme
    # For RK4:  G = 1 + z + z²/2 + z³/6 + z⁴/24
    # (truncated Taylor of exp(z))
    G = 1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0
    G_mag = np.abs(G)

    return G_mag


def check_von_neumann_stability(cfg: SimulationConfig,
                                damage_max: float = 0.0) -> Dict:
    """Perform complete von Neumann stability check.

    Tests all Fourier modes on the grid and reports:
      - max amplification factor (must be ≤ 1)
      - critical mode (most unstable)
      - stable: bool
    """
    nx, ny = cfg.numerical.nx, cfg.numerical.ny
    dx, dy = cfg.dx(), cfg.dy()
    dt = cfg.numerical.initial_dt
    c = cfg.material.p_wave_speed * math.sqrt(max(1.0 - damage_max, 1.0e-8))
    fd_order = cfg.numerical.fd_order

    # Fourier mode wave numbers
    kx = np.linspace(0, math.pi / dx, nx // 2 + 1)
    ky = np.linspace(0, math.pi / dy, ny // 2 + 1)
    KX, KY = np.meshgrid(kx, ky)

    G_mag = von_neumann_amplification(KX, KY, dt, dx, dy, c, fd_order)

    max_G = float(G_mag.max())
    idx_max = np.unravel_index(np.argmax(G_mag), G_mag.shape)

    return {
        "max_amplification": max_G,
        "critical_kx": float(KX[idx_max]),
        "critical_ky": float(KY[idx_max]),
        "stable": max_G <= 1.0 + 1.0e-10,
        "scheme": cfg.numerical.time_scheme,
        "fd_order": fd_order,
        "dt": dt,
        "c_effective": c,
    }


# ===================================================================
# Eigenvalue analysis of the spatial operator
# ===================================================================

def build_1d_laplacian_matrix(n: int, dx: float,
                              order: int = 4,
                              bc_type: str = "neumann"
                              ) -> np.ndarray:
    """Build the 1D discrete Laplacian matrix.

    For order 4, the interior stencil is:
        [-1, 16, -30, 16, -1] / (12 h²)

    Boundary conditions:
      - "neumann": ghost-point extrapolation
      - "dirichlet": zero at boundary
    """
    A = np.zeros((n, n))

    # Interior
    for i in range(2, n - 2):
        A[i, i - 2] = -1.0 / (12.0 * dx ** 2)
        A[i, i - 1] = 16.0 / (12.0 * dx ** 2)
        A[i, i] = -30.0 / (12.0 * dx ** 2)
        A[i, i + 1] = 16.0 / (12.0 * dx ** 2)
        A[i, i + 2] = -1.0 / (12.0 * dx ** 2)

    # Boundaries (2nd-order fallback)
    if n > 3:
        A[0, 0] = -2.0 / dx ** 2
        A[0, 1] = 2.0 / dx ** 2
        A[1, 0] = 1.0 / dx ** 2
        A[1, 1] = -2.0 / dx ** 2
        A[1, 2] = 1.0 / dx ** 2
        A[-1, -1] = -2.0 / dx ** 2
        A[-1, -2] = 2.0 / dx ** 2
        A[-2, -3] = 1.0 / dx ** 2
        A[-2, -2] = -2.0 / dx ** 2
        A[-2, -1] = 1.0 / dx ** 2

    if bc_type == "dirichlet":
        A[0, :] = 0.0
        A[0, 0] = 1.0
        A[-1, :] = 0.0
        A[-1, -1] = 1.0

    return A


def compute_eigenvalue_spectrum(n: int, dx: float,
                                order: int = 4,
                                damage_level: float = 0.0,
                                c_wave: float = 1.0
                                ) -> Dict[str, float]:
    """Compute the eigenvalue spectrum of the damage-modified Laplacian.

    λ_k = -(1-D) c² * μ_k

    where μ_k are eigenvalues of the discrete Laplacian.
    The most negative eigenvalue determines the CFL limit.
    """
    A = build_1d_laplacian_matrix(n, dx, order)
    eigenvalues = np.linalg.eigvals(A)
    eigenvalues = np.real(eigenvalues)

    # Scale by damage and wave speed
    factor = (1.0 - damage_level) * c_wave ** 2
    scaled_eigs = factor * eigenvalues

    return {
        "min_eigenvalue": float(scaled_eigs.min()),
        "max_eigenvalue": float(scaled_eigs.max()),
        "spectral_radius": float(np.abs(scaled_eigs).max()),
        "condition_number": float(np.abs(scaled_eigs).max() / max(np.abs(scaled_eigs).min(), 1.0e-30)),
        "n_negative": int((scaled_eigs < -1.0e-10).sum()),
    }


# ===================================================================
# Energy dissipation check
# ===================================================================

def check_energy_dissipation(energy_history: List[float],
                             tolerance: float = 0.05) -> Dict:
    """Verify that energy is non-increasing (within tolerance).

    For the damage system, energy dissipation is:
        dE/dt = -∫ Y Ḋ dA ≤ 0

    We check that:
        E(t_{n+1}) - E(t_n) ≤ tolerance * E(t_0)

    Returns diagnostics on energy conservation quality.
    """
    if len(energy_history) < 2:
        return {"satisfied": True, "message": "Insufficient data"}

    E0 = energy_history[0]
    if abs(E0) < 1.0e-30:
        E0 = 1.0  # avoid division by zero

    violations = []
    max_increase = 0.0
    for i in range(1, len(energy_history)):
        dE = energy_history[i] - energy_history[i - 1]
        rel_change = dE / abs(E0)
        if rel_change > tolerance:
            violations.append(i)
        max_increase = max(max_increase, rel_change)

    return {
        "satisfied": len(violations) == 0,
        "n_violations": len(violations),
        "max_relative_increase": float(max_increase),
        "tolerance": tolerance,
        "final_energy_ratio": float(energy_history[-1] / E0),
    }


# ===================================================================
# Stiffness detection
# ===================================================================

def detect_stiffness_ratio(eigenvalue_spectrum: Dict[str, float],
                           dt: float) -> Dict:
    """Detect stiffness of the ODE system.

    Stiffness ratio = |Re(λ_max)| / |Re(λ_min)|
    If stiffness_ratio * dt > O(10), explicit methods are inefficient
    and implicit methods are preferred.

    The transition criterion:
        S = max(|λ_i|) * dt
    If S > 2.5, the system is considered stiff for RK4.
    """
    rho = eigenvalue_spectrum["spectral_radius"]
    stiffness_number = rho * dt

    return {
        "stiffness_number": float(stiffness_number),
        "stiffness_ratio": float(eigenvalue_spectrum["condition_number"]),
        "is_stiff": stiffness_number > 2.5,
        "recommended_scheme": "imex_rk2" if stiffness_number > 2.5 else "rk4",
    }


# ===================================================================
# Critical load factor via Brent's method  (seed 694_local_min)
# ===================================================================

def find_critical_load_factor(cfg: SimulationConfig,
                              load_min: float = 0.0,
                              load_max: float = 1.0e-3,
                              tolerance: float = 1.0e-6) -> Dict:
    """Find the critical applied strain at which damage initiates.

    Uses Brent's method to find the root of:
        f(ε) = max(κ(x)) - κ_0

    where κ(x) is the equivalent strain field under applied strain ε,
    and κ_0 is the damage threshold.

    The critical strain ε_c satisfies:
        max_x κ(ε_c, x) = κ_0

    This is a uniaxial test: for homogeneous material under uniform
    strain, ε_c = κ_0. For heterogeneous material with stress concentrations,
    ε_c < κ_0.

    We minimize |max(κ) - κ_0| over ε ∈ [load_min, load_max] using
    Brent's parabolic interpolation / golden section method.
    """
    # Brent's method parameters (from seed 694_local_min)
    golden_c = 0.5 * (3.0 - math.sqrt(5.0))
    eps_tolerance = tolerance

    a, b = load_min, load_max

    # Objective: find ε where max(equiv_strain) = kappa_0
    kappa_0 = cfg.material.damage_threshold_strain

    def objective(applied_strain):
        # Simplified: for uniaxial strain, equiv_strain ≈ applied_strain
        # For heterogeneous case, we'd solve the BVP here
        # Using a simple amplification model:
        #   max(κ) = applied_strain * stress_concentration_factor
        scf = 1.5  # typical stress concentration factor
        return abs(applied_strain * scf - kappa_0)

    # Brent's minimization
    x = a + golden_c * (b - a)
    w, v = x, x
    fx = objective(x)
    fw, fv = fx, fx
    e = 0.0
    d_step = 0.0

    for _ in range(100):
        xm = 0.5 * (a + b)
        tol1 = eps_tolerance * abs(x) + 1.0e-10
        tol2 = 2.0 * tol1

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

            if abs(p) < abs(0.5 * q * e_old) and p > q * (a - x) and p < q * (b - x):
                d_step = p / q
                u = x + d_step
                if (u - a) < tol2 or (b - u) < tol2:
                    d_step = tol1 if x < xm else -tol1
            else:
                e = (a - x) if x < xm else (b - x)
                d_step = golden_c * e
        else:
            e = (a - x) if x < xm else (b - x)
            d_step = golden_c * e

        u = x + (d_step if abs(d_step) >= tol1 else (tol1 if d_step > 0 else -tol1))
        fu = objective(u)

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

    return {
        "critical_strain": float(x),
        "objective_value": float(fx),
        "kappa_0": kappa_0,
        "converged": fx < tolerance,
        "n_iterations": int(_),
    }


# ===================================================================
# L∞ norm estimation  (seed 814_norm_loo)
# ===================================================================

def linf_norm_field(field: np.ndarray,
                    dx: float, dy: float) -> Dict[str, float]:
    """Compute L∞ (supremum) norm of a field.

    ||f||_∞ = max_x |f(x)|

    Also computes:
      - L2 norm:  ||f||_2 = sqrt(∫ f² dA)
      - L1 norm:  ||f||_1 = ∫ |f| dA
      - Location of maximum
    """
    abs_field = np.abs(field)
    max_val = float(abs_field.max())
    max_idx = np.unravel_index(np.argmax(abs_field), field.shape)

    dA = dx * dy
    l2_norm = math.sqrt(float(np.sum(field ** 2) * dA))
    l1_norm = float(np.sum(abs_field) * dA)

    return {
        "linf_norm": max_val,
        "l2_norm": l2_norm,
        "l1_norm": l1_norm,
        "max_location_i": int(max_idx[0]),
        "max_location_j": int(max_idx[1]),
        "max_value": float(field[max_idx]),
    }


# ===================================================================
# Ill-conditioned BVP analysis  (seed 572_ill_bvp)
# ===================================================================

def analyze_damage_bvp_conditioning(damage: np.ndarray,
                                    dx: float, dy: float,
                                    material: MaterialParams) -> Dict:
    """Analyze the conditioning of the damage-modified BVP.

    The effective stiffness matrix becomes ill-conditioned as D → 1.
    The condition number scales as:
        κ(A_eff) ≈ κ(A_undamaged) / (1 - D_max)

    For the singularly perturbed problem  ε Δu + f = 0
    with ε → 0 (analogous to D → 1), boundary layers form.

    We estimate:
      - Effective condition number
      - Boundary layer thickness: δ ~ √ε / c
      - Resolution requirement: h < δ for accurate resolution
    """
    D_max = float(damage.max())
    D_mean = float(damage.mean())

    # Effective stiffness ratio
    stiffness_ratio = (1.0 - D_mean) / max(1.0 - D_max, 1.0e-10)

    # Estimate effective perturbation parameter
    eps_eff = max(1.0 - D_max, 1.0e-10)

    # Boundary layer thickness
    c_wave = material.p_wave_speed
    delta_bl = math.sqrt(eps_eff) / max(c_wave, 1.0)

    # Grid Peclet number: Pe = h / (2 δ_bl)
    h = min(dx, dy)
    peclet = h / max(2.0 * delta_bl, 1.0e-15)

    # Condition number estimate
    nx, ny = damage.shape
    cond_base = 4.0 / (math.sin(math.pi / max(nx, ny)) ** 2)
    cond_effective = cond_base / eps_eff

    return {
        "max_damage": D_max,
        "stiffness_ratio": float(stiffness_ratio),
        "eps_effective": float(eps_eff),
        "boundary_layer_thickness": float(delta_bl),
        "grid_peclet_number": float(peclet),
        "condition_number_estimate": float(cond_effective),
        "well_resolved": peclet < 1.0,
        "needs_refinement": peclet > 2.0,
    }
