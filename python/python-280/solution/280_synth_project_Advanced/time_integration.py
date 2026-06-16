"""
time_integration.py — Time-stepping schemes for damage-coupled elastodynamics.

Implements:
  1. Forward Euler (1st order explicit)
  2. Classical Runge-Kutta 4 (4th order explicit)
  3. IMEX-RK2 (2nd order implicit-explicit: elastic explicit, damage implicit)
  4. Newmark-β (2nd order implicit structural dynamics)

  plus adaptive time-step control via CFL condition and energy monitoring.

The semi-discrete system after FD spatial discretization is:

    M ü + C(D) u = f(t)         (momentum)
    Ḋ = g(ε(u), D, κ)           (damage evolution)

where M = ρI is the mass matrix (lumped), C(D) = (1-D)K is the
damage-dependent stiffness operator, and g is the damage rate function.

IMEX splitting:
    Explicit part: L u = K u / ρ       (linear elastic wave operator)
    Implicit part: N(u, D)             (damage coupling + nonlinear terms)
"""

import math
import numpy as np
from typing import Dict, Tuple, Callable, Optional
from config import SimulationConfig


# ===================================================================
# Adaptive time-step control
# ===================================================================

def compute_cfl_timestep(dx: float, dy: float,
                         wave_speed: float,
                         cfl_number: float,
                         fd_order: int = 4) -> float:
    """Compute the maximum stable time step from the CFL condition.

    For an explicit scheme with FD order p, the CFL condition is:
        Δt ≤ CFL * h_min / c_max

    where h_min = min(dx, dy) and c_max is the fastest wave speed.

    For the 4th-order scheme, the effective CFL limit is tighter:
        Δt_4th ≈ Δt_2nd * sqrt(3/4) ≈ 0.866 * Δt_2nd

    because the 4th-order stencil has a larger spectral radius.

    More precisely, for the 4th-order central FD applied to the
    wave equation in 2D:
        c Δt / h ≤ 1 / sqrt( (1 + 1/6) / h² + (1 + 1/6) / h² )
                = h / (c * sqrt(7/3))
                ≈ 0.6547 * h / c
    """
    h_min = min(dx, dy)

    if fd_order == 2:
        cfl_limit = 1.0 / math.sqrt(2.0)  # ≈ 0.707
    elif fd_order == 4:
        cfl_limit = 1.0 / math.sqrt(7.0 / 3.0)  # ≈ 0.6547
    elif fd_order == 6:
        cfl_limit = 1.0 / math.sqrt(41.0 / 15.0)  # ≈ 0.605
    else:
        cfl_limit = 0.5

    dt_max = cfl_number * cfl_limit * h_min / max(wave_speed, 1.0e-15)

    return max(dt_max, 1.0e-15)


def adapt_timestep(dt_current: float,
                   energy_ratio: float,
                   damage_rate_max: float,
                   cfg: SimulationConfig) -> float:
    """Adapt the time step based on energy conservation and damage rate.

    Strategy:
      - If energy is well conserved (ratio close to 1), grow dt.
      - If damage is evolving rapidly, shrink dt.
      - Respect min/max bounds.

    Energy ratio: E_current / E_previous
    Damage rate: max(Ḋ) across all grid points.

    Growth/shrinkage uses smooth factors to avoid oscillation.
    """
    params = cfg.numerical

    # Energy-based adjustment
    energy_deviation = abs(energy_ratio - 1.0)
    if energy_deviation < 0.01:
        energy_factor = params.max_dt_growth_factor
    elif energy_deviation < 0.05:
        energy_factor = 1.0
    else:
        energy_factor = 0.8  # shrink

    # Damage-rate-based adjustment
    # If damage rate is high, we need smaller time steps
    damage_threshold = 1.0 / cfg.material.characteristic_length
    if damage_rate_max > damage_threshold:
        damage_factor = damage_threshold / max(damage_rate_max, 1.0e-15)
        damage_factor = max(damage_factor, 0.5)
    else:
        damage_factor = 1.0

    # Combined
    dt_new = dt_current * min(energy_factor, damage_factor)
    dt_new = max(dt_new, params.min_dt)
    dt_new = min(dt_new, dt_current * params.max_dt_growth_factor)

    return dt_new


# ===================================================================
# Forward Euler
# ===================================================================

def forward_euler_step(u: np.ndarray, v_disp: np.ndarray,
                       u_vel: np.ndarray, v_vel: np.ndarray,
                       accel_x: np.ndarray, accel_y: np.ndarray,
                       dt: float,
                       density: float) -> Tuple[np.ndarray, ...]:
    """One step of forward Euler for the elastodynamic system.

    u^{n+1} = u^n + dt * u̇^n
    u̇^{n+1} = u̇^n + dt * (rhs_x / ρ)

    This is 1st-order accurate and conditionally stable.
    Energy is NOT conserved (artificial damping).
    """
    u_new = u + dt * u_vel
    v_new = v_disp + dt * v_vel
    u_vel_new = u_vel + dt * accel_x / density
    v_vel_new = v_vel + dt * accel_y / density

    return u_new, v_new, u_vel_new, v_vel_new


# ===================================================================
# Classical RK4
# ===================================================================

def rk4_step(state: Tuple[np.ndarray, ...],
             rhs_func: Callable,
             dt: float) -> Tuple[np.ndarray, ...]:
    """Classical 4th-order Runge-Kutta step.

    For the system  ẏ = F(t, y):
        k₁ = F(t_n, y_n)
        k₂ = F(t_n + h/2, y_n + h/2 k₁)
        k₃ = F(t_n + h/2, y_n + h/2 k₂)
        k₄ = F(t_n + h, y_n + h k₃)
        y_{n+1} = y_n + h/6 (k₁ + 2k₂ + 2k₃ + k₄)

    state = (u, v_disp, u_vel, v_vel)
    rhs_func(state) -> (du/dt, dv/dt, du_vel/dt, dv_vel/dt)
    """
    u, v_d, u_v, v_v = state

    k1 = rhs_func(u, v_d, u_v, v_v)

    u2 = u + 0.5 * dt * k1[0]
    v2 = v_d + 0.5 * dt * k1[1]
    uv2 = u_v + 0.5 * dt * k1[2]
    vv2 = v_v + 0.5 * dt * k1[3]
    k2 = rhs_func(u2, v2, uv2, vv2)

    u3 = u + 0.5 * dt * k2[0]
    v3 = v_d + 0.5 * dt * k2[1]
    uv3 = u_v + 0.5 * dt * k2[2]
    vv3 = v_v + 0.5 * dt * k2[3]
    k3 = rhs_func(u3, v3, uv3, vv3)

    u4 = u + dt * k3[0]
    v4 = v_d + dt * k3[1]
    uv4 = u_v + dt * k3[2]
    vv4 = v_v + dt * k3[3]
    k4 = rhs_func(u4, v4, uv4, vv4)

    u_new = u + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
    v_new = v_d + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
    uv_new = u_v + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
    vv_new = v_v + (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])

    return u_new, v_new, uv_new, vv_new


# ===================================================================
# IMEX-RK2 (Implicit-Explicit Runge-Kutta, 2nd order)
# ===================================================================

def imex_rk2_step(u: np.ndarray, v_disp: np.ndarray,
                  u_vel: np.ndarray, v_vel: np.ndarray,
                  explicit_rhs: Callable,
                  implicit_solve: Callable,
                  dt: float) -> Tuple[np.ndarray, ...]:
    """IMEX Runge-Kutta 2nd-order step (ARS(2,2,2) scheme).

    Splits the RHS into explicit (linear wave) and implicit (damage coupling):

    Stage 1:
        y* = y^n + Δt * L(y^n)         [explicit]
        y^{n+1/2} = y* + Δt * N(y^{n+1/2})   [implicit, nonlinear solve]

    Stage 2:
        y^{n+1} = y^n + Δt/2 * [L(y^n) + L(y^{n+1/2})]
                  + Δt/2 * [N(y^{n+1/2}) + N(y^{n+1})]

    For the damage problem, the "implicit solve" is a fixed-point iteration
    on the damage update (since the damage ODE is stiff but the wave part
    is handled explicitly).

    This is the ARS(2,2,2) scheme of Ascher, Ruuth, Spiteri (1997):
        γ = 1 - 1/√2
        a21 = γ,  ã21 = 1-γ,  b1 = b2 = 1/2,  b̃1 = b̃2 = 1/2
    """
    gamma = 1.0 - 1.0 / math.sqrt(2.0)

    # Stage 1: explicit part
    L_u, L_v, L_uv, L_vv = explicit_rhs(u, v_disp, u_vel, v_vel)
    u_star = u + dt * gamma * L_u
    v_star = v_disp + dt * gamma * L_v
    uv_star = u_vel + dt * gamma * L_uv
    vv_star = v_vel + dt * gamma * L_vv

    # Stage 1: implicit part (fixed-point iteration on damage)
    u_half, v_half, uv_half, vv_half = implicit_solve(
        u_star, v_star, uv_star, vv_star, dt * gamma
    )

    # Stage 2: explicit
    L2_u, L2_v, L2_uv, L2_vv = explicit_rhs(u_half, v_half, uv_half, vv_half)

    # Stage 2: combine
    u_new = u + dt * ((1.0 - gamma) * L_u + gamma * L2_u)
    v_new = v_disp + dt * ((1.0 - gamma) * L_v + gamma * L2_v)
    uv_new = u_vel + dt * ((1.0 - gamma) * L_uv + gamma * L2_uv)
    vv_new = v_vel + dt * ((1.0 - gamma) * L_vv + gamma * L2_vv)

    # Final implicit correction
    u_new, v_new, uv_new, vv_new = implicit_solve(
        u_new, v_new, uv_new, vv_new, dt * (1.0 - gamma)
    )

    return u_new, v_new, uv_new, vv_new


# ===================================================================
# Newmark-β method
# ===================================================================

def newmark_beta_step(u: np.ndarray, v_disp: np.ndarray,
                      u_vel: np.ndarray, v_vel: np.ndarray,
                      u_acc: np.ndarray, v_acc: np.ndarray,
                      rhs_func: Callable,
                      dt: float,
                      beta: float = 0.25,
                      gamma_nm: float = 0.50,
                      density: float = 2400.0,
                      n_iter: int = 5) -> Tuple[np.ndarray, ...]:
    """Newmark-β implicit time integration for structural dynamics.

    Displacement update:
        u^{n+1} = u^n + Δt u̇^n + Δt²/2 [(1-2β) ü^n + 2β ü^{n+1}]

    Velocity update:
        u̇^{n+1} = u̇^n + Δt [(1-γ) ü^n + γ ü^{n+1}]

    For β = 1/4, γ = 1/2 (average acceleration), the scheme is:
      - 2nd-order accurate
      - Unconditionally stable (for linear problems)
      - Non-dissipative (no numerical damping)

    We solve for ü^{n+1} via fixed-point iteration:
        ü^{n+1} = (1/ρ) * rhs(u^{n+1})
    """
    # Predict displacement
    u_pred = (u + dt * u_vel
              + 0.5 * dt ** 2 * (1.0 - 2.0 * beta) * u_acc)
    v_pred = (v_disp + dt * v_vel
              + 0.5 * dt ** 2 * (1.0 - 2.0 * beta) * v_acc)

    # Initial guess for acceleration
    acc_x_new = u_acc.copy()
    acc_y_new = v_acc.copy()

    for iteration in range(n_iter):
        # Update displacement with current acceleration guess
        u_trial = u_pred + beta * dt ** 2 * acc_x_new
        v_trial = v_pred + beta * dt ** 2 * acc_y_new

        # Compute new acceleration from RHS
        acc_x_new_trial, acc_y_new_trial = rhs_func(u_trial, v_trial)

        # Check convergence
        diff = (np.max(np.abs(acc_x_new_trial - acc_x_new))
                + np.max(np.abs(acc_y_new_trial - acc_y_new)))
        acc_x_new = acc_x_new_trial
        acc_y_new = acc_y_new_trial

        if diff < 1.0e-10:
            break

    # Final update
    u_new = u_pred + beta * dt ** 2 * acc_x_new
    v_new = v_pred + beta * dt ** 2 * acc_y_new
    uv_new = u_vel + dt * ((1.0 - gamma_nm) * u_acc + gamma_nm * acc_x_new)
    vv_new = v_vel + dt * ((1.0 - gamma_nm) * v_acc + gamma_nm * acc_y_new)

    return u_new, v_new, uv_new, vv_new, acc_x_new, acc_y_new


# ===================================================================
# Energy computation
# ===================================================================

def compute_total_energy(u: np.ndarray, v_disp: np.ndarray,
                         u_vel: np.ndarray, v_vel: np.ndarray,
                         damage: np.ndarray,
                         cfg: SimulationConfig) -> Dict[str, float]:
    """Compute kinetic, strain, and total energy of the system.

    Kinetic energy:  K = ½ ∫ ρ (u̇² + v̇²) dA
    Strain energy:   U = ½ ∫ (1-D) σ:ε dA
    Total:           E = K + U

    For the damaged system, energy is dissipated:
        dE/dt = -∫ Y Ḋ dA ≤ 0

    where Y is the energy release rate.
    """
    dx = cfg.dx()
    dy = cfg.dy()
    dA = dx * dy
    rho = cfg.material.mass_density
    lam = cfg.material.lame_lambda
    mu = cfg.material.lame_mu

    # Kinetic energy
    kinetic = 0.5 * rho * np.sum(u_vel ** 2 + v_vel ** 2) * dA

    # Strain energy (with damage degradation)
    # Compute strains
    ny, nx = u.shape
    eps_xx = np.zeros_like(u)
    eps_yy = np.zeros_like(u)
    eps_xy = np.zeros_like(u)
    if nx > 2:
        eps_xx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dx)
    if ny > 2:
        eps_yy[1:-1, :] = (v_disp[2:, :] - v_disp[:-2, :]) / (2.0 * dy)
    if nx > 2 and ny > 2:
        eps_xy[1:-1, 1:-1] = 0.5 * (
            (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dy) +
            (v_disp[1:-1, 2:] - v_disp[1:-1, :-2]) / (2.0 * dx)
        )

    stiffness = np.clip(1.0 - damage, 1.0e-8, 1.0)
    strain_energy_density = 0.5 * stiffness * (
        (lam + 2.0 * mu) * (eps_xx ** 2 + eps_yy ** 2)
        + 2.0 * lam * eps_xx * eps_yy
        + 4.0 * mu * eps_xy ** 2
    )
    strain_energy = np.sum(strain_energy_density) * dA

    total = kinetic + strain_energy

    return {
        "kinetic_energy": float(kinetic),
        "strain_energy": float(strain_energy),
        "total_energy": float(total),
    }
