"""
thermal_solver.py

Thermal evolution and chemical differentiation solver for mantle convection.

Core seed mappings:
- 488_grazing_ode -> coupled nonlinear ODE model for upper/lower mantle
                     chemical reservoir exchange (grazing model analogy)

Scientific formulas:
- Advection-diffusion equation in polar coordinates:
    ∂T/∂t + u·∇T = κ ∇²T + Q
    where Q = H/(ρ₀ C_p) is internal heating.
- Time discretization (implicit Euler for diffusion, explicit for advection):
    (T^{n+1} − T^n)/Δt = −u^n·∇T^n + κ ∇²T^{n+1} + Q
- Upper-Lower Mantle chemical exchange (grazing analogy):
    dC_um/dt = r1 * C_um * (1 − C_um/K) − c1 * C_lm * (1 − exp(−d1 * C_um))
    dC_lm/dt = −a * C_lm + c2 * C_lm * (1 − exp(−d2 * C_um))
    where C_um, C_lm are incompatible element concentrations in
    upper and lower mantle reservoirs.
- Nusselt number from surface heat flux:
    Nu = −(D/ΔT) (∂T/∂r)|_{r=R_surf}
"""

import numpy as np
from typing import Tuple, Optional
from mantle_physics import MantleConstants, ThermalPhysics


class GrazingChemicalExchange:
    """
    Coupled ODE model for chemical exchange between upper and lower mantle,
    analogous to the grazing predator-prey model (seed 488_grazing_ode).

    The model treats incompatible element enrichment in the upper mantle
    (UM) as the "prey" and depletion in the lower mantle (LM) as the
    "predator" dynamics, parameterized by:
        a   : rate of LM mixing/homogenization
        c1  : maximum transfer rate from LM to UM (subduction)
        c2  : rate of LM differentiation feedback
        d1  : efficiency of subduction transfer at low UM enrichment
        d2  : efficiency of LM feedback at low UM enrichment
        K   : carrying capacity for UM enrichment
        r1  : intrinsic growth rate of UM enrichment (radiogenic + volcanic)
    """
    def __init__(self, a: float = 1.1, c1: float = 1.2, c2: float = 1.5,
                 d1: float = 0.001, d2: float = 0.001, K: float = 3000.0,
                 r1: float = 0.8):
        self.a = a
        self.c1 = c1
        self.c2 = c2
        self.d1 = d1
        self.d2 = d2
        self.K = K
        self.r1 = r1

    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Compute dy/dt for state y = [C_um, C_lm].
        """
        C_um = max(y[0], 0.0)
        C_lm = max(y[1], 0.0)
        d_um = self.r1 * C_um * (1.0 - C_um / self.K) - self.c1 * C_lm * (1.0 - np.exp(-self.d1 * C_um))
        d_lm = -self.a * C_lm + self.c2 * C_lm * (1.0 - np.exp(-self.d2 * C_um))
        return np.array([d_um, d_lm], dtype=float)

    def integrate_rk4(self, y0: np.ndarray, t_span: Tuple[float, float],
                      n_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate the chemical exchange ODE using 4th-order Runge-Kutta.
        """
        y0 = np.asarray(y0, dtype=float)
        if len(y0) != 2:
            raise ValueError("y0 must have length 2")
        t0, t1 = t_span
        dt = (t1 - t0) / n_steps
        t_vals = np.linspace(t0, t1, n_steps + 1)
        y_vals = np.zeros((n_steps + 1, 2), dtype=float)
        y_vals[0, :] = y0
        y = y0.copy()
        for i in range(n_steps):
            k1 = self.derivatives(t_vals[i], y)
            k2 = self.derivatives(t_vals[i] + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = self.derivatives(t_vals[i] + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = self.derivatives(t_vals[i] + dt, y + dt * k3)
            y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            # Boundary handling: clip to physical range
            y[0] = np.clip(y[0], 0.0, self.K * 1.5)
            y[1] = np.clip(y[1], 0.0, self.K * 1.5)
            y_vals[i + 1, :] = y
        return t_vals, y_vals


class ThermalSolver:
    """
    Finite-difference thermal solver for mantle temperature evolution.
    """
    def __init__(self, kappa: float = MantleConstants.kappa,
                 H: float = MantleConstants.H_radio,
                 rho0: float = MantleConstants.rho0,
                 Cp: float = MantleConstants.Cp,
                 T_surf: float = MantleConstants.T_surf,
                 T_cmb: float = MantleConstants.T_cmb):
        self.physics = ThermalPhysics(kappa, H, rho0, Cp)
        self.T_surf = T_surf
        self.T_cmb = T_cmb
        self.Q = self.physics.heat_production_term()

    def initial_temperature(self, r_grid: np.ndarray, theta_grid: np.ndarray,
                           mode: str = "linear") -> np.ndarray:
        """
        Create initial temperature field.
        mode='linear': purely conductive profile
            T(r) = T_surf + (T_cmb − T_surf) * (r − R_inner)/(R_outer − R_inner)
        mode='perturbation': add sinusoidal perturbation to trigger convection
        """
        nr, ntheta = r_grid.shape
        R_inner = float(np.min(r_grid))
        R_outer = float(np.max(r_grid))
        T = np.zeros((nr, ntheta), dtype=float)
        for i in range(nr):
            for j in range(ntheta):
                r = r_grid[i, j]
                if mode == "linear":
                    T[i, j] = self.T_surf + (self.T_cmb - self.T_surf) * (r - R_inner) / (R_outer - R_inner)
                elif mode == "perturbation":
                    T_cond = self.T_surf + (self.T_cmb - self.T_surf) * (r - R_inner) / (R_outer - R_inner)
                    perturb = 50.0 * np.sin(4.0 * theta_grid[i, j]) * np.sin(np.pi * (r - R_inner) / (R_outer - R_inner))
                    T[i, j] = T_cond + perturb
                else:
                    raise ValueError(f"Unknown mode: {mode}")
        # Boundary handling
        T[0, :] = self.T_cmb   # inner boundary (CMB)
        T[-1, :] = self.T_surf  # outer boundary (surface)
        return T

    def step_forward(self, T: np.ndarray, u_r: np.ndarray, u_theta: np.ndarray,
                     r_grid: np.ndarray, theta_grid: np.ndarray,
                     dt: float) -> np.ndarray:
        """
        Advance temperature field by one time step using operator splitting:
        1. Explicit advection
        2. Implicit diffusion (simplified via forward Euler with CFL check)
        """
        T = np.asarray(T, dtype=float)
        nr, ntheta = T.shape
        if nr < 3 or ntheta < 3:
            raise ValueError("Grid too small")
        dr = float(np.mean(np.diff(r_grid[:, 0])))
        dtheta = float(np.mean(np.diff(theta_grid[0, :])))
        # CFL check for stability
        u_max = max(np.max(np.abs(u_r)), np.max(np.abs(u_theta)))
        if u_max > 1e-15:
            dt_cfl = min(dr, np.min(r_grid) * dtheta) / u_max
            if dt > 0.5 * dt_cfl:
                dt = 0.5 * dt_cfl
        # Diffusion stability
        dt_diff = 0.25 * min(dr ** 2, (np.min(r_grid) * dtheta) ** 2) / self.physics.kappa
        if dt > 0.5 * dt_diff:
            dt = 0.5 * dt_diff
        # <HOLE 2: 显式时间推进更新公式 T_new = T + dt * (-adv + κ·lap + Q)>
        raise NotImplementedError("HOLE 2: 需要实现显式时间推进更新公式")
        # Enforce boundary conditions
        T_new[0, :] = self.T_cmb
        T_new[-1, :] = self.T_surf
        # Periodic in theta
        T_new[:, 0] = T_new[:, -2]
        T_new[:, -1] = T_new[:, 1]
        # Physical bounds
        T_new = np.clip(T_new, self.T_surf, self.T_cmb)
        return T_new

    def surface_heat_flux(self, T: np.ndarray, r_grid: np.ndarray) -> float:
        """
        Compute non-dimensional surface heat flux:
            q* = −(∂T/∂r) / (ΔT / D_eff)
        where D_eff is the radial extent of the domain.
        This gives Nu = q* when compared with conductive flux.
        """
        dr = float(np.mean(np.diff(r_grid[:, 0])))
        dTdr = (T[-1, :] - T[-2, :]) / dr
        r_min = float(np.min(r_grid))
        r_max = float(np.max(r_grid))
        D_eff = max(r_max - r_min, 1e-15)
        delta_T = max(self.T_cmb - self.T_surf, 1e-15)
        q_nd = -np.mean(dTdr) / (delta_T / D_eff)
        return float(q_nd)
