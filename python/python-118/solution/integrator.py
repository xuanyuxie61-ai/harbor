"""
integrator.py

Time integration algorithms for molecular dynamics.

Synthesizes concepts from:
    - 833_ode_trapezoidal: Implicit trapezoidal method for ODEs
    - 270_dfield9: Direction field analysis, Dormand-Prince ODE solver
    - 1025_ripple_ode: Nonlinear ODE dynamics
    
Physical Model:
    Newton's equations of motion:
        dr_i/dt = v_i
        m_i * dv_i/dt = F_i(r_1, ..., r_N)
    
    This is a 3N-dimensional system of coupled second-order ODEs.
    
    Velocity Verlet Algorithm (symplectic integrator):
        v(t + dt/2) = v(t) + (dt/2) * F(t) / m
        r(t + dt)   = r(t) + dt * v(t + dt/2)
        Compute F(t + dt)
        v(t + dt)   = v(t + dt/2) + (dt/2) * F(t + dt) / m
    
    Properties:
        - Time-reversible
        - Symplectic (preserves phase space volume)
        - Second-order accurate: O(dt^2)
        - Energy drift is bounded (no secular growth)
    
    Implicit Trapezoidal Method (for stiff ODEs, e.g., thermostat):
        y_{n+1} = y_n + (dt/2) * [ f(t_n, y_n) + f(t_{n+1}, y_{n+1}) ]
    
    This is an implicit second-order method solved via fixed-point iteration:
        z^{(0)} = y_n
        z^{(k+1)} = y_n + (dt/2) * [ f(t_n, y_n) + f(t_{n+1}, z^{(k)}) ]
"""

import numpy as np
from config import TIME_STEP, MASS_A, MASS_B


# =============================================================================
# Velocity Verlet Integrator
# =============================================================================

class VelocityVerlet:
    """
    Velocity Verlet integrator for Hamiltonian dynamics.
    
    The algorithm preserves the symplectic structure of phase space:
        omega = sum_i dq_i ^ dp_i
    
    For a time step dt, the phase space map is:
        Phi_dt : (r, p) -> (r', p')
    
    where the map is a composition of two symplectic maps:
        Phi_dt = Phi_{dt/2}^B o Phi_{dt}^A o Phi_{dt/2}^B
    
    with:
        Phi^A: (r, p) -> (r + dt * p/m, p)
        Phi^B: (r, p) -> (r, p + dt * F(r))
    """
    
    def __init__(self, dt=TIME_STEP):
        self.dt = dt
        self.half_dt = 0.5 * dt
    
    def step(self, positions, velocities, forces, masses, box):
        """
        Perform one Velocity Verlet step.
        
        Args:
            positions: (N, 3) array
            velocities: (N, 3) array
            forces: (N, 3) array
            masses: (N,) array
            box: (3,) array
            
        Returns:
            new_positions, new_velocities
        """
        # v(t + dt/2) = v(t) + (dt/2) * F(t) / m
        inv_m = 1.0 / masses[:, np.newaxis]
        velocities_half = velocities + self.half_dt * forces * inv_m
        
        # r(t + dt) = r(t) + dt * v(t + dt/2)
        new_positions = positions + self.dt * velocities_half
        
        # Apply periodic boundary conditions
        new_positions -= box * np.round(new_positions / box)
        
        return new_positions, velocities_half
    
    def finalize_velocities(self, velocities_half, forces_new, masses):
        """
        Complete the velocity update after new forces are computed.
        
        Args:
            velocities_half: (N, 3) array at half step
            forces_new: (N, 3) array at new positions
            masses: (N,) array
            
        Returns:
            new_velocities
        """
        inv_m = 1.0 / masses[:, np.newaxis]
        new_velocities = velocities_half + self.half_dt * forces_new * inv_m
        return new_velocities


# =============================================================================
# Implicit Trapezoidal Integrator (from 833_ode_trapezoidal)
# =============================================================================

def implicit_trapezoidal_step(y, f_func, dt, max_iter=10, tol=1e-12):
    """
    Solve one step of the implicit trapezoidal method via fixed-point iteration.
    
    The implicit trapezoidal rule for dy/dt = f(t, y):
        y_{n+1} = y_n + (dt/2) * [ f(t_n, y_n) + f(t_{n+1}, y_{n+1}) ]
    
    Fixed-point iteration:
        z^{(0)} = y_n
        z^{(k+1)} = y_n + (dt/2) * [ f(t_n, y_n) + f(t_{n+1}, z^{(k)}) ]
    
    Convergence is guaranteed for sufficiently small dt if f is Lipschitz.
    
    Args:
        y: current state vector
        f_func: callable f(t, y) returning derivative
        dt: time step
        max_iter: maximum iterations
        tol: convergence tolerance
        
    Returns:
        y_new: updated state
        converged: whether iteration converged
    """
    t = 0.0  # local time (relative)
    f_n = f_func(t, y)
    
    z = y.copy()
    for _ in range(max_iter):
        z_new = y + 0.5 * dt * (f_n + f_func(t + dt, z))
        
        if np.linalg.norm(z_new - z) < tol * max(1.0, np.linalg.norm(z)):
            return z_new, True
        
        z = z_new
    
    return z, False


# =============================================================================
# ODE Dynamics for Interface Position (from 1025_ripple_ode)
# =============================================================================

def interface_ode_deriv(t, y, alpha=0.1, beta=1.0):
    """
    Nonlinear ODE for interface position dynamics.
    
    Inspired by the ripple ODE y' = sin(t*y), this models the interface
    position z_interface(t) with thermal fluctuations:
    
        dz/dt = alpha * sin(beta * t * z) + gamma * xi(t)
    
    where xi(t) is white noise and alpha, beta control the nonlinearity.
    
    The deterministic part has interesting properties:
        - For small alpha: nearly linear drift
        - For large alpha: chaotic oscillations
    
    Args:
        t: time
        y: interface position
        alpha: amplitude parameter
        beta: frequency parameter
        
    Returns:
        derivative
    """
    dydt = alpha * np.sin(beta * t * y[0])
    return np.array([dydt])


def solve_interface_dynamics(z0, t_span, dt, alpha=0.1, beta=1.0):
    """
    Solve the interface position ODE using explicit Euler with small steps.
    
    Args:
        z0: initial interface position
        t_span: (t_start, t_end)
        dt: time step
        alpha, beta: ODE parameters
        
    Returns:
        t_array, z_array
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    t_array = np.linspace(t_start, t_end, n_steps)
    z_array = np.zeros(n_steps)
    z_array[0] = z0
    
    y = np.array([z0])
    for i in range(1, n_steps):
        dydt = interface_ode_deriv(t_array[i - 1], y, alpha, beta)
        y = y + dt * dydt
        z_array[i] = y[0]
    
    return t_array, z_array


# =============================================================================
# Predator-Prey Like Species Dynamics (from 908_predator_prey_ode)
# =============================================================================

def species_ode_deriv(t, y, alpha_diss=0.01, beta_precip=0.005,
                      gamma_diff=0.1, delta_ads=0.02):
    """
    Coupled ODE system modeling species concentration dynamics at interface.
    
    Variables:
        y[0] = C_s: concentration of species B in solid
        y[1] = C_l: concentration of species B in liquid
    
    Equations (inspired by Lotka-Volterra):
        dC_s/dt = -alpha_diss * C_s + beta_precip * C_l + gamma_diff * (C_l - C_s)
        dC_l/dt = +alpha_diss * C_s - beta_precip * C_l - gamma_diff * (C_l - C_s)
                  + delta_ads * C_l * (1 - C_l / C_max)
    
    The last term is a logistic growth term for the liquid concentration,
    representing adsorption-limited enrichment.
    
    Args:
        t: time
        y: state vector [C_s, C_l]
        alpha_diss: dissolution rate
        beta_precip: precipitation rate
        gamma_diff: diffusion exchange rate
        delta_ads: adsorption rate
        
    Returns:
        dydt: derivative vector
    """
    C_s, C_l = y
    C_max = 1.0
    
    dCs = -alpha_diss * C_s + beta_precip * C_l + gamma_diff * (C_l - C_s)
    dCl = alpha_diss * C_s - beta_precip * C_l - gamma_diff * (C_l - C_s) \
          + delta_ads * C_l * (1.0 - C_l / C_max)
    
    return np.array([dCs, dCl])


def solve_species_dynamics(C_s0, C_l0, t_span, dt, **kwargs):
    """
    Solve the species concentration ODEs using implicit trapezoidal rule.
    
    Args:
        C_s0: initial solid concentration
        C_l0: initial liquid concentration
        t_span: (t_start, t_end)
        dt: time step
        **kwargs: parameters for species_ode_deriv
        
    Returns:
        t_array, Cs_array, Cl_array
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    t_array = np.linspace(t_start, t_end, n_steps)
    
    Cs = np.zeros(n_steps)
    Cl = np.zeros(n_steps)
    Cs[0] = C_s0
    Cl[0] = C_l0
    
    y = np.array([C_s0, C_l0])
    
    for i in range(1, n_steps):
        def f_local(t, y_local):
            return species_ode_deriv(t, y_local, **kwargs)
        
        y, converged = implicit_trapezoidal_step(y, f_local, dt, max_iter=10, tol=1e-10)
        
        # Boundary handling: enforce physical constraints
        y[0] = np.clip(y[0], 0.0, 1.0)
        y[1] = np.clip(y[1], 0.0, 1.0)
        
        Cs[i] = y[0]
        Cl[i] = y[1]
    
    return t_array, Cs, Cl
