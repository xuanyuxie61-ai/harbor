"""
particle_dynamics.py
====================
Dust particle trajectory and stochastic dynamics synthesized from seed projects:
  - 1049_rubber_band_ode (chaotic ODE with piecewise forcing)
  - 1039_rng_cliff (Cliff random number generator)

Core algorithms:
  - Cliff deterministic chaotic RNG for thermal noise initialization
  - Dust particle equations of motion with piecewise ion drag forcing
  - 4th-order Runge-Kutta (RK4) integration
  - Trajectory analysis for kinetic energy and mean square displacement
"""

import numpy as np
from yukawa_physics import yukawa_force_vector


def cliff_next(x):
    """
    Single-step Cliff random number generator.
    
    Based on seed 1039_rng_cliff (rng_cliff_next).
    Recurrence relation:
      x_{n+1} = (-100 * ln(x_n)) mod 1
    
    Domain: 0 < x < 1.
    Returns NaN if x is outside (0,1).
    """
    if x <= 0.0 or x >= 1.0:
        return np.nan
    return np.mod(-100.0 * np.log(x), 1.0)


def cliff_sequence(n, seed=0.5):
    """Generate n values from the Cliff RNG."""
    if seed <= 0.0 or seed >= 1.0:
        seed = 0.5
    seq = np.zeros(n, dtype=float)
    x = float(seed)
    for i in range(n):
        x = cliff_next(x)
        if np.isnan(x):
            x = 0.5
        seq[i] = x
    return seq


def dust_trajectory_deriv(t, y, params):
    """
    Compute time derivative for dust particle motion in a plasma sheath.
    
    Based on seed 1049_rubber_band_ode (piecewise forcing ODE).
    
    The equation of motion for particle i is:
      m_d * d^2 r_i / dt^2 = F_Yukawa + F_ion_drag + F_gravity + F_neutral_drag
    
    where:
      F_Yukawa    = sum_{j!=i} Q^2/(4*pi*eps0*r) * exp(-r/lD) * (r_i-r_j)/|r_i-r_j|
      F_ion_drag  = piecewise function of height z relative to equilibrium z_eq:
                    F_z = F_ion_base * 1.5  if z > z_eq  (above equilibrium, stronger)
                    F_z = F_ion_base * 0.5  if z <= z_eq (below equilibrium, weaker)
      F_gravity   = -m_d * g * z_hat
      F_neutral   = -nu_n * m_d * v  (Stokes-like neutral gas drag)
    
    State vector y = [r_1, ..., r_N, v_1, ..., v_N] with dimension 6N.
    """
    N = params['N']
    m_d = params['m_d']
    Q_eff = params['Q_eff']
    lambda_D = params['lambda_D']
    nu_n = params['nu_n']
    g = params['g']
    F_ion_base = params['F_ion_base']
    z_eq = params['z_eq']
    
    pos = y[:3*N].reshape((N, 3))
    vel = y[3*N:].reshape((N, 3))
    
    acc = np.zeros((N, 3), dtype=float)
    
    # Gravity (downward, negative z)
    acc[:, 2] -= g
    
    # Neutral gas drag: F = -nu_n * m_d * v  =>  a = -nu_n * v
    acc -= nu_n * vel
    
    # Inter-particle Yukawa forces
    for i in range(N):
        for j in range(i + 1, N):
            r_vec = pos[i] - pos[j]
            f = yukawa_force_vector(r_vec, Q_eff, lambda_D)
            acc[i] += f / m_d
            acc[j] -= f / m_d
    
    # Ion drag force: piecewise depending on height (sheath asymmetry)
    for i in range(N):
        z = pos[i, 2]
        if z > z_eq:
            acc[i, 2] += F_ion_base * 1.5 / m_d
        else:
            acc[i, 2] += F_ion_base * 0.5 / m_d
    
    dydt = np.zeros_like(y)
    dydt[:3*N] = vel.flatten()
    dydt[3*N:] = acc.flatten()
    return dydt


def rk4_step(f, t, y, dt, params):
    """Single step of 4th-order Runge-Kutta integration."""
    k1 = f(t, y, params)
    k2 = f(t + 0.5*dt, y + 0.5*dt*k1, params)
    k3 = f(t + 0.5*dt, y + 0.5*dt*k2, params)
    k4 = f(t + dt, y + dt*k3, params)
    return y + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


def integrate_trajectories(y0, t_end, dt, params):
    """
    Integrate dust particle trajectories using RK4.
    
    Returns the final state vector.
    """
    n_steps = max(1, int(t_end / dt))
    y = y0.copy()
    t = 0.0
    for _ in range(n_steps):
        y = rk4_step(dust_trajectory_deriv, t, y, dt, params)
        t += dt
    return y


def compute_mean_square_displacement(positions_t0, positions_t):
    """
    Compute mean square displacement:
      MSD = (1/N) * sum_i |r_i(t) - r_i(0)|^2
    """
    disp = positions_t - positions_t0
    return np.mean(np.sum(disp**2, axis=1))


def compute_kinetic_temperature(velocities, m_d, k_B=1.380649e-23):
    """
    Compute kinetic temperature from velocities:
      T_kin = (m_d / (3*k_B)) * <v^2>
    """
    v2 = np.mean(np.sum(velocities**2, axis=1))
    return m_d * v2 / (3.0 * k_B)
