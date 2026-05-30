
import numpy as np
from config import TIME_STEP, MASS_A, MASS_B






class VelocityVerlet:
    
    def __init__(self, dt=TIME_STEP):
        self.dt = dt
        self.half_dt = 0.5 * dt
    
    def step(self, positions, velocities, forces, masses, box):

        inv_m = 1.0 / masses[:, np.newaxis]
        velocities_half = velocities + self.half_dt * forces * inv_m
        

        new_positions = positions + self.dt * velocities_half
        

        new_positions -= box * np.round(new_positions / box)
        
        return new_positions, velocities_half
    
    def finalize_velocities(self, velocities_half, forces_new, masses):
        inv_m = 1.0 / masses[:, np.newaxis]
        new_velocities = velocities_half + self.half_dt * forces_new * inv_m
        return new_velocities






def implicit_trapezoidal_step(y, f_func, dt, max_iter=10, tol=1e-12):
    t = 0.0
    f_n = f_func(t, y)
    
    z = y.copy()
    for _ in range(max_iter):
        z_new = y + 0.5 * dt * (f_n + f_func(t + dt, z))
        
        if np.linalg.norm(z_new - z) < tol * max(1.0, np.linalg.norm(z)):
            return z_new, True
        
        z = z_new
    
    return z, False






def interface_ode_deriv(t, y, alpha=0.1, beta=1.0):
    dydt = alpha * np.sin(beta * t * y[0])
    return np.array([dydt])


def solve_interface_dynamics(z0, t_span, dt, alpha=0.1, beta=1.0):
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






def species_ode_deriv(t, y, alpha_diss=0.01, beta_precip=0.005,
                      gamma_diff=0.1, delta_ads=0.02):
    C_s, C_l = y
    C_max = 1.0
    
    dCs = -alpha_diss * C_s + beta_precip * C_l + gamma_diff * (C_l - C_s)
    dCl = alpha_diss * C_s - beta_precip * C_l - gamma_diff * (C_l - C_s) \
          + delta_ads * C_l * (1.0 - C_l / C_max)
    
    return np.array([dCs, dCl])


def solve_species_dynamics(C_s0, C_l0, t_span, dt, **kwargs):
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
        

        y[0] = np.clip(y[0], 0.0, 1.0)
        y[1] = np.clip(y[1], 0.0, 1.0)
        
        Cs[i] = y[0]
        Cl[i] = y[1]
    
    return t_array, Cs, Cl
