
import numpy as np
from yukawa_physics import yukawa_force_vector


def cliff_next(x):
    if x <= 0.0 or x >= 1.0:
        return np.nan
    return np.mod(-100.0 * np.log(x), 1.0)


def cliff_sequence(n, seed=0.5):
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
    

    acc[:, 2] -= g
    

    acc -= nu_n * vel
    

    for i in range(N):
        for j in range(i + 1, N):
            r_vec = pos[i] - pos[j]
            f = yukawa_force_vector(r_vec, Q_eff, lambda_D)
            acc[i] += f / m_d
            acc[j] -= f / m_d
    

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
    k1 = f(t, y, params)
    k2 = f(t + 0.5*dt, y + 0.5*dt*k1, params)
    k3 = f(t + 0.5*dt, y + 0.5*dt*k2, params)
    k4 = f(t + dt, y + dt*k3, params)
    return y + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


def integrate_trajectories(y0, t_end, dt, params):
    n_steps = max(1, int(t_end / dt))
    y = y0.copy()
    t = 0.0
    for _ in range(n_steps):
        y = rk4_step(dust_trajectory_deriv, t, y, dt, params)
        t += dt
    return y


def compute_mean_square_displacement(positions_t0, positions_t):
    disp = positions_t - positions_t0
    return np.mean(np.sum(disp**2, axis=1))


def compute_kinetic_temperature(velocities, m_d, k_B=1.380649e-23):
    v2 = np.mean(np.sum(velocities**2, axis=1))
    return m_d * v2 / (3.0 * k_B)
