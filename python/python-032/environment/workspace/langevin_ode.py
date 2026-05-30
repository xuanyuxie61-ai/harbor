
import numpy as np
from typing import Tuple, Callable, Optional


def euler_maruyama_step(
    x: np.ndarray,
    drift: np.ndarray,
    diffusion_sqrt: np.ndarray,
    dt: float,
) -> np.ndarray:
    ndim = len(x)
    dW = np.random.normal(0.0, np.sqrt(dt), size=ndim)
    x_new = x + drift * dt + diffusion_sqrt @ dW
    return x_new


def langevin_dynamics_1d(
    V_func: Callable[[float], float],
    gamma: float,
    T: float,
    x0: float,
    t_max: float,
    dt: float,
    x_bounds: Tuple[float, float] = (-2.0, 2.0),
) -> Tuple[np.ndarray, np.ndarray]:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if T < 0:
        T = 0.0
    if dt <= 0:
        raise ValueError("dt must be positive")
    
    n_steps = int(t_max / dt) + 1
    t_arr = np.linspace(0.0, t_max, n_steps)
    x_arr = np.zeros(n_steps)
    x_arr[0] = x0
    
    h = 1e-5
    mu = 1.0 / gamma
    diff_coeff = np.sqrt(2.0 * T * mu)
    
    x_min, x_max = x_bounds
    
    for n in range(n_steps - 1):
        xn = x_arr[n]

        dVdx = (V_func(xn + h) - V_func(xn - h)) / (2.0 * h)
        drift = -mu * dVdx

        dW = np.random.normal(0.0, np.sqrt(dt))
        x_new = xn + drift * dt + diff_coeff * dW
        

        if x_new < x_min:
            x_new = 2.0 * x_min - x_new
        elif x_new > x_max:
            x_new = 2.0 * x_max - x_new
        
        if not np.isfinite(x_new):
            x_new = xn
        
        x_arr[n + 1] = x_new
    
    return t_arr, x_arr


def langevin_ensemble_mass_distribution(
    V_func: Callable[[float], float],
    gamma: float,
    T: float,
    x0: float,
    t_max: float,
    dt: float,
    n_trajectories: int,
    mass_number: int,
    n_bins: int = 80,
    x_bounds: Tuple[float, float] = (-1.5, 1.5),
) -> Tuple[np.ndarray, np.ndarray]:
    from collective_coordinates import mass_asymmetry_to_fragment_mass
    
    final_masses = np.zeros(n_trajectories)
    
    for i in range(n_trajectories):
        t_arr, x_arr = langevin_dynamics_1d(
            V_func, gamma, T, x0, t_max, dt, x_bounds
        )

        beta3_final = x_arr[-1]
        A_L, A_H = mass_asymmetry_to_fragment_mass(beta3_final, mass_number)

        if np.random.rand() < 0.5:
            final_masses[i] = A_L
        else:
            final_masses[i] = A_H
    







    raise NotImplementedError("Hole_3: Langevin 质量分布直方图构建待修复")
    counts = np.zeros(n_bins)
    mass_centers = np.zeros(n_bins)
    return mass_centers, counts


def robertson_like_stiff_test(t: float, y: np.ndarray) -> np.ndarray:
    y1, y2, y3 = y[0], y[1], y[2]
    lam1 = 0.04
    lam2 = 1e4
    lam3 = 3e7
    dydt = np.zeros(3)
    dydt[0] = -lam1 * y1 + lam2 * y2 * y3
    dydt[1] = lam1 * y1 - lam2 * y2 * y3 - lam3 * y2 * y2
    dydt[2] = lam3 * y2 * y2
    return dydt


def rk4_step(f: Callable, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def fission_decay_dynamics(
    lambda_fission: float,
    lambda_neutron: float,
    lambda_gamma: float,
    t_max: float,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_steps = int(t_max / dt) + 1
    t_arr = np.linspace(0.0, t_max, n_steps)
    N_c = np.zeros(n_steps)
    N_f = np.zeros(n_steps)
    N_n = np.zeros(n_steps)
    N_g = np.zeros(n_steps)
    
    N_c[0] = 1.0
    lam_total = lambda_fission + lambda_neutron + lambda_gamma
    
    for i in range(n_steps - 1):
        decay = lam_total * N_c[i]
        N_c[i + 1] = N_c[i] - decay * dt
        N_f[i + 1] = N_f[i] + lambda_fission * N_c[i] * dt
        N_n[i + 1] = N_n[i] + lambda_neutron * N_c[i] * dt
        N_g[i + 1] = N_g[i] + lambda_gamma * N_c[i] * dt
    
    return t_arr, N_c, N_f, N_n, N_g
