
import numpy as np
from typing import Callable, Tuple


def rk12_adaptive(yprime: Callable[[float, np.ndarray], np.ndarray],
                  tspan: Tuple[float, float],
                  y0: np.ndarray,
                  dt: float = 0.01,
                  tol: float = 1e-6,
                  max_steps: int = 100000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y0 = np.asarray(y0, dtype=float)
    m = len(y0)
    
    t_list = [tspan[0]]
    y_list = [y0.copy()]
    e_list = [0.0]
    
    t_current = tspan[0]
    y_current = y0.copy()
    step_count = 0
    
    while t_current < tspan[1] and step_count < max_steps:
        step_count += 1
        

        dt_actual = min(dt, tspan[1] - t_current)
        if dt_actual <= 0:
            break
        

        k1 = dt_actual * yprime(t_current, y_current)
        y1 = y_current + k1
        
        k2 = dt_actual * yprime(t_current + dt_actual, y_current + k1)
        y2 = y_current + 0.5 * k1 + 0.5 * k2
        
        error = np.linalg.norm(y2 - y1)
        

        if error > tol * dt_actual and dt_actual > 1e-14:

            dt = dt_actual / 2.0
            continue
        elif error < tol * dt_actual / 16.0 and dt_actual > 1e-14:

            dt = dt_actual * 2.0
        else:
            dt = dt_actual
        

        t_current += dt_actual
        y_current = y2.copy()
        
        t_list.append(t_current)
        y_list.append(y_current.copy())
        e_list.append(error)
    
    t = np.array(t_list)
    y = np.array(y_list)
    e = np.array(e_list)
    
    return t, y, e


def semiclassical_eom(state: np.ndarray, t: float,
                      hamiltonian_func: Callable,
                      berry_curvature_func: Callable,
                      e_field: np.ndarray = None,
                      b_field: np.ndarray = None,
                      hbar: float = 1.0) -> np.ndarray:
    if e_field is None:
        e_field = np.zeros(3)
    if b_field is None:
        b_field = np.zeros(3)
    
    r = state[:3]
    k = state[3:6]
    

    e_val, de_dk = hamiltonian_func(k)
    omega = berry_curvature_func(k)
    

    omega_vec = np.array([omega[1, 2], omega[2, 0], omega[0, 1]])
    

    dk_dt = -e_field / hbar
    



    cross = np.cross(dk_dt, omega_vec)
    dr_dt = de_dk / hbar - cross
    
    dstate_dt = np.concatenate([dr_dt, dk_dt])
    return dstate_dt


def periodic_lattice_dynamics(y: np.ndarray, force: float = 8.0) -> np.ndarray:
    n = len(y)
    if n < 4:
        raise ValueError("至少需要4个变量")
    
    dydt = np.zeros(n)
    for i in range(n):
        ip1 = (i + 1) % n
        im1 = (i - 1) % n
        im2 = (i - 2) % n
        dydt[i] = (y[ip1] - y[im2]) * y[im1] - y[i] + force
    
    return dydt


def evolve_berry_phase_along_path(ham: object,
                                   path_func: Callable[[float], np.ndarray],
                                   tspan: Tuple[float, float],
                                   band_index: int = 0,
                                   n_points: int = 200) -> float:
    from berry_curvature import berry_connection_numeric
    
    t_vals = np.linspace(tspan[0], tspan[1], n_points)
    dt = (tspan[1] - tspan[0]) / (n_points - 1)
    
    phase = 0.0
    for i in range(n_points - 1):
        t_mid = 0.5 * (t_vals[i] + t_vals[i + 1])
        k_mid = path_func(t_mid)
        

        A = berry_connection_numeric(ham, k_mid, band_index)
        

        k1 = path_func(t_vals[i])
        k2 = path_func(t_vals[i + 1])
        dk_dt = (k2 - k1) / dt
        

        phase += np.dot(A, dk_dt) * dt
    
    return phase
