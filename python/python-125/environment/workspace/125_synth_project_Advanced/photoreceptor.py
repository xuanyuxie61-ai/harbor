
import numpy as np
from typing import Tuple






def solve_light_adaptation_steady_state(
    nx: int, ny: int,
    I_top: float, I_bottom: float, I_left: float, I_right: float,
    source_term: np.ndarray,
    epsilon: float = 1e-8,
    max_iter: int = 50000
) -> Tuple[np.ndarray, int, float]:

    C = np.zeros((nx, ny), dtype=np.float64)
    

    C[0, :] = I_top
    C[-1, :] = I_bottom
    C[:, 0] = I_left
    C[:, -1] = I_right
    
    C_new = C.copy()
    h2 = 1.0
    
    for iteration in range(1, max_iter + 1):

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                C_new[i, j] = 0.25 * (
                    C[i - 1, j] + C[i + 1, j] +
                    C[i, j - 1] + C[i, j + 1] +
                    h2 * source_term[i, j]
                )
        

        diff = np.max(np.abs(C_new - C))
        C, C_new = C_new, C
        
        if diff < epsilon:
            return C, iteration, diff
    
    return C, max_iter, diff






def clenshaw_curtis_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("n must be at least 2")
    

    j = np.arange(n, dtype=np.float64)
    x = np.cos(j * np.pi / (n - 1))
    

    theta = j * np.pi / (n - 1)
    
    c = np.ones(n, dtype=np.float64)
    c[0] = 0.5
    c[-1] = 0.5
    
    w = np.zeros(n, dtype=np.float64)
    
    half_nm1 = (n - 1) / 2.0
    
    for j_idx in range(n):
        sum_val = 0.0
        for k in range(1, int(np.floor(half_nm1)) + 1):
            if k < half_nm1:
                b_k = 1.0
            else:
                b_k = 0.5
            sum_val += b_k * np.cos(2.0 * k * theta[j_idx]) / (4.0 * k * k - 1.0)
        w[j_idx] = c[j_idx] / (n - 1.0) * (1.0 - sum_val)
    
    return x, w


def integrate_photocurrent_clenshaw_curtis(
    intensity_profile: callable,
    a: float, b: float,
    n: int = 64
) -> float:
    x_nodes, w = clenshaw_curtis_nodes_weights(n)
    

    scale = (b - a) / 2.0
    shift = (b + a) / 2.0
    t_nodes = scale * x_nodes + shift
    

    f_vals = np.array([intensity_profile(t) for t in t_nodes], dtype=np.float64)
    photocurrent = scale * np.sum(w * f_vals)
    
    return float(photocurrent)






def phototransduction_ode(
    t: float,
    y: np.ndarray,
    I_light: float,
    params: dict
) -> np.ndarray:
    PDE_star, cGMP, Ca = y[0], y[1], y[2]
    

    alpha = params.get('alpha_pde', 2.0)
    beta = params.get('beta_pde', 5.0)
    alpha_gc_max = params.get('alpha_gc_max', 10.0)
    K_gc = params.get('K_gc', 0.1)
    n_gc = params.get('n_gc', 4.0)
    gamma = params.get('gamma', 1.0)
    eta = params.get('eta', 1.0)
    g_max = params.get('g_max', 1.0)
    K_cGMP = params.get('K_cGMP', 0.05)
    V_m = params.get('V_m', -40.0)
    E_Ca = params.get('E_Ca', 40.0)
    

    PDE_star = max(PDE_star, 0.0)
    cGMP = max(cGMP, 1e-10)
    Ca = max(Ca, 0.0)
    

    dPDE = alpha * I_light - beta * PDE_star
    

    gc_activity = alpha_gc_max / (1.0 + (Ca / K_gc) ** n_gc)
    d_cGMP = gc_activity - gamma * PDE_star * cGMP
    

    I_Ca = g_max * (cGMP ** 3) / (cGMP ** 3 + K_cGMP ** 3) * (V_m - E_Ca)
    dCa = -eta * I_Ca
    
    return np.array([dPDE, d_cGMP, dCa], dtype=np.float64)


def solve_phototransduction_rk4(
    I_light_func: callable,
    y0: np.ndarray,
    t_span: Tuple[float, float],
    dt: float,
    params: dict
) -> Tuple[np.ndarray, np.ndarray]:
    t_start, t_end = t_span
    n_steps = int(np.ceil((t_end - t_start) / dt))
    dt = (t_end - t_start) / n_steps
    
    t_array = np.zeros(n_steps + 1, dtype=np.float64)
    y_array = np.zeros((n_steps + 1, 3), dtype=np.float64)
    
    t_array[0] = t_start
    y_array[0] = y0
    
    y = y0.copy()
    
    for n in range(n_steps):
        t = t_array[n]
        I_light = I_light_func(t)
        
        k1 = phototransduction_ode(t, y, I_light, params)
        k2 = phototransduction_ode(t + 0.5 * dt, y + 0.5 * dt * k1, I_light_func(t + 0.5 * dt), params)
        k3 = phototransduction_ode(t + 0.5 * dt, y + 0.5 * dt * k2, I_light_func(t + 0.5 * dt), params)
        k4 = phototransduction_ode(t + dt, y + dt * k3, I_light_func(t + dt), params)
        
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        

        y = np.maximum(y, 0.0)
        y[1] = max(y[1], 1e-10)
        
        t_array[n + 1] = t + dt
        y_array[n + 1] = y
    
    return t_array, y_array
