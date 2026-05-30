
import numpy as np



RHO_ROCK = 3000.0
CP_ROCK = 1000.0
K_THERMAL = 3.0
ALPHA_THERMAL = 3e-5
H_RADIOGENIC = 1e-6


def jacobi_thermal_2d(nx, ny, dx, dy, k_field, H_field, T_boundary,
                       epsilon=1e-8, max_iter=50000):
    if nx < 3 or ny < 3:
        raise ValueError("Grid size must be at least 3x3")
    

    T = np.ones((nx, ny), dtype=float) * 300.0
    

    if isinstance(T_boundary, dict):
        if 'top' in T_boundary:
            T[0, :] = T_boundary['top']
        if 'bottom' in T_boundary:
            T[-1, :] = T_boundary['bottom']
        if 'left' in T_boundary:
            T[:, 0] = T_boundary['left']
        if 'right' in T_boundary:
            T[:, -1] = T_boundary['right']
    else:
        T[0, :] = T_boundary
        T[-1, :] = T_boundary
        T[:, 0] = T_boundary
        T[:, -1] = T_boundary
    
    T_new = T.copy()
    iterations = 0
    diff = epsilon + 1.0
    
    while diff >= epsilon and iterations < max_iter:
        T_old = T.copy()
        
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):

                ke = 2.0 * k_field[i, j] * k_field[i + 1, j] / (k_field[i, j] + k_field[i + 1, j] + 1e-12)
                kw = 2.0 * k_field[i, j] * k_field[i - 1, j] / (k_field[i, j] + k_field[i - 1, j] + 1e-12)
                kn = 2.0 * k_field[i, j] * k_field[i, j + 1] / (k_field[i, j] + k_field[i, j + 1] + 1e-12)
                ks = 2.0 * k_field[i, j] * k_field[i, j - 1] / (k_field[i, j] + k_field[i, j - 1] + 1e-12)
                



                k_eq = (ke + kw + kn + ks) / 4.0
                source_term = H_field[i, j] * dx * dx / (k_eq + 1e-30)
                
                numerator = ke * T_old[i + 1, j] + kw * T_old[i - 1, j] + \
                            kn * T_old[i, j + 1] + ks * T_old[i, j - 1]
                denominator = ke + kw + kn + ks
                
                if denominator > 1e-15:
                    T_new[i, j] = numerator / denominator + source_term
                else:
                    T_new[i, j] = T_old[i, j]
        
        T = T_new.copy()
        diff = np.max(np.abs(T - T_old))
        iterations += 1
    
    return T, iterations


def adi_thermal_2d(nx, ny, dx, dy, k, H, T_boundary, dt, t_max,
                    rho=3000.0, cp=1000.0):
    if nx < 3 or ny < 3:
        raise ValueError("Grid too small")
    
    n_steps = int(t_max / dt)
    if n_steps < 1:
        n_steps = 1
        dt = t_max
    
    T = np.ones((nx, ny), dtype=float) * 300.0
    

    if isinstance(T_boundary, dict):
        if 'top' in T_boundary: T[0, :] = T_boundary['top']
        if 'bottom' in T_boundary: T[-1, :] = T_boundary['bottom']
        if 'left' in T_boundary: T[:, 0] = T_boundary['left']
        if 'right' in T_boundary: T[:, -1] = T_boundary['right']
    else:
        T[0, :] = T_boundary
        T[-1, :] = T_boundary
        T[:, 0] = T_boundary
        T[:, -1] = T_boundary
    
    k = float(k)
    kappa = k / (rho * cp)
    rx = kappa * dt / (2.0 * dx**2)
    ry = kappa * dt / (2.0 * dy**2)
    
    T_history = []
    
    for step in range(n_steps):
        T_half = T.copy()
        

        for j in range(1, ny - 1):

            a = np.zeros(nx - 2)
            b = np.zeros(nx - 2)
            c = np.zeros(nx - 2)
            d = np.zeros(nx - 2)
            
            for i in range(1, nx - 1):
                idx = i - 1
                a[idx] = -rx
                b[idx] = 1.0 + 2.0 * rx
                c[idx] = -rx
                d[idx] = T[i, j] + ry * (T[i, j + 1] - 2.0 * T[i, j] + T[i, j - 1]) + (H * dt / 2.0) / (rho * cp)
            

            sol = _thomas_algorithm(a, b, c, d)
            T_half[1:nx - 1, j] = sol
        

        if isinstance(T_boundary, dict):
            if 'top' in T_boundary: T_half[0, :] = T_boundary['top']
            if 'bottom' in T_boundary: T_half[-1, :] = T_boundary['bottom']
            if 'left' in T_boundary: T_half[:, 0] = T_boundary['left']
            if 'right' in T_boundary: T_half[:, -1] = T_boundary['right']
        

        for i in range(1, nx - 1):
            a = np.zeros(ny - 2)
            b = np.zeros(ny - 2)
            c = np.zeros(ny - 2)
            d = np.zeros(ny - 2)
            
            for j in range(1, ny - 1):
                idx = j - 1
                a[idx] = -ry
                b[idx] = 1.0 + 2.0 * ry
                c[idx] = -ry
                d[idx] = T_half[i, j] + rx * (T_half[i + 1, j] - 2.0 * T_half[i, j] + T_half[i - 1, j]) + (H * dt / 2.0) / (rho * cp)
            
            sol = _thomas_algorithm(a, b, c, d)
            T[i, 1:ny - 1] = sol
        

        if isinstance(T_boundary, dict):
            if 'top' in T_boundary: T[0, :] = T_boundary['top']
            if 'bottom' in T_boundary: T[-1, :] = T_boundary['bottom']
            if 'left' in T_boundary: T[:, 0] = T_boundary['left']
            if 'right' in T_boundary: T[:, -1] = T_boundary['right']
        
        if step % max(1, n_steps // 20) == 0:
            T_history.append(T.copy())
    
    return T, T_history


def _thomas_algorithm(a, b, c, d):
    n = len(d)
    if n == 0:
        return np.array([])
    
    cp = np.zeros(n - 1)
    dp = np.zeros(n)
    
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if abs(denom) < 1e-15:
            denom = 1e-15
        if i < n - 1:
            cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
    
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    
    return x


def stiff_thermal_decay(tspan, T0, lambda_stiff, T_ambient, n_steps=1000):
    t0, t1 = tspan
    dt = (t1 - t0) / n_steps
    t = np.linspace(t0, t1, n_steps + 1)
    T = np.zeros(n_steps + 1)
    T[0] = T0
    
    for i in range(n_steps):

        T[i + 1] = (T[i] + dt * lambda_stiff * T_ambient) / (1.0 + dt * lambda_stiff)
    
    T_exact = T_ambient + (T0 - T_ambient) * np.exp(-lambda_stiff * (t - t0))
    return t, T, T_exact


def euler_explicit_thermal(tspan, T0, source_func, n_steps=1000):
    t0, t1 = tspan
    dt = (t1 - t0) / n_steps
    t = np.linspace(t0, t1, n_steps + 1)
    
    T0 = np.asarray(T0, dtype=float)
    if T0.ndim == 0:
        T = np.zeros(n_steps + 1)
        T[0] = T0
        for i in range(n_steps):
            dT = source_func(t[i], T[i])
            T[i + 1] = T[i] + dt * float(dT)
    else:
        m = len(T0)
        T = np.zeros((n_steps + 1, m))
        T[0, :] = T0
        for i in range(n_steps):
            dT = source_func(t[i], T[i, :])
            dT = np.asarray(dT, dtype=float).flatten()
            T[i + 1, :] = T[i, :] + dt * dT
    
    return t, T


def thermal_expansion_density(rho0, T, T0, alpha=ALPHA_THERMAL):
    T = np.asarray(T, dtype=float)
    dT = T - T0
    rho = rho0 * (1.0 - alpha * dT)

    rho = np.maximum(rho, 100.0)
    return rho


def coupled_thermal_density_evolution(nx, ny, dx, dy, rho0, T0, T_boundary,
                                       dt, n_steps, k=K_THERMAL, H=H_RADIOGENIC,
                                       alpha=ALPHA_THERMAL, rho_ref=3000.0, cp=CP_ROCK):
    T = T0.copy()
    rho = rho0.copy()
    history = []
    
    for step in range(n_steps):

        T_new, _ = adi_thermal_2d(nx, ny, dx, dy, k, H, T_boundary, dt, dt)
        T = T_new
        

        rho = thermal_expansion_density(rho_ref, T, 300.0, alpha)
        
        if step % max(1, n_steps // 10) == 0:
            history.append((T.copy(), rho.copy()))
    
    return T, rho, history
