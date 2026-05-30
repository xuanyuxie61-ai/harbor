
import numpy as np






def euler_forward(dydt, t_span, y0, n_steps):
    t0, t_stop = t_span
    dt = (t_stop - t0) / n_steps
    m = len(y0)
    
    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1, m))
    
    t[0] = t0
    y[0, :] = y0
    
    for i in range(n_steps):
        y[i+1, :] = y[i, :] + dt * dydt(t[i], y[i, :])
        t[i+1] = t[i] + dt
    
    return t, y






def autocatalytic_carbonate_deriv(t, state, alpha=0.002, beta=0.08, gamma=0.5):
    w, x, y, z = state
    

    w = max(w, 0.0)
    x = max(x, 0.0)
    y = max(y, 0.0)
    z = max(z, 0.0)
    
    dwdt = -alpha * w
    dxdt = alpha * w - beta * x - x * y * y
    dydt = beta * x + x * y * y - gamma * y
    dzdt = gamma * y
    
    return np.array([dwdt, dxdt, dydt, dzdt])






def vertical_carbon_transport_model(
    z_grid, DIC_initial, T_profile, S_profile,
    dt_days=1.0, n_days=365, w=0.0, Kz=1e-4,
    mu_max=0.1, z_euphotic=50.0, K_half=10.0,
    pCO2_atm=410.0, u10=5.0,
    remin_rate=0.01, remin_depth_scale=1000.0
):
    nz = len(z_grid)
    dz = np.diff(z_grid)
    if not np.allclose(dz, dz[0]):
        raise ValueError("当前实现要求均匀深度网格")
    dz = dz[0]
    

    dt = dt_days * 86400.0
    n_steps = int(n_days / dt_days)
    
    DIC = DIC_initial.copy().astype(float)
    DIC_history = np.zeros((n_steps + 1, nz))
    DIC_history[0, :] = DIC
    t_history = np.zeros(n_steps + 1)
    

    from carbonate_chemistry import air_sea_co2_flux, solve_carbonate_system
    
    for step in range(n_steps):









        raise NotImplementedError("HOLE 2: 上边界海-气通量计算与单位转换待补全")
        
        DIC_new = DIC.copy()
        

        for i in range(1, nz - 1):

            adv = -w * (DIC[i+1] - DIC[i-1]) / (2.0 * abs(dz))

            diff = Kz * (DIC[i+1] - 2.0 * DIC[i] + DIC[i-1]) / (dz**2)

            z_depth = abs(z_grid[i])
            J_bio = -mu_max * np.exp(-z_depth / z_euphotic) * DIC[i] / (DIC[i] + K_half)

            J_remin = remin_rate * np.exp(-z_depth / remin_depth_scale) * (2000.0 - DIC[i])
            
            dDICdt = adv + diff + J_bio + J_remin
            DIC_new[i] = DIC[i] + dt_days * dDICdt
        

        z_depth = abs(z_grid[0])
        J_bio0 = -mu_max * np.exp(-z_depth / z_euphotic) * DIC[0] / (DIC[0] + K_half)
        J_remin0 = remin_rate * np.exp(-z_depth / remin_depth_scale) * (2000.0 - DIC[0])
        adv0 = -w * (DIC[1] - DIC[0]) / abs(dz) if nz > 1 else 0.0
        diff0 = Kz * (DIC[1] - DIC[0]) / (dz**2) if nz > 1 else 0.0
        DIC_new[0] = DIC[0] + dt_days * (adv0 + diff0 + J_bio0 + J_remin0 + flux_top_conc)
        

        DIC_new[-1] = DIC[-1]
        if nz > 1:
            adv_bot = -w * (DIC[-1] - DIC[-2]) / abs(dz)
            diff_bot = Kz * (DIC[-2] - DIC[-1]) / (dz**2)
            J_bio_bot = -mu_max * np.exp(-abs(z_grid[-1]) / z_euphotic) * DIC[-1] / (DIC[-1] + K_half)
            J_remin_bot = remin_rate * np.exp(-abs(z_grid[-1]) / remin_depth_scale) * (2000.0 - DIC[-1])
            DIC_new[-1] = DIC[-1] + dt_days * (adv_bot + diff_bot + J_bio_bot + J_remin_bot)
        

        DIC_new = np.maximum(DIC_new, 0.0)
        DIC = DIC_new
        DIC_history[step + 1, :] = DIC
        t_history[step + 1] = (step + 1) * dt_days
    
    return DIC_history, t_history






def box_carbon_cycle_model(t_span, y0, n_steps, 
                           k12=0.1, k21=0.05, k23=0.02, k32=0.01,
                           F_anthro=8.0, buffer_factor=10.0):
    N2_0 = y0[1]
    
    def dydt(t, y):
        N1, N2, N3 = y

        N1 = max(N1, 0.0)
        N2 = max(N2, 0.0)
        N3 = max(N3, 0.0)
        
        dN1 = -k12 * N1 + k21 * N2 + F_anthro
        dN2 = k12 * N1 - k21 * N2 - k23 * N2 + k32 * N3 - (N2 - N2_0) / buffer_factor
        dN3 = k23 * N2 - k32 * N3
        return np.array([dN1, dN2, dN3])
    
    return euler_forward(dydt, t_span, y0, n_steps)


def compute_anthropogenic_carbon_inventory(DIC_pre, DIC_post, rho, thickness):
    delta_DIC = np.array(DIC_post) - np.array(DIC_pre)
    if np.isscalar(rho):
        rho = np.full_like(delta_DIC, float(rho))
    if np.isscalar(thickness):
        thickness = np.full_like(delta_DIC, float(thickness))
    

    inventory = np.sum(delta_DIC * rho * thickness * 1e-6)
    return inventory
