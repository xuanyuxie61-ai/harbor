
import numpy as np
from config import (
    DIFFUSION_COEFFICIENT_SOLID, DIFFUSION_COEFFICIENT_LIQUID,
    INTERFACE_MOBILITY, DISSOLUTION_RATE, PRECIPITATION_RATE,
    TIME_STEP
)






def minmod(a, b):
    if a * b <= 0:
        return 0.0
    return np.sign(a) * min(abs(a), abs(b))


def advection_flux(c, v, dz):
    n = len(c)
    flux = np.zeros(n + 1)
    
    for i in range(1, n):

        if i > 1:
            slope_left = (c[i - 1] - c[i - 2]) if i >= 2 else 0.0
        else:
            slope_left = 0.0
        
        if i < n - 1:
            slope_right = (c[i + 1] - c[i]) if i < n - 1 else 0.0
        else:
            slope_right = 0.0
        
        slope_center = c[i] - c[i - 1]
        

        if v >= 0:
            delta = minmod(slope_center, slope_left)
            flux[i] = v * c[i - 1] + 0.5 * v * delta
        else:
            delta = minmod(slope_center, slope_right)
            flux[i] = v * c[i] - 0.5 * abs(v) * delta
    

    flux[0] = v * c[0]
    flux[n] = v * c[n - 1]
    
    return flux


def reaction_term(c_solid, c_liquid, rate_dissolve=DISSOLUTION_RATE,
                  rate_precipitate=PRECIPITATION_RATE):
    R_s = -rate_dissolve * c_solid + rate_precipitate * c_liquid
    R_l = rate_dissolve * c_solid - rate_precipitate * c_liquid
    return R_s, R_l


def solve_transport_1d(c_initial, z_grid, v_interface, D_solid, D_liquid,
                       z_interface, dt, n_steps, interface_width=3.0):
    n_z = len(z_grid)
    dz = z_grid[1] - z_grid[0]
    
    c = c_initial.copy()
    c_history = np.zeros((n_steps, n_z))
    
    for step in range(n_steps):
        c_history[step] = c.copy()
        

        phi = 0.5 * (1.0 + np.tanh((z_grid - z_interface) / interface_width))
        D_profile = D_solid + (D_liquid - D_solid) * phi
        

        flux_adv = advection_flux(c, v_interface, dz)
        

        dcdz = np.zeros(n_z)
        for i in range(1, n_z - 1):
            dcdz[i] = (c[i + 1] - c[i - 1]) / (2.0 * dz)
        
        d2cdz2 = np.zeros(n_z)
        for i in range(1, n_z - 1):
            D_plus = 0.5 * (D_profile[i + 1] + D_profile[i])
            D_minus = 0.5 * (D_profile[i] + D_profile[i - 1])
            d2cdz2[i] = (D_plus * (c[i + 1] - c[i]) - D_minus * (c[i] - c[i - 1])) / (dz ** 2)
        

        R = np.zeros(n_z)
        for i in range(n_z):
            if abs(z_grid[i] - z_interface) < 2.0 * interface_width:
                c_s = c[i]
                c_l = c[i]
                R[i] = reaction_term(c_s, c_l)[1]
        


        dc_dt = np.zeros(n_z)
        for i in range(n_z):
            if 0 < i < n_z - 1:
                dc_dt[i] = -(flux_adv[i + 1] - flux_adv[i]) / dz + d2cdz2[i] + R[i]
        

        max_D = max(D_solid, D_liquid)
        cfl_diff = max_D * dt / (dz ** 2)
        cfl_adv = abs(v_interface) * dt / dz
        

        if cfl_diff > 0.5 or cfl_adv > 1.0:
            dt_safe = min(0.25 * dz ** 2 / max_D, 0.5 * dz / max(abs(v_interface), 1e-10))

            n_sub = int(np.ceil(dt / dt_safe))
            dt_sub = dt / n_sub
            
            for _ in range(n_sub):

                flux_adv_sub = advection_flux(c, v_interface, dz)
                d2cdz2_sub = np.zeros(n_z)
                for j in range(1, n_z - 1):
                    D_plus = 0.5 * (D_profile[j + 1] + D_profile[j])
                    D_minus = 0.5 * (D_profile[j] + D_profile[j - 1])
                    d2cdz2_sub[j] = (D_plus * (c[j + 1] - c[j]) - D_minus * (c[j] - c[j - 1])) / (dz ** 2)
                
                dc_dt_sub = np.zeros(n_z)
                for j in range(n_z):
                    if 0 < j < n_z - 1:
                        dc_dt_sub[j] = -(flux_adv_sub[j + 1] - flux_adv_sub[j]) / dz + d2cdz2_sub[j] + R[j]
                
                c += dt_sub * dc_dt_sub
                c = np.clip(c, 0.0, 1.0)
        else:
            c += dt * dc_dt
        

        c[0] = c[1]
        c[-1] = c[-2]
        

        c = np.clip(c, 0.0, 1.0)
    
    return c, c_history






def interface_velocity(chemical_potential_solid, chemical_potential_liquid,
                       mobility=INTERFACE_MOBILITY):
    delta_mu = chemical_potential_liquid - chemical_potential_solid
    v = mobility * delta_mu
    

    v_max = 1.0
    v = np.clip(v, -v_max, v_max)
    
    return v






def solve_heat_equation_1d(T_initial, z_grid, alpha_thermal, dt, n_steps,
                           T_left=None, T_right=None):
    n_z = len(z_grid)
    dz = z_grid[1] - z_grid[0]
    
    T = T_initial.copy()
    T_history = np.zeros((n_steps, n_z))
    

    r = alpha_thermal * dt / (2.0 * dz ** 2)
    

    main_diag = np.ones(n_z) * (1.0 + 2.0 * r)
    off_diag = np.ones(n_z - 1) * (-r)
    

    if T_left is not None:
        main_diag[0] = 1.0
        off_diag[0] = 0.0
    else:
        main_diag[0] = 1.0 + r
    
    if T_right is not None:
        main_diag[-1] = 1.0
    else:
        main_diag[-1] = 1.0 + r
    

    for step in range(n_steps):
        T_history[step] = T.copy()
        

        rhs = np.zeros(n_z)
        for i in range(1, n_z - 1):
            rhs[i] = r * T[i - 1] + (1.0 - 2.0 * r) * T[i] + r * T[i + 1]
        
        if T_left is not None:
            rhs[0] = T_left
        else:
            rhs[0] = T[0] + r * (T[1] - T[0])
        
        if T_right is not None:
            rhs[-1] = T_right
        else:
            rhs[-1] = T[-1] + r * (T[-2] - T[-1])
        

        T = thomas_algorithm(main_diag, off_diag, off_diag, rhs)
    
    return T, T_history


def thomas_algorithm(a, b, c, d):
    n = len(a)
    cp = np.zeros(n - 1)
    dp = np.zeros(n)
    x = np.zeros(n)
    

    cp[0] = b[0] / a[0]
    dp[0] = d[0] / a[0]
    
    for i in range(1, n - 1):
        denom = a[i] - c[i - 1] * cp[i - 1]
        if abs(denom) < 1e-14:
            denom = 1e-14
        cp[i] = b[i] / denom
        dp[i] = (d[i] - c[i - 1] * dp[i - 1]) / denom
    
    denom = a[n - 1] - c[n - 2] * cp[n - 2]
    if abs(denom) < 1e-14:
        denom = 1e-14
    dp[n - 1] = (d[n - 1] - c[n - 2] * dp[n - 2]) / denom
    

    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    
    return x
