
import numpy as np



GAS_CONSTANT = 8.314462618


def kelvin_solubility(r, gamma, V_m, T, C_sat_inf):
    if r <= 0:
        raise ValueError("半径 r 必须为正")
    if T <= 0:
        raise ValueError("温度 T 必须为正")
    
    exponent = (2.0 * gamma * V_m) / (r * GAS_CONSTANT * T)

    exponent = np.clip(exponent, -50, 50)
    
    return C_sat_inf * np.exp(exponent)


def critical_radius(gamma, V_m, T, C_bulk, C_sat_inf):
    if C_bulk <= 0 or C_sat_inf <= 0:
        return 1e-8
    
    ratio = C_bulk / C_sat_inf
    if ratio <= 1.0:
        return 1e-8
    
    ln_ratio = np.log(ratio)
    
    if abs(ln_ratio) < 1e-12:
        return 1e20
    
    rc = (2.0 * gamma * V_m) / (GAS_CONSTANT * T * ln_ratio)
    return max(rc, 1e-10)


def ripening_rate(r, D, V_m, C_sat_inf, C_bulk, gamma, T):
    if r <= 0:
        return 0.0
    
    C_sat_r = kelvin_solubility(r, gamma, V_m, T, C_sat_inf)
    rate = (D * V_m / r) * (C_bulk - C_sat_r)
    


    max_rate = 1e-9
    rate = np.clip(rate, -max_rate, max_rate)
    
    return rate


def evolve_size_distribution(radii, D, V_m, C_sat_inf, C_bulk, gamma, T, dt, n_steps):
    if dt <= 0 or n_steps < 0:
        raise ValueError("dt>0, n_steps>=0")
    

    C_bulk_safe = max(float(C_bulk), 1.5 * float(C_sat_inf))
    if C_bulk_safe <= 0 or C_sat_inf <= 0:

        N = len(radii)
        radii_history = np.zeros((n_steps + 1, N))
        for s in range(n_steps + 1):
            radii_history[s, :] = np.clip(radii, 0.5e-9, 1e-6)
        return radii_history
    
    N = len(radii)
    radii_history = np.zeros((n_steps + 1, N))
    radii_history[0, :] = np.clip(radii, 0.5e-9, 1e-6)
    
    for step in range(n_steps):
        r_current = radii_history[step, :]
        
        rates = np.array([ripening_rate(float(ri), float(D), float(V_m), 
                                         float(C_sat_inf), C_bulk_safe, 
                                         float(gamma), float(T)) 
                         for ri in r_current])
        

        rates = np.array([r if np.isfinite(r) else 0.0 for r in rates])
        
        r_new = r_current + rates * dt
        

        r_new = np.clip(r_new, 0.5e-9, 1e-6)
        
        radii_history[step + 1, :] = r_new
    
    return radii_history


def lsw_analytical_r3(t, r0, gamma, D, V_m, C_sat_inf, T):
    K_LSW = (8.0 * gamma * D * V_m ** 2 * C_sat_inf) / (9.0 * GAS_CONSTANT * T)
    r0_cubed = r0 ** 3
    return (r0_cubed + K_LSW * t) ** (1.0 / 3.0)


def disk_distance_stats_monte_carlo(radii1, radii2, n_samples=1000):
    if len(radii1) == 0 or len(radii2) == 0:
        return 0.0, 0.0
    
    distances = np.zeros(n_samples)
    
    for i in range(n_samples):

        theta1 = 2.0 * np.pi * np.random.random()
        theta2 = 2.0 * np.pi * np.random.random()
        

        rad1 = np.sqrt(np.random.random())
        rad2 = np.sqrt(np.random.random())
        
        p1 = np.array([rad1 * np.cos(theta1), rad1 * np.sin(theta1)])
        p2 = np.array([rad2 * np.cos(theta2), rad2 * np.sin(theta2)])
        
        distances[i] = np.linalg.norm(p1 - p2)
    
    mean_dist = np.mean(distances)
    if n_samples > 1:
        var_dist = np.sum((distances - mean_dist) ** 2) / (n_samples - 1)
    else:
        var_dist = 0.0
    
    return mean_dist, var_dist


def gauss_legendre_integral_exactness(f, order, w, x, a=-1, b=1):
    if len(w) != len(x):
        raise ValueError("权重 w 和节点 x 长度必须一致")
    

    t = 0.5 * (b - a) * x + 0.5 * (a + b)
    jac = 0.5 * (b - a)
    
    integral = jac * np.sum(w * f(t))
    return integral


def moment_size_distribution(radii, w=None, k=1):
    if len(radii) == 0:
        return 0.0
    
    if w is None:
        w = np.ones(len(radii))
    
    w = np.array(w)
    radii = np.array(radii)
    
    total_w = np.sum(w)
    if total_w < 1e-30:
        return 0.0
    
    moment = np.sum(w * (radii ** k)) / total_w
    return moment


def pt_dissolution_parameters():
    params = {
        'gamma': 2.5,
        'V_m': 9.09e-6,
        'D_Pt2': 1e-12,
        'C_sat_inf': 1e-6,
        'T': 353.15,
        'rho_Pt': 21450,
    }
    return params


if __name__ == "__main__":
    p = pt_dissolution_parameters()
    r0 = np.array([2e-9, 3e-9, 4e-9, 5e-9, 6e-9])
    hist = evolve_size_distribution(r0, p['D_Pt2'], p['V_m'], p['C_sat_inf'],
                                     2e-6, p['gamma'], p['T'], dt=3600, n_steps=24)
    print(f"24h后平均半径: {np.mean(hist[-1])*1e9:.2f} nm")
    mu, var = disk_distance_stats_monte_carlo(r0, r0)
    print(f"颗粒间距统计: mean={mu:.4f}, var={var:.6f}")
