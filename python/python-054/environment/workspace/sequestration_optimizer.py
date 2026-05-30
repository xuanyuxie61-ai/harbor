
import numpy as np






def golden_section_search(f, a, b, n_iterations=100, x_tol=1e-6):
    g = (np.sqrt(5.0) - 1.0) / 2.0
    
    x1 = g * a + (1.0 - g) * b
    x2 = (1.0 - g) * a + g * b
    f1 = f(x1)
    f2 = f(x2)
    
    for it in range(n_iterations):
        if abs(b - a) <= x_tol:
            break
        
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = g * a + (1.0 - g) * b
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = (1.0 - g) * a + g * b
            f2 = f(x2)
    
    x_opt = 0.5 * (a + b)
    f_opt = f(x_opt)
    
    return {
        'x_opt': x_opt,
        'f_opt': f_opt,
        'iterations': it + 1,
        'interval': (a, b),
    }






def hydrate_stability_depth(T_profile_func, P_profile_func, z):
    T = T_profile_func(z)
    P = P_profile_func(z)
    

    P_eq = 10.0**(1.847 + 0.0419 * T)
    
    stability = min(1.0, max(0.0, (P - P_eq) / (P_eq + 0.1)))
    return stability


def reexposure_time_scale(z, K_z=1e-4, w=0.0):
    if z <= 0:
        return 0.0
    
    tau_diff = z**2 / (2.0 * K_z)
    tau_adv = z / w if w > 1e-10 else np.inf
    
    tau = min(tau_diff, tau_adv)
    tau_years = tau / (86400.0 * 365.25)
    return tau_years


def acidification_impact(z, injection_rate, K_z=1e-4, dilution_factor=1.0):
    if z <= 0:
        return 1.0
    
    impact = injection_rate / (K_z**1.5 * z**2) * dilution_factor

    impact_norm = min(1.0, impact / (impact + 1e-3))
    return impact_norm


def sequestration_efficiency(z, T_profile_func, P_profile_func,
                              K_z=1e-4, w=0.0,
                              w1=0.4, w2=0.4, w3=0.2,
                              injection_rate=1.0, tau0=1000.0):
    if z <= 0 or z > 6000:
        return 1e6
    
    h = hydrate_stability_depth(T_profile_func, P_profile_func, z)
    tau = reexposure_time_scale(z, K_z, w)
    acid = acidification_impact(z, injection_rate, K_z)
    
    E = w1 * h + w2 * (1.0 - np.exp(-tau / tau0)) - w3 * acid
    return -E


def optimize_sequestration_depth(z_min=500, z_max=4000,
                                  T_profile_func=None, P_profile_func=None,
                                  K_z=1e-4, w=0.0, n_iter=100):
    if T_profile_func is None:
        def T_profile_func(z):

            return max(2.0, 20.0 * np.exp(-z / 200.0))
    
    if P_profile_func is None:
        def P_profile_func(z):

            rho = 1025.0
            g = 9.81
            return 0.1013 + rho * g * z * 1e-6
    
    def objective(z):
        return sequestration_efficiency(z, T_profile_func, P_profile_func, K_z, w)
    
    result = golden_section_search(objective, z_min, z_max, n_iterations=n_iter, x_tol=1.0)
    z_opt = result['x_opt']
    

    h_opt = hydrate_stability_depth(T_profile_func, P_profile_func, z_opt)
    tau_opt = reexposure_time_scale(z_opt, K_z, w)
    acid_opt = acidification_impact(z_opt, 1.0, K_z)
    
    return {
        'optimal_depth_m': z_opt,
        'efficiency_score': -result['f_opt'],
        'iterations': result['iterations'],
        'hydrate_stability': h_opt,
        'reexposure_time_years': tau_opt,
        'acidification_impact': acid_opt,
    }


def multi_scenario_optimization(scenarios):
    results = []
    for sc in scenarios:
        res = optimize_sequestration_depth(
            K_z=sc.get('K_z', 1e-4),
            w=sc.get('w', 0.0),
            n_iter=80
        )
        res['scenario_name'] = sc.get('name', 'unnamed')
        res['injection_rate'] = sc.get('injection_rate', 1.0)
        results.append(res)
    return results
