
import numpy as np
import cmath



FARADAY = 96485.33212
GAS_CONSTANT = 8.314462618


def butler_volmer_current(eta, j0, alpha_a, alpha_c, n, T):
    if T <= 0:
        raise ValueError("温度 T 必须为正")
    if j0 < 0:
        raise ValueError("交换电流密度 j0 必须非负")
    
    rt_nf = GAS_CONSTANT * T / (n * FARADAY)
    

    arg_a = alpha_a * eta / rt_nf
    arg_c = -alpha_c * eta / rt_nf
    
    arg_a = np.clip(arg_a, -700, 700)
    arg_c = np.clip(arg_c, -700, 700)
    
    j = j0 * (np.exp(arg_a) - np.exp(arg_c))
    

    if abs(eta) < 1e-12:
        j = j0 * (alpha_a + alpha_c) * eta / rt_nf
    
    return j


def exchange_current_density(T, C_O2, j0_ref=1e-4, C_O2_ref=1.2, 
                             gamma=0.5, E_a=73200, T_ref=298.15):
    if T <= 0 or C_O2 <= 0:
        raise ValueError("T 和 C_O2 必须为正")
    
    conc_ratio = C_O2 / C_O2_ref
    if conc_ratio <= 0:
        raise ValueError("浓度比必须为正")
    
    thermal_factor = np.exp(-(E_a / GAS_CONSTANT) * (1.0 / T - 1.0 / T_ref))
    
    j0 = j0_ref * (conc_ratio ** gamma) * thermal_factor
    

    if not np.isfinite(j0) or j0 <= 0:
        j0 = j0_ref * 1e-6
    
    return j0


def solve_overpotential_muller(E, E_eq, R_ct, j0, alpha_a, alpha_c, n, T,
                                tol=1e-12, max_iter=100):
    rt_nf = GAS_CONSTANT * T / (n * FARADAY)
    E_cell = E - E_eq
    beta_a = alpha_a / rt_nf
    beta_c = alpha_c / rt_nf
    
    def safe_exp(x):
        return np.exp(np.clip(x, -350, 350))
    
    def f(eta):
        e_pos = safe_exp(beta_a * eta)
        e_neg = safe_exp(-beta_c * eta)
        return eta - E_cell + R_ct * j0 * (e_pos - e_neg)
    
    def df(eta):
        e_pos = safe_exp(beta_a * eta)
        e_neg = safe_exp(-beta_c * eta)
        return 1.0 + R_ct * j0 * (beta_a * e_pos + beta_c * e_neg)
    


    eta = float(E_cell)
    

    if abs(df(eta)) < 1e-12:
        eta = 0.0
    




    raise NotImplementedError("Hole_1: 请实现 solve_overpotential_muller 的迭代求解逻辑")



def solve_overpotential_wdk(E, E_eq, R_ct, j0, alpha_a, alpha_c, n, T,
                            tol=1e-12, max_iter=200):
    rt_nf = GAS_CONSTANT * T / (n * FARADAY)
    E_cell = E - E_eq
    



    
    coeffs = np.zeros(6, dtype=complex)
    coeffs[0] = -E_cell
    coeffs[1] = 1.0 + R_ct * j0 * (alpha_a + alpha_c) / rt_nf
    coeffs[2] = R_ct * j0 * (alpha_a**2 - alpha_c**2) / (2.0 * rt_nf**2)
    coeffs[3] = R_ct * j0 * (alpha_a**3 + alpha_c**3) / (6.0 * rt_nf**3)
    coeffs[4] = R_ct * j0 * (alpha_a**4 - alpha_c**4) / (24.0 * rt_nf**4)
    coeffs[5] = R_ct * j0 * (alpha_a**5 + alpha_c**5) / (120.0 * rt_nf**5)
    


    return solve_overpotential_muller(E, E_eq, R_ct, j0, alpha_a, alpha_c, n, T)


def orr_kinetic_parameters(T=353.15):
    params = {
        'alpha_a': 0.5,
        'alpha_c': 0.5,
        'n': 4,
        'j0_ref': 1e-4,
        'C_O2_ref': 1.2,
        'gamma': 0.5,
        'E_a': 73200,
        'T_ref': 298.15,
        'E_eq': 1.23 - 0.9e-3 * (T - 298.15),
        'R_ct': 1e-4,
    }
    return params


if __name__ == "__main__":
    p = orr_kinetic_parameters()
    eta_m = solve_overpotential_muller(0.7, p['E_eq'], p['R_ct'], p['j0_ref'],
                                       p['alpha_a'], p['alpha_c'], p['n'], 353.15)
    eta_w = solve_overpotential_wdk(0.7, p['E_eq'], p['R_ct'], p['j0_ref'],
                                    p['alpha_a'], p['alpha_c'], p['n'], 353.15)
    print(f"Muller 法过电位: {eta_m:.6f} V")
    print(f"WDK 法过电位: {eta_w:.6f} V")
