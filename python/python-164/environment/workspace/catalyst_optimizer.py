
import numpy as np


GOLDEN_RATIO = (np.sqrt(5.0) - 1.0) / 2.0


def golden_section_search(f, a, b, n_max=100, x_tol=1e-8):
    if a >= b:
        raise ValueError("必须满足 a < b")
    if n_max < 1:
        raise ValueError("n_max 必须 >= 1")
    if x_tol <= 0:
        raise ValueError("x_tol 必须为正")
    
    x1 = GOLDEN_RATIO * a + (1.0 - GOLDEN_RATIO) * b
    x2 = (1.0 - GOLDEN_RATIO) * a + GOLDEN_RATIO * b
    
    f1 = f(x1)
    f2 = f(x2)
    
    it = 0
    for it in range(1, n_max + 1):
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = GOLDEN_RATIO * a + (1.0 - GOLDEN_RATIO) * b
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = (1.0 - GOLDEN_RATIO) * a + GOLDEN_RATIO * b
            f2 = f(x2)
        
        if abs(b - a) <= x_tol:
            break
    
    x_opt = 0.5 * (a + b)
    f_opt = f(x_opt)
    
    return a, b, it, x_opt, f_opt


def power_performance(L_pt, P_max=1.0, k_eff=5.0, L_ref=0.1):
    if L_pt < 0:
        return 0.0
    
    exponent = -k_eff * L_pt / max(L_ref, 1e-10)
    exponent = np.clip(exponent, -50, 0)
    
    P = P_max * (1.0 - np.exp(exponent))
    return float(P)


def catalyst_cost(L_pt, area_active=250.0, price_pt=50.0):
    if L_pt < 0 or area_active < 0:
        return 0.0
    

    mass_mg = L_pt * area_active
    mass_g = mass_mg / 1000.0
    cost = mass_g * price_pt
    
    return float(cost)


def objective_function(L_pt, w_cost=0.3, w_power=0.5, w_penalty=0.2,
                        P_max=1.0, k_eff=5.0, L_ref=0.1,
                        area_active=250.0, price_pt=50.0, L_max=0.5):
    if L_pt < 0:
        return 1e10
    

    C_ref = catalyst_cost(0.4, area_active, price_pt)
    C_pt = catalyst_cost(L_pt, area_active, price_pt)
    cost_term = C_pt / max(C_ref, 1e-10)
    

    P_val = power_performance(L_pt, P_max, k_eff, L_ref)
    power_term = 1.0 - P_val / max(P_max, 1e-10)
    

    penalty_term = max(0.0, L_pt - L_max) ** 2 * 100.0
    
    J = w_cost * cost_term + w_power * power_term + w_penalty * penalty_term
    
    return float(J)


def optimize_catalyst_loading(w_cost=0.3, w_power=0.5, w_penalty=0.2,
                               L_min=0.02, L_max=0.8):
    def f(L):
        return objective_function(L, w_cost, w_power, w_penalty)
    
    a, b, it, L_opt, J_opt = golden_section_search(f, L_min, L_max, 
                                                     n_max=200, x_tol=1e-8)
    

    L_opt = np.clip(L_opt, L_min, L_max)
    
    info = {
        'interval': (a, b),
        'iterations': it,
        'power_at_opt': power_performance(L_opt),
        'cost_at_opt': catalyst_cost(L_opt),
        'objective_value': J_opt
    }
    
    return L_opt, J_opt, info


def sensitivity_analysis(L_opt, delta_L=0.01):
    J_plus = objective_function(L_opt + delta_L)
    J_minus = objective_function(L_opt - delta_L)
    
    sensitivity = (J_plus - J_minus) / (2.0 * delta_L)
    
    return sensitivity


if __name__ == "__main__":
    L_opt, J_opt, info = optimize_catalyst_loading()
    print(f"最优 Pt 负载: {L_opt:.4f} mg/cm^2")
    print(f"最优目标值: {J_opt:.6f}")
    print(f"迭代次数: {info['iterations']}")
    sens = sensitivity_analysis(L_opt)
    print(f"敏感性: {sens:.6f}")
