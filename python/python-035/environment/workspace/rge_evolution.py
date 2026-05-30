import numpy as np
from constants import G_F, TINY
from utils import rk2_integrate




def sm_rge_beta(t, y):
    g1, g2, g3, yt, yb, ytau, lam = y
    

    g1 = max(g1, 0.0)
    g2 = max(g2, 0.0)
    g3 = max(g3, 0.0)
    yt = max(yt, 0.0)
    yb = max(yb, 0.0)
    ytau = max(ytau, 0.0)
    
    factor = 1.0 / (16.0 * np.pi ** 2)
    
    beta_g1 = (41.0 / 10.0) * g1 ** 3
    beta_g2 = (-19.0 / 6.0) * g2 ** 3
    beta_g3 = (-7.0) * g3 ** 3
    
    beta_yt = yt * (4.5 * yt ** 2 + yb ** 2 - 0.85 * g1 ** 2 - 2.25 * g2 ** 2 - 8.0 * g3 ** 2)
    beta_yb = yb * (yt ** 2 + 4.5 * yb ** 2 + ytau ** 2 - 0.25 * g1 ** 2 - 2.25 * g2 ** 2 - 8.0 * g3 ** 2)
    beta_ytau = ytau * (3.0 * yb ** 2 + 2.5 * ytau ** 2 - 2.25 * g1 ** 2 - 2.25 * g2 ** 2)
    
    beta_lambda = (6.0 * lam ** 2 
                   + 2.0 * lam * (yt ** 2 + yb ** 2 + ytau ** 2)
                   - (yt ** 4 + yb ** 4 + ytau ** 4)
                   + 0.375 * (2.0 * g2 ** 4 + (g1 ** 2 + g2 ** 2) ** 2)
                   - 3.0 * lam * (3.0 * g2 ** 2 + g1 ** 2))
    
    return np.array([
        beta_g1 * factor,
        beta_g2 * factor,
        beta_g3 * factor,
        beta_yt * factor,
        beta_yb * factor,
        beta_ytau * factor,
        beta_lambda * factor
    ])





def sm_initial_conditions():
    alpha_em = 1.0 / 127.9
    sin2w = 0.23121
    e = np.sqrt(4.0 * np.pi * alpha_em)
    
    g1 = np.sqrt(5.0 / 3.0) * e / np.sqrt(1.0 - sin2w)
    g2 = e / np.sqrt(sin2w)
    g3 = np.sqrt(4.0 * np.pi * 0.118)
    
    v = 1.0 / np.sqrt(np.sqrt(2.0) * G_F)
    m_t = 173.1
    m_b = 4.18
    m_tau = 1.777
    m_h = 125.1
    
    yt = np.sqrt(2.0) * m_t / v
    yb = np.sqrt(2.0) * m_b / v
    ytau = np.sqrt(2.0) * m_tau / v
    lam = m_h ** 2 / (2.0 * v ** 2)
    
    return np.array([g1, g2, g3, yt, yb, ytau, lam])





def effective_potential(y):
    lam = max(y[6], TINY)
    m_h = 125.1
    v_sq = m_h ** 2 / lam
    return 0.25 * lam * v_sq ** 2


def check_rge_stability(t_array, y_array):
    issues = []
    for i, y in enumerate(y_array):
        if np.any(y[:3] < 0):
            issues.append((i, t_array[i], "negative gauge coupling"))
        if y[3] > 2.0:
            issues.append((i, t_array[i], "Yukawa Landau pole warning"))
    
    v_eff = [effective_potential(y) for y in y_array]

    v_diff = np.diff(v_eff)
    if np.any(np.abs(v_diff) > 1.0e6):
        issues.append((-1, -1, "effective potential unstable"))
    
    return len(issues) == 0, issues





def higgs_vv_coupling_evolution(mu_values):
    t_span = (0.0, np.log(np.max(mu_values) / 91.1876))
    y0 = sm_initial_conditions()
    n_steps = max(500, int(t_span[1] * 100))
    
    t_arr, y_arr = rk2_integrate(sm_rge_beta, t_span, y0, n_steps)
    

    log_mu_request = np.log(mu_values / 91.1876)
    g_hzz = []
    lam_vals = []
    
    for lmu in log_mu_request:

        idx = np.searchsorted(t_arr, lmu)
        if idx <= 0:
            y = y_arr[0]
        elif idx >= len(t_arr):
            y = y_arr[-1]
        else:
            frac = (lmu - t_arr[idx - 1]) / (t_arr[idx] - t_arr[idx - 1] + TINY)
            y = y_arr[idx - 1] + frac * (y_arr[idx] - y_arr[idx - 1])
        
        g1, g2 = y[0], y[1]
        lam = max(y[6], TINY)
        v = 246.0 / np.sqrt(lam / 0.13)
        g_hzz.append(np.sqrt(g1 ** 2 + g2 ** 2) * v / 2.0)
        lam_vals.append(lam)
    
    return np.array(g_hzz), np.array(lam_vals)





def landau_pole_estimate():
    y0 = sm_initial_conditions()
    yt0 = y0[3]
    if yt0 < TINY:
        return np.inf
    exponent = 8.0 * np.pi ** 2 / (9.0 * yt0 ** 2)
    return 91.1876 * np.exp(exponent)





def rge_analysis_report(mu_high=1.0e4, n_steps=1000):
    t_span = (0.0, np.log(mu_high / 91.1876))
    y0 = sm_initial_conditions()
    
    t_arr, y_arr = rk2_integrate(sm_rge_beta, t_span, y0, n_steps)
    mu_arr = 91.1876 * np.exp(t_arr)
    
    stable, issues = check_rge_stability(t_arr, y_arr)
    lp = landau_pole_estimate()
    
    g_hzz, lam_vals = higgs_vv_coupling_evolution(mu_arr)
    
    return {
        "mu": mu_arr,
        "t": t_arr,
        "y": y_arr,
        "g1": y_arr[:, 0],
        "g2": y_arr[:, 1],
        "g3": y_arr[:, 2],
        "yt": y_arr[:, 3],
        "yb": y_arr[:, 4],
        "ytau": y_arr[:, 5],
        "lambda": y_arr[:, 6],
        "g_hzz": g_hzz,
        "lambda_values": lam_vals,
        "stable": stable,
        "issues": issues,
        "landau_pole_gev": lp,
    }
