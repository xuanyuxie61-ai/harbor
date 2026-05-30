
import numpy as np
from physics_core import (
    C_0, coupled_mode_equations, bragg_reflectivity,
    normalized_frequency, bandgap_ratio, cavity_q_factor
)






def task_division(task_number, proc_first, proc_last):
    if task_number < 1:
        raise ValueError("任务数必须 >= 1")
    if proc_first > proc_last:
        raise ValueError("proc_first 必须 <= proc_last")
    
    p = proc_last + 1 - proc_first
    divisions = []
    i_hi = 0
    task_remain = task_number
    proc_remain = p
    
    for proc in range(proc_first, proc_last + 1):
        task_proc = int(np.round(task_remain / proc_remain))
        proc_remain -= 1
        task_remain -= task_proc
        
        i_lo = i_hi + 1
        i_hi = i_hi + task_proc
        
        divisions.append((proc, task_proc, i_lo, i_hi))
    
    return divisions


def divide_k_points(k_points, n_workers):
    N = len(k_points)
    if n_workers < 1:
        raise ValueError("工作进程数必须 >= 1")
    if N < n_workers:
        n_workers = N
    
    divisions = task_division(N, 0, n_workers - 1)
    subsets = []
    for _, _, i_lo, i_hi in divisions:
        subsets.append(k_points[i_lo - 1:i_hi])
    return subsets






def rk4(dydt, tspan, y0, n_steps):
    y0 = np.asarray(y0, dtype=complex).flatten()
    m = len(y0)
    
    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1, m), dtype=complex)
    
    tfirst, tlast = tspan
    dt = (tlast - tfirst) / n_steps
    
    t[0] = tfirst
    y[0, :] = y0
    
    for i in range(n_steps):
        f1 = dydt(t[i], y[i, :])
        f2 = dydt(t[i] + dt / 2.0, y[i, :] + dt * f1 / 2.0)
        f3 = dydt(t[i] + dt / 2.0, y[i, :] + dt * f2 / 2.0)
        f4 = dydt(t[i] + dt, y[i, :] + dt * f3)
        
        t[i + 1] = t[i] + dt
        y[i + 1, :] = y[i, :] + dt * (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0
    
    return t, y


def propagate_bragg_grating(kappa, delta_beta, L, n_z=200):













    raise NotImplementedError("Hole 2: propagate_bragg_grating shooting method needs to be implemented.")






def coupled_mode_fdtm(kappa_profile, delta_beta, L, nz, dt_factor=0.5):
    if L <= 0 or nz < 3:
        raise ValueError("参数超出允许范围")
    
    dz = L / (nz - 1)
    z = np.linspace(0, L, nz)
    

    if callable(kappa_profile):
        kappa_z = np.array([kappa_profile(zi) for zi in z], dtype=complex)
    else:
        kappa_z = np.full(nz, kappa_profile, dtype=complex)
    

    beta_max = abs(delta_beta) + np.max(np.abs(kappa_z))
    if beta_max < 1e-15:
        beta_max = 1.0
    

    A_plus = np.zeros(nz, dtype=complex)
    A_minus = np.zeros(nz, dtype=complex)
    A_plus[0] = 1.0
    

    for i in range(nz - 1):

        kappa_local = 0.5 * (kappa_z[i] + kappa_z[i + 1])
        

        Ap_mid = 0.5 * (A_plus[i] + A_plus[i + 1]) if i < nz - 2 else A_plus[i]
        Am_mid = 0.5 * (A_minus[i] + A_minus[i + 1]) if i < nz - 2 else A_minus[i]
        

        dAp = 1j * delta_beta * A_plus[i] + 1j * kappa_local * A_minus[i]
        A_plus[i + 1] = A_plus[i] + dz * dAp
        


        dAm = -1j * delta_beta * A_minus[i] + 1j * np.conj(kappa_local) * A_plus[i]
        A_minus[i + 1] = A_minus[i] + dz * dAm
    
    return z, A_plus, A_minus






def detect_bandgaps(omega_bands, threshold_ratio=0.05):
    N_k, n_bands = omega_bands.shape
    if n_bands < 2:
        return []
    
    gaps = []
    
    for band_idx in range(n_bands - 1):
        band_n_max = np.max(omega_bands[:, band_idx])
        band_np1_min = np.min(omega_bands[:, band_idx + 1])
        
        if band_np1_min > band_n_max:
            gap_width = band_np1_min - band_n_max
            gap_center = 0.5 * (band_n_max + band_np1_min)
            gap_ratio = bandgap_ratio(band_n_max, band_np1_min)
            
            if gap_ratio >= threshold_ratio:
                gaps.append({
                    'lower_band': band_idx,
                    'upper_band': band_idx + 1,
                    'omega_lower': band_n_max,
                    'omega_upper': band_np1_min,
                    'omega_center': gap_center,
                    'gap_width': gap_width,
                    'relative_width': gap_ratio,
                    'mid_gap_frequency': gap_center
                })
    
    return gaps


def gap_mismatch_parameter(eps_bg, eps_hole, fill_factor):
    if eps_bg <= 0 or eps_hole <= 0:
        raise ValueError("介电常数必须为正")
    if not (0 <= fill_factor <= 1):
        raise ValueError("填充因子必须在 [0, 1] 区间内")
    
    n_bg = np.sqrt(eps_bg)
    n_hole = np.sqrt(eps_hole)
    
    mismatch = abs(n_bg - n_hole) / (n_bg + n_hole)
    geometric_factor = abs(np.sin(np.pi * fill_factor))
    
    return (4.0 / np.pi) * mismatch * geometric_factor


def defect_mode_frequency(omega_gap_center, defect_strength, Q_factor):
    if omega_gap_center <= 0 or Q_factor <= 0:
        raise ValueError("频率和 Q 值必须为正")
    

    omega_defect = omega_gap_center * (1.0 - 0.5 * defect_strength)
    delta_omega = omega_defect / Q_factor
    
    return omega_defect, delta_omega


def slow_light_group_index(omega, k, band_index=0):
    if len(omega) < 3 or len(k) < 3:
        raise ValueError("数据点必须至少 3 个")
    if len(omega) != len(k):
        raise ValueError("omega 和 k 长度必须一致")
    

    dk_domega = np.zeros(len(omega))
    
    domega_0 = omega[1] - omega[0]
    if abs(domega_0) > 1e-18:
        dk_domega[0] = (k[1] - k[0]) / domega_0
    
    domega_end = omega[-1] - omega[-2]
    if abs(domega_end) > 1e-18:
        dk_domega[-1] = (k[-1] - k[-2]) / domega_end
    
    for i in range(1, len(omega) - 1):
        domega = omega[i + 1] - omega[i - 1]
        if abs(domega) < 1e-18:
            dk_domega[i] = 0.0
        else:
            dk_domega[i] = (k[i + 1] - k[i - 1]) / domega
    
    n_g = C_0 * dk_domega

    n_g = np.clip(n_g, -1e6, 1e6)
    
    return n_g
