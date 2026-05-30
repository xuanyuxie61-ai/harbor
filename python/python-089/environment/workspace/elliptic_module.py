
import numpy as np
from scipy.special import ellipk, ellipe, ellipj


def complete_elliptic_k(m):
    m = np.asarray(m, dtype=float)
    m = np.clip(m, 0.0, 1.0 - 1e-15)
    return ellipk(m)


def complete_elliptic_e(m):
    m = np.asarray(m, dtype=float)
    m = np.clip(m, 0.0, 1.0 - 1e-15)
    return ellipe(m)


def jacobi_elliptic_functions(u, m):
    m = float(m)
    m = np.clip(m, 0.0, 1.0)
    u = np.asarray(u, dtype=float)
    sn, cn, dn = ellipj(u, m)
    return sn, cn, dn


def elastica_beam_deflection(P, EI, L, n_points=100):
    if P <= 0 or EI <= 0 or L <= 0:
        return np.linspace(0, L, n_points), np.zeros(n_points), 0.0, 0.0
    

    lam_sq = P * L ** 2 / EI
    lam = np.sqrt(lam_sq)
    



    
    def residual(theta_m):
        if theta_m <= 0 or theta_m >= np.pi:
            return 1e10
        k = np.sin(theta_m / 2.0)
        m = k ** 2
        return complete_elliptic_k(m) - lam / 2.0
    

    th_lo, th_hi = 0.01, np.pi - 0.01
    for _ in range(50):
        th_mid = (th_lo + th_hi) / 2.0
        if residual(th_mid) > 0:
            th_hi = th_mid
        else:
            th_lo = th_mid
    
    theta_max = (th_lo + th_hi) / 2.0
    k_modulus = np.sin(theta_max / 2.0)
    m = k_modulus ** 2
    

    phi = np.linspace(0, np.pi / 2.0, n_points)
    




    
    scale = np.sqrt(EI / P)
    


    K_val = complete_elliptic_k(m)
    u_vals = np.linspace(0, K_val, n_points)
    
    sn, cn, dn = jacobi_elliptic_functions(u_vals, m)
    

    x = scale * (u_vals - complete_elliptic_e(m) * u_vals / K_val + 
                 np.cumsum(dn ** 2) * (K_val / n_points))

    x = scale * (u_vals - ellipe(m) * u_vals / K_val)



    s_vals = np.linspace(0, L, n_points)
    u_param = s_vals * np.sqrt(P / EI)
    
    sn_s, cn_s, dn_s = jacobi_elliptic_functions(u_param, m)
    theta_s = 2.0 * np.arcsin(np.clip(k_modulus * sn_s, -1.0, 1.0))
    

    dx = np.cos(theta_s)
    dy = np.sin(theta_s)
    x = np.cumsum(dx) * (L / n_points)
    y = np.cumsum(dy) * (L / n_points)
    
    return x, y, theta_max, k_modulus


def nonlinear_vibration_period(amplitude, omega_linear, alpha_nonlin):
    if amplitude <= 0 or omega_linear <= 0:
        return 2 * np.pi / omega_linear, 2 * np.pi / omega_linear
    
    omega_sq = omega_linear ** 2
    
    if alpha_nonlin > 0:

        denom = np.sqrt(omega_sq + alpha_nonlin * amplitude ** 2)
        m = alpha_nonlin * amplitude ** 2 / (2.0 * (omega_sq + alpha_nonlin * amplitude ** 2))
        m = np.clip(m, 0.0, 1.0)
        T = 4.0 * complete_elliptic_k(m) / denom
    elif alpha_nonlin < 0:

        alpha_abs = abs(alpha_nonlin)
        if amplitude >= np.sqrt(omega_sq / alpha_abs):
            amplitude = 0.99 * np.sqrt(omega_sq / alpha_abs)
        denom = np.sqrt(omega_sq - alpha_abs * amplitude ** 2 / 2.0)
        m = alpha_abs * amplitude ** 2 / (2.0 * omega_sq - alpha_abs * amplitude ** 2)
        m = np.clip(m, 0.0, 1.0)
        T = 4.0 * complete_elliptic_k(m) / denom
    else:
        T = 2 * np.pi / omega_linear
    
    T_linear = 2 * np.pi / omega_linear
    return T, T_linear


def elliptical_hole_stress_concentration(a, b, sigma_inf, theta):
    if a < b:
        a, b = b, a
    

    if b > 1e-14:
        kt = 1.0 + 2.0 * a / b
    else:
        kt = 1e10
    




    sigma_theta = sigma_inf * (1.0 + 2.0 * a / b * np.cos(2.0 * theta) + 
                                (a / b) ** 2 * np.sin(theta) ** 2)
    
    return sigma_theta, kt
