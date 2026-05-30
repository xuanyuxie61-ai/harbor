
import numpy as np
from typing import Tuple


def sncndn(u: float, m: float, tol: float = 1e-10, max_iter: int = 100) -> Tuple[float, float, float]:

    if abs(m) < tol:
        return np.sin(u), np.cos(u), 1.0
    
    if abs(m - 1.0) < tol:
        su = np.sinh(u)
        cu = np.cosh(u)
        sech_u = 1.0 / cu if abs(cu) > tol else 0.0
        return np.tanh(u), sech_u, sech_u
    

    if m > 1.0:
        mp = 1.0 / m
        up = u * np.sqrt(m)
        snp, cnp, dnp = sncndn(up, mp, tol, max_iter)
        sn_val = snp / np.sqrt(m)
        cn_val = dnp
        dn_val = cnp

        if abs(sn_val) > 1.0:
            sn_val = np.sign(sn_val)
            cn_val = 0.0
        return sn_val, cn_val, dn_val
    

    if m < 0.0:
        mp = -m / (1.0 - m)
        up = u / np.sqrt(1.0 - m)
        snp, cnp, dnp = sncndn(up, mp, tol, max_iter)
        denom = 1.0 - mp * snp**2
        if abs(denom) < tol:
            denom = tol
        sn_val = snp * np.sqrt(1.0 - mp * (1.0 - snp**2)) / denom
        cn_val = cnp * dnp / denom
        dn_val = (1.0 - mp * snp**2) / denom
        return sn_val, cn_val, dn_val
    

    a = 1.0
    b = np.sqrt(1.0 - m)
    c_val = np.sqrt(m)
    
    n_iter = 0
    while abs(c_val) > tol and n_iter < max_iter:
        a_next = 0.5 * (a + b)
        b_next = np.sqrt(a * b)
        c_next = 0.5 * (a - b)
        a, b, c_val = a_next, b_next, c_next
        n_iter += 1
    

    phi = 2**n_iter * a * u
    


    sn_val = np.sin(phi)
    cn_val = np.cos(phi)
    dn_val = 1.0 - 0.5 * m * sn_val**2
    

    if abs(sn_val) > 1.0:
        sn_val = np.sign(sn_val)
        cn_val = 0.0
    
    return sn_val, cn_val, dn_val


def jacobi_sn(u: float, m: float) -> float:
    sn, _, _ = sncndn(u, m)
    return sn


def jacobi_cn(u: float, m: float) -> float:
    _, cn, _ = sncndn(u, m)
    return cn


def jacobi_dn(u: float, m: float) -> float:
    _, _, dn = sncndn(u, m)
    return dn


def burgers_periodic_solution(x: np.ndarray, t: float, A: float = 1.0,
                               nu: float = 0.01, m: float = 0.5) -> np.ndarray:


    k_wave = np.sqrt(A / (2.0 * nu))
    c_wave = A * (2.0 - m) / 3.0
    
    u = np.zeros_like(x)
    for i, xi in enumerate(x):
        phase = k_wave * (xi - c_wave * t)
        _, cn_val, _ = sncndn(phase, m)
        u[i] = A * cn_val**2
    
    return u


def shock_wave_formation(x: np.ndarray, t: float, u0: float = 1.0,
                         x0: float = 0.5, L: float = 1.0) -> np.ndarray:
    t_shock = L / u0
    u = np.zeros_like(x)
    
    if t < t_shock:

        for i, xi in enumerate(x):


            u[i] = u0 * (xi - x0) / (L + u0 * t)
    else:


        x_shock = x0 + 0.5 * u0 * t_shock + 0.5 * u0 * (t - t_shock)
        
        for i, xi in enumerate(x):
            if xi < x_shock:
                u[i] = u0 * (xi - x0) / (L + u0 * t)
                if u[i] < 0:
                    u[i] = 0.0
            else:
                u[i] = 0.0
    
    return u


def nonlinear_acoustic_parameter_estimation(pressure_amplitudes: np.ndarray,
                                            frequencies: np.ndarray) -> dict:
    if len(pressure_amplitudes) < 2:
        return {'error': '需要至少2个频率点的数据'}
    


    p1 = pressure_amplitudes[0]
    p2 = pressure_amplitudes[1]
    f = frequencies[0]
    

    rho0 = 1000.0
    c0 = 1540.0
    z = 0.05
    
    if abs(p1) < 1e-14:
        return {'error': '基波幅值过小'}
    

    efficiency = p2 / p1
    denom = np.pi * f * z * p1 / (2.0 * rho0 * c0**3)
    
    if abs(denom) < 1e-14:
        return {'error': '分母过小，无法估计'}
    
    BA_estimated = efficiency / denom - 2.0
    
    return {
        'B_over_A': float(BA_estimated),
        'efficiency': float(efficiency),
        'fundamental_pressure': float(p1),
        'harmonic_pressure': float(p2),
        'frequency_MHz': float(f / 1e6)
    }
