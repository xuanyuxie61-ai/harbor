#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import wofz


def plasma_dispersion_function(zeta):
    zeta = np.asarray(zeta, dtype=complex)
    



    

    abs_z = np.abs(zeta)
    
    if np.isscalar(abs_z):
        if abs_z > 50.0:

            return -1.0/zeta - 1.0/(2.0*zeta**3) - 3.0/(4.0*zeta**5)
        else:
            return 1j * np.sqrt(np.pi) * wofz(zeta)
    else:

        Z = np.zeros_like(zeta, dtype=complex)
        mask_large = abs_z > 50.0
        mask_small = ~mask_large
        
        if np.any(mask_large):
            z_l = zeta[mask_large]
            Z[mask_large] = -1.0/z_l - 1.0/(2.0*z_l**3) - 3.0/(4.0*z_l**5)
        
        if np.any(mask_small):
            Z[mask_small] = 1j * np.sqrt(np.pi) * wofz(zeta[mask_small])
        
        return Z


def d_plasma_dispersion_function(zeta):
    Z = plasma_dispersion_function(zeta)
    return -2.0 * (1.0 + zeta * Z)


def whistler_dispersion_residual(omega, k, params):
    q_e = params['q_e']
    m_e = params['m_e']
    c = params['c']
    eps0 = params['eps0']
    B0 = params['B0']
    n0 = params['n0']
    Omega_e = params['Omega_e']
    omega_pe = params['omega_pe']
    v_te = params['v_te']
    

    if np.abs(omega) < 1e-20:
        omega = 1e-20 + 0j
    if np.abs(k) < 1e-20:
        k = 1e-20
    

    zeta_e = (omega - Omega_e) / (np.abs(k) * v_te)
    

    Z_e = plasma_dispersion_function(zeta_e)
    Zp_e = d_plasma_dispersion_function(zeta_e)
    


    prefactor = omega_pe**2 / (2.0 * omega * Omega_e)
    
    bracket = Z_e - (1.0 - omega / (k * v_te)) * Zp_e
    D = 1.0 - prefactor * bracket
    

    dzeta_domega = 1.0 / (np.abs(k) * v_te)
    dZ_domega = Zp_e * dzeta_domega
    

    Zpp_e = -2.0 * Z_e - 2.0 * zeta_e * Zp_e
    dZp_domega = Zpp_e * dzeta_domega
    
    dbracket_domega = dZ_domega - (-1.0/(k*v_te)) * Zp_e - (1.0 - omega/(k*v_te)) * dZp_domega
    
    dprefactor_domega = -omega_pe**2 / (2.0 * Omega_e * omega**2)
    
    dD_domega = -dprefactor_domega * bracket - prefactor * dbracket_domega
    
    return D, dD_domega


def solve_whistler_dispersion(k, params, omega_guess=None, tol=1e-10, max_iter=50):
    Omega_e = params['Omega_e']
    omega_pe = params['omega_pe']
    

    if omega_guess is None:

        c = params['c']
        omega_r = Omega_e * k**2 * c**2 / (omega_pe**2 + k**2 * c**2)
        gamma = -0.05 * omega_r
        omega_guess = complex(omega_r, gamma)
    

    ido = 0
    omega = complex(omega_guess)
    
    epsilon = np.sqrt(np.sqrt(np.finfo(float).eps))
    ncall = 0
    
    for iteration in range(max_iter):
        ncall += 1
        

        D_val, dD_val = whistler_dispersion_residual(omega, k, params)
        

        if not np.isfinite(D_val) or not np.isfinite(dD_val):

            omega = omega * (1.0 + epsilon)
            continue
        

        if np.abs(dD_val) < 1e-30:

            dD_val = np.sign(dD_val.real) * 1e-30 + 1j * np.sign(dD_val.imag) * 1e-30 if dD_val != 0 else 1e-30 + 0j
        
        delta_omega = -D_val / dD_val
        

        if np.abs(delta_omega) > 0.5 * np.abs(omega):
            delta_omega = delta_omega * (0.5 * np.abs(omega) / np.abs(delta_omega))
        
        omega_new = omega + delta_omega
        

        if np.abs(D_val) < tol * (np.abs(D_val) + 1.0):
            return omega
        

        if iteration > 15:
            if np.abs(D_val) > 0.95 * np.abs(D_val):

                pass
        
        omega = omega_new
        

        if omega.real < 0:
            omega = complex(0.01 * Omega_e, omega.imag)
    

    D_final, _ = whistler_dispersion_residual(omega, k, params)
    if np.abs(D_final) < 100 * tol:
        return omega
    
    return None
