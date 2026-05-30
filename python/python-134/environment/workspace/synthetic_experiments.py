#!/usr/bin/env python3

import numpy as np


def generate_polarization_curve(params, n_points=100):
    T = params['T']
    R = params['R']
    F = params['F']
    alpha = params['alpha_a']
    j0 = params['j_0_ref']
    t_m = params['t_membrane']
    sigma_m = params['sigma_m_ref']
    lambda_eq = params['lambda_eq']


    P_H2 = 1.0
    P_O2 = 0.21
    P_H2O = 0.5
    E_rev = params['E_0'] - (R * T / (2.0 * F)) * np.log(P_H2 * np.sqrt(P_O2) / P_H2O)


    j_min = 1e-4
    j_max = 2.0
    j = np.logspace(np.log10(j_min), np.log10(j_max), n_points)


    eta_act = (R * T) / (alpha * F) * np.arcsinh(j / (2.0 * j0))



    sigma_lambda = sigma_m * (0.005139 * lambda_eq - 0.00326)
    sigma_lambda = max(sigma_lambda, 1e-3)
    eta_ohm = j * (t_m / sigma_lambda) * 1e-4


    j_L = 3.0
    j_ratio = np.clip(j / j_L, 0.0, 0.99)
    eta_conc = -(R * T / (4.0 * F)) * np.log(1.0 - j_ratio)


    V_cell = E_rev - eta_act - eta_ohm - eta_conc
    V_cell = np.clip(V_cell, 0.0, E_rev)

    return V_cell, j


def generate_impedance_spectrum(params, n_freq=80):
    T = params['T']
    R_ohm = 0.05
    R_ct = 0.2
    C_dl = 0.02

    freq = np.logspace(-3, 5, n_freq)
    omega = 2.0 * np.pi * freq

    Z_real = R_ohm + R_ct / (1.0 + (omega * R_ct * C_dl) ** 2)
    Z_imag = -omega * R_ct ** 2 * C_dl / (1.0 + (omega * R_ct * C_dl) ** 2)

    return freq, Z_real, Z_imag


def generate_humidity_scan_data(params, n_rh=20):
    RH_range = np.linspace(0.3, 1.0, n_rh)
    T_range = np.linspace(313.15, 363.15, 5)

    data_matrix = np.zeros((n_rh, len(T_range)))
    for i, rh in enumerate(RH_range):
        for j, T in enumerate(T_range):

            lambda_w = 0.043 + 17.81 * rh - 39.85 * rh ** 2 + 36.0 * rh ** 3
            lambda_w = np.clip(lambda_w, 0.0, 22.0)


            sigma = 0.005139 * lambda_w - 0.00326
            sigma = max(sigma, 1e-3)
            P_max = 0.5 * sigma * (1.0 - T / 400.0)
            data_matrix[i, j] = max(P_max, 0.0)

    return RH_range, T_range, data_matrix


if __name__ == '__main__':
    p = {
        'T': 353.15, 'R': 8.314, 'F': 96485.0, 'alpha_a': 0.5,
        'j_0_ref': 1e-3, 't_membrane': 50e-6, 'sigma_m_ref': 10.0,
        'lambda_eq': 14.0, 'E_0': 1.229
    }
    V, I = generate_polarization_curve(p)
    print("V range:", V.min(), V.max())
