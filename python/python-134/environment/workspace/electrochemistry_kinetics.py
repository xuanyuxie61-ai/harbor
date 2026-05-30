#!/usr/bin/env python3

import numpy as np


def compute_exchange_current_density(params):
    T = params['T']
    j0_ref = params['j_0_ref']

    theta = np.exp(-5000.0 / params['R'] * (1.0 / T - 1.0 / 298.15))
    j0 = j0_ref * max(theta, 1e-6)
    return j0


def butler_volmer_kinetics(eta, params):
    T = params['T']
    F = params['F']
    R = params['R']
    alpha_a = params['alpha_a']
    alpha_c = params['alpha_c']
    j0 = compute_exchange_current_density(params)


    arg_max = 500.0
    arg_fwd = np.clip(alpha_a * F * eta / (R * T), -arg_max, arg_max)
    arg_rev = np.clip(-alpha_c * F * eta / (R * T), -arg_max, arg_max)

    j = j0 * (np.exp(arg_fwd) - np.exp(arg_rev))
    return j


def reaction_source_terms(c, state, params):
    c_H2, c_O2, c_Hp, lambda_w, T_loc = state


    j_local = butler_volmer_kinetics(c, params)


    n_e = 2.0
    r = j_local / (n_e * params['F'])





    S = np.array([
        [-1.0,  0.0],
        [ 0.0, -1.0],
        [ 2.0, -4.0],
        [ 0.0,  2.0],
        [ 0.0,  0.0],
    ], dtype=float)

    rate_vec = np.array([r, r * 0.5], dtype=float)
    source = S @ rate_vec
    return source


def compute_conserved_quantities(state):
    c_H2, c_O2, c_Hp, lambda_w, T_loc = state


    E = np.array([
        [2.0, 0.0, 1.0, 2.0, 0.0],
        [0.0, 2.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
    ], dtype=float)
    h = E @ np.array(state, dtype=float)
    return h


def compute_activation_overpotential(j_target, params, side='cathode'):
    T = params['T']
    F = params['F']
    R = params['R']
    j0 = compute_exchange_current_density(params)

    if side == 'cathode':
        alpha = params['alpha_c']

        eta = (R * T) / (alpha * F) * np.log(np.maximum(j_target, 1e-10) / j0)
    else:
        alpha = params['alpha_a']
        eta = (R * T) / (alpha * F) * np.arcsinh(j_target / (2.0 * j0))
    return eta


if __name__ == '__main__':
    p = {
        'T': 353.15, 'P': 1.5, 'R': 8.314, 'F': 96485.0,
        'alpha_a': 0.5, 'alpha_c': 0.5, 'j_0_ref': 1e-3
    }
    eta = np.linspace(-0.3, 0.3, 100)
    j = butler_volmer_kinetics(eta, p)
    print("Butler-Volmer max j:", np.max(np.abs(j)))
