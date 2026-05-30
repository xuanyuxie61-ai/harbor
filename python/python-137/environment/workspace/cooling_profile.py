# -*- coding: utf-8 -*-

import numpy as np


def linear_cooling(t, T0, Tf, t_total):
    t = np.asarray(t, dtype=float)
    if t_total <= 0:
        return np.full_like(t, T0)
    ratio = np.clip(t / t_total, 0.0, 1.0)
    return T0 + (Tf - T0) * ratio


def natural_cooling(t, T0, T_env, tau):
    t = np.asarray(t, dtype=float)
    if tau <= 0:
        return np.full_like(t, T0)
    return T_env + (T0 - T_env) * np.exp(-t / tau)


def sawtooth_cooling(t, T_base, delta_T, period, phase=0.0):
    t = np.asarray(t, dtype=float)
    if period <= 0:
        period = 1.0


    s = np.mod(t + phase, period) / period - 0.5

    if callable(T_base):
        Tb = T_base(t)
    else:
        Tb = np.full_like(t, float(T_base), dtype=float) if np.isscalar(T_base) else np.asarray(T_base, dtype=float)

    return Tb + delta_T * s


def optimal_cooling_polynomial(t, T0, Tf, t_total, order=3):
    t = np.asarray(t, dtype=float)
    if t_total <= 0:
        return np.full_like(t, T0)
    ratio = np.clip(t / t_total, 0.0, 1.0)
    return T0 + (Tf - T0) * (ratio ** order)


def solubility_vanthoff(T, H_diss, S_diss, R=8.314):
    T = np.asarray(T, dtype=float)
    T = np.where(T <= 0, 1e-6, T)
    return np.exp(-H_diss / (R * T) + S_diss / R)


def supersaturation(c, T, H_diss, S_diss):
    c_sat = solubility_vanthoff(T, H_diss, S_diss)

    c_sat = np.where(np.abs(c_sat) < 1e-300, 1e-300, c_sat)
    return (c - c_sat) / c_sat
