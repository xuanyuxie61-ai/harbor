# -*- coding: utf-8 -*-

import numpy as np
from math import pi, sqrt, sin, cos


def rubber_band_resonator(t, y, a=1.0, b=2.0, lam=5.0, mu=1.0):
    y = np.asarray(y, dtype=complex)
    if y.shape[0] != 2:
        raise ValueError("状态向量必须为 [u, v]")
    u = float(np.real(y[0]))
    v = float(np.real(y[1]))
    up = v
    vp = (10.0 + lam * sin(mu * t) - 0.01 * v -
          a * max(u, 0.0) + b * max(-u, 0.0))
    return np.array([up, vp], dtype=complex)


def pendulum_meta_atom(t, y, g=9.81, l=1.0):
    y = np.asarray(y, dtype=complex)
    if y.shape[0] != 2:
        raise ValueError("状态向量必须为 [θ, ω]")
    theta = float(np.real(y[0]))
    omega = float(np.real(y[1]))

    theta = ((theta + pi) % (2.0 * pi)) - pi
    dtheta = omega
    domega = -(g / l) * sin(theta)
    return np.array([dtheta, domega], dtype=complex)


def duffing_resonator(t, y, alpha=1.0, beta=0.1, gamma=0.05, omega=1.0, F=1.0):
    y = np.asarray(y, dtype=complex)
    if y.shape[0] != 2:
        raise ValueError("状态向量必须为 [u, v]")
    u = float(np.real(y[0]))
    v = float(np.real(y[1]))
    up = v
    vp = F * cos(omega * t) - gamma * v - alpha * u - beta * (u ** 3)

    if abs(vp) > 1e6:
        vp = np.sign(vp) * 1e6
    return np.array([up, vp], dtype=complex)


def effective_phase_shift_nonlinear(incident_intensity, params):
    I = float(incident_intensity)
    if I < 0:
        I = 0.0
    omega0 = params.get('omega0', 1.0)
    omega = params.get('omega', 1.0)
    gamma = params.get('gamma', 0.05)
    kappa = params.get('kappa', 0.01)
    I_sat = params.get('I_sat', 1.0)

    if I_sat < 1e-15:
        I_sat = 1e-15


    detuning = omega0 ** 2 - omega ** 2
    denom = max(gamma * omega, 1e-15)
    phi_linear = np.arctan2(detuning, denom)


    saturation_factor = 1.0 / sqrt(1.0 + kappa * I / I_sat)
    phi_eff = phi_linear * saturation_factor
    return phi_eff


def pendulum_period_small_angle(g, l):
    if g <= 0 or l <= 0:
        raise ValueError("g 和 l 必须为正")
    return 2.0 * pi * sqrt(l / g)


def pendulum_period_elliptic(theta0, g, l, n_terms=10):
    if g <= 0 or l <= 0:
        raise ValueError("g 和 l 必须为正")
    if abs(theta0) >= pi:
        theta0 = np.sign(theta0) * (pi - 1e-6)
    k = sin(theta0 / 2.0) ** 2
    K = pi / 2.0
    term = 1.0
    for n in range(1, n_terms):

        term *= ((2.0 * n - 1.0) / (2.0 * n)) ** 2 * k
        K += term
        if abs(term) < 1e-15:
            break
    T = 4.0 * sqrt(l / g) * K
    return T


def nonlinear_transmission_coefficient(intensity, params):
    I = float(intensity)
    if I < 0:
        I = 0.0
    A0 = params.get('A0', 1.0)
    I_sat = params.get('I_sat', 1.0)
    if I_sat < 1e-15:
        I_sat = 1e-15
    phi = effective_phase_shift_nonlinear(I, params)
    A = A0 / sqrt(1.0 + I / I_sat)
    A = np.clip(A, 0.0, 1.0)
    return A * np.exp(1j * phi)
