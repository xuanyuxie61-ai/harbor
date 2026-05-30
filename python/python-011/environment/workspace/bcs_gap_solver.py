# -*- coding: utf-8 -*-

import numpy as np
from utils import safe_sqrt, fermi_dirac
from bz_integration import integrate_bz_gauss_legendre_2d


def d_wave_form_factor(kx, ky):
    return 0.5 * (np.cos(kx) - np.cos(ky))


def quasiparticle_energy(kx, ky, Delta0, t=1.0, tp=0.3, mu=0.0):
    eps = -2.0 * t * (np.cos(kx) + np.cos(ky)) + 4.0 * tp * np.cos(kx) * np.cos(ky) - mu
    dk = Delta0 * d_wave_form_factor(kx, ky)
    return safe_sqrt(eps ** 2 + dk ** 2)


def gap_equation_integrand(k_points, Delta0, U, beta, t=1.0, tp=0.3, mu=0.0):

    raise NotImplementedError("Hole_2: implement f(k) = (U/2) * phi_k^2 * tanh(beta*E_k/2) / E_k")


def gap_equation_rhs(Delta0, U, beta, t=1.0, tp=0.3, mu=0.0, n_k=48):
    if Delta0 < 0:
        Delta0 = abs(Delta0)

    def f(kpts):
        return gap_equation_integrand(kpts, Delta0, U, beta, t, tp, mu)

    val = integrate_bz_gauss_legendre_2d(f, n_per_dim=n_k)



    bz_vol = (2.0 * np.pi) ** 2
    return val / bz_vol


def solve_gap_self_consistent(U, beta, t=1.0, tp=0.3, mu=0.0,
                               Delta_max=5.0, n_k=48, tol=1e-8, max_iter=200):
    if U <= 0:
        return 0.0, [0.0], True

    history = []
    Delta = 0.5
    alpha_mix = 0.3

    for it in range(max_iter):
        rhs = gap_equation_rhs(Delta, U, beta, t, tp, mu, n_k)






        break


    def g(D):
        return gap_equation_rhs(D, U, beta, t, tp, mu, n_k) - 1.0



    g0 = g(0.0)
    if g0 < 0:

        return 0.0, [0.0], False


    d_hi = Delta_max
    for _ in range(50):
        if g(d_hi) < 0:
            break
        d_hi *= 2.0
        if d_hi > 1e4:
            raise RuntimeError("无法找到能隙方程根的上界。")

    d_lo = 0.0
    history = []
    for it in range(max_iter):
        d_mid = (d_lo + d_hi) * 0.5
        g_mid = g(d_mid)
        history.append(d_mid)
        if abs(g_mid) < tol or (d_hi - d_lo) < tol * max(1.0, d_mid):
            return d_mid, history, True
        if g_mid > 0:
            d_lo = d_mid
        else:
            d_hi = d_mid

    Delta_sc = (d_lo + d_hi) * 0.5
    return Delta_sc, history, False


def compute_critical_temperature(U, t=1.0, tp=0.3, mu=0.0,
                                  beta_max=100.0, n_k=48, tol=1e-6):
    if U <= 0:
        return 0.0

    def h(beta):
        def f(kpts):
            kx = kpts[:, 0]
            ky = kpts[:, 1]
            phi = d_wave_form_factor(kx, ky)
            eps = (-2.0 * t * (np.cos(kx) + np.cos(ky))
                   + 4.0 * tp * np.cos(kx) * np.cos(ky) - mu)
            eps = np.where(np.abs(eps) < 1e-12, 1e-12, eps)
            tanh_term = np.tanh(beta * eps * 0.5)
            return 0.5 * U * phi ** 2 * tanh_term / eps
        val = integrate_bz_gauss_legendre_2d(f, n_per_dim=n_k)
        bz_vol = (2.0 * np.pi) ** 2
        return val / bz_vol - 1.0


    h0 = h(1e-6)
    if h0 < 0:
        return 0.0
    h_hi = h(beta_max)
    if h_hi > 0:

        return 1.0 / beta_max

    b_lo = 1e-6
    b_hi = beta_max
    for _ in range(80):
        b_mid = (b_lo + b_hi) * 0.5
        h_mid = h(b_mid)
        if abs(h_mid) < tol:
            break
        if h_mid > 0:
            b_lo = b_mid
        else:
            b_hi = b_mid
    beta_c = (b_lo + b_hi) * 0.5
    return 1.0 / beta_c


def box_behnken_parameter_sweep(U_range, T_range, tp_range, mu_range):
    from utils import box_behnken
    ranges = np.array([
        [U_range[0], U_range[1]],
        [T_range[0], T_range[1]],
        [tp_range[0], tp_range[1]],
        [mu_range[0], mu_range[1]]
    ], dtype=float)
    design = box_behnken(4, ranges)
    return design


def compute_free_energy(Delta0, U, beta, t=1.0, tp=0.3, mu=0.0, n_k=32):
    def integrand(kpts):
        kx = kpts[:, 0]
        ky = kpts[:, 1]
        eps = -2.0 * t * (np.cos(kx) + np.cos(ky)) + 4.0 * tp * np.cos(kx) * np.cos(ky) - mu
        phi = d_wave_form_factor(kx, ky)
        dk = Delta0 * phi
        Ek = safe_sqrt(eps ** 2 + dk ** 2)
        Ek = np.maximum(Ek, 1e-14)

        f = -(2.0 / beta) * np.log(2.0 * np.cosh(beta * Ek * 0.5))
        f += eps - Ek
        return f

    val = integrate_bz_gauss_legendre_2d(integrand, n_per_dim=n_k)
    bz_vol = (2.0 * np.pi) ** 2
    F = val / bz_vol + Delta0 ** 2 / U
    return F
