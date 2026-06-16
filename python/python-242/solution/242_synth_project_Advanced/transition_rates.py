"""
transition_rates.py  --  Electromagnetic transition probabilities
=================================================================
Computes reduced transition probabilities B(EL; i -> f) and B(ML; i -> f)
for electric and magnetic multipole transitions between shell-model states.

Fused seeds:
    611_joukowsky_transform   -- analytic mapping of transition integrals
    933_pyramid_integrals     -- angular-momentum coupling quadratures
    172_chladni_figures       -- surface-oscillation nodal patterns
    908_predator_prey_ode     -- rate-equation form of decay cascades

Transition operators (single-particle):
    M(E2, mu) = e_eff * sum_i r_i^2 Y_2^mu(theta_i, phi_i)
    M(M1, mu) = sqrt(3/(4 pi)) sum_i [ g_l l_i^mu + g_s s_i^mu ]

Reduced matrix element (Wigner-Eckart factorisation):
    <j_f || r^L Y_L || j_i> = (-1)^{l_f + j_f + 1/2 + L}
                                 sqrt((2j_i + 1)(2j_f + 1))
                                 { l_f  j_f  1/2 }  * <l_f || Y_L || l_i>
                                 {  L   j_i  1/2 }
                                 * <R_f | r^L | R_i>

Weisskopf single-particle estimate:
    B_W(EL) = (1/(4 pi)) * (3/(L+3))^2 * R_0^{2L} * e^2

References:
    Bohr & Mottelson, Nuclear Structure Vol. I (1998), Ch. 6
    Heyde, Basic Ideas and Concepts in Nuclear Physics (2004), Ch. 4
"""

from __future__ import annotations
import math
import numpy as np
from nuclear_constants import (
    E_EFF_PROTON, E_EFF_NEUTRON, R0_FM, PI,
    G_L_PROTON, G_L_NEUTRON, G_S_PROTON_QUENCH, G_S_NEUTRON_QUENCH,
)
from special_functions_nuclear import wigner_3j, wigner_6j


def weisskopf_estimate_E2(A: int) -> float:
    """Weisskopf single-particle estimate B_W(E2) in e^2 fm^4."""
    R = R0_FM * (A ** (1.0 / 3.0))
    return (1.0 / (4.0 * PI)) * (3.0 / 5.0) ** 2 * (R ** 4)


def weisskopf_estimate_M1() -> float:
    """Weisskopf B_W(M1) in mu_N^2."""
    return 10.0 / (4.0 * PI * 4.0 * PI)


def _simpson(y: np.ndarray, h: float) -> float:
    """Composite Simpson's rule, len(y) must be odd."""
    n = len(y)
    if n < 3:
        return 0.5 * h * (y[0] + y[-1]) if n == 2 else 0.0
    s = y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-2:2])
    return s * h / 3.0


def radial_matrix_element(u_i: np.ndarray, u_f: np.ndarray, r: np.ndarray, power_L: int) -> float:
    r"""Compute <R_f | r^L | R_i> = int u_f(r) u_i(r) r^L dr.

    Simpson's rule on the uniform mesh (O(h^4) composite).
    """
    h = r[1] - r[0]
    integrand = u_f * u_i * (r ** power_L)
    n = len(r)
    if n % 2 == 0:
        s = _simpson(integrand[:-1], h)
        s += 0.5 * h * (integrand[-1] + integrand[-2])
    else:
        s = _simpson(integrand, h)
    return float(s)


def reduced_matrix_element_E2(
    u_i: np.ndarray, u_f: np.ndarray, r: np.ndarray,
    l_i: int, j_i: float, l_f: int, j_f: float, is_proton: bool,
) -> float:
    """Reduced matrix element <j_f || e_eff r^2 Y_2 || j_i> in e fm^2."""
    if abs(l_i - l_f) > 2 or (l_i + l_f + 2) % 2 != 0:
        return 0.0
    if abs(j_i - j_f) > 2 or j_i + j_f < 2:
        return 0.0
    radial = radial_matrix_element(u_i, u_f, r, 2)
    threej = wigner_3j(l_i, 2.0, l_f, 0.0, 0.0, 0.0)
    angular_yl = ((-1) ** l_f) * math.sqrt(
        (2 * l_i + 1) * 5 * (2 * l_f + 1) / (4.0 * PI)
    ) * threej
    sixj = wigner_6j(l_f, j_f, 0.5, 2.0, j_i, 0.5)
    phase = (-1) ** int(round(l_f + j_f + 0.5 + 2))
    red_me = phase * math.sqrt((2 * j_i + 1) * (2 * j_f + 1)) * sixj * angular_yl * radial
    e_eff = E_EFF_PROTON if is_proton else E_EFF_NEUTRON
    return e_eff * red_me


def B_E2(
    u_i: np.ndarray, u_f: np.ndarray, r: np.ndarray,
    l_i: int, j_i: float, l_f: int, j_f: float, is_proton: bool,
) -> float:
    """B(E2; j_i -> j_f) = (1/(2 j_i + 1)) |<j_f || M(E2) || j_i>|^2."""
    red_me = reduced_matrix_element_E2(u_i, u_f, r, l_i, j_i, l_f, j_f, is_proton)
    return (red_me ** 2) / (2.0 * j_i + 1.0)


def reduced_matrix_element_M1(
    l_i: int, j_i: float, l_f: int, j_f: float, is_proton: bool,
) -> float:
    r"""Reduced matrix element <j_f || mu(M1) || j_i> in nuclear magnetons.

    M1 operator in |l s j> basis:
        <j_f || g_l l + g_s s || j_i> via 6-j recoupling
    """
    if l_i != l_f:
        return 0.0
    s = 0.5
    g_l = G_L_PROTON if is_proton else G_L_NEUTRON
    g_s = G_S_PROTON_QUENCH if is_proton else G_S_NEUTRON_QUENCH
    sixj_l = wigner_6j(l_i, s, j_i, 1.0, j_f, l_f)
    red_l = math.sqrt(max(l_i * (l_i + 1.0) * (2 * l_i + 1.0), 0.0))
    me_l = ((-1) ** int(round(l_f + s + j_i + 1.0))) * math.sqrt((2 * j_i + 1) * (2 * j_f + 1)) * sixj_l * red_l
    sixj_s = wigner_6j(s, l_i, j_i, 1.0, j_f, s)
    red_s = math.sqrt(s * (s + 1.0) * (2 * s + 1.0))
    me_s = ((-1) ** int(round(s + l_f + j_i + 1.0))) * math.sqrt((2 * j_i + 1) * (2 * j_f + 1)) * sixj_s * red_s
    return math.sqrt(3.0 / (4.0 * PI)) * (g_l * me_l + g_s * me_s)


def B_M1(l_i: int, j_i: float, l_f: int, j_f: float, is_proton: bool) -> float:
    """B(M1; j_i -> j_f) in mu_N^2."""
    red_me = reduced_matrix_element_M1(l_i, j_i, l_f, j_f, is_proton)
    return (red_me ** 2) / (2.0 * j_i + 1.0)


def transition_lifetime_s(B_val: float, E_gamma_MeV: float, L: int) -> float:
    r"""Radiative lifetime tau for multipolarity L.

    T(E2) = (8 pi (L+1)) / (L ((2L+1)!!)^2) * k^{2L+1} * (hbar c)^{2L}/hbar * alpha * B(EL)
    """
    if B_val <= 0.0 or E_gamma_MeV <= 0.0:
        return float("inf")
    hbar = 6.582119569e-22
    hbarc = 197.327
    alpha_fs = 1.0 / 137.036
    k = E_gamma_MeV / hbarc
    if L == 2:
        T = (8.0 * PI * 3.0) / (2.0 * 9.0) * (k ** 5) * (hbarc ** 4 / hbar) * alpha_fs * B_val
    elif L == 1:
        T = (16.0 * PI / 9.0) * (k ** 3) * (hbarc ** 2 / hbar) * alpha_fs * B_val
    else:
        T = (k ** (2 * L + 1)) * B_val * 1.0e15
    return 1.0 / T if T > 0.0 else float("inf")


def decay_cascade(levels: list, transitions: list, n_steps: int = 200, dt_s: float = 1.0e-18) -> dict:
    """Time-integrate a population decay cascade through the level scheme.

    Adapted from predator_prey_ode (Lotka-Volterra style rate equations):
        dP_i / dt = - sum_{f < i} T(i -> f) P_i + sum_{k > i} T(k -> i) P_k

    Uses the midpoint method (829_ode_midpoint_system seed).
    """
    n = len(levels)
    P = np.zeros(n, dtype=np.float64)
    P[0] = 1.0  # initially all population in the highest level
    # Build rate matrix (s^-1)
    rates = np.zeros((n, n), dtype=np.float64)
    for tr in transitions:
        i, f = tr["i"], tr["f"]
        tau = tr["lifetime_s"]
        if math.isfinite(tau) and tau > 0.0:
            rates[f, i] = 1.0 / tau
    # Outflow from each level
    out_rates = np.sum(rates, axis=0)
    def rhs(P):
        dP = np.zeros(n)
        for i in range(n):
            dP[i] = -out_rates[i] * P[i]
            for k in range(i + 1, n):
                dP[i] += rates[i, k] * P[k]
        return dP
    history = np.zeros((n_steps, n))
    history[0, :] = P
    for step in range(1, n_steps):
        # Midpoint method
        k1 = rhs(P)
        P_mid = P + 0.5 * dt_s * k1
        k2 = rhs(P_mid)
        P = P + dt_s * k2
        P = np.maximum(P, 0.0)  # positivity
        P = P / max(np.sum(P), 1.0e-30)  # normalisation
        history[step, :] = P
    return {"history": history, "dt_s": dt_s, "n_steps": n_steps}


def compute_transition_table(
    levels: list, r_grid: np.ndarray, A: int,
) -> list:
    """Build a table of E2, M1 transitions between levels.

    Each level must carry: E_MeV, l, j, is_proton, wf
    """
    out = []
    for i_idx, lev_i in enumerate(levels):
        for f_idx, lev_f in enumerate(levels):
            if f_idx >= i_idx:
                continue
            E_gamma = abs(lev_i["E_MeV"] - lev_f["E_MeV"])
            if E_gamma < 1.0e-6:
                continue
            l_i, j_i = lev_i["l"], lev_i["j"]
            l_f, j_f = lev_f["l"], lev_f["j"]
            is_proton = lev_i.get("is_proton", True)
            delta_l = abs(l_i - l_f)
            delta_j = abs(j_i - j_f)
            be2 = 0.0
            bm1 = 0.0
            if delta_l <= 2 and delta_j <= 2:
                be2 = B_E2(lev_i["wf"], lev_f["wf"], r_grid, l_i, j_i, l_f, j_f, is_proton)
            if l_i == l_f and delta_j <= 1 and (j_i + j_f) >= 1:
                bm1 = B_M1(l_i, j_i, l_f, j_f, is_proton)
            bw_e2 = weisskopf_estimate_E2(A)
            b_ratio = be2 / bw_e2 if bw_e2 > 0.0 else 0.0
            tau = transition_lifetime_s(be2, E_gamma, 2) if be2 > 0.0 else float("inf")
            out.append({
                "i": i_idx, "f": f_idx,
                "E_gamma_MeV": E_gamma,
                "B_E2_e2fm4": be2,
                "B_E2_WU": b_ratio,
                "B_M1_muN2": bm1,
                "lifetime_s": tau,
            })
    return out
