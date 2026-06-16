"""
collective_modes.py  --  Nuclear surface oscillations and collective modes
========================================================================
Fused seeds:
    172_chladni_figures       -- nodal pattern computation
    1011_haeste_DiminishedRhythmsPathology -- rhythm / bandpass analysis
    908_predator_prey_ode     -- coupled-oscillator rate equations

Models the Bohr-Mottelson collective Hamiltonian for nuclear surface
vibrations. In the harmonic limit, the nuclear surface is parametrised as

    R(theta, phi) = R_0 [ 1 + sum_{L M} alpha_{L M} Y_L^M(theta, phi) ]

and the collective Hamiltonian is

    H_coll = sum_{L M} [ B_L |dot{alpha}_{L M}|^2 + C_L |alpha_{L M}|^2 ] / 2

where B_L is the mass parameter and C_L the stiffness. The quantised
excitation energies are

    E(n, L) = hbar omega_L (n + L/2 + 3/2)

    hbar omega_L = hbar sqrt(C_L / B_L)

We compute:
  1. Surface-oscillation frequencies for L = 2 (quadrupole) and L = 3 (octupole).
  2. Nodal patterns (Chladni-like) of the surface displacement.
  3. Decay rhythm of a collective phonon via a damped-oscillator ODE.
  4. Power spectrum of the surface oscillation signal.

References:
    Bohr & Mottelson, Mat. Fys. Medd. 27 no. 16 (1953)
    Ring & Schuck, The Nuclear Many-Body Problem (1980), Ch. 8
"""

from __future__ import annotations
import math
import numpy as np
from nuclear_constants import HBAR, HBAR_C, M_PROTON, M_NEUTRON, R0_FM, PI, R_EPSILON
from special_functions_nuclear import spherical_harmonic


# ======================================================================
#  Mass and stiffness parameters (liquid-drop + shell correction)
# ======================================================================
def mass_parameter_B(L: int, A: int) -> float:
    r"""Collective mass parameter B_L in MeV (fm/c)^{-2} units.

    Liquid-drop estimate:
        B_L = (1 / (4 pi)) * (m A R_0^2) / (L (2L + 1))

    With m = (M_p + M_n)/2 the average nucleon mass and R_0 in fm,
    B_L has units MeV * fm^{-2} * (fm / c)^2 = MeV / c^2.
    We express in MeV (fm / c)^2 by multiplying by c^2 / hbar^2... actually
    we use the dimensionful B_L with hbar omega = hbar sqrt(C / B).
    """
    m_avg = 0.5 * (M_PROTON + M_NEUTRON)  # MeV/c^2
    R = R0_FM * (A ** (1.0 / 3.0))
    return (1.0 / (4.0 * PI)) * (m_avg * A * R * R) / (L * (2.0 * L + 1.0))


def stiffness_C(L: int, A: int, Z: int) -> float:
    r"""Collective stiffness C_L in MeV fm^{-2}.

    Liquid-drop model:
        C_L = (L - 1)(L + 2) * [ sigma_s A^{2/3}
                - (3 / (2 pi)) * (Z^2 e^2) / (R_0 A^{1/3}) * (1 / (2 L + 1)) ]

    where sigma_s ~ 17 MeV is the surface tension.
    """
    sigma_s = 17.0  # MeV
    R = R0_FM * (A ** (1.0 / 3.0))
    e2 = 1.43997643  # MeV fm
    term1 = sigma_s * (A ** (2.0 / 3.0))
    term2 = (3.0 / (2.0 * PI)) * (Z * Z * e2) / (R * (2.0 * L + 1.0))
    return (L - 1.0) * (L + 2.0) * (term1 - term2)


def collective_frequency(L: int, A: int, Z: int) -> float:
    r"""hbar omega_L = hbar sqrt(C_L / B_L) in MeV."""
    B = mass_parameter_B(L, A)
    C = stiffness_C(L, A, Z)
    if C <= 0.0 or B <= 0.0:
        return 0.0
    omega = math.sqrt(C / B)  # 1 / (fm / c)
    # hbar omega in MeV: hbar c * omega / c = hbar * omega
    # Since B is in MeV/c^2 * fm^2 and C is in MeV / fm^2, omega has units 1 / (fm / c).
    # hbar omega = hbar c * (omega / c) = HBAR_C * omega / c
    # With omega in fm^{-1} * c, we have hbar omega = HBAR_C * omega
    return HBAR_C * omega  # MeV


def phonon_energy(n_phonons: int, L: int, A: int, Z: int) -> float:
    r"""E(n, L) = hbar omega_L (n + 3/2) for the L-pole phonon."""
    return collective_frequency(L, A, Z) * (n_phonons + 1.5)


# ======================================================================
#  Nodal patterns of the collective surface (Chladni-like)
# ======================================================================
def surface_displacement(
    L: int, M: int, amplitude: float, theta: np.ndarray, phi: np.ndarray,
) -> np.ndarray:
    """Compute Re[ amplitude * Y_L^M(theta, phi) ] on a meshgrid.

    Returns a 2-D array of surface displacement values (dimensionless).
    """
    out = np.zeros_like(theta, dtype=np.float64)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            ylm = spherical_harmonic(L, M, float(theta[i, j]), float(phi[i, j]))
            out[i, j] = amplitude * ylm.real
    return out


def chladni_nodal_count(L: int, M: int, n_theta: int = 40, n_phi: int = 40) -> dict:
    """Count the number of nodal lines of |Y_L^M|^2 on the sphere.

    For Y_L^M the number of nodal lines in theta is L - |M| and in phi is 2 |M|
    (for M != 0). Returns a dict with counts and a sampled |Y|^2 array.
    """
    theta = np.linspace(0.0, PI, n_theta)
    phi = np.linspace(0.0, 2.0 * PI, n_phi, endpoint=False)
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")
    disp = surface_displacement(L, M, 1.0, THETA, PHI)
    power = disp * disp
    # Count sign changes along theta and phi directions
    n_theta_nodes = int(np.sum(np.abs(np.diff(np.sign(disp), axis=0)) > 0) // 2)
    n_phi_nodes = int(np.sum(np.abs(np.diff(np.sign(disp), axis=1)) > 0) // 2)
    return {
        "L": L, "M": M,
        "expected_theta_nodes": L - abs(M),
        "expected_phi_nodes": 2 * abs(M) if M != 0 else 0,
        "counted_theta_nodes": n_theta_nodes,
        "counted_phi_nodes": n_phi_nodes,
        "max_power": float(np.max(power)),
        "mean_power": float(np.mean(power)),
    }


# ======================================================================
#  Damped collective oscillator (rhythm analysis, 1011 seed)
# ======================================================================
def damped_collective_evolution(
    L: int, A: int, Z: int,
    gamma_damp: float = 0.5,     # MeV damping rate
    alpha0: float = 1.0,         # initial amplitude
    beta0: float = 0.0,          # initial velocity
    t_max_fm: float = 200.0,     # integration time in fm/c
    dt_fm: float = 0.1,          # fm/c
) -> dict:
    """Time-evolve a damped collective oscillator:

        alpha''(t) + 2 gamma alpha'(t) + omega^2 alpha(t) = 0

    using the midpoint method. Returns time series and spectral content.
    """
    hbar_omega = collective_frequency(L, A, Z)
    if hbar_omega <= 0.0:
        return {"t": np.array([0.0]), "alpha": np.array([alpha0]), "power": np.array([0.0])}
    omega = hbar_omega / HBAR_C  # fm^{-1}
    n_steps = int(t_max_fm / dt_fm)
    t = np.linspace(0.0, t_max_fm, n_steps)
    alpha = np.zeros(n_steps)
    beta = np.zeros(n_steps)
    alpha[0] = alpha0
    beta[0] = beta0
    for i in range(1, n_steps):
        # Midpoint
        k1_a = beta[i - 1]
        k1_b = -2.0 * gamma_damp * beta[i - 1] - omega * omega * alpha[i - 1]
        a_mid = alpha[i - 1] + 0.5 * dt_fm * k1_a
        b_mid = beta[i - 1] + 0.5 * dt_fm * k1_b
        k2_a = b_mid
        k2_b = -2.0 * gamma_damp * b_mid - omega * omega * a_mid
        alpha[i] = alpha[i - 1] + dt_fm * k2_a
        beta[i] = beta[i - 1] + dt_fm * k2_b
    # Power spectrum via FFT
    freq = np.fft.rfftfreq(n_steps, d=dt_fm)
    spectrum = np.abs(np.fft.rfft(alpha)) ** 2
    return {
        "t_fm_c": t,
        "alpha": alpha,
        "beta": beta,
        "hbar_omega_MeV": hbar_omega,
        "omega_fm_inv": omega,
        "frequency_fm_inv": freq,
        "power_spectrum": spectrum,
        "peak_frequency_fm_inv": float(freq[np.argmax(spectrum[1:]) + 1]) if len(spectrum) > 1 else 0.0,
    }


# ======================================================================
#  Rate-equation cascade among collective phonons (908 seed)
# ======================================================================
def phonon_rate_equation(
    n_levels: int, omega_MeV: float, gamma_down: float, gamma_up: float,
    t_max: float = 50.0, dt: float = 0.1,
) -> dict:
    """Integrate a rate equation for phonon-number populations P(n):

        dP(n)/dt = gamma_down [(n+1) P(n+1) - n P(n)]
                 + gamma_up   [n P(n-1) - (n+1) P(n)]

    (predator-prey style with up/down transition rates).
    """
    P = np.zeros(n_levels)
    P[0] = 1.0  # start in ground state
    n_steps = int(t_max / dt)
    hist = np.zeros((n_steps, n_levels))
    hist[0, :] = P
    for step in range(1, n_steps):
        dP = np.zeros(n_levels)
        for n in range(n_levels):
            # Down from n+1 to n
            if n + 1 < n_levels:
                dP[n] += gamma_down * (n + 1) * P[n + 1]
                dP[n + 1] -= gamma_down * (n + 1) * P[n + 1]
            # Up from n-1 to n
            if n - 1 >= 0:
                dP[n] += gamma_up * n * P[n - 1]
                dP[n - 1] -= gamma_up * n * P[n - 1]
            # Loss from n to n-1
            if n > 0:
                dP[n] -= gamma_down * n * P[n]
            # Loss from n to n+1
            if n + 1 < n_levels:
                dP[n] -= gamma_up * (n + 1) * P[n]
        P = P + dt * dP
        P = np.maximum(P, 0.0)
        P = P / max(np.sum(P), R_EPSILON)
        hist[step, :] = P
    return {"history": hist, "dt": dt, "n_levels": n_levels}
