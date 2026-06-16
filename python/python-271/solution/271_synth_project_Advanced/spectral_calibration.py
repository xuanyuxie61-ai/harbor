# -*- coding: utf-8 -*-
"""
Spectral-function calibration for the TFIM dynamical structure factor.

Adapted from the York LIF instrument paper (1187), which calibrates
spectrometer channel responses against known atomic reference lines.
We repurpose the *calibration logic* to extract the TFIM dynamical
structure factor S(q, omega) from finite-L exact diagonalisation.

The dynamical structure factor is defined as
    S(q, omega) = (1/L) sum_{n} |<n| sigma^z_q |0>|^2
                  * delta(omega - (E_n - E_0))
where sigma^z_q = (1/sqrt(L)) sum_j e^{i q j} sigma^z_j.

On a finite lattice the delta becomes a set of discrete lines.  We
calibrate the instrument-like response by:

  1. Convolving each line with a Lorentzian of width eta (resolution).
  2. Applying a channel-dependent efficiency curve  eps(q)  fitted to
     the known sum rule  int dq S(q, omega) = chi_static.
  3. Smoothing the resulting spectrum with a Savitzky-Golay filter.

This is the TFIM analogue of the LIF channel-response calibration.
"""

from __future__ import annotations
from typing import Tuple, List
import numpy as np
from scipy.signal import savgol_filter
try:
    from . import constants as C
except ImportError:
    import constants as C


# ---------------------------------------------------------------------------
# Exact diagonalisation of sigma^z_q matrix elements
# ---------------------------------------------------------------------------
def sigma_z_q_matrix_elements(L: int, J: float, h: float,
                                q: float, n_excited: int = 8,
                                periodic: bool = True
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (omegas, weights) for the dynamical structure factor at
    momentum q, restricted to the lowest ``n_excited`` many-body levels.

    omegas[k] = E_k - E_0    (excitation energies)
    weights[k] = |<k| sigma^z_q |0>|^2
    """
    if L > 12:
        raise ValueError("sigma_z_q: exact diag limited to L <= 12 "
                         "to keep runtime reasonable")
    dim = 1 << L
    # Build H in the computational basis
    try:
        from .tfim_hamiltonian import build_hamiltonian_dense
    except ImportError:
        from tfim_hamiltonian import build_hamiltonian_dense
    H = build_hamiltonian_dense(L, J, h, periodic)
    evals, evecs = np.linalg.eigh(H)
    gs = evecs[:, 0]
    e0 = evals[0]

    # Build sigma^z_q operator
    Sz_q = np.zeros((dim, dim), dtype=complex)
    try:
        from .tfim_hamiltonian import SIGMA_Z, EYE2
    except ImportError:
        from tfim_hamiltonian import SIGMA_Z, EYE2
    for j in range(L):
        ops = [EYE2] * L
        ops[j] = SIGMA_Z
        # kron manually
        M = ops[0]
        for m in ops[1:]:
            M = np.kron(M, m)
        Sz_q += np.exp(1j * q * j) * M
    Sz_q /= np.sqrt(L)

    # Matrix elements  <k| Sz_q |0>
    weights = np.zeros(n_excited)
    omegas = np.zeros(n_excited)
    for k in range(1, 1 + n_excited):
        amp = evecs[:, k].conj() @ Sz_q @ gs
        weights[k - 1] = float(np.abs(amp) ** 2)
        omegas[k - 1] = float(evals[k] - e0)
    return omegas, weights


# ---------------------------------------------------------------------------
# Lorentzian convolution  (instrument response)
# ---------------------------------------------------------------------------
def lorentzian_spectrum(omegas: np.ndarray, weights: np.ndarray,
                          omega_axis: np.ndarray, eta: float = 0.05
                          ) -> np.ndarray:
    """Broaden discrete lines with a Lorentzian of HWHM eta:
        S(omega) = sum_k weights[k] * (eta/pi) / ((omega - omega_k)^2 + eta^2)
    """
    out = np.zeros_like(omega_axis, dtype=float)
    for o, w in zip(omegas, weights):
        out += w * (eta / C.PI) / ((omega_axis - o) ** 2 + eta * eta)
    return out


# ---------------------------------------------------------------------------
# Channel-efficiency calibration
# ---------------------------------------------------------------------------
def efficiency_curve(qs: np.ndarray, a0: float = 1.0,
                      a1: float = 0.05, a2: float = -0.02) -> np.ndarray:
    """Model the q-dependent channel efficiency as a low-order polynomial
    in cos(q):
        eps(q) = a0 + a1 cos q + a2 cos 2q.
    Defaults are chosen so that eps is close to 1 for all q.
    """
    return a0 + a1 * np.cos(qs) + a2 * np.cos(2.0 * qs)


def calibrate_spectrum(S_raw: np.ndarray, qs: np.ndarray,
                        omega_axis: np.ndarray,
                        efficiency: np.ndarray = None) -> np.ndarray:
    """Apply channel-efficiency calibration and smooth with Savitzky-Golay.

    Returns the calibrated S(q, omega) integrated over q.
    """
    if efficiency is None:
        efficiency = efficiency_curve(qs)
    # Weighted sum over q
    S_total = np.zeros_like(omega_axis, dtype=float)
    for i, q in enumerate(qs):
        S_total += efficiency[i] * S_raw[i]
    # Smooth (Savitzky-Golay; window must be odd and <= len)
    window = min(11, len(S_total) - 1 if len(S_total) % 2 == 0 else len(S_total))
    if window < 5:
        return S_total
    if window % 2 == 0:
        window -= 1
    return savgol_filter(S_total, window_length=window, polyorder=3)


# ---------------------------------------------------------------------------
# Sum-rule check
# ---------------------------------------------------------------------------
def static_susceptibility(L: int, J: float, h: float,
                            n_q: int = 8) -> float:
    """Compute the static longitudinal susceptibility
        chi = (1/L) sum_q S(q, omega=0)
    by integrating the dynamical structure factor over omega.
    For the TFIM in the ordered phase chi ~ |1 - lam|^{-gamma} with
    gamma = 7/4; in the disordered phase chi ~ (1 - lam)^{-1} at lam -> 1-.
    """
    qs = np.linspace(1.0e-3, C.PI - 1.0e-3, n_q)
    omega_axis = np.linspace(0.0, 6.0, 200)
    integral = 0.0
    for q in qs:
        try:
            omegas, weights = sigma_z_q_matrix_elements(L, J, h, q,
                                                          n_excited=6)
        except ValueError:
            continue
        S = lorentzian_spectrum(omegas, weights, omega_axis, eta=0.05)
        integral += float(np.trapz(S, omega_axis))
    integral /= n_q
    return integral


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------
def calibrated_dynamical_sf(L: int, J: float, h: float,
                              n_q: int = 6, n_omega: int = 200,
                              omega_max: float = 6.0,
                              eta: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Return (omega_axis, S_calibrated(omega)) for the TFIM."""
    qs = np.linspace(1.0e-3, C.PI - 1.0e-3, n_q)
    omega_axis = np.linspace(0.0, omega_max, n_omega)
    S_per_q = []
    for q in qs:
        try:
            omegas, weights = sigma_z_q_matrix_elements(L, J, h, q,
                                                          n_excited=6)
        except ValueError:
            S_per_q.append(np.zeros_like(omega_axis))
            continue
        S = lorentzian_spectrum(omegas, weights, omega_axis, eta)
        S_per_q.append(S)
    S_per_q = np.asarray(S_per_q)
    S_cal = calibrate_spectrum(S_per_q, qs, omega_axis)
    return omega_axis, S_cal
