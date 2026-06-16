# -*- coding: utf-8 -*-
"""
Physical constants, units and dimensionless coupling conventions
used throughout the quantum-phase-transition finite-size-scaling
code base.  All quantities are expressed in natural units with
hbar = k_B = 1 unless explicitly noted.

Derived from the modular physical-quantity philosophy of the
Venus O2 redox code (aowarren/Venus_O2): every derived number is
built from a small set of immutable SI anchors so that round-off
and unit drift are eliminated a priori.
"""

from __future__ import annotations
import math

# ---------------------------------------------------------------------------
# SI anchors  (CODATA 2018 recommended values)
# ---------------------------------------------------------------------------
HBAR_SI: float = 1.054_571_817e-34        # J s
K_B_SI: float = 1.380_649e-23             # J K^-1
MU_B_SI: float = 9.274_010_078e-24        # J T^-1
MU_0_SI: float = 1.256_637_062e-6         # N A^-2
EPS_0_SI: float = 8.854_187_812e-12       # F m^-1
E_CHARGE_SI: float = 1.602_176_634e-19    # C
M_E_SI: float = 9.109_383_701e-31         # kg
A_BOHR_SI: float = 5.291_772_109_03e-11   # m
PI: float = math.pi
TAU: float = 2.0 * PI
EULER_GAMMA: float = 0.577_215_664_901_532_8606
GLAISHER: float = 1.282_427_129_100_622_6368

# ---------------------------------------------------------------------------
# Lattice / model scales  (transverse-field Ising model)
# ---------------------------------------------------------------------------
# We choose a reference exchange J0 as energy unit.  All energies
# below are reported in units of J0; all temperatures in units of
# J0 / k_B.  This mirrors the dimensionless reduction used in
# planetary geochemistry redox networks.
J0_DEFAULT: float = 1.0       # reference Ising exchange (model unit)
LATTICE_SPACING_A0: float = 1.0   # Angstrom-like reference

# ---------------------------------------------------------------------------
# Critical-point prior knowledge for the 1D TFIM
# ---------------------------------------------------------------------------
#  H = - J sum sigma^z_i sigma^z_{i+1} - h sum sigma^x_i
#  Quantum critical point: h_c / J = 1  (exact, Lieb-Schultz-Mattis).
#  Critical exponents  nu = 1,  beta = 1/8,  gamma = 7/4,  z = 1,
#  alpha = 0  (log divergence of C),  eta = 1/4.
TFIM_HC_OVER_J: float = 1.0
NU_TFIM: float = 1.0
BETA_TFIM: float = 0.125
GAMMA_TFIM: float = 1.75
ETA_TFIM: float = 0.25
Z_DYN_TFIM: float = 1.0

# ---------------------------------------------------------------------------
# Universality-class constants used in FSS formulae
# ---------------------------------------------------------------------------
def binder_cumulant_ising_2d() -> float:
    """Approximate universal Binder ratio U* for the 2D Ising class.
    U4 / <m^2>^2 -> 1 - <m^4>/(3 <m^2>^2) at criticality.
    The value 0.6107 is the accepted numerical estimate."""
    return 0.6107


def central_charge_c_minimal(p: int, q: int) -> float:
    """Central charge of the (p,q) Virasoro minimal model.
    C = 1 - 6 (p-q)^2 / (p q).
    For the Ising model (p,q) = (4,3) one recovers c = 1/2."""
    if p <= 0 or q <= 0:
        raise ValueError("p and q must be positive integers")
    return 1.0 - 6.0 * (p - q) ** 2 / (p * q)


CENTRAL_CHARGE_ISING: float = central_charge_c_minimal(4, 3)


def correlation_length_amplitude_xi0(d: int = 1) -> float:
    """Non-universal amplitude xi0 entering xi = xi0 |t|^{-nu}.
    For the 1D TFIM at T=0 the exact amplitude is xi0 = 1/(2 J)."""
    return 0.5 / J0_DEFAULT


def fidelity_susceptibility_prefactor(chi_fs_0: float = 1.0) -> float:
    """Prefactor of the logarithmic divergence of the fidelity
    susceptibility at the QCP:  chi_F ~ chi_fs_0 * ln(L)."""
    return chi_fs_0


# ---------------------------------------------------------------------------
# Numerical-safety floors (robustness)
# ---------------------------------------------------------------------------
EPS_NUM: float = 1.0e-14          # generic floor for denominators
SAFE_LOG_FLOOR: float = 1.0e-300  # floor before log()
SQRT_EPS: float = math.sqrt(EPS_NUM)
