# -*- coding: utf-8 -*-
"""
physics_constants.py
====================

Physical constants, dimensionless plasma parameters and equilibrium profile
constructors for the 1D slab-geometry gyrokinetic system.

Governing equations (Hasegawa-Mima / gyrokinetic Poisson limit)
---------------------------------------------------------------
The (linearised) gyrokinetic equation for species s in a slab with magnetic
field B = B0 e_z + (x/L_s) B0 e_y reads

    df0_s/dt + v_parallel b0 . grad <chi_s>_R
        = (e_s / T_s) <dF0_s/dt>_R  +  C[delta f_s]

with chi_s = phi - v_parallel A_parallel / c (electrostatic limit chi_s = phi),
<...>_R the gyroaverage at constant gyrocentre R, and C a linearised
pitch-angle scattering operator. The equilibrium is Maxwellian

    F0_s(x, v) = n0(x) (m_s / 2 pi T_s(x))^{3/2} exp(- m_s v^2 / 2 T_s(x))

and the logarithmic gradients entering the drive are

    omega_T_s  = (T_s / n_s) dn_s/dx * (L_n / T_s)       (density gradient)
    eta_s      = d ln T_s / d ln n_s                        (temperature ratio)

The normalisations used throughout the code are Bohm units:
    length        -> L_ref  = c_s / omega_ci
    time          -> 1 / omega_ci
    velocity      -> c_s = sqrt(T_e0 / m_i)
    potential     -> T_e0 / e
    distribution  -> n0 / c_s^3

References:
    [1] Frieman & Chen, Phys. Fluids 25, 502 (1982)
    [2] Hasegawa & Mima, Phys. Fluids 21, 87 (1978)
    [3] Rosenbluth & Hinton, Phys. Rev. Lett. 80, 724 (1998)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ----------------------------------------------------------------------------
# Fundamental physical constants (SI)
# ----------------------------------------------------------------------------
ELECTRON_CHARGE = 1.602176634e-19        # C
ELECTRON_MASS   = 9.1093837015e-31       # kg
PROTON_MASS     = 1.67262192369e-27      # kg
VACUUM_PERM     = 8.8541878128e-12       # F / m
VACUUM_PERME    = 1.25663706212e-6       # H / m
BOLTZMANN       = 1.380649e-23           # J / K
SPEED_OF_LIGHT  = 2.99792458e8           # m / s
PI              = math.pi
TWOPI           = 2.0 * PI
SQRTPI          = math.sqrt(PI)
SQRT2           = math.sqrt(2.0)
SQRT2PI         = math.sqrt(TWOPI)
EPS_SQRT        = np.sqrt(np.finfo(np.float64).eps)   # ~1.49e-8


# ----------------------------------------------------------------------------
# Derived / reference quantities
# ----------------------------------------------------------------------------
def ion_sound_speed(T_e_eV: float, m_i_amu: float = 1.0) -> float:
    """Ion sound speed  c_s = sqrt(T_e / m_i)  in m/s.

    T_e_eV   : electron temperature in electron-volts
    m_i_amu  : ion mass in atomic mass units (default = proton)
    """
    T_e_J = T_e_eV * ELECTRON_CHARGE
    m_i   = m_i_amu * PROTON_MASS
    if m_i <= 0.0 or T_e_J < 0.0:
        raise ValueError("ion_sound_speed: non-positive mass or temperature")
    return math.sqrt(T_e_J / m_i)


def ion_cyclotron_freq(B0: float, m_i_amu: float = 1.0, Z_i: float = 1.0) -> float:
    """Ion cyclotron frequency omega_ci = Z e B / m_i   (rad/s)."""
    if B0 <= 0.0:
        raise ValueError("ion_cyclotron_freq: B0 must be positive")
    return Z_i * ELECTRON_CHARGE * B0 / (m_i_amu * PROTON_MASS)


def gyroradius(T_i_eV: float, B0: float, m_i_amu: float = 1.0, Z_i: float = 1.0) -> float:
    """Thermal ion gyroradius rho_i = sqrt(T_i / m_i) / omega_ci   (m)."""
    if T_i_eV <= 0.0 or B0 <= 0.0:
        raise ValueError("gyroradius: T_i and B0 must be positive")
    T_i_J = T_i_eV * ELECTRON_CHARGE
    m_i   = m_i_amu * PROTON_MASS
    v_ti  = math.sqrt(T_i_J / m_i)
    w_ci  = Z_i * ELECTRON_CHARGE * B0 / m_i
    return v_ti / w_ci


def debye_length(T_e_eV: float, n_e_m3: float) -> float:
    """Electron Debye length lambda_De = sqrt(eps0 T_e / n_e e)."""
    if T_e_eV <= 0.0 or n_e_m3 <= 0.0:
        raise ValueError("debye_length: T_e, n_e must be positive")
    T_e_J = T_e_eV * ELECTRON_CHARGE
    return math.sqrt(VACUUM_PERM * T_e_J / (n_e_m3 * ELECTRON_CHARGE))


def plasma_frequency(n_m3: float, m_kg: float, Z: float = 1.0) -> float:
    """Plasma frequency omega_p = sqrt(n Z^2 e^2 / (eps0 m))."""
    if n_m3 <= 0.0 or m_kg <= 0.0:
        raise ValueError("plasma_frequency: non-positive n or m")
    return math.sqrt(n_m3 * (Z * ELECTRON_CHARGE) ** 2 / (VACUUM_PERM * m_kg))


def beta_plasma(n_e_m3: float, T_e_eV: float, T_i_eV: float, B0: float) -> float:
    """Total plasma beta = (n_e T_e + n_i T_i) / (B^2 / 2 mu0)."""
    p = n_e_m3 * ELECTRON_CHARGE * (T_e_eV + T_i_eV)     # Pa
    return p / (B0 * B0 / (2.0 * VACUUM_PERME))


def coulomb_logarithm(n_e_m3: float, T_e_eV: float) -> float:
    """Classical Coulomb logarithm ln_Lambda (NRL form).

        ln_Lambda = 23 - 0.5 ln(n_e [cm^-3] * 1e-6) + ln(T_e [eV])
    clamped to >= 2 for physical validity.
    """
    if n_e_m3 <= 0.0 or T_e_eV <= 0.0:
        return 10.0
    n_cgs = n_e_m3 * 1.0e-6
    lnL = 23.0 - 0.5 * math.log(max(n_cgs, 1.0)) + math.log(T_e_eV)
    return max(lnL, 2.0)


# ----------------------------------------------------------------------------
# Equilibrium profiles (radial coordinate x in [0, Lx])
# ----------------------------------------------------------------------------
@dataclass
class EquilibriumProfiles:
    """Analytic equilibrium profiles for slab ITG / ETG studies.

    Normalised such that rho_s = c_s / omega_ci = 1 in the code units.

    n0(x)     = n_ref * (1 - delta_n * x / Lx)
    T0_i(x)   = T_ref * (1 - eta_i * delta_n * x / Lx)   (eV)
    T0_e(x)   = T_ref * (1 - eta_e * delta_n * x / Lx)
    B0        : reference field (T)

    Gradient scales entering the gyrokinetic drive are

        1 / L_n    = - (1 / n0) dn0 / dx   =   delta_n / Lx
        1 / L_Ti   = - (1 / Ti) dTi / dx   =   eta_i delta_n / Lx
        omega_T*   = k_y rho_s c_s (1 + eta_i (v^2 / 2 v_th^2 - 3/2))
    """

    Lx: float
    n_ref: float = 1.0
    T_ref: float = 1.0
    B0: float = 1.0
    m_i_amu: float = 1.0
    Z_i: float = 1.0
    delta_n: float = 0.1
    eta_i: float = 3.0
    eta_e: float = 3.0
    k_y_rho: float = 0.3
    # Safety-factor-like magnetic shear enters via L_s:  b0 = e_z + (x/L_s) e_y
    L_s: float = 10.0

    # derived caches
    _c_s: float = field(init=False, repr=False)
    _omega_ci: float = field(init=False, repr=False)
    _rho_s: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._c_s = ion_sound_speed(self.T_ref, self.m_i_amu)
        self._omega_ci = ion_cyclotron_freq(self.B0, self.m_i_amu, self.Z_i)
        self._rho_s = self._c_s / self._omega_ci

    @property
    def c_s(self) -> float:
        return self._c_s

    @property
    def omega_ci(self) -> float:
        return self._omega_ci

    @property
    def rho_s(self) -> float:
        return self._rho_s

    # ---------------- profile callables ----------------
    def n0(self, x: np.ndarray) -> np.ndarray:
        """Equilibrium density profile (normalised)."""
        x = np.asarray(x, dtype=np.float64)
        n = self.n_ref * (1.0 - self.delta_n * x / self.Lx)
        # positivity floor
        return np.maximum(n, 1.0e-6 * self.n_ref)

    def T_i(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        T = self.T_ref * (1.0 - self.eta_i * self.delta_n * x / self.Lx)
        return np.maximum(T, 1.0e-6 * self.T_ref)

    def T_e(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        T = self.T_ref * (1.0 - self.eta_e * self.delta_n * x / self.Lx)
        return np.maximum(T, 1.0e-6 * self.T_ref)

    def L_n(self) -> float:
        """Density gradient scale length  L_n = - n0 / n0'."""
        return self.Lx / self.delta_n

    def L_Ti(self) -> float:
        return self.Lx / (self.eta_i * self.delta_n)

    def R_LTi(self) -> float:
        """Normalised inverse ion-temperature gradient  R / L_Ti, using R = L_s.

        This is the primary ITG-drive parameter.  Typical threshold
        (R/L_Ti)_c ~ 3--5 depending on geometry.
        """
        return abs(self.L_s) / self.L_Ti()

    def R_LTc(self) -> float:
        """Critical R/L_Ti for slab ITG onset (simple slab estimate).

        (R/L_Ti)_c  =  2 (1 + T_e/T_i) / (k_y rho_s)^2 * (L_n / L_s)^2  (rough)
        In slab geometry the real threshold also depends on the branch of
        the solution; this estimate captures the scaling.
        """
        tau = self.T_e(np.zeros(1))[0] / self.T_i(np.zeros(1))[0]
        kperp_rho = self.k_y_rho
        LnLs = self.Lx / (self.L_s * self.delta_n + 1.0e-30)
        return 2.0 * (1.0 + tau) / (kperp_rho * kperp_rho + 1.0e-30) * LnLs * LnLs

    def omega_star_i(self, k_y: Optional[float] = None) -> float:
        """Ion diamagnetic frequency (signed).

            omega*_i  =  - k_y rho_s c_s / L_n
        The sign encodes the ion drift direction in the slab.
        """
        if k_y is None:
            k_y = self.k_y_rho / self._rho_s
        return -k_y * self._rho_s * self._c_s / self.L_n()

    def drift_frequency(self, k_y: float, v_parallel: np.ndarray, v_th: float) -> np.ndarray:
        """Magnetic-drift frequency entering the slab gyrokinetic equation.

            omega_d(v)  =  k_y rho_s c_s (v_parallel^2 / v_th^2 - 1/2) / L_s
        """
        vv = v_parallel / max(v_th, 1.0e-30)
        return k_y * self._rho_s * self._c_s * (vv * vv - 0.5) / self.L_s

    def Maxwellian(self, v_parallel: np.ndarray, x0: float) -> np.ndarray:
        """Normalised 1D Maxwellian along v_parallel (integrated over v_perp).

            F0(v) = (1 / sqrt(pi) v_th) exp(- v^2 / v_th^2)
        """
        v_th = math.sqrt(2.0 * self.T_i(np.asarray([x0]))[0]
                         / (self.m_i_amu * PROTON_MASS)) / self._c_s
        vv = v_parallel / max(v_th, 1.0e-30)
        return np.exp(-vv * vv) / (math.sqrt(PI) * max(v_th, 1.0e-30))

    def drive_frequency(self, k_y: float, v_parallel: np.ndarray, x0: float) -> np.ndarray:
        """Combined diamagnetic + curvature drive for slab ITG.

            Omega_drive(v) = omega*_{ni} [ 1 + omega_Ti (v^2/v_th^2 - 3/2) ]
        with  omega*_{ni} = k_y rho_s c_s / L_n   and  omega_Ti = L_n / L_Ti = eta_i.
        """
        v_th = math.sqrt(2.0 * self.T_i(np.asarray([x0]))[0]
                         / (self.m_i_amu * PROTON_MASS)) / self._c_s
        vv = v_parallel / max(v_th, 1.0e-30)
        omstar = k_y * self._rho_s * self._c_s / self.L_n()
        return omstar * (1.0 + self.eta_i * (vv * vv - 1.5))


# ----------------------------------------------------------------------------
# Sanity self-check
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    eq = EquilibriumProfiles(Lx=10.0, T_ref=1000.0, B0=2.5)
    x = np.linspace(0.0, eq.Lx, 5)
    print("c_s   = {:.4e} m/s".format(eq.c_s))
    print("rho_s = {:.4e} m".format(eq.rho_s))
    print("L_n   = {:.3f}  L_Ti = {:.3f}".format(eq.L_n(), eq.L_Ti()))
    print("R/L_Ti = {:.3f}    (R/L_Ti)_c ~ {:.3f}".format(eq.R_LTi(), eq.R_LTc()))
    print("n0   :", eq.n0(x))
    print("T_i  :", eq.T_i(x))
    print("T_e  :", eq.T_e(x))
    print("Maxwellian at mid-radius:",
          eq.Maxwellian(np.linspace(-3, 3, 7), 0.5 * eq.Lx))
