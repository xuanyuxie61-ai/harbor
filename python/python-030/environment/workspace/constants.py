# -*- coding: utf-8 -*-
"""
constants.py
============
Nuclear physics fundamental constants and conversion factors.

All constants are given in SI or natural units appropriate for
low-energy nuclear structure calculations (MeV, fm, seconds).
"""

import numpy as np

# ------------------------------------------------------------------
#  Physical constants
# ------------------------------------------------------------------
HBAR_C         = 197.3269804       # MeV·fm
MASS_NUCLEON   = 938.918           # MeV/c^2  (average nucleon mass)
MASS_PROTON    = 938.2720813       # MeV/c^2
MASS_NEUTRON   = 939.5654133       # MeV/c^2
ELECTRON_MASS  = 0.5109989461      # MeV/c^2
SPEED_OF_LIGHT = 299792458.0       # m/s
FINE_STRUCTURE = 1.0 / 137.035999084

# ------------------------------------------------------------------
#  Conversion factors
# ------------------------------------------------------------------
FM_TO_M        = 1.0e-15           # 1 fm = 1e-15 m
MEV_TO_JOULE   = 1.602176634e-13   # 1 MeV = 1.602e-13 J
SECOND_TO_FS   = 1.0e15            # 1 s = 1e15 fs

# ------------------------------------------------------------------
#  Nuclear liquid-drop model coefficients (MeV)
# ------------------------------------------------------------------
LDA_VOLUME     = 15.75
LDA_SURFACE    = 17.8
LDA_COULOMB    = 0.711
LDA_ASYMMETRY  = 23.7
LDA_PAIRING    = 11.18

# ------------------------------------------------------------------
#  Woods-Saxon default parameters
# ------------------------------------------------------------------
WS_V0          = -51.0             # MeV, central depth
WS_R0          = 1.27              # fm, radius parameter
WS_A           = 0.67              # fm, diffuseness
WS_VSO         = -0.44 * WS_V0     # MeV, spin-orbit depth
WS_RSO         = 1.27              # fm, spin-orbit radius
WS_ASO         = 0.67              # fm, spin-orbit diffuseness

# ------------------------------------------------------------------
#  Pairing strength (schematic BCS)
# ------------------------------------------------------------------
G_PAIRING      = 25.0 / 41.0       # MeV, empirical G ≈ 25/A MeV

# ------------------------------------------------------------------
#  Quadrature / numerical defaults
# ------------------------------------------------------------------
DEFAULT_FEKETE_DEGREE = 7
DEFAULT_SPARSE_LEVEL  = 4

# ------------------------------------------------------------------
#  Helper: reduced mass
# ------------------------------------------------------------------
def reduced_mass(m1, m2):
    """
    Reduced mass μ = m1*m2 / (m1+m2).

    Parameters
    ----------
    m1, m2 : float
        Masses in MeV/c^2.

    Returns
    -------
    mu : float
        Reduced mass in MeV/c^2.
    """
    return (m1 * m2) / (m1 + m2)


def hbar2_over_2m(mass=MASS_NUCLEON):
    r"""
    Compute :math:`\hbar^2 / (2m)` in MeV·fm².

    From :math:`\hbar c = 197.327` MeV·fm and mass in MeV/c²:

    .. math::
        \frac{\hbar^2}{2m}
        = \frac{(\hbar c)^2}{2 m c^2}
        \;\;[\text{MeV·fm}^2]

    Parameters
    ----------
    mass : float, optional
        Nucleon mass in MeV/c².  Default is average nucleon mass.

    Returns
    -------
    val : float
        :math:`\hbar^2/(2m)` in MeV·fm².
    """
    return (HBAR_C ** 2) / (2.0 * mass)
