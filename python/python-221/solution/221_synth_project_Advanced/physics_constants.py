"""
physics_constants.py
====================

High-energy physics constants, kinematic thresholds, and coupling parameters
for the 2->3 scattering process simulation in the simplified QCD model.

All energies are in GeV, cross-sections in pb, angles in radians unless noted.

Scientific background:
----------------------
The strong coupling alpha_s(Q^2) runs according to the 1-loop beta function:

    alpha_s(Q^2) = (4 pi) / (beta0 * ln(Q^2 / Lambda_QCD^2))

where beta0 = 11 - 2*n_f/3 is the first beta-function coefficient
and Lambda_QCD ~ 0.250 GeV for n_f=5 active flavors.

The Mandelstam invariants for a 2->3 process p1 + p2 -> p3 + p4 + p5 satisfy:

    s + t1 + t2 + s34 + s45 + s35 = m1^2 + m2^2 + m3^2 + m4^2 + m5^2

where s = (p1+p2)^2 is the total CM energy squared.
"""

import math
from typing import Tuple

# ---------------------------------------------------------------------------
# Fundamental constants (PDG 2024 values)
# ---------------------------------------------------------------------------
HBAR_C = 0.1973269804           # hbar*c in GeV*fm
ALPHA_EM = 1.0 / 137.035999084  # fine-structure constant at zero momentum
GF = 1.1663788e-5               # Fermi constant in GeV^-2
SIN2THETA_W = 0.23122           # weak mixing angle
PI = math.pi
EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni constant

# ---------------------------------------------------------------------------
# QCD parameters
# ---------------------------------------------------------------------------
LAMBDA_QCD_5 = 0.250            # Lambda_QCD^(5) in GeV
NF_ACTIVE = 5                   # number of active flavors
BETA0_QCD = 11.0 - 2.0 * NF_ACTIVE / 3.0  # first beta-function coefficient

# Heavy quark thresholds
MC_QUARK = 1.27                 # charm mass in GeV
MB_QUARK = 4.18                 # bottom mass in GeV
MT_QUARK = 172.76               # top mass in GeV

# ---------------------------------------------------------------------------
# Electroweak boson masses
# ---------------------------------------------------------------------------
MZ_BOSON = 91.1876              # Z boson mass in GeV
MW_BOSON = 80.379               # W boson mass in GeV
MH_HIGGS = 125.10               # Higgs mass in GeV

# ---------------------------------------------------------------------------
# Proton / parton parameters
# ---------------------------------------------------------------------------
MP_PROTON = 0.938272            # proton mass in GeV


def alpha_s_1loop(q2: float) -> float:
    """
    1-loop running of the strong coupling alpha_s(Q^2).

    alpha_s(Q^2) = 4*pi / (beta0 * ln(Q^2 / Lambda_QCD^2))

    Parameters
    ----------
    q2 : float
        Momentum transfer squared in GeV^2. Must satisfy Q^2 > Lambda_QCD^2.

    Returns
    -------
    float
        Value of alpha_s(Q^2). Clamped to [1e-4, 4*pi] for safety.
    """
    lam2 = LAMBDA_QCD_5 ** 2
    if q2 <= lam2:
        q2 = lam2 + 1e-8
    arg = q2 / lam2
    if arg <= 1.0:
        arg = 1.0 + 1e-10
    val = (4.0 * PI) / (BETA0_QCD * math.log(arg))
    # Perturbativity bound: alpha_s < 4*pi/ beta0
    val = max(1e-4, min(val, 4.0 * PI / BETA0_QCD))
    return val


def alpha_s_2loop(q2: float) -> float:
    """
    2-loop running of alpha_s including beta1:

    alpha_s(Q^2) = (4*pi)/(beta0*L) * [1 - beta1/(beta0^2) * ln(L)/L]

    where L = ln(Q^2/Lambda^2), beta1 = 102 - 38*n_f/3.
    """
    lam2 = LAMBDA_QCD_5 ** 2
    if q2 <= lam2:
        q2 = lam2 + 1e-8
    arg = q2 / lam2
    if arg <= 1.0:
        arg = 1.0 + 1e-10
    L = math.log(arg)
    beta1 = 102.0 - 38.0 * NF_ACTIVE / 3.0
    a1 = (4.0 * PI) / (BETA0_QCD * L)
    correction = 1.0 - (beta1 / (BETA0_QCD ** 2)) * math.log(L) / L
    val = a1 * correction
    val = max(1e-4, min(val, 4.0 * PI / BETA0_QCD))
    return val


def mandelstam_s(sqrts: float) -> float:
    """
    Compute s = (p1+p2)^2 for a CM collision with energy sqrt(s).
    """
    if sqrts <= 0.0:
        return 0.0
    return sqrts * sqrts


def threshold_s_2to3(m3: float, m4: float, m5: float,
                     m1: float = MP_PROTON, m2: float = MP_PROTON) -> float:
    """
    Minimum value of s for a 2->3 process: s_th = (m3 + m4 + m5)^2.

    This is the kinematic threshold for producing particles of masses
    m3, m4, m5 in the final state.
    """
    return (m3 + m4 + m5) ** 2


def flux_factor(s: float, m1: float, m2: float) -> float:
    """
    Flux factor for 2->n processes:

    F = sqrt( (s - (m1+m2)^2)(s - (m1-m2)^2) ) / (2*sqrt(s))
    """
    if s <= (m1 + m2) ** 2:
        return 1e-10
    a = s - (m1 + m2) ** 2
    b = s - (m1 - m2) ** 2
    arg = a * b
    if arg < 0.0:
        arg = 0.0
    return math.sqrt(arg) / (2.0 * math.sqrt(s) + 1e-30)


def kallen_function(x: float, y: float, z: float) -> float:
    """
    Kallen (triangle) function:

    lambda(x, y, z) = x^2 + y^2 + z^2 - 2*x*y - 2*x*z - 2*y*z

    Used for 2-body phase space: lambda(s, m1^2, m2^2) >= 0 for physical region.
    """
    return x * x + y * y + z * z - 2.0 * x * y - 2.0 * x * z - 2.0 * y * z


def parton_luminosity_tau(s: float, m_final: float) -> float:
    """
    Parton luminosity variable tau = m_final^2 / s.
    For physical events, 0 < tau <= 1.
    """
    if s <= 0.0:
        return 1.0
    tau = m_final * m_final / s
    return max(0.0, min(tau, 1.0))


def color_factor_fundamental() -> float:
    """CF = (Nc^2 - 1)/(2*Nc) for SU(3): CF = 4/3."""
    nc = 3.0
    return (nc * nc - 1.0) / (2.0 * nc)


def color_factor_adjoint() -> float:
    """CA = Nc for SU(3): CA = 3."""
    return 3.0


def splitting_pqq(z: float) -> float:
    """
    Altarelli-Parisi quark->quark splitting function at LO:

    P_qq(z) = CF * (1 + z^2) / (1 - z)_+

    The + prescription is implemented by dropping the singular piece;
    we regularize with z in [eps, 1-eps].
    """
    cf = color_factor_fundamental()
    z = max(1e-8, min(z, 1.0 - 1e-8))
    return cf * (1.0 + z * z) / (1.0 - z)


def splitting_pgg(z: float) -> float:
    """
    Altarelli-Parisi gluon->gluon splitting function at LO:

    P_gg(z) = 2*CA * [ z/(1-z) + (1-z)/z + z*(1-z) ]
    """
    ca = color_factor_adjoint()
    z = max(1e-8, min(z, 1.0 - 1e-8))
    return 2.0 * ca * (z / (1.0 - z) + (1.0 - z) / z + z * (1.0 - z))
