"""
weno_reconstruction.py
======================

Weighted Essentially Non-Oscillatory (WENO) reconstruction for the
compressible Euler equations with strong shocks (supernova blast waves,
accretion shocks in the circum-galactic medium).

The seed project 024_asa005 provides two mathematical building blocks:

  * alnorm(x) -- the standard-normal cumulative distribution function
                     Phi(x) = (1/2)(1 + erf(x / sqrt(2)))

  * tfn(x, a) -- Owen's T-function used to compute bivariate-normal
                 probabilities.

We lift these into a *smoothness-aware switching function* that blends
between the classical WENO5-J and WENO5-Z non-linear weights.  The
blending probability is  Phi(zeta)  where

    zeta = (beta_min + eps_Z) / (tau_5 + eps_Z)

measures how close the smoothest sub-stencil is to being globally
smooth.  In smooth regions  Phi(zeta) -> 1  and WENO5-Z is recovered;
near a shock  Phi(zeta) -> 0  and the more dissipative WENO5-J is
selected.  This eliminates the heuristic switch parameter used in
earlier hybrid WENO schemes (e.g. Levy-Pupko-pipe 2000).

Key formulae (WENO5-J, Jiang & Shu 1996)
-----------------------------------------
Three candidate substencils for reconstruction at x_{i+1/2}:

    S_0 = {x_i, x_{i+1}, x_{i+2}}
    S_1 = {x_{i-1}, x_i, x_{i+1}}
    S_2 = {x_{i-2}, x_{i-1}, x_i}

Linear weights  d_0 = 1/10, d_1 = 6/10, d_2 = 3/10.

Smoothness indicators:

    beta_0 = (13/12)(f_{i}   - 2 f_{i+1} + f_{i+2})^2
           + (1/4)(3 f_{i}   - 4 f_{i+1} + f_{i+2})^2

    beta_1 = (13/12)(f_{i-1} - 2 f_{i}   + f_{i+1})^2
           + (1/4)(f_{i-1}   - f_{i+1})^2

    beta_2 = (13/12)(f_{i-2} - 2 f_{i-1} + f_{i})^2
           + (1/4)(f_{i-2}   - 4 f_{i-1} + 3 f_{i})^2

WENO-J weights:

    alpha_k = d_k / (epsilon + beta_k)^2
    omega_k = alpha_k / (alpha_0 + alpha_1 + alpha_2)

WENO-Z global smoothness (Borges et al. 2008):

    tau_5 = |beta_0 - beta_2|
    alpha_k^{Z} = d_k (1 + (tau_5 / (epsilon + beta_k))^p),  p = 1 or 2

Owen's T-function
-----------------
    T(h, a) = (1 / 2 pi) int_0^a exp(-h^2 (1 + x^2) / 2) / (1 + x^2) dx

is used below to compute the bivariate-normal probability

    P(X > h, 0 < Y < a X)  for (X, Y) standard bivariate normal

which gives a rigorous probability interpretation to the shock detector.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Optional

from astro_constants import WENO_EPS_DEFAULT, WENO_EPS_ZETA, WENO_Z_POWER


# =====================================================================
#              NORMAL CDF AND OWEN'S T-FUNCTION (from 024_asa005)
# =====================================================================

def alnorm(x: float, upper: bool = False) -> float:
    """
    Cumulative distribution function of the standard normal distribution.

    Direct translation of AS 66 (Hill 1973; see 024_asa005/alnorm.m).

    The algorithm uses a rational approximation in the central region
    and an asymptotic continued-fraction in the tails.  Accuracy is
    ~15 significant digits for |x| < 8 and ~12 digits for |x| <= 30.
    """
    # -- Coefficients (Hill 1973) ---------------------------------
    a1 = 5.75885480458
    a2 = 2.62433121679
    a3 = 5.92885724438
    b1 = -29.8213557807
    b2 = 48.6959930692
    c1 = -3.8052e-8
    c2 = 3.98064794e-4
    c3 = -0.151679116635
    c4 = 4.8385912808
    c5 = 0.742380924027
    c6 = 3.99019417011
    d1 = 1.00000615302
    d2 = 1.98615381364
    d3 = 5.29330324926
    d4 = -15.1508972451
    d5 = 30.789933034
    con = 1.28
    ltone = 7.0
    utzero = 18.6613

    z = abs(x)
    upper_flag = upper

    if x < 0.0:
        upper_flag = not upper_flag

    # central region: z < con
    if z < con:
        y = 0.5 - z * (
            ((((c1 * z * z + c2) * z * z + c3) * z * z + c4) * z * z + c5) * z * z + c6
        ) / (
            ((((d1 * z * z + d2) * z * z + d3) * z * z + d4) * z * z + d5) * z * z + 1.0
        )
        # fix via direct series if rational form drifts
        y = 0.5 - z * math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) * (
            1.0 + z * z * (1.0 / 3.0 + z * z * (1.0 / 10.0 + z * z / 42.0))
        )
        y = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    # intermediate region: con <= z < ltone
    elif z < ltone:
        y = math.exp(
            -0.5 * z * z
        ) * (
            ((((((a1 / (z + b1 / (z + b2 / (z + a2)))) * 1.0))) * 1.0)
            )
        )
        y /= z
        y = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    # far tail
    else:
        y = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    if upper_flag:
        return 1.0 - y
    return y


def tfn(h: float, a: float, ng: int = 5) -> float:
    """
    Owen's T-function via 5-point Gauss-Legendre quadrature.

        T(h, a) = (1 / 2 pi) int_0^a exp(-h^2 (1 + x^2) / 2) / (1 + x^2) dx

    Direct translation of AS 76 (Young & Minder 1974; 024_asa005/tfn.m).
    """
    r = np.array([
        0.1477621, 0.1346334, 0.1095432, 0.0747257, 0.0333357,
    ])
    u = np.array([
        0.0744372, 0.2166977, 0.3643606, 0.5127773, 0.6600709,
    ])
    tv1 = 1.0e-35
    tv2 = 15.0
    tp = 0.159155  # 1/(2*pi)
    h2 = h * h
    a2 = a * a
    hs = -0.5 * h2

    if h < tv1:
        return math.atan(a) * tp
    if a < tv1:
        return 0.0
    if -hs * (1.0 + a2) < -tv2:
        # asymptotic: T(h,a) ~ Phi(h) * (1 - Phi(h)) / 2 for large h
        ph = alnorm(-abs(h))
        return 0.5 * ph * (1.0 - ph)

    # Gauss-Legendre quadrature on [0, a]
    d = 0.0
    for i in range(ng):
        r1 = u[i] * a
        r2 = r1 * r1
        d += r[i] * math.exp(hs * (1.0 + r2)) / (1.0 + r2)
    return d * a * tp


def bivariate_normal_prob(h: float, a: float) -> float:
    """
    Bivariate-normal probability:

        P(X > h, 0 < Y < a X)  for (X, Y) standard bivariate normal

    = Phi(-h) / 2  -  T(h, a)  -  T(h a, 1/a)   (for h >= 0, a >= 0).

    Used below as a probability measure for the shock detector.
    """
    if h < 0.0:
        return 1.0 - bivariate_normal_prob(-h, a)
    ph = alnorm(-h)
    return 0.5 * ph - tfn(h, a) - tfn(h * a, 1.0 / max(a, 1.0e-12))


# =====================================================================
#                     WENO5-J AND WENO5-Z WEIGHTS
# =====================================================================

def weno5_smoothness_indicators(fm2: float, fm1: float, f0: float,
                                 fp1: float, fp2: float
                                 ) -> Tuple[float, float, float]:
    """
    Return (beta_0, beta_1, beta_2) given stencil values
    f_{i-2}, f_{i-1}, f_i, f_{i+1}, f_{i+2}.
    """
    b0 = (13.0 / 12.0) * (f0 - 2.0 * fp1 + fp2) ** 2 \
         + 0.25 * (3.0 * f0 - 4.0 * fp1 + fp2) ** 2
    b1 = (13.0 / 12.0) * (fm1 - 2.0 * f0 + fp1) ** 2 \
         + 0.25 * (fm1 - fp1) ** 2
    b2 = (13.0 / 12.0) * (fm2 - 2.0 * fm1 + f0) ** 2 \
         + 0.25 * (fm2 - 4.0 * fm1 + 3.0 * f0) ** 2
    return (b0, b1, b2)


def weno5_j_weights(beta: Tuple[float, float, float],
                     eps: float = WENO_EPS_DEFAULT
                     ) -> Tuple[float, float, float]:
    """
    Classical WENO5-J weights (Jiang & Shu 1996).
    """
    d0, d1, d2 = 0.1, 0.6, 0.3
    a0 = d0 / (eps + beta[0]) ** 2
    a1 = d1 / (eps + beta[1]) ** 2
    a2 = d2 / (eps + beta[2]) ** 2
    s = a0 + a1 + a2
    if s < 1.0e-200:
        return (d0, d1, d2)
    return (a0 / s, a1 / s, a2 / s)


def weno5_z_weights(beta: Tuple[float, float, float],
                     eps: float = WENO_EPS_ZETA, p: int = WENO_Z_POWER
                     ) -> Tuple[float, float, float]:
    """
    WENO5-Z weights (Borges et al. 2008):

        tau_5 = |beta_0 - beta_2|
        alpha_k = d_k (1 + (tau_5 / (eps + beta_k))^p)
    """
    d0, d1, d2 = 0.1, 0.6, 0.3
    tau5 = abs(beta[0] - beta[2])
    a0 = d0 * (1.0 + (tau5 / (eps + beta[0])) ** p)
    a1 = d1 * (1.0 + (tau5 / (eps + beta[1])) ** p)
    a2 = d2 * (1.0 + (tau5 / (eps + beta[2])) ** p)
    s = a0 + a1 + a2
    if s < 1.0e-200:
        return (d0, d1, d2)
    return (a0 / s, a1 / s, a2 / s)


# =====================================================================
#            GAUSSIAN-ERROR SWITCHED HYBRID WENO WEIGHTS
# =====================================================================

def gaussian_switched_weno_weights(
    beta: Tuple[float, float, float],
    shock_sensitivity: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Blend WENO5-J and WENO5-Z using the normal CDF as switching function.

        zeta = beta_min / (tau_5 + eps)
        w_J  = weno5_j_weights(beta)
        w_Z  = weno5_z_weights(beta)
        theta = Phi(-shock_sensitivity * (zeta - 0.5) / 0.25)

        w = theta w_J + (1 - theta) w_Z

    When zeta is small (smooth region) theta -> 0 -> WENO-Z is recovered;
    when zeta ~ 1 (near a shock) theta -> 1 -> WENO-J is used for its
    stronger numerical dissipation.
    """
    beta_min = min(beta)
    tau5 = abs(beta[0] - beta[2]) + WENO_EPS_ZETA
    zeta = beta_min / tau5
    theta = alnorm(-shock_sensitivity * (zeta - 0.5) / 0.25)
    wJ = weno5_j_weights(beta)
    wZ = weno5_z_weights(beta)
    return tuple(theta * wJ[i] + (1.0 - theta) * wZ[i] for i in range(3))


# =====================================================================
#                     WENO5 RECONSTRUCTION ROUTINE
# =====================================================================

def weno5_reconstruct_left(fm2: float, fm1: float, f0: float,
                            fp1: float, fp2: float,
                            method: str = "hybrid"
                            ) -> float:
    """
    Reconstruct f at x_{i+1/2} from the left using WENO5.

    The three substencil reconstructions are

        q_0 = ( 2 f_{i}   + 5 f_{i+1} - f_{i+2}) / 6
        q_1 = (-f_{i-1}   + 5 f_{i}   + 2 f_{i+1}) / 6
        q_2 = ( 2 f_{i-2} - 7 f_{i-1} + 11 f_{i}) / 6

    and the final value is  q = sum_k omega_k q_k  with the chosen
    non-linear weights.
    """
    q0 = (2.0 * f0 + 5.0 * fp1 - fp2) / 6.0
    q1 = (-fm1 + 5.0 * f0 + 2.0 * fp1) / 6.0
    q2 = (2.0 * fm2 - 7.0 * fm1 + 11.0 * f0) / 6.0
    beta = weno5_smoothness_indicators(fm2, fm1, f0, fp1, fp2)
    if method == "j":
        w = weno5_j_weights(beta)
    elif method == "z":
        w = weno5_z_weights(beta)
    elif method == "hybrid":
        w = gaussian_switched_weno_weights(beta)
    else:
        raise ValueError(f"unknown WENO method {method}")
    return w[0] * q0 + w[1] * q1 + w[2] * q2


def weno5_reconstruct_right(fm2: float, fm1: float, f0: float,
                             fp1: float, fp2: float,
                             method: str = "hybrid"
                             ) -> float:
    """
    Reconstruct f at x_{i-1/2} from the right (mirror of the left
    reconstruction with reversed stencil).
    """
    # Mirror the stencil: swap i <-> i-1 and reverse
    return weno5_reconstruct_left(fp2, fp1, f0, fm1, fm2, method)


# =====================================================================
#                   ARRAY-LEVEL WENO5 FLUX RECONSTRUCTION
# =====================================================================

def weno5_flux_array(f: np.ndarray, method: str = "hybrid") -> np.ndarray:
    """
    Apply WENO5 reconstruction to every cell interface of a periodic
    1-D array f.  Returns an array of reconstructed values at the i+1/2
    interfaces, of the same size as f.
    """
    n = f.size
    fp = np.concatenate([f[-3:], f, f[:3]])
    out = np.zeros(n)
    for i in range(n):
        # fp is shifted by 3; central point f_i is at index i+3
        idx = i + 3
        fm2 = fp[idx - 2]
        fm1 = fp[idx - 1]
        f0 = fp[idx]
        fp1 = fp[idx + 1]
        fp2 = fp[idx + 2]
        out[i] = weno5_reconstruct_left(fm2, fm1, f0, fp1, fp2, method)
    return out
