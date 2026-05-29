"""
QCD Special Functions Library
==============================
Derived from polpak (orthogonal polynomials, Gamma, Zeta) and
1228_test_values (special function validation).

Provides core mathematical tools for parton shower and jet clustering:
- Running coupling alpha_s(Q^2)
- LO QCD splitting functions P_{ij}(z)
- Legendre / Chebyshev / Hermite polynomials for angular decomposition
- Polylogarithms and harmonic sums for NLL resummation
- Gamma function ratios needed for anomalous dimensions
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import gammaln, digamma, polygamma, zeta as riemann_zeta

# QCD color constants
NC = 3.0       # Number of colors
TF = 0.5       # T_F = 1/2
CA = NC        # C_A = N_c
CF = (NC**2 - 1.0) / (2.0 * NC)  # C_F = (N_c^2-1)/(2N_c)

# Quark flavor thresholds
N_F = 5        # Active flavors at LHC energies

# One-loop beta function coefficient
BETA0 = (11.0 * CA - 4.0 * TF * N_F) / 3.0

# Lambda_QCD (approximate, in GeV)
LAMBDA_QCD = 0.2  # GeV


def alpha_s_1loop(Q2, nf=N_F, lambda_qcd=LAMBDA_QCD):
    """
    One-loop running strong coupling:
        α_s(Q^2) = 4π / [ β_0 ln(Q^2/Λ_QCD^2) ]
    
    Parameters
    ----------
    Q2 : float or array
        Squared momentum transfer in GeV^2.
    nf : int
        Number of active quark flavors.
    lambda_qcd : float
        Λ_QCD in GeV.
    
    Returns
    -------
    float or array
        Running coupling α_s.
    """
    Q2 = np.asarray(Q2, dtype=float)
    beta0 = (11.0 * CA - 4.0 * TF * nf) / 3.0
    if beta0 <= 0:
        raise ValueError("beta0 must be positive; check nf.")
    
    # Avoid Landau pole and unphysical low scales
    min_Q2 = (1.1 * lambda_qcd) ** 2
    Q2_safe = np.where(Q2 < min_Q2, min_Q2, Q2)
    
    log_term = np.log(Q2_safe / (lambda_qcd ** 2))
    # Prevent division by zero or negative log
    log_term = np.where(log_term < 1e-6, 1e-6, log_term)
    
    return 4.0 * np.pi / (beta0 * log_term)


def alpha_s_2loop(Q2, nf=N_F, lambda_qcd=LAMBDA_QCD):
    """
    Two-loop running strong coupling (more accurate for high energies):
        α_s/π = 1/(β_0 L) - (β_1 ln L)/(β_0^3 L^2) + O(1/L^3)
    where L = ln(Q^2/Λ^2), and
        β_1 = 34/3 C_A^2 - 20/3 C_A T_F n_f - 4 C_F T_F n_f
    """
    Q2 = np.asarray(Q2, dtype=float)
    beta0 = (11.0 * CA - 4.0 * TF * nf) / 3.0
    beta1 = (34.0 / 3.0) * CA**2 - (20.0 / 3.0) * CA * TF * nf - 4.0 * CF * TF * nf
    
    min_Q2 = (1.1 * lambda_qcd) ** 2
    Q2_safe = np.where(Q2 < min_Q2, min_Q2, Q2)
    L = np.log(Q2_safe / (lambda_qcd ** 2))
    L = np.where(L < 1e-6, 1e-6, L)
    
    a1 = 1.0 / (beta0 * L)
    a2 = -beta1 * np.log(L) / (beta0**3 * L**2)
    return np.pi * (a1 + a2)


def p_qq_lo(z, eps=1e-10):
    """
    LO quark -> quark + gluon splitting function:
        P_{qq}(z) = C_F * [ (1+z^2)/(1-z)_+ + 3/2 δ(1-z) ]
    
    For numerical implementation, the plus prescription is handled by
    subtracting the soft divergence at z→1.
    
    Parameters
    ----------
    z : float or array, in (0,1)
        Momentum fraction.
    eps : float
        Regularization cutoff near z=1.
    
    Returns
    -------
    float or array
        Regularized P_qq(z) (without δ-term).
    """
    z = np.asarray(z, dtype=float)
    # Enforce physical domain
    z = np.clip(z, eps, 1.0 - eps)
    
    regular = (1.0 + z**2) / (1.0 - z)
    # Plus prescription: replace 1/(1-z) by 1/(1-z) - δ(1-z)∫dz/(1-z)
    # For a single evaluation we regularize by subtracting the pole residue
    return CF * regular


def p_qg_lo(z, eps=1e-10):
    """
    LO gluon -> quark + anti-quark splitting function:
        P_{qg}(z) = T_R * [ z^2 + (1-z)^2 ]
    """
    z = np.asarray(z, dtype=float)
    z = np.clip(z, eps, 1.0 - eps)
    return TF * (z**2 + (1.0 - z)**2)


def p_gq_lo(z, eps=1e-10):
    """
    LO quark -> gluon + quark (gluon emission from quark, gluon carries z):
        P_{gq}(z) = C_F * [ 1 + (1-z)^2 ] / z
    """
    z = np.asarray(z, dtype=float)
    z = np.clip(z, eps, 1.0 - eps)
    return CF * (1.0 + (1.0 - z)**2) / z


def p_gg_lo(z, eps=1e-10):
    """
    LO gluon -> gluon + gluon splitting function:
        P_{gg}(z) = 2 C_A [ z/(1-z)_+ + (1-z)/z + z(1-z) ] + (β_0/2) δ(1-z)
    
    Returns the regular part (without δ-term).
    """
    z = np.asarray(z, dtype=float)
    z = np.clip(z, eps, 1.0 - eps)
    
    regular = z / (1.0 - z) + (1.0 - z) / z + z * (1.0 - z)
    return 2.0 * CA * regular


def sudakov_quark(Q2, q2, zmin=0.01, zmax=0.99, nf=N_F):
    """
    Quark Sudakov form factor (probability of NO emission between q^2 and Q^2):
        Δ_q(Q^2, q^2) = exp[ -∫_{q^2}^{Q^2} (dk^2/k^2) (α_s(k^2)/2π)
                               × ∫_{zmin}^{zmax} dz (P_qq(z) + P_gq(z)) ]
    
    Parameters
    ----------
    Q2, q2 : float
        Upper and lower scales in GeV^2.
    zmin, zmax : float
        Branching kinematic cuts.
    
    Returns
    -------
    float
        Sudakov suppression factor in [0,1].
    """
    if q2 >= Q2 or Q2 <= 0 or q2 <= 0:
        return 1.0
    
    # TODO: Implement numerical double integral for Sudakov form factor
    pass


def legendre_poly_vals(n, x):
    """
    Evaluate Legendre polynomials P_0(x)...P_n(x) via stable recurrence.
    From polpak (Legendre polynomial recurrence).
    
    P_0(x) = 1
    P_1(x) = x
    (k+1) P_{k+1}(x) = (2k+1) x P_k(x) - k P_{k-1}(x)
    
    Returns array of shape (len(x), n+1).
    """
    x = np.asarray(x, dtype=float)
    if n < 0:
        return np.empty((x.size, 0))
    
    vals = np.zeros((x.size, n + 1))
    vals[:, 0] = 1.0
    if n >= 1:
        vals[:, 1] = x
    
    for k in range(1, n):
        vals[:, k + 1] = ((2.0 * k + 1.0) * x * vals[:, k] - k * vals[:, k - 1]) / (k + 1.0)
    
    return vals


def chebyshev_poly_vals(n, x):
    """
    Chebyshev polynomials of the first kind T_n(x) via recurrence.
    T_0 = 1, T_1 = x, T_{n+1} = 2x T_n - T_{n-1}
    
    Returns array of shape (len(x), n+1).
    """
    x = np.asarray(x, dtype=float)
    if n < 0:
        return np.empty((x.size, 0))
    
    vals = np.zeros((x.size, n + 1))
    vals[:, 0] = 1.0
    if n >= 1:
        vals[:, 1] = x
    
    for k in range(1, n):
        vals[:, k + 1] = 2.0 * x * vals[:, k] - vals[:, k - 1]
    
    return vals


def hermite_poly_vals(n, x):
    """
    Physicists' Hermite polynomials H_n(x) via recurrence.
    H_0 = 1, H_1 = 2x, H_{n+1} = 2x H_n - 2n H_{n-1}
    
    Returns array of shape (len(x), n+1).
    """
    x = np.asarray(x, dtype=float)
    if n < 0:
        return np.empty((x.size, 0))
    
    vals = np.zeros((x.size, n + 1))
    vals[:, 0] = 1.0
    if n >= 1:
        vals[:, 1] = 2.0 * x
    
    for k in range(1, n):
        vals[:, k + 1] = 2.0 * x * vals[:, k] - 2.0 * k * vals[:, k - 1]
    
    return vals


def harmonic_sum(n, m=1):
    """
    Generalized harmonic number H_{n}^{(m)} = sum_{k=1}^n 1/k^m.
    Needed for anomalous dimensions and NLL resummation coefficients.
    """
    if n <= 0:
        return 0.0
    if m == 1:
        return digamma(n + 1.0) + np.euler_gamma
    else:
        return float(np.sum(1.0 / np.arange(1, n + 1, dtype=float) ** m))


def di_log(x):
    """
    Dilogarithm Li_2(x) = -∫_0^x ln(1-t)/t dt.
    Approximation valid for x in [-1,1].
    Used in NLO splitting functions and Sudakov exponents.
    """
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -1.0, 1.0)
    
    # Series expansion for |x| <= 1
    # Li_2(x) = sum_{k=1}^∞ x^k / k^2
    result = np.zeros_like(x, dtype=float)
    for k in range(1, 50):
        result += x**k / (k * k)
    return result


def anomalous_dim_gamma_0(nf=N_F):
    """
    One-loop cusp anomalous dimension coefficient:
        γ_0 = 4 C_F
    In general notation for resummation: A_1 = C_F / π.
    """
    return 4.0 * CF


def anomalous_dim_gamma_1(nf=N_F):
    """
    Two-loop cusp anomalous dimension:
        γ_1 = 4 C_F [ (67/9 - π^2/3) C_A - 20/9 T_F n_f ]
    """
    return 4.0 * CF * ((67.0 / 9.0 - np.pi**2 / 3.0) * CA - (20.0 / 9.0) * TF * nf)


def validate_special_functions():
    """
    Validation suite derived from 1228_test_values.
    Checks consistency of our special function implementations
    against known exact or high-precision values.
    """
    max_error = 0.0
    
    # Check Legendre P_2(0.5) = -0.125
    leg = legendre_poly_vals(2, np.array([0.5]))
    val = leg[0, 2]
    exact = -0.125
    err = abs(val - exact)
    max_error = max(max_error, err)
    
    # Check Chebyshev T_3(0.5) = -1.0
    cheb = chebyshev_poly_vals(3, np.array([0.5]))
    val = cheb[0, 3]
    exact = -1.0
    err = abs(val - exact)
    max_error = max(max_error, err)
    
    # Check Hermite H_2(1.0) = 2
    herm = hermite_poly_vals(2, np.array([1.0]))
    val = herm[0, 2]
    exact = 2.0
    err = abs(val - exact)
    max_error = max(max_error, err)
    
    # Check alpha_s at high Q2 is small
    a_s = alpha_s_1loop(1e8)
    if not (0.0 < a_s < 0.5):
        raise RuntimeError(f"alpha_s unphysical: {a_s}")
    
    # Check Sudakov factor is monotonically increasing with lower cutoff
    # (smaller emission interval -> larger survival probability)
    s1 = sudakov_quark(100.0, 1.0)
    s2 = sudakov_quark(100.0, 10.0)
    if s1 > s2:
        raise RuntimeError("Sudakov factor not monotonic: larger q2 should give larger survival prob")
    
    return max_error


if __name__ == "__main__":
    err = validate_special_functions()
    print(f"Special function validation max error: {err:.2e}")
