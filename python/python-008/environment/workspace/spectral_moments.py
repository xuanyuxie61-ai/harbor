"""
Spectral Moments Module
=======================
Based on seed project 506_hankel_spd:
- hankel_spd_cholesky_lower.m  →  Hankel SPD matrix construction

Physics:
--------
The spectral energy distribution (SED) of a GRB afterglow can be
characterized by its moments:

    μ_k = ∫_0^{∞} ν^k F_ν dν    [erg cm⁻² s⁻¹ Hz^{k}]

The Hankel moment matrix is:

    H_{ij} = μ_{i+j}    for i, j = 0, ..., N-1

By the Hamburger moment problem, H is symmetric positive definite
(SPD) if and only if the spectral measure dΦ(ν) = F_ν dν is a
positive measure.  The Cholesky factorization H = L·Lᵀ provides
a representation of the spectrum in terms of orthogonal polynomials.

In GRB physics, the first few moments have direct physical meaning:

    μ_0 = bolometric flux  [erg cm⁻² s⁻¹]
    μ_1 = mean frequency   [Hz]
    μ_2 = spectral width   [Hz²]

The moment matrix also appears in the Christoffel-Darboux kernel
for spectral line shape analysis:

    K_N(ν, ν') = Σ_{k=0}^{N-1} p_k(ν) p_k(ν')

where p_k are orthonormal polynomials with respect to dΦ.
"""

import numpy as np


def hankel_spd_cholesky_lower(n, lii, liim1):
    """
    Constructs a lower-triangular Cholesky factor L such that
    H = L·Lᵀ is a symmetric positive-definite Hankel matrix.

    Algorithm (Al-Homidan & Alshahrani 2009):

        L(i,i)   = lii[i]
        L(i+1,i) = liim1[i]

        For i = 3..N, j = 1..i-2:
            if (i+j) even:  q = (i+j)/2, r = q
            else:           q = (i+j-1)/2, r = q+1

            α = Σ_{s=1}^{q} L(q,s) L(r,s)
            β = Σ_{t=1}^{j-1} L(i,t) L(j,t)
            L(i,j) = (α - β) / L(j,j)

    Parameters
    ----------
    n : int
        Matrix order.
    lii : ndarray, shape (n,)
        Diagonal entries of L.
    liim1 : ndarray, shape (n-1,)
        First subdiagonal entries of L.

    Returns
    -------
    L : ndarray, shape (n, n)
        Lower Cholesky factor.
    """
    # TODO: Implement Hankel SPD Cholesky factorization.
    # Algorithm (Al-Homidan & Alshahrani 2009):
    #   L(i,i)   = lii[i]
    #   L(i+1,i) = liim1[i]
    #   For i = 3..N, j = 1..i-2:
    #       compute L(i,j) using Hankel symmetry constraints.
    # Return lower-triangular L.
    pass


def build_hankel_from_moments(moments):
    """
    Build Hankel moment matrix H_{ij} = μ_{i+j}.

    Parameters
    ----------
    moments : ndarray, shape (2*n-1,)
        Spectral moments.

    Returns
    -------
    H : ndarray
        Hankel matrix.
    """
    m = moments.size
    n = (m + 1) // 2
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            H[i, j] = moments[i + j]
    return H


def compute_spectral_moments_from_hankel(H):
    """
    Recover spectral statistics from Hankel moment matrix.

    Returns
    -------
    stats : dict
        bolometric_flux, mean_freq, spectral_width
    """
    # Moments are the first row/column anti-diagonal entries
    n = H.shape[0]
    mu = np.zeros(2 * n - 1, dtype=float)
    for k in range(2 * n - 1):
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                mu[k] = H[i, j]
                break

    mu_0 = mu[0] if mu.size > 0 else 1.0
    mu_1 = mu[1] if mu.size > 1 else 0.0
    mu_2 = mu[2] if mu.size > 2 else 0.0

    mean_freq = mu_1 / (mu_0 + 1e-30)
    spectral_width = np.sqrt(max(0.0, mu_2 / (mu_0 + 1e-30) - mean_freq ** 2))

    return {
        "bolometric_flux": mu_0,
        "mean_frequency": mean_freq,
        "spectral_width": spectral_width,
        "moments": mu,
    }


def synthetic_grb_moments(n):
    """
    Generate synthetic spectral moments for a GRB afterglow
    with a broken power-law SED:

        F_ν ∝ ν^{-α}   for ν < ν_b
        F_ν ∝ ν^{-β}   for ν > ν_b

    The k-th moment integral gives:

        μ_k = ν_b^{k+1-α} / (k+1-α) + ν_b^{k+1-β} / (β-k-1)
    """
    alpha = 0.5
    beta = 2.5
    nu_b = 1e15  # Hz

    moments = np.zeros(2 * n - 1, dtype=float)
    for k in range(2 * n - 1):
        term1 = nu_b ** (k + 1 - alpha) / (k + 1 - alpha)
        term2 = nu_b ** (k + 1 - beta) / (beta - k - 1)
        moments[k] = term1 + term2

    return moments
