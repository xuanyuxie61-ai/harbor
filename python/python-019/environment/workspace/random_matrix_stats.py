"""
random_matrix_stats.py
----------------------
Statistical analysis of energy level spacing for non-Hermitian
random matrices, using the incomplete beta function.

Adapted from seed project 031_asa063 (incomplete beta function).

Scientific Background
=====================
Non-Hermitian random matrix theory (RMT) describes universal spectral
fluctuations of complex quantum systems. For the complex Ginibre
ensemble (matrices with i.i.d. complex Gaussian entries), the eigenvalues
fill a disk in the complex plane with uniform density.

The nearest-neighbor spacing distribution s = |z_{n+1} - z_n| / ⟨s⟩
in the bulk of the Ginibre ensemble is well-approximated by

    P_Ginibre(s) ≈ (32 / π^2) s^3 exp(-4 s^2 / π).

For real Ginibre ensembles, the statistics are more subtle due to
real eigenvalues.

More generally, for non-Hermitian Hamiltonians H = H_0 + i V with
H_0 and V drawn from Gaussian Orthogonal Ensembles (GOE), the level
spacing distribution interpolates between Wigner-Dyson (Hermitian
limit) and Ginibre (strongly non-Hermitian limit).

The Wigner surmise for the GOE spacing distribution is

    P_Wigner(s) = (π s / 2) exp(-π s^2 / 4),

while the Poisson (integrable) distribution is

    P_Poisson(s) = exp(-s).

To test whether a given spectrum follows a particular distribution,
one computes the cumulative distribution function (CDF) and compares
it to the theoretical CDF. The incomplete beta function arises when
computing confidence intervals for binomial tests of level-spacing
histograms.

The incomplete beta function is defined as

    I_x(a, b) = (1 / B(a,b)) ∫_0^x t^{a-1} (1-t)^{b-1} dt,

with B(a,b) = Γ(a)Γ(b) / Γ(a+b).
"""

import numpy as np
from scipy.special import gammaln


def log_beta(a, b):
    """
    Compute log(B(a,b)) = log(Γ(a)) + log(Γ(b)) - log(Γ(a+b)).
    """
    return gammaln(a) + gammaln(b) - gammaln(a + b)


def incomplete_beta(x, p, q):
    """
    Compute the incomplete beta function ratio I_x(p, q) using a
    series expansion (Soper's reduction formula).

    Adapted from Algorithm AS 63 (Majumder & Bhattacharjee, 1973).

    Parameters
    ----------
    x : float
        Upper limit of integration, must be in [0, 1].
    p, q : float
        Shape parameters, must be positive.

    Returns
    -------
    value : float
        I_x(p, q).
    ifault : int
        0 = success, 1 = invalid parameters, 2 = x out of range.
    """
    acu = 1e-14
    value = x
    ifault = 0

    if p <= 0.0 or q <= 0.0:
        ifault = 1
        return value, ifault
    if x < 0.0 or x > 1.0:
        ifault = 2
        return value, ifault
    if x == 0.0 or x == 1.0:
        return value, ifault

    psq = p + q
    cx = 1.0 - x

    if p < psq * x:
        xx = cx
        cx = x
        pp = q
        qq = p
        indx = 1
    else:
        xx = x
        pp = p
        qq = q
        indx = 0

    term = 1.0
    ai = 1.0
    value = 1.0
    ns = int(np.floor(qq + cx * psq))

    rx = xx / cx
    temp = qq - ai
    if ns == 0:
        rx = xx

    while True:
        term = term * temp * rx / (pp + ai)
        value = value + term
        temp = abs(term)

        if temp <= acu and temp <= acu * value:
            beta_log = log_beta(pp, qq)
            value = value * np.exp(pp * np.log(xx) + (qq - 1.0) * np.log(cx) - beta_log) / pp
            if indx:
                value = 1.0 - value
            break

        ai += 1.0
        ns -= 1
        if ns >= 0:
            temp = qq - ai
            if ns == 0:
                rx = xx
        else:
            temp = psq
            psq += 1.0

    return value, ifault


def level_spacing_ratios(eigenvalues, sort_by='real'):
    """
    Compute nearest-neighbor level spacing ratios for a set of
    complex eigenvalues.

    For complex spectra, we sort by real part (or modulus) and compute
    s_n = |E_{n+1} - E_n|.

    Parameters
    ----------
    eigenvalues : ndarray, dtype=complex
    sort_by : str
        'real', 'imag', or 'abs'.

    Returns
    -------
    spacings : ndarray
        Unnormalized nearest-neighbor spacings.
    ratios : ndarray
        r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}).
    """
    if sort_by == 'real':
        idx = np.argsort(eigenvalues.real)
    elif sort_by == 'imag':
        idx = np.argsort(eigenvalues.imag)
    else:
        idx = np.argsort(np.abs(eigenvalues))

    E_sorted = eigenvalues[idx]
    spacings = np.abs(np.diff(E_sorted))
    if len(spacings) < 2:
        return spacings, np.array([])

    ratios = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
    return spacings, ratios


def wigner_poisson_mixture_cdf(s, alpha):
    """
    Cumulative distribution function for a mixture of Wigner-Dyson
    and Poisson spacing distributions:

        P(s) = α P_Wigner(s) + (1-α) P_Poisson(s)

    Parameters
    ----------
    s : float or ndarray
        Normalized spacing.
    alpha : float
        Mixing parameter in [0, 1].

    Returns
    -------
    cdf : float or ndarray
    """
    wigner_cdf = 1.0 - np.exp(-np.pi * s ** 2 / 4.0)
    poisson_cdf = 1.0 - np.exp(-s)
    return alpha * wigner_cdf + (1.0 - alpha) * poisson_cdf


def generate_ginibre_spectrum(N, seed=42):
    """
    Generate eigenvalues of an N×N complex Ginibre random matrix.

    Entries are i.i.d. complex Gaussian with variance 1/N.

    Parameters
    ----------
    N : int
    seed : int

    Returns
    -------
    eigenvalues : ndarray, shape (N,), dtype=complex
    """
    rng = np.random.default_rng(seed)
    M = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) / np.sqrt(2.0 * N)
    return np.linalg.eigvals(M)


def analyze_spacing_statistics(eigenvalues, num_bins=20):
    """
    Analyze level-spacing statistics and fit a Wigner-Poisson mixture.

    Returns histogram data and a goodness-of-fit metric.
    """
    spacings, ratios = level_spacing_ratios(eigenvalues, sort_by='real')
    if len(spacings) == 0:
        return None
    mean_s = spacings.mean()
    if mean_s < 1e-15:
        return None
    s_norm = spacings / mean_s

    hist, bin_edges = np.histogram(s_norm, bins=num_bins, range=(0.0, 4.0), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Fit alpha by minimizing chi-square against mixture CDF derivative
    best_alpha = 0.0
    best_chi2 = 1e308
    for alpha in np.linspace(0.0, 1.0, 21):
        pdf = (
            alpha * (np.pi * bin_centers / 2.0) * np.exp(-np.pi * bin_centers ** 2 / 4.0)
            + (1.0 - alpha) * np.exp(-bin_centers)
        )
        chi2 = np.sum((hist - pdf) ** 2 / (pdf + 1e-10))
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_alpha = alpha

    return {
        'bin_centers': bin_centers,
        'histogram': hist,
        'alpha_fit': best_alpha,
        'mean_spacing': mean_s,
        'ratios': ratios,
    }
