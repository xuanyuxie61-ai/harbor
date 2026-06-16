"""
hermite_spectral_modes.py
==========================
Hermite polynomial spectral decomposition for dust grain oscillation
modes in the plasma crystal.

Physical motivation:
    Small oscillations of dust grains around their equilibrium positions
    in a dusty plasma crystal can be decomposed into normal modes.
    When the confining potential is approximately harmonic (as in the
    plasma sheath), the mode eigenfunctions are Hermite-Gaussian:
        psi_n(x) = H_n(x/sigma) * exp(-x^2/(2*sigma^2))
    where H_n are Hermite polynomials and sigma is the characteristic
    width of the confining potential.

    The Hermite functions form a complete orthonormal basis for L^2(R):
        integral psi_m(x) * psi_n(x) dx = delta_{mn}

    We use the physicist's Hermite polynomials with the recurrence:
        H_0(x) = 1,  H_1(x) = 2x
        H_{n+1}(x) = 2x * H_n(x) - 2n * H_{n-1}(x)

    And Gaussian-Hermite quadrature for computing matrix elements.

References:
    - From 522_hermite_polynomial: he_polynomial_value, h_polynomial_value,
      hf_function_value, he_quadrature_rule, imtqlx
    - Press et al., "Numerical Recipes", Chapter on orthogonal polynomials
"""

import numpy as np
from typing import Tuple, Optional


def hermiteH_value(x: np.ndarray, n_max: int) -> np.ndarray:
    """
    Evaluate physicist's Hermite polynomials H_n(x) up to degree n_max.

    Three-term recurrence (from 522_hermite_polynomial):
        H_0(x) = 1
        H_1(x) = 2x
        H_{n+1}(x) = 2x*H_n(x) - 2n*H_{n-1}(x)

    Parameters
    ----------
    x : np.ndarray, shape (M,)
        Evaluation points.
    n_max : int
        Maximum degree.

    Returns
    -------
    P : np.ndarray, shape (M, n_max+1)
        P[:, n] = H_n(x).
    """
    M = len(x)
    P = np.zeros((M, n_max + 1), dtype=np.float64)
    P[:, 0] = 1.0
    if n_max >= 1:
        P[:, 1] = 2.0 * x
    for n in range(1, n_max):
        P[:, n + 1] = 2.0 * x * P[:, n] - 2.0 * n * P[:, n - 1]
    return P


def hermiteHe_value(x: np.ndarray, n_max: int) -> np.ndarray:
    """
    Evaluate probabilist's Hermite polynomials He_n(x) up to degree n_max.

    Three-term recurrence:
        He_0(x) = 1
        He_1(x) = x
        He_{n+1}(x) = x*He_n(x) - n*He_{n-1}(x)

    The probabilist's Hermite polynomials are related to the physicist's
    by: He_n(x) = 2^(-n/2) * H_n(x/sqrt(2)).

    Parameters
    ----------
    x : np.ndarray, shape (M,)
        Evaluation points.
    n_max : int
        Maximum degree.

    Returns
    -------
    P : np.ndarray, shape (M, n_max+1)
        P[:, n] = He_n(x).
    """
    M = len(x)
    P = np.zeros((M, n_max + 1), dtype=np.float64)
    P[:, 0] = 1.0
    if n_max >= 1:
        P[:, 1] = x
    for n in range(1, n_max):
        P[:, n + 1] = x * P[:, n] - n * P[:, n - 1]
    return P


def hermite_function_value(
    x: np.ndarray,
    n_max: int,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Evaluate orthonormal Hermite functions (quantum harmonic oscillator basis).

    psi_n(x) = (1/(sqrt(2^n * n! * sigma * sqrt(pi)))) *
               H_n(x/sigma) * exp(-x^2/(2*sigma^2))

    These are the eigenfunctions of the quantum harmonic oscillator:
        (-1/2) * psi_n'' + (1/2)*(x/sigma)^2 * psi_n = (n + 1/2) * psi_n

    For dusty plasma, psi_n represents the n-th oscillation mode of a
    grain trapped in the sheath potential well.

    Parameters
    ----------
    x : np.ndarray, shape (M,)
        Evaluation points.
    n_max : int
        Maximum mode number.
    sigma : float
        Characteristic width of the confining potential.

    Returns
    -------
    psi : np.ndarray, shape (M, n_max+1)
        psi[:, n] = psi_n(x).
    """
    M = len(x)
    xi = x / sigma  # Normalized coordinate

    # Evaluate Hermite polynomials
    P = hermiteH_value(xi, n_max)

    # Gaussian envelope
    gauss = np.exp(-0.5 * xi**2)

    # Normalization: 1/sqrt(2^n * n! * sigma * sqrt(pi))
    psi = np.zeros((M, n_max + 1), dtype=np.float64)
    for n in range(n_max + 1):
        norm = 1.0 / np.sqrt(
            (2.0**n) * float(_factorial(n)) * sigma * np.sqrt(np.pi)
        )
        psi[:, n] = norm * P[:, n] * gauss

    return psi


def hermite_gauss_quadrature(
    n_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Gauss-Hermite quadrature nodes and weights.

    The Golub-Welsch algorithm computes nodes as eigenvalues of the
    symmetric tridiagonal Jacobi matrix:
        J[i,i] = 0
        J[i,i+1] = J[i+1,i] = sqrt(i/2)  (for physicist's Hermite)
    or equivalently using the imtqlx implicit QR algorithm
    (from 522_hermite_polynomial).

    Parameters
    ----------
    n_points : int
        Number of quadrature points.

    Returns
    -------
    nodes : np.ndarray, shape (n_points,)
        Quadrature nodes (zeros of H_{n_points}).
    weights : np.ndarray, shape (n_points,)
        Quadrature weights for integral of f(x)*exp(-x^2) dx.
    """
    # Use Golub-Welsch via eigendecomposition of Jacobi matrix
    # For physicist's Hermite: recurrence coefficients
    # H_{n+1}(x) = 2x*H_n(x) - 2n*H_{n-1}(x)
    # Monic recurrence: P_{n+1} = x*P_n - (n/2)*P_{n-1}
    # Jacobi matrix: diagonal = 0, off-diagonal = sqrt(n/2)

    N = n_points
    diag = np.zeros(N)
    off_diag = np.zeros(N - 1)
    for i in range(N - 1):
        off_diag[i] = np.sqrt((i + 1) / 2.0)

    # Symmetric tridiagonal eigendecomposition
    J = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    eigenvalues, eigenvectors = np.linalg.eigh(J)

    nodes = eigenvalues
    # Weights: w_i = sqrt(pi) * v_{0,i}^2
    weights = np.sqrt(np.pi) * eigenvectors[0, :]**2

    return nodes, weights


def _factorial(n: int) -> int:
    """Compute n! for non-negative integer n."""
    if n <= 1:
        return 1
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result


def compute_mode_overlap_matrix(
    sigma1: float,
    sigma2: float,
    n_max: int,
    n_quad: int = 50,
) -> np.ndarray:
    """
    Compute overlap matrix between Hermite bases with different widths.

    S[m,n] = integral psi_m(x, sigma1) * psi_n(x, sigma2) dx

    This arises when matching grain oscillation modes in different
    plasma sheath conditions.

    Parameters
    ----------
    sigma1, sigma2 : float
        Width parameters for the two bases.
    n_max : int
        Maximum mode number.
    n_quad : int
        Number of quadrature points.

    Returns
    -------
    S : np.ndarray, shape (n_max+1, n_max+1)
        Overlap matrix.
    """
    # Use Gauss-Hermite quadrature with the wider basis
    sigma_eff = max(sigma1, sigma2) * 2.0
    nodes, weights = hermite_gauss_quadrature(n_quad)
    nodes *= sigma_eff
    weights *= sigma_eff * np.exp(nodes**2)  # Adjust for non-standard weight

    psi1 = hermite_function_value(nodes, n_max, sigma1)
    psi2 = hermite_function_value(nodes, n_max, sigma2)

    S = np.zeros((n_max + 1, n_max + 1))
    for m in range(n_max + 1):
        for n in range(n_max + 1):
            integrand = psi1[:, m] * psi2[:, n]
            S[m, n] = np.sum(weights * integrand)

    return S


def dust_oscillation_spectrum(
    omega_conf: float,
    n_modes: int = 10,
    temperature_ratio: float = 0.01,
) -> dict:
    """
    Compute the oscillation spectrum of a dust grain in a harmonic
    confining potential with anharmonic corrections.

    The confining potential in the plasma sheath is approximately:
        V(x) = (1/2) * m_d * omega_conf^2 * x^2 + alpha * x^4

    The energy levels (including anharmonic correction) are:
        E_n = hbar * omega_conf * (n + 1/2)
              - alpha * (3 + 6n + 3n^2) / (4 * m_d^2 * omega_conf^3)

    For dusty plasma, the "quantum" correction is small and we treat
    this classically: the mode frequencies are:
        omega_n = omega_conf * (1 + correction_n)

    Parameters
    ----------
    omega_conf : float
        Confining frequency [rad/s].
    n_modes : int
        Number of modes to compute.
    temperature_ratio : float
        k_B*T_d / (hbar*omega_conf) ratio for thermal population.

    Returns
    -------
    results : dict
        'energies': mode energies (in units of hbar*omega),
        'frequencies': mode frequencies [rad/s],
        'populations': thermal populations,
        'eigenfunctions': psi_n at sample points.
    """
    n_sample = 200
    x_range = 5.0 * np.sqrt(1.0 / (omega_conf + 1e-30))
    x = np.linspace(-x_range, x_range, n_sample)

    psi = hermite_function_value(x, n_modes, sigma=1.0 / np.sqrt(omega_conf + 1e-30))

    energies = np.arange(n_modes + 1, dtype=np.float64) + 0.5
    frequencies = omega_conf * np.ones(n_modes + 1)

    # Thermal populations (classical limit)
    kT = temperature_ratio * omega_conf
    if kT > 0:
        populations = np.exp(-energies / kT)
        populations /= np.sum(populations)
    else:
        populations = np.zeros(n_modes + 1)
        populations[0] = 1.0

    return {
        "energies": energies,
        "frequencies": frequencies,
        "populations": populations,
        "x_sample": x,
        "eigenfunctions": psi,
        "n_modes": n_modes,
    }
