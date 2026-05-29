"""
hamiltonian_builder.py
----------------------
Construct non-Hermitian Hamiltonians for condensed-matter models.

This module implements tight-binding non-Hermitian Hamiltonians with
balanced gain and loss, inspired by parity-time (PT) symmetric photonic
crystals and non-Hermitian topological insulators.

Scientific Background
=====================
A general non-Hermitian Hamiltonian in second-quantized form reads

    H = Σ_{i,j} t_{ij} c_i^† c_j  +  i Σ_j γ_j c_j^† c_j

where t_{ij} = t_{ji}^* for the Hermitian part, and γ_j are real gain/loss
rates. In momentum space for a 1D or 2D lattice,

    H(k) = H_0(k) + i Γ,

with H_0(k) Hermitian and Γ a real diagonal or non-diagonal matrix.

For a 2D square lattice with nearest-neighbor hopping t and on-site
potential V, the Bloch Hamiltonian is

    H(k) =  -2t [ cos(k_x a) σ_x  +  cos(k_y a) σ_y ]  +  (m + iγ) σ_z

where σ_{x,y,z} are Pauli matrices, m is a mass term, and γ controls
the non-Hermitian strength. This model hosts exceptional points where
the discriminant of the characteristic polynomial vanishes.

Exceptional points satisfy

    det[ H(k_EP) - E_EP I ] = 0,
    ∂_E det[ H(k_EP) - E_EP I ] = 0,

simultaneously, meaning both eigenvalues and eigenvectors coalesce.
"""

import numpy as np


# Pauli matrices (convention: acting on pseudospin space)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY2 = np.eye(2, dtype=complex)


def build_pt_symmetric_hamiltonian_1d(k, t=1.0, m=0.5, gamma=0.3):
    """
    1D PT-symmetric non-Hermitian two-band Hamiltonian.

    H(k) = (m + t cos(k)) σ_z + t sin(k) σ_y + i γ σ_x

    Parameters
    ----------
    k : float or ndarray
        Crystal momentum.
    t : float
        Hopping amplitude.
    m : float
        Mass term.
    gamma : float
        Non-Hermitian strength (gain/loss imbalance).

    Returns
    -------
    H : ndarray, shape (2, 2), dtype=complex
    """
    H = (m + t * np.cos(k)) * PAULI_Z + t * np.sin(k) * PAULI_Y + 1j * gamma * PAULI_X
    return H


def build_pt_symmetric_hamiltonian_2d(kx, ky, t=1.0, m=0.5, gamma=0.3, a=1.0):
    """
    2D non-Hermitian Hamiltonian on a square lattice.

    H(kx, ky) = -2t [ cos(kx a) σ_x + cos(ky a) σ_y ] + (m + iγ) σ_z

    Parameters
    ----------
    kx, ky : float
        Crystal momenta.
    t : float
        Hopping amplitude.
    m : float
        Mass term.
    gamma : float
        Non-Hermitian strength.
    a : float
        Lattice constant.

    Returns
    -------
    H : ndarray, shape (2, 2), dtype=complex
    """
    H = -2.0 * t * (np.cos(kx * a) * PAULI_X + np.cos(ky * a) * PAULI_Y)
    H += (m + 1j * gamma) * PAULI_Z
    return H


def build_nonhermitian_ssh_hamiltonian(k, t1=1.0, t2=0.5, gamma=0.2):
    """
    Non-Hermitian Su-Schrieffer-Heeger (SSH) model with on-site gain/loss.

    The Bloch Hamiltonian in momentum space is

        H(k) = [t1 + t2 cos(k)] σ_x + t2 sin(k) σ_y + i γ σ_z

    Parameters
    ----------
    k : float
        Crystal momentum.
    t1 : float
        Intra-cell hopping.
    t2 : float
        Inter-cell hopping.
    gamma : float
        On-site non-Hermitian potential (±γ on A/B sublattices).

    Returns
    -------
    H : ndarray, shape (2, 2), dtype=complex
    """
    # TODO: Implement the non-Hermitian SSH Bloch Hamiltonian.
    # H(k) = (t1 + t2 cos k) σ_x + t2 sin k σ_y + i γ σ_z
    raise NotImplementedError("SSH Hamiltonian construction is missing.")


def build_nonhermitian_hofstadter_hamiltonian(kx, ky, phi, t=1.0, gamma=0.1, q=4):
    """
    Non-Hermitian Hofstadter model on a square lattice with flux φ = p/q
    per plaquette. The Peierls substitution introduces complex phases in
    hopping, while γ adds non-Hermitian on-site terms.

    For flux φ = p/q, the magnetic unit cell contains q sites, yielding
    a q×q Bloch Hamiltonian H(kx, ky).

    Parameters
    ----------
    kx, ky : float
    phi : float
        Magnetic flux per plaquette (in units of flux quantum).
    t : float
        Hopping amplitude.
    gamma : float
        Non-Hermitian strength.
    q : int
        Denominator of flux rational approximation.

    Returns
    -------
    H : ndarray, shape (q, q), dtype=complex
    """
    if q <= 0:
        raise ValueError("q must be a positive integer.")
    H = np.zeros((q, q), dtype=complex)
    for n in range(q):
        # On-site non-Hermitian term
        H[n, n] = 1j * gamma * ((-1) ** n)
        # Hopping in x direction with Peierls phase
        H[n, (n + 1) % q] += -t * np.exp(1j * kx)
        H[(n + 1) % q, n] += -t * np.exp(-1j * kx)
        # Hopping in y direction (Landau gauge)
        H[n, n] += -2.0 * t * np.cos(ky - 2.0 * np.pi * phi * n)
    return H


def characteristic_polynomial_2x2(H):
    """
    Compute coefficients of the characteristic polynomial for a 2×2 matrix:

        p(E) = E^2 - Tr(H) E + det(H)

    Returns (c2, c1, c0) with c2=1, so p(E) = c2 E^2 + c1 E + c0.
    """
    if H.shape != (2, 2):
        raise ValueError("Only 2x2 matrices supported.")
    c2 = 1.0 + 0.0j
    c1 = -np.trace(H)
    c0 = np.linalg.det(H)
    return c2, c1, c0


def discriminant_2x2(H):
    """
    Discriminant of the characteristic polynomial of a 2×2 matrix.
    An exceptional point occurs when Δ = 0.

    Δ = c1^2 - 4 c2 c0 = Tr(H)^2 - 4 det(H)
    """
    c2, c1, c0 = characteristic_polynomial_2x2(H)
    delta = c1 ** 2 - 4.0 * c2 * c0
    return delta
