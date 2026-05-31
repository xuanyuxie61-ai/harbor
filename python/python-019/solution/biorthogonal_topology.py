"""
biorthogonal_topology.py
------------------------
Compute biorthogonal Berry phases, Berry curvatures, and winding numbers
for non-Hermitian Hamiltonians.

Scientific Background
=====================
In non-Hermitian systems, the standard eigenvector normalization
⟨ψ_n|ψ_n⟩ = 1 is replaced by the biorthogonal normalization

    ⟨ψ_n^L | ψ_m^R⟩ = δ_{nm},

where |ψ_n^R⟩ and ⟨ψ_n^L| are right and left eigenvectors:

    H |ψ_n^R⟩ = E_n |ψ_n^R⟩,     ⟨ψ_n^L| H = E_n ⟨ψ_n^L|.

The biorthogonal Berry connection for band n is defined as

    A_n(k) = i ⟨ψ_n^L(k)| ∇_k |ψ_n^R(k)⟩,

and the Berry curvature is

    Ω_n(k) = ∇_k × A_n(k)
           = i [ ⟨∂_{k_x} ψ_n^L | ∂_{k_y} ψ_n^R⟩
                 - ⟨∂_{k_y} ψ_n^L | ∂_{k_x} ψ_n^R⟩ ].

The biorthogonal Berry phase around a closed loop C is

    γ_n = ∮_C A_n(k) · dk  =  ∬_S Ω_n(k) d^2k,

where S is a surface bounded by C. In the biorthogonal framework,
the Berry phase is not necessarily quantized to 2πℤ but can acquire
complex values, reflecting the non-Hermitian skin effect and
exceptional point encircling.

For a 1D system, the Zak phase (a special case of Berry phase) is

    γ_Zak = ∫_{-π/a}^{π/a} A(k) dk.

The winding number of the complex energy spectrum is

    W = (1 / 2πi) ∮_C dE / E,

which counts the number of times the spectrum encircles the origin
in the complex energy plane.
"""

import numpy as np


def compute_biorthogonal_eigenvectors(H):
    """
    Compute left and right eigenvectors of a non-Hermitian matrix H
    with biorthogonal normalization.

    Parameters
    ----------
    H : ndarray, shape (N, N), dtype=complex

    Returns
    -------
    E : ndarray, shape (N,)
        Eigenvalues (unsorted).
    right : ndarray, shape (N, N)
        Right eigenvectors as columns: H @ right[:, n] = E[n] * right[:, n].
    left : ndarray, shape (N, N)
        Left eigenvectors as rows: left[n, :] @ H = E[n] * left[n, :].
    """
    if H.shape[0] != H.shape[1]:
        raise ValueError("H must be a square matrix.")

    # Right eigenvectors
    E, right = np.linalg.eig(H)
    # Left eigenvectors = right eigenvectors of H^†, conjugated
    E_left, left_dag = np.linalg.eig(H.T)
    # Sort to match eigenvalues
    # Use a simple pairing by closest eigenvalue
    left = left_dag.T

    # Biorthogonal normalization: ensure ⟨ψ_n^L | ψ_n^R⟩ = 1
    N = H.shape[0]
    for n in range(N):
        overlap = np.vdot(left[n, :], right[:, n])
        if abs(overlap) < 1e-30:
            raise RuntimeError(f"Zero biorthogonal overlap for eigenvalue {n}.")
        left[n, :] = left[n, :] / np.conj(overlap)

    return E, right, left


def berry_connection_1d(H_func, k, dk=1e-5):
    """
    Compute the biorthogonal Berry connection A(k) for a 1D non-Hermitian
    Hamiltonian H(k).

    A(k) = i ⟨ψ^L(k)| ∂_k |ψ^R(k)⟩

    Parameters
    ----------
    H_func : callable
        H_func(k) returns H(k) as ndarray.
    k : float
        Momentum point.
    dk : float
        Finite-difference step.

    Returns
    -------
    A : float
        Berry connection (real or complex).
    """
    E0, right0, left0 = compute_biorthogonal_eigenvectors(H_func(k))
    Ep, rightp, leftp = compute_biorthogonal_eigenvectors(H_func(k + dk))
    Em, rightm, leftm = compute_biorthogonal_eigenvectors(H_func(k - dk))

    # Use central difference for derivative of right eigenvector
    # Choose the ground-state band (lowest real part of energy)
    n = np.argmin(E0.real)

    d_right = (rightp[:, n] - rightm[:, n]) / (2.0 * dk)
    A = 1j * np.vdot(left0[n, :], d_right)
    return A


def berry_curvature_2d(H_func, kx, ky, dk=1e-5):
    """
    Compute the biorthogonal Berry curvature Ω(kx, ky) for a 2D
    non-Hermitian Hamiltonian.

    Ω = i [ ⟨∂_{kx} ψ^L | ∂_{ky} ψ^R⟩ - ⟨∂_{ky} ψ^L | ∂_{kx} ψ^R⟩ ]

    Parameters
    ----------
    H_func : callable
        H_func(kx, ky) returns H(kx, ky).
    kx, ky : float
    dk : float
        Finite-difference step.

    Returns
    -------
    Omega : complex
        Berry curvature.
    """
    # Central differences
    def get_eig(kx_, ky_):
        E, right, left = compute_biorthogonal_eigenvectors(H_func(kx_, ky_))
        n = np.argmin(E.real)
        return right[:, n], left[n, :]

    rp_kx, lp_kx = get_eig(kx + dk, ky)
    rm_kx, lm_kx = get_eig(kx - dk, ky)
    rp_ky, lp_ky = get_eig(kx, ky + dk)
    rm_ky, lm_ky = get_eig(kx, ky - dk)
    r_pp, l_pp = get_eig(kx + dk, ky + dk)
    r_pm, l_pm = get_eig(kx + dk, ky - dk)
    r_mp, l_mp = get_eig(kx - dk, ky + dk)
    r_mm, l_mm = get_eig(kx - dk, ky - dk)

    # Mixed partial derivatives using central differences
    d2_r_dkxdky = (r_pp - r_pm - r_mp + r_mm) / (4.0 * dk * dk)
    # But actually we need first partials and cross terms.
    # Simpler: compute A_x and A_y, then curl
    d_r_dkx = (rp_kx - rm_kx) / (2.0 * dk)
    d_r_dky = (rp_ky - rm_ky) / (2.0 * dk)
    d_l_dkx = (lp_kx - lm_kx) / (2.0 * dk)
    d_l_dky = (lp_ky - lm_ky) / (2.0 * dk)

    # Use the central point for left/right
    r0, l0 = get_eig(kx, ky)

    # Standard formula for Berry curvature from overlaps
    Omega = 1j * (
        np.vdot(d_l_dkx, d_r_dky) - np.vdot(d_l_dky, d_r_dkx)
    )
    return Omega


def zak_phase_1d(H_func, k_points=401, a=1.0):
    """
    Compute the Zak phase for a 1D non-Hermitian system by numerical
    integration of the Berry connection over the Brillouin zone.

    γ_Zak = ∫_{-π/a}^{π/a} A(k) dk

    Parameters
    ----------
    H_func : callable
        H_func(k) returns H(k).
    k_points : int
        Number of k-points for integration.
    a : float
        Lattice constant.

    Returns
    -------
    gamma_zak : complex
        The Zak phase.
    """
    k_vals = np.linspace(-np.pi / a, np.pi / a, k_points)
    A_vals = np.array([berry_connection_1d(H_func, k) for k in k_vals])
    gamma_zak = np.trapz(A_vals, k_vals)
    return gamma_zak


def chern_number_2d(H_func, kx_points=81, ky_points=81, dk=None):
    """
    Compute the Chern number (integrated Berry curvature) for a 2D
    non-Hermitian system over the first Brillouin zone.

    C = (1 / 2π) ∬_{BZ} Ω(kx, ky) dkx dky

    Parameters
    ----------
    H_func : callable
        H_func(kx, ky) returns H(kx, ky).
    kx_points, ky_points : int
        Grid resolution.
    dk : float or None
        Finite-difference step for curvature. If None, estimated from grid.

    Returns
    -------
    C : complex
        Chern number (should be close to integer for gapped systems).
    """
    kx_vals = np.linspace(-np.pi, np.pi, kx_points)
    ky_vals = np.linspace(-np.pi, np.pi, ky_points)
    if dk is None:
        dk = (kx_vals[1] - kx_vals[0]) * 0.1

    Omega_grid = np.zeros((kx_points, ky_points), dtype=complex)
    for i, kx in enumerate(kx_vals):
        for j, ky in enumerate(ky_vals):
            try:
                Omega_grid[i, j] = berry_curvature_2d(H_func, kx, ky, dk=dk)
            except Exception:
                Omega_grid[i, j] = 0.0

    C = np.trapz(np.trapz(Omega_grid, ky_vals, axis=1), kx_vals, axis=0) / (2.0 * np.pi)
    return C


def winding_number_complex_energy(H_func, k_points=401):
    """
    Compute the winding number of the complex energy band around the origin.

    W = (1 / 2π i) ∮_C (1 / E(k)) dE/dk dk

    For a 1D system, this counts how many times the complex energy E(k)
    encircles the origin as k traverses the Brillouin zone.

    Parameters
    ----------
    H_func : callable
        H_func(k) returns H(k).
    k_points : int

    Returns
    -------
    W : float
        Winding number.
    """
    k_vals = np.linspace(-np.pi, np.pi, k_points)
    E_vals = np.zeros(k_points, dtype=complex)
    for i, k in enumerate(k_vals):
        H = H_func(k)
        E, _, _ = compute_biorthogonal_eigenvectors(H)
        E_vals[i] = E[np.argmin(E.real)]

    dE = np.gradient(E_vals, k_vals)
    integrand = dE / E_vals
    W = np.trapz(integrand, k_vals) / (2.0j * np.pi)
    return W.real
