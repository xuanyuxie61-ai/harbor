"""
transfer_matrix.py
------------------
Non-Hermitian transfer-matrix methods for 1D and quasi-1D systems.

Adapted from seed project 1094_snakes_matrix (Markov transition matrix).

Scientific Background
=====================
In 1D non-Hermitian lattice models, the transfer matrix T(E) relates
wavefunction amplitudes across unit cells:

    [ ψ_{n+1} ]     [ ψ_n   ]
    [ ψ_n     ] = T(E) [ ψ_{n-1} ]

For a two-band system with Bloch Hamiltonian H(k), the transfer matrix
at energy E is a 2×2 complex matrix satisfying

    det( T(E) - e^{ik} I ) = 0.

The eigenvalues of T(E) are e^{±ik}, and the spectrum is determined by
the condition that |Tr T(E)| ≤ 2 for propagating (Bloch) states, while
|Tr T(E)| > 2 corresponds to evanescent states.

In non-Hermitian systems, the transfer matrix itself is generally
non-unitary, and its singular values determine the localization length
via the Furstenberg formula:

    1/ξ = lim_{N→∞} (1/N) Σ_{n=1}^{N} ln σ_max(T_n),

where σ_max is the largest singular value. This is closely related to
the non-Hermitian skin effect, where eigenstates accumulate at boundaries.

We also construct non-Hermitian Markov chains (continuous-time) where
the transition matrix L satisfies Σ_j L_{ij} = 0 with L_{ii} < 0,
and L is non-Hermitian when forward/backward rates differ.
"""

import numpy as np


def transfer_matrix_ssh(E, t1, t2, gamma):
    """
    Compute the transfer matrix for the non-Hermitian SSH model at
    energy E.

    H(k) = (t1 + t2 cos k) σ_x + t2 sin k σ_y + iγ σ_z

    The real-space recurrence for amplitudes ψ_A(n), ψ_B(n) yields
    the transfer matrix:

        T(E) = [ (E - iγ)/t2   -t1/t2 ]
               [ 1              0      ]

    for the A-sublattice basis.

    Parameters
    ----------
    E : complex
        Energy.
    t1, t2 : float
        Hopping amplitudes.
    gamma : float
        Non-Hermitian strength.

    Returns
    -------
    T : ndarray, shape (2, 2), dtype=complex
    """
    if abs(t2) < 1e-15:
        raise ValueError("t2 must be non-zero.")
    T = np.array([
        [(E - 1j * gamma) / t2, -t1 / t2],
        [1.0, 0.0]
    ], dtype=complex)
    return T


def spectrum_from_transfer_matrix(E_grid, t1, t2, gamma):
    """
    Determine whether energies in E_grid correspond to propagating
    or evanescent states by analyzing |Tr T(E)|.

    Returns the trace and discriminant for each energy.
    """
    traces = np.zeros(len(E_grid), dtype=complex)
    discriminants = np.zeros(len(E_grid), dtype=complex)
    for i, E in enumerate(E_grid):
        T = transfer_matrix_ssh(E, t1, t2, gamma)
        traces[i] = np.trace(T)
        discriminants[i] = np.trace(T) ** 2 - 4.0 * np.linalg.det(T)
    return traces, discriminants


def lyapunov_exponent_ssh(E, t1, t2, gamma, N=1000, seed=42):
    """
    Compute the largest Lyapunov exponent (inverse localization length)
    for the non-Hermitian SSH model with random disorder using the
    transfer-matrix product method.

    λ = lim_{N→∞} (1 / 2N) ln || Π_{n=1}^{N} T_n ||^2

    Parameters
    ----------
    E : complex
        Energy.
    t1, t2 : float
        Clean hoppings.
    gamma : float
        Non-Hermitian strength.
    N : int
        Number of unit cells.
    seed : int

    Returns
    -------
    lyap : float
        Largest Lyapunov exponent.
    """
    rng = np.random.default_rng(seed)
    # Disorder: t1 fluctuates by 10%
    t1_disorder = 0.1

    vec = np.array([1.0, 0.0], dtype=complex)
    norm_log = 0.0

    for _ in range(N):
        t1_n = t1 + t1_disorder * (rng.random() - 0.5)
        T = transfer_matrix_ssh(E, t1_n, t2, gamma)
        vec = T @ vec
        nv = np.linalg.norm(vec)
        if nv < 1e-30:
            return 1e308
        norm_log += np.log(nv)
        vec = vec / nv

    lyap = norm_log / N
    return lyap


def nonhermitian_markov_chain(N, p_forward, p_backward, loss_rate):
    """
    Construct a non-Hermitian continuous-time Markov transition matrix
    for a 1D chain with asymmetric hopping and uniform loss.

    L_{i,i+1} = p_forward      (jump to right)
    L_{i,i-1} = p_backward     (jump to left)
    L_{i,i}   = -(p_forward + p_backward + loss_rate)

    The matrix L is non-Hermitian when p_forward ≠ p_backward.
    The steady-state distribution satisfies L^T π = 0.

    Parameters
    ----------
    N : int
        Number of sites.
    p_forward, p_backward : float
        Hopping rates.
    loss_rate : float
        Uniform loss rate.

    Returns
    -------
    L : ndarray, shape (N, N), dtype=float
    """
    if N <= 1:
        raise ValueError("N must be > 1.")
    L = np.zeros((N, N))
    for i in range(N):
        if i > 0:
            L[i, i - 1] = p_backward
        if i < N - 1:
            L[i, i + 1] = p_forward
        L[i, i] = -(p_forward + p_backward + loss_rate)
    return L


def steady_state_distribution(L):
    """
    Compute the steady-state probability distribution π for a
    continuous-time Markov chain with transition matrix L.

    Solves L^T π = 0 with Σ_i π_i = 1.
    """
    N = L.shape[0]
    # Replace last equation with normalization
    M = L.T.copy()
    M[-1, :] = 1.0
    b = np.zeros(N)
    b[-1] = 1.0
    pi = np.linalg.solve(M, b)
    # Ensure non-negative
    pi = np.maximum(pi, 0.0)
    pi = pi / np.sum(pi)
    return pi
