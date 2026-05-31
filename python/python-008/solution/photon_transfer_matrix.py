"""
Photon Transfer Matrix Module
=============================
Based on seed project 1405_web_matrix:
- incidence_to_transition.m  →  incidence → transition matrix
- power_rank.m               →  power iteration for steady-state

Physics:
--------
In the GRB afterglow, photons undergo multiple scattering events
(comptonization) before escaping.  The photon state space is
partitioned into N energy bins; the probability of a photon
transitioning from bin i to bin j in one scattering is encoded
in an incidence matrix A.

The transition matrix is:

    T_{ji} = A_{ij} / Σ_k A_{ik}

so that each column of T sums to 1 (Markov chain).  The steady-state
photon occupation number n* satisfies:

    T · n* = n*

which is the eigenvector equation for eigenvalue λ = 1.  The power
method iterates:

    n_{k+1} = T · n_k

and converges to n* provided T is irreducible and aperiodic
(Perron-Frobenius theorem).

The mean number of scatterings before escape is:

    ⟨N_sc⟩ = Σ_i n*_i / (Σ_j A_{escape,j})

and the Compton-y parameter accumulated per photon is:

    y = Σ_i n*_i · (4 k T_e / (m_e c²)) · τ_{es,i}
"""

import numpy as np


def incidence_to_transition(A):
    """
    Converts an incidence matrix to a column-stochastic transition matrix.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Incidence matrix (nonnegative entries).

    Returns
    -------
    T : ndarray, shape (n, n)
        Transition matrix (columns sum to 1).
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    s = np.sum(A, axis=0)
    T = np.zeros((n, n), dtype=float)
    for i in range(n):
        if s[i] > 0.0:
            T[:, i] = A[:, i] / s[i]
        else:
            T[i, i] = 1.0
    return T


def power_rank(A, max_iter=200, tol=1e-12):
    """
    Power iteration to find the dominant eigenvector of the transition
    matrix (eigenvalue 1).

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Incidence matrix.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    n_ss : ndarray
        Steady-state distribution.
    iterations : int
        Number of iterations performed.
    """
    T = incidence_to_transition(A)
    n = T.shape[0]
    x = np.ones(n, dtype=float) / n

    for it in range(max_iter):
        x_new = T @ x
        # Normalization not strictly needed for exact transition matrix,
        # but improves numerical stability
        norm = np.sum(x_new)
        if norm > 0.0:
            x_new = x_new / norm

        if np.linalg.norm(x_new - x, ord=1) < tol:
            x = x_new
            break
        x = x_new

    return x, it + 1


def build_compton_transfer_matrix(n_bins, T_e, tau_es, p_scatter=0.9):
    """
    Build a simplified Compton-scattering transfer matrix for GRB
    afterglow photons.

    Parameters
    ----------
    n_bins : int
        Number of energy bins.
    T_e : float
        Electron temperature (K).
    tau_es : float
        Electron-scattering optical depth.
    p_scatter : float
        Probability of scattering (vs. escape).

    Returns
    -------
    A : ndarray, shape (n_bins+1, n_bins+1)
        Incidence matrix (last row/column = escape state).
    """
    k_B = 1.380649e-16  # erg/K
    m_e_c2 = 8.18710565e-7  # erg
    eps = 4.0 * k_B * T_e / m_e_c2
    eps = max(eps, 1e-6)

    n = n_bins + 1
    A = np.zeros((n, n), dtype=float)

    for i in range(n_bins):
        # Up-scattering to higher energy bins
        for j in range(i + 1, n_bins):
            # Klein-Nishina-weighted upscattering probability
            delta = j - i
            prob = np.exp(-delta / (eps * (i + 1))) / (delta + 1.0) ** 2
            A[j, i] = prob

        # Down-scattering
        for j in range(max(0, i - 3), i):
            A[j, i] = 0.1 / (i - j + 1.0)

        # Self-scattering
        A[i, i] = 0.2

        # Escape probability
        A[n_bins, i] = (1.0 - p_scatter) * np.exp(-tau_es)

    # Escape state is absorbing
    A[n_bins, n_bins] = 1.0

    # Normalize columns
    for i in range(n_bins):
        col_sum = np.sum(A[:, i])
        if col_sum > 0.0:
            A[:, i] /= col_sum

    return A


def compute_photon_stats(A):
    """
    Compute steady-state photon statistics.

    Returns
    -------
    dict with keys: steady_state, mean_scatterings, y_param
    """
    n_ss, iters = power_rank(A)
    n_bins = A.shape[0] - 1

    # Mean scatterings before escape
    p_escape = A[n_bins, :n_bins]
    mean_scat = np.sum(n_ss[:n_bins]) / (np.sum(p_escape * n_ss[:n_bins]) + 1e-30)

    # Approximate Compton-y parameter
    y_param = np.sum(n_ss[:n_bins] * np.arange(1, n_bins + 1))

    return {
        "steady_state": n_ss,
        "mean_scatterings": mean_scat,
        "y_param": y_param,
        "iterations": iters,
    }
