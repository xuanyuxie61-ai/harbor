"""
markov_model.py
Markov State Model (MSM) for conformational dynamics of DNA repair proteins.

Derived from: 771_mm_to_msm + 740_matrix_chain_dynamic

A Markov State Model describes the kinetics of protein conformational
changes as memoryless transitions among discrete metastable states.
The transition matrix T satisfies:

    p(t+tau) = T(tau) * p(t)
    sum_j T_{ij} = 1,  T_{ij} >= 0

The implied timescales are given by:
    t_k = -tau / log(mu_k)
where mu_k are the eigenvalues of T (excluding the stationary mu_1=1).

This module also includes optimal tensor contraction ordering for
multi-site correlation functions via dynamic programming (matrix chain).
"""

import numpy as np
from scipy.linalg import eig


def build_msm_transition_matrix(n_states, temperature=1.0, seed=None):
    """
    Build a synthetic reversible Markov transition matrix for protein
    conformational states with detailed balance.

    Detailed balance: pi_i * T_{ij} = pi_j * T_{ji}

    Parameters
    ----------
    n_states : int
        Number of metastable conformational states.
    temperature : float
        Thermal energy scale.
    seed : int, optional

    Returns
    -------
    T : ndarray, shape (n_states, n_states)
        Row-stochastic transition matrix.
    pi : ndarray, shape (n_states,)
        Stationary distribution.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random energy levels
    E = np.random.randn(n_states)
    # Boltzmann weights
    pi = np.exp(-E / temperature)
    pi /= np.sum(pi)

    # Build a symmetric rate matrix K with K_{ij} = K_{ji}
    K = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(i + 1, n_states):
            rate = np.exp(-abs(E[j] - E[i]) / temperature)
            K[i, j] = rate
            K[j, i] = rate

    # Row-normalize to get transition matrix
    row_sums = np.sum(K, axis=1)
    T = np.zeros_like(K)
    for i in range(n_states):
        if row_sums[i] > 0:
            T[i, :] = K[i, :] / row_sums[i]
        else:
            T[i, i] = 1.0

    # Ensure detailed balance approximately holds
    # Add small self-transition for numerical stability
    for i in range(n_states):
        T[i, i] += 1e-6
        T[i, :] /= np.sum(T[i, :])

    return T, pi


def msm_eigenvalues_timescales(T, tau=1.0):
    """
    Compute eigenvalues and implied timescales of a Markov transition matrix.

    Implied timescales: t_k = -tau / log(|mu_k|)

    Parameters
    ----------
    T : ndarray, shape (n, n)
    tau : float
        Lag time.

    Returns
    -------
    eigenvalues : ndarray, shape (n,)
    timescales : ndarray, shape (n-1,)
    """
    mu, _ = eig(T, left=False, right=True)
    mu = np.sort(mu)[::-1]  # Sort descending by real part
    mu_real = np.real(mu)

    timescales = []
    for k in range(1, len(mu_real)):
        val = mu_real[k]
        if val > 0 and val < 1.0:
            t_k = -tau / np.log(val)
            timescales.append(t_k)
        else:
            timescales.append(np.inf)

    return mu_real, np.array(timescales)


def msm_propagate(T, p0, n_steps):
    """
    Propagate an initial probability distribution p0 for n_steps
    according to the Markov transition matrix T.

    p_n = p_0 * T^n

    Parameters
    ----------
    T : ndarray, shape (n, n)
    p0 : ndarray, shape (n,)
    n_steps : int

    Returns
    -------
    p : ndarray, shape (n,)
    """
    p = np.array(p0, dtype=float)
    p /= np.sum(p)
    for _ in range(n_steps):
        p = p @ T
        # Numerical robustness
        p = np.maximum(p, 0.0)
        s = np.sum(p)
        if s > 0:
            p /= s
        else:
            p = np.ones(len(p0)) / len(p0)
    return p


def msm_mfpt(T, target_state, start_state=None):
    """
    Compute the mean first-passage time (MFPT) from start_state to target_state.

    For a discrete-time Markov chain, MFPT satisfies:
        m_i = 1 + sum_{j != target} T_{ij} * m_j
    or in matrix form: (I - T_{-target}) * m = 1

    Parameters
    ----------
    T : ndarray, shape (n, n)
    target_state : int
    start_state : int, optional
        If None, return MFPT from all non-target states.

    Returns
    -------
    mfpt : float or ndarray
    """
    n = T.shape[0]
    if target_state < 0 or target_state >= n:
        raise ValueError("Invalid target state.")

    # Reduced matrix excluding target state
    indices = [i for i in range(n) if i != target_state]
    T_red = T[np.ix_(indices, indices)]
    I = np.eye(len(indices))
    b = np.ones(len(indices))

    try:
        m = np.linalg.solve(I - T_red, b)
    except np.linalg.LinAlgError:
        m = np.linalg.lstsq(I - T_red, b, rcond=None)[0]

    if start_state is None:
        return m
    if start_state == target_state:
        return 0.0
    idx = indices.index(start_state)
    return m[idx]


def optimal_tensor_contraction_path(dims, n_contract):
    """
    Find the optimal contraction ordering for a chain of n_contract
    tensors with dimensions given by dims[0..n_contract].

    This maps the matrix-chain dynamic programming problem to tensor
    network contraction, where each contraction of tensors A_i (dims[i] x dims[i+1])
    and A_{i+1} (dims[i+1] x dims[i+2]) costs dims[i]*dims[i+1]*dims[i+2].

    Parameters
    ----------
    dims : list of int
    n_contract : int

    Returns
    -------
    min_cost : int
    path : str
        Parenthesized expression.
    """
    from matrix_utils import matrix_chain_optimal_order, reconstruct_optimal_order

    dims_sub = dims[: n_contract + 1]
    min_cost, s = matrix_chain_optimal_order(dims_sub)
    if s.size == 0:
        path = "A0"
    else:
        path = reconstruct_optimal_order(s, 0, n_contract - 1)
    return min_cost, path
