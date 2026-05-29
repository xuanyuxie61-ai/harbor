"""
risk_fracture.py
================
Markov chain model for fracture network evolution and connectivity
in geothermal reservoirs.

Incorporates algorithms from:
  - 1026_risk_matrix: transition matrix and adjacency matrix operations

Mathematical formulation:
The fracture network is modeled as a Markov chain where each state
represents a fracture connectivity configuration.

For a fracture network with N fracture clusters, the transition
probability matrix P satisfies:

  P_{ij} = \text{Pr}(\text{state } j \text{ at } t+1 \mid \text{state } i \text{ at } t)

with \sum_{j} P_{ij} = 1 for all i.

The steady-state distribution \pi satisfies:
  \pi = P^T \pi, \quad \sum_i \pi_i = 1

For fracture aperture evolution under thermal stress cycling:
  a_{n+1} = a_n + \Delta a \cdot \text{sgn}(\sigma_{\text{thermal}})

The connectivity probability evolves as:
  C(t+1) = P \cdot C(t)

where C(t) is the vector of connectivity probabilities.
"""

import numpy as np


class FractureMarkovModel:
    """
    Markov chain model for fracture aperture and connectivity evolution.
    """

    def __init__(self, num_states=42):
        """
        Parameters
        ----------
        num_states : int
            Number of discrete fracture states.
        """
        self.num_states = int(num_states)
        self.adjacency = self._build_adjacency_matrix()
        self.transition = self._build_transition_matrix()

    def _build_adjacency_matrix(self):
        """
        Build a banded adjacency matrix representing nearest-neighbor
        fracture state transitions.
        """
        n = self.num_states
        A = np.zeros((n, n))
        # Band structure: each state connects to adjacent states
        for i in range(n):
            if i > 0:
                A[i, i - 1] = 1.0
            if i < n - 1:
                A[i, i + 1] = 1.0
            A[i, i] = 1.0  # self-connection
        return A

    def _build_transition_matrix(self):
        """
        Build row-stochastic transition matrix from adjacency.
        P_{ij} = A_{ij} / \sum_k A_{ik}
        """
        A = self.adjacency
        P = np.zeros_like(A)
        for i in range(A.shape[0]):
            row_sum = np.sum(A[i, :])
            if row_sum == 0:
                P[i, :] = 1.0 / A.shape[1]
            else:
                P[i, :] = A[i, :] / row_sum
        return P

    def steady_state(self, max_iter=10000, tol=1.0e-12):
        """
        Compute steady-state distribution via power iteration.

        Returns
        -------
        pi : np.ndarray
            Steady-state probability vector.
        """
        n = self.num_states
        pi = np.ones(n) / n
        P = self.transition

        for _ in range(max_iter):
            pi_new = P.T @ pi
            if np.linalg.norm(pi_new - pi, ord=1) < tol:
                break
            pi = pi_new

        pi = pi_new / np.sum(pi_new)
        return pi

    def evolve(self, initial_dist, num_steps):
        """
        Evolve probability distribution over num_steps.

        Parameters
        ----------
        initial_dist : np.ndarray
            Initial probability distribution.
        num_steps : int
            Number of time steps.

        Returns
        -------
        dist : np.ndarray
            Final distribution.
        history : np.ndarray, shape (num_steps+1, num_states)
            Distribution history.
        """
        initial_dist = np.asarray(initial_dist, dtype=np.float64)
        if initial_dist.size != self.num_states:
            raise ValueError("Initial distribution size must match num_states.")
        if abs(np.sum(initial_dist) - 1.0) > 1.0e-10:
            initial_dist = initial_dist / np.sum(initial_dist)

        P = self.transition
        dist = initial_dist.copy()
        history = [dist.copy()]

        for _ in range(num_steps):
            dist = P.T @ dist
            history.append(dist.copy())

        return dist, np.array(history)

    def mean_first_passage_time(self, target_state):
        """
        Compute mean first passage time to target_state using fundamental matrix.

        For an absorbing Markov chain with target_state as absorbing:
          Q = P_{transient}
          N = (I - Q)^{-1}
          t = N \mathbf{1}

        Returns
        -------
        mfpt : np.ndarray
            Mean first passage times from each transient state.
        """
        n = self.num_states
        target = int(target_state)
        if target < 0 or target >= n:
            raise ValueError("Invalid target state.")

        # Build transient submatrix
        transient = [i for i in range(n) if i != target]
        Q = self.transition[np.ix_(transient, transient)]
        I = np.eye(len(transient))
        N = np.linalg.solve(I - Q, I)
        ones = np.ones(len(transient))
        mfpt = N @ ones

        # Map back to full state space
        full_mfpt = np.zeros(n)
        for idx, state in enumerate(transient):
            full_mfpt[state] = mfpt[idx]
        full_mfpt[target] = 0.0
        return full_mfpt


def fracture_aperture_markov_evolution(a_initial, thermal_cycles, delta_a=1.0e-5,
                                       closure_prob=0.3, opening_prob=0.4):
    """
    Model fracture aperture evolution under thermal stress cycling
    as a discrete-state Markov process.

    Parameters
    ----------
    a_initial : float
        Initial aperture (m).
    thermal_cycles : int
        Number of thermal loading cycles.
    delta_a : float
        Aperture increment per state (m).
    closure_prob : float
        Probability of aperture decrease.
    opening_prob : float
        Probability of aperture increase.

    Returns
    -------
    a_history : np.ndarray
        Aperture history.
    """
    if delta_a <= 0:
        raise ValueError("delta_a must be positive.")
    if closure_prob < 0 or opening_prob < 0 or closure_prob + opening_prob > 1.0:
        raise ValueError("Probabilities must be non-negative and sum <= 1.")

    num_states = 20
    state = int(np.clip(a_initial / delta_a, 0, num_states - 1))

    # Transition: P(close) = closure_prob, P(open) = opening_prob, P(stay) = rest
    P = np.zeros((num_states, num_states))
    for i in range(num_states):
        if i > 0:
            P[i, i - 1] = closure_prob
        if i < num_states - 1:
            P[i, i + 1] = opening_prob
        P[i, i] = 1.0 - closure_prob - opening_prob
        # Normalize
        P[i, :] /= np.sum(P[i, :])

    a_history = np.zeros(thermal_cycles + 1)
    a_history[0] = state * delta_a

    for t in range(thermal_cycles):
        state = np.random.choice(num_states, p=P[state, :])
        a_history[t + 1] = state * delta_a

    return a_history


def effective_permeability_from_fracture_network(apertures, fracture_density,
                                                  matrix_perm, max_perm=1.0e-12):
    """
    Compute effective permeability using the cubic law for fracture flow
    combined with matrix permeability via the Snow (1969) model:

    k_{\text{eff}} = k_m + \frac{\rho_f g}{12 \mu} \sum_i a_i^3 b_i

    where a_i is aperture and b_i is fracture spacing.
    For a simplified model:

    k_{\text{eff}} = k_m + C \cdot \rho_{\text{frac}} \cdot \bar{a}^3

    Parameters
    ----------
    apertures : np.ndarray
        Fracture apertures (m).
    fracture_density : float
        Fracture density (1/m).
    matrix_perm : float
        Matrix permeability (m^2).
    max_perm : float
        Upper bound for permeability.

    Returns
    -------
    k_eff : float
        Effective permeability (m^2).
    """
    if fracture_density < 0 or matrix_perm < 0:
        raise ValueError("Densities and permeabilities must be non-negative.")

    a_mean = np.mean(apertures)
    # Cubic law coefficient: 1/12 for parallel plate
    k_frac = fracture_density * (a_mean ** 3) / 12.0
    k_eff = matrix_perm + k_frac
    k_eff = min(k_eff, max_perm)
    return k_eff
