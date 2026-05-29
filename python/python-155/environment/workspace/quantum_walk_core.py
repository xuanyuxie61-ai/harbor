"""
Core quantum walk dynamics engine.
Implements discrete-time coined quantum walks and continuous-time quantum walks.
"""
import numpy as np
from typing import List, Optional, Tuple
from utils import normalize_vector, clamp
from quantum_operators import (
    hadamard_coin, grover_coin, fourier_coin, shift_operator_1d,
    shift_operator_graph, unitary_evolution, ctqw_hamiltonian,
    ctqw_hamiltonian_with_marked, oracle_operator
)


# ---------------------------------------------------------------------------
# Discrete-Time Quantum Walk (DTQW)
# ---------------------------------------------------------------------------
class DiscreteTimeQuantumWalk:
    """Discrete-time coined quantum walk on 1D line or arbitrary graph."""

    def __init__(self, n: int, coin_dim: int = 2,
                 graph_adj: Optional[List[List[int]]] = None,
                 coin_type: str = "hadamard",
                 periodic: bool = True):
        self.n = n
        self.coin_dim = coin_dim
        self.periodic = periodic
        self.graph_adj = graph_adj
        self.state = np.zeros(n * coin_dim, dtype=complex)

        # Build coin operator
        if coin_type == "hadamard":
            self.C = hadamard_coin(coin_dim)
        elif coin_type == "grover":
            self.C = grover_coin(coin_dim)
        elif coin_type == "fourier":
            self.C = fourier_coin(coin_dim)
        else:
            self.C = hadamard_coin(coin_dim)

        # Build shift operator
        if graph_adj is not None:
            self.S = shift_operator_graph(graph_adj, coin_dim=coin_dim)
        else:
            self.S = shift_operator_1d(n, coin_dim=coin_dim, periodic=periodic)

        # Full evolution operator W = S @ (C ⊗ I)
        self._build_evolution_operator()

    def _build_evolution_operator(self):
        """Construct W = S @ (C ⊗ I_P)."""
        N = self.n * self.coin_dim
        C_full = np.zeros((N, N), dtype=complex)
        for c1 in range(self.coin_dim):
            for c2 in range(self.coin_dim):
                val = self.C[c1, c2]
                for x in range(self.n):
                    idx1 = c1 * self.n + x
                    idx2 = c2 * self.n + x
                    C_full[idx1, idx2] = val
        self.W = self.S @ C_full

    def set_initial_state(self, position: int = 0, coin_state: Optional[np.ndarray] = None):
        """Set initial state localized at position with given coin state."""
        self.state[:] = 0.0
        if coin_state is None:
            coin_state = normalize_vector(np.ones(self.coin_dim, dtype=complex))
        for c in range(self.coin_dim):
            idx = c * self.n + position
            self.state[idx] = coin_state[c]

    def step(self, num_steps: int = 1):
        """Evolve the quantum walk for num_steps."""
        for _ in range(num_steps):
            self.state = self.W @ self.state

    def get_position_distribution(self) -> np.ndarray:
        """Return probability distribution over position space."""
        prob = np.zeros(self.n, dtype=float)
        for x in range(self.n):
            for c in range(self.coin_dim):
                idx = c * self.n + x
                prob[x] += np.abs(self.state[idx]) ** 2
        return prob

    def get_state_norm(self) -> float:
        """Check norm conservation (should be 1.0)."""
        return float(np.linalg.norm(self.state))

    def get_coin_distribution_at(self, position: int) -> np.ndarray:
        """Return coin-state distribution at a given position."""
        dist = np.zeros(self.coin_dim, dtype=float)
        for c in range(self.coin_dim):
            idx = c * self.n + position
            dist[c] = np.abs(self.state[idx]) ** 2
        return dist


# ---------------------------------------------------------------------------
# Search-Enhanced Quantum Walk
# ---------------------------------------------------------------------------
class QuantumWalkSearch:
    """Quantum walk search algorithm with marked vertices."""

    def __init__(self, n: int, coin_dim: int = 2,
                 graph_adj: Optional[List[List[int]]] = None,
                 coin_type: str = "grover",
                 periodic: bool = True):
        self.n = n
        self.coin_dim = coin_dim
        self.graph_adj = graph_adj
        self.periodic = periodic

        # Base walk
        self.qw = DiscreteTimeQuantumWalk(n, coin_dim, graph_adj, coin_type, periodic)

        # Marked vertices
        self.marked = []
        self.oracle = None
        self.W_search = None

    def set_marked(self, marked_vertices: List[int]):
        """Set the marked vertices for search."""
        self.marked = [v for v in marked_vertices if 0 <= v < self.n]
        if not self.marked:
            self.W_search = self.qw.W.copy()
            return
        self.oracle = oracle_operator(self.n, self.coin_dim, self.marked, phase=np.pi)
        # Search operator: W_search = W @ O
        self.W_search = self.qw.W @ self.oracle

    def set_initial_state_uniform(self):
        """Set uniform superposition over all states."""
        self.qw.state = normalize_vector(np.ones(self.n * self.coin_dim, dtype=complex))

    def search_step(self, num_steps: int = 1):
        """Perform search evolution steps."""
        if self.W_search is None:
            raise RuntimeError("Marked vertices not set")
        for _ in range(num_steps):
            self.qw.state = self.W_search @ self.qw.state

    def get_success_probability(self) -> float:
        """Total probability of being in a marked state."""
        if not self.marked:
            return 0.0
        prob = 0.0
        for v in self.marked:
            for c in range(self.coin_dim):
                idx = c * self.n + v
                prob += np.abs(self.qw.state[idx]) ** 2
        return float(clamp(prob, 0.0, 1.0))

    def get_optimal_steps_estimate(self) -> int:
        """Estimate optimal number of steps using spectral properties.
        For Grover coin on regular graph, optimal steps ~ pi/4 * sqrt(N/M).
        """
        if not self.marked:
            return 0
        N = self.n * self.coin_dim
        M = len(self.marked)
        return int(np.ceil(0.25 * np.pi * np.sqrt(N / M)))


# ---------------------------------------------------------------------------
# Continuous-Time Quantum Walk (CTQW)
# ---------------------------------------------------------------------------
class ContinuousTimeQuantumWalk:
    """Continuous-time quantum walk on a graph."""

    def __init__(self, adj: List[List[int]], gamma: float = 1.0):
        self.adj = adj
        self.n = len(adj)
        self.gamma = gamma
        self.state = np.zeros(self.n, dtype=complex)
        self.H = ctqw_hamiltonian(adj, gamma=gamma)

    def set_initial_state(self, vertex: int):
        """Set initial state localized at a vertex."""
        self.state[:] = 0.0
        if 0 <= vertex < self.n:
            self.state[vertex] = 1.0

    def evolve(self, t: float):
        """Evolve for time t: psi(t) = exp(-i H t) psi(0)."""
        U = unitary_evolution(self.H, t)
        self.state = U @ self.state

    def get_position_distribution(self) -> np.ndarray:
        """Return probability distribution over vertices."""
        return np.abs(self.state) ** 2

    def get_state_norm(self) -> float:
        return float(np.linalg.norm(self.state))


# ---------------------------------------------------------------------------
# CTQW Search
# ---------------------------------------------------------------------------
class CTQWSearch:
    """Continuous-time quantum walk search with oracle Hamiltonian."""

    def __init__(self, adj: List[List[int]], marked_vertices: List[int],
                 gamma: float = 1.0):
        self.adj = adj
        self.n = len(adj)
        self.marked = [v for v in marked_vertices if 0 <= v < self.n]
        self.gamma = gamma
        self.state = np.zeros(self.n, dtype=complex)
        self.H = ctqw_hamiltonian_with_marked(adj, self.marked, gamma)

    def set_initial_state_uniform(self):
        self.state = normalize_vector(np.ones(self.n, dtype=complex))

    def evolve(self, t: float):
        U = unitary_evolution(self.H, t)
        self.state = U @ self.state

    def get_success_probability(self) -> float:
        prob = 0.0
        for v in self.marked:
            prob += np.abs(self.state[v]) ** 2
        return float(clamp(prob, 0.0, 1.0))

    def get_optimal_time_estimate(self) -> float:
        """Estimate optimal evolution time using spectral gap."""
        # TODO: Estimate the optimal evolution time for CTQW search.
        # HINT: Compute the spectral gap of self.H (difference between two smallest eigenvalues).
        # If the gap is too small, fall back to heuristic: 0.25 * pi * sqrt(n / M).
        # Otherwise return pi / (2.0 * gap).
        raise NotImplementedError("Hole 2: get_optimal_time_estimate not implemented")


# ---------------------------------------------------------------------------
# Multi-coin quantum walk on high-dimensional lattice
# ---------------------------------------------------------------------------
class MultiDimensionalQuantumWalk:
    """Quantum walk on a d-dimensional Cartesian lattice."""

    def __init__(self, dims: Tuple[int, ...], coin_type: str = "hadamard",
                 periodic: bool = True):
        self.dims = dims
        self.dim = len(dims)
        self.periodic = periodic
        self.n = int(np.prod(dims))
        self.coin_dim = 2 * self.dim  # One pair of directions per dimension

        # Build coin
        if coin_type == "hadamard":
            self.C = hadamard_coin(self.coin_dim)
        elif coin_type == "grover":
            self.C = grover_coin(self.coin_dim)
        elif coin_type == "fourier":
            self.C = fourier_coin(self.coin_dim)
        else:
            self.C = hadamard_coin(self.coin_dim)

        self.state = np.zeros(self.n * self.coin_dim, dtype=complex)
        self._build_shift_operator()
        self._build_evolution_operator()

    def _build_shift_operator(self):
        """Shift operator for multi-dimensional lattice."""
        N = self.n * self.coin_dim
        self.S = np.zeros((N, N), dtype=complex)
        from utils import flat_to_tensor_index, tensor_index_to_flat
        for flat_idx in range(self.n):
            idx = flat_to_tensor_index(flat_idx, self.dims)
            for c in range(self.coin_dim):
                dim_dir = c // 2
                sign = 1 if c % 2 == 0 else -1
                new_idx = list(idx)
                new_idx[dim_dir] += sign
                if self.periodic:
                    new_idx[dim_dir] %= self.dims[dim_dir]
                else:
                    new_idx[dim_dir] = clamp(new_idx[dim_dir], 0, self.dims[dim_dir] - 1)
                if tuple(new_idx) != idx or self.periodic:
                    new_flat = tensor_index_to_flat(tuple(new_idx), self.dims)
                    from_idx = c * self.n + flat_idx
                    to_idx = c * self.n + new_flat
                    self.S[to_idx, from_idx] = 1.0

    def _build_evolution_operator(self):
        N = self.n * self.coin_dim
        C_full = np.zeros((N, N), dtype=complex)
        for c1 in range(self.coin_dim):
            for c2 in range(self.coin_dim):
                val = self.C[c1, c2]
                for x in range(self.n):
                    idx1 = c1 * self.n + x
                    idx2 = c2 * self.n + x
                    C_full[idx1, idx2] = val
        self.W = self.S @ C_full

    def set_initial_state(self, center: Optional[Tuple[int, ...]] = None):
        if center is None:
            center = tuple(d // 2 for d in self.dims)
        flat = 0
        try:
            from utils import tensor_index_to_flat
            flat = tensor_index_to_flat(center, self.dims)
        except Exception:
            flat = 0
        self.state[:] = 0.0
        for c in range(self.coin_dim):
            self.state[c * self.n + flat] = 1.0 / np.sqrt(self.coin_dim)

    def step(self, num_steps: int = 1):
        for _ in range(num_steps):
            self.state = self.W @ self.state

    def get_position_distribution(self) -> np.ndarray:
        prob = np.zeros(self.n, dtype=float)
        for x in range(self.n):
            for c in range(self.coin_dim):
                idx = c * self.n + x
                prob[x] += np.abs(self.state[idx]) ** 2
        return prob
