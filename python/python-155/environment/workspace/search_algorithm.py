"""
Quantum walk search algorithms with advanced analysis.
Implements Grover-like search on graphs using both DTQW and CTQW.
"""
import numpy as np
from typing import List, Optional, Tuple
from quantum_walk_core import QuantumWalkSearch, CTQWSearch, DiscreteTimeQuantumWalk
from quantum_operators import spectral_gap, eigenstate_localization
from utils import normalize_vector


# ---------------------------------------------------------------------------
# Optimal search parameter analysis
# ---------------------------------------------------------------------------
def estimate_search_complexity(n: int, num_marked: int,
                                graph_degree: float = 4.0) -> dict:
    """Estimate quantum search complexity on a regular graph.
    For d-regular graphs with M marked vertices:
      - Optimal steps T* ~ (pi/4) * sqrt(N / (M * d))
      - Success probability P* ~ 1 - O(M/N)
      - Classical comparison: O(N/M)
    """
    if num_marked <= 0 or n <= 0:
        return {"optimal_steps": 0, "success_prob": 0.0, "speedup": 1.0}
    N = n
    M = num_marked
    T_star = 0.25 * np.pi * np.sqrt(N / (M * graph_degree))
    P_star = 1.0 - M / N if M < N else 1.0
    classical = N / M
    quantum = T_star / P_star if P_star > 0 else float('inf')
    speedup = classical / quantum if quantum > 0 else 1.0
    return {
        "optimal_steps": int(np.ceil(T_star)),
        "optimal_time": float(T_star),
        "success_probability": float(P_star),
        "classical_complexity": float(classical),
        "quantum_complexity": float(quantum),
        "quadratic_speedup": float(speedup)
    }


# ---------------------------------------------------------------------------
# Spatial search on 2D grid with optimal coin tuning
# ---------------------------------------------------------------------------
def spatial_search_2d_grid(nx: int, ny: int, marked: List[Tuple[int, int]],
                           max_steps: int = 200) -> dict:
    """Perform quantum walk search on a 2D grid.
    For 2D grids, the standard Grover speedup is lost due to recurrence,
    but can be recovered with proper coin tuning (Ambainis, Kempe, Rivosh).
    We use the AKR coin: C = -I + 2|s><s| where |s> is uniform.
    """
    n = nx * ny
    marked_flat = [x + y * nx for x, y in marked]

    # Build grid adjacency
    adj = []
    for y in range(ny):
        for x in range(nx):
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx2, ny2 = x + dx, y + dy
                if 0 <= nx2 < nx and 0 <= ny2 < ny:
                    neighbors.append(nx2 + ny2 * nx)
            adj.append(neighbors)

    # Use Grover coin (optimal for spatial search)
    search = QuantumWalkSearch(n, coin_dim=4, graph_adj=adj, coin_type="grover")
    search.set_marked(marked_flat)
    search.set_initial_state_uniform()

    probs = []
    for t in range(max_steps + 1):
        if t > 0:
            search.search_step(1)
        probs.append(search.get_success_probability())

    probs = np.array(probs)
    t_opt = int(np.argmax(probs))
    p_opt = float(probs[t_opt])

    return {
        "optimal_steps": t_opt,
        "max_success_probability": p_opt,
        "probability_evolution": probs.tolist(),
        "marked_vertices": marked,
        "grid_size": (nx, ny)
    }


# ---------------------------------------------------------------------------
# Search on arbitrary graph with spectral analysis
# ---------------------------------------------------------------------------
def spectral_search_analysis(adj: List[List[int]], marked: List[int]) -> dict:
    """Analyze the search Hamiltonian spectrum to predict search performance.
    The key insight: search succeeds when the marked state component
    of the ground state is large, which occurs near the critical gamma value:
      gamma_c = 1 / N * sum_{k>0} 1 / lambda_k
    where lambda_k are the nonzero eigenvalues of the graph Laplacian.
    """
    # TODO: Implement spectral search analysis.
    # HINT:
    # 1. Build graph Laplacian L from adjacency list.
    # 2. Compute its eigenvalues. Critical gamma: gamma_c = (1/n) * sum_{k>0} 1/lambda_k.
    # 3. Build Hamiltonian H = gamma_c * L, then subtract 1.0 from diagonal at marked vertices.
    # 4. Compute spectral gap of H (eigs_H[1] - eigs_H[0]).
    # 5. Compute adiabatic time T_ad = pi / (2*gap) and success probability bound.
    # 6. Return a dict with keys: critical_gamma, spectral_gap, adiabatic_time,
    #    success_bound, laplacian_eigenvalues.
    raise NotImplementedError("Hole 3: spectral_search_analysis not implemented")


# ---------------------------------------------------------------------------
# Multi-target search with phase estimation
# ---------------------------------------------------------------------------
def multi_target_search_phase_estimation(n: int, marked_sets: List[List[int]],
                                         num_steps_each: int = 50) -> dict:
    """Search for multiple target sets and estimate their relative phase structures.
    This simulates a quantum walk that sequentially searches for different targets.
    """
    results = []
    for targets in marked_sets:
        search = QuantumWalkSearch(n, coin_dim=2)
        search.set_marked(targets)
        search.set_initial_state_uniform()
        search.search_step(num_steps_each)
        prob = search.get_success_probability()
        results.append({
            "targets": targets,
            "success_probability": prob,
            "num_targets": len(targets)
        })
    return {"searches": results}


# ---------------------------------------------------------------------------
# Quantum walk search on hexagonal lattice
# ---------------------------------------------------------------------------
def hexagonal_lattice_search(n_rings: int, marked: List[int],
                             max_steps: int = 100) -> dict:
    """Search on a hexagonal lattice (graphene-like structure)."""
    from geometry_mesh import generate_hexagonal_lattice, hexagonal_adjacency
    points = generate_hexagonal_lattice(n_rings)
    adj = hexagonal_adjacency(points)
    n = len(adj)

    search = QuantumWalkSearch(n, coin_dim=3, graph_adj=adj, coin_type="grover")
    valid_marked = [v for v in marked if 0 <= v < n]
    search.set_marked(valid_marked)
    search.set_initial_state_uniform()

    probs = []
    for t in range(max_steps + 1):
        if t > 0:
            search.search_step(1)
        probs.append(search.get_success_probability())

    probs = np.array(probs)
    t_opt = int(np.argmax(probs))
    return {
        "num_vertices": n,
        "optimal_steps": t_opt,
        "max_success_probability": float(probs[t_opt]),
        "probability_evolution": probs.tolist()
    }


# ---------------------------------------------------------------------------
# Quantum walk search on meshed domain
# ---------------------------------------------------------------------------
def meshed_domain_search(boundary: np.ndarray, marked_nodes: List[int],
                         hmax: float = 0.5, max_steps: int = 100) -> dict:
    """Search on a triangulated 2D domain."""
    from geometry_mesh import generate_2d_mesh, mesh_adjacency
    nodes, elems = generate_2d_mesh(boundary, hmax)
    adj = mesh_adjacency(nodes, elems)
    n = len(adj)

    search = QuantumWalkSearch(n, coin_dim=3, graph_adj=adj, coin_type="grover")
    valid_marked = [v for v in marked_nodes if 0 <= v < n]
    search.set_marked(valid_marked)
    search.set_initial_state_uniform()

    probs = []
    for t in range(max_steps + 1):
        if t > 0:
            search.search_step(1)
        probs.append(search.get_success_probability())

    probs = np.array(probs)
    t_opt = int(np.argmax(probs))
    return {
        "num_vertices": n,
        "num_elements": elems.shape[0],
        "optimal_steps": t_opt,
        "max_success_probability": float(probs[t_opt]),
        "probability_evolution": probs.tolist()
    }


# ---------------------------------------------------------------------------
# High-dimensional hypercube search
# ---------------------------------------------------------------------------
def hypercube_search(dim: int, marked: List[int], max_steps: int = 200) -> dict:
    """Quantum walk search on a dim-dimensional hypercube.
    For hypercubes, the optimal quantum walk search achieves
    T* ~ (pi/4) * sqrt(2^dim / M) with high probability.
    """
    n = 2 ** dim
    # Hypercube adjacency
    adj = []
    for v in range(n):
        neighbors = []
        for d in range(dim):
            neighbors.append(v ^ (1 << d))
        adj.append(neighbors)

    search = QuantumWalkSearch(n, coin_dim=dim, graph_adj=adj, coin_type="grover")
    valid_marked = [v for v in marked if 0 <= v < n]
    search.set_marked(valid_marked)
    search.set_initial_state_uniform()

    probs = []
    for t in range(max_steps + 1):
        if t > 0:
            search.search_step(1)
        probs.append(search.get_success_probability())

    probs = np.array(probs)
    t_opt = int(np.argmax(probs))
    p_opt = float(probs[t_opt])
    theoretical_opt = int(0.25 * np.pi * np.sqrt(n / max(len(valid_marked), 1)))

    return {
        "dimension": dim,
        "num_vertices": n,
        "optimal_steps": t_opt,
        "theoretical_optimal": theoretical_opt,
        "max_success_probability": p_opt,
        "probability_evolution": probs.tolist()
    }
