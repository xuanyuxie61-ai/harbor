import numpy as np
from typing import List, Optional, Tuple
from quantum_walk_core import QuantumWalkSearch, CTQWSearch, DiscreteTimeQuantumWalk
from quantum_operators import spectral_gap, eigenstate_localization
from utils import normalize_vector





def estimate_search_complexity(n: int, num_marked: int,
                                graph_degree: float = 4.0) -> dict:
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





def spatial_search_2d_grid(nx: int, ny: int, marked: List[Tuple[int, int]],
                           max_steps: int = 200) -> dict:
    n = nx * ny
    marked_flat = [x + y * nx for x, y in marked]


    adj = []
    for y in range(ny):
        for x in range(nx):
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx2, ny2 = x + dx, y + dy
                if 0 <= nx2 < nx and 0 <= ny2 < ny:
                    neighbors.append(nx2 + ny2 * nx)
            adj.append(neighbors)


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





def spectral_search_analysis(adj: List[List[int]], marked: List[int]) -> dict:









    raise NotImplementedError("Hole 3: spectral_search_analysis not implemented")





def multi_target_search_phase_estimation(n: int, marked_sets: List[List[int]],
                                         num_steps_each: int = 50) -> dict:
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





def hexagonal_lattice_search(n_rings: int, marked: List[int],
                             max_steps: int = 100) -> dict:
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





def meshed_domain_search(boundary: np.ndarray, marked_nodes: List[int],
                         hmax: float = 0.5, max_steps: int = 100) -> dict:
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





def hypercube_search(dim: int, marked: List[int], max_steps: int = 200) -> dict:
    n = 2 ** dim

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
