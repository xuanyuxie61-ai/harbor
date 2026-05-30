
import numpy as np
from typing import List, Tuple, Dict, Optional


class FissionPathwayGraph:
    
    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self.adjacency = {i: [] for i in range(n_nodes)}
        self.edge_weights = {}
        self.node_positions = {}
    
    def add_edge(self, i: int, j: int, weight: float):
        if 0 <= i < self.n_nodes and 0 <= j < self.n_nodes:
            if j not in self.adjacency[i]:
                self.adjacency[i].append(j)
            self.edge_weights[(i, j)] = weight
    
    def set_node_position(self, node: int, position: np.ndarray):
        self.node_positions[node] = position.copy()
    
    def out_degree(self, node: int) -> int:
        return len(self.adjacency.get(node, []))
    
    def in_degree(self, node: int) -> int:
        count = 0
        for i in range(self.n_nodes):
            if node in self.adjacency.get(i, []):
                count += 1
        return count
    
    def degree_sequence(self) -> Tuple[np.ndarray, np.ndarray]:
        indeg = np.zeros(self.n_nodes, dtype=int)
        outdeg = np.zeros(self.n_nodes, dtype=int)
        for i in range(self.n_nodes):
            outdeg[i] = self.out_degree(i)
            indeg[i] = self.in_degree(i)
        return indeg, outdeg


def digraph_is_eulerian(graph: FissionPathwayGraph) -> int:
    indeg, outdeg = graph.degree_sequence()
    n_plus = 0
    n_minus = 0
    
    for i in range(graph.n_nodes):
        if indeg[i] == outdeg[i]:
            continue
        elif n_plus == 0 and indeg[i] == outdeg[i] + 1:
            n_plus = 1
        elif n_minus == 0 and indeg[i] == outdeg[i] - 1:
            n_minus = 1
        else:
            return 0
    
    if n_plus == 0 and n_minus == 0:
        return 2
    elif n_plus == 1 and n_minus == 1:
        return 1
    else:
        return 0


def dijkstra_min_energy_path(
    graph: FissionPathwayGraph,
    source: int,
    target: int,
) -> Tuple[List[int], float]:
    n = graph.n_nodes
    dist = np.full(n, np.inf)
    prev = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    
    dist[source] = 0.0
    
    for _ in range(n):

        u = -1
        min_dist = np.inf
        for i in range(n):
            if not visited[i] and dist[i] < min_dist:
                min_dist = dist[i]
                u = i
        
        if u == -1:
            break
        visited[u] = True
        
        for v in graph.adjacency.get(u, []):
            if not visited[v]:
                w = graph.edge_weights.get((u, v), 0.0)
                if w <= 0:
                    cost = np.inf
                else:
                    cost = -np.log(w)
                alt = dist[u] + cost
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
    

    if dist[target] == np.inf:
        return [], np.inf
    
    path = []
    u = target
    while u != -1:
        path.append(u)
        u = prev[u]
    path.reverse()
    return path, float(dist[target])


def build_fission_network_from_pes(
    mass_number: int,
    charge_number: int,
    excitation_energy: float,
    n_grid_beta2: int = 15,
    n_grid_beta3: int = 15,
) -> FissionPathwayGraph:
    from potential_energy_surface import potential_energy
    from diffusion_coefficient import nuclear_temperature
    
    T = nuclear_temperature(excitation_energy, mass_number)
    if T < 0.1:
        T = 0.1
    
    beta2_min, beta2_max = -0.3, 2.5
    beta3_min, beta3_max = -1.2, 1.2
    
    b2_grid = np.linspace(beta2_min, beta2_max, n_grid_beta2)
    b3_grid = np.linspace(beta3_min, beta3_max, n_grid_beta3)
    

    V_grid = np.zeros((n_grid_beta2, n_grid_beta3))
    for i, b2 in enumerate(b2_grid):
        for j, b3 in enumerate(b3_grid):
            q = np.array([b2, b3, 0.0, 0.0, 0.0])
            V_grid[i, j] = potential_energy(q, mass_number, charge_number)
    

    nodes = []
    node_indices = {}
    for i in range(1, n_grid_beta2 - 1):
        for j in range(1, n_grid_beta3 - 1):
            V_center = V_grid[i, j]
            neighbors = [
                V_grid[i - 1, j], V_grid[i + 1, j],
                V_grid[i, j - 1], V_grid[i, j + 1],
            ]
            if all(V_center <= v for v in neighbors):
                idx = len(nodes)
                nodes.append((i, j))
                node_indices[(i, j)] = idx
    
    graph = FissionPathwayGraph(len(nodes))
    
    for idx, (i, j) in enumerate(nodes):
        pos = np.array([b2_grid[i], b3_grid[j]])
        graph.set_node_position(idx, pos)
    

    for idx, (i, j) in enumerate(nodes):
        V_i = V_grid[i, j]

        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if (ni, nj) in node_indices:
                nidx = node_indices[(ni, nj)]
                V_j = V_grid[ni, nj]

                delta_V = V_j - V_i
                w = np.exp(-delta_V / T)
                w = np.clip(w, 1e-10, 1.0)
                graph.add_edge(idx, nidx, w)
    
    return graph


def pathway_entropy(graph: FissionPathwayGraph, source: int) -> np.ndarray:
    n = graph.n_nodes
    entropy = np.zeros(n)
    for i in range(n):
        neighbors = graph.adjacency.get(i, [])
        if not neighbors:
            entropy[i] = 0.0
            continue
        total_w = sum(graph.edge_weights.get((i, j), 0.0) for j in neighbors)
        if total_w <= 0:
            entropy[i] = 0.0
            continue
        S = 0.0
        for j in neighbors:
            w = graph.edge_weights.get((i, j), 0.0)
            if w > 0:
                p = w / total_w
                S -= p * np.log(p)
        entropy[i] = S
    return entropy
