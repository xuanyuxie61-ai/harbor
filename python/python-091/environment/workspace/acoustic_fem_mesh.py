
import numpy as np
from typing import List, Tuple, Set, Dict
from collections import deque


def build_adjacency_structure(nodes: np.ndarray, triangles: np.ndarray) -> Dict[int, Set[int]]:
    n_nodes = nodes.shape[0]
    adjacency = {i: set() for i in range(n_nodes)}
    
    for tri in triangles:
        i, j, k = tri
        adjacency[i].add(j)
        adjacency[i].add(k)
        adjacency[j].add(i)
        adjacency[j].add(k)
        adjacency[k].add(i)
        adjacency[k].add(j)
    
    return adjacency


def bfs_levels(adjacency: Dict[int, Set[int]], start: int) -> Tuple[Dict[int, int], int, int]:
    n_nodes = len(adjacency)
    visited = np.zeros(n_nodes, dtype=bool)
    levels = {}
    
    queue = deque([start])
    visited[start] = True
    levels[start] = 0
    max_level = 0
    
    while queue:
        node = queue.popleft()
        current_level = levels[node]
        
        for neighbor in adjacency[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                levels[neighbor] = current_level + 1
                max_level = max(max_level, current_level + 1)
                queue.append(neighbor)
    
    nodes_at_max = sum(1 for v in levels.values() if v == max_level)
    return levels, max_level, nodes_at_max


def find_pseudo_peripheral(adjacency: Dict[int, Set[int]]) -> int:
    n_nodes = len(adjacency)
    

    degrees = {i: len(adjacency[i]) for i in range(n_nodes)}
    min_degree = min(degrees.values())
    candidates = [i for i, d in degrees.items() if d == min_degree]
    start = candidates[0]
    
    while True:
        levels, max_level, nodes_at_max = bfs_levels(adjacency, start)
        

        deepest_nodes = [i for i, lev in levels.items() if lev == max_level]
        deepest_degrees = {i: degrees[i] for i in deepest_nodes}
        min_deg = min(deepest_degrees.values())
        next_candidates = [i for i, d in deepest_degrees.items() if d == min_deg]
        next_start = next_candidates[0]
        

        _, new_max_level, new_nodes_at_max = bfs_levels(adjacency, next_start)
        

        if new_nodes_at_max <= nodes_at_max:
            return next_start
        
        start = next_start


def rcm_reorder(adjacency: Dict[int, Set[int]]) -> np.ndarray:
    n_nodes = len(adjacency)
    

    start = find_pseudo_peripheral(adjacency)
    

    levels, max_level, _ = bfs_levels(adjacency, start)
    

    level_groups = [[] for _ in range(max_level + 1)]
    degrees = {i: len(adjacency[i]) for i in range(n_nodes)}
    
    for node, lev in levels.items():
        level_groups[lev].append(node)
    

    for lev in range(max_level + 1):
        level_groups[lev].sort(key=lambda x: degrees[x])
    


    cm_order = []
    for lev in range(max_level + 1):
        cm_order.extend(level_groups[lev])
    

    rcm_order = cm_order[::-1]
    
    reorder = np.array(rcm_order, dtype=int)
    return reorder


def apply_reorder_to_mesh(nodes: np.ndarray, triangles: np.ndarray,
                          reorder: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_nodes = nodes.shape[0]
    old_to_new = np.zeros(n_nodes, dtype=int)
    for new_idx, old_idx in enumerate(reorder):
        old_to_new[old_idx] = new_idx
    
    new_nodes = nodes[reorder]
    new_triangles = old_to_new[triangles]
    
    return new_nodes, new_triangles, old_to_new


def generate_acoustic_domain(nx: int = 41, ny: int = 41,
                             xlim: Tuple[float, float] = (0.0, 0.1),
                             ylim: Tuple[float, float] = (0.0, 0.1)) -> Tuple[np.ndarray, np.ndarray]:
    if nx < 2 or ny < 2:
        raise ValueError(f"nx和ny必须至少为2，当前nx={nx}, ny={ny}")
    
    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    

    nodes = []
    for j in range(ny):
        for i in range(nx):
            nodes.append([x[i], y[j]])
    nodes = np.array(nodes)
    

    triangles = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n1 = j * nx + i
            n2 = j * nx + (i + 1)
            n3 = (j + 1) * nx + i
            n4 = (j + 1) * nx + (i + 1)
            

            triangles.append([n1, n2, n4])
            triangles.append([n1, n4, n3])
    
    triangles = np.array(triangles)
    
    return nodes, triangles


def generate_optimized_acoustic_mesh(nx: int = 41, ny: int = 41,
                                     xlim: Tuple[float, float] = (0.0, 0.1),
                                     ylim: Tuple[float, float] = (0.0, 0.1)) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nodes, triangles = generate_acoustic_domain(nx, ny, xlim, ylim)
    adjacency = build_adjacency_structure(nodes, triangles)
    reorder = rcm_reorder(adjacency)
    new_nodes, new_triangles, old_to_new = apply_reorder_to_mesh(nodes, triangles, reorder)
    
    return new_nodes, new_triangles, reorder, old_to_new
