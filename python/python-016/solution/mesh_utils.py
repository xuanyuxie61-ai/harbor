"""
Mesh Processing, Triangulation Refinement, and Reverse Cuthill-McKee
====================================================================
Provides utilities for handling 2D triangular and quadrilateral meshes
that arise in the real-space discretization of moiré superlattices,
including linear-to-quadratic element upgrade and bandwidth reduction
for sparse Hamiltonian matrices.

Scientific Background
---------------------
Finite-element or finite-difference calculations on irregular domains
require a mesh.  For the moiré supercell, a triangular mesh allows
flexible handling of the hexagonal geometry.  Quadratic (P2) elements
provide higher-order accuracy:

    Linear (P1) triangle: 3 nodes (vertices)
    Quadratic (P2) triangle: 6 nodes (vertices + edge midpoints)

The element upgrade adds edge-midpoint nodes while preserving
compatibility (shared edges get the same midpoint node).

For sparse matrix computations, the bandwidth

    B = max_{A_{ij} ≠ 0} |i − j|

strongly affects the performance of direct solvers and preconditioners.
The Reverse Cuthill-McKee (RCM) algorithm reorders nodes to minimize B
by performing a breadth-first search from a pseudo-peripheral node and
reversing the ordering.

The profile (envelope) is defined as

    P = Σ_i (max_{j≤i, A_{ij}≠0} j) .

RCM typically reduces both bandwidth and profile by one to two orders
of magnitude for graphs with local connectivity (e.g., tight-binding
Hamiltonians).
"""

import numpy as np
from typing import Tuple, List, Dict


def linear_to_quadratic_triangles(
    nodes: np.ndarray,
    elements: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Upgrade a linear (3-node) triangular mesh to a quadratic (6-node)
    triangular mesh by adding edge-midpoint nodes.

    For each edge shared by two triangles, the midpoint is computed once
    and assigned to both elements.  Boundary edges get their own
    midpoint nodes.

    Parameters
    ----------
    nodes : np.ndarray of shape (n_nodes, 2)
        Vertex coordinates.
    elements : np.ndarray of shape (n_elements, 3)
        Linear element connectivity (0-based indices).

    Returns
    -------
    new_nodes : np.ndarray of shape (n_nodes + n_edge_nodes, 2)
    new_elements : np.ndarray of shape (n_elements, 6)
        Quadratic connectivity: [v0, v1, v2, m01, m12, m20].
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]

    if elements.shape[1] != 3:
        raise ValueError("Elements must have 3 vertices for linear triangles.")

    # Identify edges and shared edges
    edge_to_midpoint: Dict[Tuple[int, int], int] = {}
    new_nodes_list = list(nodes)

    def get_sorted_edge(v1: int, v2: int) -> Tuple[int, int]:
        return (min(v1, v2), max(v1, v2))

    # First pass: create midpoints for all edges
    for e in range(n_elements):
        verts = elements[e]
        for edge_idx in range(3):
            v1 = verts[edge_idx]
            v2 = verts[(edge_idx + 1) % 3]
            edge = get_sorted_edge(v1, v2)
            if edge not in edge_to_midpoint:
                mid = 0.5 * (nodes[v1] + nodes[v2])
                new_idx = n_nodes + len(edge_to_midpoint)
                edge_to_midpoint[edge] = new_idx
                new_nodes_list.append(mid)

    n_new_nodes = len(new_nodes_list)
    new_elements = np.zeros((n_elements, 6), dtype=int)

    for e in range(n_elements):
        verts = elements[e]
        new_elements[e, 0:3] = verts
        # Edge midpoints
        for edge_idx in range(3):
            v1 = verts[edge_idx]
            v2 = verts[(edge_idx + 1) % 3]
            edge = get_sorted_edge(v1, v2)
            new_elements[e, 3 + edge_idx] = edge_to_midpoint[edge]

    new_nodes = np.array(new_nodes_list)
    return new_nodes, new_elements


def build_adjacency_from_elements(
    elements: np.ndarray,
    n_nodes: int,
    element_type: str = "triangle",
) -> List[List[int]]:
    """
    Build a node adjacency graph from element connectivity.

    Two nodes are adjacent if they share at least one element.

    Parameters
    ----------
    elements : np.ndarray
        Element connectivity array.
    n_nodes : int
        Total number of nodes.
    element_type : str
        "triangle" (3 or 6 nodes) or "quadrilateral" (4 nodes).

    Returns
    -------
    adjacency : list of lists
        adjacency[i] contains the neighbors of node i.
    """
    adjacency = [set() for _ in range(n_nodes)]

    for elem in elements:
        if element_type == "triangle":
            n_vert = min(3, len(elem))
        elif element_type == "quadrilateral":
            n_vert = min(4, len(elem))
        else:
            raise ValueError("Unsupported element type.")

        verts = elem[:n_vert]
        for i in range(n_vert):
            for j in range(i + 1, n_vert):
                vi = int(verts[i])
                vj = int(verts[j])
                if 0 <= vi < n_nodes and 0 <= vj < n_nodes:
                    adjacency[vi].add(vj)
                    adjacency[vj].add(vi)

    return [sorted(list(s)) for s in adjacency]


def pseudo_peripheral_node(
    adjacency: List[List[int]],
    start: int = 0,
) -> int:
    """
    Find a pseudo-peripheral node using the Gibbs-Poole-Stockmeyer
    strategy: start from an arbitrary node, find the farthest node in
    a BFS level set, and repeat until the level-set depth stops growing.

    Parameters
    ----------
    adjacency : list of lists
    start : int
        Starting node index.

    Returns
    -------
    int
        A pseudo-peripheral node.
    """
    n = len(adjacency)
    if n == 0:
        return 0

    current = start % n
    prev_depth = -1

    while True:
        # BFS from current
        visited = np.full(n, -1, dtype=int)
        queue = [current]
        visited[current] = 0
        level_nodes = {0: [current]}
        head = 0

        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adjacency[u]:
                if visited[v] == -1:
                    visited[v] = visited[u] + 1
                    level = visited[v]
                    if level not in level_nodes:
                        level_nodes[level] = []
                    level_nodes[level].append(v)
                    queue.append(v)

        max_level = max(level_nodes.keys())
        if max_level <= prev_depth:
            return current
        prev_depth = max_level
        # Pick the node in the last level with minimum degree
        last_level = level_nodes[max_level]
        degrees = [len(adjacency[v]) for v in last_level]
        current = last_level[int(np.argmin(degrees))]


def rcm_order(adjacency: List[List[int]]) -> np.ndarray:
    """
    Compute the Reverse Cuthill-McKee (RCM) ordering for a sparse graph.

    Algorithm:
      1. Find a pseudo-peripheral node R.
      2. Initialize ordering with R.
      3. For each level set (breadth-first layers), sort nodes within
         the level by increasing degree.
      4. Reverse the final ordering.

    Parameters
    ----------
    adjacency : list of lists

    Returns
    -------
    np.ndarray of shape (n,)
        RCM permutation (new_index = perm[old_index]).
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)

    root = pseudo_peripheral_node(adjacency, start=0)

    visited = np.full(n, False, dtype=bool)
    perm = []
    queue = [root]
    visited[root] = True

    while queue:
        # Collect current level
        level = []
        next_queue = []
        for u in queue:
            level.append(u)
            for v in adjacency[u]:
                if not visited[v]:
                    visited[v] = True
                    next_queue.append(v)
        # Sort level by degree (ascending)
        level.sort(key=lambda x: len(adjacency[x]))
        perm.extend(level)
        queue = next_queue

    perm = np.array(perm, dtype=int)
    # Reverse
    return perm[::-1]


def apply_rcm_to_sparse_matrix(
    H: np.ndarray,
    adjacency: List[List[int]],
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Apply RCM reordering to a sparse Hamiltonian matrix and return the
    reordered matrix along with bandwidth metrics.

    Parameters
    ----------
    H : np.ndarray of shape (n, n)
    adjacency : list of lists

    Returns
    -------
    H_perm : np.ndarray
        Reordered matrix.
    perm : np.ndarray
        Permutation array.
    bandwidth_old : int
    bandwidth_new : int
    """
    n = H.shape[0]
    perm = rcm_order(adjacency)
    H_perm = H[np.ix_(perm, perm)]

    def compute_bandwidth(M):
        bw = 0
        for i in range(M.shape[0]):
            row = M[i]
            nonzero = np.where(np.abs(row) > 1e-14)[0]
            if len(nonzero) > 0:
                bw = max(bw, max(nonzero) - i)
        return bw

    bw_old = compute_bandwidth(H)
    bw_new = compute_bandwidth(H_perm)
    return H_perm, perm, bw_old, bw_new


def write_triangle_mesh(
    filename: str,
    nodes: np.ndarray,
    elements: np.ndarray,
) -> None:
    """
    Write a triangular mesh in a simplified TRIANGLE format (.node and
    .ele files).

    Parameters
    ----------
    filename : str
        Base filename (without extension).
    nodes : np.ndarray of shape (n_nodes, 2)
    elements : np.ndarray of shape (n_elements, n_nodes_per_element)
    """
    node_file = filename + ".node"
    ele_file = filename + ".ele"

    n_nodes = nodes.shape[0]
    with open(node_file, "w") as f:
        f.write(f"{n_nodes} 2 0 0\n")
        for i in range(n_nodes):
            f.write(f"{i} {nodes[i, 0]:.12e} {nodes[i, 1]:.12e}\n")

    n_elements = elements.shape[0]
    n_per_elem = elements.shape[1]
    with open(ele_file, "w") as f:
        f.write(f"{n_elements} {n_per_elem} 0\n")
        for e in range(n_elements):
            idx_str = " ".join(str(int(elements[e, j])) for j in range(n_per_elem))
            f.write(f"{e} {idx_str}\n")


def read_triangle_mesh(
    node_file: str,
    ele_file: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a TRIANGLE-format mesh from .node and .ele files.

    Returns
    -------
    nodes : np.ndarray
    elements : np.ndarray
    """
    nodes = []
    with open(node_file, "r") as f:
        header = f.readline().strip().split()
        n_nodes = int(header[0])
        for _ in range(n_nodes):
            parts = f.readline().strip().split()
            nodes.append([float(parts[1]), float(parts[2])])

    elements = []
    with open(ele_file, "r") as f:
        header = f.readline().strip().split()
        n_elements = int(header[0])
        n_per = int(header[1])
        for _ in range(n_elements):
            parts = f.readline().strip().split()
            elems = [int(parts[j]) for j in range(1, 1 + n_per)]
            elements.append(elems)

    return np.array(nodes, dtype=float), np.array(elements, dtype=int)
