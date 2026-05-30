
import numpy as np
from typing import Tuple, List, Dict


def linear_to_quadratic_triangles(
    nodes: np.ndarray,
    elements: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]

    if elements.shape[1] != 3:
        raise ValueError("Elements must have 3 vertices for linear triangles.")


    edge_to_midpoint: Dict[Tuple[int, int], int] = {}
    new_nodes_list = list(nodes)

    def get_sorted_edge(v1: int, v2: int) -> Tuple[int, int]:
        return (min(v1, v2), max(v1, v2))


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
    n = len(adjacency)
    if n == 0:
        return 0

    current = start % n
    prev_depth = -1

    while True:

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

        last_level = level_nodes[max_level]
        degrees = [len(adjacency[v]) for v in last_level]
        current = last_level[int(np.argmin(degrees))]


def rcm_order(adjacency: List[List[int]]) -> np.ndarray:
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)

    root = pseudo_peripheral_node(adjacency, start=0)

    visited = np.full(n, False, dtype=bool)
    perm = []
    queue = [root]
    visited[root] = True

    while queue:

        level = []
        next_queue = []
        for u in queue:
            level.append(u)
            for v in adjacency[u]:
                if not visited[v]:
                    visited[v] = True
                    next_queue.append(v)

        level.sort(key=lambda x: len(adjacency[x]))
        perm.extend(level)
        queue = next_queue

    perm = np.array(perm, dtype=int)

    return perm[::-1]


def apply_rcm_to_sparse_matrix(
    H: np.ndarray,
    adjacency: List[List[int]],
) -> Tuple[np.ndarray, np.ndarray, int, int]:
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
