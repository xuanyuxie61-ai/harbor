
import numpy as np
from itertools import combinations
from typing import List, Tuple, Optional


def build_incidence_matrix(nodes: np.ndarray, members: List[Tuple[int, int]]) -> np.ndarray:
    n_nodes = nodes.shape[0]
    n_members = len(members)
    A = np.zeros((n_members, n_nodes), dtype=np.float64)
    for k, (i, j) in enumerate(members):
        if not (0 <= i < n_nodes and 0 <= j < n_nodes):
            raise ValueError(f"节点索引越界: ({i}, {j}), 总节点数={n_nodes}")
        if i == j:
            raise ValueError(f"自环杆件不允许: ({i}, {j})")
        A[k, i] = -1.0
        A[k, j] = +1.0
    return A


def build_equilibrium_matrix(nodes: np.ndarray, members: List[Tuple[int, int]]) -> np.ndarray:
    n_nodes = nodes.shape[0]
    n_members = len(members)
    B = np.zeros((3 * n_nodes, n_members), dtype=np.float64)
    for k, (i, j) in enumerate(members):
        diff = nodes[j] - nodes[i]
        length = np.linalg.norm(diff)
        if length < 1e-14:
            raise ValueError(f"杆件 {k} 长度为零: 节点 {i} 与 {j} 重合")
        c = diff / length
        B[3 * i:3 * i + 3, k] = -c
        B[3 * j:3 * j + 3, k] = +c
    return B


def check_static_determinacy(n_nodes: int, n_members: int, n_reactions: int,
                            spatial_dim: int = 3) -> Tuple[bool, int, int]:
    expected = spatial_dim * n_nodes
    actual = n_members + n_reactions
    deficiency = expected - actual
    is_determinate = (deficiency == 0)

    rank_deficiency = max(0, deficiency)
    return is_determinate, deficiency, rank_deficiency


def nullspace_orthogonality_check(B: np.ndarray, tol: float = 1e-10) -> Tuple[np.ndarray, int]:
    if B.size == 0:
        return np.zeros((0, 0)), 0
    _, s, Vt = np.linalg.svd(B, full_matrices=False)
    rank = np.sum(s > tol)
    dim_null = B.shape[1] - rank
    if dim_null <= 0:
        return np.zeros((B.shape[1], 0)), 0
    N = Vt[rank:, :].T
    return N, dim_null


def enumerate_dof_indices(n_nodes: int, spatial_dim: int = 3,
                          fixed_nodes: Optional[List[int]] = None) -> np.ndarray:
    total_dofs = spatial_dim * n_nodes
    all_dofs = np.arange(total_dofs, dtype=np.int32)
    if fixed_nodes is None or len(fixed_nodes) == 0:
        return all_dofs
    mask = np.ones(total_dofs, dtype=bool)
    for node in fixed_nodes:
        if not (0 <= node < n_nodes):
            raise ValueError(f"固定节点索引越界: {node}")
        mask[spatial_dim * node:spatial_dim * node + spatial_dim] = False
    return all_dofs[mask]


def lexicographic_joint_configs(n_joints: int, n_states: int) -> np.ndarray:
    total = n_states ** n_joints
    if total > 1_000_000:
        raise ValueError(f"构型空间过大: {total} > 1e6，请使用生成器版本")
    configs = np.zeros((total, n_joints), dtype=np.int32)
    for i in range(total):
        tmp = i
        for j in range(n_joints):
            configs[i, j] = tmp % n_states
            tmp //= n_states
    return configs


def triangulation_boundary_edges(triangles: np.ndarray) -> np.ndarray:
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles 必须是 (n, 3) 形状")
    n_tri = triangles.shape[0]
    edges = np.zeros((3 * n_tri, 2), dtype=np.int32)

    edges[0::3, 0] = triangles[:, 0]
    edges[0::3, 1] = triangles[:, 1]
    edges[1::3, 0] = triangles[:, 1]
    edges[1::3, 1] = triangles[:, 2]
    edges[2::3, 0] = triangles[:, 2]
    edges[2::3, 1] = triangles[:, 0]

    edges_sorted = np.sort(edges, axis=1)

    edge_dict = {}
    for e in edges_sorted:
        key = (int(e[0]), int(e[1]))
        edge_dict[key] = edge_dict.get(key, 0) + 1
    boundary = [np.array(k) for k, v in edge_dict.items() if v == 1]
    if len(boundary) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    boundary = np.vstack(boundary)

    chained = _chain_boundary_edges(boundary)
    return chained


def _chain_boundary_edges(edges: np.ndarray) -> np.ndarray:
    if edges.shape[0] == 0:
        return edges
    n_edges = edges.shape[0]
    used = np.zeros(n_edges, dtype=bool)
    result = []
    current = 0
    used[current] = True
    chain = [edges[current].copy()]
    while np.any(~used):
        tail = chain[-1][1]
        found = False
        for i in range(n_edges):
            if used[i]:
                continue
            if edges[i, 0] == tail:
                chain.append(edges[i].copy())
                used[i] = True
                found = True
                break
            elif edges[i, 1] == tail:
                chain.append(np.array([edges[i, 1], edges[i, 0]]))
                used[i] = True
                found = True
                break
        if not found:
            result.append(np.vstack(chain))

            remaining = np.where(~used)[0]
            if len(remaining) == 0:
                break
            current = remaining[0]
            used[current] = True
            chain = [edges[current].copy()]
    if len(chain) > 0:
        result.append(np.vstack(chain))
    if len(result) == 1:
        return result[0]
    return np.vstack(result)


def generate_truss_topology(num_bays: int = 2, bay_length: float = 1.0,
                            height: float = 1.0) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    if num_bays < 1:
        raise ValueError("num_bays 必须 ≥ 1")
    n_nodes = 2 * (num_bays + 1)
    nodes = np.zeros((n_nodes, 3), dtype=np.float64)
    for i in range(num_bays + 1):
        nodes[i, :] = [i * bay_length, 0.0, 0.0]
        nodes[i + num_bays + 1, :] = [i * bay_length, height, 0.0]
    members = []

    for i in range(num_bays):
        members.append((i, i + 1))

    for i in range(num_bays):
        members.append((i + num_bays + 1, i + num_bays + 2))

    for i in range(num_bays + 1):
        members.append((i, i + num_bays + 1))

    for i in range(num_bays):
        members.append((i, i + num_bays + 2))
        members.append((i + 1, i + num_bays + 1))
    return nodes, members
