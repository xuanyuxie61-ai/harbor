
import numpy as np
from typing import List, Tuple






def greedy_graph_partition(weights: np.ndarray, adjacency: np.ndarray) -> Tuple[np.ndarray, float, float]:
    n = len(weights)
    partition = np.zeros(n, dtype=np.int32)

    degrees = adjacency.sum(axis=1) * weights
    order = np.argsort(-degrees)
    sum0, sum1 = 0.0, 0.0
    for idx in order:
        if sum0 <= sum1:
            partition[idx] = 0
            sum0 += weights[idx]
        else:
            partition[idx] = 1
            sum1 += weights[idx]
    return partition, sum0, sum1






def fermat_factor(n: int) -> Tuple[int, int]:
    if n < 2:
        return (1, n)
    a = int(np.floor(np.sqrt(n)))
    if a * a == n:
        return (a, a)
    while True:
        a += 1
        b2 = a * a - n
        if b2 < 0:
            continue
        b = int(np.sqrt(b2))
        if b * b == b2:
            return (a - b, a + b)
        if a > n:
            return (1, n)


def graph_hash_fingerprint(n_nodes: int, n_edges: int) -> int:
    val = n_nodes * n_edges + 1
    f1, f2 = fermat_factor(val)
    return abs(f1 - n_nodes)






def diophantine_nonnegative_solutions(target: int, n_vars: int) -> List[np.ndarray]:
    solutions = []
    def backtrack(remain, start, current):
        if start == n_vars - 1:
            current.append(remain)
            solutions.append(np.array(current, dtype=np.int32))
            current.pop()
            return
        for v in range(remain + 1):
            current.append(v)
            backtrack(remain - v, start + 1, current)
            current.pop()
    backtrack(target, 0, [])
    return solutions


def parity_violation_check(atom_counts: np.ndarray, required_parity: int = 0) -> bool:
    total = np.sum(atom_counts)
    return (total % 2) != required_parity






def hypercube_distance_stats(descriptors: np.ndarray, n_pairs: int = 500) -> Tuple[float, float]:
    n = descriptors.shape[0]
    if n < 2:
        return 0.0, 0.0
    distances = []
    for _ in range(n_pairs):
        i, j = np.random.randint(0, n, 2)
        if i == j:
            continue
        d = np.linalg.norm(descriptors[i] - descriptors[j])
        distances.append(d)
    if not distances:
        return 0.0, 0.0
    dists = np.array(distances, dtype=np.float64)
    return float(dists.mean()), float(dists.var())


def descriptor_space_uniformity(descriptors: np.ndarray) -> float:
    mu, var = hypercube_distance_stats(descriptors, n_pairs=min(1000, len(descriptors) * 10))
    if mu < 1e-12:
        return 0.0
    return 1.0 / (1.0 + var / (mu ** 2))






def spherical_basis_angles(n_points: int = 16, rotation: float = 0.0) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False) + np.radians(rotation)
    return np.column_stack([np.cos(theta), np.sin(theta)])


def angular_descriptor(atoms: np.ndarray, center_idx: int, n_angles: int = 8) -> np.ndarray:
    n = atoms.shape[0]
    if center_idx >= n:
        return np.zeros(n_angles)
    dirs = spherical_basis_angles(n_angles)
    center = atoms[center_idx]
    desc = np.zeros(n_angles, dtype=np.float64)
    for i in range(n):
        if i == center_idx:
            continue
        dr = atoms[i] - center
        r = np.linalg.norm(dr)
        if r < 1e-6:
            continue
        dr_u = dr / r

        for a in range(n_angles):
            proj = np.dot(dr_u[:2], dirs[a])
            desc[a] += proj / r

    norm = np.linalg.norm(desc)
    return desc / norm if norm > 1e-12 else desc
