
import numpy as np
from typing import List, Tuple, Dict


def binary_to_gray(n: int, m: int) -> np.ndarray:
    gray = n ^ (n >> 1)
    bits = np.zeros(m, dtype=int)
    for i in range(m):
        bits[i] = (gray >> i) & 1
    return bits


def gray_to_binary(g: np.ndarray) -> int:
    n = 0
    mask = 0
    for i in range(len(g) - 1, -1, -1):
        mask ^= g[i]
        n = (n << 1) | mask
    return n


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(np.abs(a - b)))


def quantize_to_gray_code(x: np.ndarray, bounds: np.ndarray, m_bits: int = 8) -> np.ndarray:
    D = len(x)
    indices = np.zeros(D, dtype=int)
    for d in range(D):
        xmin, xmax = bounds[d]
        if xmax <= xmin:
            indices[d] = 0
            continue

        norm = (x[d] - xmin) / (xmax - xmin)
        norm = np.clip(norm, 0.0, 1.0)
        idx = int(norm * ((1 << m_bits) - 1))
        indices[d] = idx
    return indices


def gray_code_neighborhood_search(data: np.ndarray, query: np.ndarray,
                                   bounds: np.ndarray, m_bits: int = 8,
                                   max_hamming: int = 2) -> np.ndarray:
    N, D = data.shape
    query_gray = quantize_to_gray_code(query, bounds, m_bits)
    candidates = []
    for i in range(N):
        pt_gray = quantize_to_gray_code(data[i], bounds, m_bits)
        hd = hamming_distance(query_gray, pt_gray)
        if hd <= max_hamming:
            candidates.append(i)
    if len(candidates) == 0:

        dists = np.linalg.norm(data - query, axis=1)
        candidates = [int(np.argmin(dists))]
    return np.array(candidates, dtype=int)


def levenshtein_distance(s: List, t: List) -> int:
    m, n = len(s), len(t)
    d = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1):
        d[i, 0] = i
    for j in range(n + 1):
        d[0, j] = j
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1,
                          min(d[i, j - 1] + 1,
                              d[i - 1, j - 1] + cost))
    return int(d[m, n])


def mixed_distance(x: np.ndarray, y: np.ndarray,
                   categorical_dims: List[int] = None,
                   sequence_dims: List[int] = None) -> float:
    D = len(x)
    if categorical_dims is None:
        categorical_dims = []
    if sequence_dims is None:
        sequence_dims = []
    numerical_dims = [d for d in range(D)
                      if d not in categorical_dims and d not in sequence_dims]
    dist = 0.0

    if len(numerical_dims) > 0:
        diff = x[numerical_dims] - y[numerical_dims]
        dist += np.sum(diff ** 2)

    for d in categorical_dims:
        if x[d] != y[d]:
            dist += 1.0

    for d in sequence_dims:
        s = list(str(x[d]))
        t = list(str(y[d]))
        max_len = max(len(s), len(t))
        if max_len > 0:
            dist += levenshtein_distance(s, t) / max_len
    return np.sqrt(dist)


def build_knn_graph(data: np.ndarray, k: int = 10,
                    method: str = "exact") -> Tuple[np.ndarray, np.ndarray]:
    N = len(data)
    edges = []
    weights = []

    all_dists = []
    for i in range(min(N, 100)):
        dists = np.linalg.norm(data - data[i], axis=1)
        all_dists.extend(dists[dists > 0])
    median_dist = np.median(all_dists) if len(all_dists) > 0 else 1.0
    bandwidth = median_dist
    for i in range(N):
        dists = np.linalg.norm(data - data[i], axis=1)
        idx = np.argsort(dists)[1:k + 1]
        for j in idx:
            w = np.exp(-dists[j] ** 2 / (2.0 * bandwidth ** 2))
            edges.append([i, j])
            weights.append(w)
    return np.array(edges, dtype=int), np.array(weights, dtype=np.float64)


def graph_laplacian(edges: np.ndarray, weights: np.ndarray,
                    n_vertices: int, normalize: bool = True) -> np.ndarray:
    W = np.zeros((n_vertices, n_vertices), dtype=np.float64)
    for (i, j), w in zip(edges, weights):
        W[i, j] = w
        W[j, i] = w
    D = np.diag(np.sum(W, axis=1))
    if normalize:
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-15))
        L = np.eye(n_vertices) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        L = D - W
    return L
