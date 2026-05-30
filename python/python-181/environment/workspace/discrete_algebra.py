
import numpy as np
from typing import Tuple


def threshold_encode(data: np.ndarray, thresholds: np.ndarray = None) -> np.ndarray:
    if thresholds is None:
        thresholds = np.median(data, axis=0)
    binary = (data > thresholds).astype(int)
    return binary


def gray_code_hypercube_adjacency(n_dim: int) -> np.ndarray:
    n_vertices = 1 << n_dim
    A = np.zeros((n_vertices, n_vertices), dtype=int)
    for i in range(n_vertices):
        gray_i = i ^ (i >> 1)
        for d in range(n_dim):
            j = i ^ (1 << d)
            gray_j = j ^ (j >> 1)
            if bin(gray_i ^ gray_j).count('1') == 1:
                A[i, j] = 1
                A[j, i] = 1
    return A


def boolean_pca(data_binary: np.ndarray, n_components: int = 3) -> np.ndarray:

    mean = np.mean(data_binary, axis=0)
    centered = data_binary - mean
    cov = centered.T @ centered / len(data_binary)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    components = eigvecs[:, idx[:n_components]]
    embedding = centered @ components
    return embedding


def mod2_rank(matrix: np.ndarray) -> int:
    A = matrix.copy() % 2
    m, n = A.shape
    rank = 0
    row = 0
    for col in range(n):
        if row >= m:
            break

        pivot = -1
        for r in range(row, m):
            if A[r, col] == 1:
                pivot = r
                break
        if pivot == -1:
            continue

        A[[row, pivot]] = A[[pivot, row]]

        for r in range(m):
            if r != row and A[r, col] == 1:
                A[r] = (A[r] + A[row]) % 2
        row += 1
        rank += 1
    return rank


def binary_feature_hash(data_binary: np.ndarray, n_bits: int = 16) -> np.ndarray:
    n, d = data_binary.shape
    np.random.seed(42)
    R = np.random.randn(n_bits, d)
    projected = R @ data_binary.T
    hash_codes = (projected > 0).astype(int).T
    return hash_codes


def hamming_distance_matrix(binary_data: np.ndarray) -> np.ndarray:
    n = len(binary_data)
    D = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = int(np.sum(np.abs(binary_data[i] - binary_data[j])))
            D[i, j] = d
            D[j, i] = d
    return D


def discrete_wasserstein_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = p / (np.sum(p) + 1e-15)
    q = q / (np.sum(q) + 1e-15)
    cum_p = np.cumsum(p)
    cum_q = np.cumsum(q)
    return float(np.sum(np.abs(cum_p - cum_q)))


def lights_out_feature_transform(data: np.ndarray, grid_size: int = 5) -> np.ndarray:
    n = len(data)
    from topological_invariants import lights_out_matrix
    A = lights_out_matrix(grid_size, grid_size)

    features = []
    for pt in data:

        vec = np.zeros(grid_size * grid_size)
        d = min(len(pt), grid_size * grid_size)
        vec[:d] = pt[:d]

        median = np.median(vec)
        binary = (vec > median).astype(int)

        p = np.linalg.lstsq(A.astype(float), binary.astype(float), rcond=None)[0]
        p = (p > 0.5).astype(int)
        features.append(p)
    return np.array(features)
