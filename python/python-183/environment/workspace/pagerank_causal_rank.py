
import numpy as np
from typing import List, Tuple, Optional


def adjacency_from_edges(edges: List[Tuple[int, int, float]], n: int,
                         use_weights: bool = True) -> np.ndarray:
    A = np.zeros((n, n), dtype=float)
    for i, j, w in edges:
        if 0 <= i < n and 0 <= j < n:
            A[j, i] = abs(w) if use_weights else 1.0
    return A


def build_google_matrix(A: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha 必须在 (0,1) 区间内。")
    n = A.shape[0]

    col_sums = A.sum(axis=0)
    S = np.zeros_like(A)
    for j in range(n):
        if col_sums[j] > 0.0:
            S[:, j] = A[:, j] / col_sums[j]
        else:

            S[:, j] = 1.0 / n


    G = alpha * S + (1.0 - alpha) / n * np.ones((n, n))
    return G


def power_method_rank(G: np.ndarray,
                      max_iter: int = 200,
                      tol: float = 1e-10) -> np.ndarray:
    n = G.shape[0]
    pi = np.ones(n) / n
    for it in range(max_iter):
        pi_new = G @ pi

        s = np.sum(pi_new)
        if s > 0.0:
            pi_new = pi_new / s
        diff = np.linalg.norm(pi_new - pi, 1)
        pi = pi_new
        if diff < tol:
            break
    return pi


def surf_rank(pi_history: np.ndarray) -> np.ndarray:
    if pi_history.ndim != 2:
        raise ValueError("pi_history 必须是二维数组 (iterations, n_nodes)。")
    return np.sum(np.abs(np.diff(pi_history, axis=0)), axis=0)


def identify_confounders_by_rank(edges: List[Tuple[int, int, float]],
                                  n: int,
                                  top_k: int = 3) -> List[Tuple[int, float]]:
    A = adjacency_from_edges(edges, n, use_weights=False)
    G = build_google_matrix(A, alpha=0.85)
    pi = power_method_rank(G)

    indeg = A.sum(axis=0)
    outdeg = A.sum(axis=1)

    scores = []
    for i in range(n):
        ratio = (outdeg[i] + 1.0) / (indeg[i] + 1.0)
        score = pi[i] * np.log1p(ratio)
        scores.append((i, float(score)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def demo():
    np.random.seed(3)
    n = 10

    edges = [(0, 1, 0.5), (0, 2, 0.4), (1, 3, 0.3), (2, 3, 0.3),
             (3, 4, 0.6), (4, 5, 0.5), (0, 5, 0.2), (6, 7, 0.4),
             (7, 8, 0.5), (8, 9, 0.3), (6, 9, 0.2)]
    A = adjacency_from_edges(edges, n)
    G = build_google_matrix(A, alpha=0.85)
    pi = power_method_rank(G, max_iter=300)
    confounders = identify_confounders_by_rank(edges, n, top_k=3)
    print(f"[pagerank_causal_rank] CausalRank (前5): {pi[:5].round(4)}")
    print(f"[pagerank_causal_rank] 潜在混淆变量: {confounders}")
    return pi, confounders


if __name__ == "__main__":
    demo()
