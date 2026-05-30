
import numpy as np


def greedy_partition(weights, num_partitions):
    n = weights.size
    if n < 1:
        raise ValueError("weights must be non-empty.")
    if num_partitions < 1:
        raise ValueError("num_partitions must be positive.")

    sorted_indices = np.argsort(-weights)
    assignment = np.zeros(n, dtype=int)
    subset_sums = np.zeros(num_partitions)

    for idx in sorted_indices:
        p = np.argmin(subset_sums)
        assignment[idx] = p
        subset_sums[p] += weights[idx]

    discrepancy = np.max(subset_sums) - np.min(subset_sums)
    return assignment, subset_sums, discrepancy


def estimate_workload(positions, interaction_radius):
    N = positions.shape[0]
    weights = np.zeros(N)
    for i in range(N):
        count = 0
        for j in range(N):
            if i == j:
                continue
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < interaction_radius:
                count += 1

        weights[i] = max(1, count ** 2)
    return weights


def spectral_partition_laplacian(adjacency_matrix, num_partitions):
    N = adjacency_matrix.shape[0]
    degree = np.sum(adjacency_matrix, axis=1)
    L = np.diag(degree) - adjacency_matrix

    eigvals, eigvecs = np.linalg.eigh(L)

    fiedler = eigvecs[:, 1]


    indices = np.argsort(fiedler)
    assignment = np.zeros(N, dtype=int)
    counts = np.zeros(num_partitions)

    for idx in indices:
        p = np.argmin(counts)
        assignment[idx] = p
        counts[p] += 1

    return assignment
