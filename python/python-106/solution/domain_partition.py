"""
domain_partition.py
===================
Greedy domain partitioning for load-balanced parallel plasmonic computation.

In large-scale simulations of disordered nanoparticle assemblies, the
computational load is unevenly distributed: dense clusters require more
memory and CPU time than sparse regions.  We partition the set of
nanoparticles into P subsets (domains) such that the total workload
is as evenly distributed as possible.

Given workload weights {w_i} for N particles, the greedy partition
algorithm (from partition_greedy seed) sorts weights in descending
order and assigns each particle to the currently lighter subset:

    S₀_sum = 0,  S₁_sum = 0
    for w in sorted_weights:
        if S₀_sum < S₁_sum:
            assign to subset 0; S₀_sum += w
        else:
            assign to subset 1; S₁_sum += w

For P > 2 subsets, we maintain P running sums and always assign to
the minimum-sum subset.

The discrepancy is defined as:

    D = | max_p S_p − min_p S_p |

An ideal partition has D = 0.

Load estimation for plasmonics:
    w_i = n_neighbors_i²  (dipole-dipole interaction scales as N² per domain)
"""

import numpy as np


def greedy_partition(weights, num_partitions):
    """
    Greedy balanced partitioning of weighted items.

    Parameters
    ----------
    weights : ndarray, shape (N,)
    num_partitions : int

    Returns
    -------
    assignment : ndarray, shape (N,)
        Subset index (0..num_partitions-1) for each item.
    subset_sums : ndarray, shape (num_partitions,)
    discrepancy : float
    """
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
    """
    Estimate per-particle workload based on number of neighbors within
    an interaction cutoff.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
    interaction_radius : float

    Returns
    -------
    weights : ndarray, shape (N,)
    """
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
        # Workload scales roughly as count² (pairwise interactions)
        weights[i] = max(1, count ** 2)
    return weights


def spectral_partition_laplacian(adjacency_matrix, num_partitions):
    """
    Spectral graph partitioning using the Fiedler vector of the graph
    Laplacian.  Particles are embedded in a 1D spectral coordinate and
then split greedily.

    Parameters
    ----------
    adjacency_matrix : ndarray, shape (N, N)
    num_partitions : int

    Returns
    -------
    assignment : ndarray
    """
    N = adjacency_matrix.shape[0]
    degree = np.sum(adjacency_matrix, axis=1)
    L = np.diag(degree) - adjacency_matrix

    eigvals, eigvecs = np.linalg.eigh(L)
    # Fiedler vector is the eigenvector for the second-smallest eigenvalue
    fiedler = eigvecs[:, 1]

    # Sort by Fiedler value and assign greedily to balance count
    indices = np.argsort(fiedler)
    assignment = np.zeros(N, dtype=int)
    counts = np.zeros(num_partitions)

    for idx in indices:
        p = np.argmin(counts)
        assignment[idx] = p
        counts[p] += 1

    return assignment
