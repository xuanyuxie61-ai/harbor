"""
Sparse Matrix Reordering and Permutation Analysis Module.

Integrates:
  - 696_locker_simulation: permutation cycle analysis

Scientific formulas:
  Permutation matrix P:
    P_{i,j} = 1 if sigma(i) = j, else 0
    P^T * A * P reorders rows and columns of A.

  Cuthill-McKee (CM) bandwidth reduction:
    Based on BFS starting from a peripheral node.
    Produces ordering that minimizes matrix bandwidth:
      bandwidth(A) = max{ |i-j| : A_{i,j} != 0 }

  AMD (Approximate Minimum Degree) heuristic:
    At each step, eliminate the node with minimum degree.
    Fill-in F at elimination step k:
      F_k = sum_{i in adj(k)} sum_{j in adj(k), j>i} e_{ij}
    where e_{ij} is the edge indicator.

  Cycle decomposition of permutation:
    Any permutation decomposes into disjoint cycles.
    Cycle length distribution affects sparse LU factorization fill.
"""

import numpy as np


def random_permutation(n, seed=42):
    """
    Generate a random permutation of [0, n-1].
    Directly from seed 696_locker_simulation concept.
    """
    rng = np.random.default_rng(seed)
    return rng.permutation(n)


def permutation_to_matrix(perm):
    """
    Convert a permutation vector to a permutation matrix P.
    P[i, perm[i]] = 1, satisfying P^T * P = I.
    """
    n = len(perm)
    P = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = perm[i]
        if 0 <= j < n:
            P[i, j] = 1.0
    return P


def cycle_decomposition(perm):
    """
    Decompose a permutation into disjoint cycles.
    Returns list of cycles, each cycle is a list of indices.
    From seed 696_locker_simulation.

    Example: perm = [1, 2, 0, 4, 3]
    Cycles: [[0, 1, 2], [3, 4]]
    """
    n = len(perm)
    visited = [False] * n
    cycles = []
    for i in range(n):
        if not visited[i]:
            cycle = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = perm[j]
                if j < 0 or j >= n:
                    break
            if len(cycle) > 0:
                cycles.append(cycle)
    return cycles


def analyze_permutation_cycles(n=100, n_trials=1000, seed=42):
    """
    Monte-Carlo analysis of cycle length distributions in random permutations.
    Theoretical expectation for cycle length L in random permutation of size n:
      E[#cycles of length L] = 1/L
      E[total #cycles] = H_n = sum_{k=1}^n 1/k  (harmonic number)

    Returns:
      dict with empirical cycle statistics.
    """
    rng = np.random.default_rng(seed)
    total_cycles = []
    max_cycle_lengths = []
    cycle_length_hist = np.zeros(n + 1)

    for _ in range(n_trials):
        perm = rng.permutation(n)
        cycles = cycle_decomposition(perm)
        total_cycles.append(len(cycles))
        max_len = max((len(c) for c in cycles), default=0)
        max_cycle_lengths.append(max_len)
        for c in cycles:
            if len(c) <= n:
                cycle_length_hist[len(c)] += 1

    harmonic_n = sum(1.0 / k for k in range(1, n + 1))
    return {
        'n': n,
        'n_trials': n_trials,
        'expected_total_cycles': harmonic_n,
        'mean_total_cycles': np.mean(total_cycles),
        'mean_max_cycle_length': np.mean(max_cycle_lengths),
        'cycle_length_histogram': cycle_length_hist / n_trials,
    }


def reverse_cuthill_mckee(A):
    """
    Reverse Cuthill-McKee (RCM) ordering for bandwidth reduction.
    Simplified implementation for small dense matrices.

    Algorithm:
      1. Find a peripheral node (node with minimum degree on boundary)
      2. BFS from this node to generate level sets
      3. Sort nodes within each level by degree
      4. Reverse the ordering

    Scientific impact:
      For sparse matrix A with bandwidth B, Cholesky factorization
      fill-in is bounded by O(B^2 * N).
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    # Build adjacency list (ignore diagonal)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and abs(A[i, j]) > 1e-15:
                adj[i].add(j)
                adj[j].add(i)

    def degree(i):
        return len(adj[i])

    # Find starting node: node with minimum degree
    start = min(range(n), key=degree)

    # BFS to get level structure (handle disconnected components)
    visited = [False] * n
    order = []

    while len(order) < n:
        # Find unvisited node with minimum degree
        unvisited = [i for i in range(n) if not visited[i]]
        if not unvisited:
            break
        start = min(unvisited, key=degree)
        queue = [start]
        visited[start] = True

        while queue:
            # Sort current level by degree
            queue.sort(key=degree)
            next_queue = []
            for node in queue:
                order.append(node)
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        next_queue.append(neighbor)
            queue = next_queue

    # Reverse ordering
    rcm_order = order[::-1]
    return np.array(rcm_order, dtype=int)


def apply_reordering(A, order):
    """
    Apply permutation ordering to matrix A:
      A_perm = P^T * A * P
    where P is the permutation matrix corresponding to 'order'.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    order = np.asarray(order, dtype=int)
    # Validate
    if len(order) != n or set(order) != set(range(n)):
        raise ValueError("Invalid permutation order")
    A_perm = np.zeros_like(A)
    for i in range(n):
        for j in range(n):
            A_perm[i, j] = A[order[i], order[j]]
    return A_perm


def bandwidth(A):
    """
    Compute matrix bandwidth:
      lower_bw = max{i-j : A_{i,j} != 0, i > j}
      upper_bw = max{j-i : A_{i,j} != 0, j > i}
      bandwidth = lower_bw + upper_bw + 1
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    lower_bw = 0
    upper_bw = 0
    for i in range(n):
        for j in range(n):
            if abs(A[i, j]) > 1e-15:
                if i > j:
                    lower_bw = max(lower_bw, i - j)
                elif j > i:
                    upper_bw = max(upper_bw, j - i)
    return lower_bw + upper_bw + 1
