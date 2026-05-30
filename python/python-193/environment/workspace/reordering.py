
import numpy as np


def random_permutation(n, seed=42):
    rng = np.random.default_rng(seed)
    return rng.permutation(n)


def permutation_to_matrix(perm):
    n = len(perm)
    P = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = perm[i]
        if 0 <= j < n:
            P[i, j] = 1.0
    return P


def cycle_decomposition(perm):
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
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if n == 0:
        return np.array([], dtype=int)


    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and abs(A[i, j]) > 1e-15:
                adj[i].add(j)
                adj[j].add(i)

    def degree(i):
        return len(adj[i])


    start = min(range(n), key=degree)


    visited = [False] * n
    order = []

    while len(order) < n:

        unvisited = [i for i in range(n) if not visited[i]]
        if not unvisited:
            break
        start = min(unvisited, key=degree)
        queue = [start]
        visited[start] = True

        while queue:

            queue.sort(key=degree)
            next_queue = []
            for node in queue:
                order.append(node)
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        next_queue.append(neighbor)
            queue = next_queue


    rcm_order = order[::-1]
    return np.array(rcm_order, dtype=int)


def apply_reordering(A, order):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    order = np.asarray(order, dtype=int)

    if len(order) != n or set(order) != set(range(n)):
        raise ValueError("Invalid permutation order")
    A_perm = np.zeros_like(A)
    for i in range(n):
        for j in range(n):
            A_perm[i, j] = A[order[i], order[j]]
    return A_perm


def bandwidth(A):
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
