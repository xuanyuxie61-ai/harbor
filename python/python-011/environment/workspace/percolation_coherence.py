# -*- coding: utf-8 -*-

import numpy as np


def components_2d(A):
    A = np.asarray(A, dtype=int)
    m, n = A.shape
    C = np.zeros((m, n), dtype=int)
    label = 0
    sizes = []

    for i in range(m):
        for j in range(n):
            if A[i, j] == 1 and C[i, j] == 0:
                label += 1
                stack = [(i, j)]
                C[i, j] = label
                count = 0
                while stack:
                    ci, cj = stack.pop()
                    count += 1
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < m and 0 <= nj < n:
                            if A[ni, nj] == 1 and C[ni, nj] == 0:
                                C[ni, nj] = label
                                stack.append((ni, nj))
                sizes.append(count)

    return C, label, np.array(sizes, dtype=int)


def detect_spanning_clusters(C, component_num):
    if component_num == 0:
        return False, []
    m, n = C.shape
    span_labels = set()

    left_labels = set(C[:, 0]) - {0}
    right_labels = set(C[:, n - 1]) - {0}
    span_labels |= left_labels & right_labels

    top_labels = set(C[0, :]) - {0}
    bottom_labels = set(C[m - 1, :]) - {0}
    span_labels |= top_labels & bottom_labels
    return len(span_labels) > 0, list(span_labels)


def percolation_simulation(m, n, p, seed=None):
    if not (0.0 <= p <= 1.0):
        raise ValueError("p 必须在 [0,1] 内。")
    if seed is not None:
        np.random.seed(seed)
    u = (np.random.rand(m, n) < p).astype(int)
    C, comp_num, sizes = components_2d(u)
    spanning, span_labels = detect_spanning_clusters(C, comp_num)
    mean_size = np.mean(sizes) if sizes.size > 0 else 0.0
    largest = np.max(sizes) if sizes.size > 0 else 0
    return {
        'occupation_matrix': u,
        'labels': C,
        'component_num': comp_num,
        'mean_size': mean_size,
        'spanning': spanning,
        'span_labels': span_labels,
        'largest_size': largest
    }


def find_percolation_threshold(m=100, n=100, n_trials=20):
    p_low = 0.0
    p_high = 1.0
    for _ in range(n_trials):
        p_mid = (p_low + p_high) * 0.5
        spans = 0
        for t in range(n_trials):
            res = percolation_simulation(m, n, p_mid, seed=t * 1000 + _)
            if res['spanning']:
                spans += 1
        ratio = spans / n_trials
        if ratio > 0.5:
            p_high = p_mid
        else:
            p_low = p_mid
    return (p_low + p_high) * 0.5


def superconducting_percolation_analysis(order_parameter_field, threshold=0.1):
    field = np.abs(np.asarray(order_parameter_field, dtype=complex))
    occupied = (field > threshold).astype(int)
    C, comp_num, sizes = components_2d(occupied)
    spanning, span_labels = detect_spanning_clusters(C, comp_num)
    filling = np.mean(occupied)
    return {
        'filling_fraction': filling,
        'component_num': comp_num,
        'mean_cluster_size': np.mean(sizes) if sizes.size > 0 else 0.0,
        'spanning': spanning,
        'spanning_labels': span_labels,
        'largest_cluster_size': np.max(sizes) if sizes.size > 0 else 0,
        'labels': C
    }
