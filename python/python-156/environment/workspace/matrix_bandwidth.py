
import numpy as np


def compute_matrix_bandwidth(element_nodes):
    element_nodes = np.asarray(element_nodes, dtype=int)
    element_num, element_order = element_nodes.shape

    ml = 0
    mu = 0

    for element in range(element_num):
        for local_i in range(element_order):
            global_i = element_nodes[element, local_i]
            for local_j in range(element_order):
                global_j = element_nodes[element, local_j]

                mu = max(mu, global_j - global_i)
                ml = max(ml, global_i - global_j)

    bandwidth = ml + 1 + mu
    return ml, mu, bandwidth


def estimate_sparse_storage(n_nodes, bandwidth, element_order=2):
    dense_elements = n_nodes * n_nodes
    band_elements = n_nodes * bandwidth

    dense_bytes = dense_elements * 8
    band_bytes = band_elements * 8

    sparse_ratio = band_elements / dense_elements if dense_elements > 0 else 1.0

    return dense_bytes, band_bytes, sparse_ratio


def analyze_flamelet_bandwidth(n, fem_type='linear'):
    if fem_type == 'linear':
        element_order = 2

        e_num = n - 1
        element_nodes = np.zeros((e_num, 2), dtype=int)
        for e in range(e_num):
            element_nodes[e] = [e, e + 1]
    elif fem_type == 'quadratic':
        element_order = 3
        if n % 2 == 0:
            n += 1
        e_num = (n - 1) // 2
        element_nodes = np.zeros((e_num, 3), dtype=int)
        for e in range(e_num):
            element_nodes[e] = [2 * e, 2 * e + 1, 2 * e + 2]
    else:
        raise ValueError("fem_type 必须是 'linear' 或 'quadratic'")

    ml, mu, bandwidth = compute_matrix_bandwidth(element_nodes)
    dense_bytes, band_bytes, sparse_ratio = estimate_sparse_storage(n, bandwidth, element_order)

    analysis = {
        'n_nodes': n,
        'fem_type': fem_type,
        'element_order': element_order,
        'n_elements': e_num,
        'lower_bandwidth': ml,
        'upper_bandwidth': mu,
        'total_bandwidth': bandwidth,
        'dense_storage_bytes': dense_bytes,
        'band_storage_bytes': band_bytes,
        'sparse_ratio': sparse_ratio,
    }

    return analysis
