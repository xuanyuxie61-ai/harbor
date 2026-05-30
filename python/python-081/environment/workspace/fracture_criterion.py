
import numpy as np


def threshold_damage_field(damage_values, threshold):
    damage_values = np.array(damage_values, dtype=float)
    threshold = float(np.clip(threshold, 0.0, 1.0))
    fractured = damage_values >= threshold
    return fractured


def otsu_threshold(damage_values):
    vals = np.array(damage_values, dtype=float)
    vals = np.clip(vals, 0.0, 1.0)

    n_bins = 256
    hist, bin_edges = np.histogram(vals, bins=n_bins, range=(0.0, 1.0))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    total = len(vals)
    total_sum = np.sum(vals)

    max_var = -1.0
    best_t = 0.5

    w0 = 0
    sum0 = 0.0

    for i in range(n_bins):
        w0 += hist[i]
        sum0 += hist[i] * bin_centers[i]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        mu0 = sum0 / w0
        mu1 = (total_sum - sum0) / w1
        var_between = (w0 / total) * (w1 / total) * (mu0 - mu1) ** 2
        if var_between > max_var:
            max_var = var_between
            best_t = bin_centers[i]

    return float(best_t)


def fracture_energy_release_rate(damage_history, strain_energy_history, element_volume):
    dD = np.diff(damage_history, prepend=0.0)

    Y = np.zeros_like(damage_history)
    for i in range(len(damage_history)):
        denom = max(1.0 - damage_history[i], 1e-12)
        Y[i] = strain_energy_history[i] / denom

    dissipated = np.sum(Y * dD) * element_volume

    A_crack = element_volume ** (2.0 / 3.0)
    G = dissipated / max(A_crack, 1e-12)
    return G


def extract_fracture_clusters(fractured_flags, connectivity):
    n_elem = len(fractured_flags)
    visited = np.zeros(n_elem, dtype=bool)
    clusters = []


    adjacency = [set() for _ in range(n_elem)]
    for i in range(n_elem):
        if not fractured_flags[i]:
            continue
        nodes_i = set(connectivity[i])
        for j in range(i + 1, n_elem):
            if not fractured_flags[j]:
                continue
            nodes_j = set(connectivity[j])
            if len(nodes_i & nodes_j) >= 3:
                adjacency[i].add(j)
                adjacency[j].add(i)


    for i in range(n_elem):
        if not fractured_flags[i] or visited[i]:
            continue
        stack = [i]
        cluster = []
        visited[i] = True
        while stack:
            cur = stack.pop()
            cluster.append(cur)
            for nb in adjacency[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        clusters.append(cluster)

    return clusters


def critical_damage_criterion(equivalent_stress, yield_stress, damage_evolution_rate,
                               material_fracture_toughness, element_size):
    sigma_eq = float(equivalent_stress)
    sigma_y = float(yield_stress)
    dD_dt = float(damage_evolution_rate)
    G_c = float(material_fracture_toughness)
    h = float(element_size)

    if sigma_eq < sigma_y or dD_dt <= 1e-15:
        return False, 0.0


    D_c = 1.0 - G_c / max(sigma_y * h, 1e-12)
    D_c = np.clip(D_c, 0.1, 0.99)

    return True, D_c
