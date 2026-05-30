
import numpy as np


def voronoi_area_2d(seeds, bbox=((-1, 1), (-1, 1)), resolution=500):
    seeds = np.asarray(seeds)
    N = len(seeds)
    if N == 0:
        return np.array([]), 1.0

    xmin, xmax = bbox[0]
    ymin, ymax = bbox[1]

    x_grid = np.linspace(xmin, xmax, resolution)
    y_grid = np.linspace(ymin, ymax, resolution)
    dx = (xmax - xmin) / resolution
    dy = (ymax - ymin) / resolution

    areas = np.zeros(N)

    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            dists = np.sqrt((seeds[:, 0] - x) ** 2 + (seeds[:, 1] - y) ** 2)
            nearest = np.argmin(dists)
            areas[nearest] += dx * dy

    min_area = np.min(areas) if np.min(areas) > 0 else 1.0e-12
    max_area = np.max(areas) if np.max(areas) > 0 else 1.0
    load_balance = max_area / min_area

    return areas, load_balance


def domain_decomposition_1d(Z_nodes, n_subdomains):
    n = len(Z_nodes)
    nodes_per_sub = max(1, n // n_subdomains)

    subdomain_indices = []
    subdomain_bounds = []

    for i in range(n_subdomains):
        start = i * nodes_per_sub
        end = min((i + 1) * nodes_per_sub + 2, n) if i < n_subdomains - 1 else n

        if i > 0:
            start = max(0, start - 1)

        subdomain_indices.append((start, end))
        subdomain_bounds.append((Z_nodes[start], Z_nodes[end - 1]))

    return subdomain_indices, subdomain_bounds


def compute_partition_quality(areas):
    mean_area = np.mean(areas)
    std_area = np.std(areas)

    if mean_area < 1.0e-12:
        mean_area = 1.0e-12

    quality = {
        'load_imbalance': np.max(areas) / mean_area,
        'coefficient_of_variation': std_area / mean_area,
        'min_area': np.min(areas),
        'max_area': np.max(areas),
        'mean_area': mean_area,
        'num_partitions': len(areas),
    }

    return quality
