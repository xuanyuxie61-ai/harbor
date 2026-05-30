
import numpy as np


def density_transform(s, exponent=0.3):
    s = np.asarray(s, dtype=float)
    return 0.5 + 0.5 * np.abs(2.0 * s - 1.0) ** exponent * np.sign(2.0 * s - 1.0)


def cvt_generate(n_generators=32, n_samples=5000, n_iterations=50, seed=42):
    rng = np.random.default_rng(seed)

    generators = rng.random((n_generators, 2))

    for _iter in range(n_iterations):

        raw = rng.random((n_samples, 2))
        samples = density_transform(raw)




        diff = samples[:, np.newaxis, :] - generators[np.newaxis, :, :]
        dists = np.sum(diff ** 2, axis=2)
        nearest = np.argmin(dists, axis=1)


        new_generators = np.zeros_like(generators)
        counts = np.zeros(n_generators)
        for j in range(n_generators):
            mask = nearest == j
            if np.any(mask):
                new_generators[j, :] = np.mean(samples[mask, :], axis=0)
                counts[j] = np.sum(mask)
            else:

                new_generators[j, :] = generators[j, :]
                counts[j] = 1
        generators = new_generators.copy()

    return generators


def delaunay_triangulation(nodes):
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(nodes)
        return tri.simplices.astype(int)
    except ImportError:


        nodes = np.asarray(nodes)
        n = nodes.shape[0]
        elements = []


        center = np.mean(nodes, axis=0)
        angles = np.arctan2(nodes[:, 1] - center[1], nodes[:, 0] - center[0])
        order = np.argsort(angles)
        for i in range(n):
            a = order[i]
            b = order[(i + 1) % n]
            c = order[(i + 2) % n]

            area = 0.5 * ((nodes[b, 0] - nodes[a, 0]) * (nodes[c, 1] - nodes[a, 1]) -
                          (nodes[b, 1] - nodes[a, 1]) * (nodes[c, 0] - nodes[a, 0]))
            if area > 1e-12:
                elements.append([a, b, c])
        return np.array(elements, dtype=int)


def node_to_element_average(nodal_values, elements):
    nodal_values = np.asarray(nodal_values, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_elements = elements.shape[0]
    element_values = np.zeros(n_elements)
    for e in range(n_elements):
        i, j, k = elements[e, :]

        max_idx = len(nodal_values) - 1
        i = min(max(i, 0), max_idx)
        j = min(max(j, 0), max_idx)
        k = min(max(k, 0), max_idx)
        element_values[e] = (nodal_values[i] + nodal_values[j] + nodal_values[k]) / 3.0
    return element_values


def element_area(nodes, elements):
    nodes = np.asarray(nodes)
    elements = np.asarray(elements)
    n_elements = elements.shape[0]
    areas = np.zeros(n_elements)
    for e in range(n_elements):
        i, j, k = elements[e, :]
        x1, y1 = nodes[i]
        x2, y2 = nodes[j]
        x3, y3 = nodes[k]
        areas[e] = 0.5 * abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
    return areas


def compute_mesh_quality(nodes, elements):
    nodes = np.asarray(nodes)
    elements = np.asarray(elements)
    n_elements = elements.shape[0]
    quality = np.zeros(n_elements)
    areas = element_area(nodes, elements)
    for e in range(n_elements):
        i, j, k = elements[e, :]
        a2 = np.sum((nodes[j] - nodes[i]) ** 2)
        b2 = np.sum((nodes[k] - nodes[j]) ** 2)
        c2 = np.sum((nodes[i] - nodes[k]) ** 2)
        denom = a2 + b2 + c2
        if denom > 1e-15:
            quality[e] = 4.0 * np.sqrt(3.0) * areas[e] / denom
        else:
            quality[e] = 0.0
    return quality
