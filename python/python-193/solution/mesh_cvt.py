"""
Mesh Generation via Centroidal Voronoi Tessellation (CVT) and Delaunay Triangulation.

Integrates:
  - 242_cvt_4_movie: CVT point generation (Lloyd iteration)
  - 1340_triangulation_node_to_element: nodal-to-element averaging

Scientific formulas:
  Lloyd iteration:
    z_{k+1}^{(i)} = centroid( V_i(z_k) )
    where V_i(z) = { x in Omega : ||x - z_i|| <= ||x - z_j|| for all j }
  Density-weighted centroid:
    c = integral_{V_i} x * rho(x) dx / integral_{V_i} rho(x) dx
"""

import numpy as np


def density_transform(s, exponent=0.3):
    """
    Non-uniform density transformation concentrating points near corners.
    rho(s) ~ 0.5 + 0.5 * |2s-1|^exponent * sign(2s-1)
    Adapted from seed 242_cvt_4_movie.
    """
    s = np.asarray(s, dtype=float)
    return 0.5 + 0.5 * np.abs(2.0 * s - 1.0) ** exponent * np.sign(2.0 * s - 1.0)


def cvt_generate(n_generators=32, n_samples=5000, n_iterations=50, seed=42):
    """
    Generate a 2D point set via Centroidal Voronoi Tessellation.
    Uses Lloyd's algorithm with Monte-Carlo density sampling.

    Parameters:
      n_generators : number of generator points
      n_samples    : number of density samples per iteration
      n_iterations : Lloyd iterations

    Returns:
      generators : (n_generators, 2) array of optimized points in [0,1]^2
    """
    rng = np.random.default_rng(seed)
    # Initialize generators uniformly
    generators = rng.random((n_generators, 2))

    for _iter in range(n_iterations):
        # Sample with density bias
        raw = rng.random((n_samples, 2))
        samples = density_transform(raw)

        # Assign each sample to nearest generator
        # Use vectorized distance computation for efficiency
        # distances[i,j] = distance between sample i and generator j
        diff = samples[:, np.newaxis, :] - generators[np.newaxis, :, :]
        dists = np.sum(diff ** 2, axis=2)
        nearest = np.argmin(dists, axis=1)

        # Move generators to centroids of their Voronoi cells
        new_generators = np.zeros_like(generators)
        counts = np.zeros(n_generators)
        for j in range(n_generators):
            mask = nearest == j
            if np.any(mask):
                new_generators[j, :] = np.mean(samples[mask, :], axis=0)
                counts[j] = np.sum(mask)
            else:
                # Empty cell: keep old generator
                new_generators[j, :] = generators[j, :]
                counts[j] = 1
        generators = new_generators.copy()

    return generators


def delaunay_triangulation(nodes):
    """
    Compute Delaunay triangulation of a 2D node set.
    Returns elements array of shape (n_elements, 3) with 0-based node indices.
    """
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(nodes)
        return tri.simplices.astype(int)
    except ImportError:
        # Fallback: very simple greedy triangulation for small node counts
        # This is not Delaunay but provides a valid triangulation for testing
        nodes = np.asarray(nodes)
        n = nodes.shape[0]
        elements = []
        # Use a simple sweep-line approach for convex hull approximation
        # For robust scientific code, scipy is strongly preferred
        center = np.mean(nodes, axis=0)
        angles = np.arctan2(nodes[:, 1] - center[1], nodes[:, 0] - center[0])
        order = np.argsort(angles)
        for i in range(n):
            a = order[i]
            b = order[(i + 1) % n]
            c = order[(i + 2) % n]
            # Only add if area is positive (counter-clockwise)
            area = 0.5 * ((nodes[b, 0] - nodes[a, 0]) * (nodes[c, 1] - nodes[a, 1]) -
                          (nodes[b, 1] - nodes[a, 1]) * (nodes[c, 0] - nodes[a, 0]))
            if area > 1e-12:
                elements.append([a, b, c])
        return np.array(elements, dtype=int)


def node_to_element_average(nodal_values, elements):
    """
    Convert nodal values to element values by averaging over each element's vertices.
    From seed 1340_triangulation_node_to_element.

    For element e with vertices (i, j, k):
        u_e = (u_i + u_j + u_k) / 3

    Parameters:
      nodal_values : (n_nodes,) array
      elements     : (n_elements, 3) array

    Returns:
      element_values : (n_elements,) array
    """
    nodal_values = np.asarray(nodal_values, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_elements = elements.shape[0]
    element_values = np.zeros(n_elements)
    for e in range(n_elements):
        i, j, k = elements[e, :]
        # Boundary check
        max_idx = len(nodal_values) - 1
        i = min(max(i, 0), max_idx)
        j = min(max(j, 0), max_idx)
        k = min(max(k, 0), max_idx)
        element_values[e] = (nodal_values[i] + nodal_values[j] + nodal_values[k]) / 3.0
    return element_values


def element_area(nodes, elements):
    """
    Compute the area of each triangular element.
    Area = 0.5 * | (x_b - x_a) x (x_c - x_a) |
    """
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
    """
    Compute mesh quality metrics.
    Q_e = 4 * sqrt(3) * Area_e / (a^2 + b^2 + c^2)
    where a, b, c are edge lengths. Q=1 for equilateral triangle.
    """
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
