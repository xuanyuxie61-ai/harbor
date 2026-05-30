
import numpy as np


def compute_voronoi_cells(sample_points, generators):
    if sample_points.ndim == 1:
        sample_points = sample_points.reshape(-1, 1)
    if generators.ndim == 1:
        generators = generators.reshape(-1, 1)

    n_samples = sample_points.shape[0]
    n_gen = generators.shape[0]


    diff = sample_points[:, np.newaxis, :] - generators[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    nearest = np.argmin(dist_sq, axis=1)
    return nearest


def cvt_iterate(generators, sample_points, density_values=None):
    n_gen = generators.shape[0]
    dim = generators.shape[1]

    if density_values is None:
        density_values = np.ones(sample_points.shape[0])

    nearest = compute_voronoi_cells(sample_points, generators)

    new_generators = np.zeros_like(generators)
    counts = np.zeros(n_gen)
    energy = 0.0

    for j in range(n_gen):
        mask = (nearest == j)
        count = np.sum(mask)
        if count > 0:
            weights = density_values[mask]
            weighted_sum = np.sum(
                sample_points[mask] * weights[:, np.newaxis], axis=0
            )
            total_weight = np.sum(weights)
            new_generators[j] = weighted_sum / total_weight
            counts[j] = total_weight


            diff = sample_points[mask] - generators[j]
            energy += np.sum(weights[:, np.newaxis] * (diff ** 2))
        else:

            new_generators[j] = generators[j]

    energy = energy / sample_points.shape[0]

    it_diff = np.sum(np.sqrt(np.sum((new_generators - generators) ** 2, axis=1)))

    return new_generators, it_diff, energy


def cvt_energy(generators, sample_points, density_values=None):
    if density_values is None:
        density_values = np.ones(sample_points.shape[0])

    nearest = compute_voronoi_cells(sample_points, generators)
    energy = 0.0
    for j in range(generators.shape[0]):
        mask = (nearest == j)
        if np.sum(mask) > 0:
            diff = sample_points[mask] - generators[j]
            energy += np.sum(
                density_values[mask][:, np.newaxis] * (diff ** 2)
            )

    return energy / sample_points.shape[0]


def generate_cvt_mesh(
    n_cells,
    domain_bounds,
    density_func=None,
    it_max=50,
    sample_multiplier=100,
    tol=1e-5
):
    dim = 2
    sample_num = n_cells * sample_multiplier


    rng = np.random.default_rng(seed=42)
    generators = rng.random((n_cells, dim))
    generators[:, 0] = generators[:, 0] * (domain_bounds[0][1] - domain_bounds[0][0]) + domain_bounds[0][0]
    generators[:, 1] = generators[:, 1] * (domain_bounds[1][1] - domain_bounds[1][0]) + domain_bounds[1][0]

    energy_history = []

    for it in range(it_max):
        sample_points = rng.random((sample_num, dim))
        sample_points[:, 0] = sample_points[:, 0] * (domain_bounds[0][1] - domain_bounds[0][0]) + domain_bounds[0][0]
        sample_points[:, 1] = sample_points[:, 1] * (domain_bounds[1][1] - domain_bounds[1][0]) + domain_bounds[1][0]

        if density_func is not None:
            density_values = np.array([
                density_func(p[0], p[1]) for p in sample_points
            ])

            density_values = np.maximum(density_values, 1e-10)
        else:
            density_values = None

        generators, it_diff, energy = cvt_iterate(
            generators, sample_points, density_values
        )
        energy_history.append(energy)


        generators[:, 0] = np.clip(generators[:, 0], domain_bounds[0][0], domain_bounds[0][1])
        generators[:, 1] = np.clip(generators[:, 1], domain_bounds[1][0], domain_bounds[1][1])

        if it_diff < tol:
            break

    return generators, energy_history


def compute_delaunay_triangulation(points):

    tol = 1e-10
    unique_points = []
    for p in points:
        is_duplicate = False
        for up in unique_points:
            if np.linalg.norm(p - up) < tol:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_points.append(p)
    nodes = np.array(unique_points)
    n = len(nodes)

    if n < 3:
        raise ValueError("compute_delaunay_triangulation: 至少需要3个节点")


    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(nodes)
        triangles = tri.simplices.astype(int)
        return triangles, nodes
    except ImportError:
        pass



    xmin, ymin = nodes.min(axis=0)
    xmax, ymax = nodes.max(axis=0)



    triangles = []
    used = set()

    for i in range(n):
        dists = np.linalg.norm(nodes - nodes[i], axis=1)
        dists[i] = np.inf
        neighbors = np.argsort(dists)[:min(6, n - 1)]
        for j in range(len(neighbors)):
            for k in range(j + 1, len(neighbors)):
                a, b, c = i, neighbors[j], neighbors[k]
                key = tuple(sorted([a, b, c]))
                if key not in used:

                    x1, y1 = nodes[a]
                    x2, y2 = nodes[b]
                    x3, y3 = nodes[c]
                    area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                    if area > 1e-12:
                        triangles.append([a, b, c])
                        used.add(key)
                    elif area < -1e-12:
                        triangles.append([a, c, b])
                        used.add(key)

    if len(triangles) == 0:

        triangles = [[0, 1, 2]] if n >= 3 else []

    return np.array(triangles, dtype=int), nodes


def triangle_area(nodes, triangle):
    p1 = nodes[triangle[0]]
    p2 = nodes[triangle[1]]
    p3 = nodes[triangle[2]]
    area = 0.5 * abs(
        (p2[0] - p1[0]) * (p3[1] - p1[1]) -
        (p3[0] - p1[0]) * (p2[1] - p1[1])
    )
    return area
