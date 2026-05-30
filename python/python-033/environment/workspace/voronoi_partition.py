
import numpy as np


def voronoi_nearest_generator(query_points, generators, metric='euclidean', p=2):
    query_points = np.asarray(query_points, dtype=float)
    generators = np.asarray(generators, dtype=float)

    if metric == 'manhattan':
        dists = np.sum(np.abs(query_points[:, None, :] - generators[None, :, :]), axis=2)
    elif metric == 'chebyshev':
        dists = np.max(np.abs(query_points[:, None, :] - generators[None, :, :]), axis=2)
    elif metric == 'euclidean':
        dists = np.sqrt(np.sum((query_points[:, None, :] - generators[None, :, :]) ** 2, axis=2))
    else:

        dists = np.sum(np.abs(query_points[:, None, :] - generators[None, :, :]) ** p, axis=2) ** (1.0 / p)

    indices = np.argmin(dists, axis=1)
    distances = dists[np.arange(len(query_points)), indices]
    return indices, distances


def voronoi_cell_centroid(generators, density_samples, indices):
    ng = generators.shape[0]
    dim = generators.shape[1]
    centroids = np.zeros((ng, dim))
    counts = np.zeros(ng)
    for i in range(len(density_samples)):
        idx = indices[i]
        centroids[idx] += density_samples[i]
        counts[idx] += 1

    counts = np.where(counts < 1, 1, counts)
    centroids = centroids / counts[:, None]
    return centroids


def interpolate_nuclear_data(query_nz, known_nz, known_data, metric='euclidean'):
    indices, _ = voronoi_nearest_generator(query_nz, known_nz, metric=metric)
    interpolated = known_data[indices]
    return interpolated


def partition_nuclear_chart(nuclides, n_partitions, max_iter=50):
    coords = np.array([(z, n) for z, n, a in nuclides], dtype=float)
    if len(coords) == 0:
        return np.array([]), np.array([])


    z_min, z_max = coords[:, 0].min(), coords[:, 0].max()
    n_min, n_max = coords[:, 1].min(), coords[:, 1].max()
    generators = np.zeros((n_partitions, 2))
    generators[:, 0] = np.linspace(z_min, z_max, n_partitions)
    generators[:, 1] = np.linspace(n_min, n_max, n_partitions)

    for _ in range(max_iter):
        indices, _ = voronoi_nearest_generator(coords, generators)
        new_gens = voronoi_cell_centroid(generators, coords, indices)

        for i in range(n_partitions):
            if np.all(new_gens[i] == 0):
                new_gens[i] = generators[i]
        generators = new_gens

    labels, _ = voronoi_nearest_generator(coords, generators)
    return generators, labels


def test_voronoi_partition():
    np.random.seed(42)
    generators = np.random.rand(10, 2)
    queries = np.random.rand(100, 2)
    indices, dists = voronoi_nearest_generator(queries, generators)
    print(f"[voronoi_partition] Nearest generator indices range: [{indices.min()}, {indices.max()}]")


    known_nz = np.array([[50, 80], [82, 126], [92, 146]], dtype=float)
    known_data = np.array([1.0, 2.0, 3.0])
    query_nz = np.array([[60, 90], [85, 130]], dtype=float)
    interp = interpolate_nuclear_data(query_nz, known_nz, known_data)
    print(f"[voronoi_partition] Interpolated values: {interp}")


if __name__ == "__main__":
    test_voronoi_partition()
