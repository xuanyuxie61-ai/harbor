
import numpy as np
from typing import Tuple


def legendre_zeros(n: int) -> np.ndarray:
    eps = 1e-14
    roots = np.zeros(n, dtype=np.float64)
    for i in range(1, n + 1):
        z = np.cos(np.pi * (i - 0.25) / (n + 0.5))
        z1 = 0.0
        while abs(z - z1) > eps:
            p1 = 1.0
            p2 = 0.0
            for j in range(1, n + 1):
                p3 = p2
                p2 = p1
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
            pp = n * (z * p1 - p2) / (z * z - 1.0)
            z1 = z
            z = z1 - p1 / pp
        roots[i - 1] = z
    return roots


def gauss_legendre_rule(n: int) -> Tuple[np.ndarray, np.ndarray]:
    x = legendre_zeros(n)
    w = np.zeros(n, dtype=np.float64)
    for i in range(n):

        p1 = 1.0
        p2 = 0.0
        for j in range(1, n + 1):
            p3 = p2
            p2 = p1
            p1 = ((2.0 * j - 1.0) * x[i] * p2 - (j - 1.0) * p3) / j
        pp = n * (x[i] * p1 - p2) / (x[i] * x[i] - 1.0)
        w[i] = 2.0 / ((1.0 - x[i] ** 2) * pp ** 2)
    return x, w


def gauss_quadrature_1d(f, a: float, b: float, n: int = 8) -> float:
    x, w = gauss_legendre_rule(n)

    x_mapped = 0.5 * (b - a) * x + 0.5 * (a + b)
    w_mapped = 0.5 * (b - a) * w
    fx = f(x_mapped)
    return float(np.dot(w_mapped, fx))


def gauss_quadrature_nd(f, bounds: np.ndarray, n_per_dim: int = 5) -> float:
    D = bounds.shape[0]
    x_1d, w_1d = gauss_legendre_rule(n_per_dim)

    grids = np.meshgrid(*[x_1d] * D, indexing='ij')
    weights_grid = np.meshgrid(*[w_1d] * D, indexing='ij')
    total_weight = np.ones_like(grids[0])
    for d in range(D):

        grids[d] = 0.5 * (bounds[d, 1] - bounds[d, 0]) * grids[d] + 0.5 * (bounds[d, 1] + bounds[d, 0])
        total_weight *= 0.5 * (bounds[d, 1] - bounds[d, 0]) * weights_grid[d]

    points = np.stack([g.ravel() for g in grids], axis=1)
    weights = total_weight.ravel()
    fx = f(points)
    return float(np.dot(weights, fx))


def tetrahedron_reference_rule(degree: int = 5) -> Tuple[np.ndarray, np.ndarray]:

    if degree <= 1:

        nodes = np.array([[0.25, 0.25, 0.25]]).T
        weights = np.array([1.0 / 6.0])
    elif degree <= 2:

        a = 0.58541020
        b = 0.13819660
        nodes = np.array([
            [a, b, b],
            [b, a, b],
            [b, b, a],
            [b, b, b]
        ]).T
        weights = np.ones(4) / 24.0
    elif degree <= 3:

        nodes = np.array([
            [0.25, 0.25, 0.25],
            [0.5, 1.0/6.0, 1.0/6.0],
            [1.0/6.0, 0.5, 1.0/6.0],
            [1.0/6.0, 1.0/6.0, 0.5],
            [1.0/6.0, 1.0/6.0, 1.0/6.0]
        ]).T
        weights = np.array([-2.0/15.0, 3.0/40.0, 3.0/40.0, 3.0/40.0, 3.0/40.0])
    elif degree <= 5:


        nodes_list = [[0.25, 0.25, 0.25]]
        weights_list = [8.0 / 405.0]

        face_centers = [
            [1.0/3.0, 1.0/3.0, 1.0/3.0],
            [0.0, 1.0/3.0, 1.0/3.0],
            [1.0/3.0, 0.0, 1.0/3.0],
            [1.0/3.0, 1.0/3.0, 0.0]
        ]
        nodes_list.extend(face_centers)
        weights_list.extend([-(1.0/30.0)] * 4)

        edge_pts = []
        for i in range(4):
            for j in range(i + 1, 4):
                pt = [0.0, 0.0, 0.0]
                if i == 0:
                    pt = [0.5, 0.0, 0.0]
                elif i == 1 and j == 2:
                    pt = [0.5, 0.5, 0.0]
                elif i == 1 and j == 3:
                    pt = [0.5, 0.0, 0.5]
                elif i == 2 and j == 3:
                    pt = [0.0, 0.5, 0.5]
                edge_pts.append(pt)


        a1 = 0.25
        a2 = 1.0 / 3.0
        a3 = 0.5
        a4 = 1.0 / 6.0
        b1 = 8.0 / 405.0
        b2 = -1.0 / 30.0
        b3 = 1.0 / 45.0
        nodes_list = [
            [a1, a1, a1],
            [a2, a2, a2], [0.0, a2, a2], [a2, 0.0, a2], [a2, a2, 0.0],
            [a3, a4, a4], [a4, a3, a4], [a4, a4, a3], [a4, a4, a4]
        ]
        weights_list = [b1, b2, b2, b2, b2, b3, b3, b3, b3]

        vol = 1.0 / 6.0
        w_sum = sum(weights_list)
        weights_list = [w * vol / w_sum for w in weights_list]
        nodes = np.array(nodes_list).T
        weights = np.array(weights_list)
    else:

        nodes, weights = tetrahedron_reference_rule(5)
    return nodes, weights


def integrate_on_manifold(data: np.ndarray, f, sigma: float = 1.0,
                          n_samples: int = 1000) -> float:
    N = len(data)
    if N == 0:
        return 0.0

    densities = np.zeros(N)
    for i in range(N):
        dists = np.linalg.norm(data - data[i], axis=1)
        densities[i] = np.sum(np.exp(-dists ** 2 / (2.0 * sigma ** 2)))

    weights = 1.0 / (densities + 1e-10)
    weights = weights / np.sum(weights)

    fx = f(data)
    return float(np.dot(weights, fx))


def manifold_volume_element(data: np.ndarray, k: int = 10) -> np.ndarray:
    N, D = data.shape
    volumes = np.zeros(N)
    for i in range(N):
        dists = np.linalg.norm(data - data[i], axis=1)
        idx = np.argsort(dists)[1:k + 1]
        local_data = data[idx] - data[i]

        if D <= k:

            gram = local_data[:D].T @ local_data[:D]
            vol = np.sqrt(max(np.linalg.det(gram), 0.0))
            volumes[i] = vol
        else:

            vol = np.prod(dists[idx] ** (1.0 / k))
            volumes[i] = vol
    return volumes
