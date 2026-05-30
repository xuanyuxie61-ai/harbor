import numpy as np
from combustion_utils import check_positive, check_nonnegative, cholesky_factor, solve_lower_triangular


def triangle_area(t):
    t = np.asarray(t, dtype=float)
    if t.shape != (2, 3):
        raise ValueError("t must be shape (2,3), got " + str(t.shape))
    area = abs(t[0, 0] * (t[1, 1] - t[1, 2]) +
               t[0, 1] * (t[1, 2] - t[1, 0]) +
               t[0, 2] * (t[1, 0] - t[1, 1]))
    if area < 1.0e-14:
        return 0.0
    return area * 0.5


def basis_t3(t, i, p):
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    area = triangle_area(t)
    if area <= 0.0:
        raise ValueError("Triangle has zero or negative area")
    if i not in (0, 1, 2):
        raise ValueError("Node index i must be 0, 1, or 2")

    ip1 = (i + 1) % 3
    ip2 = (i + 2) % 3

    phi = ((t[0, ip2] - t[0, ip1]) * (p[1] - t[1, ip1]) -
           (t[1, ip2] - t[1, ip1]) * (p[0] - t[0, ip1])) / (2.0 * area)

    dphi_dx = -(t[1, ip2] - t[1, ip1]) / (2.0 * area)
    dphi_dy = (t[0, ip2] - t[0, ip1]) / (2.0 * area)
    return phi, dphi_dx, dphi_dy


def sample_square_uniform(n):
    return np.random.rand(n, 2)


def cvt_iteration(points, n_samples=1000, n_iter=20):
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    check_positive(n, "n_points")
    check_positive(n_samples, "n_samples")
    check_positive(n_iter, "n_iter")

    for it in range(n_iter):
        samples = sample_square_uniform(n_samples)

        counts = np.zeros(n)
        centroids = np.zeros((n, 2))
        for s in samples:
            dists = np.sum((points - s) ** 2, axis=1)
            j = np.argmin(dists)
            counts[j] += 1
            centroids[j] += s

        for j in range(n):
            if counts[j] > 0:
                points[j] = centroids[j] / counts[j]
            else:

                points[j] = np.random.rand(2)
    return points


def adaptive_density_function(x, y, wave_x=0.5, wave_width=0.05,
                              max_density=10.0, min_density=1.0):
    dx = x - wave_x
    gauss = np.exp(-(dx / wave_width) ** 2)
    return min_density + (max_density - min_density) * gauss


class AdaptiveDetonationMesh:

    def __init__(self, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
                 n_base=400, wave_x=0.5, wave_width=0.05):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.n_base = n_base
        self.wave_x = wave_x
        self.wave_width = wave_width

    def generate(self, cvt_samples=2000, cvt_iter=15):

        points = sample_square_uniform(self.n_base)


        accepted = []
        max_trials = self.n_base * 20
        trial = 0
        while len(accepted) < self.n_base and trial < max_trials:
            trial += 1
            p = np.random.rand(2)
            x = self.x_min + p[0] * (self.x_max - self.x_min)
            y = self.y_min + p[1] * (self.y_max - self.y_min)
            dens = adaptive_density_function(
                x, y, self.wave_x, self.wave_width,
                max_density=10.0, min_density=1.0
            )
            if np.random.rand() < dens / 10.0:
                accepted.append([x, y])

        if len(accepted) < self.n_base // 2:

            accepted = sample_square_uniform(self.n_base).tolist()

        points = np.array(accepted[:self.n_base])


        points = cvt_iteration(points, n_samples=cvt_samples, n_iter=cvt_iter)


        points[:, 0] = self.x_min + points[:, 0] * (self.x_max - self.x_min)
        points[:, 1] = self.y_min + points[:, 1] * (self.y_max - self.y_min)




        nodes = points
        elements = self._simple_triangulation(nodes)
        return nodes, elements

    def _simple_triangulation(self, nodes):
        n = nodes.shape[0]

        if n < 3:
            return np.zeros((0, 3), dtype=int)


        elements = []
        used = set()
        for i in range(n):
            dists = np.sum((nodes - nodes[i]) ** 2, axis=1)
            dists[i] = np.inf

            j = np.argmin(dists)
            dists[j] = np.inf
            k = np.argmin(dists)
            tri = tuple(sorted((i, j, k)))
            if tri not in used:

                t = nodes[[i, j, k]].T
                area = triangle_area(t)
                if area > 1.0e-12:
                    used.add(tri)
                    elements.append([i, j, k])
            if len(elements) >= 2 * n:
                break

        if len(elements) == 0:
            return np.zeros((0, 3), dtype=int)
        return np.array(elements, dtype=int)

    def element_quality(self, nodes, elements):
        qualities = []
        for elem in elements:
            t = nodes[elem].T
            area = triangle_area(t)
            if area < 1.0e-14:
                qualities.append(0.0)
                continue

            a = np.linalg.norm(nodes[elem[1]] - nodes[elem[0]])
            b = np.linalg.norm(nodes[elem[2]] - nodes[elem[1]])
            c = np.linalg.norm(nodes[elem[0]] - nodes[elem[2]])




            denom = (a + b + c) * a * b * c
            if denom < 1.0e-14:
                qualities.append(0.0)
            else:
                q = 8.0 * area * area / denom
                qualities.append(min(q, 1.0))
        return np.mean(qualities) if qualities else 0.0
