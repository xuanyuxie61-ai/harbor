
import numpy as np


def hex_grid_points(nodes_per_layer, layers, box):
    nodes_per_layer = int(nodes_per_layer)
    layers = int(layers)
    if nodes_per_layer < 1:
        return np.zeros((0, 2))
    if nodes_per_layer == 1:
        pt = (box[:, 0] + box[:, 1]) / 2.0
        return pt.reshape(1, 2)

    hx = (box[0, 1] - box[0, 0]) / (nodes_per_layer - 1)
    hy = hx * np.sqrt(3.0) / 2.0

    points = []
    for j in range(layers):
        y = box[1, 0] + hy * j
        jmod = j % 2
        if jmod == 0:
            for i in range(nodes_per_layer):
                x = box[0, 0] + (box[0, 1] - box[0, 0]) * i / (nodes_per_layer - 1)
                points.append([x, y])
        else:
            for i in range(nodes_per_layer - 1):
                x = box[0, 0] + (box[0, 1] - box[0, 0]) * (2 * i + 1) / (2 * nodes_per_layer - 2)
                points.append([x, y])

    return np.array(points, dtype=float)


def hex_grid_approximate_n(nodes_per_layer, layers):
    if nodes_per_layer < 1:
        return 0
    if nodes_per_layer == 1:
        return 1
    n_odd = ((layers + 1) // 2) * nodes_per_layer
    n_even = (layers // 2) * (nodes_per_layer - 1)
    return n_odd + n_even


def find_closest(ndim, n_generators, n_samples, samples, generators):



    s = samples.reshape(ndim, n_samples, 1)
    g = generators.reshape(ndim, 1, n_generators)
    dists = np.sum((s - g) ** 2, axis=0)
    nearest = np.argmin(dists, axis=1)
    return nearest


def cvt_iterate_2d(generators, ratio, region_box):
    ndim = 2
    n = generators.shape[1]
    sample_num = ratio * n


    samples = np.zeros((ndim, sample_num), dtype=float)
    samples[0, :] = region_box[0, 0] + np.random.rand(sample_num) * (region_box[0, 1] - region_box[0, 0])
    samples[1, :] = region_box[1, 0] + np.random.rand(sample_num) * (region_box[1, 1] - region_box[1, 0])

    nearest = find_closest(ndim, n, sample_num, samples, generators)

    generators_new = np.zeros_like(generators)
    counts = np.zeros(n, dtype=int)
    energy = 0.0

    for j in range(sample_num):
        idx = nearest[j]
        generators_new[:, idx] += samples[:, j]
        energy += np.sum((generators[:, idx] - samples[:, j]) ** 2)
        counts[idx] += 1


    for j in range(n):
        if counts[j] > 0:
            generators_new[:, j] /= counts[j]
        else:
            generators_new[:, j] = generators[:, j]

    energy /= sample_num
    diff = np.sum(np.sqrt(np.sum((generators_new - generators) ** 2, axis=0)))

    return generators_new, diff, energy


def cvt_optimize_2d(initial_points, region_box, it_max=50, ratio=1000, tol=1e-5):
    generators = initial_points.T.copy()
    n = generators.shape[1]
    if n == 0:
        return initial_points

    for it in range(it_max):
        generators_new, diff, energy = cvt_iterate_2d(generators, ratio, region_box)
        generators = generators_new
        if diff < tol * n:
            break

    return generators.T.copy()


def adaptive_density_function(x, y, shock_center, shock_width, base_density=1.0,
                              peak_density=10.0):
    dx = x - shock_center[0]
    dy = y - shock_center[1]
    r2 = dx ** 2 + dy ** 2
    sigma2 = 2.0 * shock_width ** 2
    if sigma2 <= 0.0:
        return np.full_like(x, base_density) if isinstance(x, np.ndarray) else base_density
    rho = base_density + (peak_density - base_density) * np.exp(-r2 / sigma2)
    return rho


def rejection_sampling_adaptive(n_points, region_box, shock_center, shock_width,
                                base_density=1.0, peak_density=10.0, max_trials=1000000):
    points = []
    trials = 0
    while len(points) < n_points and trials < max_trials:
        x = region_box[0, 0] + np.random.rand() * (region_box[0, 1] - region_box[0, 0])
        y = region_box[1, 0] + np.random.rand() * (region_box[1, 1] - region_box[1, 0])
        rho_val = adaptive_density_function(x, y, shock_center, shock_width,
                                            base_density, peak_density)

        if np.random.rand() < (rho_val / peak_density):
            points.append([x, y])
        trials += 1

    if len(points) < n_points:

        remaining = n_points - len(points)
        x_rand = region_box[0, 0] + np.random.rand(remaining) * (region_box[0, 1] - region_box[0, 0])
        y_rand = region_box[1, 0] + np.random.rand(remaining) * (region_box[1, 1] - region_box[1, 0])
        extra = np.column_stack((x_rand, y_rand))
        if len(points) > 0:
            return np.vstack((np.array(points), extra))
        return extra

    return np.array(points, dtype=float)


class AcousticMesh:

    def __init__(self, box, method='hex', nodes_per_layer=20, layers=20,
                 cvt_iters=30, adaptive=False, shock_center=None, shock_width=None):
        self.box = np.asarray(box, dtype=float)
        if self.box.shape != (2, 2):
            raise ValueError("box must have shape (2, 2)")
        if np.any(self.box[:, 1] <= self.box[:, 0]):
            raise ValueError("box max must be greater than min in each dimension.")

        self.method = method
        self.nodes_per_layer = int(nodes_per_layer)
        self.layers = int(layers)
        self.cvt_iters = int(cvt_iters)

        if method == 'hex':
            self.points = hex_grid_points(nodes_per_layer, layers, self.box)
        elif method == 'cvt':
            n = hex_grid_approximate_n(nodes_per_layer, layers)
            init = np.random.rand(n, 2)
            init[:, 0] = self.box[0, 0] + init[:, 0] * (self.box[0, 1] - self.box[0, 0])
            init[:, 1] = self.box[1, 0] + init[:, 1] * (self.box[1, 1] - self.box[1, 0])
            self.points = cvt_optimize_2d(init, self.box, it_max=cvt_iters)
        elif method == 'adaptive_cvt':
            if shock_center is None or shock_width is None:
                raise ValueError("adaptive_cvt requires shock_center and shock_width.")
            n = hex_grid_approximate_n(nodes_per_layer, layers)
            init = rejection_sampling_adaptive(n, self.box, shock_center, shock_width)
            self.points = cvt_optimize_2d(init, self.box, it_max=cvt_iters)
        else:
            raise ValueError(f"Unknown mesh method: {method}")

        self.n_points = self.points.shape[0]
        if self.n_points == 0:
            raise ValueError("Mesh generation produced zero points.")

    def compute_element_size(self):
        if self.n_points < 2:
            return 0.0

        pts = self.points

        if self.n_points > 2000:
            idx = np.random.choice(self.n_points, 2000, replace=False)
            pts_sub = pts[idx]
        else:
            idx = np.arange(self.n_points)
            pts_sub = pts


        diff = pts_sub[:, np.newaxis, :] - pts_sub[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        return float(np.mean(min_dists))

    def compute_mesh_quality(self):
        if self.n_points == 0:
            return np.inf
        ratio = max(10, min(1000, 50000 // self.n_points))
        sample_num = ratio * self.n_points
        samples = np.zeros((2, sample_num), dtype=float)
        samples[0, :] = self.box[0, 0] + np.random.rand(sample_num) * (self.box[0, 1] - self.box[0, 0])
        samples[1, :] = self.box[1, 0] + np.random.rand(sample_num) * (self.box[1, 1] - self.box[1, 0])

        generators = self.points.T
        nearest = find_closest(2, self.n_points, sample_num, samples, generators)
        energy = 0.0
        for j in range(sample_num):
            energy += np.sum((generators[:, nearest[j]] - samples[:, j]) ** 2)
        energy /= sample_num
        return float(energy)
