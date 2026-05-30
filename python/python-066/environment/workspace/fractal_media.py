
import numpy as np
from typing import List, Tuple


def hexagon_vertices(center: Tuple[float, float] = (0.0, 0.0),
                      radius: float = 1.0) -> np.ndarray:
    if radius <= 0:
        raise ValueError("半径必须为正")
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    verts = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])
    return verts


def sample_uniform_hexagon(n_samples: int, center: Tuple[float, float] = (0.0, 0.0),
                           radius: float = 1.0, seed: int = 42) -> np.ndarray:
    if n_samples < 1:
        raise ValueError("样本数必须 ≥ 1")
    rng = np.random.default_rng(seed)
    verts = hexagon_vertices(center, radius)

    c = np.array(center)

    samples = np.zeros((n_samples, 2))
    for i in range(n_samples):

        k = rng.integers(0, 6)
        v0 = c
        v1 = verts[k]
        v2 = verts[(k + 1) % 6]

        u1 = rng.random()
        u2 = rng.random()
        if u1 + u2 > 1.0:
            u1 = 1.0 - u1
            u2 = 1.0 - u2
        samples[i] = v0 + u1 * (v1 - v0) + u2 * (v2 - v0)
    return samples


def hexagon_grid(nx: int, ny: int, radius: float = 1.0) -> Tuple[np.ndarray, List[List[int]]]:
    if nx < 1 or ny < 1:
        raise ValueError("nx, ny 必须 ≥ 1")
    dy = radius * np.sqrt(3)
    nodes = []
    node_id = {}
    for j in range(ny):
        y = j * dy
        x_offset = 0.0 if j % 2 == 0 else 1.5 * radius
        for i in range(nx):
            x = x_offset + i * 3.0 * radius
            nid = len(nodes)
            node_id[(i, j)] = nid
            nodes.append([x, y])

    coords = np.array(nodes)


    elements = []
    for j in range(ny - 1):
        for i in range(nx - 1):

            n0 = node_id.get((i, j))
            n1 = node_id.get((i + 1, j))
            n2 = node_id.get((i, j + 1))
            n3 = node_id.get((i + 1, j + 1))
            if None not in (n0, n1, n2, n3):

                elements.append([n0, n1, n3])
                elements.append([n0, n3, n2])

    return coords, elements


class FractalPorousMedia:

    def __init__(self, n_iterations: int = 4, n_points: int = 10000,
                 seed: int = 42):
        self.n_iterations = n_iterations
        self.n_points = n_points
        self.rng = np.random.default_rng(seed)

    def generate_sierpinski_carpet_permeability(self, grid_res: int = 64) -> np.ndarray:
        if grid_res < 3:
            raise ValueError("网格分辨率必须 ≥ 3")
        K = np.ones((grid_res, grid_res), dtype=float)
        K_min = 1e-4
        K_max = 10.0

        def remove_center(arr, level):
            if level == 0:
                return
            m, n = arr.shape
            if m < 3 or n < 3:
                return

            cm, cn = m // 3, n // 3
            arr[cm:2 * cm, cn:2 * cn] = K_max

            for i in range(3):
                for j in range(3):
                    if i == 1 and j == 1:
                        continue
                    sub = arr[i * cm:(i + 1) * cm, j * cn:(j + 1) * cn]
                    remove_center(sub, level - 1)

        remove_center(K, self.n_iterations)
        K[K == 1.0] = K_min
        return K

    def generate_ifs_attractor(self, n_points: int = 5000) -> np.ndarray:

        transforms = [
            {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.16, "e": 0.0, "f": 0.0, "prob": 0.01},
            {"a": 0.85, "b": 0.04, "c": -0.04, "d": 0.85, "e": 0.0, "f": 1.6, "prob": 0.85},
            {"a": 0.2, "b": -0.26, "c": 0.23, "d": 0.22, "e": 0.0, "f": 1.6, "prob": 0.07},
            {"a": -0.15, "b": 0.28, "c": 0.26, "d": 0.24, "e": 0.0, "f": 0.44, "prob": 0.07},
        ]
        probs = np.cumsum([t["prob"] for t in transforms])

        points = np.zeros((n_points, 2))
        p = np.array([0.0, 0.0])
        for i in range(n_points):
            r = self.rng.random()
            idx = np.searchsorted(probs, r)
            t = transforms[idx]
            x = t["a"] * p[0] + t["b"] * p[1] + t["e"]
            y = t["c"] * p[0] + t["d"] * p[1] + t["f"]
            p = np.array([x, y])
            points[i] = p
        return points

    def fractal_dimension_boxcount(self, points: np.ndarray, n_boxes: int = 20) -> float:
        if len(points) == 0:
            raise ValueError("点云不能为空")
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        L = max(xmax - xmin, ymax - ymin)
        if L <= 0:
            return 0.0

        counts = []
        epsilons = []
        for k in range(1, n_boxes + 1):
            eps = L / k

            ix = np.floor((points[:, 0] - xmin) / eps).astype(int)
            iy = np.floor((points[:, 1] - ymin) / eps).astype(int)
            boxes = set(zip(ix, iy))
            counts.append(len(boxes))
            epsilons.append(eps)

        counts = np.array(counts, dtype=float)
        epsilons = np.array(epsilons, dtype=float)
        valid = counts > 0
        if valid.sum() < 2:
            return 0.0
        logN = np.log(counts[valid])
        logE = np.log(1.0 / epsilons[valid])

        A = np.vstack([logE, np.ones(len(logE))]).T
        D_f, _ = np.linalg.lstsq(A, logN, rcond=None)[0]
        return float(max(0.0, D_f))


if __name__ == "__main__":
    samples = sample_uniform_hexagon(1000, radius=2.0)
    assert samples.shape == (1000, 2)

    dists = np.linalg.norm(samples, axis=1)
    assert np.all(dists <= 2.0 + 1e-6)

    fpm = FractalPorousMedia(n_iterations=3)
    K_field = fpm.generate_sierpinski_carpet_permeability(grid_res=81)
    assert K_field.shape == (81, 81)

    pts = fpm.generate_ifs_attractor(2000)
    D_est = fpm.fractal_dimension_boxcount(pts)
    assert D_est > 0.5
    print("fractal_media: 自测试通过")
