
import numpy as np
from typing import Tuple, List


def line_grid(n: int, a: float, b: float, c: int = 1) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be >= 1")
    if a >= b:
        raise ValueError("must have a < b")
    if c < 1 or c > 5:
        raise ValueError("c must be in [1,5]")

    x = np.zeros(n)
    if c == 1:
        if n == 1:
            x[0] = 0.5 * (a + b)
        else:
            for j in range(n):
                x[j] = ((n - 1 - j) * a + j * b) / (n - 1)
    elif c == 2:
        for j in range(n):
            x[j] = ((n - j) * a + (j + 1) * b) / (n + 1)
    elif c == 3:
        for j in range(n):
            x[j] = ((n - j) * a + j * b) / n
    elif c == 4:
        for j in range(n):
            x[j] = ((n - 1 - j) * a + (j + 1) * b) / n
    elif c == 5:
        for j in range(n):
            x[j] = ((2 * n - 2 * j - 1) * a + (2 * j + 1) * b) / (2 * n)
    return x


def chebyshev_grid(n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be >= 1")
    j = np.arange(n + 1)
    return np.cos(np.pi * j / n)


def triangulate_polygon(vertices: np.ndarray, hmax: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    nv = vertices.shape[0]
    if nv < 3:
        raise ValueError("polygon must have at least 3 vertices")


    area = 0.0
    for i in range(nv):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % nv]
        area += x1 * y2 - x2 * y1
    if area <= 0:
        raise ValueError("vertices must be ordered counter-clockwise and enclose positive area")


    triangles = []
    for i in range(1, nv - 1):
        triangles.append([0, i, i + 1])

    return vertices.copy(), np.array(triangles, dtype=int)


def diaphony_compute(points: np.ndarray) -> float:
    if points.ndim != 2:
        raise ValueError("points must be 2D array")
    n, dim = points.shape
    if n < 2:
        return 0.0


    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    rng = pmax - pmin
    rng[rng == 0] = 1.0
    pts = (points - pmin) / rng


    C = (1.0 + np.pi ** 2 / 3.0) ** dim - 1.0
    if abs(C) < 1e-14:
        return 0.0

    total = 0.0
    for i in range(n):
        for j in range(n):
            z = np.mod(pts[i] - pts[j], 1.0)

            kernel = -1.0 + np.prod(1.0 + 2.0 * np.pi ** 2 * (z ** 2 - z + 1.0 / 6.0))
            total += kernel

    diaphony_val = np.sqrt(total / (n * n * C))
    return diaphony_val


def sample_ellipse(a: float, b: float, n: int) -> np.ndarray:
    if a <= 0 or b <= 0 or n < 1:
        raise ValueError("invalid ellipse parameters")
    points = []
    max_iter = n * 100
    count = 0
    while len(points) < n and count < max_iter:
        x = np.random.uniform(-a, a)
        y = np.random.uniform(-b, b)
        if (x / a) ** 2 + (y / b) ** 2 <= 1.0:
            points.append([x, y])
        count += 1
    if len(points) < n:

        nx = int(np.ceil(np.sqrt(n * a / b))) + 1
        ny = int(np.ceil(np.sqrt(n * b / a))) + 1
        xs = np.linspace(-a, a, nx)
        ys = np.linspace(-b, b, ny)
        for x in xs:
            for y in ys:
                if (x / a) ** 2 + (y / b) ** 2 <= 1.0:
                    points.append([x, y])
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
    return np.array(points[:n])


def compute_polygon_area(vertices: np.ndarray) -> float:
    nv = vertices.shape[0]
    area = 0.0
    for i in range(nv):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % nv]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def refine_cross_section_mesh(vertices: np.ndarray, max_area: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    nodes, tris = triangulate_polygon(vertices)

    cx = np.mean(nodes[:, 0])
    cy = np.mean(nodes[:, 1])
    center_idx = nodes.shape[0]
    nodes = np.vstack([nodes, [cx, cy]])


    new_tris = []
    for tri in tris:
        i, j, k = tri

        new_tris.append([i, j, center_idx])
        new_tris.append([j, k, center_idx])
        new_tris.append([k, i, center_idx])

    return nodes, np.array(new_tris, dtype=int)
