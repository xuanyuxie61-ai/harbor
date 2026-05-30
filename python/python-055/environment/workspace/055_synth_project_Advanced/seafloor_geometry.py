
import numpy as np


def triangle_area(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=np.float64)
    if t.shape != (3, 2):
        raise ValueError("t 必须是形状为 (3, 2) 的数组")

    area = 0.5 * (
        (t[1, 0] - t[0, 0]) * (t[2, 1] - t[0, 1])
        - (t[2, 0] - t[0, 0]) * (t[1, 1] - t[0, 1])
    )
    return float(area)


def triangle_unit_monomial_integral(i: int, j: int) -> float:
    if i < 0 or j < 0:
        raise ValueError("指数 i, j 必须为非负整数")


    k = 0
    q = 1.0
    for l in range(1, i + 1):
        k += 1
        q *= l / k
    for l in range(1, j + 1):
        k += 1
        q *= l / k
    for _ in range(2):
        k += 1
        q /= k
    return float(q)


def triangle_monomial_integral(t: np.ndarray, i: int, j: int) -> float:
    area = triangle_area(t)
    jac = 2.0 * abs(area)


    if i == 0 and j == 0:
        return float(abs(area))



    if i + j <= 4:
        return _triangle_monomial_integral_analytic(t, i, j)
    else:
        return _triangle_monomial_integral_numerical(t, i, j)


def _triangle_monomial_integral_analytic(t: np.ndarray, i: int, j: int) -> float:
    x1, y1 = t[0]
    x2, y2 = t[1]
    x3, y3 = t[2]
    area = abs(triangle_area(t))




    bary = np.array([
        [2.0/3.0, 1.0/6.0, 1.0/6.0],
        [1.0/6.0, 2.0/3.0, 1.0/6.0],
        [1.0/6.0, 1.0/6.0, 2.0/3.0],
    ])
    weights = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])

    integral = 0.0
    for w, b in zip(weights, bary):
        x = b[0]*x1 + b[1]*x2 + b[2]*x3
        y = b[0]*y1 + b[1]*y2 + b[2]*y3
        integral += w * (x**i) * (y**j)

    return float(integral * 2.0 * area)


def _triangle_monomial_integral_numerical(t: np.ndarray, i: int, j: int) -> float:
    x1, y1 = t[0]
    x2, y2 = t[1]
    x3, y3 = t[2]
    area = abs(triangle_area(t))


    bary = np.array([
        [1.0/3.0, 1.0/3.0, 1.0/3.0],
        [0.05971587, 0.47014206, 0.47014206],
        [0.47014206, 0.05971587, 0.47014206],
        [0.47014206, 0.47014206, 0.05971587],
        [0.79742699, 0.10128651, 0.10128651],
        [0.10128651, 0.79742699, 0.10128651],
        [0.10128651, 0.10128651, 0.79742699],
    ])
    weights = np.array([
        0.22500000,
        0.13239415,
        0.13239415,
        0.13239415,
        0.12593918,
        0.12593918,
        0.12593918,
    ])

    integral = 0.0
    for w, b in zip(weights, bary):
        x = b[0]*x1 + b[1]*x2 + b[2]*x3
        y = b[0]*y1 + b[1]*y2 + b[2]*y3
        integral += w * (x**i) * (y**j)

    return float(integral * 2.0 * area)


def triangle_centroid(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    return np.mean(t, axis=0)


def triangle_normal_2d(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    ab = t[1] - t[0]
    ac = t[2] - t[0]

    cross_z = ab[0]*ac[1] - ab[1]*ac[0]
    norm_cross = abs(cross_z)
    if norm_cross < 1e-15:
        return np.array([0.0, 0.0])

    n = np.array([ab[1], -ab[0]])
    n = n / np.linalg.norm(n)
    return n


def point_to_triangle_distance(t: np.ndarray, p: np.ndarray) -> float:
    t = np.asarray(t, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    a, b, c = t[0], t[1], t[2]

    def line_param(pt1, pt2, pt):
        v1 = pt2 - pt1
        v2 = pt - pt1
        nv = np.array([v1[1], -v1[0]])
        norm_nv = np.linalg.norm(nv)
        if norm_nv < 1e-15:
            return np.linalg.norm(v2)
        return float((nv @ v2) / norm_nv)

    dab = line_param(a, b, p)
    dbc = line_param(b, c, p)
    dca = line_param(c, a, p)


    if dab <= 0.0 and dbc <= 0.0 and dca <= 0.0:
        return 0.0


    if dab >= 0.0 and dbc <= 0.0 and dca <= 0.0:
        return dab
    if dab <= 0.0 and dbc >= 0.0 and dca <= 0.0:
        return dbc
    if dab <= 0.0 and dbc <= 0.0 and dca >= 0.0:
        return dca


    if dab <= 0.0:
        return float(np.linalg.norm(c - p))
    elif dbc <= 0.0:
        return float(np.linalg.norm(a - p))
    elif dca <= 0.0:
        return float(np.linalg.norm(b - p))
    else:

        return float(min(np.linalg.norm(a - p),
                         np.linalg.norm(b - p),
                         np.linalg.norm(c - p)))


def ray_triangle_intersection_2d(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    t: np.ndarray,
    max_dist: float = 1e6
) -> tuple:
    ray_origin = np.asarray(ray_origin, dtype=np.float64)
    ray_dir = np.asarray(ray_dir, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)


    dir_norm = np.linalg.norm(ray_dir)
    if dir_norm < 1e-15:
        return False, float('inf'), None
    ray_dir = ray_dir / dir_norm


    edges = [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]
    best_dist = float('inf')
    best_point = None

    for v1, v2 in edges:




        d = v2 - v1
        det = ray_dir[0] * (-d[1]) - ray_dir[1] * (-d[0])
        if abs(det) < 1e-15:
            continue
        rhs = v1 - ray_origin
        tt = (rhs[0] * (-d[1]) - rhs[1] * (-d[0])) / det
        ss = (ray_dir[0] * rhs[1] - ray_dir[1] * rhs[0]) / det
        if tt >= 0.0 and 0.0 <= ss <= 1.0 and tt < best_dist and tt <= max_dist:
            best_dist = tt
            best_point = ray_origin + tt * ray_dir

    if best_point is not None:
        return True, best_dist, best_point
    return False, float('inf'), None


class SeafloorTriangulation:

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int64)
        self._validate()
        self._compute_patch_properties()

    def _validate(self):
        nv = self.vertices.shape[0]
        nt = self.triangles.shape[0]
        if nt == 0:
            return
        if self.triangles.min() < 0 or self.triangles.max() >= nv:
            raise ValueError("三角形顶点索引越界")

    def _compute_patch_properties(self):
        nt = self.triangles.shape[0]
        self.patch_areas = np.zeros(nt, dtype=np.float64)
        self.patch_centroids = np.zeros((nt, 2), dtype=np.float64)
        for i in range(nt):
            tri = self.vertices[self.triangles[i]]
            self.patch_areas[i] = abs(triangle_area(tri))
            self.patch_centroids[i] = triangle_centroid(tri)

    def total_area(self) -> float:
        return float(np.sum(self.patch_areas))

    def mean_patch_area(self) -> float:
        nt = self.triangles.shape[0]
        if nt == 0:
            return 0.0
        return float(np.mean(self.patch_areas))

    def find_containing_triangle(self, p: np.ndarray) -> int:
        p = np.asarray(p, dtype=np.float64)
        for i in range(self.triangles.shape[0]):
            tri = self.vertices[self.triangles[i]]
            a, b, c = tri[0], tri[1], tri[2]
            denom = (b[1] - c[1])*(a[0] - c[0]) + (c[0] - b[0])*(a[1] - c[1])
            if abs(denom) < 1e-15:
                continue
            alpha = ((b[1] - c[1])*(p[0] - c[0]) + (c[0] - b[0])*(p[1] - c[1])) / denom
            beta  = ((c[1] - a[1])*(p[0] - c[0]) + (a[0] - c[0])*(p[1] - c[1])) / denom
            gamma = 1.0 - alpha - beta
            if alpha >= -1e-12 and beta >= -1e-12 and gamma >= -1e-12:
                return i
        return -1

    def integrate_scalar_field(self, field_values: np.ndarray, power: int = 0) -> float:
        if len(field_values) != self.vertices.shape[0]:
            raise ValueError("场值数组长度必须与顶点数一致")

        integral = 0.0
        for i in range(self.triangles.shape[0]):
            tri = self.triangles[i]

            avg_val = np.mean(field_values[tri])
            integral += avg_val * self.patch_areas[i]
        return float(integral)
