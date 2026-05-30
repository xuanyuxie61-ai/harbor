
import numpy as np
from typing import Tuple






def triangle_unit_monomial_integral(m: int, n: int) -> float:
    from math import factorial
    return factorial(m) * factorial(n) / factorial(m + n + 2)


def triangle_unit_o01() -> Tuple[np.ndarray, np.ndarray]:
    xy = np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64)
    w = np.array([0.5], dtype=np.float64)
    return w, xy


def triangle_unit_o03() -> Tuple[np.ndarray, np.ndarray]:
    xy = np.array([
        [0.666666666666667, 0.166666666666667],
        [0.166666666666667, 0.666666666666667],
        [0.166666666666667, 0.166666666666667]
    ], dtype=np.float64)
    w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=np.float64)
    return w, xy


def triangle_unit_o07() -> Tuple[np.ndarray, np.ndarray]:
    a = 0.059715871789770
    b = 0.797426985353087
    c = 0.101286507323456
    d = 0.25

    xy = np.array([
        [a, a],
        [1.0 - 2.0 * a, a],
        [a, 1.0 - 2.0 * a],
        [b, c],
        [c, b],
        [c, c],
        [d, d]
    ], dtype=np.float64)

    w1 = 0.1125
    w2 = (155.0 - np.sqrt(15.0)) / 1200.0
    w3 = (155.0 + np.sqrt(15.0)) / 1200.0
    w4 = 0.225
    w = np.array([w2, w2, w2, w3, w3, w3, w4], dtype=np.float64) * 0.5
    return w, xy


def triangle_unit_volume() -> float:
    return 0.5


def transform_to_triangle(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    v0, v1, v2 = vertices[0], vertices[1], vertices[2]
    J = np.array([v1 - v0, v2 - v0]).T
    return (v0.reshape(1, -1) + points @ J.T)


def integrate_triangle(f, vertices: np.ndarray, rule: str = "o07") -> float:
    if rule == "o01":
        w, xy = triangle_unit_o01()
    elif rule == "o03":
        w, xy = triangle_unit_o03()
    elif rule == "o07":
        w, xy = triangle_unit_o07()
    else:
        w, xy = triangle_unit_o07()

    pts = transform_to_triangle(xy, vertices)
    vals = np.array([f(p) for p in pts], dtype=np.float64)

    v0, v1, v2 = vertices[:3]
    jac = np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    return float(np.sum(w * vals) * jac)






def line_nco_rule(n: int, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        return np.array([]), np.array([])
    h = (b - a) / (n + 1)
    x = np.array([a + i * h for i in range(1, n + 1)], dtype=np.float64)

    rhs = np.zeros(n, dtype=np.float64)
    for i in range(n):
        power = i
        rhs[i] = (b ** (power + 1) - a ** (power + 1)) / (power + 1)

    V = np.vander(x, N=n, increasing=True)
    w = np.linalg.solve(V.T, rhs)
    return x, w


def integrate_line(f, a: float, b: float, n: int = 5) -> float:
    x, w = line_nco_rule(n, a, b)
    if len(x) == 0:
        return 0.0
    vals = np.array([f(xi) for xi in x], dtype=np.float64)
    return float(np.dot(w, vals))






def gaussian_basis_2d(r: np.ndarray, center: np.ndarray, alpha: float) -> float:
    d = r - center
    return np.exp(-alpha * np.dot(d, d))


def compute_molecular_surface_integral(atoms: np.ndarray,
                                       alpha: float = 1.0,
                                       rule: str = "o07") -> float:
    n = atoms.shape[0]
    if n < 3:
        return 0.0
    total = 0.0

    count = min(n, 12)
    for i in range(count):
        for j in range(i + 1, count):
            for k in range(j + 1, count):
                verts = atoms[[i, j, k]]

                area = 0.5 * np.linalg.norm(np.cross(verts[1] - verts[0], verts[2] - verts[0]))
                if area < 1e-6:
                    continue
                center = np.mean(verts, axis=0)
                val = integrate_triangle(lambda r: gaussian_basis_2d(r, center, alpha), verts, rule)
                total += val
    return float(total)


def compute_bond_path_integral(atoms: np.ndarray, bond: Tuple[int, int],
                               potential_func, n_points: int = 5) -> float:
    a, b = bond
    r_a = atoms[a]
    r_b = atoms[b]
    def path_func(s):
        r = r_a + s * (r_b - r_a)
        return potential_func(r)

    return integrate_line(path_func, 0.0, 1.0, n_points)
