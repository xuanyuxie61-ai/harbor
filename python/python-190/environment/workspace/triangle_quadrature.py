
import numpy as np


def triangle_unit_monomial_integral(expon: np.ndarray) -> float:
    m = int(expon[0])
    n = int(expon[1])
    if m < 0 or n < 0:
        raise ValueError("指数必须为非负整数。")
    value = 1.0
    k = 0
    for _ in range(m):
        k += 1
        value = value * 1.0 / k
    for _ in range(n):
        k += 1
        value = value * 1.0 / k
    k += 1
    value = value / k
    k += 1
    value = value / k
    return float(value)


def triangle_area(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    v3 = np.asarray(v3, dtype=float)
    if v1.shape[0] == 2:
        cross = (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v2[1] - v1[1]) * (v3[0] - v1[0])
        return 0.5 * abs(cross)
    else:
        cross = np.cross(v2 - v1, v3 - v1)
        return 0.5 * np.linalg.norm(cross)


def triangle_symq_rule(degree: int = 7) -> tuple:
    if degree <= 1:

        bary = np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
        weights = np.array([1.0])
    elif degree == 2:

        bary = np.array([
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ])
        weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    elif degree == 3:

        bary = np.array([
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
        ])
        weights = np.array([-9.0 / 16.0, 25.0 / 48.0, 25.0 / 48.0, 25.0 / 48.0])
    elif degree == 4 or degree == 5:


        a1 = 0.816847572980459
        a2 = 0.091576213509771
        b1 = 0.108103018168070
        b2 = 0.445948490915965
        bary = np.array([
            [a1, a2, a2],
            [a2, a1, a2],
            [a2, a2, a1],
            [b1, b2, b2],
            [b2, b1, b2],
            [b2, b2, b1],
        ])
        w1 = 0.109951743655322
        w2 = 0.223381589678011
        weights = np.array([w1, w1, w1, w2, w2, w2])
    else:

        a1 = 0.797426985353087
        a2 = 0.101286507323456
        b1 = 0.059715871789770
        b2 = 0.470142064105115
        c1 = 1.0 / 3.0
        bary = np.array([
            [a1, a2, a2],
            [a2, a1, a2],
            [a2, a2, a1],
            [b1, b2, b2],
            [b2, b1, b2],
            [b2, b2, b1],
            [c1, c1, c1],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        w1 = 0.062969590272413
        w2 = 0.066197076394253
        w3 = -0.074785022233841
        w4 = 0.0625
        w5 = 0.0625
        weights = np.array([w1, w1, w1, w2, w2, w2, w3, w4, w4, w4, w5, w5, w5])

    weights = weights / np.sum(weights)
    return bary, weights


def integrate_over_triangle(f, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray,
                            degree: int = 5) -> float:
    bary, w = triangle_symq_rule(degree)

    pts = (bary[:, 0:1] * v1.reshape(1, -1)
           + bary[:, 1:2] * v2.reshape(1, -1)
           + bary[:, 2:3] * v3.reshape(1, -1))
    vals = f(pts)
    area = triangle_area(v1, v2, v3)
    return float(area * np.dot(w, vals))


def sphere01_triangle_vertices_to_area(v1: np.ndarray, v2: np.ndarray,
                                       v3: np.ndarray) -> float:
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    v3 = np.asarray(v3, dtype=float)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)


    a = np.arccos(np.clip(np.dot(v2, v3), -1.0, 1.0))
    b = np.arccos(np.clip(np.dot(v3, v1), -1.0, 1.0))
    c = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    s = 0.5 * (a + b + c)

    if s > np.pi - 1e-12:
        return 2.0 * np.pi


    tan_s2 = np.tan(s * 0.5)
    tan_as = np.tan(max(0.0, (s - a) * 0.5))
    tan_bs = np.tan(max(0.0, (s - b) * 0.5))
    tan_cs = np.tan(max(0.0, (s - c) * 0.5))


    prod = tan_s2 * tan_as * tan_bs * tan_cs
    prod = max(prod, 0.0)
    E = 4.0 * np.arctan(np.sqrt(prod))
    return float(E)


def sphere01_triangle_sample(n: int, v1: np.ndarray, v2: np.ndarray,
                             v3: np.ndarray, seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)


    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    v3 = np.asarray(v3, dtype=float)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)


    r1 = rng.random(n)
    r2 = rng.random(n)
    mask = r1 + r2 > 1.0
    r1[mask] = 1.0 - r1[mask]
    r2[mask] = 1.0 - r2[mask]

    pts_local = (r1[None, :] * v2[:, None]
                 + r2[None, :] * v3[:, None]
                 + (1.0 - r1 - r2)[None, :] * v1[:, None])

    norms = np.sqrt(np.sum(pts_local ** 2, axis=0))
    norms = np.where(norms < 1e-15, 1.0, norms)
    pts = pts_local / norms[None, :]
    return pts


def sphere01_triangle_quad_00(n: int, v1: np.ndarray, v2: np.ndarray,
                              v3: np.ndarray, f, seed: int = None) -> float:
    area = sphere01_triangle_vertices_to_area(v1, v2, v3)
    pts = sphere01_triangle_sample(n, v1, v2, v3, seed)
    quad = 0.0
    for j in range(n):
        quad += f(pts[:, j])
    return quad * area / n


def integrate_pde_residual_over_mesh(residual_func, triangles: list,
                                     degree: int = 5) -> float:
    total = 0.0
    for v1, v2, v3 in triangles:
        total += integrate_over_triangle(residual_func, v1, v2, v3, degree)
    return float(total)
