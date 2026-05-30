
import numpy as np


def r8vec_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        return v
    return v / norm


def sphere01_distance_xyz(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = r8vec_normalize(v1)
    v2 = r8vec_normalize(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.arccos(dot))


def sphere01_triangle_vertices_to_angles(v1: np.ndarray, v2: np.ndarray,
                                         v3: np.ndarray) -> tuple:
    v1 = r8vec_normalize(v1)
    v2 = r8vec_normalize(v2)
    v3 = r8vec_normalize(v3)
    a = sphere01_distance_xyz(v2, v3)
    b = sphere01_distance_xyz(v3, v1)
    c = sphere01_distance_xyz(v1, v2)
    return a, b, c


def sphere01_triangle_vertices_to_area(v1: np.ndarray, v2: np.ndarray,
                                       v3: np.ndarray) -> float:
    a, b, c = sphere01_triangle_vertices_to_angles(v1, v2, v3)
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


def sphere01_triangle_vertices_to_centroid(v1: np.ndarray, v2: np.ndarray,
                                           v3: np.ndarray) -> np.ndarray:
    v1 = r8vec_normalize(v1)
    v2 = r8vec_normalize(v2)
    v3 = r8vec_normalize(v3)
    c = v1 + v2 + v3
    return r8vec_normalize(c)


def sphere01_triangle_vertices_to_midpoints(v1: np.ndarray, v2: np.ndarray,
                                            v3: np.ndarray) -> tuple:
    v1 = r8vec_normalize(v1)
    v2 = r8vec_normalize(v2)
    v3 = r8vec_normalize(v3)
    m12 = r8vec_normalize(v1 + v2)
    m23 = r8vec_normalize(v2 + v3)
    m31 = r8vec_normalize(v3 + v1)
    return m12, m23, m31


def sphere01_triangle_quad_03(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray,
                              f) -> float:
    area = sphere01_triangle_vertices_to_area(v1, v2, v3)
    m12, m23, m31 = sphere01_triangle_vertices_to_midpoints(v1, v2, v3)
    val = f(m12) + f(m23) + f(m31)
    return area * val / 3.0


def sphere01_triangle_quad_07(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray,
                              f) -> float:
    area = sphere01_triangle_vertices_to_area(v1, v2, v3)
    m12, m23, m31 = sphere01_triangle_vertices_to_midpoints(v1, v2, v3)
    c = sphere01_triangle_vertices_to_centroid(v1, v2, v3)
    w_v = area / 20.0
    w_m = area / 20.0
    w_c = 9.0 * area / 20.0
    result = (w_v * (f(v1) + f(v2) + f(v3))
              + w_m * (f(m12) + f(m23) + f(m31))
              + w_c * f(c))
    return float(result)


def icosahedron_faces() -> list:
    phi = 0.5 * (1.0 + np.sqrt(5.0))
    vertices = np.array([
        [0.0, 1.0, phi],
        [0.0, 1.0, -phi],
        [0.0, -1.0, phi],
        [0.0, -1.0, -phi],
        [1.0, phi, 0.0],
        [1.0, -phi, 0.0],
        [-1.0, phi, 0.0],
        [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0],
        [phi, 0.0, -1.0],
        [-phi, 0.0, 1.0],
        [-phi, 0.0, -1.0],
    ], dtype=float)

    vertices = vertices / np.linalg.norm(vertices[0])

    faces = [
        (0, 2, 8), (0, 8, 4), (0, 4, 6), (0, 6, 10), (0, 10, 2),
        (3, 1, 11), (3, 11, 7), (3, 7, 5), (3, 5, 9), (3, 9, 1),
        (1, 4, 6), (1, 6, 11), (1, 9, 4), (11, 6, 10), (11, 10, 7),
        (7, 10, 2), (7, 2, 5), (5, 2, 8), (5, 8, 9), (9, 8, 4),
    ]
    return faces, vertices


def integrate_on_sphere(f, rule: str = "icos1v") -> float:
    faces, vertices = icosahedron_faces()
    total = 0.0
    for tri in faces:
        v1 = vertices[tri[0]]
        v2 = vertices[tri[1]]
        v3 = vertices[tri[2]]
        if rule == "icos1v":
            total += sphere01_triangle_quad_03(v1, v2, v3, f)
        else:
            total += sphere01_triangle_quad_07(v1, v2, v3, f)
    return float(total)
