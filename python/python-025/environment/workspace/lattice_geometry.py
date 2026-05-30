
import numpy as np


def triangle_area(p1, p2, p3):
    p1, p2, p3 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float), np.asarray(p3, dtype=float)
    return 0.5 * (p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))


def barycentric_interpolate(query_points, p1, p2, p3, v1, v2, v3):
    query_points = np.asarray(query_points, dtype=float)
    if query_points.ndim == 1:
        query_points = query_points.reshape(1, -1)
    
    A = triangle_area(p1, p2, p3)
    if abs(A) < 1e-14:
        raise ValueError("Degenerate triangle: area is near zero")
    
    v1, v2, v3 = np.asarray(v1), np.asarray(v2), np.asarray(v3)
    results = []
    for p in query_points:
        A1 = triangle_area(p, p2, p3)
        A2 = triangle_area(p1, p, p3)
        A3 = triangle_area(p1, p2, p)
        w1, w2, w3 = A1 / A, A2 / A, A3 / A
        results.append(w1 * v1 + w2 * v2 + w3 * v3)
    return np.array(results)


def uniform_in_triangle(v1, v2, v3, n):
    v1, v2, v3 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float), np.asarray(v3, dtype=float)
    r1 = np.random.rand(n)
    r2 = np.random.rand(n)
    sqrt_r2 = np.sqrt(r2)
    a = 1.0 - sqrt_r2
    b = (1.0 - r1) * sqrt_r2
    c = r1 * sqrt_r2
    return (a[:, None] * v1 + b[:, None] * v2 + c[:, None] * v3)


def signed_point_line_distance(p1, p2, p):
    p1, p2, p = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float), np.asarray(p, dtype=float)
    d = p2 - p1
    norm_d = np.linalg.norm(d)
    if norm_d < 1e-14:
        return 0.0
    n = np.array([-d[1], d[0]]) / norm_d
    return float(np.dot(n, p - p1))


def build_hexagonal_lattice(n_rows, n_cols, a):
    positions = []
    for i in range(n_rows):
        for j in range(n_cols):
            x = j * a
            y = i * a * np.sqrt(3.0) / 2.0
            if i % 2 == 1:
                x += 0.5 * a
            positions.append([x, y, 0.0])
    return np.array(positions, dtype=float)


def build_square_lattice(n_rows, n_cols, a):
    positions = []
    for i in range(n_rows):
        for j in range(n_cols):
            positions.append([j * a, i * a, 0.0])
    return np.array(positions, dtype=float)


def build_simple_cubic_lattice(nx, ny, nz, a):
    positions = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                positions.append([i * a, j * a, k * a])
    return np.array(positions, dtype=float)
