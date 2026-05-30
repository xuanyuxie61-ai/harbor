
import numpy as np


def polygon_area_2d(vertices):
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("vertices must be of shape (N, 2)")
    n = vertices.shape[0]
    if n < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    area = 0.5 * np.sum(x * (np.roll(y, -1) - np.roll(y, 1)))
    return area


def polygon_contains_point_2d(vertices, point):
    vertices = np.asarray(vertices, dtype=float)
    point = np.asarray(point, dtype=float)
    n = vertices.shape[0]
    inside = False
    x, y = point[0], point[1]
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]

        if ((y1 > y) != (y2 > y)):

            xinters = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if xinters > x:
                inside = not inside
    return inside


def quadrilateral_area(quad):
    quad = np.asarray(quad, dtype=float)
    if quad.shape != (4, 2):
        raise ValueError("quad must be of shape (4, 2)")

    t1 = 0.5 * abs(
        quad[0, 0] * (quad[1, 1] - quad[2, 1]) +
        quad[1, 0] * (quad[2, 1] - quad[0, 1]) +
        quad[2, 0] * (quad[0, 1] - quad[1, 1])
    )

    t2 = 0.5 * abs(
        quad[0, 0] * (quad[2, 1] - quad[3, 1]) +
        quad[2, 0] * (quad[3, 1] - quad[0, 1]) +
        quad[3, 0] * (quad[0, 1] - quad[2, 1])
    )
    return t1 + t2


def quadrilateral_is_convex(quad):
    quad = np.asarray(quad, dtype=float)
    angles = quadrilateral_angles(quad)
    angle_sum = np.sum(angles)
    return (
        np.all(angles > 0.0) and
        np.all(angles < np.pi) and
        abs(angle_sum - 2.0 * np.pi) < 1.0
    )


def quadrilateral_angles(quad):
    quad = np.asarray(quad, dtype=float)
    angles = np.zeros(4)
    for i in range(4):
        p1 = quad[(i - 1) % 4]
        p2 = quad[i]
        p3 = quad[(i + 1) % 4]
        v1 = p1 - p2
        v2 = p3 - p2

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-14 or norm2 < 1e-14:
            angles[i] = 0.0
            continue
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles[i] = np.arccos(cos_angle)
    return angles


def quadrilateral_contains_point(quad, point):
    quad = np.asarray(quad, dtype=float)
    point = np.asarray(point, dtype=float)

    if not quadrilateral_is_convex(quad):
        return polygon_contains_point_2d(quad, point)
    for i in range(4):
        p1 = quad[i]
        p2 = quad[(i + 1) % 4]
        p3 = quad[(i + 2) % 4]

        v1 = p1 - p2
        v2 = p3 - p2
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-14 or norm2 < 1e-14:
            return False
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_quad = np.arccos(cos_angle)

        w1 = p1 - p2
        w2 = point - p2
        norm_w1 = np.linalg.norm(w1)
        norm_w2 = np.linalg.norm(w2)
        if norm_w1 < 1e-14 or norm_w2 < 1e-14:
            return False
        cos_p = np.dot(w1, w2) / (norm_w1 * norm_w2)
        cos_p = np.clip(cos_p, -1.0, 1.0)
        angle_p = np.arccos(cos_p)
        if angle_quad < angle_p:
            return False
    return True


def area_estimate_mc(boundary, width, height, sample_num, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if sample_num <= 0:
        return 0.0
    x = width * rng.random(sample_num)
    y = height * rng.random(sample_num)
    inside_count = 0
    for i in range(sample_num):
        if polygon_contains_point_2d(boundary, np.array([x[i], y[i]])):
            inside_count += 1
    return inside_count / sample_num


def area_estimate_grid(boundary, width, height, n_grid):
    if n_grid <= 0:
        return 0.0
    dx = width / (n_grid + 1)
    dy = height / (n_grid + 1)
    xlo = 0.5 * dx
    xhi = width - 0.5 * dx
    ylo = 0.5 * dy
    yhi = height - 0.5 * dy
    if n_grid == 1:
        gx = np.array([0.5 * (xlo + xhi)])
        gy = np.array([0.5 * (ylo + yhi)])
    else:
        gx = np.linspace(xlo, xhi, n_grid)
        gy = np.linspace(ylo, yhi, n_grid)
    XG, YG = np.meshgrid(gx, gy)
    inside = np.zeros_like(XG, dtype=bool)
    for i in range(n_grid):
        for j in range(n_grid):
            inside[j, i] = polygon_contains_point_2d(
                boundary, np.array([XG[j, i], YG[j, i]])
            )
    return np.sum(inside) / (n_grid * n_grid)


def hammersley_sequence(i1, i2, m, n_base):
    primes = np.array([
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
        31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
        127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
        179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
        233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
        283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
        353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
        419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
        467, 479, 487, 491, 499, 503, 509, 521, 523, 541
    ], dtype=int)
    if n_base <= 0:
        n_base = 1
    step = 1 if i1 <= i2 else -1
    l = abs(i2 - i1) + 1
    r = np.zeros((m, l))
    k_idx = 0
    for i in range(i1, i2 + step, step):
        r[0, k_idx] = (i % (n_base + 1)) / n_base
        t = np.full(m - 1, i, dtype=int)
        prime_inv = 1.0 / primes[:m - 1].astype(float)
        while np.any(t != 0):
            for j in range(m - 1):
                d = int(t[j] % primes[j])
                r[j + 1, k_idx] += d * prime_inv[j]
                prime_inv[j] /= primes[j]
                t[j] = t[j] // primes[j]
        k_idx += 1
    return r


def area_estimate_qmc(boundary, width, height, sample_num):
    if sample_num <= 0:
        return 0.0
    seq = hammersley_sequence(0, sample_num - 1, 2, sample_num - 1)
    x = width * seq[0, :]
    y = height * seq[1, :]
    inside_count = 0
    for i in range(sample_num):
        if polygon_contains_point_2d(boundary, np.array([x[i], y[i]])):
            inside_count += 1
    return inside_count / sample_num
