
import numpy as np
from typing import List, Tuple, Optional


class AsteroidGeometryError(Exception):
    pass


def sierpinski_ifs_transforms(scale: float = 1.0 / 3.0) -> List[Tuple[np.ndarray, np.ndarray]]:
    A = np.array([[scale, 0.0], [0.0, scale]])
    translations = [
        np.array([0.0, 2.0 * scale]),
        np.array([scale, 2.0 * scale]),
        np.array([2.0 * scale, 2.0 * scale]),
        np.array([0.0, scale]),
        np.array([2.0 * scale, scale]),
        np.array([0.0, 0.0]),
        np.array([scale, 0.0]),
        np.array([2.0 * scale, 0.0]),
    ]
    return [(A.copy(), b) for b in translations]


def generate_fractal_profile(
    n_iterations: int = 5000,
    scale: float = 1.0 / 3.0,
    seed: Optional[int] = None
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    transforms = sierpinski_ifs_transforms(scale)
    x = np.random.rand(2)
    points = np.zeros((n_iterations, 2))


    for _ in range(100):
        idx = np.random.randint(0, 8)
        A, b = transforms[idx]
        x = A @ x + b

    for i in range(n_iterations):
        idx = np.random.randint(0, 8)
        A, b = transforms[idx]
        x = A @ x + b
        points[i] = x.copy()

    return points


def polygon_area_2d(vertices: np.ndarray) -> float:
    n = vertices.shape[0]
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1] - vertices[j, 0] * vertices[i, 1]
    return 0.5 * area


def is_collinear(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, eps: float = 1e-10) -> bool:
    area = 0.5 * abs(
        (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1])
    )
    d12 = np.sum((v1 - v2) ** 2)
    d23 = np.sum((v2 - v3) ** 2)
    d31 = np.sum((v3 - v1) ** 2)
    area_max = max(d12, d23, d31)
    if area_max <= eps:
        return True
    return 2.0 * area <= eps * area_max


def is_convex_vertex(poly: np.ndarray, i: int) -> bool:
    n = poly.shape[0]
    im1 = (i - 1) % n
    ip1 = (i + 1) % n
    cross = (poly[i, 0] - poly[im1, 0]) * (poly[ip1, 1] - poly[i, 1]) - \
            (poly[i, 1] - poly[im1, 1]) * (poly[ip1, 0] - poly[i, 0])
    return cross > 0


def point_in_triangle_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(denom) < 1e-14:
        return False
    alpha = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / denom
    beta = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / denom
    gamma = 1.0 - alpha - beta
    return (alpha > 1e-10) and (beta > 1e-10) and (gamma > 1e-10)


def ear_clip_triangulation(poly: np.ndarray) -> np.ndarray:
    n = poly.shape[0]
    if n < 3:
        raise AsteroidGeometryError("多边形至少需要 3 个顶点")
    if n == 3:
        return np.array([[0, 1, 2]], dtype=int)


    if polygon_area_2d(poly) < 0:
        poly = poly[::-1]


    for i in range(n):
        j = (i + 1) % n
        if np.allclose(poly[i], poly[j]):
            raise AsteroidGeometryError(f"顶点 {i} 与 {j} 重合")

    indices = list(range(n))
    triangles = []

    while len(indices) > 3:
        n_cur = len(indices)
        ear_found = False
        for i in range(n_cur):
            im1 = (i - 1) % n_cur
            ip1 = (i + 1) % n_cur
            idx_i = indices[i]
            idx_im1 = indices[im1]
            idx_ip1 = indices[ip1]

            v_im1 = poly[idx_im1]
            v_i = poly[idx_i]
            v_ip1 = poly[idx_ip1]

            if is_collinear(v_im1, v_i, v_ip1):
                continue
            if not is_convex_vertex(poly[np.array(indices)], i):
                continue


            valid = True
            for j in range(n_cur):
                if j in (im1, i, ip1):
                    continue
                if point_in_triangle_2d(poly[indices[j]], v_im1, v_i, v_ip1):
                    valid = False
                    break
            if valid:
                triangles.append([idx_im1, idx_i, idx_ip1])
                del indices[i]
                ear_found = True
                break

        if not ear_found:

            for i in range(n_cur):
                im1 = (i - 1) % n_cur
                ip1 = (i + 1) % n_cur
                if is_collinear(poly[indices[im1]], poly[indices[i]], poly[indices[ip1]]):
                    del indices[i]
                    break
            else:
                raise AsteroidGeometryError("无法完成三角剖分，可能为自相交多边形")

    triangles.append([indices[0], indices[1], indices[2]])
    return np.array(triangles, dtype=int)


def generate_asteroid_cross_section(
    a: float = 2.0,
    b: float = 1.5,
    c: float = 1.0,
    n_points: int = 64,
    roughness_amplitude: float = 0.05,
    roughness_scale: float = 1.0 / 3.0,
    seed: int = 42
) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)

    x = a * np.cos(theta)
    y = b * np.sin(theta)


    fractal = generate_fractal_profile(n_iterations=n_points, scale=roughness_scale, seed=seed)
    np.random.seed(seed)
    raw_noise = 2.0 * fractal[:, 0] - 1.0

    window = 3
    pad = window // 2
    padded = np.concatenate([raw_noise[-pad:], raw_noise, raw_noise[:pad]])
    smooth_noise = np.convolve(padded, np.ones(window) / window, mode='valid')
    radial_noise = roughness_amplitude * smooth_noise

    r_base = np.sqrt(x ** 2 + y ** 2)
    r_perturbed = r_base * (1.0 + radial_noise)

    x_pert = r_perturbed * np.cos(theta)
    y_pert = r_perturbed * np.sin(theta)

    vertices = np.column_stack((x_pert, y_pert))
    return vertices


def revolve_to_3d(poly2d: np.ndarray, z_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    n_lat = poly2d.shape[0]
    n_lon = max(24, n_lat // 2)

    vertices = []
    for j in range(n_lon):
        phi = 2.0 * np.pi * j / n_lon
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        for i in range(n_lat):
            r = np.sqrt(poly2d[i, 0] ** 2 + poly2d[i, 1] ** 2)

            lat_angle = np.arctan2(poly2d[i, 1], poly2d[i, 0])
            x = r * np.cos(lat_angle) * cos_phi
            y = r * np.cos(lat_angle) * sin_phi
            z = r * np.sin(lat_angle) * z_scale
            vertices.append([x, y, z])

    vertices = np.array(vertices)

    faces = []
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        for i in range(n_lat - 1):
            v0 = j * n_lat + i
            v1 = j * n_lat + i + 1
            v2 = j_next * n_lat + i + 1
            v3 = j_next * n_lat + i
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    return vertices, np.array(faces, dtype=int)


def polyhedron_volume_and_com(vertices: np.ndarray, faces: np.ndarray) -> Tuple[float, np.ndarray]:
    vol = 0.0
    com = np.zeros(3)
    for f in faces:
        v0 = vertices[f[0]]
        v1 = vertices[f[1]]
        v2 = vertices[f[2]]

        tet_vol = np.dot(v0, np.cross(v1, v2)) / 6.0
        vol += tet_vol
        com += tet_vol * (v0 + v1 + v2) / 4.0

    if abs(vol) < 1e-14:
        return 0.0, np.zeros(3)
    com /= vol
    return abs(vol), com


def triangle_area_3d(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))


def surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    area = 0.0
    for f in faces:
        area += triangle_area_3d(vertices[f[0]], vertices[f[1]], vertices[f[2]])
    return area
