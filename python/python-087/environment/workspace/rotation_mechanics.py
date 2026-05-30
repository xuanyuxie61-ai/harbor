
import numpy as np
from typing import Tuple, List, Callable


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).flatten()
    if v.shape[0] != 3:
        raise ValueError("必须为三维向量")
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]], dtype=np.float64)


def so3_exp(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64).flatten()
    if theta.shape[0] != 3:
        raise ValueError("旋转向量必须为三维")
    angle = np.linalg.norm(theta)
    if angle < 1e-14:
        return np.eye(3)
    K = skew_symmetric(theta / angle)
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    return R


def so3_log(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError("R 必须为 3×3 矩阵")
    trace = np.trace(R)
    cos_theta = 0.5 * (trace - 1.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-14:
        return np.zeros(3)

    vec = np.array([R[2, 1] - R[1, 2],
                    R[0, 2] - R[2, 0],
                    R[1, 0] - R[0, 1]])
    if np.sin(theta) < 1e-14:


        diag = np.diag(R)
        i = np.argmax(diag)
        e = np.eye(3)[i]
        axis = e / np.linalg.norm(e)
        return theta * axis
    return (0.5 * theta / np.sin(theta)) * vec


def tangent_map_so3(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64).flatten()
    angle = np.linalg.norm(theta)
    K = skew_symmetric(theta)
    if angle < 1e-14:
        return np.eye(3)
    T = np.eye(3) + ((1.0 - np.cos(angle)) / (angle ** 2)) * K \
        + ((angle - np.sin(angle)) / (angle ** 3)) * (K @ K)
    return T


def icosahedron_vertices() -> np.ndarray:
    phi = 0.5 * (1.0 + np.sqrt(5.0))
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    return verts


def icosahedron_faces() -> np.ndarray:
    return np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]
    ], dtype=np.int32)


def sphere_triangle_area(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:

    v0 = v0 / (np.linalg.norm(v0) + 1e-18)
    v1 = v1 / (np.linalg.norm(v1) + 1e-18)
    v2 = v2 / (np.linalg.norm(v2) + 1e-18)

    a = np.arccos(np.clip(v1 @ v2, -1.0, 1.0))
    b = np.arccos(np.clip(v0 @ v2, -1.0, 1.0))
    c = np.arccos(np.clip(v0 @ v1, -1.0, 1.0))

    s = 0.5 * (a + b + c)



    if s >= np.pi - 1e-12:
        return 0.0
    t = np.tan(s / 2.0) * np.tan((s - a) / 2.0) * np.tan((s - b) / 2.0) * np.tan((s - c) / 2.0)
    t = max(t, 0.0)
    E = 4.0 * np.arctan(np.sqrt(t))
    return float(E)


def subdivide_icosahedron(level: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    verts = icosahedron_vertices()
    faces = icosahedron_faces()
    for _ in range(level):
        new_faces = []

        edge_map = {}
        def mid_vertex(i, j):
            key = tuple(sorted((i, j)))
            if key not in edge_map:
                m = verts[i] + verts[j]
                m /= np.linalg.norm(m)
                edge_map[key] = len(verts)
                verts = np.vstack([verts, m])
            return edge_map[key]

        for f in faces:
            v0, v1, v2 = f

            a = len(verts) + len(edge_map)
            b = a + 1
            c = a + 2


        break

    verts_list = [verts]
    faces_list = [faces]
    for _ in range(level):
        v_curr = verts_list[-1]
        f_curr = faces_list[-1]
        edge_map = {}
        new_v = [v for v in v_curr]
        new_f = []
        for f in f_curr:
            i0, i1, i2 = f
            key01 = tuple(sorted((i0, i1)))
            key12 = tuple(sorted((i1, i2)))
            key20 = tuple(sorted((i2, i0)))
            def get_mid(key):
                if key not in edge_map:
                    idx = len(new_v)
                    p = new_v[key[0]] + new_v[key[1]]
                    norm = np.linalg.norm(p)
                    if norm > 1e-14:
                        p /= norm
                    new_v.append(p)
                    edge_map[key] = idx
                return edge_map[key]
            m01 = get_mid(key01)
            m12 = get_mid(key12)
            m20 = get_mid(key20)
            new_f.append([i0, m01, m20])
            new_f.append([i1, m12, m01])
            new_f.append([i2, m20, m12])
            new_f.append([m01, m12, m20])
        verts_list.append(np.array(new_v))
        faces_list.append(np.array(new_f))
    return verts_list[-1], faces_list[-1]


def sphere_quadrature_rule(level: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    verts, faces = subdivide_icosahedron(level)
    points = []
    weights = []
    for f in faces:
        v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]

        centroid = (v0 + v1 + v2) / 3.0
        centroid /= np.linalg.norm(centroid) + 1e-18
        area = sphere_triangle_area(v0, v1, v2)
        points.append(centroid)
        weights.append(area)
    points = np.array(points)
    weights = np.array(weights)

    weights *= 4.0 * np.pi / weights.sum()
    return points, weights


def integrate_over_sphere(integrand: Callable[[np.ndarray], np.ndarray],
                          level: int = 2) -> float:
    pts, wts = sphere_quadrature_rule(level)
    vals = integrand(pts)
    return float(np.sum(wts * vals))


def stroud_en_r2_05_1d(func: Callable[[np.ndarray], np.ndarray],
                        dim: int = 3) -> float:


    n = dim
    a_sq = 0.5 * n + 1.0
    a = np.sqrt(a_sq)
    b_sq = a_sq / n
    b = np.sqrt(b_sq)
    w0 = 2.0 / ((n + 2.0) ** 2)
    w1 = (4.0 - n) / (2.0 * a_sq ** 2)
    w2 = n ** 2 / (2.0 ** n * b_sq ** 2)
    total = 0.0

    total += w0 * func(np.zeros(n))

    for i in range(n):
        e = np.zeros(n)
        e[i] = a
        total += w1 * func(e)
        e[i] = -a
        total += w1 * func(e)

    if n <= 6:
        from itertools import product
        for signs in product([-1, 1], repeat=n):
            pt = np.array(signs) * b
            total += w2 * func(pt)
    else:

        n_samples = 100
        for _ in range(n_samples):
            signs = np.random.choice([-1, 1], size=n)
            pt = signs * b
            total += w2 * func(pt) * (2.0 ** n / n_samples)


    return float(total)


def rotation_averaged_stiffness(K_local: np.ndarray,
                                n_orientations: int = 100) -> np.ndarray:

    indices = np.arange(0, n_orientations, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0)) * indices
    y = 1.0 - 2.0 * indices / (n_orientations - 1)
    if n_orientations > 1:
        y[0], y[-1] = 1.0, -1.0
    radius = np.sqrt(1.0 - y ** 2)
    x = radius * np.cos(phi)
    z = radius * np.sin(phi)
    dirs = np.column_stack((x, y, z))
    K_sum = np.zeros_like(K_local)
    for n in dirs:


        z_axis = np.array([0.0, 0.0, 1.0])
        if np.linalg.norm(n - z_axis) < 1e-12:
            R = np.eye(3)
        elif np.linalg.norm(n + z_axis) < 1e-12:
            R = np.diag([1.0, -1.0, -1.0])
        else:
            v = np.cross(z_axis, n)
            s = np.linalg.norm(v)
            c = z_axis @ n
            vx = skew_symmetric(v)
            R = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s ** 2))



        K_sum += K_local
    return K_sum / n_orientations
