
import numpy as np


def dsphere(p, xc, yc, zc, r):
    return np.sqrt((p[:, 0] - xc) ** 2 +
                   (p[:, 1] - yc) ** 2 +
                   (p[:, 2] - zc) ** 2) - r


def dbox(p, x_min, x_max, y_min, y_max, z_min, z_max):
    dx = np.maximum(np.maximum(x_min - p[:, 0], p[:, 0] - x_max), 0.0)
    dy = np.maximum(np.maximum(y_min - p[:, 1], p[:, 1] - y_max), 0.0)
    dz = np.maximum(np.maximum(z_min - p[:, 2], p[:, 2] - z_max), 0.0)

    inside_dist = -np.minimum(np.minimum(
        np.minimum(p[:, 0] - x_min, x_max - p[:, 0]),
        np.minimum(p[:, 1] - y_min, y_max - p[:, 1])),
        np.minimum(p[:, 2] - z_min, z_max - p[:, 2]))
    dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    dist[dist < 1e-14] = inside_dist[dist < 1e-14]
    return dist


def ddiff(d1, d2):
    return np.maximum(d1, -d2)


def dintersect(d1, d2):
    return np.maximum(d1, d2)


def huniform(p):
    return np.ones(p.shape[0])


def dshoebox_with_pillars(p):
    room = dbox(p, 0.0, 10.0, 0.0, 8.0, 0.0, 5.0)
    pillar1 = dsphere(p, 3.0, 3.0, 2.5, 0.3)
    pillar2 = dsphere(p, 7.0, 5.0, 2.5, 0.3)
    return ddiff(room, dintersect(pillar1, pillar2))


def extract_room_surfaces():
    surfaces = {}

    surfaces['floor'] = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 8.0, 0.0],
        [0.0, 0.0, 0.0], [10.0, 8.0, 0.0], [0.0, 8.0, 0.0]
    ], dtype=float)

    surfaces['ceiling'] = np.array([
        [0.0, 0.0, 5.0], [10.0, 8.0, 5.0], [10.0, 0.0, 5.0],
        [0.0, 0.0, 5.0], [0.0, 8.0, 5.0], [10.0, 8.0, 5.0]
    ], dtype=float)

    surfaces['front_wall'] = np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 5.0], [10.0, 0.0, 0.0],
        [0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [10.0, 0.0, 5.0]
    ], dtype=float)

    surfaces['back_wall'] = np.array([
        [0.0, 8.0, 0.0], [10.0, 8.0, 0.0], [10.0, 8.0, 5.0],
        [0.0, 8.0, 0.0], [10.0, 8.0, 5.0], [0.0, 8.0, 5.0]
    ], dtype=float)

    surfaces['left_wall'] = np.array([
        [0.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 8.0, 5.0],
        [0.0, 0.0, 0.0], [0.0, 8.0, 5.0], [0.0, 0.0, 5.0]
    ], dtype=float)

    surfaces['right_wall'] = np.array([
        [10.0, 0.0, 0.0], [10.0, 8.0, 5.0], [10.0, 8.0, 0.0],
        [10.0, 0.0, 0.0], [10.0, 0.0, 5.0], [10.0, 8.0, 5.0]
    ], dtype=float)
    return surfaces


def compute_surface_normals(surfaces):
    normals = {}
    for name, tris in surfaces.items():

        v0, v1, v2 = tris[0], tris[1], tris[2]
        n = np.cross(v1 - v0, v2 - v0)
        n_norm = np.linalg.norm(n)
        if n_norm > 1e-14:
            n = n / n_norm
        normals[name] = n
    return normals


def triangle_area(v0, v1, v2):
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))


def room_surface_areas(surfaces):
    areas = {}
    for name, tris in surfaces.items():
        total = 0.0
        for i in range(0, len(tris), 3):
            total += triangle_area(tris[i], tris[i + 1], tris[i + 2])
        areas[name] = total
    return areas


def room_total_volume():
    V_room = 10.0 * 8.0 * 5.0

    V_pillars = 2.0 * (np.pi * 0.3 ** 2 * 5.0)
    return V_room - V_pillars


def compute_sabine_reverberation_time(absorption_coeffs, surfaces):
    areas = room_surface_areas(surfaces)
    V = room_total_volume()
    total_absorption = 0.0
    for name, area in areas.items():
        alpha = absorption_coeffs.get(name, 0.05)
        total_absorption += area * alpha
    if total_absorption < 1e-14:
        total_absorption = 1e-14
    T60 = 0.161 * V / total_absorption
    return T60
