# -*- coding: utf-8 -*-

import numpy as np


def spherical_distance(lat1, lon1, lat2, lon2, radius):
    if radius <= 0:
        raise ValueError("半径必须为正。")

    arg = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2)
    arg = np.clip(arg, -1.0, 1.0)
    theta = np.arccos(arg)
    dist = radius * theta
    return dist


def spherical_to_cartesian(lat, lon, radius):
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    if r < 1e-30:
        return 0.0, 0.0, 0.0
    lat = np.arcsin(np.clip(z / r, -1.0, 1.0))
    lon = np.arctan2(y, x)
    return lat, lon, r


def stereographic_projection_sphere_to_plane(p):
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        if len(p) != 3:
            raise ValueError("p 必须是 3 维向量。")
        pz = p[2]
        if 1.0 + pz < 1e-15:

            s = 1e15
        else:
            s = 2.0 / (1.0 + pz)
        S = np.array([0.0, 0.0, -1.0])
        q = s * p + (1.0 - s) * S
        return q[:2]
    else:
        if p.shape[0] != 3:
            raise ValueError("p 的第一维必须为 3。")
        pz = p[2, :]
        s = np.where(1.0 + pz < 1e-15, 1e15, 2.0 / (1.0 + pz))
        q = np.zeros_like(p)
        for i in range(3):
            q[i, :] = s * p[i, :] + (1.0 - s) * (-1.0 if i == 2 else 0.0)
        return q[:2, :]


def stereographic_projection_plane_to_sphere(q):
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        u, v = q[0], q[1]
        r2 = u**2 + v**2
        denom = 4.0 + r2
        px = 4.0 * u / denom
        py = 4.0 * v / denom
        pz = (4.0 - r2) / denom
        return np.array([px, py, pz])
    else:
        u = q[0, :]
        v = q[1, :]
        r2 = u**2 + v**2
        denom = 4.0 + r2
        p = np.zeros((3, q.shape[1]))
        p[0, :] = 4.0 * u / denom
        p[1, :] = 4.0 * v / denom
        p[2, :] = (4.0 - r2) / denom
        return p


def quadrilateral_area_3d(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (3, 4):
        raise ValueError("q 必须是形状 (3, 4) 的数组。")


    p = np.zeros((3, 4))
    p[:, 0] = (q[:, 0] + q[:, 1]) / 2.0
    p[:, 1] = (q[:, 1] + q[:, 2]) / 2.0
    p[:, 2] = (q[:, 2] + q[:, 3]) / 2.0
    p[:, 3] = (q[:, 3] + q[:, 0]) / 2.0


    v1 = p[:, 1] - p[:, 0]
    v2 = p[:, 3] - p[:, 0]
    cross = np.cross(v1, v2)
    para_area = np.linalg.norm(cross)
    area = 2.0 * para_area
    return area


def icf_target_surface_mesh(R0, n_theta, n_phi):
    if R0 <= 0 or n_theta < 2 or n_phi < 2:
        raise ValueError("R0 必须为正，n_theta 和 n_phi 必须 >= 2。")

    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)

    vertices = np.zeros((n_theta, n_phi, 3))
    for i in range(n_theta):
        for j in range(n_phi):
            vertices[i, j, :] = spherical_to_cartesian(
                np.pi / 2.0 - theta[i], phi[j], R0
            )

    face_areas = np.zeros((n_theta - 1, n_phi - 1))
    for i in range(n_theta - 1):
        for j in range(n_phi - 1):
            q = np.zeros((3, 4))
            q[:, 0] = vertices[i, j, :]
            q[:, 1] = vertices[i + 1, j, :]
            q[:, 2] = vertices[i + 1, j + 1, :]
            q[:, 3] = vertices[i, j + 1, :]
            face_areas[i, j] = quadrilateral_area_3d(q)

    total_area = np.sum(face_areas)
    return vertices, face_areas, total_area


def solid_angle_subtended_by_quad(q, observer):
    q = np.asarray(q, dtype=float)
    observer = np.asarray(observer, dtype=float)


    tri_indices = [(0, 1, 2), (0, 2, 3)]
    omega_total = 0.0

    for (i1, i2, i3) in tri_indices:
        r1 = q[:, i1] - observer
        r2 = q[:, i2] - observer
        r3 = q[:, i3] - observer

        n1 = np.linalg.norm(r1)
        n2 = np.linalg.norm(r2)
        n3 = np.linalg.norm(r3)
        if n1 < 1e-20 or n2 < 1e-20 or n3 < 1e-20:
            continue

        r1u = r1 / n1
        r2u = r2 / n2
        r3u = r3 / n3

        dot12 = np.dot(r1u, r2u)
        dot13 = np.dot(r1u, r3u)
        dot23 = np.dot(r2u, r3u)

        cross12 = np.cross(r1u, r2u)
        triple = abs(np.dot(cross12, r3u))

        denom = 1.0 + dot12 + dot13 + dot23
        if denom < 1e-15:
            continue
        omega_tri = 2.0 * np.arctan2(triple, denom)
        omega_total += omega_tri

    return omega_total


def laser_incidence_angle_on_sphere(laser_dir, surface_normal):
    d = np.asarray(laser_dir, dtype=float)
    n = np.asarray(surface_normal, dtype=float)
    dn = np.linalg.norm(d)
    nn = np.linalg.norm(n)
    if dn < 1e-20 or nn < 1e-20:
        return 0.0, 1.0
    cos_val = abs(np.dot(d / dn, n / nn))
    cos_val = np.clip(cos_val, 0.0, 1.0)
    theta = np.arccos(cos_val)
    return theta, cos_val
