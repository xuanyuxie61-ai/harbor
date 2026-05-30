
import numpy as np
from math import radians, sin, cos, acos, sqrt, pi, degrees


class MeshError(Exception):
    pass


def ll_degrees_to_distance_earth(lat1, lon1, lat2, lon2, radius=6371.0):
    phi1 = radians(lat1)
    lambda1 = radians(lon1)
    phi2 = radians(lat2)
    lambda2 = radians(lon2)

    cos_theta = sin(phi1) * sin(phi2) + cos(phi1) * cos(phi2) * cos(lambda1 - lambda2)

    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = acos(cos_theta)
    return radius * theta


def generate_lat_lon_grid(n_lat, n_lon):
    if n_lat < 2 or n_lon < 2:
        raise MeshError("generate_lat_lon_grid: 网格维度至少为 2")

    lats = np.linspace(-90.0, 90.0, n_lat)
    lons = np.linspace(-180.0, 180.0, n_lon)
    nodes = []
    for lat in lats:
        for lon in lons:
            nodes.append([lat, lon])
    return np.array(nodes, dtype=np.float64)


def compute_distance_table(nodes):
    N = nodes.shape[0]
    dist = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            d = ll_degrees_to_distance_earth(
                nodes[i, 0], nodes[i, 1], nodes[j, 0], nodes[j, 1]
            )
            dist[i, j] = d
            dist[j, i] = d
    return dist


def triangle_angles_2d(p1, p2, p3):
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p3)
    c = np.linalg.norm(p1 - p2)

    if a < 1e-12 or b < 1e-12 or c < 1e-12:
        return 0.0, 0.0, 0.0

    cos_a = max(-1.0, min(1.0, (b ** 2 + c ** 2 - a ** 2) / (2.0 * b * c)))
    cos_b = max(-1.0, min(1.0, (a ** 2 + c ** 2 - b ** 2) / (2.0 * a * c)))
    cos_c = max(-1.0, min(1.0, (a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b)))

    A = degrees(acos(cos_a))
    B = degrees(acos(cos_b))
    C = degrees(acos(cos_c))
    return A, B, C


def delaunay_discrepancy_simple(nodes, triangles):
    num_tri = triangles.shape[0]
    if num_tri == 0:
        return 0.0, 0.0

    angles = []
    for t in range(num_tri):
        i, j, k = triangles[t]
        p1 = nodes[i]
        p2 = nodes[j]
        p3 = nodes[k]
        A, B, C = triangle_angles_2d(p1, p2, p3)
        angles.extend([A, B, C])

    if len(angles) == 0:
        return 0.0, 0.0

    min_angle = min(angles)

    discrepancy = max(0.0, 60.0 - min_angle)
    return discrepancy, min_angle


def define_atmospheric_layers(z_bottom, z_top, num_layers):
    if z_top <= z_bottom or num_layers < 1:
        raise MeshError("define_atmospheric_layers: 参数非法")

    boundaries = np.linspace(z_bottom, z_top, num_layers + 1)
    mid_points = 0.5 * (boundaries[:-1] + boundaries[1:])
    return boundaries, mid_points


def compute_mesh_quality_metrics(nodes, triangles):
    discrepancy, min_angle = delaunay_discrepancy_simple(nodes, triangles)
    num_tri = triangles.shape[0]
    all_angles = []
    max_angle = 0.0

    for t in range(num_tri):
        i, j, k = triangles[t]
        A, B, C = triangle_angles_2d(nodes[i], nodes[j], nodes[k])
        all_angles.extend([A, B, C])
        max_angle = max(max_angle, A, B, C)

    angle_std = float(np.std(all_angles)) if all_angles else 0.0
    return {
        "delaunay_discrepancy": float(discrepancy),
        "min_angle_deg": float(min_angle),
        "max_angle_deg": float(max_angle),
        "angle_std_deg": angle_std,
        "num_triangles": num_tri,
    }


def generate_simple_triangulation(nodes_2d):
    n = int(round(np.sqrt(nodes_2d.shape[0])))
    if n * n != nodes_2d.shape[0]:

        return np.zeros((0, 3), dtype=int)

    triangles = []
    for i in range(n - 1):
        for j in range(n - 1):
            p00 = i * n + j
            p10 = (i + 1) * n + j
            p01 = i * n + (j + 1)
            p11 = (i + 1) * n + (j + 1)
            triangles.append([p00, p10, p11])
            triangles.append([p00, p11, p01])

    return np.array(triangles, dtype=int)
