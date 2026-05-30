
import numpy as np






def naca4_symmetric(t, c, x):
    x = np.asarray(x, dtype=float)
    if np.any(x < -1e-9) or np.any(x > c + 1e-9):
        raise ValueError("x must be in [0, c].")
    x = np.clip(x, 0.0, c)
    xi = x / c
    y = 5.0 * t * c * (
        0.2969 * np.sqrt(xi)
        + ((((-0.1015) * xi + 0.2843) * xi - 0.3516) * xi - 0.1260) * xi
    )
    return y


def naca4_cambered(m, p, t, c, x):
    x = np.asarray(x, dtype=float)
    if np.any(x < -1e-9) or np.any(x > c + 1e-9):
        raise ValueError("x must be in [0, c].")
    x = np.clip(x, 0.0, c)
    xi = x / c


    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    mask = xi <= p
    if p > 0.0:
        yc[mask] = (m / p ** 2) * (2.0 * p * xi[mask] - xi[mask] ** 2)
        dyc_dx[mask] = (2.0 * m / p ** 2) * (p - xi[mask])
    if p < 1.0:
        yc[~mask] = (m / (1.0 - p) ** 2) * (
            (1.0 - 2.0 * p) + 2.0 * p * xi[~mask] - xi[~mask] ** 2)
        dyc_dx[~mask] = (2.0 * m / (1.0 - p) ** 2) * (p - xi[~mask])

    yt = naca4_symmetric(t, c, x)

    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    upper = np.column_stack((xu, yu))
    lower = np.column_stack((xl, yl))
    return upper, lower


def generate_naca_airfoil_points(t, c, n_points=100, m=0.0, p=0.0):

    theta = np.linspace(0.0, np.pi, n_points)
    x = 0.5 * c * (1.0 - np.cos(theta))
    x = np.clip(x, 0.0, c)

    if m == 0.0:
        y = naca4_symmetric(t, c, x)
        upper = np.column_stack((x, y))
        lower = np.column_stack((x, -y))
    else:
        upper, lower = naca4_cambered(m, p, t, c, x)


    surface = np.vstack([
        upper[::-1],
        lower[1:, :]
    ])
    return surface






def triangle_area(node_xy):
    node_xy = np.asarray(node_xy, dtype=float)
    if node_xy.shape != (3, 2):
        raise ValueError("node_xy must have shape (3, 2).")
    area = 0.5 * abs(
        node_xy[0, 0] * (node_xy[1, 1] - node_xy[2, 1]) +
        node_xy[1, 0] * (node_xy[2, 1] - node_xy[0, 1]) +
        node_xy[2, 0] * (node_xy[0, 1] - node_xy[1, 1])
    )
    return area


def triangle_angles(node_xy):
    node_xy = np.asarray(node_xy, dtype=float)

    a = np.linalg.norm(node_xy[1] - node_xy[2])
    b = np.linalg.norm(node_xy[0] - node_xy[2])
    c = np.linalg.norm(node_xy[0] - node_xy[1])

    angles = np.zeros(3, dtype=float)
    if a > 0.0 and b > 0.0 and c > 0.0:
        angles[0] = np.arccos(np.clip((b ** 2 + c ** 2 - a ** 2) / (2.0 * b * c), -1.0, 1.0))
        angles[1] = np.arccos(np.clip((a ** 2 + c ** 2 - b ** 2) / (2.0 * a * c), -1.0, 1.0))
        angles[2] = np.arccos(np.clip((a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b), -1.0, 1.0))
    return angles


def triangle_circumcircle(node_xy):
    node_xy = np.asarray(node_xy, dtype=float)
    A = triangle_area(node_xy)
    a = np.linalg.norm(node_xy[1] - node_xy[2])
    b = np.linalg.norm(node_xy[0] - node_xy[2])
    c = np.linalg.norm(node_xy[0] - node_xy[1])
    if A <= 1e-14:
        return np.inf, node_xy[0].copy()
    R = a * b * c / (4.0 * A)


    D = 2.0 * (
        node_xy[0, 0] * (node_xy[1, 1] - node_xy[2, 1]) +
        node_xy[1, 0] * (node_xy[2, 1] - node_xy[0, 1]) +
        node_xy[2, 0] * (node_xy[0, 1] - node_xy[1, 1])
    )
    if abs(D) < 1e-14:
        return np.inf, node_xy[0].copy()

    ux = (
        (node_xy[0, 0] ** 2 + node_xy[0, 1] ** 2) * (node_xy[1, 1] - node_xy[2, 1]) +
        (node_xy[1, 0] ** 2 + node_xy[1, 1] ** 2) * (node_xy[2, 1] - node_xy[0, 1]) +
        (node_xy[2, 0] ** 2 + node_xy[2, 1] ** 2) * (node_xy[0, 1] - node_xy[1, 1])
    ) / D
    uy = (
        (node_xy[0, 0] ** 2 + node_xy[0, 1] ** 2) * (node_xy[2, 0] - node_xy[1, 0]) +
        (node_xy[1, 0] ** 2 + node_xy[1, 1] ** 2) * (node_xy[0, 0] - node_xy[2, 0]) +
        (node_xy[2, 0] ** 2 + node_xy[2, 1] ** 2) * (node_xy[1, 0] - node_xy[0, 0])
    ) / D

    return R, np.array([ux, uy])


def triangle_incircle(node_xy):
    node_xy = np.asarray(node_xy, dtype=float)
    A = triangle_area(node_xy)
    a = np.linalg.norm(node_xy[1] - node_xy[2])
    b = np.linalg.norm(node_xy[0] - node_xy[2])
    c = np.linalg.norm(node_xy[0] - node_xy[1])
    perim = a + b + c
    if perim <= 1e-14:
        return 0.0, np.mean(node_xy, axis=0)
    r = 2.0 * A / perim
    center = (a * node_xy[0] + b * node_xy[1] + c * node_xy[2]) / perim
    return r, center


def triangle_quality(node_xy):
    node_xy = np.asarray(node_xy, dtype=float)
    A = triangle_area(node_xy)
    a = np.linalg.norm(node_xy[1] - node_xy[2])
    b = np.linalg.norm(node_xy[0] - node_xy[2])
    c = np.linalg.norm(node_xy[0] - node_xy[1])
    denom = a ** 2 + b ** 2 + c ** 2
    if denom <= 1e-14:
        return 0.0
    q = 4.0 * np.sqrt(3.0) * A / denom
    return float(np.clip(q, 0.0, 1.0))






def compute_surface_normals(nodes, triangles):
    nodes = np.asarray(nodes, dtype=float)
    triangles = np.asarray(triangles, dtype=int)

    v0 = nodes[triangles[:, 0], :]
    v1 = nodes[triangles[:, 1], :]
    v2 = nodes[triangles[:, 2], :]

    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e1, e2)

    norms = np.linalg.norm(normals, axis=1)
    norms = np.where(norms < 1e-14, 1.0, norms)
    normals = normals / norms[:, np.newaxis]
    return normals


def tri_surface_area(nodes, triangles):
    nodes = np.asarray(nodes, dtype=float)
    triangles = np.asarray(triangles, dtype=int)

    total = 0.0
    for tri in triangles:
        pts = nodes[tri, :2]
        a = np.linalg.norm(nodes[tri[1]] - nodes[tri[0]])
        b = np.linalg.norm(nodes[tri[2]] - nodes[tri[1]])
        c = np.linalg.norm(nodes[tri[0]] - nodes[tri[2]])
        s = 0.5 * (a + b + c)
        area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))
        total += area
    return total


class AcousticBoundary:

    def __init__(self, boundary_type='naca'):
        self.boundary_type = boundary_type
        self.surface_points = None
        self.triangles = None
        self.normals = None

    def generate_naca_boundary(self, t=0.12, c=1.0, n_points=200, m=0.0, p=0.0,
                                angle_of_attack=0.0):
        surf = generate_naca_airfoil_points(t, c, n_points, m, p)
        if angle_of_attack != 0.0:
            theta = np.radians(angle_of_attack)
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
            surf = (rot @ surf.T).T
        self.surface_points = surf
        return surf

    def reflect_acoustic_rays(self, ray_origins, ray_directions):
        if self.surface_points is None:
            raise ValueError("Boundary not initialized.")


        reflected = np.zeros_like(ray_directions)
        for i in range(ray_origins.shape[0]):
            origin = ray_origins[i]
            d = ray_directions[i]

            dists = np.sum((self.surface_points - origin) ** 2, axis=1)
            idx = np.argmin(dists)
            pt = self.surface_points[idx]


            if idx < len(self.surface_points) - 1:
                tangent = self.surface_points[idx + 1] - pt
            else:
                tangent = pt - self.surface_points[idx - 1]
            tangent = tangent / (np.linalg.norm(tangent) + 1e-14)
            normal = np.array([-tangent[1], tangent[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-14)


            if np.dot(d, normal) > 0.0:
                normal = -normal

            reflected[i] = d - 2.0 * np.dot(d, normal) * normal
            reflected[i] = reflected[i] / (np.linalg.norm(reflected[i]) + 1e-14)

        return reflected

    def mesh_quality_statistics(self, tri_nodes):
        qualities = []
        areas = []
        for tri in tri_nodes:
            q = triangle_quality(tri)
            a = triangle_area(tri)
            qualities.append(q)
            areas.append(a)

        qualities = np.array(qualities)
        areas = np.array(areas)

        return {
            'min_quality': float(np.min(qualities)),
            'max_quality': float(np.max(qualities)),
            'mean_quality': float(np.mean(qualities)),
            'min_area': float(np.min(areas)),
            'max_area': float(np.max(areas)),
            'total_area': float(np.sum(areas))
        }
