
import numpy as np
from math import factorial


def cartesian_to_spherical(xyz):
    xyz = np.atleast_2d(xyz)
    r = np.linalg.norm(xyz, axis=1)
    r_safe = np.where(r < 1e-15, 1.0, r)
    theta = np.arccos(np.clip(xyz[:, 2] / r_safe, -1.0, 1.0))
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    phi = np.where(phi < 0, phi + 2.0 * np.pi, phi)
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.column_stack([x, y, z])


def great_circle_distance(r, theta1, phi1, theta2, phi2):
    delta_phi = phi1 - phi2
    cos_d = (np.sin(theta1) * np.sin(theta2) * np.cos(delta_phi)
             + np.cos(theta1) * np.cos(theta2))
    cos_d = np.clip(cos_d, -1.0, 1.0)

    if np.isscalar(cos_d):
        if cos_d > 0.999999999:
            sin_half = np.sin(0.5 * np.arccos(cos_d))
            return 2.0 * r * sin_half
        return r * np.arccos(cos_d)
    else:
        close_mask = cos_d > 0.999999999
        d = np.zeros_like(cos_d)
        d[close_mask] = r * np.sqrt(2.0 * (1.0 - cos_d[close_mask]))
        d[~close_mask] = r * np.arccos(cos_d[~close_mask])
        return d


def spherical_cap_area(r, alpha):
    alpha = np.clip(alpha, 0.0, np.pi)
    return 2.0 * np.pi * r * r * (1.0 - np.cos(alpha))


def solid_angle(r, cap_area):
    if r < 1e-15:
        raise ValueError("半径必须为正数")
    return cap_area / (r * r)


def quadrilateral_area_2d(quad):
    quad = np.asarray(quad)
    if quad.shape != (4, 2):
        raise ValueError("quad必须是(4,2)数组")

    tri1 = quad[[0, 1, 2]]
    area1 = 0.5 * abs(
        tri1[0, 0] * (tri1[1, 1] - tri1[2, 1])
        + tri1[1, 0] * (tri1[2, 1] - tri1[0, 1])
        + tri1[2, 0] * (tri1[0, 1] - tri1[1, 1])
    )

    tri2 = quad[[0, 2, 3]]
    area2 = 0.5 * abs(
        tri2[0, 0] * (tri2[1, 1] - tri2[2, 1])
        + tri2[1, 0] * (tri2[2, 1] - tri2[0, 1])
        + tri2[2, 0] * (tri2[0, 1] - tri2[1, 1])
    )
    return area1 + area2


def legendre_associated_normalized(n, m, x):
    if m < 0:
        raise ValueError("m必须非负")
    if n < m:
        raise ValueError("n必须大于等于m")
    x = float(x)
    if x < -1.0 or x > 1.0:
        raise ValueError("x必须在[-1,1]范围内")

    cx = np.zeros(n + 1)
    cx[:m] = 0.0
    cx[m] = 1.0
    somx2 = np.sqrt(max(0.0, 1.0 - x * x))

    fact = 1.0
    for i in range(1, m + 1):
        cx[m] = -cx[m] * fact * somx2
        fact = fact + 2.0

    if m < n:
        cx[m + 1] = x * (2 * m + 1) * cx[m]

    for i in range(m + 2, n + 1):
        cx[i] = ((2 * i - 1) * x * cx[i - 1] + (-i - m + 1) * cx[i - 2]) / (i - m)


    for mm in range(m, n + 1):
        factor = np.sqrt(
            ((2 * mm + 1) * factorial(mm - m))
            / (4.0 * np.pi * factorial(mm + m))
        )
        cx[mm] = cx[mm] * factor

    return cx


def spherical_harmonic_basis(l_max, theta, phi):
    cos_theta = np.cos(theta)
    c_all = []
    s_all = []
    for m in range(l_max + 1):
        plm = legendre_associated_normalized(l_max, m, cos_theta)
        c = plm * np.cos(m * phi)
        s = plm * np.sin(m * phi)
        c_all.append(c)
        s_all.append(s)
    return c_all, s_all


def uniform_sphere_sample(n):
    if n <= 0:
        raise ValueError("n必须为正整数")
    p = np.random.normal(size=(n, 3))
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return p / norms


def circle_points_on_plane(center, radius, normal, num_points=32):
    center = np.asarray(center, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal = normal / (np.linalg.norm(normal) + 1e-15)


    if abs(normal[2]) < 0.9:
        arbitrary = np.array([0.0, 0.0, 1.0])
    else:
        arbitrary = np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, arbitrary)
    u = u / (np.linalg.norm(u) + 1e-15)
    v = np.cross(normal, u)
    v = v / (np.linalg.norm(v) + 1e-15)

    t = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    pts = center[:, None] + radius * (u[:, None] * np.cos(t)[None, :] + v[:, None] * np.sin(t)[None, :])
    return pts.T


def compute_bounding_sphere(points):
    points = np.asarray(points)
    if points.size == 0:
        raise ValueError("点集不能为空")
    center = np.mean(points, axis=0)
    radius = np.max(np.linalg.norm(points - center, axis=1))

    if radius < 1e-15:
        radius = 1e-10
    return center, radius
