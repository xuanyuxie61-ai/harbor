
import numpy as np


def rotation_matrix_2d(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def reflect_vector(v, axis):
    v = np.asarray(v, dtype=np.float64)
    if axis == 'x':
        R = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    elif axis == 'y':
        R = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    else:
        raise ValueError("axis 必须为 'x' 或 'y'")
    return v @ R.T if v.ndim > 1 and v.shape[1] == 2 else R @ v


def torus_distance_function(p, R_major=2.0, R_minor=0.8):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 1:
        x, y, z = p[0], p[1], p[2]
        r_xy = np.sqrt(x * x + y * y)
        return np.sqrt((r_xy - R_major) ** 2 + z * z) - R_minor
    else:
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        r_xy = np.sqrt(x * x + y * y)
        return np.sqrt((r_xy - R_major) ** 2 + z * z) - R_minor


def cylinder_distance_function(p, radius=1.0, center=(0.0, 0.0)):
    p = np.asarray(p, dtype=np.float64)
    cx, cy = center
    if p.ndim == 1:
        return np.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2) - radius
    else:
        return np.sqrt((p[:, 0] - cx) ** 2 + (p[:, 1] - cy) ** 2) - radius


def rectangle_distance_function(p, xmin, xmax, ymin, ymax):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 1:
        dx = max(max(xmin - p[0], 0.0), p[0] - xmax)
        dy = max(max(ymin - p[1], 0.0), p[1] - ymax)
        return np.sqrt(dx * dx + dy * dy)
    else:
        dx = np.maximum(np.maximum(xmin - p[:, 0], 0.0), p[:, 0] - xmax)
        dy = np.maximum(np.maximum(ymin - p[:, 1], 0.0), p[:, 1] - ymax)
        return np.sqrt(dx * dx + dy * dy)


def union_distance(d1, d2):
    return np.minimum(d1, d2)


def intersect_distance(d1, d2):
    return np.maximum(d1, d2)


def diff_distance(d1, d2):
    return np.maximum(d1, -d2)


def compute_reactor_boundary_word(n_segments=64, reactor_type='cylinder', **kwargs):
    if reactor_type == 'cylinder':
        radius = kwargs.get('radius', 1.0)
        theta = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return np.column_stack((x, y))
    elif reactor_type == 'torus':

        R_major = kwargs.get('R_major', 2.0)
        theta = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
        x = R_major * np.cos(theta)
        y = R_major * np.sin(theta)
        return np.column_stack((x, y))
    elif reactor_type == 'rectangle':
        xmin = kwargs.get('xmin', -1.0)
        xmax = kwargs.get('xmax', 1.0)
        ymin = kwargs.get('ymin', -1.0)
        ymax = kwargs.get('ymax', 1.0)

        pts = []
        n_side = n_segments // 4
        pts.extend([[xmin + (xmax - xmin) * i / n_side, ymin] for i in range(n_side)])
        pts.extend([[xmax, ymin + (ymax - ymin) * i / n_side] for i in range(n_side)])
        pts.extend([[xmax - (xmax - xmin) * i / n_side, ymax] for i in range(n_side)])
        pts.extend([[xmin, ymax - (ymax - ymin) * i / n_side] for i in range(n_side)])
        return np.array(pts, dtype=np.float64)
    else:
        raise ValueError(f"不支持的 reactor_type: {reactor_type}")


def circumcenter(p1, p2, p3):
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    d = 2.0 * ((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]))
    if abs(d) < 1e-14:
        return (p1 + p2 + p3) / 3.0
    ux = ((p1[1] - p3[1]) * (np.dot(p2 - p3, p2 - p3)) -
          (p2[1] - p3[1]) * (np.dot(p1 - p3, p1 - p3))) / d
    uy = ((p2[0] - p3[0]) * (np.dot(p1 - p3, p1 - p3)) -
          (p1[0] - p3[0]) * (np.dot(p2 - p3, p2 - p3))) / d
    return np.array([ux + p3[0], uy + p3[1]], dtype=np.float64)
