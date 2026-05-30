
import numpy as np
from scipy.spatial import ConvexHull


def fit_circle_2_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray = None) -> tuple[np.ndarray, float]:
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    if p3 is None:
        center = 0.5 * (p1 + p2)
        radius = 0.5 * np.linalg.norm(p2 - p1)
        return center, radius
    p3 = np.asarray(p3, dtype=float)


    A = 2.0 * np.array([p2 - p1, p3 - p1])
    b = np.array([
        np.dot(p2, p2) - np.dot(p1, p1),
        np.dot(p3, p3) - np.dot(p1, p1)
    ])
    center = np.linalg.solve(A, b)
    radius = np.linalg.norm(center - p1)
    return center, radius


def _is_in_circle(point: np.ndarray, center: np.ndarray, radius: float, tol: float = 1e-10) -> bool:
    return np.linalg.norm(point - center) <= radius + tol


def _welzl_recursive(points: np.ndarray, boundary: list, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    if len(points) == 0 or len(boundary) == 3:
        if len(boundary) == 0:
            return np.zeros(2), 0.0
        elif len(boundary) == 1:
            return boundary[0], 0.0
        elif len(boundary) == 2:
            return fit_circle_2_points(boundary[0], boundary[1])
        else:
            return fit_circle_2_points(boundary[0], boundary[1], boundary[2])

    p = points[-1]
    rest = points[:-1]
    center, radius = _welzl_recursive(rest, boundary, rng)
    if _is_in_circle(p, center, radius):
        return center, radius
    else:
        return _welzl_recursive(rest, boundary + [p], rng)


def minimum_bounding_circle(points: np.ndarray, seed: int = 42) -> tuple[np.ndarray, float]:
    points = np.asarray(points, dtype=float)
    if points.shape[0] == 0:
        return np.zeros(2), 0.0
    if points.shape[0] == 1:
        return points[0], 0.0

    rng = np.random.default_rng(seed)
    shuffled = points[rng.permutation(len(points))]


    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        shuffled = hull_points[rng.permutation(len(hull_points))]
    except Exception:
        pass

    return _welzl_recursive(shuffled, [], rng)


def minimum_bounding_sphere_3d(points: np.ndarray, seed: int = 42) -> tuple[np.ndarray, float]:
    points = np.asarray(points, dtype=float)
    if points.shape[0] == 0:
        return np.zeros(3), 0.0
    if points.shape[0] == 1:
        return points[0], 0.0


    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(points))
    p1 = points[idx]
    dists = np.linalg.norm(points - p1, axis=1)
    p2 = points[np.argmax(dists)]
    dists2 = np.linalg.norm(points - p2, axis=1)
    p3 = points[np.argmax(dists2)]
    center = 0.5 * (p2 + p3)
    radius = 0.5 * np.linalg.norm(p3 - p2)


    for p in points:
        d = np.linalg.norm(p - center)
        if d > radius:
            radius = (radius + d) / 2.0
            center = center + (p - center) * (d - radius) / d if d > 0 else center

    return center, radius


def compute_infected_patch_geometry(
    infected_field: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    threshold: float = 0.01
) -> dict:
    from scipy import ndimage

    mask = infected_field > threshold
    labeled, num_features = ndimage.label(mask)

    results = {
        'num_patches': int(num_features),
        'bounding_circles': [],
        'total_infected_area': 0.0,
    }

    dx = x_coords[1] - x_coords[0]
    dy = y_coords[1] - y_coords[0]
    pixel_area = dx * dy

    for i in range(1, num_features + 1):
        patch_mask = labeled == i
        coords = np.argwhere(patch_mask)
        if len(coords) == 0:
            continue


        phys_coords = np.zeros((len(coords), 2))
        phys_coords[:, 0] = x_coords[coords[:, 0]]
        phys_coords[:, 1] = y_coords[coords[:, 1]]

        center, radius = minimum_bounding_circle(phys_coords)
        area = np.sum(patch_mask) * pixel_area

        results['bounding_circles'].append({
            'center': center,
            'radius': radius,
            'area': area,
        })
        results['total_infected_area'] += area

    return results
