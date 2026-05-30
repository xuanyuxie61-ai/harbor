
import numpy as np


def point_in_polygon(pt: np.ndarray, poly: np.ndarray) -> bool:
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = poly.shape[0]
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]

        if ((y1 > y) != (y2 > y)):
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1
            if xinters > x:
                inside = not inside
    return inside


def polygon_bounding_box(poly: np.ndarray) -> tuple:
    xmin, ymin = poly.min(axis=0)
    xmax, ymax = poly.max(axis=0)
    return (xmin, xmax, ymin, ymax)


def polygon_area_mc(poly: np.ndarray, n_samples: int = 10000,
                    seed: int = None) -> tuple:
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = polygon_bounding_box(poly)
    box_area = (xmax - xmin) * (ymax - ymin)
    if box_area < 1e-15:
        return 0.0, 0.0
    x = rng.random(n_samples) * (xmax - xmin) + xmin
    y = rng.random(n_samples) * (ymax - ymin) + ymin
    pts = np.column_stack([x, y])
    inside = np.array([point_in_polygon(p, poly) for p in pts])
    p_in = float(np.mean(inside))
    area_est = p_in * box_area
    std_err = box_area * np.sqrt(p_in * (1.0 - p_in) / n_samples)
    return float(area_est), float(std_err)


def human_outline_polygon(scale: float = 1.0, n_points: int = 60) -> np.ndarray:
    pts = []

    n_head = n_points // 5
    n_torso = n_points // 5
    n_legs = n_points // 5
    n_arms = n_points // 5
    n_neck = n_points - n_head - n_torso - n_legs - n_arms


    for theta in np.linspace(np.pi * 0.5, -np.pi * 0.5, n_head):
        x = 0.5 * np.cos(theta)
        y = 2.5 + 0.5 * np.sin(theta)
        pts.append([x, y])


    for t in np.linspace(0.0, 1.0, n_neck):
        x = 0.5 + 0.1 * t
        y = 2.0 - 0.3 * t
        pts.append([x, y])


    for t in np.linspace(0.0, 1.0, n_arms):
        x = 0.6 + 0.1 * np.sin(np.pi * t)
        y = 1.7 - 1.5 * t
        pts.append([x, y])


    for t in np.linspace(0.0, 1.0, n_legs):
        x = 0.4 * (1.0 - t)
        y = 0.2 - 2.0 * t
        pts.append([x, y])


    for t in np.linspace(0.0, 1.0, n_legs):
        x = -0.4 * t
        y = -1.8 + 2.0 * t
        pts.append([x, y])


    for t in np.linspace(0.0, 1.0, n_arms):
        x = -0.6 - 0.1 * np.sin(np.pi * t)
        y = 0.2 + 1.5 * t
        pts.append([x, y])


    for t in np.linspace(0.0, 1.0, n_neck):
        x = -0.6 + 0.1 * t
        y = 1.7 + 0.3 * t
        pts.append([x, y])

    poly = np.array(pts, dtype=float) * scale
    return poly


def sample_in_polygon(poly: np.ndarray, n: int, seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = polygon_bounding_box(poly)
    samples = []
    max_attempts = n * 100
    attempts = 0
    while len(samples) < n and attempts < max_attempts:
        x = rng.random() * (xmax - xmin) + xmin
        y = rng.random() * (ymax - ymin) + ymin
        pt = np.array([x, y])
        if point_in_polygon(pt, poly):
            samples.append(pt)
        attempts += 1
    if len(samples) < n:

        return grid_sample_in_polygon(poly, n)
    return np.array(samples)


def grid_sample_in_polygon(poly: np.ndarray, n: int) -> np.ndarray:
    xmin, xmax, ymin, ymax = polygon_bounding_box(poly)

    area_est, _ = polygon_area_mc(poly, n_samples=1000)
    side = np.sqrt(area_est / max(n, 1))
    if side < 1e-10:
        side = 0.01
    nx = max(3, int(np.ceil((xmax - xmin) / side)))
    ny = max(3, int(np.ceil((ymax - ymin) / side)))
    xgrid = np.linspace(xmin, xmax, nx)
    ygrid = np.linspace(ymin, ymax, ny)
    Xg, Yg = np.meshgrid(xgrid, ygrid)
    pts = np.column_stack([Xg.ravel(), Yg.ravel()])
    inside = np.array([point_in_polygon(p, poly) for p in pts])
    valid = pts[inside]
    if valid.shape[0] < n:

        repeat = (n + valid.shape[0] - 1) // valid.shape[0]
        valid = np.tile(valid, (repeat, 1))[:n]
    return valid[:n]


def velocity_field_in_complex_domain(poly: np.ndarray, a: float, d: float,
                                     t_val: float, n_samples: int = 500,
                                     seed: int = None) -> tuple:
    pts_2d = sample_in_polygon(poly, n_samples, seed)
    x = pts_2d[:, 0]
    y = pts_2d[:, 1]
    z = np.zeros_like(x)
    t = np.full_like(x, t_val)
    from navier_stokes_exact import uvwp_ethier
    u, v, w, p = uvwp_ethier(a, d, x, y, z, t)
    coords = np.column_stack([x, y, z])
    velocity = np.column_stack([u, v, w])
    return coords, velocity, p
