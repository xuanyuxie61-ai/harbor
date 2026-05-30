
import numpy as np
from typing import Tuple, List, Optional






def generate_cvt_points(n_generators: int, domain_bounds: Tuple[float, float, float, float],
                        n_samples: int = 5000, n_lloyd: int = 20,
                        seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    xmin, xmax, ymin, ymax = domain_bounds


    generators = np.column_stack([
        np.random.uniform(xmin, xmax, n_generators),
        np.random.uniform(ymin, ymax, n_generators)
    ])

    for _ in range(n_lloyd):

        samples = np.column_stack([
            np.random.uniform(xmin, xmax, n_samples),
            np.random.uniform(ymin, ymax, n_samples)
        ])




        dx = samples[:, 0:1] - generators[:, 0].reshape(1, -1)
        dy = samples[:, 1:2] - generators[:, 1].reshape(1, -1)
        dists = dx**2 + dy**2
        nearest = np.argmin(dists, axis=1)


        new_gens = np.zeros_like(generators)
        counts = np.zeros(n_generators, dtype=np.int32)
        for i in range(n_generators):
            mask = nearest == i
            if np.any(mask):
                new_gens[i] = np.mean(samples[mask], axis=0)
                counts[i] = np.sum(mask)
            else:

                new_gens[i] = np.array([
                    np.random.uniform(xmin, xmax),
                    np.random.uniform(ymin, ymax)
                ])

        generators = new_gens

    return generators






def is_inside_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    n_points = points.shape[0]
    inside = np.zeros(n_points, dtype=bool)
    n_vert = polygon.shape[0]

    for i in range(n_points):
        x, y = points[i]
        crossings = 0
        for j in range(n_vert):
            x1, y1 = polygon[j]
            x2, y2 = polygon[(j + 1) % n_vert]

            if ((y1 > y) != (y2 > y)):
                x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1 + 1e-14)
                if x_intersect > x:
                    crossings += 1
        inside[i] = (crossings % 2 == 1)
    return inside


def clip_points_to_boundary(points: np.ndarray,
                             boundary_polygon: np.ndarray) -> np.ndarray:
    mask = is_inside_polygon(points, boundary_polygon)
    return points[mask]


def generate_hand_like_boundary(lx: float = 10.0, ly: float = 6.0,
                                 n_points: int = 40) -> np.ndarray:
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)


    a = 0.45 * min(lx, ly)
    b = 0.15 * min(lx, ly)
    c = 0.08 * min(lx, ly)
    r = a + b * np.cos(3*t) + c * np.sin(5*t)
    x = 0.5 * lx + r * np.cos(t)
    y = 0.5 * ly + r * np.sin(t) * 0.7
    return np.column_stack([x, y])






def generate_circle_lattice(centers: np.ndarray, radii: np.ndarray,
                             n_segments: int = 16) -> Tuple[np.ndarray, List[np.ndarray]]:
    n_cells = len(centers)
    all_points_list = []
    cell_rings = []
    offset = 0
    theta = np.linspace(0, 2*np.pi, n_segments, endpoint=False)

    for i in range(n_cells):
        cx, cy = centers[i]
        r = radii[i]

        px = cx + r * np.cos(theta)
        py = cy + r * np.sin(theta)
        pts = np.column_stack([px, py])
        all_points_list.append(pts)
        cell_rings.append(np.arange(offset, offset + n_segments))
        offset += n_segments

    if len(all_points_list) == 0:
        return np.zeros((0, 2)), []
    all_points = np.vstack(all_points_list)
    return all_points, cell_rings


def compute_lattice_relative_density(centers: np.ndarray, radii: np.ndarray,
                                      domain_area: float) -> float:
    total_area = np.sum(np.pi * radii**2)
    return total_area / domain_area






def boundary_word_encode(path_points: np.ndarray,
                          step_letters: str = "ABCDEFGHIJKL") -> str:
    n = len(path_points)
    if n < 2:
        return ""
    word = []
    for i in range(n - 1):
        dx = path_points[i+1, 0] - path_points[i, 0]
        dy = path_points[i+1, 1] - path_points[i, 1]
        angle = np.arctan2(dy, dx)

        sector = int(np.round(angle / (np.pi / 6.0))) % 12
        word.append(step_letters[sector])
    return "".join(word)


def decode_boundary_word(word: str, start_point: np.ndarray,
                          step_length: float = 1.0,
                          step_letters: str = "ABCDEFGHIJKL") -> np.ndarray:
    n = len(word)
    points = np.zeros((n + 1, 2), dtype=np.float64)
    points[0] = start_point
    for i, ch in enumerate(word):
        idx = step_letters.index(ch)
        angle = idx * np.pi / 6.0
        points[i+1, 0] = points[i, 0] + step_length * np.cos(angle)
        points[i+1, 1] = points[i, 1] + step_length * np.sin(angle)
    return points


def generate_print_path_word(centers: np.ndarray, radii: np.ndarray,
                              n_segments: int = 16) -> List[str]:
    words = []
    theta = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
    for i in range(len(centers)):
        cx, cy = centers[i]
        r = radii[i]
        px = cx + r * np.cos(theta)
        py = cy + r * np.sin(theta)
        pts = np.column_stack([px, py])

        pts_closed = np.vstack([pts, pts[0:1]])
        word = boundary_word_encode(pts_closed)
        words.append(word)
    return words






def generate_am_lattice(domain_bounds: Tuple[float, float, float, float],
                        n_cells: int = 50, target_density: float = 0.3,
                        use_complex_boundary: bool = False,
                        seed: Optional[int] = None) -> dict:
    xmin, xmax, ymin, ymax = domain_bounds
    domain_area = (xmax - xmin) * (ymax - ymin)


    centers = generate_cvt_points(n_cells, domain_bounds, n_samples=5000,
                                   n_lloyd=15, seed=seed)

    boundary_polygon = None
    if use_complex_boundary:
        boundary_polygon = generate_hand_like_boundary(xmax - xmin, ymax - ymin)

        boundary_polygon += np.array([xmin, ymin])
        centers = clip_points_to_boundary(centers, boundary_polygon)

        if len(centers) < max(3, n_cells // 4):
            centers = generate_cvt_points(n_cells * 2, domain_bounds,
                                           n_samples=5000, n_lloyd=15, seed=seed)
            centers = clip_points_to_boundary(centers, boundary_polygon)

    n_actual = len(centers)


    if n_actual > 0:
        r_base = np.sqrt(target_density * domain_area / (n_actual * np.pi))
    else:
        r_base = 0.1

    radii = r_base * (0.8 + 0.4 * np.random.rand(n_actual))



    if n_actual > 1:
        for i in range(n_actual):
            dists = np.linalg.norm(centers - centers[i], axis=1)
            dists[i] = np.inf
            min_dist = np.min(dists)
            max_r = 0.45 * min_dist
            radii[i] = min(radii[i], max_r)

    rel_density = compute_lattice_relative_density(centers, radii, domain_area)

    all_ring_points, cell_rings = generate_circle_lattice(centers, radii)
    words = generate_print_path_word(centers, radii)

    return {
        "centers": centers,
        "radii": radii,
        "relative_density": rel_density,
        "boundary_polygon": boundary_polygon,
        "print_path_words": words,
        "all_ring_points": all_ring_points,
        "cell_rings": cell_rings,
    }
