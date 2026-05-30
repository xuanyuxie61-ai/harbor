
import numpy as np
from typing import List, Tuple, Dict


def compute_flow_direction(surface: np.ndarray,
                           dx: float,
                           dy: float) -> np.ndarray:
    surface = np.asarray(surface, dtype=np.float64)
    ny, nx = surface.shape

    flow_dir = np.full((ny, nx), -1, dtype=np.int32)


    offsets = [
        (0, 1, dx),
        (1, 1, np.sqrt(dx**2 + dy**2)),
        (1, 0, dy),
        (1, -1, np.sqrt(dx**2 + dy**2)),
        (0, -1, dx),
        (-1, -1, np.sqrt(dx**2 + dy**2)),
        (-1, 0, dy),
        (-1, 1, np.sqrt(dx**2 + dy**2)),
    ]

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            h0 = surface[i, j]
            max_slope = -1e20
            best_dir = -1

            for code, (di, dj, dist) in enumerate(offsets):
                h_neighbor = surface[i + di, j + dj]
                slope = (h0 - h_neighbor) / dist
                if slope > max_slope:
                    max_slope = slope
                    best_dir = code


            if max_slope <= 0:
                best_dir = -1

            flow_dir[i, j] = best_dir

    return flow_dir


def label_connected_components_2d(mask: np.ndarray,
                                   connectivity: int = 4) -> Tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=np.bool_)
    ny, nx = mask.shape
    labels = np.zeros((ny, nx), dtype=np.int32)
    label_id = 0

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                     (0, 1), (1, -1), (1, 0), (1, 1)]

    for i in range(ny):
        for j in range(nx):
            if mask[i, j] and labels[i, j] == 0:
                label_id += 1
                stack = [(i, j)]
                labels[i, j] = label_id

                while stack:
                    ci, cj = stack.pop()
                    for di, dj in neighbors:
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < ny and 0 <= nj < nx:
                            if mask[ni, nj] and labels[ni, nj] == 0:
                                labels[ni, nj] = label_id
                                stack.append((ni, nj))

    return labels, label_id


def identify_catchments(surface: np.ndarray,
                         mask: np.ndarray,
                         dx: float,
                         dy: float,
                         min_area: float = 1e6) -> Dict[int, Dict]:
    labels, n_comp = label_connected_components_2d(mask, connectivity=4)
    catchments = {}

    for comp_id in range(1, n_comp + 1):
        comp_mask = (labels == comp_id)
        n_cells = int(np.sum(comp_mask))
        area = n_cells * dx * dy

        if area < min_area:
            continue

        indices = np.argwhere(comp_mask)
        ys = indices[:, 0]
        xs = indices[:, 1]

        centroid_x = float(np.mean(xs)) * dx
        centroid_y = float(np.mean(ys)) * dy
        mean_elev = float(np.mean(surface[comp_mask]))
        min_elev = float(np.min(surface[comp_mask]))
        max_elev = float(np.max(surface[comp_mask]))


        if max_elev > min_elev:
            hypsometric_integral = (mean_elev - min_elev) / (max_elev - min_elev)
        else:
            hypsometric_integral = 0.5

        catchments[comp_id] = {
            'area_m2': area,
            'n_cells': n_cells,
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'mean_elevation': mean_elev,
            'min_elevation': min_elev,
            'max_elevation': max_elev,
            'hypsometric_integral': hypsometric_integral,
        }

    return catchments


def compute_drainage_density(catchments: Dict[int, Dict],
                              total_ice_area: float) -> float:
    if total_ice_area <= 0:
        return 0.0
    return len(catchments) / total_ice_area


def merge_small_catchments(labels: np.ndarray,
                           catchments: Dict[int, Dict],
                           min_area: float) -> np.ndarray:
    new_labels = labels.copy()
    ny, nx = labels.shape

    for comp_id, info in catchments.items():
        if info['area_m2'] < min_area:

            new_labels[labels == comp_id] = 0

    return new_labels


def extract_main_flow_branches(surface: np.ndarray,
                                thickness: np.ndarray,
                                dx: float, dy: float,
                                velocity_threshold: float = 10.0) -> Dict[int, Dict]:
    ny, nx = surface.shape
    n = 3.0


    grad_x = np.zeros_like(surface)
    grad_y = np.zeros_like(surface)
    grad_x[:, 1:-1] = (surface[:, 2:] - surface[:, :-2]) / (2.0 * dx)
    grad_y[1:-1, :] = (surface[2:, :] - surface[:-2, :]) / (2.0 * dy)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag = np.maximum(grad_mag, 1e-12)

    intensity = (thickness ** (n + 1.0)) * (grad_mag ** (n - 1.0))
    mask = intensity > velocity_threshold

    branches = identify_catchments(surface, mask, dx, dy, min_area=dx*dy*10)
    return branches
