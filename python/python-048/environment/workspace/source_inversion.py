
import numpy as np
from typing import Tuple, List
from seismic_green import SeismicGreen
from moment_tensor import MomentTensor


def source_location_grid_search(stations: np.ndarray,
                                 observed_tt: np.ndarray,
                                 velocity: float,
                                 grid_bounds: Tuple[Tuple[float, float], ...],
                                 grid_dims: Tuple[int, int, int]) -> Tuple[np.ndarray, float]:
    if stations.shape[0] != observed_tt.size:
        raise ValueError("台站数与观测走时不匹配")

    nx, ny, nz = grid_dims
    x_vals = np.linspace(grid_bounds[0][0], grid_bounds[0][1], nx)
    y_vals = np.linspace(grid_bounds[1][0], grid_bounds[1][1], ny)
    z_vals = np.linspace(grid_bounds[2][0], grid_bounds[2][1], nz)

    best_misfit = np.inf
    best_loc = np.zeros(3)

    for xi in x_vals:
        for yi in y_vals:
            for zi in z_vals:
                loc = np.array([xi, yi, zi])
                dists = np.linalg.norm(stations - loc, axis=1)
                tt_calc = dists / velocity
                misfit = np.sum((observed_tt - tt_calc) ** 2)
                if misfit < best_misfit:
                    best_misfit = misfit
                    best_loc = loc

    return best_loc, best_misfit


def moment_tensor_inversion(stations: np.ndarray,
                             observed_displacements: np.ndarray,
                             source_loc: np.ndarray,
                             green: SeismicGreen) -> MomentTensor:









    raise NotImplementedError("Hole 2: 请实现矩张量反演")


def connectivity_source_cluster(grid_occupied: np.ndarray,
                                 grid_dims: Tuple[int, int, int]) -> List[np.ndarray]:
    nx, ny, nz = grid_dims
    if grid_occupied.size != nx * ny * nz:
        raise ValueError("网格尺寸不匹配")

    grid = grid_occupied.reshape((nx, ny, nz)).astype(bool)
    visited = np.zeros_like(grid, dtype=bool)
    clusters = []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if grid[i, j, k] and not visited[i, j, k]:

                    queue = [(i, j, k)]
                    visited[i, j, k] = True
                    cluster = []
                    while queue:
                        ci, cj, ck = queue.pop(0)
                        cluster.append((ci, cj, ck))
                        for di, dj, dk in [(-1, 0, 0), (1, 0, 0),
                                           (0, -1, 0), (0, 1, 0),
                                           (0, 0, -1), (0, 0, 1)]:
                            ni, nj, nk = ci + di, cj + dj, ck + dk
                            if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                                if grid[ni, nj, nk] and not visited[ni, nj, nk]:
                                    visited[ni, nj, nk] = True
                                    queue.append((ni, nj, nk))
                    clusters.append(np.array(cluster))
    return clusters
