
import numpy as np
from typing import List, Tuple, Callable, Optional


def metric_identity(x: np.ndarray) -> np.ndarray:
    return np.eye(2, dtype=float)


def metric_anisotropic(x: np.ndarray, alpha: float = 10.0) -> np.ndarray:
    M = np.eye(2, dtype=float)
    M[0, 0] = alpha
    return M


def metric_boundary_layer(x: np.ndarray, eps: float = 0.01) -> np.ndarray:
    d = x[0] + 1e-8
    s = 1.0 / (eps + d)
    M = np.eye(2, dtype=float)
    M[0, 0] = min(s, 100.0)
    return M


def anisotropic_distance(z1: np.ndarray, z2: np.ndarray,
                         metric_func: Callable) -> float:
    mid = 0.5 * (z1 + z2)
    M = metric_func(mid)
    dz = z1 - z2
    val = float(dz @ M @ dz)
    if val < 0.0:
        val = 0.0
    return np.sqrt(val)


def lloyd_iteration_cvt(
    generators: np.ndarray,
    n_samples: int = 20000,
    metric_func: Callable = metric_identity,
    domain: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
) -> np.ndarray:
    n = generators.shape[0]
    xmin, xmax, ymin, ymax = domain

    samples = np.column_stack((
        np.random.uniform(xmin, xmax, size=n_samples),
        np.random.uniform(ymin, ymax, size=n_samples)
    ))


    belongs = np.zeros(n_samples, dtype=int)
    for s in range(n_samples):
        best_d = np.inf
        best_i = 0
        for i in range(n):
            d = anisotropic_distance(samples[s], generators[i], metric_func)
            if d < best_d:
                best_d = d
                best_i = i
        belongs[s] = best_i


    new_gen = np.zeros_like(generators)
    counts = np.zeros(n, dtype=int)
    for s in range(n_samples):
        i = belongs[s]
        new_gen[i] += samples[s]
        counts[i] += 1

    for i in range(n):
        if counts[i] > 0:
            new_gen[i] /= counts[i]
        else:

            new_gen[i] = np.array([
                np.random.uniform(xmin, xmax),
                np.random.uniform(ymin, ymax)
            ])
    return new_gen


def compute_cvt(
    n_subdomains: int,
    n_iterations: int = 30,
    n_samples: int = 20000,
    metric_func: Callable = metric_identity,
    domain: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    tol: float = 1e-4
) -> np.ndarray:
    xmin, xmax, ymin, ymax = domain
    generators = np.column_stack((
        np.random.uniform(xmin, xmax, size=n_subdomains),
        np.random.uniform(ymin, ymax, size=n_subdomains)
    ))

    for it in range(n_iterations):
        old = generators.copy()
        generators = lloyd_iteration_cvt(
            generators, n_samples, metric_func, domain
        )
        shift = np.max(np.linalg.norm(generators - old, axis=1))
        if shift < tol:
            break
    return generators


def compute_subdomain_boundaries(
    generators: np.ndarray,
    metric_func: Callable = metric_identity,
    domain: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    grid_res: int = 80
) -> List[np.ndarray]:
    n = generators.shape[0]
    masks = []
    xmin, xmax, ymin, ymax = domain
    xs = np.linspace(xmin, xmax, grid_res)
    ys = np.linspace(ymin, ymax, grid_res)
    for i in range(n):
        mask = np.zeros((grid_res, grid_res), dtype=bool)
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                p = np.array([x, y])
                best_d = np.inf
                best_j = -1
                for j in range(n):
                    d = anisotropic_distance(p, generators[j], metric_func)
                    if d < best_d:
                        best_d = d
                        best_j = j
                mask[iy, ix] = (best_j == i)
        masks.append(mask)
    return masks


def subdomain_overlap_masks(
    masks: List[np.ndarray],
    overlap_layers: int = 2
) -> List[np.ndarray]:
    from scipy import ndimage

    try:
        expanded = []
        for mask in masks:
            exp = ndimage.binary_dilation(mask, iterations=overlap_layers)
            expanded.append(exp)
        return expanded
    except Exception:
        expanded = []
        ny, nx = masks[0].shape
        for mask in masks:
            exp = mask.copy()
            for _ in range(overlap_layers):
                new_exp = exp.copy()
                for iy in range(ny):
                    for ix in range(nx):
                        if exp[iy, ix]:
                            for dy in (-1, 0, 1):
                                for dx in (-1, 0, 1):
                                    jy, jx = iy + dy, ix + dx
                                    if 0 <= jy < ny and 0 <= jx < nx:
                                        new_exp[jy, jx] = True
                exp = new_exp
            expanded.append(exp)
        return expanded


def extract_interface_nodes(
    mask_i: np.ndarray,
    mask_j: np.ndarray,
    domain: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
) -> np.ndarray:
    overlap = mask_i & mask_j
    ny, nx = overlap.shape
    xmin, xmax, ymin, ymax = domain
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    points = []
    for iy in range(ny):
        for ix in range(nx):
            if overlap[iy, ix]:
                points.append([xs[ix], ys[iy]])
    return np.array(points, dtype=float)
