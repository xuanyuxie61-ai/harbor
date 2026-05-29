"""
mesh_partition.py
=================
Anisotropic Centroidal Voronoi Tessellation (CVT) for domain decomposition
mesh partitioning in parallel finite element methods.

Integrates concepts from:
  * cvt_metric (anisotropic CVT with spatially varying metric tensor)
  * ice_to_medit (mesh I/O topology)
  * tetrahedron_slice_display (geometric intersection for subdomain boundary extraction)

Mathematical background
-----------------------
Given a domain Omega in R^2 and a Riemannian metric tensor field
    M(x) = [[m11(x), m12(x)], [m12(x), m22(x)]]
that is SPD everywhere, the anisotropic distance between points x and y is:
    d_M(x,y) = sqrt( (x-y)^T M((x+y)/2) (x-y) )

A Centroidal Voronoi Tessellation (CVT) minimizes the energy functional:
    E( {z_i} ) = sum_i integral_{V_i} rho(x) d_M(x, z_i)^2 dx
where V_i = { x in Omega : d_M(x, z_i) <= d_M(x, z_j) for all j != i }

The Lloyd algorithm iterates:
  1. For fixed generators {z_i}, compute Voronoi cells V_i.
  2. Update each generator to the centroid (mass center) of V_i.
  3. Repeat until convergence: max_i ||z_i^{new} - z_i^{old}|| < tol.

In HPC domain decomposition, anisotropic CVT produces subdomain partitions
that adapt to the PDE coefficients (e.g., high diffusion in one direction),
minimizing communication volume while balancing computational load.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional


def metric_identity(x: np.ndarray) -> np.ndarray:
    """Euclidean metric: M(x) = I."""
    return np.eye(2, dtype=float)


def metric_anisotropic(x: np.ndarray, alpha: float = 10.0) -> np.ndarray:
    """
    Anisotropic metric tensor with higher cost in x-direction.
    M = diag(alpha, 1).  alpha > 1 stretches cells in x-direction.
    """
    M = np.eye(2, dtype=float)
    M[0, 0] = alpha
    return M


def metric_boundary_layer(x: np.ndarray, eps: float = 0.01) -> np.ndarray:
    """
    Metric adapted to boundary layer at x=0:
    M = diag(1/eps, 1) near x=0, transitioning to I away from boundary.
    """
    d = x[0] + 1e-8
    s = 1.0 / (eps + d)
    M = np.eye(2, dtype=float)
    M[0, 0] = min(s, 100.0)
    return M


def anisotropic_distance(z1: np.ndarray, z2: np.ndarray,
                         metric_func: Callable) -> float:
    """
    Compute d_M(z1, z2) = sqrt((z1-z2)^T M((z1+z2)/2) (z1-z2)).
    """
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
    """
    One Lloyd iteration using Monte Carlo sampling.

    Parameters
    ----------
    generators : (n, 2) array of generator points.
    n_samples  : Number of Monte Carlo sample points per iteration.
    metric_func: Function mapping (2,) -> (2,2) SPD matrix.
    domain     : (xmin, xmax, ymin, ymax).

    Returns
    -------
    new_generators : Updated generator positions (centroids of their cells).
    """
    n = generators.shape[0]
    xmin, xmax, ymin, ymax = domain
    # Monte Carlo samples uniformly in domain
    samples = np.column_stack((
        np.random.uniform(xmin, xmax, size=n_samples),
        np.random.uniform(ymin, ymax, size=n_samples)
    ))

    # Assign each sample to nearest generator using anisotropic distance
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

    # Compute centroids
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
            # Empty cell: reinitialize randomly
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
    """
    Compute anisotropic CVT for domain decomposition partitioning.

    Returns
    -------
    generators : (n_subdomains, 2) array of subdomain centers.
    """
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
    """
    Rasterize subdomain boundaries on a regular grid.
    Returns a list of boolean masks (grid_res x grid_res) for each subdomain.
    """
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
    """
    Expand each subdomain mask by overlap_layers in grid index space
    to create overlapping Schwarz subdomains.
    """
    from scipy import ndimage
    # If scipy not available, fallback to manual dilation
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
    """
    Extract grid points that lie in the overlap region between
    subdomain i and subdomain j.
    Returns an array of shape (n_points, 2) with coordinates.
    """
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
