"""
scan_pattern.py

OCT scan pattern generation and coordinate management.
Handles A-scan, B-scan, and volumetric scan coordinates.
Incorporates XY coordinate I/O concepts from xy_display / xy_io.

Scan patterns include:
- Rectilinear raster scan
- Radial scan (star pattern)
- Spiral scan
- Dense sampling for speckle reduction
"""

import numpy as np


def generate_rectilinear_scan(x_range, y_range, n_x, n_y):
    """
    Generate a rectilinear raster scan pattern.

    Parameters
    ----------
    x_range : tuple
        (x_min, x_max).
    y_range : tuple
        (y_min, y_max).
    n_x, n_y : int
        Number of scan lines in x and y.

    Returns
    -------
    coords : ndarray, shape (n_x * n_y, 2)
        Scan coordinates (x, y).
    """
    x_vals = np.linspace(x_range[0], x_range[1], n_x)
    y_vals = np.linspace(y_range[0], y_range[1], n_y)
    xx, yy = np.meshgrid(x_vals, y_vals)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    return coords


def generate_radial_scan(center, radius, n_angles, n_points_per_line):
    """
    Generate a radial (star) scan pattern.

    Parameters
    ----------
    center : tuple
        (cx, cy).
    radius : float
        Scan radius.
    n_angles : int
        Number of radial lines.
    n_points_per_line : int
        Points per radial line.

    Returns
    -------
    coords : ndarray, shape (n_angles * n_points_per_line, 2)
    """
    angles = np.linspace(0, 2.0 * np.pi, n_angles, endpoint=False)
    r_vals = np.linspace(0, radius, n_points_per_line)
    coords = []
    for theta in angles:
        for r in r_vals:
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            coords.append([x, y])
    return np.array(coords, dtype=float)


def generate_spiral_scan(center, radius, n_points):
    """
    Generate an Archimedean spiral scan pattern.

    r = a * theta, with a chosen so that max radius is reached at max angle.

    Parameters
    ----------
    center : tuple
        (cx, cy).
    radius : float
        Maximum radius.
    n_points : int
        Total number of points.

    Returns
    -------
    coords : ndarray, shape (n_points, 2)
    """
    theta_max = 8.0 * np.pi  # 4 full turns
    theta = np.linspace(0, theta_max, n_points)
    a = radius / theta_max
    r = a * theta
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.column_stack([x, y])


def generate_dense_scan_for_speckle(x_range, y_range, n_x, n_y, n_compound=4):
    """
    Generate a dense scan pattern for speckle reduction via compounding.

    Slightly offset multiple raster scans and interleave them.

    Parameters
    ----------
    x_range, y_range : tuple
    n_x, n_y : int
    n_compound : int
        Number of offset scans.

    Returns
    -------
    coords : ndarray
    offsets : list
        Offset vectors used.
    """
    dx = (x_range[1] - x_range[0]) / n_x
    dy = (y_range[1] - y_range[0]) / n_y
    all_coords = []
    offsets = []
    for i in range(n_compound):
        offset_x = (i % 2) * dx * 0.5
        offset_y = (i // 2) * dy * 0.5
        offsets.append([offset_x, offset_y])
        c = generate_rectilinear_scan(
            (x_range[0] + offset_x, x_range[1]),
            (y_range[0] + offset_y, y_range[1]),
            n_x, n_y
        )
        all_coords.append(c)
    coords = np.vstack(all_coords)
    return coords, offsets


def scan_pattern_to_ascan_coords(scan_xy, depth_array, fast_axis='x'):
    """
    Convert 2D scan pattern to 3D A-scan coordinates.

    For each (x, y) scan position, create a depth array z.

    Parameters
    ----------
    scan_xy : ndarray, shape (n_scan, 2)
    depth_array : ndarray
        1D depth array.
    fast_axis : str
        'x' or 'y' for B-scan orientation.

    Returns
    -------
    coords_3d : ndarray, shape (n_scan * n_depth, 3)
        [x, y, z] coordinates.
    scan_indices : ndarray
        Index mapping from 3D coord to scan position.
    """
    scan_xy = np.asarray(scan_xy, dtype=float)
    depth_array = np.asarray(depth_array, dtype=float)
    n_scan = scan_xy.shape[0]
    n_depth = len(depth_array)
    coords_3d = np.zeros((n_scan * n_depth, 3), dtype=float)
    scan_indices = np.repeat(np.arange(n_scan), n_depth)

    for i in range(n_scan):
        base = i * n_depth
        coords_3d[base:base + n_depth, 0] = scan_xy[i, 0]
        coords_3d[base:base + n_depth, 1] = scan_xy[i, 1]
        coords_3d[base:base + n_depth, 2] = depth_array

    return coords_3d, scan_indices


def sort_scan_for_bscan(scan_xy, fast_axis='x', tol=1e-8):
    """
    Sort scan coordinates into B-scan frames.

    A B-scan is a collection of A-scans along the fast axis.

    Parameters
    ----------
    scan_xy : ndarray
    fast_axis : str
    tol : float

    Returns
    -------
    bscan_indices : list of ndarray
        Each element contains indices for one B-scan.
    """
    scan_xy = np.asarray(scan_xy, dtype=float)
    axis_idx = 0 if fast_axis == 'x' else 1
    other_idx = 1 - axis_idx

    # Group by slow axis coordinate
    slow_vals = scan_xy[:, other_idx]
    unique_slow = np.unique(np.round(slow_vals / tol).astype(int)) * tol
    bscan_indices = []
    for val in unique_slow:
        mask = np.abs(slow_vals - val) < tol
        indices = np.where(mask)[0]
        # Sort within B-scan by fast axis
        order = np.argsort(scan_xy[indices, axis_idx])
        bscan_indices.append(indices[order])
    return bscan_indices


# ---------------------------------------------------------------------------
# Scan pattern quality metrics
# ---------------------------------------------------------------------------

def scan_uniformity_metric(coords):
    """
    Compute uniformity metric based on nearest-neighbor distances.

    For uniform sampling, nearest-neighbor distances should be similar.
    Metric = std(d_nn) / mean(d_nn), lower is better.

    Parameters
    ----------
    coords : ndarray, shape (n, 2)

    Returns
    -------
    uniformity : float
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    if n < 2:
        return 0.0
    d_nn = []
    for i in range(n):
        dists = np.linalg.norm(coords - coords[i, :], axis=1)
        dists[i] = np.inf
        d_nn.append(np.min(dists))
    d_nn = np.array(d_nn)
    mean_d = np.mean(d_nn)
    if mean_d < 1e-14:
        return 0.0
    return np.std(d_nn) / mean_d


def scan_coverage_metric(coords, domain_bounds):
    """
    Compute coverage fraction of scan pattern within domain.

    Uses a simple histogram approach.

    Parameters
    ----------
    coords : ndarray, shape (n, 2)
    domain_bounds : tuple
        ((xmin, xmax), (ymin, ymax)).

    Returns
    -------
    coverage : float
        Fraction of domain bins that contain at least one point.
    """
    coords = np.asarray(coords, dtype=float)
    (xmin, xmax), (ymin, ymax) = domain_bounds
    n_bins = max(int(np.sqrt(coords.shape[0])), 2)
    H, _, _ = np.histogram2d(coords[:, 0], coords[:, 1],
                              bins=n_bins,
                              range=[[xmin, xmax], [ymin, ymax]])
    coverage = np.sum(H > 0) / (n_bins * n_bins)
    return coverage
