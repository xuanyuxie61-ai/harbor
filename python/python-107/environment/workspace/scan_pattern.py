
import numpy as np


def generate_rectilinear_scan(x_range, y_range, n_x, n_y):
    x_vals = np.linspace(x_range[0], x_range[1], n_x)
    y_vals = np.linspace(y_range[0], y_range[1], n_y)
    xx, yy = np.meshgrid(x_vals, y_vals)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    return coords


def generate_radial_scan(center, radius, n_angles, n_points_per_line):
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
    theta_max = 8.0 * np.pi
    theta = np.linspace(0, theta_max, n_points)
    a = radius / theta_max
    r = a * theta
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.column_stack([x, y])


def generate_dense_scan_for_speckle(x_range, y_range, n_x, n_y, n_compound=4):
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
    scan_xy = np.asarray(scan_xy, dtype=float)
    axis_idx = 0 if fast_axis == 'x' else 1
    other_idx = 1 - axis_idx


    slow_vals = scan_xy[:, other_idx]
    unique_slow = np.unique(np.round(slow_vals / tol).astype(int)) * tol
    bscan_indices = []
    for val in unique_slow:
        mask = np.abs(slow_vals - val) < tol
        indices = np.where(mask)[0]

        order = np.argsort(scan_xy[indices, axis_idx])
        bscan_indices.append(indices[order])
    return bscan_indices






def scan_uniformity_metric(coords):
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
    coords = np.asarray(coords, dtype=float)
    (xmin, xmax), (ymin, ymax) = domain_bounds
    n_bins = max(int(np.sqrt(coords.shape[0])), 2)
    H, _, _ = np.histogram2d(coords[:, 0], coords[:, 1],
                              bins=n_bins,
                              range=[[xmin, xmax], [ymin, ymax]])
    coverage = np.sum(H > 0) / (n_bins * n_bins)
    return coverage
