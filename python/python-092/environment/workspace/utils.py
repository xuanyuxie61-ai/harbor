
import numpy as np


def safe_divide(a, b, fill_value=0.0):
    b = np.asarray(b, dtype=float)
    result = np.empty_like(np.asarray(a, dtype=float))
    mask = np.abs(b) > 1e-14
    result[mask] = np.asarray(a, dtype=float)[mask] / b[mask]
    result[~mask] = fill_value
    return result


def normalize_vector(v):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm < 1e-14:
        return np.zeros_like(v)
    return v / norm


def compute_mean_free_path(volume, surface_area):
    if surface_area < 1e-14:
        return 0.0
    return 4.0 * volume / surface_area


def sabine_absorption_to_t60(volume, total_absorption):
    if total_absorption < 1e-14:
        total_absorption = 1e-14
    return 0.161 * volume / total_absorption


def eyring_absorption_to_t60(volume, surface_area, avg_absorption):
    if avg_absorption >= 1.0:
        avg_absorption = 0.999
    if avg_absorption < 1e-14:
        return sabine_absorption_to_t60(volume, surface_area * avg_absorption)
    denom = -surface_area * np.log(1.0 - avg_absorption)
    if abs(denom) < 1e-14:
        denom = 1e-14
    return 0.161 * volume / denom


def write_matrix_file(filename, matrix, fmt='%.6f'):
    np.savetxt(filename, matrix, fmt=fmt)


def read_node_element_files(node_file, element_file):
    nodes = np.loadtxt(node_file, dtype=float)
    elements = np.loadtxt(element_file, dtype=int)
    return nodes, elements


def compute_bounding_box(points):
    return np.min(points, axis=0), np.max(points, axis=0)


def is_point_inside_box(point, box_min, box_max, tol=1e-10):
    point = np.asarray(point)
    return np.all(point >= box_min - tol) and np.all(point <= box_max + tol)


def linear_regression(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return 0.0, 0.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    if abs(ss_xx) < 1e-14:
        return 0.0, y_mean
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    return slope, intercept


def db_to_linear(db):
    return 10.0 ** (db / 20.0)


def linear_to_db(linear):
    linear = np.maximum(np.asarray(linear, dtype=float), 1e-15)
    return 20.0 * np.log10(linear)


def energy_to_db(energy):
    energy = np.maximum(np.asarray(energy, dtype=float), 1e-15)
    return 10.0 * np.log10(energy)


def check_finite_and_real(arr, name="array"):
    arr = np.asarray(arr)
    if not np.all(np.isfinite(arr)):
        bad_count = np.sum(~np.isfinite(arr))
        raise ValueError(f"{name} contains {bad_count} non-finite values")
    return True


def compute_statistics(data):
    data = np.asarray(data, dtype=float)
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data)),
        'q25': float(np.percentile(data, 25)),
        'q75': float(np.percentile(data, 75)),
    }
