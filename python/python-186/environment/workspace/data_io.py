
import numpy as np
from typing import Tuple, List, Optional


def generate_helical_point_cloud(n_points: int = 100,
                                 radius: float = 1.0,
                                 pitch: float = 0.5) -> np.ndarray:
    t = np.linspace(0.0, 4.0 * np.pi, n_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = pitch * t / (2.0 * np.pi)

    return np.column_stack([x, y, z])


def normalize_features(data: np.ndarray,
                       method: str = 'minmax',
                       axis: int = 0) -> Tuple[np.ndarray, dict]:
    if method == 'minmax':
        dmin = np.min(data, axis=axis, keepdims=True)
        dmax = np.max(data, axis=axis, keepdims=True)
        eps = 1e-10
        norm_data = (data - dmin) / (dmax - dmin + eps)
        params = {'method': 'minmax', 'min': dmin, 'max': dmax}

    elif method == 'zscore':
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        eps = 1e-10
        norm_data = (data - mean) / (std + eps)
        params = {'method': 'zscore', 'mean': mean, 'std': std}

    elif method == 'robust':
        median = np.median(data, axis=axis, keepdims=True)
        q1 = np.percentile(data, 25, axis=axis, keepdims=True)
        q3 = np.percentile(data, 75, axis=axis, keepdims=True)
        iqr = q3 - q1
        eps = 1e-10
        norm_data = (data - median) / (iqr + eps)
        params = {'method': 'robust', 'median': median, 'iqr': iqr}

    else:
        norm_data = data.copy()
        params = {'method': 'none'}

    return norm_data, params


def write_xyz_data(filename: str, points: np.ndarray):
    with open(filename, 'w') as f:
        f.write(f"# XYZ point cloud, {points.shape[0]} points\n")
        for i in range(points.shape[0]):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}\n")


def read_xyz_data(filename: str) -> np.ndarray:
    points = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])

    return np.array(points, dtype=np.float64)


def compute_pca_features(data: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, np.ndarray]:

    mean = np.mean(data, axis=0)
    Xc = data - mean


    C = (Xc.T @ Xc) / max(data.shape[0] - 1, 1)


    eigvals, eigvecs = np.linalg.eigh(C)


    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]


    V_k = eigvecs[:, :n_components]
    projected = Xc @ V_k

    total_var = np.sum(eigvals)
    if total_var > 1e-15:
        explained_ratio = eigvals[:n_components] / total_var
    else:
        explained_ratio = np.zeros(n_components)

    return projected, explained_ratio
