"""
data_io.py
多维数据I/O与预处理模块

基于种子项目:
- 1424_xyz_io: 3D点云文件I/O
- 582_image_normalize: 图像归一化
"""

import numpy as np
from typing import Tuple, List, Optional


def generate_helical_point_cloud(n_points: int = 100,
                                 radius: float = 1.0,
                                 pitch: float = 0.5) -> np.ndarray:
    """
    生成螺旋3D点云数据。

    参数方程:
        x(t) = r * cos(t)
        y(t) = r * sin(t)
        z(t) = p * t / (2*pi)

    t in [0, 4*pi]
    """
    t = np.linspace(0.0, 4.0 * np.pi, n_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = pitch * t / (2.0 * np.pi)

    return np.column_stack([x, y, z])


def normalize_features(data: np.ndarray,
                       method: str = 'minmax',
                       axis: int = 0) -> Tuple[np.ndarray, dict]:
    """
    特征归一化/标准化。

    方法:
        'minmax': x' = (x - min) / (max - min + eps)
        'zscore': x' = (x - mean) / (std + eps)
        'robust': x' = (x - median) / (IQR + eps)

    返回:
        normalized_data: 归一化后的数据
        params: 包含归一化参数的字典
    """
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
    """
    写入XYZ格式点云文件。

    格式:
        # 注释行
        x y z
        ...
    """
    with open(filename, 'w') as f:
        f.write(f"# XYZ point cloud, {points.shape[0]} points\n")
        for i in range(points.shape[0]):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}\n")


def read_xyz_data(filename: str) -> np.ndarray:
    """
    读取XYZ格式点云文件。
    """
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
    """
    主成分分析 (PCA)。

    数学推导:
        1. 中心化: X_c = X - mean(X)
        2. 协方差: C = X_c^T X_c / (n-1)
        3. 特征分解: C = V Lambda V^T
        4. 投影: Z = X_c V[:, :k]

    参数:
        data: (n_samples, n_features)
        n_components: 保留的主成分数

    返回:
        projected: (n_samples, n_components)
        explained_variance_ratio: 各主成分方差贡献率
    """
    # 中心化
    mean = np.mean(data, axis=0)
    Xc = data - mean

    # 协方差矩阵
    C = (Xc.T @ Xc) / max(data.shape[0] - 1, 1)

    # 特征分解
    eigvals, eigvecs = np.linalg.eigh(C)

    # 按特征值降序
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # 投影
    V_k = eigvecs[:, :n_components]
    projected = Xc @ V_k

    total_var = np.sum(eigvals)
    if total_var > 1e-15:
        explained_ratio = eigvals[:n_components] / total_var
    else:
        explained_ratio = np.zeros(n_components)

    return projected, explained_ratio
