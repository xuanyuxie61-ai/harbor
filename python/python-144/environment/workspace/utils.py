"""
utils.py
通用工具模块，融合了数据扰动、矩阵上采样、凸包计算与多维尺度分析。

融入的原项目核心算法：
- 132_caesar: Caesar移位思想用于金融数据敏感性扰动
- 578_image_double: 矩阵双线性插值/上采样
- 891_polygonal_surface_display: 多边形顶点与面的数据解析思想
- 306_distance_to_position: 欧氏距离到坐标的MDS嵌入
"""

import numpy as np
from scipy.spatial import ConvexHull
from scipy.linalg import eigh


def caesar_perturb(data: np.ndarray, k: int = 3, axis: int = -1) -> np.ndarray:
    """
    基于Caesar移位思想对数值序列进行循环扰动，用于敏感性分析与隐私保护。

    数学原理：
    对输入向量 v，构造扰动向量
        v'_i = v_{(i + k) mod n}
    并添加高斯噪声 ε ~ N(0, σ²)，其中 σ = std(v) / 10。

    参数
    ----------
    data : np.ndarray
        输入金融数据序列。
    k : int
        循环移位步长，默认 3。
    axis : int
        扰动轴，默认最后一个轴。

    返回
    -------
    np.ndarray
        扰动后的数据。
    """
    if data.size == 0:
        raise ValueError("caesar_perturb: 输入数据不能为空。")
    shifted = np.roll(data, shift=k, axis=axis)
    sigma = np.std(data) / 10.0
    if sigma < 1e-12:
        sigma = 1e-6
    noise = np.random.normal(0.0, sigma, size=data.shape)
    return shifted + noise


def matrix_interpolation_upsample(A: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    矩阵上采样（双线性插值思想），用于协方差矩阵或收益率矩阵的精细化。

    对 M×N 矩阵 A，生成 (factor*M)×(factor*N) 矩阵 B，其中每个原像素
    复制到 factor×factor 的块中，随后用简单的均值滤波做平滑：
        B_{2i,2j} = A_{i,j}
        B_{2i+1,2j} = (A_{i,j} + A_{i+1,j}) / 2
        B_{2i,2j+1} = (A_{i,j} + A_{i,j+1}) / 2
        B_{2i+1,2j+1} = (A_{i,j} + A_{i+1,j} + A_{i,j+1} + A_{i+1,j+1}) / 4

    参数
    ----------
    A : np.ndarray
        输入矩阵。
    factor : int
        上采样倍数，默认 2，仅支持 2 的幂次或等于 2。

    返回
    -------
    np.ndarray
        上采样后的矩阵。
    """
    if factor != 2:
        raise ValueError("matrix_interpolation_upsample: 当前仅支持 factor=2。")
    m, n = A.shape
    if m < 2 or n < 2:
        raise ValueError("matrix_interpolation_upsample: 矩阵维度至少为 2×2。")
    B = np.zeros((2 * m, 2 * n), dtype=A.dtype)
    B[0::2, 0::2] = A
    # 水平插值
    B[0::2, 1::2][:, :-1] = (A[:, :-1] + A[:, 1:]) / 2.0
    B[0::2, -1] = A[:, -1]
    # 垂直插值
    for i in range(m - 1):
        B[2*i+1, :] = (B[2*i, :] + B[2*i+2, :]) / 2.0
    B[-1, :] = B[-2, :]
    return B


def polygonal_convex_hull(points: np.ndarray) -> dict:
    """
    计算点集的凸包，用于构建马科维茨有效前沿的凸近似。

    数学背景：
    给定 d 维空间中的点集 {x_i}_{i=1}^n，凸包 Conv(P) 定义为
        Conv(P) = { Σ λ_i x_i | λ_i ≥ 0, Σ λ_i = 1 }。
    有效前沿在收益-风险空间中即为凸包的下边界。

    参数
    ----------
    points : np.ndarray, shape (n, d)
        输入点集。

    返回
    -------
    dict
        包含 'vertices'（凸包顶点索引）、'volume'（体积/面积）、
        'simplices'（构成凸包的单纯形）的字典。
    """
    if points.ndim != 2:
        raise ValueError("polygonal_convex_hull: 输入必须是二维数组。")
    if points.shape[0] < points.shape[1] + 1:
        raise ValueError("polygonal_convex_hull: 点数量不足，无法构成非退化凸包。")
    hull = ConvexHull(points)
    return {
        "vertices": hull.vertices,
        "volume": hull.volume,
        "simplices": hull.simplices,
        "n_points": points.shape[0],
    }


def distance_to_position_mds(distance: np.ndarray, dim: int = 2,
                              max_iter: int = 200, tol: float = 1e-6) -> np.ndarray:
    """
    经典多维尺度分析（Classical MDS）：从距离矩阵恢复低维嵌入坐标。

    数学推导：
    设 D 为 n×n 距离矩阵，定义双中心化矩阵
        B = -0.5 * J * D^{(2)} * J,
    其中 D^{(2)}_{ij} = D_{ij}^2，J = I - (1/n) 1 1^T 为中心化矩阵。
    对 B 进行谱分解 B = V Λ V^T，取前 dim 个最大正特征值对应的特征向量，
    则嵌入坐标为
        X = V_dim * sqrt(Λ_dim)。

    参数
    ----------
    distance : np.ndarray
        对称距离矩阵，对角线为 0。
    dim : int
        目标维度。
    max_iter : int
        未使用（保留接口一致性）。
    tol : float
        未使用（保留接口一致性）。

    返回
    -------
    np.ndarray, shape (n, dim)
        低维嵌入坐标。
    """
    n = distance.shape[0]
    if distance.shape[0] != distance.shape[1]:
        raise ValueError("distance_to_position_mds: 距离矩阵必须是方阵。")
    # 平方距离
    D2 = distance ** 2
    # 双中心化
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    # 对称化并谱分解
    B = 0.5 * (B + B.T)
    eigvals, eigvecs = eigh(B)
    # 取最大的 dim 个正特征值
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    positive = eigvals > 1e-10
    if np.sum(positive) < dim:
        raise RuntimeError("distance_to_position_mds: 正特征值数量不足，无法嵌入到指定维度。")
    Lambda = np.diag(np.sqrt(np.maximum(eigvals[:dim], 0.0)))
    X = eigvecs[:, :dim] @ Lambda
    return X


def r8mat_condition_number(A: np.ndarray) -> float:
    """
    计算矩阵条件数（2-范数），用于评估协方差矩阵的数值稳定性。

    κ_2(A) = σ_max(A) / σ_min(A)。
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("r8mat_condition_number: 输入必须是方阵。")
    s = np.linalg.svd(A, compute_uv=False)
    if s[-1] < 1e-15:
        return np.inf
    return s[0] / s[-1]
