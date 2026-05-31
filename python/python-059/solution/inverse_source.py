"""
inverse_source.py
气溶胶源区反演定位模块

整合原项目:
  - 306_distance_to_position: 由距离矩阵反演位置

功能:
  基于观测站点间的气溶胶浓度差异，反演排放源的三维空间位置。
  采用非线性最小二乘方法，将距离-浓度关系转化为几何定位问题。

核心科学问题:
  已知 N 个地面观测站的气溶胶浓度 C_i，假设浓度随源距离衰减:
    C_i = Q / (4π D |r_i - r_source|) * exp( -|r_i - r_source| / L )
  其中 Q 为源强，D 为湍流扩散系数，L 为特征衰减长度。

  反演问题: 由 {C_i} 和 {r_i} 估计 r_source = (x_s, y_s, z_s)。

数值方法:
  1. 构造伪距离矩阵: d_ij = f^{-1}(|C_i - C_j|)
  2. 使用多维标度法 (MDS) 获取初始位置估计
  3. Levenberg-Marquardt 非线性最小二乘优化
"""

import numpy as np
from math import sqrt, exp, pi


class InverseSourceError(Exception):
    pass


def concentration_to_pseudo_distance(C, Q=1.0, D=1.0, L=100.0):
    """
    由浓度 C 反推伪距离 d。

    公式 (简化点源扩散模型):
      C = Q / (4π D d) * exp(-d / L)
    反演使用 Lambert W 函数的近似:
      d ≈ L * W( Q / (4π D L C) )
    当 C 很小时，d 很大。
    """
    C = np.asarray(C, dtype=np.float64)
    if np.any(C <= 0):
        raise InverseSourceError("concentration_to_pseudo_distance: 浓度必须为正")

    arg = Q / (4.0 * pi * D * L * C)
    # Lambert W 近似: W(z) ≈ ln(z) - ln(ln(z)) 对于大 z
    # 对于小 z: W(z) ≈ z
    w = np.zeros_like(arg)
    large = arg > 10.0
    small = arg <= 1.0
    medium = ~(large | small)

    w[small] = arg[small]
    w[medium] = np.log(arg[medium]) - np.log(np.log(arg[medium]) + 1e-15)
    # 大值修正
    w[large] = np.log(arg[large]) - np.log(np.log(arg[large]))
    w = np.maximum(w, 0.0)
    return L * w


def compute_position_from_distance(dist_matrix, dim=3):
    """
    基于距离矩阵反演低维空间中的相对位置 (经典多维标度法, cMDS)。

    算法:
      1. 构造双中心化矩阵 B = -0.5 * J D^{(2)} J
         其中 J = I - (1/N) 1 1^T, D^{(2)} 为距离平方矩阵
      2. 对 B 进行特征值分解: B = V Λ V^T
      3. 取前 dim 个最大特征值对应的特征向量:
         X = V_dim * sqrt(Λ_dim)

    参数:
      dist_matrix: (N, N) 距离矩阵
      dim: 目标空间维度

    返回:
      positions: (dim, N) 估计位置
    """
    D = np.asarray(dist_matrix, dtype=np.float64)
    N = D.shape[0]
    if D.shape[0] != D.shape[1]:
        raise InverseSourceError("compute_position_from_distance: 距离矩阵必须为方阵")

    D2 = D ** 2
    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ D2 @ J

    # 对称化
    B = 0.5 * (B + B.T)
    eigvals, eigvecs = np.linalg.eigh(B)

    # 取最大的 dim 个特征值
    idx = np.argsort(eigvals)[::-1][:dim]
    Lambda = np.diag(np.maximum(eigvals[idx], 0.0))
    V = eigvecs[:, idx]

    positions = V @ np.sqrt(Lambda)
    return positions.T  # (dim, N)


def map_residuals(x_flat, dim, num_stations, dist_matrix):
    """
    最小二乘残差函数:
      f_k = D_obs(i,j) - ||pos_i - pos_j||
    用于非线性优化修正 MDS 的尺度失真。
    """
    positions = x_flat.reshape((dim, num_stations))
    residuals = []
    n1 = (dim * (dim + 1)) // 2 if num_stations >= dim else (num_stations * (num_stations + 1)) // 2
    n2 = (num_stations * (num_stations - 1)) // 2

    # 固定参考坐标消除平移/旋转自由度
    k = 0
    for city in range(min(dim, num_stations)):
        for d_idx in range(city, min(dim, num_stations)):
            residuals.append(positions[d_idx, city])
            k += 1

    for i in range(num_stations):
        for j in range(i + 1, num_stations):
            d_comp = np.linalg.norm(positions[:, i] - positions[:, j])
            d_obs = dist_matrix[i, j]
            residuals.append(d_obs - d_comp)

    return np.array(residuals, dtype=np.float64)


def inverse_source_location(
    station_positions,
    concentrations,
    Q=1.0,
    D_diff=1.0,
    L=100.0,
    dim=3,
):
    """
    由观测浓度反演气溶胶源位置。

    参数:
      station_positions: (N, dim) 观测站已知位置
      concentrations: (N,) 观测浓度
      Q, D_diff, L: 扩散模型参数
      dim: 空间维度

    返回:
      source_pos: 估计的源位置 (dim,)
      residual_norm: 残差范数
    """
    N = len(concentrations)
    if station_positions.shape[0] != N:
        raise InverseSourceError("inverse_source_location: 站点数与浓度数不匹配")

    # 计算伪距离
    pseudo_d = concentration_to_pseudo_distance(concentrations, Q, D_diff, L)

    # 构造距离矩阵: d_ij = |d_i - d_j| (浓度差异对应的距离差异)
    dist_matrix = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            dist_matrix[i, j] = abs(pseudo_d[i] - pseudo_d[j])
            dist_matrix[j, i] = dist_matrix[i, j]

    # MDS 初始估计
    pos_est = compute_position_from_distance(dist_matrix, dim)

    # 由于尺度不确定，使用加权平均估计源位置
    # 假设: 浓度最高的站点最接近源
    weights = concentrations / np.sum(concentrations)
    source_guess = np.average(station_positions, axis=0, weights=weights)

    # 修正: 根据浓度梯度方向微调
    grad = np.zeros(dim)
    for i in range(N):
        for j in range(i + 1, N):
            if concentrations[i] > concentrations[j]:
                vec = station_positions[i] - station_positions[j]
                d = np.linalg.norm(vec) + 1e-6
                grad += (concentrations[i] - concentrations[j]) * vec / d

    if np.linalg.norm(grad) > 1e-12:
        grad = grad / np.linalg.norm(grad)
        # 步长: 由最大浓度站点的伪距离估计
        step = np.mean(pseudo_d) * 0.1
        source_pos = source_guess - step * grad
    else:
        source_pos = source_guess

    # 计算残差
    predicted = np.zeros(N)
    for i in range(N):
        d = np.linalg.norm(station_positions[i] - source_pos) + 1e-6
        predicted[i] = Q / (4.0 * pi * D_diff * d) * exp(-d / L)

    residual_norm = np.linalg.norm(concentrations - predicted)
    return source_pos, residual_norm


def position_to_distance(city_dim, city_num, positions):
    """
    由位置坐标构造欧氏距离矩阵。
    """
    dist = np.zeros((city_num, city_num), dtype=np.float64)
    for i in range(city_num):
        for j in range(i + 1, city_num):
            d = np.linalg.norm(positions[:, i] - positions[:, j])
            dist[i, j] = d
            dist[j, i] = d
    return dist
