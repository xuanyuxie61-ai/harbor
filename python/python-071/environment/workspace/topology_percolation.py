# -*- coding: utf-8 -*-
"""
topology_percolation.py
湍流相干结构的渗流拓扑分析模块

融合来源:
- 865_percolation_simulation: 渗流模拟与连通分量分析

功能:
- 基于速度梯度或涡量阈值识别湍流相干结构
- 使用渗流理论分析结构的连通性和尺度分布
- 计算渗流阈值、跨越概率、分形维数
- 分析湍流间歇性和能量级串的拓扑特征

数学背景:
  湍流相干结构（如涡管、涡片）可通过速度梯度张量的特征值或
  Q-准则、Lambda2-准则识别：

  Q-准则:
    Q = 0.5 * (||Omega||^2 - ||S||^2)
    其中 Omega 为旋转率张量，S 为应变率张量。
    Q > 0 表示涡量主导的区域（涡核）。

  渗流理论:
    对于一个 m x n x p 的网格，每个格点以概率 p 被"占据"。
    当 p > p_c（临界阈值）时，出现跨越整个系统的连通团簇。
    对于三维简单立方格子，p_c ≈ 0.3116。

  连通分量分析:
    使用 BFS/DFS 标记所有连通团簇，统计团簇大小分布 n(s) ~ s^(-tau)
    其中 tau 为 Fisher 指数，与维度相关。
"""

import numpy as np


def components_3d(A):
    """
    对三维二值数组进行连通分量标记（6-连通）。
    融合自 865_percolation_simulation 的 components_2d。

    参数:
      A: (nx, ny, nz) 二值数组（0 或 1）

    返回:
      C: 标记数组，每个连通分量有唯一标签
      component_num: 连通分量个数
    """
    A = np.asarray(A, dtype=int)
    nx, ny, nz = A.shape
    C = np.zeros_like(A, dtype=int)
    component_index = 0

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if A[i, j, k] != 0 and C[i, j, k] == 0:
                    component_index += 1
                    plist = [(i, j, k)]

                    while plist:
                        ci, cj, ck = plist.pop()
                        if C[ci, cj, ck] != 0:
                            continue
                        C[ci, cj, ck] = component_index

                        # 6-连通邻居
                        neighbors = []
                        if ci > 0 and A[ci - 1, cj, ck] != 0 and C[ci - 1, cj, ck] == 0:
                            neighbors.append((ci - 1, cj, ck))
                        if ci < nx - 1 and A[ci + 1, cj, ck] != 0 and C[ci + 1, cj, ck] == 0:
                            neighbors.append((ci + 1, cj, ck))
                        if cj > 0 and A[ci, cj - 1, ck] != 0 and C[ci, cj - 1, ck] == 0:
                            neighbors.append((ci, cj - 1, ck))
                        if cj < ny - 1 and A[ci, cj + 1, ck] != 0 and C[ci, cj + 1, ck] == 0:
                            neighbors.append((ci, cj + 1, ck))
                        if ck > 0 and A[ci, cj, ck - 1] != 0 and C[ci, cj, ck - 1] == 0:
                            neighbors.append((ci, cj, ck - 1))
                        if ck < nz - 1 and A[ci, cj, ck + 1] != 0 and C[ci, cj, ck + 1] == 0:
                            neighbors.append((ci, cj, ck + 1))

                        plist.extend(neighbors)

    return C, component_index


def q_criterion(u, v, w, dx, dy, dz):
    """
    计算 Q-准则识别涡结构。

    数学公式:
      Q = 0.5 * (||Omega||^2 - ||S||^2)
      其中:
        Omega_{ij} = 0.5 * (du_i/dx_j - du_j/dx_i)  (旋转率张量)
        S_{ij} = 0.5 * (du_i/dx_j + du_j/dx_i)      (应变率张量)
        ||A||^2 = sum_{i,j} A_{ij} * A_{ij}

    参数:
      u, v, w: 速度场
      dx, dy, dz: 网格间距

    返回:
      Q: Q-准则场
    """
    def ddx(f):
        result = np.zeros_like(f)
        result[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2.0 * dx)
        return result

    def ddy(f):
        result = np.zeros_like(f)
        result[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2.0 * dy)
        return result

    def ddz(f):
        result = np.zeros_like(f)
        result[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2.0 * dz)
        return result

    dudx = ddx(u)
    dudy = ddy(u)
    dudz = ddz(u)
    dvdx = ddx(v)
    dvdy = ddy(v)
    dvdz = ddz(v)
    dwdx = ddx(w)
    dwdy = ddy(w)
    dwdz = ddz(w)

    # 旋转率张量
    O12 = 0.5 * (dudy - dvdx)
    O13 = 0.5 * (dudz - dwdx)
    O23 = 0.5 * (dvdz - dwdy)

    # 应变率张量
    S11 = dudx
    S12 = 0.5 * (dudy + dvdx)
    S13 = 0.5 * (dudz + dwdx)
    S22 = dvdy
    S23 = 0.5 * (dvdz + dwdy)
    S33 = dwdz

    # Q = 0.5 * (||Omega||^2 - ||S||^2)
    norm_Omega_sq = 2.0 * (O12 ** 2 + O13 ** 2 + O23 ** 2)
    norm_S_sq = S11 ** 2 + S22 ** 2 + S33 ** 2 + 2.0 * S12 ** 2 + 2.0 * S13 ** 2 + 2.0 * S23 ** 2

    Q = 0.5 * (norm_Omega_sq - norm_S_sq)
    return Q


def percolation_analysis_3d(field, thresholds=None):
    """
    对湍流场进行多阈值渗流分析。
    融合自 865_percolation_simulation。

    参数:
      field: 三维标量场（如 Q-准则）
      thresholds: 阈值列表，若 None 则自动生成

    返回:
      results: 各阈值下的渗流统计结果字典列表
    """
    field = np.asarray(field, dtype=float)
    nx, ny, nz = field.shape

    if thresholds is None:
        qmin, qmax = np.min(field), np.max(field)
        thresholds = np.linspace(qmin, qmax, 11)

    results = []

    for thresh in thresholds:
        # 二值化
        binary = (field > thresh).astype(int)

        # 连通分量分析
        labels, n_components = components_3d(binary)

        # 统计各分量大小
        component_sizes = []
        for c in range(1, n_components + 1):
            size = np.sum(labels == c)
            if size > 0:
                component_sizes.append(size)

        component_sizes = np.array(component_sizes, dtype=int)

        # 占据概率
        p_occupied = np.mean(binary)

        # 检查跨越团簇（x, y, z 三个方向）
        spanning_x = 0
        spanning_y = 0
        spanning_z = 0

        for c in range(1, n_components + 1):
            mask = (labels == c)
            if np.any(mask[0, :, :]) and np.any(mask[-1, :, :]):
                spanning_x += 1
            if np.any(mask[:, 0, :]) and np.any(mask[:, -1, :]):
                spanning_y += 1
            if np.any(mask[:, :, 0]) and np.any(mask[:, :, -1]):
                spanning_z += 1

        # 分形维数估计（盒计数法简化）
        if len(component_sizes) > 0:
            max_size = np.max(component_sizes)
            # 用最大团簇的等效半径估计
            r_eq = (max_size / (4.0 / 3.0 * np.pi)) ** (1.0 / 3.0)
            fractal_dim = np.log(max_size) / (np.log(r_eq) + 1e-15) if r_eq > 1 else 0.0
        else:
            fractal_dim = 0.0

        results.append({
            'threshold': thresh,
            'p_occupied': p_occupied,
            'n_components': n_components,
            'mean_size': float(np.mean(component_sizes)) if len(component_sizes) > 0 else 0.0,
            'max_size': int(np.max(component_sizes)) if len(component_sizes) > 0 else 0,
            'spanning_x': spanning_x,
            'spanning_y': spanning_y,
            'spanning_z': spanning_z,
            'fractal_dim': float(fractal_dim)
        })

    return results


def energy_cascade_topology(u, v, w, dx, dy, dz):
    """
    分析湍流能量级串的拓扑特征。

    物理模型:
      通过多尺度滤波分析涡量场的拓扑结构变化，
      研究能量从大尺度到小尺度传递过程中的结构演化。

    返回:
      scales: 分析尺度列表
      topology_metrics: 各尺度下的拓扑度量
    """
    nx, ny, nz = u.shape
    scales = [1, 2, 4]
    topology_metrics = []

    for scale in scales:
        if nx // scale < 3 or ny // scale < 3 or nz // scale < 3:
            continue

        # 粗粒化滤波
        u_coarse = u[::scale, ::scale, ::scale]
        v_coarse = v[::scale, ::scale, ::scale]
        w_coarse = w[::scale, ::scale, ::scale]

        # 计算 Q-准则
        Q = q_criterion(u_coarse, v_coarse, w_coarse,
                        dx * scale, dy * scale, dz * scale)

        # 渗流分析
        perc = percolation_analysis_3d(Q, thresholds=[np.percentile(Q, 75)])

        topology_metrics.append({
            'scale': scale,
            'n_components': perc[0]['n_components'],
            'mean_size': perc[0]['mean_size'],
            'fractal_dim': perc[0]['fractal_dim'],
            'spanning': perc[0]['spanning_x'] + perc[0]['spanning_y'] + perc[0]['spanning_z']
        })

    return scales, topology_metrics
