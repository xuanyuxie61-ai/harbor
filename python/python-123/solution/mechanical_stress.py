"""
mechanical_stress.py

肿瘤微环境固体应力（Solid Stress）计算模块

本模块融合以下种子项目的核心算法：
  - 172_chladni_figures: 双调和算子（Biharmonic Operator）与广义特征值问题

科学背景：
  实体肿瘤内部的固体应力由细胞增殖、ECM（细胞外基质）沉积和血管塌陷共同产生。
  该应力场可用薄板弯曲的双调和方程描述：

    nabla^4 w - lambda * w = 0

  其中 w(x,y) 为位移场，lambda 为特征值，与材料常数 mu（泊松比）相关。

  在离散化形式下，双调和算子通过 5 点 Laplacian 的复合构造：
    L = D (5-point stencil)
    A = N * L   (N 为修正后的 Laplacian)

  边界条件（自由边界 / 固支边界）通过 ghost points 消除：
    A0 = A(phys,phys) - A(phys,ghost) * inv(A(ghost,ghost)) * A(ghost,phys)

  最终求解广义特征值问题：
    A0 * V = Lambda * B0 * V

  固体应力与位移的二阶导数相关：
    sigma_xx = E/(1-mu^2) * (w_xx + mu * w_yy)
    sigma_yy = E/(1-mu^2) * (w_yy + mu * w_xx)
    sigma_xy = E/(2*(1+mu)) * w_xy

  冯·米塞斯等效应力：
    sigma_vm = sqrt(sigma_xx^2 - sigma_xx*sigma_yy + sigma_yy^2 + 3*sigma_xy^2)
"""

import numpy as np
from typing import Tuple


def build_5point_laplacian_2d(nx: int, ny: int, hx: float, hy: float) -> np.ndarray:
    """
    构建二维 5 点 Laplacian 稀疏矩阵的稠密表示（用于中小规模问题）。

     stencil:
             (1/hy^2)
                |
        (1/hx^2) -2*(1/hx^2+1/hy^2) (1/hx^2)
                |
             (1/hy^2)

    参数:
        nx, ny: x 和 y 方向的内部格点数
        hx, hy: 网格步长

    返回:
        L: (nx*ny, nx*ny) 稠密矩阵
    """
    N = nx * ny
    L = np.zeros((N, N))
    cx = 1.0 / (hx * hx)
    cy = 1.0 / (hy * hy)
    c0 = -2.0 * (cx + cy)

    def idx(i, j):
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            k = idx(i, j)
            L[k, k] = c0
            if i > 0:
                L[k, idx(i - 1, j)] = cx
            if i < nx - 1:
                L[k, idx(i + 1, j)] = cx
            if j > 0:
                L[k, idx(i, j - 1)] = cy
            if j < ny - 1:
                L[k, idx(i, j + 1)] = cy

    return L


def biharmonic_stress_operator(
    nx: int, ny: int, hx: float, hy: float, mu: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构建薄板双调和应力算子并求解特征模态。

    控制方程:
        nabla^4 w = lambda * w   on Omega
        w = 0,  d^2 w / dn dt = 0  on boundary

    离散化后转化为广义特征值问题:
        A0 * phi = lambda * B0 * phi

    参数:
        nx, ny: 网格点数
        hx, hy: 步长
        mu: 材料泊松比 (0 < mu < 1)

    返回:
        eigenvalues: (k,) 前 k 个特征值
        eigenvectors: (nx*ny, k) 特征向量
        stress_vm: (nx*ny,) 冯·米塞斯等效应力分布
    """
    if not (0.0 < mu < 1.0):
        raise ValueError("biharmonic_stress_operator: mu 必须在 (0,1)")
    if nx < 3 or ny < 3:
        raise ValueError("biharmonic_stress_operator: 网格尺寸至少为 3x3")

    N = nx * ny
    L = build_5point_laplacian_2d(nx, ny, hx, hy)

    # 简化的双调和算子：A = L^2 (Laplacian 的复合)
    # 在实际薄板问题中需做边界修正，这里用 L^2 近似内部区域
    A = L @ L

    # 质量矩阵 B (单位矩阵，简化)
    B = np.eye(N)

    # 边界修正：对最外圈的行/列施加 Dirichlet 条件
    def idx(i, j):
        return i * ny + j

    boundary_indices = set()
    for i in range(nx):
        boundary_indices.add(idx(i, 0))
        boundary_indices.add(idx(i, ny - 1))
    for j in range(ny):
        boundary_indices.add(idx(0, j))
        boundary_indices.add(idx(nx - 1, j))

    boundary_indices = sorted(list(boundary_indices))
    interior_indices = [k for k in range(N) if k not in boundary_indices]

    # 提取子矩阵
    A_ii = A[np.ix_(interior_indices, interior_indices)]
    B_ii = B[np.ix_(interior_indices, interior_indices)]

    # 求解小规模特征值问题
    k = min(6, len(interior_indices))
    eigenvalues, eigenvectors = np.linalg.eigh(A_ii)

    # 排序
    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx][:k]
    eigenvectors = eigenvectors[:, sort_idx][:, :k]

    # 映射回完整网格
    full_vectors = np.zeros((N, k))
    full_vectors[interior_indices, :] = eigenvectors

    # 计算冯·米塞斯等效应力（基于第一模态）
    phi = full_vectors[:, 0]
    phi_grid = phi.reshape((nx, ny))

    # 数值差分计算二阶导数
    w_xx = np.zeros_like(phi_grid)
    w_yy = np.zeros_like(phi_grid)
    w_xy = np.zeros_like(phi_grid)

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            w_xx[i, j] = (phi_grid[i + 1, j] - 2.0 * phi_grid[i, j] +
                          phi_grid[i - 1, j]) / (hx ** 2)
            w_yy[i, j] = (phi_grid[i, j + 1] - 2.0 * phi_grid[i, j] +
                          phi_grid[i, j - 1]) / (hy ** 2)
            w_xy[i, j] = (phi_grid[i + 1, j + 1] - phi_grid[i + 1, j - 1] -
                          phi_grid[i - 1, j + 1] + phi_grid[i - 1, j - 1]) / (4.0 * hx * hy)

    E = 1.0  # 归一化杨氏模量
    denom = 1.0 - mu ** 2
    denom = max(denom, 1e-15)

    sigma_xx = E / denom * (w_xx + mu * w_yy)
    sigma_yy = E / denom * (w_yy + mu * w_xx)
    sigma_xy = E / (2.0 * (1.0 + mu)) * w_xy

    stress_vm = np.sqrt(np.maximum(
        sigma_xx ** 2 - sigma_xx * sigma_yy + sigma_yy ** 2 + 3.0 * sigma_xy ** 2,
        0.0
    ))

    return eigenvalues, full_vectors, stress_vm.ravel()


def compute_stress_induced_apoptosis(
    stress_vm: np.ndarray, threshold: float = 0.5, steepness: float = 10.0
) -> np.ndarray:
    """
    计算应力诱导的凋亡概率。

    采用 Sigmoid 响应函数：
        P_apop(sigma) = 1 / (1 + exp(-steepness * (sigma - threshold)))

    参数:
        stress_vm: 冯·米塞斯应力数组
        threshold: 凋亡阈值应力
        steepness: Sigmoid 斜率

    返回:
        prob_apop: 凋亡概率数组
    """
    stress_vm = np.asarray(stress_vm, dtype=float)
    z = steepness * (stress_vm - threshold)
    # 数值稳定性截断
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def compute_tumor_stress_metrics(stress_vm: np.ndarray) -> dict:
    """
    计算肿瘤应力场的统计指标。

    返回字典包含：
        - max_stress: 最大冯·米塞斯应力
        - mean_stress: 平均应力
        - std_stress: 应力标准差
        - high_stress_fraction: 高应力区域比例 (> 0.5 * max)
    """
    if stress_vm.size == 0:
        return {
            "max_stress": 0.0,
            "mean_stress": 0.0,
            "std_stress": 0.0,
            "high_stress_fraction": 0.0,
        }

    max_s = float(np.max(stress_vm))
    mean_s = float(np.mean(stress_vm))
    std_s = float(np.std(stress_vm))
    frac = float(np.mean(stress_vm > 0.5 * max_s))

    return {
        "max_stress": max_s,
        "mean_stress": mean_s,
        "std_stress": std_s,
        "high_stress_fraction": frac,
    }
