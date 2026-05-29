"""
lagrange_interp.py
================================================================================
多维 Lagrange 插值模块 —— 基于种子项目 638_lagrange_nd

在 LES 的后处理与粒子追踪中，经常需要在非网格点上插值物理量。
本模块提供多维（1D/2D/3D）Lagrange 多项式插值，支持完整与部分多项式空间。

核心物理公式
--------------------------------------------------------------------------------
一维 Lagrange 基函数：
    L_j(x) = Π_{k≠j} (x - x_k) / (x_j - x_k)

n 维张量积 Lagrange 插值：
    f(x) = Σ_{i1,...,in} f(x_{i1},...,x_{in})  L_{i1}(x_1) ... L_{in}(x_n)

在 LES 中，该插值用于：
1. 粒子位置追踪时获取当地速度场
2. 粗-细网格间信息传递（多级网格）
3. 采样点上的统计量计算
"""

import numpy as np


def lagrange_basis_1d(x_nodes, x_target):
    """
    计算一维 Lagrange 基函数在目标点的值。

    参数
    ----------
    x_nodes : np.ndarray, shape (n,)
        插值节点（必须互异）
    x_target : float
        目标点

    返回
    -------
    L : np.ndarray, shape (n,)
        L[j] = L_j(x_target)
    """
    n = len(x_nodes)
    L = np.ones(n, dtype=np.float64)

    for j in range(n):
        for k in range(n):
            if k != j:
                denom = x_nodes[j] - x_nodes[k]
                if abs(denom) < 1e-15:
                    # 退化为 Kronecker delta
                    L = np.zeros(n)
                    L[j] = 1.0
                    return L
                L[j] *= (x_target - x_nodes[k]) / denom

    return L


def lagrange_interp_nd(grid_nodes, values, target):
    """
    多维张量积 Lagrange 插值（支持 1D/2D/3D）。

    参数
    ----------
    grid_nodes : list of np.ndarray
        每个维度的节点列表，例如 [x_nodes, y_nodes, z_nodes]
    values : np.ndarray
        网格上的函数值，形状为 (nx, [ny, [nz]])
    target : tuple or np.ndarray
        目标点坐标 (x, y, z)

    返回
    -------
    interp_val : float
        插值结果
    """
    target = np.atleast_1d(target)
    ndim = len(grid_nodes)

    if len(target) != ndim:
        raise ValueError("lagrange_interp_nd: 目标点维度与网格维度不匹配")

    if values.ndim != ndim:
        raise ValueError(f"lagrange_interp_nd: values.ndim={values.ndim} 不等于 ndim={ndim}")

    # 计算每个维度上的 Lagrange 基
    basis_list = []
    for d in range(ndim):
        Ld = lagrange_basis_1d(grid_nodes[d], target[d])
        basis_list.append(Ld)

    # 根据维度选择求和方式
    if ndim == 1:
        return float(np.dot(basis_list[0], values))
    elif ndim == 2:
        result = 0.0
        nx, ny = values.shape
        for i in range(min(nx, len(basis_list[0]))):
            for j in range(min(ny, len(basis_list[1]))):
                result += basis_list[0][i] * basis_list[1][j] * values[i, j]
        return float(result)
    elif ndim == 3:
        result = 0.0
        nx, ny, nz = values.shape
        for i in range(min(nx, len(basis_list[0]))):
            for j in range(min(ny, len(basis_list[1]))):
                for k in range(min(nz, len(basis_list[2]))):
                    result += (basis_list[0][i] * basis_list[1][j] *
                               basis_list[2][k] * values[i, j, k])
        return float(result)
    else:
        raise NotImplementedError("lagrange_interp_nd: 仅支持 1D/2D/3D")


def interpolate_velocity_to_particles(particles, grid, u_field, v_field, w_field, order=3):
    """
    将速度场插值到拉格朗日粒子位置。

    参数
    ----------
    particles : np.ndarray, shape (n_particle, 3)
        粒子坐标
    grid : tuple of np.ndarray
        (x_grid, y_grid, z_grid)
    u_field, v_field, w_field : np.ndarray
        网格上的速度分量，形状 (nx, ny, nz)
    order : int
        插值阶数

    返回
    -------
    u_p, v_p, w_p : np.ndarray, shape (n_particle,)
    """
    n_particle = particles.shape[0]
    u_p = np.zeros(n_particle, dtype=np.float64)
    v_p = np.zeros(n_particle, dtype=np.float64)
    w_p = np.zeros(n_particle, dtype=np.float64)

    x_grid, y_grid, z_grid = grid
    nx, ny, nz = u_field.shape

    for p in range(n_particle):
        pt = particles[p]

        # 找到最近的节点并选取 order 个邻域节点
        def get_local_nodes(full_grid, x_target, n):
            idx = np.argmin(np.abs(full_grid - x_target))
            half = n // 2
            i0 = max(0, idx - half)
            i1 = min(len(full_grid), i0 + n)
            i0 = max(0, i1 - n)
            return full_grid[i0:i1], i0

        local_x, ix0 = get_local_nodes(x_grid, pt[0], order)
        local_y, iy0 = get_local_nodes(y_grid, pt[1], order)
        local_z, iz0 = get_local_nodes(z_grid, pt[2], order)

        ix1 = min(ix0 + order, nx)
        iy1 = min(iy0 + order, ny)
        iz1 = min(iz0 + order, nz)
        ix0 = max(0, ix1 - order)
        iy0 = max(0, iy1 - order)
        iz0 = max(0, iz1 - order)

        u_slice = u_field[ix0:ix1, iy0:iy1, iz0:iz1]
        v_slice = v_field[ix0:ix1, iy0:iy1, iz0:iz1]
        w_slice = w_field[ix0:ix1, iy0:iy1, iz0:iz1]

        if u_slice.size == 0:
            continue

        local_grid = [local_x, local_y, local_z]
        local_pt = pt.copy()
        # 将粒子坐标限制在局部网格范围内
        for d in range(3):
            local_pt[d] = np.clip(local_pt[d], local_grid[d][0], local_grid[d][-1])

        u_p[p] = lagrange_interp_nd(local_grid, u_slice, local_pt)
        v_p[p] = lagrange_interp_nd(local_grid, v_slice, local_pt)
        w_p[p] = lagrange_interp_nd(local_grid, w_slice, local_pt)

    return u_p, v_p, w_p
