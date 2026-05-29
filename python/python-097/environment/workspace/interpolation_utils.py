"""
interpolation_utils.py

插值与场重构模块。
融合lagrange_interp_1d的拉格朗日插值思想，
用于材料参数在非均匀网格上的插值、场值的后处理采样等。

核心数学:
---------
1. 拉格朗日基函数:
   L_j(x) = ∏_{m≠j} (x - x_m) / (x_j - x_m)

2. 拉格朗日插值:
   P(x) = Σ_j y_j · L_j(x)

3. 在电磁学中的应用:
   - 材料参数 ε(x), μ(x) 的连续化表示
   - 场值从粗网格到细网格的插值
   - 腔体边界上的场采样
"""

import numpy as np


def lagrange_basis_1d(xd, xi):
    """
    计算一维拉格朗日基函数在插值点处的值。
    基于lagrange_basis_1d.m的核心算法。

    Parameters
    ----------
    xd : ndarray, shape (nd,)
        插值节点
    xi : ndarray, shape (ni,)
        求值点

    Returns
    -------
    lb : ndarray, shape (ni, nd)
        lb[i,j] = L_j(xi[i])
    """
    xd = np.asarray(xd).flatten()
    xi = np.asarray(xi).flatten()
    nd = len(xd)
    ni = len(xi)

    lb = np.ones((ni, nd))
    for j in range(nd):
        for m in range(nd):
            if m != j:
                denom = xd[j] - xd[m]
                if abs(denom) < 1e-15:
                    denom = 1e-15
                lb[:, j] *= (xi - xd[m]) / denom

    return lb


def lagrange_value_1d(xd, yd, xi):
    """
    一维拉格朗日插值。
    基于lagrange_value_1d.m的核心算法。

    Parameters
    ----------
    xd : ndarray, shape (nd,)
        数据点横坐标
    yd : ndarray, shape (nd,)
        数据点纵坐标
    xi : ndarray, shape (ni,)
        插值点横坐标

    Returns
    -------
    yi : ndarray, shape (ni,)
        插值结果
    """
    lb = lagrange_basis_1d(xd, xi)
    yd = np.asarray(yd).flatten()
    yi = lb @ yd
    return yi


def lagrange_derivative_1d(xd, yd, xi):
    """
    计算拉格朗日插值函数的导数。
    用于计算场的空间导数。

    dP/dx = Σ_j y_j · dL_j/dx

    Parameters
    ----------
    xd, yd : ndarray
        数据点
    xi : ndarray
        求导点

    Returns
    -------
    dyi : ndarray
        导数值
    """
    xd = np.asarray(xd).flatten()
    yd = np.asarray(yd).flatten()
    xi = np.asarray(xi).flatten()
    nd = len(xd)
    ni = len(xi)

    dy = np.zeros((ni, nd))
    for j in range(nd):
        for i in range(ni):
            dLj = 0.0
            for m in range(nd):
                if m != j:
                    denom = xd[j] - xd[m]
                    if abs(denom) < 1e-15:
                        denom = 1e-15
                    prod = 1.0 / denom
                    for k in range(nd):
                        if k != j and k != m:
                            denom2 = xd[j] - xd[k]
                            if abs(denom2) < 1e-15:
                                denom2 = 1e-15
                            prod *= (xi[i] - xd[k]) / denom2
                    dLj += prod
            dy[i, j] = dLj

    dyi = dy @ yd
    return dyi


def trilinear_interpolation(field, x, y, z, x_grid, y_grid, z_grid):
    """
    三维三线性插值。
    用于从离散网格场获取任意位置的场值。

    Parameters
    ----------
    field : ndarray, shape (nx, ny, nz)
        离散场
    x, y, z : float or ndarray
        目标位置
    x_grid, y_grid, z_grid : ndarray
        网格坐标

    Returns
    -------
    value : float or ndarray
        插值后的场值
    """
    scalar_input = np.isscalar(x)
    if scalar_input:
        x, y, z = np.array([x]), np.array([y]), np.array([z])
    else:
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

    nx, ny, nz = field.shape
    dx = x_grid[1] - x_grid[0] if nx > 1 else 1.0
    dy = y_grid[1] - y_grid[0] if ny > 1 else 1.0
    dz = z_grid[1] - z_grid[0] if nz > 1 else 1.0

    # 归一化坐标
    ix = (x - x_grid[0]) / dx
    iy = (y - y_grid[0]) / dy
    iz = (z - z_grid[0]) / dz

    # 限制在有效范围
    ix = np.clip(ix, 0, nx - 2)
    iy = np.clip(iy, 0, ny - 2)
    iz = np.clip(iz, 0, nz - 2)

    i0 = np.floor(ix).astype(int)
    j0 = np.floor(iy).astype(int)
    k0 = np.floor(iz).astype(int)

    i1 = np.minimum(i0 + 1, nx - 1)
    j1 = np.minimum(j0 + 1, ny - 1)
    k1 = np.minimum(k0 + 1, nz - 1)

    tx = ix - i0
    ty = iy - j0
    tz = iz - k0

    # 三线性插值
    c000 = field[i0, j0, k0]
    c001 = field[i0, j0, k1]
    c010 = field[i0, j1, k0]
    c011 = field[i0, j1, k1]
    c100 = field[i1, j0, k0]
    c101 = field[i1, j0, k1]
    c110 = field[i1, j1, k0]
    c111 = field[i1, j1, k1]

    c00 = c000 * (1 - tz) + c001 * tz
    c01 = c010 * (1 - tz) + c011 * tz
    c10 = c100 * (1 - tz) + c101 * tz
    c11 = c110 * (1 - tz) + c111 * tz

    c0 = c00 * (1 - ty) + c01 * ty
    c1 = c10 * (1 - ty) + c11 * ty

    value = c0 * (1 - tx) + c1 * tx

    if scalar_input:
        return float(value)
    return value


def chebyshev_nodes_1d(n, a=-1.0, b=1.0):
    """
    生成切比雪夫节点，用于高精度多项式插值。

    x_k = (a+b)/2 + (b-a)/2 · cos((2k+1)π/(2n)), k=0,...,n-1

    Parameters
    ----------
    n : int
        节点数量
    a, b : float
        区间端点

    Returns
    -------
    nodes : ndarray
    """
    k = np.arange(n)
    nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2.0 * k + 1.0) * np.pi / (2.0 * n))
    return nodes


def interpolate_material_profile(z_coords, epsilon_profile, z_query, method='lagrange'):
    """
    沿z方向插值材料参数分布。

    Parameters
    ----------
    z_coords : ndarray
        已知材料参数的位置
    epsilon_profile : ndarray
        对应的介电常数
    z_query : ndarray
        查询位置
    method : str
        'lagrange' 或 'linear'

    Returns
    -------
    epsilon_interp : ndarray
    """
    if method == 'linear':
        return np.interp(z_query, z_coords, epsilon_profile)
    elif method == 'lagrange':
        # 使用分段拉格朗日插值（每段4个节点）
        result = np.zeros_like(z_query)
        for i, zq in enumerate(z_query):
            # 找到最近的4个节点
            idx = np.argsort(np.abs(z_coords - zq))[:4]
            idx = np.sort(idx)
            xd = z_coords[idx]
            yd = epsilon_profile[idx]
            result[i] = lagrange_value_1d(xd, yd, np.array([zq]))[0]
        return result
    else:
        raise ValueError(f"未知插值方法: {method}")
