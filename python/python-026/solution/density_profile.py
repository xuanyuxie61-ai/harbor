# -*- coding: utf-8 -*-
"""
density_profile.py

基于 pwc_plot_2d (piecewise constant 2D) 与 area_under_curve 的
等离子体密度剖面构建与积分模块。

原项目 924_pwc_plot_2d 提供了在矩形网格单元上定义分段常值函数的能力；
原项目 017_area_under_curve 提供了基于密集采样的曲线下面积计算。
二者融合后用于:
    1. 在 2D 空间网格上定义分段常数电子密度 n_e(x,y)。
    2. 沿任意路径对密度进行积分（用于射线光学中的光程计算）。
    3. 计算密度梯度（有质动力的源项）。

物理模型:
    - 中心 plateau: n_e = n_0 * f_plateau, 其中 0 <= f_plateau <= 1。
    - 边界层: 采用 Fermi-Dirac 型过渡剖面:
        n_e(r) = n_0 / (1 + exp((r - R_0) / L_s))
      其中 L_s 为密度标长 (scale length)。
    - 随机扰动: 叠加低幅值高斯扰动模拟束间等离子体不均匀性。
"""

import numpy as np


def piecewise_constant_density_2d(xc, yc, nxc, nyc, density_func):
    """
    在二维矩形网格单元上生成分段常数密度场。

    原 pwc_plot_2d 的核心思想被保留：每个单元 [xc[i], xc[i+1]] × [yc[j], yc[j+1]]
    具有单一密度值，取单元中心处的解析值。

    Parameters
    ----------
    xc, yc : ndarray
        网格断点坐标，长度分别为 nxc+1 和 nyc+1。
    nxc, nyc : int
        x 和 y 方向的单元数。
    density_func : callable
        密度函数 n_e(x, y)，接受标量或 ndarray 输入。

    Returns
    -------
    ne_cells : ndarray, shape (nxc, nyc)
        每个单元中心的电子密度 [m^{-3}]。
    xc, yc : ndarray
        网格断点。
    """
    if len(xc) != nxc + 1 or len(yc) != nyc + 1:
        raise ValueError("断点数组长度必须与单元数匹配。")
    if np.any(np.diff(xc) <= 0) or np.any(np.diff(yc) <= 0):
        raise ValueError("断点坐标必须严格递增。")

    ne_cells = np.zeros((nxc, nyc), dtype=float)
    for i in range(nxc):
        x_mid = 0.5 * (xc[i] + xc[i + 1])
        for j in range(nyc):
            y_mid = 0.5 * (yc[j] + yc[j + 1])
            val = density_func(x_mid, y_mid)
            if val < 0:
                val = 0.0
            ne_cells[i, j] = val

    return ne_cells, xc, yc


def icf_density_profile(x, y, n0, R0, Ls, f_plateau=0.3,
                         perturbation_amplitude=0.0, perturbation_scale=1e-6):
    """
    惯性约束聚变靶丸外覆等离子体的密度分布模型。

    采用径向 Fermi-Dirac 型过渡:
        n_e(r) = n_0 * f_plateau + n_0 * (1 - f_plateau) / (1 + exp((r - R_0)/L_s))

    其中 r = sqrt(x^2 + y^2)。

    叠加随机高斯扰动:
        δn = perturbation_amplitude * n_0 * exp(-(x^2+y^2)/(2*perturbation_scale^2)) * ξ
        ξ ~ N(0, 1)

    Parameters
    ----------
    x, y : float or ndarray
        空间坐标 [m]。
    n0 : float
        峰值密度 [m^{-3}]。
    R0 : float
        特征半径 [m]。
    Ls : float
        密度标长 [m]。
    f_plateau : float, optional
        plateau 密度比值, 默认 0.3。
    perturbation_amplitude : float, optional
        扰动相对幅值, 默认 0.0。
    perturbation_scale : float, optional
        扰动空间尺度 [m], 默认 1e-6。

    Returns
    -------
    ne : float or ndarray
        电子密度 [m^{-3}]。
    """
    if n0 < 0 or R0 <= 0 or Ls <= 0:
        raise ValueError("n0 必须非负，R0 和 Ls 必须为正。")

    r = np.sqrt(x**2 + y**2)
    # Fermi-Dirac 型过渡
    ne = n0 * f_plateau + n0 * (1.0 - f_plateau) / (1.0 + np.exp((r - R0) / Ls))

    # 边界保护：避免数值溢出
    ne = np.clip(ne, 0.0, n0 * 1.1)

    # 叠加低幅值随机扰动
    if perturbation_amplitude > 0.0:
        if np.isscalar(x):
            rng = np.random.default_rng(seed=int((x + y) * 1e12) % 2**31)
            xi = rng.standard_normal()
        else:
            xi = np.random.default_rng(42).standard_normal(size=np.broadcast(x, y).shape)
        delta = perturbation_amplitude * n0 * np.exp(-r**2 / (2.0 * perturbation_scale**2)) * xi
        ne = ne + delta
        ne = np.clip(ne, 0.0, None)

    return ne


def density_gradient_pwc(ne_cells, xc, yc):
    """
    基于分段常数密度场计算单元界面上的密度梯度近似。

    使用中心差分:
        ∂n/∂x|_{i+1/2, j} ≈ (n_{i+1,j} - n_{i,j}) / (x_{i+1} - x_i)
        ∂n/∂y|_{i, j+1/2} ≈ (n_{i,j+1} - n_{i,j}) / (y_{j+1} - y_j)

    Parameters
    ----------
    ne_cells : ndarray, shape (nxc, nyc)
        单元中心密度。
    xc, yc : ndarray
        网格断点。

    Returns
    -------
    grad_x : ndarray, shape (nxc-1, nyc)
        x 方向梯度在垂直界面上的值。
    grad_y : ndarray, shape (nxc, nyc-1)
        y 方向梯度在水平界面上的值。
    """
    nxc, nyc = ne_cells.shape
    if nxc < 2 or nyc < 2:
        raise ValueError("密度场至少需要 2x2 个单元以计算梯度。")

    dx = np.diff(xc)
    dy = np.diff(yc)

    grad_x = np.zeros((nxc - 1, nyc), dtype=float)
    for i in range(nxc - 1):
        dx_safe = max(dx[i], 1e-20)
        grad_x[i, :] = (ne_cells[i + 1, :] - ne_cells[i, :]) / dx_safe

    grad_y = np.zeros((nxc, nyc - 1), dtype=float)
    for j in range(nyc - 1):
        dy_safe = max(dy[j], 1e-20)
        grad_y[:, j] = (ne_cells[:, j + 1] - ne_cells[:, j]) / dy_safe

    return grad_x, grad_y


def integrate_density_along_path(ne_cells, xc, yc, path_points):
    """
    沿折线路径对分段常数密度场进行数值积分（光程近似）。

    基于 area_under_curve 的采样思想：
    对路径进行密集采样，在每个采样点通过双线性插值获取密度，
    然后使用梯形法则积分:
        ∫ n_e ds ≈ Σ (n_i + n_{i+1})/2 * Δs_i

    Parameters
    ----------
    ne_cells : ndarray, shape (nxc, nyc)
        单元中心密度。
    xc, yc : ndarray
        网格断点。
    path_points : ndarray, shape (M, 2)
        路径上的有序点列 [(x_0, y_0), ..., (x_{M-1}, y_{M-1})]。

    Returns
    -------
    integral : float
        沿路径的密度积分 [m^{-2}]。
    ds_total : float
        路径总长度 [m]。
    """
    if path_points.ndim != 2 or path_points.shape[1] != 2:
        raise ValueError("path_points 必须是形状为 (M, 2) 的数组。")
    M = path_points.shape[0]
    if M < 2:
        return 0.0, 0.0

    # 密集采样: 每段至少 50 个采样点
    n_samples_per_segment = 50
    sampled_points = []
    for k in range(M - 1):
        p0 = path_points[k]
        p1 = path_points[k + 1]
        t = np.linspace(0.0, 1.0, n_samples_per_segment + 1)
        seg = np.outer(1.0 - t, p0) + np.outer(t, p1)
        if k > 0:
            seg = seg[1:]
        sampled_points.append(seg)
    sampled = np.vstack(sampled_points)

    # 在采样点上插值密度
    ne_sampled = bilinear_interpolate_density(ne_cells, xc, yc, sampled[:, 0], sampled[:, 1])

    # 梯形法则积分
    ds = np.sqrt(np.sum(np.diff(sampled, axis=0)**2, axis=1))
    integral = np.sum(0.5 * (ne_sampled[:-1] + ne_sampled[1:]) * ds)
    ds_total = np.sum(ds)

    return integral, ds_total


def bilinear_interpolate_density(ne_cells, xc, yc, xq, yq):
    """
    对分段常数密度场进行双线性插值（将单元中心值视为节点值）。

    Parameters
    ----------
    ne_cells : ndarray, shape (nxc, nyc)
        单元中心密度。
    xc, yc : ndarray
        网格断点，长度分别为 nxc+1 和 nyc+1。
    xq, yq : float or ndarray
        查询点坐标。

    Returns
    -------
    ne_q : float or ndarray
        查询点处的插值密度。
    """
    xq = np.asarray(xq, dtype=float)
    yq = np.asarray(yq, dtype=float)

    nxc, nyc = ne_cells.shape
    # 构造单元中心坐标
    x_centers = 0.5 * (xc[:-1] + xc[1:])
    y_centers = 0.5 * (yc[:-1] + yc[1:])

    # 将查询点限制在网格范围内
    xq = np.clip(xq, x_centers[0], x_centers[-1])
    yq = np.clip(yq, y_centers[0], y_centers[-1])

    # 找到所在的单元索引
    ix = np.searchsorted(x_centers, xq, side='right') - 1
    iy = np.searchsorted(y_centers, yq, side='right') - 1
    ix = np.clip(ix, 0, nxc - 2)
    iy = np.clip(iy, 0, nyc - 2)

    # 局部坐标
    dx = x_centers[ix + 1] - x_centers[ix]
    dy = y_centers[iy + 1] - y_centers[iy]
    dx_safe = np.where(dx > 0, dx, 1.0)
    dy_safe = np.where(dy > 0, dy, 1.0)
    tx = (xq - x_centers[ix]) / dx_safe
    ty = (yq - y_centers[iy]) / dy_safe

    # 双线性插值
    ne_q = (1.0 - tx) * (1.0 - ty) * ne_cells[ix, iy] + \
           tx * (1.0 - ty) * ne_cells[ix + 1, iy] + \
           (1.0 - tx) * ty * ne_cells[ix, iy + 1] + \
           tx * ty * ne_cells[ix + 1, iy + 1]

    return ne_q


def total_plasma_mass(ne_cells, xc, yc, ion_mass=2.5e-26):
    """
    计算分段常数密度场对应的等离子体总质量（面密度）。

    公式:
        M = Σ_{i,j} n_{e,ij} * Δx_i * Δy_j * m_i / Z_eff

    假设完全电离且 Z_eff ≈ 1（质子-电子近似）。

    Parameters
    ----------
    ne_cells : ndarray
        单元中心密度 [m^{-3}]。
    xc, yc : ndarray
        网格断点。
    ion_mass : float, optional
        离子质量 [kg], 默认为氘核质量近似。

    Returns
    -------
    mass : float
        总质量 [kg/m] (沿第三维假设为单位长度)。
    """
    dx = np.diff(xc)
    dy = np.diff(yc)
    volumes = np.outer(dx, dy)  # (nxc, nyc)
    mass = np.sum(ne_cells * volumes) * ion_mass
    return mass
