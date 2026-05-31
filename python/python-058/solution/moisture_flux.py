"""
水汽通量辐合模块 (Moisture Flux Convergence Module)

集成种子项目:
- 213_contour_gradient_3d: 3D 梯度计算思想

用于中尺度对流系统中水汽辐合的诊断:
  水汽通量向量: F = ρ * qv * V
  水汽通量辐合: -∇·F = -∇·(ρ qv V)
  在 Cartesian 坐标下:
    -∇·F = -(∂(ρ qv u)/∂x + ∂(ρ qv v)/∂y + ∂(ρ qv w)/∂z)

核心公式 (连续形式):
  ∂qv/∂t + V·∇qv = -qv ∇·V + S_qv
  其中 S_qv 为源汇项 (凝结/蒸发).
"""

import numpy as np
from typing import Tuple


def gradient_2d_centered(field: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    二维场中心差分梯度 (基于 213_contour_gradient_3d 的核心算法,
    去除可视化, 保留纯数值梯度计算).

    内部点: 二阶中心差分
    边界点: 一阶前/后向差分 (带边界鲁棒处理)

    ∂f/∂x 使用 x 方向差分, ∂f/∂y 使用 y 方向差分.
    """
    if field.ndim != 2:
        raise ValueError("Field must be 2D")
    ny, nx = field.shape
    dfdx = np.zeros_like(field)
    dfdy = np.zeros_like(field)

    if nx < 2 or ny < 2:
        return dfdx, dfdy

    # x 方向梯度
    if nx >= 3:
        dfdx[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2.0 * dx)
    dfdx[:, 0] = (field[:, 1] - field[:, 0]) / dx
    dfdx[:, -1] = (field[:, -1] - field[:, -2]) / dx

    # y 方向梯度
    if ny >= 3:
        dfdy[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2.0 * dy)
    dfdy[0, :] = (field[1, :] - field[0, :]) / dy
    dfdy[-1, :] = (field[-1, :] - field[-2, :]) / dy

    return dfdx, dfdy


def gradient_3d_centered(field: np.ndarray, dx: float, dy: float, dz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    三维场中心差分梯度.
    对 (z, y, x) 排列的三维数组计算 (∂f/∂x, ∂f/∂y, ∂f/∂z).
    """
    if field.ndim != 3:
        raise ValueError("Field must be 3D")
    nz, ny, nx = field.shape
    dfdx = np.zeros_like(field)
    dfdy = np.zeros_like(field)
    dfdz = np.zeros_like(field)

    if nx >= 3:
        dfdx[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / (2.0 * dx)
    dfdx[:, :, 0] = (field[:, :, 1] - field[:, :, 0]) / dx
    dfdx[:, :, -1] = (field[:, :, -1] - field[:, :, -2]) / dx

    if ny >= 3:
        dfdy[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / (2.0 * dy)
    dfdy[:, 0, :] = (field[:, 1, :] - field[:, 0, :]) / dy
    dfdy[:, -1, :] = (field[:, -1, :] - field[:, -2, :]) / dy

    if nz >= 3:
        dfdz[1:-1, :, :] = (field[2:, :, :] - field[:-2, :, :]) / (2.0 * dz)
    dfdz[0, :, :] = (field[1, :, :] - field[0, :, :]) / dz
    dfdz[-1, :, :] = (field[-1, :, :] - field[-2, :, :]) / dz

    return dfdx, dfdy, dfdz


def divergence_2d(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    二维速度场的水平散度: ∇h·V = ∂u/∂x + ∂v/∂y.
    """
    dudx, _ = gradient_2d_centered(u, dx, dy)
    _, dvdy = gradient_2d_centered(v, dx, dy)
    return dudx + dvdy


def moisture_flux_convergence(qv: np.ndarray, u: np.ndarray, v: np.ndarray,
                              w: np.ndarray, rho: np.ndarray,
                              dx: float, dy: float, dz: float) -> np.ndarray:
    """
    三维水汽通量辐合场 (单位: kg/(m³·s) 或经适当缩放).

    公式:
      MFC = -∇·(ρ qv V) = -(∂(ρ qv u)/∂x + ∂(ρ qv v)/∂y + ∂(ρ qv w)/∂z)

    输入均为三维数组 (z, y, x).
    """
    if not (qv.shape == u.shape == v.shape == w.shape == rho.shape):
        raise ValueError("All input fields must have the same shape")
    if qv.ndim != 3:
        raise ValueError("Inputs must be 3D")

    flux_x = rho * qv * u
    flux_y = rho * qv * v
    flux_z = rho * qv * w

    dfx_dx, _, _ = gradient_3d_centered(flux_x, dx, dy, dz)
    _, dfy_dy, _ = gradient_3d_centered(flux_y, dx, dy, dz)
    _, _, dfz_dz = gradient_3d_centered(flux_z, dx, dy, dz)

    mfc = -(dfx_dx + dfy_dy + dfz_dz)
    # 边界保护
    mfc = np.where(np.isfinite(mfc), mfc, 0.0)
    return mfc


def moisture_flux_convergence_2d(qv: np.ndarray, u: np.ndarray, v: np.ndarray,
                                 rho: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    二维水汽通量辐合 (仅水平分量).
    """
    if not (qv.shape == u.shape == v.shape == rho.shape):
        raise ValueError("All input fields must have the same shape")
    flux_x = rho * qv * u
    flux_y = rho * qv * v
    dfx_dx, _ = gradient_2d_centered(flux_x, dx, dy)
    _, dfy_dy = gradient_2d_centered(flux_y, dx, dy)
    mfc = -(dfx_dx + dfy_dy)
    return np.where(np.isfinite(mfc), mfc, 0.0)


def laplacian_9point_torus(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    基于 1134_spiral_pde 的 9 点 Laplacian 模板,
    带有周期性 (torus) 边界条件.

    模板系数 (高阶精度):
      (1/6dx²) * [[1, 4, 1], [4, -20, 4], [1, 4, 1]]

    用于中尺度对流系统中的平滑/扩散计算.
    """
    if field.ndim != 2:
        raise ValueError("Field must be 2D")
    ny, nx = field.shape
    if nx < 3 or ny < 3:
        return np.zeros_like(field)

    lap = np.zeros_like(field)
    coeff = 1.0 / (6.0 * dx * dy)

    for j in range(ny):
        for i in range(nx):
            jp = (j + 1) % ny
            jm = (j - 1) % ny
            ip = (i + 1) % nx
            im = (i - 1) % nx

            lap[j, i] = coeff * (
                1.0 * field[jm, im] + 4.0 * field[jm, i] + 1.0 * field[jm, ip]
                + 4.0 * field[j, im] - 20.0 * field[j, i] + 4.0 * field[j, ip]
                + 1.0 * field[jp, im] + 4.0 * field[jp, i] + 1.0 * field[jp, ip]
            )
    return lap


def laplacian_5point(field: np.ndarray, dx: float, dy: float,
                     periodic_x: bool = False, periodic_y: bool = False) -> np.ndarray:
    """
    标准 5 点 Laplacian, 支持 Dirichlet 零边界或周期性边界.
    """
    if field.ndim != 2:
        raise ValueError("Field must be 2D")
    ny, nx = field.shape
    if nx < 3 or ny < 3:
        return np.zeros_like(field)

    lap = np.zeros_like(field)
    dx2 = dx * dx
    dy2 = dy * dy

    # 内部
    lap[1:-1, 1:-1] = (
        (field[1:-1, 2:] - 2.0 * field[1:-1, 1:-1] + field[1:-1, :-2]) / dx2
        + (field[2:, 1:-1] - 2.0 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / dy2
    )

    # 边界处理
    if periodic_x:
        lap[1:-1, 0] = (field[1:-1, 1] - 2.0 * field[1:-1, 0] + field[1:-1, -1]) / dx2 + (field[2:, 0] - 2.0 * field[1:-1, 0] + field[:-2, 0]) / dy2
        lap[1:-1, -1] = (field[1:-1, 0] - 2.0 * field[1:-1, -1] + field[1:-1, -2]) / dx2 + (field[2:, -1] - 2.0 * field[1:-1, -1] + field[:-2, -1]) / dy2
    if periodic_y:
        lap[0, 1:-1] = (field[0, 2:] - 2.0 * field[0, 1:-1] + field[0, :-2]) / dx2 + (field[1, 1:-1] - 2.0 * field[0, 1:-1] + field[-1, 1:-1]) / dy2
        lap[-1, 1:-1] = (field[-1, 2:] - 2.0 * field[-1, 1:-1] + field[-1, :-2]) / dx2 + (field[0, 1:-1] - 2.0 * field[-1, 1:-1] + field[-2, 1:-1]) / dy2

    return lap
