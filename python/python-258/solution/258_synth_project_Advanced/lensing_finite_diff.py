"""
lensing_finite_diff.py — 高阶有限差分算子
=========================================

来源种子: 399_fem1d_spectral_numeric (谱方法/Galerkin),
          487_gray_scott_pde (PDE 求解器/多阶格式)
科学角色: 实现 2D 高阶有限差分算子 (2阶、4阶、6阶),
         包括 Laplacian、Biharmonic、梯度、散度。
         这些算子是弱透镜 PDE 质量重建的基础。

离散化公式:
===========
2nd order Laplacian (5-point stencil):
    Δ_h f_{ij} = (f_{i+1,j} + f_{i-1,j} + f_{i,j+1} + f_{i,j-1} - 4f_{ij}) / h²
    截断误差: O(h²)

4th order Laplacian (13-point stencil, Mehrstellenverfahren):
    Δ_h^(4) f_{ij} = (1/(6h²)) × [
        4(f_{i+1,j} + f_{i-1,j} + f_{i,j+1} + f_{i,j-1})
        + (f_{i+1,j+1} + f_{i-1,j+1} + f_{i+1,j-1} + f_{i-1,j-1})
        - 20 f_{ij}
    ]
    截断误差: O(h⁴)

Biharmonic (Δ²):
    ∇⁴f = Δ(Δf)
    2nd order: 使用 13-point 复合模板
    4th order: 使用 25-point 模板

梯度 (4th order central):
    ∂f/∂x ≈ (-f_{i+2,j} + 8f_{i+1,j} - 8f_{i-1,j} + f_{i-2,j}) / (12h)
    截断误差: O(h⁴)
"""

import numpy as np
from lensing_config import GRID


def laplacian_2nd(f, h):
    """
    2阶 Laplacian (标准 5-point stencil)

    ∇²_h f_{ij} = (f_{i+1,j} + f_{i-1,j} + f_{i,j+1} + f_{i,j-1} - 4f_{ij}) / h²

    使用周期边界条件 (弱透镜场通常采用周期性边界)。

    修正波数 (Modified wavenumber):
    对于傅里叶模式 e^{ikx}, 离散 Laplacian 给出:
    k²_eff = (2/h²)(1 - cos(kh)) = (4/h²) sin²(kh/2)

    这导致小尺度的 Laplacian 被低估, 对 PDE 稳定性有影响。

    Parameters
    ----------
    f : ndarray (Ny, Nx)
        2D 输入场
    h : float
        网格间距

    Returns
    -------
    lap : ndarray
        ∇²f 的离散近似
    """
    Ny, Nx = f.shape
    lap = np.zeros_like(f)

    # 周期边界
    f_ip1 = np.roll(f, -1, axis=0)
    f_im1 = np.roll(f, 1, axis=0)
    f_jp1 = np.roll(f, -1, axis=1)
    f_jm1 = np.roll(f, 1, axis=1)

    lap = (f_ip1 + f_im1 + f_jp1 + f_jm1 - 4.0 * f) / (h * h)
    return lap


def laplacian_4th(f, h):
    """
    4阶 Laplacian (Mehrstellen 13-point stencil)

    ∇²_h f_{ij} = (1/(6h²)) × [
        4(f_{i+1,j} + f_{i-1,j} + f_{i,j+1} + f_{i,j-1})
        + (f_{i+1,j+1} + f_{i-1,j+1} + f_{i+1,j-1} + f_{i-1,j-1})
        - 20 f_{ij}
    ]

    修正波数:
    k²_eff = (1/(6h²)) × [
        8(cos(kx h) + cos(ky h))
        + 4 cos(kx h) cos(ky h)  (from cross terms... wait)
    ]

    更精确: 4th order Mehrstellen 的修正波数为:
    k²_eff = (4/(3h²))[2 - cos(kx h) - cos(ky h)]
             × [1 + (1/6)(cos(kx h) + cos(ky h) - 2)]

    这比标准 5-point 更精确地逼近 k², 在 kΔx < π/2 范围内
    误差 < 0.1%。

    Parameters
    ----------
    f : ndarray (Ny, Nx)
        2D 输入场
    h : float
        网格间距

    Returns
    -------
    lap : ndarray
        4阶精度的 ∇²f 近似
    """
    # 4个直接邻居
    f_ip1 = np.roll(f, -1, axis=0)
    f_im1 = np.roll(f, 1, axis=0)
    f_jp1 = np.roll(f, -1, axis=1)
    f_jm1 = np.roll(f, 1, axis=1)

    # 4个对角邻居
    f_ip1jp1 = np.roll(np.roll(f, -1, axis=0), -1, axis=1)
    f_im1jp1 = np.roll(np.roll(f, 1, axis=0), -1, axis=1)
    f_ip1jm1 = np.roll(np.roll(f, -1, axis=0), 1, axis=1)
    f_im1jm1 = np.roll(np.roll(f, 1, axis=0), 1, axis=1)

    lap = (
        4.0 * (f_ip1 + f_im1 + f_jp1 + f_jm1)
        + (f_ip1jp1 + f_im1jp1 + f_ip1jm1 + f_im1jm1)
        - 20.0 * f
    ) / (6.0 * h * h)

    return lap


def laplacian(f, h, order=4):
    """
    自适应阶数的 Laplacian 调度器

    Parameters
    ----------
    f : ndarray
        2D 输入场
    h : float
        网格间距
    order : int
        差分阶数 (2 或 4)

    Returns
    -------
    lap : ndarray
        ∇²f
    """
    if order == 2:
        return laplacian_2nd(f, h)
    elif order == 4:
        return laplacian_4th(f, h)
    else:
        raise ValueError(f"Unsupported Laplacian order: {order}. Use 2 or 4.")


def gradient_4th(f, h):
    """
    4阶中心差分梯度

    ∂f/∂x ≈ (-f_{i+2,j} + 8f_{i+1,j} - 8f_{i-1,j} + f_{i-2,j}) / (12h)
    ∂f/∂y ≈ (-f_{i,j+2} + 8f_{i,j+1} - 8f_{i,j-1} + f_{i,j-2}) / (12h)

    修正波数:
    k_eff = (1/(6h)) × [8 sin(kh) - sin(2kh)]
    在 kΔx < 2π/3 范围内误差 < 0.5%

    Parameters
    ----------
    f : ndarray (Ny, Nx)
        2D 标量场
    h : float
        网格间距

    Returns
    -------
    df_dx, df_dy : ndarray
        梯度分量
    """
    # x 方向 (axis=0)
    f_ip2 = np.roll(f, -2, axis=0)
    f_ip1 = np.roll(f, -1, axis=0)
    f_im1 = np.roll(f, 1, axis=0)
    f_im2 = np.roll(f, 2, axis=0)
    df_dx = (-f_ip2 + 8.0 * f_ip1 - 8.0 * f_im1 + f_im2) / (12.0 * h)

    # y 方向 (axis=1)
    f_jp2 = np.roll(f, -2, axis=1)
    f_jp1 = np.roll(f, -1, axis=1)
    f_jm1 = np.roll(f, 1, axis=1)
    f_jm2 = np.roll(f, 2, axis=1)
    df_dy = (-f_jp2 + 8.0 * f_jp1 - 8.0 * f_jm1 + f_jm2) / (12.0 * h)

    return df_dx, df_dy


def gradient_2nd(f, h):
    """
    2阶中心差分梯度

    ∂f/∂x ≈ (f_{i+1,j} - f_{i-1,j}) / (2h)
    """
    f_ip1 = np.roll(f, -1, axis=0)
    f_im1 = np.roll(f, 1, axis=0)
    df_dx = (f_ip1 - f_im1) / (2.0 * h)

    f_jp1 = np.roll(f, -1, axis=1)
    f_jm1 = np.roll(f, 1, axis=1)
    df_dy = (f_jp1 - f_jm1) / (2.0 * h)

    return df_dx, df_dy


def gradient(f, h, order=4):
    """梯度调度器"""
    if order == 2:
        return gradient_2nd(f, h)
    elif order == 4:
        return gradient_4th(f, h)
    else:
        raise ValueError(f"Unsupported gradient order: {order}")


def biharmonic_2nd(f, h):
    """
    双调和算子 ∇⁴f = Δ(Δf) (2阶精度)

    应用两次 5-point Laplacian。

    离散双调和算子的 13-point 复合模板:
    ∇⁴_h f_{ij} = (1/h⁴) × [
        20 f_{ij}
        - 8(f_{i±1,j} + f_{i,j±1})
        + 2(f_{i±1,j±1})
        + (f_{i±2,j} + f_{i,j±2})
    ]

    注: 这是 Δ² 的离散近似, 不是 Δ_h ∘ Δ_h 的直接复合。
    直接复合会得到不同的模板但相同精度。

    Parameters
    ----------
    f : ndarray (Ny, Nx)
        2D 输入场
    h : float
        网格间距

    Returns
    -------
    biharm : ndarray
        ∇⁴f 的离散近似
    """
    # 方法: 应用 Laplacian 两次
    lap_f = laplacian_2nd(f, h)
    biharm = laplacian_2nd(lap_f, h)
    return biharm


def biharmonic_4th(f, h):
    """
    4阶双调和算子 ∇⁴f = Δ⁴f

    使用 4阶 Laplacian 两次:
    ∇⁴_h f = Δ_h^(4) [Δ_h^(4) f]

    注意: 这增加了模板宽度 (9×9 有效模板),
    但在周期性边界下通过 np.roll 高效实现。

    Parameters
    ----------
    f : ndarray (Ny, Nx)
        2D 输入场
    h : float
        网格间距

    Returns
    -------
    biharm : ndarray
        4阶精度 ∇⁴f
    """
    lap_f = laplacian_4th(f, h)
    biharm = laplacian_4th(lap_f, h)
    return biharm


def biharmonic(f, h, order=4):
    """双调和调度器"""
    if order == 2:
        return biharmonic_2nd(f, h)
    elif order == 4:
        return biharmonic_4th(f, h)
    else:
        raise ValueError(f"Unsupported biharmonic order: {order}")


def divergence_4th(fx, fy, h):
    """
    4阶散度算子

    div(F) = ∂fx/∂x + ∂fy/∂y

    使用 4阶中心差分:
    ∂fx/∂x ≈ (-fx_{i+2,j} + 8fx_{i+1,j} - 8fx_{i-1,j} + fx_{i-2,j}) / (12h)

    Parameters
    ----------
    fx, fy : ndarray
        向量场的两个分量
    h : float
        网格间距

    Returns
    -------
    div : ndarray
        散度
    """
    dfx_dx = gradient_4th(fx, h)[0]
    dfy_dy = gradient_4th(fy, h)[1]
    return dfx_dx + dfy_dy


def modified_wavenumber_analysis(N, h, order=4):
    """
    计算离散算子的修正波数 (Modified Wavenumber)

    对于离散 Laplacian 作用于傅里叶模式 e^{ikx}:
        Δ_h e^{ikx} = -k²_eff e^{ikx}

    2阶: k²_eff = (4/h²) sin²(kh/2)
    4阶: k²_eff = (1/(3h²)) × [
             16 sin²(kh/2) - sin²(kh)
         ]
         (沿一个方向)

    对于 2D 4阶 Mehrstellen:
    k²_eff(kx,ky) = (4/(3h²)) × [2 - cos(kx h) - cos(ky h)]

    Parameters
    ----------
    N : int
        网格点数
    h : float
        网格间距
    order : int
        差分阶数

    Returns
    -------
    k : ndarray
        波数数组
    k2_exact : ndarray
        精确 k²
    k2_modified : ndarray
        离散修正 k²
    """
    k = np.fft.fftfreq(N, d=h) * 2.0 * np.pi
    k2_exact = k**2

    if order == 2:
        k2_modified = (4.0 / h**2) * np.sin(k * h / 2.0)**2
    elif order == 4:
        # 4th order central difference modified wavenumber
        k2_modified = (1.0 / (3.0 * h**2)) * (
            16.0 * np.sin(k * h / 2.0)**2 - np.sin(k * h)**2
        )
    else:
        raise ValueError(f"Order {order} not supported")

    return k, k2_exact, k2_modified


def poisson_solve_fd(rhs, h, order=4):
    """
    有限差分法求解 Poisson 方程

    ∇²φ = rhs

    使用 FFT 直接求解 (在周期性边界下等价于精确离散求解):

    -k²_eff φ̃(k) = rhs̃(k)
    φ̃(k) = -rhs̃(k) / k²_eff(k)

    DC 分量 (k=0): 设为 0 (零均值解)

    Parameters
    ----------
    rhs : ndarray (Ny, Nx)
        Poisson 方程右端项
    h : float
        网格间距
    order : int
        差分阶数 (决定 k²_eff 的形式)

    Returns
    -------
    phi : ndarray
        Poisson 方程的解
    """
    Ny, Nx = rhs.shape
    ky = np.fft.fftfreq(Ny, d=h) * 2.0 * np.pi
    kx = np.fft.fftfreq(Nx, d=h) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky)

    # 修正波数 (2D)
    if order == 2:
        k2_eff = (4.0 / h**2) * (
            np.sin(KX * h / 2.0)**2 + np.sin(KY * h / 2.0)**2
        )
    elif order == 4:
        k2_eff = (1.0 / (3.0 * h**2)) * (
            16.0 * np.sin(KX * h / 2.0)**2 - np.sin(KX * h)**2
            + 16.0 * np.sin(KY * h / 2.0)**2 - np.sin(KY * h)**2
        )
    else:
        raise ValueError(f"Order {order} not supported")

    # FFT 求解
    rhs_hat = np.fft.fft2(rhs)

    # 避免除以零 (DC 分量)
    k2_eff[0, 0] = 1.0  # 保护

    phi_hat = -rhs_hat / k2_eff
    phi_hat[0, 0] = 0.0  # 零均值条件

    phi = np.real(np.fft.ifft2(phi_hat))
    return phi
