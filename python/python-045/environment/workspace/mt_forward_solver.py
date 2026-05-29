#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mt_forward_solver.py
大地电磁正演求解器

融合种子项目：
  - 365_fd1d_predator_prey_plot: 一维有限差分法
  - 141_cavity_flow_display: 矢量场网格处理思想（用于 E/H 场后处理）

核心物理：
   Maxwell 方程在准静态近似下：
    ∇ × E = iωμ₀ H
    ∇ × H = σ E

  一维层状介质解析解（递推公式）：
    对于第 n 层，波数 k_n = √(iωμ₀σ_n)
    层顶阻抗：Z_n = (iωμ₀ / k_n) * coth(k_n h_n + arccoth(k_{n+1} Z_{n+1} / (iωμ₀)))
    底层半无限空间：Z_N = iωμ₀ / k_N

  二维 TE 模式有限差分：
    ∂²E_x / ∂y² + ∂²E_x / ∂z² + k² E_x = 0
    其中 k² = iωμ₀σ

  阻抗张量：
    Z_xy = E_x / H_y
    Z_yx = E_y / H_x

  视电阻率与相位：
    ρ_a = |Z|² / (ωμ₀)
    φ = arg(Z)  [度]
"""

import numpy as np
from parameter_manager import PhysicalConstants
from mesh_generator import StructuredMesh2D
from sparse_matrix_tools import DenseLUSolver


def mt_1d_analytic(resistivities, thicknesses, frequencies):
    """
    一维层状大地 MT 正演解析解

    Parameters
    ----------
    resistivities : ndarray, shape (n_layers,)
        各层电阻率 [Ω·m]
    thicknesses : ndarray, shape (n_layers - 1,)
        各层厚度 [m]
    frequencies : ndarray
        频率 [Hz]

    Returns
    -------
    Z_xy : ndarray, shape (n_freq,)
        阻抗 [V/A·m]
    rho_a : ndarray
        视电阻率 [Ω·m]
    phi : ndarray
        相位 [度]
    """
    mu0 = PhysicalConstants.MU_0
    n_layers = len(resistivities)
    n_freq = len(frequencies)
    Z_xy = np.zeros(n_freq, dtype=np.complex128)

    # TODO [Hole 1]: 实现一维层状介质MT正演的核心递推算法
    # 需要完成：
    # 1. 对每个频率，从底层半无限空间向上逐层递推计算层顶阻抗
    # 2. 使用波数 k_n = sqrt(i*omega*mu0*sigma_n) 和双曲函数关系
    # 3. 底层阻抗: Z_N = i*omega*mu0 / k_N
    # 4. 递推公式: Z_n = (i*omega*mu0/k) / tanh(k*h + arctanh(k*Z_{n+1}/(i*omega*mu0)))
    # 5. 数值稳定性处理（避免arctanh奇点）
    # 6. 由最终阻抗 Z_xy 计算视电阻率 rho_a = |Z|^2 / (omega*mu0) 和相位 phi = arg(Z)
    # 关键科学知识：Maxwell方程准静态近似下的层状介质电磁响应
    raise NotImplementedError("Hole 1: 一维层状介质MT正演递推公式待实现")


def mt_1d_analytic_cole_cole(resistivities, thicknesses, dispersion_list, frequencies):
    """
    一维层状大地 MT 正演（含 Cole-Cole 频散）

    使用复电导率替代直流电导率。
    """
    mu0 = PhysicalConstants.MU_0
    n_layers = len(resistivities)
    n_freq = len(frequencies)
    Z_xy = np.zeros(n_freq, dtype=np.complex128)

    for ifreq, f in enumerate(frequencies):
        omega = 2.0 * np.pi * f

        # 获取底层复电导率
        if dispersion_list[-1] is not None:
            sigma_star = dispersion_list[-1].complex_conductivity(omega)
        else:
            sigma_star = 1.0 / resistivities[-1]
        k_N = np.sqrt(1j * omega * mu0 * sigma_star)
        Z_top = 1j * omega * mu0 / k_N

        for ilayer in range(n_layers - 2, -1, -1):
            if dispersion_list[ilayer] is not None:
                sigma_star = dispersion_list[ilayer].complex_conductivity(omega)
            else:
                sigma_star = 1.0 / resistivities[ilayer]
            k = np.sqrt(1j * omega * mu0 * sigma_star)
            h = thicknesses[ilayer]
            ratio = k * Z_top / (1j * omega * mu0)
            inv_ratio = 1.0 / ratio
            if np.abs(inv_ratio) >= 0.999:
                inv_ratio = 0.999 * inv_ratio / np.abs(inv_ratio)
            arg = k * h + np.arctanh(inv_ratio)
            # 数值稳定性处理
            if np.abs(np.sinh(arg)) < 1e-15:
                arg = arg + 1e-15j
            Z_top = (1j * omega * mu0 / k) / np.tanh(arg)

        Z_xy[ifreq] = Z_top

    omega_all = 2.0 * np.pi * frequencies
    rho_a = np.abs(Z_xy) ** 2 / (omega_all * mu0)
    phi = np.angle(Z_xy, deg=True)
    return Z_xy, rho_a, phi


def mt_2d_te_fd(conductivity_map, mesh, frequency, boundary_value_func):
    """
    二维 TE 模式有限差分正演求解器

    求解方程：∂²E_x / ∂y² + ∂²E_x / ∂z² + iωμ₀σ E_x = 0

    Parameters
    ----------
    conductivity_map : callable or ndarray
        电导率分布 [S/m]，可以是函数 σ(y,z) 或节点数组
    mesh : StructuredMesh2D
        结构化网格
    frequency : float
        频率 [Hz]
    boundary_value_func : callable
        边界条件函数 bc(y, z) -> E_x

    Returns
    -------
    E_x : ndarray, shape (n_nodes,)
        电场 x 分量
    H_y : ndarray, shape (n_nodes,)
        磁场 y 分量（由法拉第定律计算）
    Z_xy : ndarray, shape (n_nodes,)
        局部阻抗
    """
    mu0 = PhysicalConstants.MU_0
    omega = 2.0 * np.pi * frequency

    n_nodes = mesh.n_nodes

    # 获取各节点的电导率
    if callable(conductivity_map):
        sigma_nodes = np.zeros(n_nodes, dtype=np.complex128)
        for idx in range(n_nodes):
            y, z = mesh.node_coords[idx]
            sigma_nodes[idx] = conductivity_map(y, z)
    else:
        sigma_nodes = np.asarray(conductivity_map, dtype=np.complex128)
        if len(sigma_nodes) != n_nodes:
            raise ValueError("电导率数组长度与节点数不匹配")

    # 构建系数矩阵 A
    # A u = 0, 其中 u = E_x
    # 内部节点: (u_{i+1,j} - 2u + u_{i-1,j}) / dx² + (u_{i,j+1} - 2u + u_{i,j-1}) / dy² + k² u = 0
    dx2 = mesh.dx ** 2
    dy2 = mesh.dy ** 2

    # 构建线性系统
    # 对边界节点直接赋值，内部节点用有限差分
    n_int = len(mesh.interior_nodes)
    n_bnd = len(mesh.boundary_nodes)

    # 构建全局方程
    A = np.zeros((n_nodes, n_nodes), dtype=np.complex128)
    rhs = np.zeros(n_nodes, dtype=np.complex128)

    # 内部节点
    for idx in mesh.interior_nodes:
        i, j = mesh.inv_map[idx]
        neighbors = mesh.get_neighbors(idx)

        coeff = 0.0
        for nidx, direction in neighbors:
            if direction in ('E', 'W'):
                A[idx, nidx] += 1.0 / dx2
                coeff -= 1.0 / dx2
            else:
                A[idx, nidx] += 1.0 / dy2
                coeff -= 1.0 / dy2

        k2 = 1j * omega * mu0 * sigma_nodes[idx]
        A[idx, idx] = coeff + k2

    # 边界节点：Dirichlet 边界条件
    for idx in mesh.boundary_nodes:
        y, z = mesh.node_coords[idx]
        A[idx, idx] = 1.0
        rhs[idx] = boundary_value_func(y, z)

    # 求解线性系统
    solver = DenseLUSolver(A)
    info = solver.dgefa()
    if info != 0:
        # 矩阵可能病态，尝试使用 numpy 的 lstsq
        E_x = np.linalg.lstsq(A, rhs, rcond=None)[0]
    else:
        E_x = solver.solve(rhs)

    # 计算磁场 H_y = (1 / (iωμ₀)) * ∂E_x / ∂z
    # 使用中心差分
    H_y = np.zeros(n_nodes, dtype=np.complex128)
    for idx in range(n_nodes):
        i, j = mesh.inv_map[idx]
        nidx_s = mesh.get_node_index(i, j - 1)
        nidx_n = mesh.get_node_index(i, j + 1)

        if nidx_s >= 0 and nidx_n >= 0:
            dEdz = (E_x[nidx_n] - E_x[nidx_s]) / (2.0 * mesh.dy)
        elif nidx_n >= 0:
            dEdz = (E_x[nidx_n] - E_x[idx]) / mesh.dy
        elif nidx_s >= 0:
            dEdz = (E_x[idx] - E_x[nidx_s]) / mesh.dy
        else:
            dEdz = 0.0

        H_y[idx] = dEdz / (1j * omega * mu0)

    # 计算局部阻抗 Z_xy = E_x / H_y
    Z_xy = np.zeros(n_nodes, dtype=np.complex128)
    for idx in range(n_nodes):
        if abs(H_y[idx]) > 1e-20:
            Z_xy[idx] = E_x[idx] / H_y[idx]
        else:
            Z_xy[idx] = 0.0

    return E_x, H_y, Z_xy


def compute_apparent_resistivity_phase(Z, frequencies):
    """
    由阻抗计算视电阻率和相位

    Parameters
    ----------
    Z : ndarray
        阻抗 [V/A·m]
    frequencies : ndarray
        频率 [Hz]

    Returns
    -------
    rho_a : ndarray
        视电阻率 [Ω·m]
    phi : ndarray
        相位 [度]
    """
    mu0 = PhysicalConstants.MU_0
    omega = 2.0 * np.pi * frequencies
    rho_a = np.abs(Z) ** 2 / (omega * mu0)
    phi = np.angle(Z, deg=True)
    return rho_a, phi


def add_noise_to_mt_data(rho_a, phi, noise_level=0.05):
    """
    给 MT 数据添加高斯噪声

    Parameters
    ----------
    rho_a : ndarray
        视电阻率
    phi : ndarray
        相位 [度]
    noise_level : float
        噪声水平（相对标准差）

    Returns
    -------
    rho_a_noisy, phi_noisy : ndarray
    """
    rho_a = np.asarray(rho_a, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)

    rho_noise = rho_a * noise_level * np.random.randn(len(rho_a))
    # 相位噪声通常较小，约 1-3 度
    phi_noise = np.random.randn(len(phi)) * noise_level * 5.0

    rho_a_noisy = np.maximum(rho_a + rho_noise, 0.1)
    phi_noisy = phi + phi_noise
    # 相位限制在合理范围
    phi_noisy = np.clip(phi_noisy, -90.0, 90.0)

    return rho_a_noisy, phi_noisy


def thin_field_data(coords, field_values, thin_factor=2):
    """
    对场数据进行稀疏采样（基于 cavity_flow_display 的 thin_index 思想）

    用于减少二维正演输出数据量，模拟实际观测中的测点分布。
    """
    coords = np.asarray(coords)
    field_values = np.asarray(field_values)
    n = len(coords)

    x_unique = np.unique(coords[:, 0])
    y_unique = np.unique(coords[:, 1])

    # 简单的格网化稀疏
    kept = []
    for i in range(n):
        xi = coords[i, 0]
        yi = coords[i, 1]
        ix = np.searchsorted(x_unique, xi)
        iy = np.searchsorted(y_unique, yi)
        if iy % thin_factor == thin_factor // 2 and ix % thin_factor == thin_factor // 2:
            kept.append(i)

    return coords[kept], field_values[kept]


if __name__ == "__main__":
    # 一维解析解自检
    resistivities = np.array([100.0, 50.0, 10.0])
    thicknesses = np.array([500.0, 1000.0])
    frequencies = np.logspace(-2, 2, 20)
    Z, rho_a, phi = mt_1d_analytic(resistivities, thicknesses, frequencies)
    print("1D 解析正演结果 (前5个频率):")
    for i in range(min(5, len(frequencies))):
        print(f"  f={frequencies[i]:.4f} Hz, Z={Z[i]:.4e}, "
              f"ρ_a={rho_a[i]:.2f} Ω·m, φ={phi[i]:.2f}°")

    # 二维正演自检
    from mesh_generator import generate_rectangular_mesh
    mesh = generate_rectangular_mesh(0.0, 10000.0, 0.0, 5000.0, 21, 11)

    def sigma_map(y, z):
        if z < 500.0:
            return 0.01
        elif z < 2000.0:
            return 0.02
        else:
            return 0.1

    def bc_func(y, z):
        # 顶部边界：单位电场
        if abs(z) < 1.0:
            return 1.0
        # 其他边界：衰减的平面波近似
        return np.exp(-z / 1000.0)

    E, H, Z2d = mt_2d_te_fd(sigma_map, mesh, 10.0, bc_func)
    print(f"\n2D 正演: {len(E)} 节点, E_x 范围: [{np.min(np.abs(E)):.4e}, {np.max(np.abs(E)):.4e}]")
