"""
fem_gravity.py

基于 wathen_ge (稀疏有限元质量矩阵)、tumor_pde (PDE 系数与通量函数)
以及 r8blt_sl (带状下三角矩阵求解器) 核心算法，
实现小行星内部引力势的有限元求解。

科学背景：
小行星内部引力势 φ 满足泊松方程：
    ∇²φ = 4πGρ(r)
其中 ρ(r) 为内部密度分布。

对于非均匀密度（碎石堆结构），需数值求解。
采用 Galerkin 有限元方法，在规则网格上离散：
    ∫_Ω ∇φ · ∇ψ dΩ = −4πG ∫_Ω ρ ψ dΩ  +  边界项

核心离散公式（二维截面简化，轴对称假设）：
    K u = f
其中刚度矩阵 K 由 Wathen 型单元组装，使用带状下三角存储格式求解。
"""

import numpy as np
from typing import Tuple, Optional


class FEMGravityError(Exception):
    pass


def wathen_element_matrix() -> np.ndarray:
    """
    返回 Wathen 8 节点 serendipity 单元的参考质量/刚度矩阵。
    基于 wathen_ge.m 中的 em 矩阵，但 reinterpret 为二维刚度矩阵。

    参考单元的节点编号：
         3---2---1
         |       |
         4       8
         |       |
         5---6---7
    """
    em = np.array([
        [6.0, -6.0, 2.0, -8.0, 3.0, -8.0, 2.0, -6.0],
        [-6.0, 32.0, -6.0, 20.0, -8.0, 16.0, -8.0, 20.0],
        [2.0, -6.0, 6.0, -6.0, 2.0, -8.0, 3.0, -8.0],
        [-8.0, 20.0, -6.0, 32.0, -6.0, 20.0, -8.0, 16.0],
        [3.0, -8.0, 2.0, -6.0, 6.0, -6.0, 2.0, -8.0],
        [-8.0, 16.0, -8.0, 20.0, -6.0, 32.0, -6.0, 20.0],
        [2.0, -8.0, 3.0, -8.0, 2.0, -6.0, 6.0, -6.0],
        [-6.0, 20.0, -8.0, 16.0, -8.0, 20.0, -6.0, 32.0]
    ], dtype=float)
    return em


def assemble_fem_system_2d(
    nx: int,
    ny: int,
    density_grid: np.ndarray,
    g_const: float = 6.67430e-11,
    dx: float = 1.0,
    dy: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    在 nx × ny 网格上组装二维泊松方程有限元系统。
    采用简化的一阶矩形单元（4节点），避免 8 节点 serendipity 的复杂度。

    方程:  ∇·(∇φ) = 4πGρ
    弱形式:  ∫ ∇φ·∇ψ dA = −4πG ∫ ρ ψ dA

    单元刚度矩阵（双线性矩形，参考 [0,1]×[0,1]）:
        K_e = (1/6) [ 4 -1 -2 -1; -1 4 -1 -2; -2 -1 4 -1; -1 -2 -1 4 ]
              × (dy/dx + dx/dy)

    返回:
        A: (n, n) 稀疏存储的带状下三角矩阵（ML=nx+1）
        b: (n,) 右端项
        n: 总自由度
    """
    if density_grid.shape != (ny, nx):
        raise FEMGravityError(f"密度网格形状 {density_grid.shape} 不匹配 (ny={ny}, nx={nx})")

    n = nx * ny
    # 估计下带宽: 一维排序下，节点(i,j)与(i,j+1)相距1，与(i+1,j)相距nx
    ml = nx + 1

    # 使用常规 dense 矩阵先组装，再转为带状下三角
    K = np.zeros((n, n))
    F = np.zeros(n)

    # 4 节点矩形单元刚度（简化，使用标准拉普拉斯离散）
    factor = 1.0 / 6.0 * (dy / dx + dx / dy)
    ke = factor * np.array([
        [4.0, -1.0, -2.0, -1.0],
        [-1.0, 4.0, -1.0, -2.0],
        [-2.0, -1.0, 4.0, -1.0],
        [-1.0, -2.0, -1.0, 4.0]
    ])

    for j in range(ny - 1):
        for i in range(nx - 1):
            # 局部节点编号:
            # 2--3
            # |  |
            # 0--1
            nodes = [
                j * nx + i,
                j * nx + i + 1,
                (j + 1) * nx + i,
                (j + 1) * nx + i + 1
            ]
            # 单元平均密度
            rho_avg = 0.25 * (
                density_grid[j, i] + density_grid[j, i + 1] +
                density_grid[j + 1, i] + density_grid[j + 1, i + 1]
            )
            # 右端项: -4πGρ * (单元体积/4) 分配到 4 个节点
            rhs_per_node = -4.0 * np.pi * g_const * rho_avg * dx * dy / 4.0
            for idx, node in enumerate(nodes):
                F[node] += rhs_per_node
                for jdx, node_j in enumerate(nodes):
                    K[node, node_j] += ke[idx, jdx]

    # Dirichlet 边界条件: 边界势设为 0（简化）
    for i in range(nx):
        K[i, :] = 0.0
        K[:, i] = 0.0
        K[i, i] = 1.0
        F[i] = 0.0
        bottom = (ny - 1) * nx + i
        K[bottom, :] = 0.0
        K[:, bottom] = 0.0
        K[bottom, bottom] = 1.0
        F[bottom] = 0.0
    for j in range(ny):
        left = j * nx
        right = j * nx + nx - 1
        K[left, :] = 0.0
        K[:, left] = 0.0
        K[left, left] = 1.0
        F[left] = 0.0
        K[right, :] = 0.0
        K[:, right] = 0.0
        K[right, right] = 1.0
        F[right] = 0.0

    # 将 dense K 转换为 R8BLT (banded lower triangular) 格式
    # 注意：K 是对称的，我们只存下三角部分用于 Cholesky 或直接求解
    A_blt = dense_to_r8blt(K, ml)

    return A_blt, F, n, ml


def dense_to_r8blt(K: np.ndarray, ml: int) -> np.ndarray:
    """
    将稠密矩阵 K 的下三角部分转换为 R8BLT 格式 A(ml+1, n)。
    R8BLT 存储规则（MATLAB/FORTRAN 风格，列优先）:
        A(1, j) = K(j, j)          对角线
        A(i-j+1, j) = K(i, j)      下三角非零元
    """
    n = K.shape[0]
    A = np.zeros((ml + 1, n))
    for j in range(n):
        A[0, j] = K[j, j]
        ihi = min(j + ml, n - 1)
        for i in range(j + 1, ihi + 1):
            A[i - j, j] = K[i, j]
    return A


def r8blt_sl(n: int, ml: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    基于 r8blt_sl.m 的带状下三角前代求解器。
    求解 A x = b，其中 A 为下三角带状矩阵。

    算法:
        for j = 1 .. n:
            x(j) = b(j) / A(1,j)
            for i = j+1 .. min(j+ml, n):
                b(i) = b(i) - A(i-j+1, j) * x(j)
    """
    x = b.copy()
    for j in range(n):
        if abs(a[0, j]) < 1e-16:
            raise FEMGravityError(f"零对角元在索引 {j}，矩阵奇异")
        x[j] = x[j] / a[0, j]
        ihi = min(j + ml, n - 1)
        for i in range(j + 1, ihi + 1):
            x[i] = x[i] - a[i - j, j] * x[j]
    return x


def solve_internal_potential_2d(
    nx: int = 32,
    ny: int = 32,
    density_profile: Optional[np.ndarray] = None,
    r_max: float = 1e3,
    g_const: float = 6.67430e-11
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    求解二维轴对称小行星的内部引力势分布。

    参数:
        nx, ny: 网格分辨率
        density_profile: (ny, nx) 密度分布 (kg/m³)，若 None 则使用均匀密度
        r_max: 物理尺寸 (m)
        g_const: 万有引力常数

    返回:
        phi: (ny, nx) 引力势 (m²/s²)
        x_coords: (nx,) 网格 x 坐标
        y_coords: (ny,) 网格 y 坐标
    """
    dx = r_max / (nx - 1)
    dy = r_max / (ny - 1)
    x_coords = np.linspace(-r_max / 2, r_max / 2, nx)
    y_coords = np.linspace(-r_max / 2, r_max / 2, ny)

    if density_profile is None:
        # 碎石堆密度：中心高、边缘低
        X, Y = np.meshgrid(x_coords, y_coords)
        r = np.sqrt(X ** 2 + Y ** 2)
        density_profile = 3000.0 * np.exp(-r / (0.3 * r_max))
        density_profile = np.clip(density_profile, 100.0, 5000.0)

    A_blt, F, n, ml = assemble_fem_system_2d(nx, ny, density_profile, g_const, dx, dy)

    # 求解
    phi_flat = r8blt_sl(n, ml, A_blt, F)
    phi = phi_flat.reshape((ny, nx))

    return phi, x_coords, y_coords


def internal_gravity_from_potential(
    phi: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    由势场数值梯度计算内部引力加速度:
        g_x = −∂φ/∂x,  g_y = −∂φ/∂y
    使用中心差分。
    """
    ny, nx = phi.shape
    gx = np.zeros_like(phi)
    gy = np.zeros_like(phi)

    dx = x_coords[1] - x_coords[0] if nx > 1 else 1.0
    dy = y_coords[1] - y_coords[0] if ny > 1 else 1.0

    # 内部中心差分
    if nx > 2:
        gx[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2.0 * dx)
        gx[:, 0] = -(phi[:, 1] - phi[:, 0]) / dx
        gx[:, -1] = -(phi[:, -1] - phi[:, -2]) / dx
    if ny > 2:
        gy[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2.0 * dy)
        gy[0, :] = -(phi[1, :] - phi[0, :]) / dy
        gy[-1, :] = -(phi[-1, :] - phi[-2, :]) / dy

    return gx, gy
