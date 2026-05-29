"""
finite_difference.py
高阶有限差分模块

基于种子项目 282_differ 的核心算法：
- differ_stencil: 通过 Vandermonde 矩阵求解有限差分系数
- differ_matrix: 构建差分矩阵

在离子通道问题中的应用：
用于求解 Poisson-Nernst-Planck 方程组的空间离散化。
"""

import numpy as np
from numpy.polynomial import polynomial as P


def vandermonde_like(stencil):
    """
    构建广义 Vandermonde 矩阵（源自 differ_matrix.m）
    A[i,j] = stencil[j]**i
    """
    n = len(stencil)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = stencil[j] ** i
    return A


def differ_stencil(x0, order, precision, x):
    """
    计算有限差分系数（源自 differ_stencil.m）

    求解系数 c 使得：
        d^o f / dx^o (x0) = sum_i c[i] * f(x[i]) + O(h^p)

    其中 n = o + p 为模板点数。

    Parameters
    ----------
    x0 : float
        求导点
    order : int
        导数阶数 o >= 1
    precision : int
        误差阶数 p
    x : ndarray
        模板点，长度 n = o + p

    Returns
    -------
    c : ndarray
        差分系数
    """
    n = order + precision
    if len(x) != n:
        raise ValueError("模板点数量必须等于 order + precision")
    dx = x - x0
    A = vandermonde_like(dx)
    b = np.zeros(n)
    b[order] = 1.0
    # 求解线性方程组 A * c = b，然后乘以阶乘
    c = np.linalg.solve(A, b)
    c = c * np.math.factorial(order)
    return c


def build_laplacian_1d(N, dx, bc_type='dirichlet'):
    """
    构建一维 Laplacian 的紧凑差分矩阵（四阶精度）。

    采用中心差分：
        f''(x_i) ≈ (-f_{i-2} + 16 f_{i-1} - 30 f_i + 16 f_{i+1} - f_{i+2}) / (12 h^2)

    边界处理采用降阶或外推。
    """
    A = np.zeros((N, N))
    coeff = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / (12.0 * dx ** 2)
    for i in range(2, N - 2):
        A[i, i - 2:i + 3] = coeff

    # 边界 i=0, 1 和 i=N-2, N-1 使用二阶中心差分
    bc_coeff = np.array([1.0, -2.0, 1.0]) / (dx ** 2)
    A[0, 0:3] = bc_coeff
    A[1, 0:3] = bc_coeff
    A[N - 2, N - 3:N] = bc_coeff
    A[N - 1, N - 3:N] = bc_coeff

    if bc_type == 'dirichlet':
        # Dirichlet 边界: 固定值，Laplacian 在边界退化为恒等
        A[0, :] = 0.0
        A[0, 0] = 1.0
        A[N - 1, :] = 0.0
        A[N - 1, N - 1] = 1.0

    return A


def build_laplacian_3d(Nx, Ny, Nz, dx, dy, dz):
    """
    构建三维各向异性 Laplacian 矩阵（Kronecker 和形式）。

    利用 Laplacian 的可分离性：
        ∇^2 = ∂^2/∂x^2 + ∂^2/∂y^2 + ∂^2/∂z^2

    通过 Kronecker 积构造稀疏表示的算子（返回函数形式，避免 O(N^6) 内存）。
    """
    Lx = build_laplacian_1d(Nx, dx, bc_type='neumann')
    Ly = build_laplacian_1d(Ny, dy, bc_type='neumann')
    Lz = build_laplacian_1d(Nz, dz, bc_type='neumann')

    # 返回作用在展平数组上的函数
    Ix = np.eye(Nx)
    Iy = np.eye(Ny)
    Iz = np.eye(Nz)

    # 使用 Kronecker 积构造完整矩阵（对于小规模问题）
    L = (np.kron(np.kron(Lx, Iy), Iz) +
         np.kron(np.kron(Ix, Ly), Iz) +
         np.kron(np.kron(Ix, Iy), Lz))
    return L


def apply_laplacian_3d(phi, dx, dy, dz):
    """
    对三维场 phi 应用各向同性 Laplacian（紧凑模板，无矩阵存储）。

    内部点采用 7 点模板：
        ∇^2 φ ≈ (φ_{i-1,j,k} - 2φ_{i,j,k} + φ_{i+1,j,k}) / dx^2
                + (φ_{i,j-1,k} - 2φ_{i,j,k} + φ_{i,j+1,k}) / dy^2
                + (φ_{i,j,k-1} - 2φ_{i,j,k} + φ_{i,j,k+1}) / dz^2
    """
    Nx, Ny, Nz = phi.shape
    out = np.zeros_like(phi)

    # 内部点
    # TODO: Hole 1 — 实现三维各向同性 Laplacian 的 7 点紧凑模板
    # 科学背景：∇²φ = ∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z²
    # 提示：各方向采用二阶中心差分，注意 dx, dy, dz 可能不相等
    raise NotImplementedError("Hole 1: 请实现 apply_laplacian_3d 的内部点计算")

    # 边界使用 Neumann 条件（零法向导数）
    out[0, :, :] = out[1, :, :]
    out[-1, :, :] = out[-2, :, :]
    out[:, 0, :] = out[:, 1, :]
    out[:, -1, :] = out[:, -2, :]
    out[:, :, 0] = out[:, :, 1]
    out[:, :, -1] = out[:, :, -2]

    return out
