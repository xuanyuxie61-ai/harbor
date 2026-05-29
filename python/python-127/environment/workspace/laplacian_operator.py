"""
laplacian_operator.py
=====================
高阶 Laplacian 离散算子模块

基于种子项目:
  - 487_gray_scott_pde: 9点 Laplacian stencil（环形区域周期边界）
  - 282_differ: Vandermonde 型差分模板矩阵生成

科学背景:
  耳蜗组织中的电场分布涉及电流连续性方程:
      ∇·(σ ∇V) = -I_e

  当 σ 为常数时，退化为 Poisson 方程:
      σ ∇²V = -I_e

  在结构化网格上，需要高精度 Laplacian 离散格式。
  本模块提供:
    1) 5点/9点 compact stencil
    2) 基于差分模板矩阵的高阶格式
    3) 非均匀网格上的 Laplacian
"""

import numpy as np
from scipy.sparse import diags, csr_matrix


def laplacian_5point(V, dx, dy):
    """
    5点 Laplacian stencil:
        ∇²V ≈ (V_{i+1,j} + V_{i-1,j} + V_{i,j+1} + V_{i,j-1} - 4V_{i,j}) / h²

    Parameters
    ----------
    V : ndarray, shape (nx, ny)
        二维场量
    dx, dy : float
        网格间距

    Returns
    -------
    L : ndarray, shape (nx, ny)
        Laplacian 结果
    """
    V = np.asarray(V, dtype=float)
    if V.ndim != 2:
        raise ValueError("V 必须为二维数组")

    nx, ny = V.shape
    L = np.zeros_like(V)

    # 内部点
    L[1:-1, 1:-1] = (
        (V[2:, 1:-1] - 2.0 * V[1:-1, 1:-1] + V[:-2, 1:-1]) / (dx**2)
        + (V[1:-1, 2:] - 2.0 * V[1:-1, 1:-1] + V[1:-1, :-2]) / (dy**2)
    )

    # 边界处理: Neumann 零通量
    L[0, :] = L[1, :]
    L[-1, :] = L[-2, :]
    L[:, 0] = L[:, 1]
    L[:, -1] = L[:, -2]

    return L


def laplacian_9point(V, dx, dy):
    """
    9点高阶 Laplacian stencil:
        ∇²V ≈ [
            1*V_{i-1,j-1} + 4*V_{i-1,j} + 1*V_{i-1,j+1}
          + 4*V_{i,j-1}   -20*V_{i,j}   + 4*V_{i,j+1}
          + 1*V_{i+1,j-1} + 4*V_{i+1,j} + 1*V_{i+1,j+1}
        ] / (6 h²)

    截断误差 O(h⁴)，基于种子 487_gray_scott_pde。

    Parameters
    ----------
    V : ndarray, shape (nx, ny)
    dx, dy : float

    Returns
    -------
    L : ndarray, shape (nx, ny)
    """
    V = np.asarray(V, dtype=float)
    if V.ndim != 2:
        raise ValueError("V 必须为二维数组")

    nx, ny = V.shape
    L = np.zeros_like(V)

    # TODO: 修复 Hole 2
    # 9点高阶 Laplacian stencil (截断误差 O(h⁴))
    # 提示: 标准 9点 Laplacian 系数为
    #   [ 1,  4,  1]
    #   [ 4,-20,  4] / (6 * dx²)
    #   [ 1,  4,  1]
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            L[i, j] = (
                # TODO: 填入正确的 9点 stencil 系数和分母
                0.0  # placeholder
            )

    # 边界
    L[0, :] = L[1, :]
    L[-1, :] = L[-2, :]
    L[:, 0] = L[:, 1]
    L[:, -1] = L[:, -2]

    return L


def build_differ_matrix(n, stencil):
    """
    构建差分模板矩阵 (Vandermonde 型)。

    基于种子 282_differ 的思想:
        A_ij = stencil_j^i,  i=0..n-1, j=0..n-1

    用于求解差分系数:
        A * c = r.h.s
    其中 r.h.s = [0, 1, 0, ..., 0]^T 对应一阶导数。

    Parameters
    ----------
    n : int
        模板点数
    stencil : ndarray, shape (n,)
        模板位置（网格单位），必须互异且非零

    Returns
    -------
    A : ndarray, shape (n, n)
        模板矩阵
    """
    stencil = np.asarray(stencil, dtype=float)
    if len(stencil) != n:
        raise ValueError("stencil 长度必须等于 n")
    if len(np.unique(stencil)) != n:
        raise ValueError("stencil 中的点必须互异")
    if np.any(np.isclose(stencil, 0.0)):
        raise ValueError("stencil 中不能包含零点（边界处理请单独处理）")

    A = np.zeros((n, n))
    A[0, :] = stencil
    for i in range(1, n):
        A[i, :] = A[i - 1, :] * stencil
    return A


def high_order_derivative_coefficients(order, stencil):
    """
    计算高阶导数差分系数。

    Parameters
    ----------
    order : int
        导数阶数 (1=一阶, 2=二阶)
    stencil : ndarray
        模板位置

    Returns
    -------
    coeffs : ndarray
        差分系数
    """
    n = len(stencil)
    A = build_differ_matrix(n, stencil)
    rhs = np.zeros(n)
    if 1 <= order <= n - 1:
        rhs[order] = np.math.factorial(order)
    else:
        raise ValueError("order 必须在 1 到 n-1 之间")
    coeffs = np.linalg.solve(A, rhs)
    return coeffs


def laplacian_matrix_1d(n, dx, stencil_type='central'):
    """
    构建一维 Laplacian 的稀疏矩阵表示。

    Parameters
    ----------
    n : int
        网格点数
    dx : float
        网格间距
    stencil_type : str
        'central' (二阶), 'compact4' (四阶 compact)

    Returns
    -------
    L : csr_matrix, shape (n, n)
    """
    if stencil_type == 'central':
        main = -2.0 * np.ones(n)
        off = np.ones(n - 1)
        L = diags([off, main, off], [-1, 0, 1], format='csr') / (dx**2)
        # Neumann 边界
        L[0, 0] = -1.0 / (dx**2)
        L[0, 1] = 1.0 / (dx**2)
        L[-1, -1] = -1.0 / (dx**2)
        L[-1, -2] = 1.0 / (dx**2)
    elif stencil_type == 'compact4':
        # 四阶 compact: (1/10)L_{i-1} + L_i + (1/10)L_{i+1} = ...
        # 简化为高阶中心差分
        main = -2.5 * np.ones(n)
        off1 = (4.0 / 3.0) * np.ones(n - 1)
        off2 = (-1.0 / 12.0) * np.ones(n - 2)
        L = diags([off2, off1, main, off1, off2],
                  [-2, -1, 0, 1, 2], format='csr') / (dx**2)
        # 边界用二阶
        L[0, 0] = -2.0 / (dx**2)
        L[0, 1] = 1.0 / (dx**2)
        L[-1, -1] = -2.0 / (dx**2)
        L[-1, -2] = 1.0 / (dx**2)
    else:
        raise ValueError(f"未知的 stencil_type: {stencil_type}")

    return L


def anisotropic_laplacian_5point(V, dx, dy, sigma_xx, sigma_yy):
    """
    各向异性 Laplacian:
        ∂/∂x (σ_xx ∂V/∂x) + ∂/∂y (σ_yy ∂V/∂y)

    Parameters
    ----------
    V : ndarray, shape (nx, ny)
    dx, dy : float
    sigma_xx, sigma_yy : float
        各方向电导率

    Returns
    -------
    L : ndarray, shape (nx, ny)
    """
    V = np.asarray(V, dtype=float)
    nx, ny = V.shape
    L = np.zeros_like(V)

    # 内部: 守恒型离散
    L[1:-1, 1:-1] = (
        sigma_xx * (V[2:, 1:-1] - 2.0 * V[1:-1, 1:-1] + V[:-2, 1:-1]) / (dx**2)
        + sigma_yy * (V[1:-1, 2:] - 2.0 * V[1:-1, 1:-1] + V[1:-1, :-2]) / (dy**2)
    )

    # 边界
    L[0, :] = L[1, :]
    L[-1, :] = L[-2, :]
    L[:, 0] = L[:, 1]
    L[:, -1] = L[:, -2]

    return L
