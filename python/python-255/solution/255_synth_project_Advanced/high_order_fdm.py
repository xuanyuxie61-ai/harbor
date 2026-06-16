# -*- coding: utf-8 -*-
"""
high_order_fdm.py
======================================================================
高阶有限差分算子 —— 4 阶紧致格式与 6 阶中心差分

物理背景:
    辐射传输方程的空间离散要求高阶精度以抑制数值耗散。
    考虑一般对流-扩散型方程:
        du/dt + a * du/dx = nu * d^2u/dx^2 + S(x, t)

    一阶导数:
        - 6 阶中心差分 (6th-order central):
            u'_i = (-u_{i+3} + 9 u_{i+2} - 45 u_{i+1}
                    + 45 u_{i-1} - 9 u_{i-2} + u_{i-3}) / (60 dx)
            截断误差: O(dx^6)

        - 4 阶紧致格式 (Pade, tridiagonal):
            (1/4) u'_{i-1} + u'_i + (1/4) u'_{i+1}
                = (3/(4 dx)) (u_{i+1} - u_{i-1})
            截断误差: O(dx^4)
            需解三对角系统 (依赖 tridiagonal_solver)

    二阶导数:
        - 4 阶中心差分:
            u''_i = (-u_{i+2} + 16 u_{i+1} - 30 u_i
                     + 16 u_{i-1} - u_{i-2}) / (12 dx^2)
            截断误差: O(dx^4)

依赖: numpy, tridiagonal 子程序 (从 1355_tridiagonal_solver 移植)
======================================================================
"""

import numpy as np
from typing import Tuple, Optional


# ============================================================
# 三对角求解器 (直接移植自 1355_tridiagonal_solver)
# ============================================================
def tridiagonal_mv(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """
    三对角矩阵与向量的乘积。
    A = tridiag(a, b, c), 计算 y = A @ x。
    a[0] 和 c[-1] 不使用。
    """
    m = len(b)
    rhs = np.zeros(m, dtype=np.float64)
    rhs[:] = b[:] * x[:]
    rhs[1:m] += a[1:m] * x[: m - 1]
    rhs[: m - 1] += c[: m - 1] * x[1:m]
    return rhs


def tridiagonal_solver(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """
    Thomas 算法求解三对角线性系统 A x = d。
    与 1355_tridiagonal_solver 的 MATLAB 版本保持相同接口。

    边界处理: 当 b(i) == 0 时抛出异常。
    """
    a = a.astype(np.float64, copy=True)
    b = b.astype(np.float64, copy=True)
    c = c.astype(np.float64, copy=True)
    d = d.astype(np.float64, copy=True)

    m = len(b)
    for i in range(1, m):
        if abs(b[i - 1]) < 1.0e-30:
            raise ValueError(f"Thomas 算法: b[{i-1}] = 0, 主元消失")
        s = a[i] / b[i - 1]
        b[i] = b[i] - s * c[i - 1]
        d[i] = d[i] - s * d[i - 1]

    x = np.zeros(m, dtype=np.float64)
    for i in range(m - 1, -1, -1):
        if abs(b[i]) < 1.0e-30:
            raise ValueError(f"Thomas 算法: 回代时 b[{i}] = 0")
        if i == m - 1:
            x[i] = d[i] / b[i]
        else:
            x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


# ============================================================
# 高阶有限差分算子
# ============================================================
def fd_first_derivative_6th(
    u: np.ndarray, dx: float
) -> np.ndarray:
    """
    6 阶中心差分一阶导数。
    边界采用 4 阶单侧差分。

    公式:
        u'_i = (-u_{i+3} + 9 u_{i+2} - 45 u_{i+1}
                + 45 u_{i-1} - 9 u_{i-2} + u_{i-3}) / (60 dx)
    截断误差 O(dx^6)。
    """
    n = len(u)
    if n < 7:
        raise ValueError(f"6阶差分需要至少7个点, 当前 n={n}")

    dudx = np.zeros(n, dtype=np.float64)
    dudx[3 : n - 3] = (
        -u[6:n]
        + 9.0 * u[5 : n - 1]
        - 45.0 * u[4 : n - 2]
        + 45.0 * u[2 : n - 4]
        - 9.0 * u[1 : n - 5]
        + u[0 : n - 6]
    ) / (60.0 * dx)

    # 左边界 4 阶前向
    dudx[0] = (-25.0 * u[0] + 48.0 * u[1] - 36.0 * u[2] + 16.0 * u[3] - 3.0 * u[4]) / (12.0 * dx)
    dudx[1] = (-3.0 * u[0] - 10.0 * u[1] + 18.0 * u[2] - 6.0 * u[3] + u[4]) / (12.0 * dx)
    dudx[2] = (u[0] - 8.0 * u[1] + 8.0 * u[3] - u[4]) / (12.0 * dx)

    # 右边界 4 阶后向
    dudx[n - 1] = (25.0 * u[n - 1] - 48.0 * u[n - 2] + 36.0 * u[n - 3] - 16.0 * u[n - 4] + 3.0 * u[n - 5]) / (12.0 * dx)
    dudx[n - 2] = (3.0 * u[n - 1] + 10.0 * u[n - 2] - 18.0 * u[n - 3] + 6.0 * u[n - 4] - u[n - 5]) / (12.0 * dx)
    dudx[n - 3] = (-u[n - 1] + 8.0 * u[n - 2] - 8.0 * u[n - 4] + u[n - 5]) / (12.0 * dx)

    return dudx


def fd_first_derivative_4th_compact(
    u: np.ndarray, dx: float
) -> np.ndarray:
    """
    4 阶紧致 (Pade) 一阶导数, 需解三对角系统。

    隐式格式:
        (1/4) u'_{i-1} + u'_i + (1/4) u'_{i+1}
            = (3/(4 dx)) (u_{i+1} - u_{i-1})

    边界条件: u'_0 = u'_1, u'_{n-1} = u'_{n-2} (Neumann 型,
    类似 Allen-Cahn 边界处理)。
    """
    n = len(u)
    if n < 5:
        raise ValueError(f"紧致差分需要至少5个点, 当前 n={n}")

    alpha = 0.25
    rhs_interior = (3.0 / (4.0 * dx)) * (u[2:n] - u[0 : n - 2])

    a = np.full(n, alpha, dtype=np.float64)
    b = np.ones(n, dtype=np.float64)
    c = np.full(n, alpha, dtype=np.float64)
    rhs = np.zeros(n, dtype=np.float64)

    rhs[1 : n - 1] = rhs_interior
    a[0] = 0.0
    c[n - 1] = 0.0

    # Neumann 边界: 镜像
    rhs[0] = rhs[1]
    rhs[n - 1] = rhs[n - 2]
    b[0] = 1.0
    c[0] = -1.0
    a[n - 1] = -1.0
    b[n - 1] = 1.0

    return tridiagonal_solver(a, b, c, rhs)


def fd_second_derivative_4th(
    u: np.ndarray, dx: float
) -> np.ndarray:
    """
    4 阶中心差分二阶导数。

    公式:
        u''_i = (-u_{i+2} + 16 u_{i+1} - 30 u_i
                 + 16 u_{i-1} - u_{i-2}) / (12 dx^2)
    截断误差 O(dx^4)。

    此算子与 Allen-Cahn 的 laplacian_interval 对应,
    但精度从 O(dx^2) 提升到 O(dx^4)。
    """
    n = len(u)
    if n < 5:
        raise ValueError(f"4阶二阶差分需要至少5个点, 当前 n={n}")

    d2u = np.zeros(n, dtype=np.float64)
    d2u[2 : n - 2] = (
        -u[4:n]
        + 16.0 * u[3 : n - 1]
        - 30.0 * u[2 : n - 2]
        + 16.0 * u[1 : n - 3]
        - u[0 : n - 4]
    ) / (12.0 * dx * dx)

    # 2 阶边界
    d2u[0] = (u[0] - 2.0 * u[1] + u[2]) / (dx * dx)
    d2u[1] = (u[0] - 2.0 * u[1] + u[2]) / (dx * dx)
    d2u[n - 2] = (u[n - 3] - 2.0 * u[n - 2] + u[n - 1]) / (dx * dx)
    d2u[n - 1] = (u[n - 3] - 2.0 * u[n - 2] + u[n - 1]) / (dx * dx)

    return d2u


def fd_fourth_derivative_compact(
    u: np.ndarray, dx: float
) -> np.ndarray:
    """
    4 阶导数的紧致近似 (用于辐射传输中的超扩散项)。

    实现: 两次应用二阶差分算子 D2:
        D4 = D2 @ D2
    这是双调和算子 d^4/dx^4 的离散形式。
    """
    d2u = fd_second_derivative_4th(u, dx)
    return fd_second_derivative_4th(d2u, dx)


def fd_first_derivative_nonuniform(
    u: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """
    非均匀网格上的一阶导数 (移植自 003_allen_cahn 的 laplacian_interval)。

    对非等间距节点 x_i, 采用二阶差分:
        u'_i = [(u_{i+1} - u_i)/(x_{i+1} - x_i) * (x_i - x_{i-1})^2
              + (u_i - u_{i-1})/(x_i - x_{i-1}) * (x_{i+1} - x_i)^2]
              / [(x_{i+1} - x_i) * (x_i - x_{i-1}) * (x_{i+1} - x_{i-1})]
    """
    n = len(u)
    if n < 3:
        raise ValueError(f"非均匀差分至少需要 3 个点, 当前 n={n}")

    dudx = np.zeros(n, dtype=np.float64)

    for i in range(1, n - 1):
        dxl = x[i] - x[i - 1]
        dxr = x[i + 1] - x[i]
        dx = x[i + 1] - x[i - 1]

        if abs(dxl) < 1.0e-30 or abs(dxr) < 1.0e-30 or abs(dx) < 1.0e-30:
            dudx[i] = 0.0
            continue

        ul = (u[i] - u[i - 1]) / dxl
        ur = (u[i + 1] - u[i]) / dxr

        dudx[i] = (ul * dxr + ur * dxl) / dx

    dudx[0] = (u[1] - u[0]) / max(x[1] - x[0], 1.0e-30)
    dudx[n - 1] = (u[n - 1] - u[n - 2]) / max(x[n - 1] - x[n - 2], 1.0e-30)

    return dudx


def fd_second_derivative_nonuniform(
    u: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """
    非均匀网格上的二阶导数。
    """
    n = len(u)
    if n < 3:
        raise ValueError(f"非均匀二阶差分至少需要 3 个点, 当前 n={n}")

    d2u = np.zeros(n, dtype=np.float64)
    for i in range(1, n - 1):
        dxl = x[i] - x[i - 1]
        dxr = x[i + 1] - x[i]
        if abs(dxl) < 1.0e-30 or abs(dxr) < 1.0e-30:
            d2u[i] = 0.0
            continue
        d2u[i] = 2.0 * (
            (u[i + 1] - u[i]) / dxr - (u[i] - u[i - 1]) / dxl
        ) / (dxl + dxr)

    d2u[0] = d2u[1]
    d2u[n - 1] = d2u[n - 2]
    return d2u


def build_derivative_matrix(
    n: int, dx: float, order: int = 4
) -> np.ndarray:
    """
    构造全局一阶差分矩阵 D (n x n)。
    用于 von Neumann 稳定性分析中计算特征值。
    """
    D = np.zeros((n, n), dtype=np.float64)
    if order == 2:
        for i in range(1, n - 1):
            D[i, i + 1] = 1.0 / (2.0 * dx)
            D[i, i - 1] = -1.0 / (2.0 * dx)
    elif order == 4:
        for i in range(2, n - 2):
            D[i, i + 1] = 45.0 / (60.0 * dx)
            D[i, i - 1] = -45.0 / (60.0 * dx)
            D[i, i + 2] = -9.0 / (60.0 * dx)
            D[i, i - 2] = 9.0 / (60.0 * dx)
            D[i, i + 3] = 1.0 / (60.0 * dx)
            D[i, i - 3] = -1.0 / (60.0 * dx)
    else:
        raise ValueError(f"不支持的差分阶数: {order}")
    return D
