"""
chebyshev_spectral.py
=====================
Chebyshev 谱插值与微分。

源自 interp_chebyshev 项目的核心算法，扩展为支持自动微分
的谱方法工具箱。Chebyshev 节点具有极小化 Runge 现象的最优
分布特性，在区间 [-1,1] 上定义为：

    x_k = cos(πk / N),    k = 0, 1, ..., N

这些节点是 Chebyshev 多项式 T_N(x) = cos(N·arccos(x)) 的
极值点。

核心数学公式
------------
插值多项式（Newton 形式，差商）：

    P_N(x) = d₀ + d₁(x-x₀) + d₂(x-x₀)(x-x₁) + ... + d_N Π_{k=0}^{N-1}(x-x_k)

其中差商 d_k 递归计算：

    d_k^{(0)} = f(x_k)
    d_k^{(j)} = (d_k^{(j-1)} - d_{k-1}^{(j-1)}) / (x_k - x_{k-j})

Chebyshev 微分矩阵 D（谱精度）：

    D_{ij} = (c_i / c_j) · (-1)^{i+j} / (x_i - x_j)   (i ≠ j)
    D_{ii} = -x_i / (2(1-x_i²))                      (i ≠ 0, N)
    D_{00} = (2N² + 1) / 6
    D_{NN} = -(2N² + 1) / 6

其中 c₀ = c_N = 2, c_i = 1 (0 < i < N)。

通过 D 可计算任意网格函数的导数：
    f'(x_i) = Σ_j D_{ij} f(x_j)

Clenshaw 递推（稳定求值）：
    对 P_N(x) = Σ a_k T_k(x)，定义递推：
        b_{N+2} = b_{N+1} = 0
        b_k = 2x·b_{k+1} - b_{k+2} + a_k
    则 P_N(x) = (b₀ - b₂)/2
"""

import numpy as np
from typing import Callable, Tuple


def chebyshev_nodes(a: float, b: float, n: int) -> np.ndarray:
    """
    生成区间 [a, b] 上的 Chebyshev 节点（含端点）。
    
    x_k = (b+a)/2 + (b-a)/2 · cos(πk/(n-1)),  k=0,...,n-1
    
    参数:
        a, b: 区间端点
        n: 节点数
    """
    if n < 1:
        return np.array([])
    if n == 1:
        return np.array([(a + b) / 2.0])
    k = np.arange(n)
    theta = np.pi * k / (n - 1)
    x = np.cos(theta)
    # 映射到 [a, b]
    return 0.5 * (b - a) * x + 0.5 * (b + a)


def divided_differences(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    计算差商表。
    
    输入:
        x: 节点数组（长度 n）
        y: 函数值数组（长度 n）
    
    返回:
        d: 差商数组，d[k] 为 k 阶差商 f[x₀,...,x_k]
    """
    n = len(x)
    d = np.array(y, dtype=float)
    for j in range(1, n):
        for k in range(n - 1, j - 1, -1):
            denom = x[k] - x[k - j]
            if abs(denom) < 1e-30:
                denom = 1e-30 if denom >= 0 else -1e-30
            d[k] = (d[k] - d[k - 1]) / denom
    return d


def newton_interpolate(xd: np.ndarray, dd: np.ndarray,
                       xp: np.ndarray) -> np.ndarray:
    """
    使用 Newton 差商形式在点 xp 上求插值多项式的值。
    
    P(x) = dd[n-1] + (x-xd[n-2])·[dd[n-2] + (x-xd[n-3])·[...]]
    """
    nd = len(dd)
    yp = dd[-1] * np.ones_like(xp, dtype=float)
    for i in range(nd - 2, -1, -1):
        yp = dd[i] + (xp - xd[i]) * yp
    return yp


def chebyshev_interpolate(func: Callable, a: float, b: float,
                          n: int, xp: np.ndarray) -> np.ndarray:
    """
    对函数 func 在 [a,b] 上进行 n 点 Chebyshev 插值，在 xp 处求值。
    
    步骤:
        1. 生成 Chebyshev 节点 xd
        2. 计算函数值 yd = func(xd)
        3. 计算差商 dd = divided_differences(xd, yd)
        4. 在 xp 处用 Newton 插值求值
    """
    xd = chebyshev_nodes(a, b, n)
    yd = func(xd)
    dd = divided_differences(xd, yd)
    return newton_interpolate(xd, dd, xp)


def chebyshev_differentiation_matrix(n: int) -> np.ndarray:
    """
    计算 Chebyshev 谱微分矩阵 D（n×n）。
    
    D 满足：f'(x_i) ≈ Σ_j D_{ij} f(x_j)，具有谱精度。
    """
    if n < 2:
        return np.zeros((n, n))
    x = np.cos(np.pi * np.arange(n) / (n - 1))
    c = np.ones(n)
    c[0] = 2.0
    c[-1] = 2.0

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            D[i, j] = (c[i] / c[j]) * ((-1) ** (i + j)) / (x[i] - x[j])
    # 对角元
    D[0, 0] = (2.0 * (n - 1) ** 2 + 1.0) / 6.0
    D[-1, -1] = -D[0, 0]
    for i in range(1, n - 1):
        D[i, i] = -x[i] / (2.0 * (1.0 - x[i] ** 2))
    return D


def chebyshev_derivative(f_vals: np.ndarray) -> np.ndarray:
    """
    使用谱微分矩阵计算 Chebyshev 网格上函数的导数。
    
    参数:
        f_vals: 在 Chebyshev 节点上的函数值（标准区间 [-1,1]）
    
    返回:
        导数值数组
    """
    n = len(f_vals)
    D = chebyshev_differentiation_matrix(n)
    return D @ f_vals


def chebyshev_spectral_solve_ode_bvp(coeff_func: Callable,
                                      rhs_func: Callable,
                                      n: int = 32,
                                      bc_left: Tuple[float, float] = (0.0, 0.0),
                                      bc_right: Tuple[float, float] = (0.0, 0.0)) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 Chebyshev 谱方法求解二阶常微分方程边值问题：
    
        a(x) u''(x) + b(x) u'(x) + c(x) u(x) = f(x)
        u(a) = u_l,  u(b) = u_r
    
    参数:
        coeff_func: 返回 (a(x), b(x), c(x)) 的函数
        rhs_func:   右端项 f(x)
        n:          Chebyshev 点数
        bc_left:    (u_l, u'_l) 左边界条件
        bc_right:   (u_r, u'_r) 右边界条件
    
    返回:
        (x_nodes, u_solution)
    """
    x = np.cos(np.pi * np.arange(n) / (n - 1))
    D = chebyshev_differentiation_matrix(n)
    D2 = D @ D

    # 组装系统矩阵
    A = np.zeros((n, n))
    b = np.zeros(n)

    for i in range(n):
        a_i, b_i, c_i = coeff_func(x[i])
        A[i, :] = a_i * D2[i, :] + b_i * D[i, :] + c_i * np.eye(n)[i, :]
        b[i] = rhs_func(x[i])

    # 施加边界条件（替换首末行）
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = bc_left[0]

    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    b[-1] = bc_right[0]

    # 如果提供了导数边界条件，修改第二行和倒数第二行
    if bc_left[1] is not None and n > 2:
        A[1, :] = D[0, :]
        b[1] = bc_left[1]
    if bc_right[1] is not None and n > 2:
        A[-2, :] = D[-1, :]
        b[-2] = bc_right[1]

    u = np.linalg.solve(A, b)
    return x, u


def chebyshev_clenshaw_eval(coeffs: np.ndarray, x: float) -> float:
    """
    使用 Clenshaw 递推计算 Chebyshev 级数 Σ_k coeffs[k] T_k(x)。
    
    递推关系：
        b_{N+2} = b_{N+1} = 0
        b_k = 2x·b_{k+1} - b_{k+2} + coeffs[k]
        P(x) = (b_0 - b_2) / 2
    """
    N = len(coeffs) - 1
    b2 = 0.0
    b1 = 0.0
    for k in range(N, -1, -1):
        b0 = 2.0 * x * b1 - b2 + coeffs[k]
        b2 = b1
        b1 = b0
    return 0.5 * (b0 - b2)
