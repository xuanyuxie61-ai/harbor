"""
biharmonic_elasticity.py
========================
一维双调和方程有限差分解法，用于纳米结构弹性变形分析。

源自 biharmonic_fd1d 项目的核心算法，应用于计算纳米尺度薄膜
或纳米梁在热应力作用下的弯曲变形。双调和方程是薄板弯曲理论的
控制方程（Kirchhoff-Love 理论）。

核心物理公式
------------
一维双调和方程：

    d⁴u/dx⁴ = f(x),    x ∈ [a, b]

边界条件（固支梁，Clamped）：
    u(a) = u(b) = 0          (位移为零)
    u'(a) = u'(b) = 0        (转角为零)

对于均匀离散网格 x_i = a + i·h, h = (b-a)/(n-1)，四阶中心差分：

    (u_{i-2} - 4u_{i-1} + 6u_i - 4u_{i+1} + u_{i+2}) / h⁴ = f(x_i)

边界处理（虚构节点法）：
    左端 u'(a)=0: (u_1 - u_{-1})/(2h) = 0  →  u_{-1} = u_1
    因此 i=0 处方程变为：
        (u_{-2} - 4u_{-1} + 6u_0 - 4u_1 + u_2)/h⁴ = f_0
    需用 u_0=0 替换第一行，u_1 处用 u_{-1}=u_1 修正。

纳米梁弯曲与热膨胀耦合：
    当存在温度梯度 ΔT(x) 时，等效载荷：
        f(x) = -α·E·h³/(12(1-ν²)) · d²ΔT/dx²
    
    其中 α 为热膨胀系数，E 为杨氏模量，ν 为泊松比，h 为梁厚度。

弹性常数与 LJ 参数的关系（粗略估计）：
    E ≈ 75 ε/σ³   （对于面心立方结构）
"""

import numpy as np
from typing import Callable, Tuple


def build_biharmonic_matrix(n: int) -> np.ndarray:
    """
    构造一维双调和方程的稀疏矩阵（带状五对角）。
    
    内部点：
        A[i,i-2] =  1
        A[i,i-1] = -4
        A[i,i]   =  6
        A[i,i+1] = -4
        A[i,i+2] =  1
    """
    A = np.zeros((n, n))
    for i in range(n):
        if i - 2 >= 0:
            A[i, i - 2] = 1.0
        if i - 1 >= 0:
            A[i, i - 1] = -4.0
        A[i, i] = 6.0
        if i + 1 < n:
            A[i, i + 1] = -4.0
        if i + 2 < n:
            A[i, i + 2] = 1.0
    return A


def solve_biharmonic_fd1d(f_func: Callable,
                          n: int = 65,
                          xlim: Tuple[float, float] = (-1.0, 1.0),
                          bc_displacement: Tuple[float, float] = (0.0, 0.0),
                          bc_slope: Tuple[float, float] = (0.0, 0.0)) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用有限差分求解一维双调和方程。
    
    方程: d⁴u/dx⁴ = f(x)
    边界: u(x_l)=u_l, u(x_r)=u_r, u'(x_l)=u'_l, u'(x_r)=u'_r
    
    参数:
        f_func: 右端项函数
        n: 网格点数
        xlim: (x_left, x_right)
        bc_displacement: (u_left, u_right)
        bc_slope: (u'_left, u'_right)
    
    返回:
        (x, u)
    """
    x_left, x_right = xlim
    x = np.linspace(x_left, x_right, n)
    h = (x_right - x_left) / (n - 1)

    # 右端项（含 h⁴ 因子）
    b = np.array([f_func(xi) for xi in x]) * (h ** 4)

    A = build_biharmonic_matrix(n)

    ul, ur = bc_displacement
    upl, upr = bc_slope

    # 边界条件处理
    # 左端位移
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = ul

    # 右端位移
    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    b[-1] = ur

    # 左端斜率（使用 i=1 的方程和虚构节点 u_{-1}=u_1 - 2h·upl）
    # 原 i=1 方程: u_{-1} - 4u_0 + 6u_1 - 4u_2 + u_3 = b_1
    # u_0 = ul, u_{-1} = u_1 - 2h·upl
    # → -4ul + 7u_1 - 4u_2 + u_3 = b_1 + 2h·upl
    if n > 2:
        A[1, :] = 0.0
        A[1, 0] = 0.0
        A[1, 1] = 7.0
        if n > 3:
            A[1, 2] = -4.0
        if n > 4:
            A[1, 3] = 1.0
        b[1] = b[1] + 2.0 * h * upl - ul  # 已经考虑了 -4u_0 = -4ul
        # 修正：实际上上面的推导中，u_{-1} = u_1 - 2h*upl 代入后
        # u_1 - 2h*upl - 4ul + 6u_1 - 4u_2 + u_3 = b_1
        # 7u_1 - 4u_2 + u_3 = b_1 + 2h*upl + 4ul
        b[1] = b[1] + 4.0 * ul

    # 右端斜率
    # u_{n} = u_{n-2} + 2h·upr
    # i=n-2: u_{n-4} - 4u_{n-3} + 6u_{n-2} - 4u_{n-1} + u_n = b_{n-2}
    # → u_{n-4} - 4u_{n-3} + 7u_{n-2} - 4u_{n-1} = b_{n-2} - 2h·upr
    if n > 3:
        A[-2, :] = 0.0
        if n > 5:
            A[-2, -4] = 1.0
        if n > 4:
            A[-2, -3] = -4.0
        A[-2, -2] = 7.0
        A[-2, -1] = 0.0
        b[-2] = b[-2] - 2.0 * h * upr

    # 求解
    try:
        u = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # 使用最小二乘作为备用
        u = np.linalg.lstsq(A, b, rcond=None)[0]

    return x, u


def compute_curvature(u: np.ndarray, h: float) -> np.ndarray:
    """
    计算解的曲率 κ ≈ d²u/dx²（中心差分）。
    
    κ_i = (u_{i-1} - 2u_i + u_{i+1}) / h²
    """
    n = len(u)
    kappa = np.zeros(n)
    for i in range(1, n - 1):
        kappa[i] = (u[i - 1] - 2.0 * u[i] + u[i + 1]) / (h * h)
    # 边界用单侧差分
    if n > 2:
        kappa[0] = (2.0 * u[0] - 5.0 * u[1] + 4.0 * u[2] - u[3]) / (h * h)
        kappa[-1] = (2.0 * u[-1] - 5.0 * u[-2] + 4.0 * u[-3] - u[-4]) / (h * h)
    return kappa


def compute_strain_energy(u: np.ndarray, h: float,
                          young_modulus: float = 1.0,
                          moment_of_inertia: float = 1.0) -> float:
    """
    计算梁的弯曲应变能。
    
    U = (E·I)/2 ∫ (d²u/dx²)² dx ≈ (E·I)/2 Σ κ_i² · h
    """
    kappa = compute_curvature(u, h)
    return 0.5 * young_modulus * moment_of_inertia * np.sum(kappa ** 2) * h


def thermal_load_from_gradient(temp_func: Callable,
                               x: np.ndarray,
                               alpha: float = 1.0,
                               young_modulus: float = 1.0,
                               thickness: float = 1.0,
                               poisson_ratio: float = 0.3) -> np.ndarray:
    """
    由温度梯度生成双调和方程的等效载荷。
    
    对于薄板，热等效载荷：
        f(x) = -[α·E/(1-ν)] · d²T/dx²
    
    参数:
        temp_func: 温度分布函数 T(x)
        x: 网格点
        alpha: 热膨胀系数
        young_modulus: 杨氏模量 E
        thickness: 板厚 h
        poisson_ratio: 泊松比 ν
    """
    n = len(x)
    h = x[1] - x[0] if n > 1 else 1.0
    T = np.array([temp_func(xi) for xi in x])

    # 二阶导数 d²T/dx²
    d2T = np.zeros(n)
    for i in range(1, n - 1):
        d2T[i] = (T[i - 1] - 2.0 * T[i] + T[i + 1]) / (h * h)
    if n > 2:
        d2T[0] = d2T[1]
        d2T[-1] = d2T[-2]

    coeff = alpha * young_modulus / (1.0 - poisson_ratio)
    # 对于梁理论，等效载荷与 h³ 相关
    # f = -coeff * d2T * h³ / 12
    return -coeff * d2T * (thickness ** 3) / 12.0
