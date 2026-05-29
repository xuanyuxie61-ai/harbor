"""
Chebyshev 谱插值与势能面 (PES) 逼近模块
基于 chebyshev 核心算法：Chebyshev零点生成、系数计算、Clenshaw递推求值。

在蛋白质折叠中，Chebyshev插值用于：
- 高精度拟合自由能函数 F(x) 或 F(Q, RMSD)
- 径向分布函数 (RDF) 的光滑解析延拓
- 二面角势能面的谱展开

数学基础:
    T_n(x) = cos(n * arccos(x)),  x in [-1, 1]
    插值节点: x_j = cos( pi*(2j-1)/(2n) ),  j=1,...,n
    展开系数: c_j = (2/n) * sum_{k=1}^{n} f(x_k) * cos( pi*(j-1)*(2k-1)/(2n) )
    插值函数: C(f)(x) = sum_{i=1}^{n} c_i * T_{i-1}(x) - 0.5*c_1
"""

import numpy as np
from typing import Callable


def chebyshev_zeros(n: int) -> np.ndarray:
    """
    生成 n 阶 Chebyshev 多项式（第一类）的零点。
    
    零点公式:
        x_j = cos( pi * (2*j - 1) / (2*n) ),  j = 1, ..., n
    
    这些节点在 [-1, 1] 上按 arcsine 分布密集，可有效抑制 Runge 效应。
    
    Parameters
    ----------
    n : int
        零点个数，必须 >= 1。
    
    Returns
    -------
    zeros : np.ndarray, shape (n,)
        Chebyshev 零点。
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    j = np.arange(1, n + 1)
    zeros = np.cos(np.pi * (2.0 * j - 1.0) / (2.0 * n))
    return zeros


def chebyshev_coefficients(a: float, b: float, n: int, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    计算函数 f 在区间 [a, b] 上的 n 点 Chebyshev 展开系数。
    
    数学步骤:
        1. 将 [a, b] 映射到 [-1, 1]:  x' = (2x - a - b) / (b - a)
        2. 在 Chebyshev 零点 x_k 处采样 f
        3. 计算离散余弦变换 (DCT) 得到系数 c_j
    
    系数公式:
        c_j = (2/n) * sum_{k=1}^{n} f(x_k) * cos( pi*(j-1)*(2k-1)/(2n) )
    
    Parameters
    ----------
    a, b : float
        目标区间，要求 a < b。
    n : int
        插值阶数。
    f : callable
        输入为 np.ndarray，输出为 np.ndarray 的函数。
    
    Returns
    -------
    coeffs : np.ndarray, shape (n,)
        Chebyshev 展开系数。
    """
    if a >= b:
        raise ValueError("Require a < b")
    if n < 1:
        raise ValueError("n must be at least 1")
    
    z = chebyshev_zeros(n)
    # 将 [-1,1] 的零点映射到 [a,b]
    x_nodes = 0.5 * (a + b) + 0.5 * (b - a) * z
    f_vals = f(x_nodes)
    
    coeffs = np.zeros(n)
    for j in range(n):
        angle = np.pi * j * (2.0 * np.arange(1, n + 1) - 1.0) / (2.0 * n)
        coeffs[j] = (2.0 / n) * np.sum(f_vals * np.cos(angle))
    return coeffs


def chebyshev_interpolant(a: float, b: float, n: int, coeffs: np.ndarray,
                          x_query: np.ndarray) -> np.ndarray:
    """
    在查询点 x_query 上求值 Chebyshev 插值函数。
    使用 Clenshaw 递推算法保证数值稳定性。
    
    Clenshaw 递推:
        令 y = (2x - a - b) / (b - a)，将 x 映射到 [-1, 1]
        d_{n+1} = d_{n+2} = 0
        d_i = 2*y*d_{i+1} - d_{i+2} + c_i,   i = n, n-1, ..., 1
        结果: C(f)(x) = y*d_2 - d_3 + 0.5*c_0
    
    其中索引从0开始，c_i 对应 coeffs[i]，d_i 对应 d[i]。
    
    Parameters
    ----------
    a, b : float
        原始区间。
    n : int
        系数个数。
    coeffs : np.ndarray, shape (n,)
        Chebyshev 系数。
    x_query : np.ndarray
        查询点数组。
    
    Returns
    -------
    values : np.ndarray
        插值函数在查询点处的值。
    """
    if a >= b:
        raise ValueError("Require a < b")
    if coeffs.size != n:
        raise ValueError("coeffs length must equal n")
    
    # TODO: Hole 1 - 实现 Clenshaw 递推算法
    # 要求: 将 x_query 映射到 [-1, 1]，然后用递推公式求值 Chebyshev 插值
    # 提示: y = (2x - a - b) / (b - a)
    #       d_{n+1} = d_{n+2} = 0
    #       d_i = 2*y*d_{i+1} - d_{i+2} + c_i
    #       结果 = y*d_2 - d_3 + 0.5*c_0
    raise NotImplementedError("Hole 1: 请补全 Clenshaw 递推算法")


def chebyshev_derivative(a: float, b: float, n: int, coeffs: np.ndarray,
                         x_query: np.ndarray) -> np.ndarray:
    """
    计算 Chebyshev 插值函数的一阶导数。
    
    利用 Chebyshev 多项式导数的递推关系:
        T_0'(x) = 0
        T_1'(x) = 1
        T_n'(x) = n * U_{n-1}(x)  (U_n 为第二类 Chebyshev 多项式)
    
    或者通过系数递推:
        c'_n = 0
        c'_{n-1} = 2*(n-1) * c_n
        c'_{k} = c'_{k+2} + 2*(k+1) * c_{k+1},  k = n-2, ..., 0
        然后除以 (b-a)/2 以补偿区间映射的导数
    
    Parameters
    ----------
    a, b : float
        原始区间。
    n : int
        系数个数。
    coeffs : np.ndarray, shape (n,)
        Chebyshev 系数。
    x_query : np.ndarray
        查询点。
    
    Returns
    -------
    deriv : np.ndarray
        一阶导数值。
    """
    if n < 2:
        return np.zeros_like(x_query)
    
    dc = np.zeros(n)
    dc[n - 1] = 0.0
    dc[n - 2] = 2.0 * (n - 1) * coeffs[n - 1]
    for k in range(n - 3, -1, -1):
        dc[k] = dc[k + 2] + 2.0 * (k + 1) * coeffs[k + 1]
    
    # 区间映射导致的缩放因子: dx/dy = (b-a)/2
    scale = 2.0 / (b - a)
    return scale * chebyshev_interpolant(a, b, n, dc, x_query)


def fit_free_energy_profile(coordinate_values: np.ndarray, free_energy: np.ndarray,
                            order: int = 16) -> tuple:
    """
    用 Chebyshev 插值拟合一维自由能剖面。
    
    自由能定义:
        F(x) = -k_B T * ln( P(x) / P_max )
    
    由于 P(x) 可能在边界趋近于0导致 ln(0) 问题，本函数对极小概率做截断处理。
    
    Parameters
    ----------
    coordinate_values : np.ndarray
        反应坐标采样点（已排序）。
    free_energy : np.ndarray
        对应采样点的自由能值。
    order : int
        Chebyshev 插值阶数。
    
    Returns
    -------
    a, b : float
        拟合区间。
    coeffs : np.ndarray
        Chebyshev 系数。
    """
    if len(coordinate_values) != len(free_energy):
        raise ValueError("Lengths of coordinate_values and free_energy must match")
    a = float(coordinate_values.min())
    b = float(coordinate_values.max())
    if a >= b:
        raise ValueError("Invalid coordinate range")
    
    # 构建插值函数，通过线性插值在 Chebyshev 节点采样
    def f_interp(x):
        return np.interp(x, coordinate_values, free_energy)
    
    coeffs = chebyshev_coefficients(a, b, order, f_interp)
    return a, b, coeffs


def approximate_potential_energy_surface_2d(x_vals: np.ndarray, y_vals: np.ndarray,
                                            energy_grid: np.ndarray,
                                            nx_cheb: int = 12, ny_cheb: int = 12) -> tuple:
    """
    用张量积 Chebyshev 插值逼近二维势能面/自由能面。
    
    张量积展开:
        F(x, y) ≈ sum_{i=0}^{nx-1} sum_{j=0}^{ny-1} c_{ij} * T_i(x') * T_j(y')
    
    其中 x' 和 y' 为映射到 [-1, 1] 的局部坐标。
    
    Parameters
    ----------
    x_vals : np.ndarray, shape (nx,)
        x 方向网格坐标。
    y_vals : np.ndarray, shape (ny,)
        y 方向网格坐标。
    energy_grid : np.ndarray, shape (nx, ny)
        能量值网格。
    nx_cheb, ny_cheb : int
        Chebyshev 方向阶数。
    
    Returns
    -------
    ax, bx, ay, by : float
        区间边界。
    coeffs_2d : np.ndarray, shape (nx_cheb, ny_cheb)
        二维 Chebyshev 系数。
    """
    ax, bx = float(x_vals.min()), float(x_vals.max())
    ay, by = float(y_vals.min()), float(y_vals.max())
    
    # 在 Chebyshev 节点采样
    zx = chebyshev_zeros(nx_cheb)
    zy = chebyshev_zeros(ny_cheb)
    x_nodes = 0.5 * (ax + bx) + 0.5 * (bx - ax) * zx
    y_nodes = 0.5 * (ay + by) + 0.5 * (by - ay) * zy
    
    from scipy.interpolate import RectBivariateSpline
    # 使用 RectBivariateSpline 进行平滑插值采样
    spline = RectBivariateSpline(x_vals, y_vals, energy_grid, kx=1, ky=1)
    sample_grid = spline.ev(x_nodes[:, None], y_nodes[None, :])
    
    coeffs_2d = np.zeros((nx_cheb, ny_cheb))
    for i in range(nx_cheb):
        angle_x = np.pi * i * (2.0 * np.arange(1, nx_cheb + 1) - 1.0) / (2.0 * nx_cheb)
        cx = (2.0 / nx_cheb) * np.cos(angle_x)
        for j in range(ny_cheb):
            angle_y = np.pi * j * (2.0 * np.arange(1, ny_cheb + 1) - 1.0) / (2.0 * ny_cheb)
            cy = (2.0 / ny_cheb) * np.cos(angle_y)
            coeffs_2d[i, j] = np.sum(sample_grid * cx[:, None] * cy[None, :])
    return ax, bx, ay, by, coeffs_2d
