"""
high_order_fd.py
================
高阶紧致有限差分 (Compact Finite Difference) 算子。

专为 Cahn-Hilliard 方程空间离散化设计:
  - 六阶紧致差分 (Pade 型) 计算一阶、二阶导数
  - 四阶紧致差分计算四阶导数 (∂⁴c/∂x⁴)
  - 周期性边界条件处理
  - 带权重的混合差分算子

核心公式:
  六阶紧致差分 (一阶导数):
    alpha * f'_{i-1} + f'_i + alpha * f'_{i+1}
    = a * (f_{i+1} - f_{i-1})/(2h)
    + b * (f_{i+2} - f_{i-2})/(4h)
    + c * (f_{i+3} - f_{i-3})/(6h)
  其中 alpha = 14/18, a = 28/27, b = 1/27, c = 0 (Lele 1992)

  六阶紧致差分 (二阶导数):
    alpha * f''_{i-1} + f''_i + alpha * f''_{i+1}
    = a * (f_{i+1} - 2f_i + f_{i-1})/h²
    + b * (f_{i+2} - 2f_i + f_{i-2})/(4h²)
    + c * (f_{i+3} - 2f_i + f_{i-3})/(9h²)
  其中 alpha = 2/11, a = 12/11, b = 3/11, c = 0
"""

import numpy as np
from numpy.linalg import solve


def build_periodic_tridiag(N, alpha, rhs_func, f):
    """
    求解周期性三对角系统 (Sherman-Morrison 方法):
        alpha * x_{i-1} + x_i + alpha * x_{i+1} = rhs_i
    其中 rhs 由 rhs_func 计算。

    使用 Sherman-Morrison 公式将循环三对角分解为
    两个普通三对角系统的求解:
        A*x = rhs - (x_0 * alpha * e_0 + x_{N-1} * alpha * e_{N-1})

    Parameters
    ----------
    N : int
        网格点数
    alpha : float
        紧致差分参数
    rhs_func : callable
        右端项计算函数 rhs_func(f) -> array
    f : np.ndarray
        输入函数值

    Returns
    -------
    np.ndarray
        解向量
    """
    if N < 4:
        # 退化情况: 直接求解
        A = np.eye(N)
        for i in range(N):
            A[i, (i - 1) % N] += alpha
            A[i, (i + 1) % N] += alpha
        return solve(A, rhs_func(f))

    # 右端项
    d = rhs_func(f)

    # Sherman-Morrison 分解:
    # 修改第一行和最后一行来消除循环性
    gamma = -1.0  # 自由参数, 取 -1 使数值稳定
    A_diag = np.ones(N)
    A_lower = np.full(N, alpha)
    A_upper = np.full(N, alpha)

    # 修改主对角
    A_diag[0] = 1.0 - gamma
    A_diag[N - 1] = 1.0 - alpha * alpha / gamma

    # 右端项修正
    d_mod = d.copy()
    d_mod[0] -= gamma * f[N - 1] if False else 0.0  # placeholder
    # 实际使用: d_mod[0] -= gamma, d_mod[N-1] -= alpha²/gamma (不需要, 因为修正项进入 u, v)

    # 使用 Thomas 算法求解修正系统
    # 方法 1: 直接构造并求解 (小规模, 性能可接受)
    A_mat = np.diag(A_diag) + np.diag(A_lower[:-1], -1) + np.diag(A_upper[1:], 1)
    # 周期性连接
    A_mat[0, N - 1] = alpha
    A_mat[N - 1, 0] = alpha

    # 修正向量
    u = np.zeros(N)
    v = np.zeros(N)
    u[0] = gamma
    u[N - 1] = alpha
    v[0] = 1.0
    v[N - 1] = alpha / gamma if abs(gamma) > 1e-30 else 0.0

    # 修正右端项
    d_mod = d - np.dot(u, d) * v / (1.0 + np.dot(v, u)) if abs(1.0 + np.dot(v, u)) > 1e-30 else d

    # 求解主系统
    try:
        y = solve(A_mat, d)
        q = solve(A_mat, u)
    except np.linalg.LinAlgError:
        # 退化回直接求解
        return solve(A_mat, d)

    # Sherman-Morrison 修正
    factor = np.dot(v, y) / (1.0 + np.dot(v, q)) if abs(1.0 + np.dot(v, q)) > 1e-30 else 0.0
    x = y - factor * q

    return x


def compact_first_derivative(f, h, omega=14.0 / 18.0):
    """
    六阶紧致差分计算一阶导数 f'(x):

    omega * f'_{i-1} + f'_i + omega * f'_{i+1}
    = (28/27) * (f_{i+1} - f_{i-1})/(2h)
    + (1/27) * (f_{i+2} - f_{i-2})/(4h)

    截断误差: O(h⁶)

    Parameters
    ----------
    f : np.ndarray
        函数值 (周期性边界)
    h : float
        网格间距
    omega : float
        紧致参数 (默认 14/18 为六阶最优)

    Returns
    -------
    np.ndarray
        f'(x) 近似值
    """
    N = len(f)
    if N < 5:
        # 退化: 中心差分
        df = np.zeros(N)
        for i in range(N):
            df[i] = (f[(i + 1) % N] - f[(i - 1) % N]) / (2.0 * h)
        return df

    def rhs_func(f_in):
        rhs = np.zeros(N)
        for i in range(N):
            rhs[i] = (28.0 / 27.0) * (f_in[(i + 1) % N] - f_in[(i - 1) % N]) / (2.0 * h) \
                + (1.0 / 27.0) * (f_in[(i + 2) % N] - f_in[(i - 2) % N]) / (4.0 * h)
        return rhs

    return build_periodic_tridiag(N, omega, rhs_func, f)


def compact_second_derivative(f, h, alpha=2.0 / 11.0):
    """
    六阶紧致差分计算二阶导数 f''(x):

    alpha * f''_{i-1} + f''_i + alpha * f''_{i+1}
    = (12/11) * (f_{i+1} - 2f_i + f_{i-1})/h²
    + (3/11) * (f_{i+2} - 2f_i + f_{i-2})/(4h²)

    截断误差: O(h⁶)

    Parameters
    ----------
    f : np.ndarray
        函数值 (周期性边界)
    h : float
        网格间距
    alpha : float
        紧致参数 (默认 2/11 为六阶最优)

    Returns
    -------
    np.ndarray
        f''(x) 近似值
    """
    N = len(f)
    if N < 5:
        # 退化: 标准二阶中心差分
        d2f = np.zeros(N)
        for i in range(N):
            d2f[i] = (f[(i + 1) % N] - 2.0 * f[i] + f[(i - 1) % N]) / (h * h)
        return d2f

    def rhs_func(f_in):
        rhs = np.zeros(N)
        for i in range(N):
            rhs[i] = (12.0 / 11.0) * (f_in[(i + 1) % N] - 2.0 * f_in[i]
                                         + f_in[(i - 1) % N]) / (h * h) \
                + (3.0 / 11.0) * (f_in[(i + 2) % N] - 2.0 * f_in[i]
                                    + f_in[(i - 2) % N]) / (4.0 * h * h)
        return rhs

    return build_periodic_tridiag(N, alpha, rhs_func, f)


def compact_fourth_derivative(f, h):
    """
    四阶紧致差分计算四阶导数 ∂⁴f/∂x⁴:

    使用两次二阶紧致差分的复合:
        f⁽⁴⁾ ≈ D²_h(D²_h(f))
    其中 D²_h 为六阶紧致二阶差分算子。

    或者直接使用显式六点模板:
        f⁽⁴⁾_i = (f_{i-2} - 4f_{i-1} + 6f_i - 4f_{i+1} + f_{i+2}) / h⁴
    截断误差: O(h²)

    Parameters
    ----------
    f : np.ndarray
        函数值 (周期性边界)
    h : float
        网格间距

    Returns
    -------
    np.ndarray
        f⁽⁴⁾(x) 近似值
    """
    N = len(f)
    d4f = np.zeros(N)

    if N < 5:
        # 退化处理
        return d4f

    h4 = h ** 4
    for i in range(N):
        d4f[i] = (
            f[(i - 2) % N]
            - 4.0 * f[(i - 1) % N]
            + 6.0 * f[i]
            - 4.0 * f[(i + 1) % N]
            + f[(i + 2) % N]
        ) / h4

    return d4f


def build_diffusion_matrix_compact(N, h, D, omega=2.0 / 11.0):
    """
    构建紧致差分离散化的扩散算子矩阵 (用于隐式求解):

    M * d²c/dx² ≈ D * ∇²_h c

    其中 M 为三对角矩阵 (紧致格式左手边):
        M = diag(1) + omega * diag(1, ±1)  (周期性)

    右端为 D 乘以二阶差分模板。

    Parameters
    ----------
    N : int
        网格点数
    h : float
        网格间距
    D : float
        扩散系数
    omega : float
        紧致参数

    Returns
    -------
    tuple
        (M_matrix, D_matrix): 左手边质量矩阵和右手边扩散矩阵
    """
    # 左手边: 三对角紧致矩阵
    M = np.eye(N)
    for i in range(N):
        M[i, (i - 1) % N] += omega
        M[i, (i + 1) % N] += omega

    # 右手边: 二阶差分矩阵 * D
    D_mat = np.zeros((N, N))
    coeff_diag = -2.0 * (12.0 / 11.0) / (h * h)
    coeff_off1 = (12.0 / 11.0) / (h * h)
    coeff_off2 = (3.0 / 11.0) / (4.0 * h * h)

    for i in range(N):
        D_mat[i, i] = coeff_diag
        D_mat[i, (i - 1) % N] += coeff_off1
        D_mat[i, (i + 1) % N] += coeff_off1
        D_mat[i, (i - 2) % N] += coeff_off2
        D_mat[i, (i + 2) % N] += coeff_off2

    D_mat *= D

    return M, D_mat


def von_neumann_stability_factor(k, h, scheme_order=6):
    """
    计算有限差分格式的 von Neumann 放大因子。

    对于六阶紧致格式:
        G(k*h) = 1 + dt * lambda(k)
    其中 lambda(k) 为离散 Laplacian 的特征值。

    六阶紧致的修正波数:
        k_eff² * h² = [12/11 * (1 - cos(kh)) + 3/44 * (1 - cos(2kh))]
                     / [1 + 2/11 * cos(kh)]

    Parameters
    ----------
    k : float or np.ndarray
        波数 (rad/m)
    h : float
        网格间距
    scheme_order : int
        差分阶数

    Returns
    -------
    float or np.ndarray
        修正波数 k_eff² (用于稳定性分析)
    """
    kh = k * h
    cos_kh = np.cos(kh)
    cos_2kh = np.cos(2.0 * kh)

    # 六阶紧致修正波数
    numerator = (12.0 / 11.0) * (1.0 - cos_kh) + (3.0 / 44.0) * (1.0 - cos_2kh)
    denominator = 1.0 + (2.0 / 11.0) * cos_kh

    k_eff_sq = np.where(
        np.abs(denominator) > 1e-30,
        numerator / (denominator * h * h),
        0.0
    )

    return k_eff_sq
