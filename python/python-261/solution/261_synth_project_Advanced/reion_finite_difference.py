"""
reion_finite_difference.py
==========================
高阶有限差分算子

本模块实现再电离方程空间离散化所需的高阶有限差分模板与矩阵构建.
包含:
  - Fornberg 算法: 任意非均匀点集上任意阶导数的有限差分系数
  - 周期性边界条件差分矩阵
  - 一阶/二阶导数模板 (2阶、4阶、6阶精度)
  - 迎风/中心加权模板 (Peclet 自适应)
  - Laplacian 算子构建

核心公式 (周期性 Laplacian, 4阶精度):
    (D2 u)_i = (-u_{i-2} + 16 u_{i-1} - 30 u_i + 16 u_{i+1} - u_{i+2}) / (12 h^2)

Fornberg 算法 (Fornberg 1988, Math. Comp. 51, 699-706):
    给定 N 个点 x_0, ..., x_{N-1} 与导数阶 M,
    递归计算在 x_0 处的 M 阶导数有限差分系数.

对应种子项目:
  - 387_fem1d_bvp_quadratic (1D BVP 思想 → 再电离辐射传输 BVP)
"""

import math
import numpy as np


def fornberg_weights(x_points, x0, deriv_order):
    """Fornberg 算法计算有限差分权重.

    给定点集 x_points 与目标点 x0, 返回 M 阶导数的有限差分系数.
    采用 Vandermonde 矩阵求解 (数值稳定版本).

    Parameters
    ----------
    x_points : array-like
        模板点坐标 (长度 N)
    x0 : float
        求导点
    deriv_order : int
        导数阶数 (0, 1, 2, ...)

    Returns
    -------
    weights : array [N]
        各点处的有限差分系数
    """
    x_points = np.asarray(x_points, dtype=float)
    N = len(x_points)
    M = deriv_order

    # 使用 Vandermonde 方法: 构造 A[k, j] = (x_j - x0)^k / k!
    # 求解 A @ w = e_M (M阶导数对应的单位向量)
    # 实际上: sum_j w_j * (x_j - x0)^k = delta_{k,M} * M!
    A = np.zeros((N, N))
    for k in range(N):
        A[k, :] = (x_points - x0) ** k
    rhs = np.zeros(N)
    rhs[M] = float(math.factorial(M))

    try:
        weights = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        # 若矩阵病态, 使用最小二乘
        weights, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    return weights


def _compute_FD_coefficients(order, accuracy):
    """计算中心差分模板系数 (整数网格间距 dx=1).

    Parameters
    ----------
    order : int
        导数阶数 (1 = 一阶, 2 = 二阶)
    accuracy : int
        精度阶数 (2, 4, 6, 8)

    Returns
    -------
    coeffs : array [M + 1]
        对应于点 [-M/2, ..., 0, ..., M/2] 的系数
    half_width : int
        M/2 (模板半宽)
    """
    if accuracy % 2 != 0:
        raise ValueError("精度阶数必须为偶数")
    M = order + accuracy - 1
    if M % 2 == 1:
        M += 1  # 保证对称模板
    N_pts = M + 1
    x_pts = np.arange(N_pts, dtype=float) - M / 2.0

    # 使用简化的解析公式 (对均匀网格)
    # 对于一阶导数, 4阶精度: c = [1/12, -2/3, 0, 2/3, -1/12]
    # 对于二阶导数, 4阶精度: c = [-1/12, 4/3, -5/2, 4/3, -1/12]
    # 采用直接构造方法

    # 使用 Vandermonde 方法求解
    # 模板点: x_i = i - M/2, i = 0, ..., M
    # 需要求解: sum_i c_i * x_i^k = k! delta_{k, order},  k = 0, ..., M
    A = np.zeros((N_pts, N_pts))
    for k in range(N_pts):
        A[k, :] = x_pts**k
    rhs = np.zeros(N_pts)
    rhs[order] = float(math.factorial(order))

    coeffs = np.linalg.solve(A, rhs)
    half_width = M // 2
    return coeffs, half_width


def fd_first_derivative(u, dx, accuracy=4, periodic=True):
    """高阶有限差分一阶导数.

    Parameters
    ----------
    u : array [N]
        场变量
    dx : float
        网格间距
    accuracy : int
        精度阶数 (2, 4, 6)
    periodic : bool
        周期边界

    Returns
    -------
    du_dx : array [N]
    """
    N = len(u)
    coeffs, hw = _compute_FD_coefficients(1, accuracy)
    du = np.zeros(N)
    for i in range(N):
        s = 0.0
        for j, c in enumerate(coeffs):
            idx = i + j - hw
            if periodic:
                idx = idx % N
            elif idx < 0 or idx >= N:
                continue
            s += c * u[idx]
        du[i] = s / dx
    return du


def fd_second_derivative(u, dx, accuracy=4, periodic=True):
    """高阶有限差分二阶导数 (Laplacian 1D).

    Parameters
    ----------
    u : array [N]
    dx : float
    accuracy : int
        精度阶数 (2, 4, 6)
    periodic : bool

    Returns
    -------
    d2u_dx2 : array [N]
    """
    N = len(u)
    coeffs, hw = _compute_FD_coefficients(2, accuracy)
    d2u = np.zeros(N)
    for i in range(N):
        s = 0.0
        for j, c in enumerate(coeffs):
            idx = i + j - hw
            if periodic:
                idx = idx % N
            elif idx < 0 or idx >= N:
                continue
            s += c * u[idx]
        d2u[i] = s / (dx * dx)
    return d2u


def fd_laplacian_matrix(N, dx, accuracy=4, periodic=True):
    """构建 Laplacian 差分矩阵 D2, 使得 (D2 @ u) 给出 Laplacian.

    Parameters
    ----------
    N : int
    dx : float
    accuracy : int
    periodic : bool

    Returns
    -------
    D2 : sparse-like dense matrix [N x N]
    """
    coeffs, hw = _compute_FD_coefficients(2, accuracy)
    D2 = np.zeros((N, N))
    for i in range(N):
        for j, c in enumerate(coeffs):
            col = i + j - hw
            if periodic:
                col = col % N
            elif col < 0 or col >= N:
                continue
            D2[i, col] += c
    D2 /= (dx * dx)
    return D2


def fd_first_derivative_matrix(N, dx, accuracy=4, periodic=True):
    """构建一阶导数差分矩阵 D1.

    Returns
    -------
    D1 : array [N x N]
    """
    coeffs, hw = _compute_FD_coefficients(1, accuracy)
    D1 = np.zeros((N, N))
    for i in range(N):
        for j, c in enumerate(coeffs):
            col = i + j - hw
            if periodic:
                col = col % N
            elif col < 0 or col >= N:
                continue
            D1[i, col] += c
    D1 /= dx
    return D1


def upwind_first_derivative(u, dx, velocity, periodic=True):
    """迎风一阶导数 (对 advection 项).

    对正速度采用向后差分, 负速度采用向前差分.

    Parameters
    ----------
    u : array [N]
    dx : float
    velocity : array [N] or float
        风速 (电离锋面传播速度)
    periodic : bool

    Returns
    -------
    du_dx_upwind : array [N]
    """
    N = len(u)
    du = np.zeros(N)
    v_arr = np.broadcast_to(np.asarray(velocity, dtype=float), N).copy()
    for i in range(N):
        if v_arr[i] >= 0:
            # 向后差分
            ip = i - 1 if periodic else max(i - 1, 0)
            if not periodic and i == 0:
                du[i] = 0.0
            else:
                du[i] = (u[i] - u[ip]) / dx
        else:
            ip = i + 1 if periodic else min(i + 1, N - 1)
            if not periodic and i == N - 1:
                du[i] = 0.0
            else:
                du[i] = (u[ip] - u[i]) / dx
    return du


def compact_pade_first(u, dx, alpha=1.0 / 3.0, periodic=True):
    """紧致 Padé 格式求一阶导数 (4阶精度).

    隐式格式:
        alpha * u'_{i-1} + u'_i + alpha * u'_{i+1}
        = a * (u_{i+1} - u_{i-1}) / (2h)

    其中 a = (3/2) * (1 + 2*alpha) / 3 = (1 + 2*alpha) / 2 * 3 / 3 ...
    标准取 alpha = 1/3, a = 4/3 (4阶精度).

    通过 Thomas 算法求解三对角系统得到 u'.
    """
    N = len(u)
    a_coeff = (1.0 + 2.0 * alpha) / 2.0  # 对 alpha=1/3, a=2/3

    # 右端
    rhs = np.zeros(N)
    for i in range(N):
        ip = (i + 1) % N
        im = (i - 1) % N
        rhs[i] = a_coeff * (u[ip] - u[im]) / (2.0 * dx)

    # Thomas 算法求解 (alpha * du[i-1] + du[i] + alpha * du[i+1] = rhs[i])
    # 周期边界需特殊处理 (Sherman-Morrison)
    du = np.zeros(N)
    if not periodic:
        # 标准 Thomas
        c_prime = np.zeros(N)
        d_prime = np.zeros(N)
        c_prime[0] = alpha / 1.0
        d_prime[0] = rhs[0] / 1.0
        for i in range(1, N):
            m = 1.0 - alpha * c_prime[i - 1]
            c_prime[i] = alpha / m if i < N - 1 else 0.0
            d_prime[i] = (rhs[i] - alpha * d_prime[i - 1]) / m
        du[N - 1] = d_prime[N - 1]
        for i in range(N - 2, -1, -1):
            du[i] = d_prime[i] - c_prime[i] * du[i + 1]
    else:
        # Sherman-Morrison 处理周期角点
        gamma_val = -alpha
        a_diag = np.ones(N)
        b_diag = np.full(N, alpha)
        c_diag = np.full(N, alpha)
        # 修正: A' = A - u v^T
        a_diag[0] -= gamma_val
        a_diag[N - 1] -= alpha * alpha / gamma_val
        # Thomas 求解修正系统
        c_prime = np.zeros(N)
        d_prime = np.zeros(N)
        c_prime[0] = b_diag[0] / a_diag[0]
        d_prime[0] = rhs[0] / a_diag[0]
        for i in range(1, N - 1):
            m = a_diag[i] - c_diag[i] * c_prime[i - 1]
            c_prime[i] = b_diag[i] / m
            d_prime[i] = (rhs[i] - c_diag[i] * d_prime[i - 1]) / m
        m = a_diag[N - 1] - c_diag[N - 1] * c_prime[N - 2]
        d_prime[N - 1] = (rhs[N - 1] - c_diag[N - 1] * d_prime[N - 2]) / m
        # 回代
        du_temp = np.zeros(N)
        du_temp[N - 1] = d_prime[N - 1]
        for i in range(N - 2, -1, -1):
            du_temp[i] = d_prime[i] - c_prime[i] * du_temp[i + 1]
        # Sherman-Morrison 修正
        u_vec = np.zeros(N)
        u_vec[0] = gamma_val
        u_vec[N - 1] = alpha
        v_vec = np.zeros(N)
        v_vec[0] = 1.0
        v_vec[N - 1] = c_diag[N - 1] / gamma_val
        # 解 A z = u, A w = v (已经隐含在修正中)
        # 简化: 直接用 Sherman-Morrison 公式
        z_vec = du_temp.copy()
        # 解 A y = u_vec
        y_vec = np.zeros(N)
        d_y = np.zeros(N)
        c_y = np.zeros(N)
        d_y[0] = u_vec[0] / a_diag[0]
        c_y[0] = b_diag[0] / a_diag[0]
        for i in range(1, N - 1):
            m = a_diag[i] - c_diag[i] * c_y[i - 1]
            c_y[i] = b_diag[i] / m
            d_y[i] = (u_vec[i] - c_diag[i] * d_y[i - 1]) / m
        d_y[N - 1] = (u_vec[N - 1] - c_diag[N - 1] * d_y[N - 2]) / (
            a_diag[N - 1] - c_diag[N - 1] * c_y[N - 2])
        y_vec[N - 1] = d_y[N - 1]
        for i in range(N - 2, -1, -1):
            y_vec[i] = d_y[i] - c_y[i] * y_vec[i + 1]
        factor = np.dot(v_vec, z_vec) / (1.0 + np.dot(v_vec, y_vec))
        du = z_vec - factor * y_vec

    return du


def spectral_derivative(u, L_comoving):
    """谱方法 (FFT) 求一阶导数 (最高精度参考解).

    Parameters
    ----------
    u : array [N]
    L_comoving : float
        盒尺寸 [cm]

    Returns
    -------
    du_dx : array [N]
    """
    N = len(u)
    u_hat = np.fft.fft(u)
    k_arr = 2.0 * np.pi * np.fft.fftfreq(N, d=L_comoving / N)
    du_hat = 1j * k_arr * u_hat
    return np.real(np.fft.ifft(du_hat))
