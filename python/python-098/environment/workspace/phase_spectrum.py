# -*- coding: utf-8 -*-
"""
phase_spectrum.py
基于 r8poly（Horner 求值、多项式运算）与 polpak/Legendre 多项式，
构建超表面全息相位剖面的谱展开设计模块。

核心科学问题：
  利用正交多项式（Legendre、Chebyshev）对二维全息相位函数 φ(x,y) 进行谱展开，
  实现低阶模态驱动的超表面相位调控：
      φ(x, y) = Σ_{m=0}^{M} Σ_{n=0}^{N} c_{mn} P_m(2x/L_x) P_n(2y/L_y)
  其中 P_m 为 m 阶 Legendre 多项式。

关键公式：
  1. Horner 法则（r8poly_value_horner）:
       p(x) = c_0 + x·(c_1 + x·(c_2 + ... ))
  2. Legendre 递推（legendre_poly）:
       P_0(x) = 1
       P_1(x) = x
       P_n(x) = ((2n-1)·x·P_{n-1}(x) - (n-1)·P_{n-2}(x)) / n
  3. 导数递推:
       P'_0(x) = 0
       P'_1(x) = 1
       P'_n(x) = ((2n-1)·(P_{n-1}(x) + x·P'_{n-1}(x)) - (n-1)·P'_{n-2}(x)) / n
  4. Chebyshev 节点（谱配置点）:
       x_k = cos((2k+1)π / (2N+2)),  k = 0,...,N
  5. 二维相位展开:
       φ(x,y) = Σ_m Σ_n c_{mn} P_m(x̃) P_n(ỹ),  x̃ = 2x/L_x, ỹ = 2y/L_y
"""

import numpy as np


def horner_eval(coeffs, x):
    """
    Horner 法则求多项式值（参考 r8poly_value_horner）。
    coeffs: [c0, c1, ..., cd]
    p(x) = c0 + c1*x + c2*x^2 + ... + cd*x^d
    """
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    d = coeffs.shape[0] - 1
    if d < 0:
        return np.zeros_like(x)
    value = np.full_like(x, coeffs[d])
    for i in range(d - 1, -1, -1):
        value = value * x + coeffs[i]
    return value


def legendre_polynomials(n, x):
    """
    计算 0~n 阶 Legendre 多项式及其导数（参考 legendre_poly）。
    返回 (cx, cpx)，其中 cx[i] = P_i(x), cpx[i] = P'_i(x)。
    """
    x = float(x)
    if n < 0:
        return np.array([]), np.array([])
    cx = np.zeros(n + 1)
    cpx = np.zeros(n + 1)
    cx[0] = 1.0
    cpx[0] = 0.0
    if n < 1:
        return cx, cpx
    cx[1] = x
    cpx[1] = 1.0
    for i in range(2, n + 1):
        cx[i] = ((2.0 * i - 1.0) * x * cx[i - 1] - (i - 1.0) * cx[i - 2]) / i
        cpx[i] = ((2.0 * i - 1.0) * (cx[i - 1] + x * cpx[i - 1]) -
                  (i - 1.0) * cpx[i - 2]) / i
    return cx, cpx


def legendre_polynomials_array(n, x_arr):
    """
    对数组 x_arr 批量计算 Legendre 多项式值，返回形状 (len(x_arr), n+1)。
    """
    x_arr = np.asarray(x_arr, dtype=float)
    m = x_arr.shape[0]
    cx = np.zeros((m, n + 1))
    if n >= 0:
        cx[:, 0] = 1.0
    if n >= 1:
        cx[:, 1] = x_arr
    for i in range(2, n + 1):
        cx[:, i] = ((2.0 * i - 1.0) * x_arr * cx[:, i - 1] -
                    (i - 1.0) * cx[:, i - 2]) / i
    return cx


def chebyshev_nodes(n):
    """
    返回 n 阶 Chebyshev 节点（参考 r8poly_chebyshev）。
    x_k = cos((2k+1)π / (2n)), k=0,...,n-1
    """
    k = np.arange(n, dtype=float)
    return np.cos((2.0 * k + 1.0) * np.pi / (2.0 * n))


def chebyshev_polynomials(n, x):
    """
    计算第一类 Chebyshev 多项式 T_0(x)...T_n(x)。
    T_0 = 1, T_1 = x, T_n = 2x·T_{n-1} - T_{n-2}
    """
    x = float(x)
    if n < 0:
        return np.array([])
    T = np.zeros(n + 1)
    T[0] = 1.0
    if n >= 1:
        T[1] = x
    for i in range(2, n + 1):
        T[i] = 2.0 * x * T[i - 1] - T[i - 2]
    return T


def design_hologram_phase_2d(x_grid, y_grid, legendre_coeffs, Lx=1.0, Ly=1.0):
    """
    利用二维 Legendre 张量积展开设计全息相位剖面。

    参数:
        x_grid, y_grid: 1-D array
        legendre_coeffs: 2-D array shape (M+1, N+1)
    返回:
        phase: 2-D array shape (len(y_grid), len(x_grid))
    """
    legendre_coeffs = np.asarray(legendre_coeffs, dtype=float)
    M, N = legendre_coeffs.shape
    M -= 1
    N -= 1
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)

    # HOLE 1: 实现二维 Legendre 张量积相位合成
    # 需要将 x_grid, y_grid 映射到 [-1, 1]，计算 Legendre 多项式矩阵，
    # 并通过张量积展开 phase(y, x) = Σ_m Σ_n c_{mn} P_m(x̃) P_n(ỹ)
    raise NotImplementedError("Hole 1: design_hologram_phase_2d 需要实现")


def reconstruct_phase_from_spectrum(phase_samples, x_nodes, y_nodes, max_degree):
    """
    从离散相位样本通过最小二乘拟合 Legendre 谱系数。
    对应离散广义 Fourier-Legendre 变换：
        c_{mn} = (2m+1)(2n+1)/4 ∫∫ φ(x,y) P_m(x) P_n(y) dx dy
    这里用配置点最小二乘逼近。
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    phase_samples = np.asarray(phase_samples, dtype=float)
    nx = x_nodes.shape[0]
    ny = y_nodes.shape[0]
    if phase_samples.shape != (ny, nx):
        raise ValueError("phase_samples shape must match (len(y_nodes), len(x_nodes))")

    x_tilde = np.clip(x_nodes, -1.0, 1.0)
    y_tilde = np.clip(y_nodes, -1.0, 1.0)

    Px = legendre_polynomials_array(max_degree, x_tilde)  # (nx, M+1)
    Py = legendre_polynomials_array(max_degree, y_tilde)  # (ny, N+1)

    # 构建最小二乘问题：将 2D 展平为 1D
    A = np.kron(Px, Py)  # (nx*ny, (M+1)*(N+1))
    b = phase_samples.ravel()
    # 使用正规方程（小矩阵）
    ATA = A.T @ A
    ATb = A.T @ b
    # Tikhonov 正则化保证数值稳定
    reg = 1e-10 * np.eye(ATA.shape[0])
    coeffs_flat = np.linalg.solve(ATA + reg, ATb)
    coeffs = coeffs_flat.reshape(max_degree + 1, max_degree + 1)
    return coeffs


def phase_gradient_2d(x_grid, y_grid, legendre_coeffs, Lx=1.0, Ly=1.0):
    """
    计算二维相位剖面的梯度 ∇φ = (∂φ/∂x, ∂φ/∂y)。
    利用 Legendre 多项式导数的解析表达式：
        ∂φ/∂x = Σ_m Σ_n c_{mn} · (2/Lx) · P'_m(2x/Lx) · P_n(2y/Ly)
    """
    legendre_coeffs = np.asarray(legendre_coeffs, dtype=float)
    M, N = legendre_coeffs.shape
    M -= 1
    N -= 1
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    x_tilde = np.clip(2.0 * x_grid / Lx, -1.0, 1.0)
    y_tilde = np.clip(2.0 * y_grid / Ly, -1.0, 1.0)

    # 计算导数矩阵
    Px = np.zeros((len(x_grid), M + 1))
    Py = np.zeros((len(y_grid), N + 1))
    for i, xt in enumerate(x_tilde):
        _, dPx = legendre_polynomials(M, xt)
        Px[i, :] = dPx
    for j, yt in enumerate(y_tilde):
        _, dPy = legendre_polynomials(N, yt)
        Py[j, :] = dPy

    dphi_dx = (2.0 / Lx) * np.einsum('ij,ni,mj->mn', legendre_coeffs, Py, Px)

    # 重新计算 P_n 值用于 y 导数
    Px_val = legendre_polynomials_array(M, x_tilde)
    Py_val = legendre_polynomials_array(N, y_tilde)
    # 需要导数版本重新计算
    Px_d = np.zeros_like(Px_val)
    Py_d = np.zeros_like(Py_val)
    for i, xt in enumerate(x_tilde):
        _, dPx = legendre_polynomials(M, xt)
        Px_d[i, :] = dPx
    for j, yt in enumerate(y_tilde):
        _, dPy = legendre_polynomials(N, yt)
        Py_d[j, :] = dPy

    dphi_dy = (2.0 / Ly) * np.einsum('ij,ni,mj->mn', legendre_coeffs, Py_d, Px_val)
    return dphi_dx, dphi_dy
