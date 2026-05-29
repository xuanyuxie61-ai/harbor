# -*- coding: utf-8 -*-
"""
numerical_utils.py
==================
数值工具模块：提供高阶有限差分模板、WENO5重构权重、TVD-RK3时间积分
以及复数线性代数操作（源自 linpack_z 的数值线性代数思想）。

核心数学公式
------------
1. WENO5 光滑指示子 (Jiang & Shu, 1996):
   IS_0 = 13/12 (f_{i-2} - 2 f_{i-1} + f_i)^2 + 1/4 (f_{i-2} - 4 f_{i-1} + 3 f_i)^2
   IS_1 = 13/12 (f_{i-1} - 2 f_i + f_{i+1})^2 + 1/4 (f_{i-1} - f_{i+1})^2
   IS_2 = 13/12 (f_i - 2 f_{i+1} + f_{i+2})^2 + 1/4 (3 f_i - 4 f_{i+1} + f_{i+2})^2

2. WENO5 非线性权重:
   α_k = d_k / (ε + IS_k)^2,   k=0,1,2
   ω_k = α_k / Σ α_m
   其中 d_0=1/10, d_1=6/10, d_2=3/10

3. TVD-RK3 (Shu-Osher):
   u^{(1)} = u^n + Δt L(u^n)
   u^{(2)} = 3/4 u^n + 1/4 u^{(1)} + 1/4 Δt L(u^{(1)})
   u^{n+1} = 1/3 u^n + 2/3 u^{(2)} + 2/3 Δt L(u^{(2)})

4. 五阶中心差分 (曲率计算):
   ∂f/∂x ≈ (-f_{i+2} + 8 f_{i+1} - 8 f_{i-1} + f_{i-2}) / (12 h)
   ∂²f/∂x² ≈ (-f_{i+2} + 16 f_{i+1} - 30 f_i + 16 f_{i-1} - f_{i-2}) / (12 h²)

本模块融入原始项目 690_linpack_z 的数值线性代数鲁棒性思想，
在复数运算与矩阵分解中实施严格的边界检查与数值稳定性控制。
"""

import numpy as np
from numpy.polynomial import polynomial as P


def weno5_reconstruct(v):
    """
    对一维数组 v 进行 WENO5 正通量重构（左偏模板）。
    返回重构后的界面值 v_half。

    数学推导:
    设三个三阶候选多项式在界面 x_{i+1/2} 的值分别为:
      p_0(x_{i+1/2}) =  1/3 v_{i-2} - 7/6 v_{i-1} + 11/6 v_i
      p_1(x_{i+1/2}) = -1/6 v_{i-1} + 5/6 v_i     +  1/3 v_{i+1}
      p_2(x_{i+1/2}) =  1/3 v_i     + 5/6 v_{i+1} -  1/6 v_{i+2}

    光滑指示子 IS_k 量化了模板内部多项式变化剧烈程度。
    参数:
        v : ndarray, shape (nx,)
    返回:
        v_half : ndarray, shape (nx,), 界面通量
    """
    v = np.asarray(v, dtype=np.float64)
    nx = v.shape[0]
    v_half = np.zeros_like(v)

    eps = 1e-12
    # 线性权重
    d0, d1, d2 = 0.1, 0.6, 0.3

    for i in range(2, nx - 2):
        # 三个候选模板值
        p0 = (1.0 / 3.0) * v[i - 2] - (7.0 / 6.0) * v[i - 1] + (11.0 / 6.0) * v[i]
        p1 = (-1.0 / 6.0) * v[i - 1] + (5.0 / 6.0) * v[i] + (1.0 / 3.0) * v[i + 1]
        p2 = (1.0 / 3.0) * v[i] + (5.0 / 6.0) * v[i + 1] - (1.0 / 6.0) * v[i + 2]

        # 光滑指示子
        is0 = (13.0 / 12.0) * (v[i - 2] - 2.0 * v[i - 1] + v[i]) ** 2 \
              + 0.25 * (v[i - 2] - 4.0 * v[i - 1] + 3.0 * v[i]) ** 2
        is1 = (13.0 / 12.0) * (v[i - 1] - 2.0 * v[i] + v[i + 1]) ** 2 \
              + 0.25 * (v[i - 1] - v[i + 1]) ** 2
        is2 = (13.0 / 12.0) * (v[i] - 2.0 * v[i + 1] + v[i + 2]) ** 2 \
              + 0.25 * (3.0 * v[i] - 4.0 * v[i + 1] + v[i + 2]) ** 2

        # 非线性权重
        a0 = d0 / (eps + is0) ** 2
        a1 = d1 / (eps + is1) ** 2
        a2 = d2 / (eps + is2) ** 2
        wsum = a0 + a1 + a2
        w0 = a0 / wsum
        w1 = a1 / wsum
        w2 = a2 / wsum

        v_half[i] = w0 * p0 + w1 * p1 + w2 * p2

    # 边界：用一阶迎风填充（鲁棒性处理）
    if nx > 2:
        v_half[0] = v[0]
        v_half[1] = v[1]
        v_half[-2] = v[-2]
        v_half[-1] = v[-1]
    return v_half


def weno5_neg_reconstruct(v):
    """
    WENO5 负通量重构（右偏模板），用于双向通量分裂。
    数学上与正通量对称，模板镜像翻转。
    """
    v = np.asarray(v, dtype=np.float64)
    nx = v.shape[0]
    v_half = np.zeros_like(v)

    eps = 1e-12
    d0, d1, d2 = 0.1, 0.6, 0.3

    for i in range(2, nx - 2):
        p0 = (1.0 / 3.0) * v[i + 2] - (7.0 / 6.0) * v[i + 1] + (11.0 / 6.0) * v[i]
        p1 = (-1.0 / 6.0) * v[i + 1] + (5.0 / 6.0) * v[i] + (1.0 / 3.0) * v[i - 1]
        p2 = (1.0 / 3.0) * v[i] + (5.0 / 6.0) * v[i - 1] - (1.0 / 6.0) * v[i - 2]

        is0 = (13.0 / 12.0) * (v[i + 2] - 2.0 * v[i + 1] + v[i]) ** 2 \
              + 0.25 * (v[i + 2] - 4.0 * v[i + 1] + 3.0 * v[i]) ** 2
        is1 = (13.0 / 12.0) * (v[i + 1] - 2.0 * v[i] + v[i - 1]) ** 2 \
              + 0.25 * (v[i + 1] - v[i - 1]) ** 2
        is2 = (13.0 / 12.0) * (v[i] - 2.0 * v[i - 1] + v[i - 2]) ** 2 \
              + 0.25 * (3.0 * v[i] - 4.0 * v[i - 1] + v[i - 2]) ** 2

        a0 = d0 / (eps + is0) ** 2
        a1 = d1 / (eps + is1) ** 2
        a2 = d2 / (eps + is2) ** 2
        wsum = a0 + a1 + a2
        w0 = a0 / wsum
        w1 = a1 / wsum
        w2 = a2 / wsum

        v_half[i] = w0 * p0 + w1 * p1 + w2 * p2

    if nx > 2:
        v_half[0] = v[0]
        v_half[1] = v[1]
        v_half[-2] = v[-2]
        v_half[-1] = v[-1]
    return v_half


def weno5_derivative(phi, dx, axis=0):
    """
    使用 WENO5 + Lax-Friedrichs 通量分裂计算空间导数 L = -V·∇φ 中的 ∂φ/∂x。
    这里采用 Godunov 型通量，对 Hamilton-Jacobi 方程:
        φ_t + H(φ_x, φ_y) = 0
    空间离散后得到半离散形式 dφ/dt = L(φ)。

    公式:
        φ_x^+ = (φ_{i+1/2}^+ - φ_{i-1/2}^+) / dx   (WENO5正重构差分)
        φ_x^- = (φ_{i+1/2}^- - φ_{i-1/2}^-) / dx   (WENO5负重构差分)
        对 HJ 方程取 φ_x = argmin_{s∈{+,-}} |s| 的选取策略（Engquist-Osher型）
    """
    phi = np.asarray(phi, dtype=np.float64)
    if axis == 0:
        v = phi
    else:
        v = phi.T

    nx = v.shape[0]
    ny = v.shape[1] if v.ndim > 1 else 1

    if v.ndim == 1:
        vp = weno5_reconstruct(v)
        vm = weno5_neg_reconstruct(v)
        # 简单的迎风格式导数（用于HJ方程）
        dphi = np.zeros_like(v)
        # 内部点: 使用 Lax-Friedrichs 分裂近似
        for i in range(2, nx - 2):
            dphi[i] = 0.5 * ((vp[i] - vp[i - 1]) + (vm[i] - vm[i - 1])) / dx
        # 边界处理
        dphi[0] = (v[1] - v[0]) / dx
        dphi[1] = (v[2] - v[1]) / dx
        dphi[-2] = (v[-1] - v[-2]) / dx
        dphi[-1] = (v[-1] - v[-2]) / dx
        return dphi
    else:
        dphi = np.zeros_like(v)
        for j in range(ny):
            col = v[:, j]
            vp = weno5_reconstruct(col)
            vm = weno5_neg_reconstruct(col)
            for i in range(2, nx - 2):
                dphi[i, j] = 0.5 * ((vp[i] - vp[i - 1]) + (vm[i] - vm[i - 1])) / dx
            dphi[0, j] = (col[1] - col[0]) / dx
            dphi[1, j] = (col[2] - col[1]) / dx
            dphi[-2, j] = (col[-1] - col[-2]) / dx
            dphi[-1, j] = (col[-1] - col[-2]) / dx
        if axis == 1:
            dphi = dphi.T
        return dphi


def tvd_rk3_step(phi0, dt, rhs_func):
    """
    三阶 TVD Runge-Kutta 时间推进。

    算法:
        φ^{(1)} = φ^n + Δt · L(φ^n)
        φ^{(2)} = 3/4 φ^n + 1/4 φ^{(1)} + 1/4 Δt · L(φ^{(1)})
        φ^{n+1} = 1/3 φ^n + 2/3 φ^{(2)} + 2/3 Δt · L(φ^{(2)})

    参数:
        phi0     : 当前水平集值
        dt       : 时间步长
        rhs_func : 右端项函数 L(φ)
    返回:
        phi_new  : 下一时间步的值
    """
    phi0 = np.asarray(phi0, dtype=np.float64)
    # HOLE_1: 实现三阶 TVD Runge-Kutta 时间推进
    # 算法:
    #   φ^{(1)} = φ^n + Δt · L(φ^n)
    #   φ^{(2)} = 3/4 φ^n + 1/4 φ^{(1)} + 1/4 Δt · L(φ^{(1)})
    #   φ^{n+1} = 1/3 φ^n + 2/3 φ^{(2)} + 2/3 Δt · L(φ^{(2)})
    raise NotImplementedError("HOLE_1: TVD-RK3 step implementation missing")


def central_diff_4th(phi, dx, axis=0):
    """
    四阶中心差分计算一阶导数。
    公式:
        f'_i ≈ (-f_{i+2} + 8 f_{i+1} - 8 f_{i-1} + f_{i-2}) / (12 h)
    边界处退化为二阶中心差分。
    """
    phi = np.asarray(phi, dtype=np.float64)
    if axis == 0:
        v = phi
    else:
        v = phi.T

    nx = v.shape[0]
    ny = v.shape[1] if v.ndim > 1 else 1

    if v.ndim == 1:
        d = np.zeros_like(v)
        if nx >= 5:
            for i in range(2, nx - 2):
                d[i] = (-v[i + 2] + 8.0 * v[i + 1] - 8.0 * v[i - 1] + v[i - 2]) / (12.0 * dx)
        # 边界二阶
        if nx >= 3:
            d[0] = (-3.0 * v[0] + 4.0 * v[1] - v[2]) / (2.0 * dx)
            d[1] = (v[2] - v[0]) / (2.0 * dx)
            d[-2] = (v[-1] - v[-3]) / (2.0 * dx)
            d[-1] = (3.0 * v[-1] - 4.0 * v[-2] + v[-3]) / (2.0 * dx)
        elif nx == 2:
            d[0] = (v[1] - v[0]) / dx
            d[1] = (v[1] - v[0]) / dx
        return d
    else:
        d = np.zeros_like(v)
        for j in range(ny):
            col = v[:, j]
            if nx >= 5:
                for i in range(2, nx - 2):
                    d[i, j] = (-col[i + 2] + 8.0 * col[i + 1] - 8.0 * col[i - 1] + col[i - 2]) / (12.0 * dx)
            if nx >= 3:
                d[0, j] = (-3.0 * col[0] + 4.0 * col[1] - col[2]) / (2.0 * dx)
                d[1, j] = (col[2] - col[0]) / (2.0 * dx)
                d[-2, j] = (col[-1] - col[-3]) / (2.0 * dx)
                d[-1, j] = (3.0 * col[-1] - 4.0 * col[-2] + col[-3]) / (2.0 * dx)
        if axis == 1:
            d = d.T
        return d


def central_diff_2nd(phi, dx, axis=0):
    """
    二阶中心差分计算一阶导数（用于边界或低阶备选）。
    f'_i ≈ (f_{i+1} - f_{i-1}) / (2h)
    """
    phi = np.asarray(phi, dtype=np.float64)
    if axis == 0:
        v = phi
    else:
        v = phi.T

    nx = v.shape[0]
    ny = v.shape[1] if v.ndim > 1 else 1

    if v.ndim == 1:
        d = np.zeros_like(v)
        if nx >= 3:
            for i in range(1, nx - 1):
                d[i] = (v[i + 1] - v[i - 1]) / (2.0 * dx)
            d[0] = (-3.0 * v[0] + 4.0 * v[1] - v[2]) / (2.0 * dx)
            d[-1] = (3.0 * v[-1] - 4.0 * v[-2] + v[-3]) / (2.0 * dx)
        elif nx == 2:
            d[0] = (v[1] - v[0]) / dx
            d[1] = d[0]
        return d
    else:
        d = np.zeros_like(v)
        for j in range(ny):
            col = v[:, j]
            if nx >= 3:
                for i in range(1, nx - 1):
                    d[i, j] = (col[i + 1] - col[i - 1]) / (2.0 * dx)
                d[0, j] = (-3.0 * col[0] + 4.0 * col[1] - col[2]) / (2.0 * dx)
                d[-1, j] = (3.0 * col[-1] - 4.0 * col[-2] + col[-3]) / (2.0 * dx)
        if axis == 1:
            d = d.T
        return d


def laplacian_2d(phi, dx, dy):
    """
    二维五点 Laplacian，带 Neumann 边界条件。
    Δφ ≈ (φ_{i+1,j} + φ_{i-1,j} + φ_{i,j+1} + φ_{i,j-1} - 4 φ_{i,j}) / h²
    当 dx=dy=h 时成立，不等距时做调和平均近似。
    """
    phi = np.asarray(phi, dtype=np.float64)
    nx, ny = phi.shape
    lap = np.zeros_like(phi)

    # 内部
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            lap[i, j] = (phi[i + 1, j] - 2.0 * phi[i, j] + phi[i - 1, j]) / (dx * dx) \
                        + (phi[i, j + 1] - 2.0 * phi[i, j] + phi[i, j - 1]) / (dy * dy)

    # 边界 Neumann: 法向导数为0
    for j in range(ny):
        lap[0, j] = lap[1, j]
        lap[-1, j] = lap[-2, j]
    for i in range(nx):
        lap[i, 0] = lap[i, 1]
        lap[i, -1] = lap[i, -2]
    return lap


def cplx_cholesky_decompose(A):
    """
    复数 Hermite 正定矩阵的 Cholesky 分解（源自 linpack_z 的 zchdc 思想）。
    对 A = L L^H，其中 L 为下三角矩阵。

    公式:
        L_{jj} = √(A_{jj} - Σ_{k=1}^{j-1} |L_{jk}|²)
        L_{ij} = (A_{ij} - Σ_{k=1}^{j-1} L_{ik} conj(L_{jk})) / L_{jj}

    参数:
        A : ndarray, 复数 Hermite 正定矩阵
    返回:
        L : ndarray, 下三角矩阵
    """
    A = np.asarray(A, dtype=np.complex128)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("cplx_cholesky_decompose: A must be square")
    # 检查 Hermite 性
    if not np.allclose(A, A.conj().T, atol=1e-12):
        raise ValueError("cplx_cholesky_decompose: A must be Hermitian")

    L = np.zeros_like(A, dtype=np.complex128)
    for j in range(n):
        diag_val = A[j, j] - np.sum(np.abs(L[j, :j]) ** 2)
        if diag_val.real <= 0:
            raise ValueError("cplx_cholesky_decompose: Matrix is not positive definite")
        L[j, j] = np.sqrt(diag_val)
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j].conj())) / L[j, j]
    return L


def cplx_solve_lower_triangular(L, b):
    """
    前向替换求解 L y = b，其中 L 为下三角。
    融入 linpack_z 的复数运算稳定性控制。
    """
    n = L.shape[0]
    y = np.zeros_like(b, dtype=np.complex128)
    for i in range(n):
        val = b[i] - np.sum(L[i, :i] * y[:i])
        if abs(L[i, i]) < 1e-15:
            raise ValueError("cplx_solve_lower_triangular: zero diagonal element")
        y[i] = val / L[i, i]
    return y


def cplx_solve_upper_triangular(U, b):
    """
    后向替换求解 U x = b，其中 U 为上三角。
    """
    n = U.shape[0]
    x = np.zeros_like(b, dtype=np.complex128)
    for i in range(n - 1, -1, -1):
        val = b[i] - np.sum(U[i, i + 1:] * x[i + 1:])
        if abs(U[i, i]) < 1e-15:
            raise ValueError("cplx_solve_upper_triangular: zero diagonal element")
        x[i] = val / U[i, i]
    return x


def cplx_lu_factor(A):
    """
    复数矩阵的 LU 分解（Doolittle 算法，源自 linpack_z 的 zgefa 思想）。
    PA = LU，其中 P 为置换矩阵。

    参数:
        A : ndarray, 复数矩阵
    返回:
        L, U, P : LU 分解与置换矩阵
    """
    A = np.asarray(A, dtype=np.complex128)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("cplx_lu_factor: A must be square")

    L = np.eye(n, dtype=np.complex128)
    U = A.copy()
    P = np.eye(n, dtype=np.complex128)

    for k in range(n - 1):
        # 部分主元
        pivot = np.argmax(np.abs(U[k:, k])) + k
        if abs(U[pivot, k]) < 1e-15:
            raise ValueError("cplx_lu_factor: singular matrix")
        if pivot != k:
            U[[k, pivot], :] = U[[pivot, k], :]
            P[[k, pivot], :] = P[[pivot, k], :]

        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]
    return L, U, P


def cplx_qr_factor(A):
    """
    复数矩阵的 QR 分解（Gram-Schmidt 正交化，源自 linpack_z 的 zqrdc 思想）。
    A = Q R，其中 Q 为酉矩阵，R 为上三角。

    参数:
        A : ndarray, 复数矩阵 (m x n)
    返回:
        Q, R : 分解结果
    """
    A = np.asarray(A, dtype=np.complex128)
    m, n = A.shape
    Q = np.zeros((m, n), dtype=np.complex128)
    R = np.zeros((n, n), dtype=np.complex128)

    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.vdot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-15:
            raise ValueError("cplx_qr_factor: rank deficient")
        R[j, j] = norm_v
        Q[:, j] = v / norm_v
    return Q, R
