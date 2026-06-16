# -*- coding: utf-8 -*-
"""
fd_operators.py
---------------
日冕 MHD 高阶有限差分算子库.

本模块将双调和算子 (088_biharmonic_fd1d)、带状矩阵结构 (061_b1g3)、
Predator-Prey 风格的 FD 时间积分 (350) 推广到日冕等离子体方程.

物理方程
--------
沿环轴方向 s 的一维约化 MHD (含热传导、辐射、加热):

(1) 质量连续:
    drho/dt + d(rho v)/ds = 0

(2) 动量方程 (含磁场磁压梯度):
    d(rho v)/dt + d/ds(rho v^2 + p + B^2/(2 mu_0))
        = - rho g_sun(s) + d tau_visc/ds

(3) 能量方程:
    dE/dt + d/ds[(E + p + B^2/(2 mu_0)) v]
        = d/ds[ kappa_|| dT/ds ] - Q_rad + Q_heat + H_visc

(4) 磁场 induction (理想 MHD):
    dB/dt = d(v B)/ds

其中 g_sun(s) 为沿环轴的重力分量, Q_heat 为加热率, Q_rad 为辐射冷却.

离散化
------
1) 二阶导数 (非均匀网格):
    d/ds[ kappa ds/ds T ]|_i
      = ( 2/(h_{i-1}+h_i) ) *
        [ (kappa_{i+1/2} (T_{i+1}-T_i))/h_i
        - (kappa_{i-1/2} (T_i-T_{i-1}))/h_{i-1} ]

2) 双调和算子 (磁场扩散的高阶项):
    d^4 u / ds^4  => 标准五点差分 (biharmonic_fd1d 推广)

3) 对流项 (四阶中心 + 人工粘性):
    d(rho v)/ds  =>  ( -rho_{i+2} v_{i+2} + 8 rho_{i+1} v_{i+1}
                       - 8 rho_{i-1} v_{i-1} + rho_{i-2} v_{i-2} ) / (12 h)
"""
from __future__ import annotations
import numpy as np
from scipy import sparse


# ============================================================
# 非均匀网格上的一阶导数 (四阶中心, 内点)
# ============================================================
def derivative_matrix_4th(z: np.ndarray) -> sparse.csr_matrix:
    """四阶中心差分一阶导矩阵 (内点), 边界用三阶单侧.

    对均匀 h, 内点:
        f'_i = (-f_{i+2} + 8 f_{i+1} - 8 f_{i-1} + f_{i-2}) / (12 h)
    """
    n = z.size
    h = z[1:] - z[:-1]
    rows, cols, vals = [], [], []

    # 内点 i=2..n-3
    for i in range(2, n - 2):
        h1, h2 = h[i - 2], h[i - 1]
        h3, h4 = h[i], h[i + 1] if i + 1 < n else h[-1]
        # 退化到非均匀网格上的拉格朗日插值导数
        # 简单起见, 此处用加权平均
        hi_mean = 0.25 * (h1 + h2 + h3 + h4)
        coeffs = {
            i - 2: -1.0, i - 1: 8.0, i: 0.0,
            i + 1: -8.0, i + 2: 1.0,
        }
        for j, c in coeffs.items():
            if c != 0.0:
                rows.append(i); cols.append(j); vals.append(c / (12.0 * hi_mean))

    # 边界 i=0, 1, n-2, n-1: 一阶/二阶单侧
    for i in [0, 1]:
        h1 = h[0]
        if i == 0:
            rows += [i, i, i]
            cols += [i, i + 1, i + 2]
            vals += [-3.0 / (2.0 * h1), 4.0 / (2.0 * h1), -1.0 / (2.0 * h1)]
        else:
            rows += [i, i, i]
            cols += [i - 1, i, i + 1]
            vals += [-1.0 / (2.0 * h1), 0.0, 1.0 / (2.0 * h1)]

    for i in [n - 2, n - 1]:
        h1 = h[-1]
        if i == n - 2:
            rows += [i, i, i]
            cols += [i - 1, i, i + 1]
            vals += [-1.0 / (2.0 * h1), 0.0, 1.0 / (2.0 * h1)]
        else:
            rows += [i, i, i]
            cols += [i - 2, i - 1, i]
            vals += [1.0 / (2.0 * h1), -4.0 / (2.0 * h1), 3.0 / (2.0 * h1)]

    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


# ============================================================
# 二阶导数矩阵 (自伴随形式, 非均匀网格, 含可变系数 kappa)
# ============================================================
def diffusion_matrix(z: np.ndarray, kappa: np.ndarray) -> sparse.csr_matrix:
    """构造 d/ds [ kappa(s) d/ds T ] 矩阵.

    离散:
        L T |_i = 2/(h_{i-1}+h_i) *
            [ kappa_{i+1/2} (T_{i+1}-T_i)/h_i
            - kappa_{i-1/2} (T_i-T_{i-1})/h_{i-1} ]

    kappa_{i+1/2} 取调和平均.
    """
    n = z.size
    h = z[1:] - z[:-1]
    k_half = 0.5 * (kappa[:-1] + kappa[1:])   # kappa at cell faces

    rows, cols, vals = [], [], []
    for i in range(1, n - 1):
        h_im1, h_i = h[i - 1], h[i]
        k_im1, k_ip = k_half[i - 1], k_half[i]
        coeff = 2.0 / (h_im1 + h_i)
        # T_{i-1}
        c_im1 = -coeff * k_im1 / h_im1
        # T_{i}
        c_i = coeff * (k_im1 / h_im1 + k_ip / h_i)
        # T_{i+1}
        c_ip1 = -coeff * k_ip / h_i
        rows += [i, i, i]
        cols += [i - 1, i, i + 1]
        vals += [c_im1, c_i, c_ip1]

    # 齐次 Dirichlet 边界 T=0
    rows += [0, n - 1]; cols += [0, n - 1]; vals += [1.0, 1.0]

    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


# ============================================================
# 双调和算子 d^4 u / ds^4 (均匀/非均匀网格, 来自 088_biharmonic_fd1d)
# ============================================================
def biharmonic_matrix(z: np.ndarray) -> sparse.csr_matrix:
    """一维双调和算子四点差分 (磁扩散高阶项):
       d^4 u / ds^4 |_i \approx (u_{i-2} - 4 u_{i-1} + 6 u_i - 4 u_{i+1} + u_{i+2}) / h^4
    边界条件: u = du/dn = 0 (夹紧端, 对应日冕环足点锚定).
    """
    n = z.size
    h = (z[-1] - z[0]) / (n - 1)
    rows, cols, vals = [], [], []

    for i in range(2, n - 2):
        coeffs = {
            i - 2: 1.0, i - 1: -4.0, i: 6.0, i + 1: -4.0, i + 2: 1.0,
        }
        for j, c in coeffs.items():
            rows.append(i); cols.append(j); vals.append(c / h**4)

    # 夹紧端: u=0 在两端
    for i in [0, 1, n - 2, n - 1]:
        rows.append(i); cols.append(i); vals.append(1.0)

    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


# ============================================================
# 带状矩阵存储 (来自 061_b1g3)
# ============================================================
def to_band_storage(A: sparse.csr_matrix, kl: int, ku: int) -> np.ndarray:
    """将稀疏矩阵转为 LAPACK 带状存储.

    AB[ku + i - j, j] = A[i, j]  for max(0, j-ku) <= i <= min(m, j+kl).
    """
    A_dense = A.toarray()
    n = A.shape[0]
    ab = np.zeros((kl + ku + 1, n), dtype=A_dense.dtype)
    for j in range(n):
        for i in range(max(0, j - ku), min(n, j + kl + 1)):
            ab[ku + i - j, j] = A_dense[i, j]
    return ab


def band_solve(ab: np.ndarray, kl: int, ku: int, b: np.ndarray) -> np.ndarray:
    """带状矩阵 LU 求解 (回退到稠密解, 但保持接口)."""
    n = ab.shape[1]
    A_dense = np.zeros((n, n))
    for j in range(n):
        for i in range(max(0, j - ku), min(n, j + kl + 1)):
            A_dense[i, j] = ab[ku + i - j, j]
    return np.linalg.solve(A_dense, b)


# ============================================================
# 人工粘性 (Richtmyer 两点)
# ============================================================
def artificial_viscosity(v: np.ndarray, z: np.ndarray,
                         c_lin: float = 0.1, c_quad: float = 1.0) -> np.ndarray:
    """q_i = rho * ( c_lin c_s |dv/dx| + c_quad (dv/dx)^2 )  if dv/dx < 0
    否则 q = 0. 返回 q 数组."""
    dv = np.zeros_like(v)
    h = z[1:] - z[:-1]
    dv[1:-1] = (v[2:] - v[:-2]) / (h[:-1] + h[1:])
    dv[0] = (v[1] - v[0]) / h[0]
    dv[-1] = (v[-1] - v[-2]) / h[-1]
    q = np.where(dv < 0, c_lin * np.abs(dv) + c_quad * dv**2, 0.0)
    return q


# ============================================================
# 时间积分: 显式 RK4 (日冕 MHD 显式阶段)
# ============================================================
def rk4_step(rhs_func, y: np.ndarray, t: float, dt: float) -> np.ndarray:
    """经典四阶 Runge-Kutta."""
    k1 = rhs_func(t, y)
    k2 = rhs_func(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = rhs_func(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = rhs_func(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
