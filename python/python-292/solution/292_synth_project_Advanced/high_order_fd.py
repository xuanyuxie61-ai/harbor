"""
high_order_fd.py — 高阶有限差分算子
=====================================

融合自:
  - 393_fem1d_lagrange: Lagrange 多项式基函数与导数构造
  - 605_jacobi_exactness: 多项式精确度验证

物理背景:
  磁重联 MHD 方程中对流项 d(vB)/dx 和扩散项 eta*d^2B/dx^2
  需要高阶精度以准确捕获撕裂模不稳定性增长率 gamma ~ S^{-3/5}。

  2p 阶中心差分:
    f'(x_i) = sum_{k=-p}^{p} c_k * f(x_{i+k}) / dx
  系数 c_k 由 Vandermonde 系统求解:
    sum_k c_k * k^m = delta_{m,1}  for m = 0, 1, ..., 2p

  色散关系验证:
    对 f = exp(i*k*x)，数值波数 k_num 与精确波数 k 的偏差
    k_num * dx = -i * sum c_k * exp(i*k*dx*k)
"""

import numpy as np


class HighOrderFD:
    """高阶有限差分算子，支持中心差分和迎风差分。"""

    def __init__(self, order=6, nx=128, dx=0.1):
        self.order = order
        self.nx = nx
        self.dx = dx
        self.half_width = order // 2
        self.central_coeffs = None
        self.central_2nd = None
        self.upwind_coeffs = None
        self.max_dispersion_error = None

    def build_central_stencils(self):
        """构建中心差分系数 (一阶和二阶导数)。"""
        p = self.half_width
        n_stencil = 2 * p + 1
        k_vals = np.arange(-p, p + 1, dtype=float)
        V = np.zeros((n_stencil, n_stencil))
        for m in range(n_stencil):
            V[m, :] = k_vals ** m
        rhs1 = np.zeros(n_stencil)
        rhs1[1] = 1.0
        self.central_coeffs = np.linalg.solve(V, rhs1)
        rhs2 = np.zeros(n_stencil)
        rhs2[2] = 2.0
        self.central_2nd = np.linalg.solve(V, rhs2)

    def build_upwind_stencils(self):
        """构建迎风差分系数。"""
        p = self.half_width
        n_stencil = 2 * p
        k_vals = np.arange(-p + 1, p + 1, dtype=float)
        V = np.zeros((n_stencil, n_stencil))
        for m in range(n_stencil):
            V[m, :] = k_vals ** m
        rhs = np.zeros(n_stencil)
        rhs[1] = 1.0
        self.upwind_coeffs = np.linalg.solve(V, rhs)

    def apply_first_derivative(self, f):
        """对周期数组施加中心差分一阶导数。"""
        p = self.half_width
        df = np.zeros_like(f)
        for k_idx in range(len(self.central_coeffs)):
            k = k_idx - p
            df += self.central_coeffs[k_idx] * np.roll(f, -k) / self.dx
        return df

    def apply_second_derivative(self, f):
        """对周期数组施加中心差分二阶导数。"""
        p = self.half_width
        d2f = np.zeros_like(f)
        for k_idx in range(len(self.central_2nd)):
            k = k_idx - p
            d2f += self.central_2nd[k_idx] * np.roll(f, -k) / (self.dx ** 2)
        return d2f

    def apply_upwind_derivative(self, f, velocity_sign=1.0):
        """迎风差分一阶导数。"""
        p = self.half_width
        df = np.zeros_like(f)
        if velocity_sign > 0:
            coeffs = self.upwind_coeffs
        else:
            coeffs = -self.upwind_coeffs[::-1]
        for k_idx in range(len(coeffs)):
            k = k_idx - p + 1
            df += coeffs[k_idx] * np.roll(f, -k) / self.dx
        return df

    def compute_dispersion_relation(self):
        """计算数值色散关系误差。"""
        p = self.half_width
        n_k = 200
        kdx = np.linspace(1e-6, np.pi, n_k)
        errors = np.zeros(n_k)
        for idx, kd in enumerate(kdx):
            k_num_complex = 0.0 + 0.0j
            for m, m_val in enumerate(range(-p, p + 1)):
                k_num_complex += self.central_coeffs[m] * np.exp(1j * kd * m_val)
            k_num = -1j * k_num_complex
            errors[idx] = abs(k_num / kd - 1.0) if abs(kd) > 1e-12 else 0.0
        self.max_dispersion_error = float(np.max(errors))

    def verify_exactness(self, poly_degree=None):
        """验证差分算子对多项式的精确度。"""
        if poly_degree is None:
            poly_degree = self.order
        p = self.half_width
        n_pts = max(2 * p + 5, 20)
        x = np.linspace(-1, 1, n_pts)
        dx = x[1] - x[0]
        self.dx = dx
        max_error = 0.0
        for d in range(1, poly_degree + 1):
            f = x ** d
            df_num = self.apply_first_derivative(f)
            df_exact = d * x[1:-1] ** max(0, d - 1)
            df_num_inner = df_num[1:-1]
            if len(df_num_inner) == len(df_exact):
                err = np.max(np.abs(df_num_inner - df_exact))
                max_error = max(max_error, err)
        self.dx = 0.1
        return max_error
