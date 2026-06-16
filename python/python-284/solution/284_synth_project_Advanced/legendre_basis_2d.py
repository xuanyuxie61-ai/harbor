# -*- coding: utf-8 -*-
"""
legendre_basis_2d.py — Legendre 乘积多项式基
================================================
核心科学问题: 使用 Legendre 多项式作为谱方法基函数
展开波函数和势能, 用于高精度求解薛定谔方程.

融合种子项目:
  - 664_legendre_product_polynomial: Legendre 乘积多项式展开
"""

import numpy as np


def legendre_coefficients(n):
    """
    计算 n 阶 Legendre 多项式的系数.
    递推公式:
      P_0(x) = 1
      P_1(x) = x
      (n+1)P_{n+1}(x) = (2n+1)x·P_n(x) - n·P_{n-1}(x)

    返回系数数组 c, 使得 P_n(x) = sum_k c_k * x^k.
    """
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 1.0])

    # 递推
    P_prev = np.array([1.0])  # P_0
    P_curr = np.array([0.0, 1.0])  # P_1

    for k in range(1, n):
        # (k+1)P_{k+1} = (2k+1)*x*P_k - k*P_{k-1}
        # x*P_k: 将 P_k 的系数上移一位
        xPk = np.zeros(len(P_curr) + 1)
        xPk[1:] = P_curr

        P_next = ((2 * k + 1) * xPk - k * np.pad(P_prev, (0, max(0, len(xPk) - len(P_prev))))) / (k + 1)
        # 确保长度一致
        min_len = min(len(P_next), max(len(xPk), len(P_prev) + 2))
        P_next = P_next[:min_len]

        P_prev = P_curr
        P_curr = P_next

    return P_curr


def legendre_value(n, x):
    """
    计算 P_n(x) (使用 Bonnet 递推, 数值稳定).
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()

    P_prev = np.ones_like(x)
    P_curr = x.copy()

    for k in range(1, n):
        P_next = ((2 * k + 1) * x * P_curr - k * P_prev) / (k + 1)
        P_prev = P_curr
        P_curr = P_next

    return P_curr


def legendre_product_polynomial(indices, x_multi):
    """
    Legendre 乘积多项式:
        L(x_1, ..., x_d) = P_{l_1}(x_1) * P_{l_2}(x_2) * ... * P_{l_d}(x_d)

    融合 664_legendre_product_polynomial.

    参数
    ----
    indices : tuple of int
        各维度的 Legendre 阶数 (l_1, ..., l_d)
    x_multi : ndarray, shape (..., d)
        各维度的坐标

    返回
    ----
    ndarray
        乘积多项式值
    """
    indices = tuple(indices)
    d = len(indices)
    x_multi = np.atleast_2d(x_multi)

    result = np.ones(x_multi.shape[0])
    for dim in range(d):
        result *= legendre_value(indices[dim], x_multi[:, dim])
    return result


class LegendreSpectralBasis:
    """
    Legendre 谱方法基.

    用于将薛定谔方程展开为 Legendre 级数:
        ψ(z) = sum_{n=0}^{N} a_n * P_n(ζ(z))

    其中 ζ(z) = (2z - (z_min+z_max))/(z_max-z_min) ∈ [-1,1]
    是映射到参考区间的坐标.
    """

    def __init__(self, N_basis, z_min, z_max):
        """
        参数
        ----
        N_basis : int
            基函数个数
        z_min, z_max : float
            物理域边界 [m]
        """
        self.N = N_basis
        self.z_min = z_min
        self.z_max = z_max
        self.L = z_max - z_min

    def map_to_reference(self, z):
        """
        物理坐标 → 参考坐标:
            ζ = (2z - z_min - z_max) / (z_max - z_min)
        """
        return (2.0 * z - self.z_min - self.z_max) / self.L

    def map_to_physical(self, zeta):
        """参考坐标 → 物理坐标."""
        return 0.5 * (self.L * zeta + self.z_min + self.z_max)

    def basis_function(self, n, z):
        """第 n 个基函数 P_n(ζ(z))."""
        zeta = self.map_to_reference(z)
        return legendre_value(n, np.atleast_1d(zeta))

    def basis_derivative(self, n, z):
        """
        基函数导数:
            dP_n/dz = dP_n/dζ · dζ/dz = (2/L) · P'_n(ζ)

        P'_n(ζ) 递推:
            P'_n = (2n-1)P_{n-1} + P'_{n-2}  (简化)
        或直接用:
            (1-ζ²)P'_n = n(P_{n-1} - ζP_n)
        """
        zeta = self.map_to_reference(z)
        zeta = np.atleast_1d(zeta)

        if n == 0:
            return np.zeros_like(zeta)

        Pn = legendre_value(n, zeta)
        Pn_1 = legendre_value(n - 1, zeta)

        # 避免 ζ = ±1
        one_minus_zeta2 = np.maximum(1.0 - zeta**2, 1e-14)
        dPn_dzeta = n * (Pn_1 - zeta * Pn) / one_minus_zeta2

        return (2.0 / self.L) * dPn_dzeta

    def build_mass_matrix(self):
        """
        质量矩阵 M_{mn} = <P_m|P_n>:
            M_{mn} = (L/2) * integral_{-1}^{1} P_m(ζ) P_n(ζ) dζ
                    = (L/2) * 2/(2n+1) * δ_{mn}
        """
        M = np.zeros((self.N, self.N))
        for n in range(self.N):
            M[n, n] = self.L / (2 * n + 1)
        return M

    def build_stiffness_matrix(self):
        """
        刚度矩阵 S_{mn} = <P'_m|P'_n>:
            S_{mn} = (2/L) * integral_{-1}^{1} P'_m(ζ) P'_n(ζ) dζ

        解析公式:
            S_{mn} = (2/L) * sum over appropriate terms
        对于 Legendre: S_{mn} = 0 if m+n is odd
        """
        S = np.zeros((self.N, self.N))
        for m in range(self.N):
            for n in range(self.N):
                if (m + n) % 2 == 0 and m != n:
                    # 非对角项 (m+n 偶数, m≠n)
                    # S_mn = 2/L * min(m,n)*(min(m,n)+1) if |m-n| > 0 and even
                    p = min(m, n)
                    if abs(m - n) >= 2 and (m - n) % 2 == 0:
                        S[m, n] = (2.0 / self.L) * p * (p + 1)
                elif m == n and m > 0:
                    S[m, m] = (2.0 / self.L) * m * (m + 1) / (2.0)
        return S

    def build_potential_matrix(self, V_func, n_quad=None):
        """
        势能矩阵 V_{mn} = <P_m|V|P_n>:
            V_{mn} = (L/2) integral_{-1}^{1} V(ζ) P_m(ζ) P_n(ζ) dζ

        用 Gauss-Legendre 求积.
        """
        if n_quad is None:
            n_quad = max(2 * self.N, 20)

        # Gauss-Legendre 节点和权重
        zeta_q, w_q = np.polynomial.legendre.leggauss(n_quad)
        z_q = self.map_to_physical(zeta_q)
        V_q = np.array([V_func(zi) for zi in z_q])

        V_mat = np.zeros((self.N, self.N))
        for m in range(self.N):
            Pm = legendre_value(m, zeta_q)
            for n in range(m, self.N):
                Pn = legendre_value(n, zeta_q)
                integrand = V_q * Pm * Pn
                val = np.sum(w_q * integrand) * self.L / 2.0
                V_mat[m, n] = val
                V_mat[n, m] = val

        return V_mat

    def expand_function(self, f_func, n_quad=None):
        """
        将函数 f(z) 展开为 Legendre 级数:
            f(z) ≈ sum_n a_n P_n(ζ(z))
            a_n = (2n+1)/L * integral f(z) P_n(ζ) dz
        """
        if n_quad is None:
            n_quad = max(2 * self.N + 2, 30)

        zeta_q, w_q = np.polynomial.legendre.leggauss(n_quad)
        z_q = self.map_to_physical(zeta_q)
        f_q = np.array([f_func(zi) for zi in z_q])

        coeffs = np.zeros(self.N)
        for n in range(self.N):
            Pn = legendre_value(n, zeta_q)
            coeffs[n] = np.sum(w_q * f_q * Pn) * (2 * n + 1) / 2.0

        return coeffs
