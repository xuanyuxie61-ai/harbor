#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
basis_approximation.py
Bernstein 多项式基函数逼近地下电性结构

融合种子项目 013_approx_bernstein 的核心算法，
将 Bernstein 多项式应用于电阻率剖面的参数化表示。

Bernstein 多项式在 [0, 1] 上的定义：
    B_{n,k}(x) = C(n,k) * x^k * (1-x)^{n-k},  k = 0,...,n

电阻率剖面的 Bernstein 参数化：
    ρ(z) = Σ_{k=0}^{n} c_k * B_{n,k}(z / z_max)

Bernstein 基函数的优点：
  1. 保正性：若 c_k ≥ 0，则 ρ(z) ≥ 0
  2. 端点插值：ρ(0) = c_0, ρ(z_max) = c_n
  3. 变差缩减性：剖面平滑，适合地球物理反演
"""

import numpy as np


def binomial_coefficient(n, k):
    """计算二项式系数 C(n, k)"""
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    k = min(k, n - k)
    result = 1.0
    for i in range(1, k + 1):
        result = result * (n - k + i) / i
    return result


def bernstein_basis(n, k, t):
    """
    计算 Bernstein 基函数 B_{n,k}(t)

    B_{n,k}(t) = C(n,k) * t^k * (1-t)^{n-k}

    Parameters
    ----------
    n : int
        多项式次数
    k : int
        基函数索引, 0 <= k <= n
    t : float or ndarray
        自变量, 应在 [0, 1] 内

    Returns
    -------
    float or ndarray
        基函数值
    """
    t = np.asarray(t, dtype=np.float64)
    if np.any(t < 0.0) or np.any(t > 1.0):
        # 软截断到边界，保证数值稳定性
        t = np.clip(t, 0.0, 1.0)
    if k < 0 or k > n:
        return np.zeros_like(t)
    # 使用对数空间计算避免下溢
    log_c = np.log(binomial_coefficient(n, k))
    # 处理 t=0 或 t=1 的边界
    result = np.zeros_like(t)
    mask = (t > 0.0) & (t < 1.0)
    result[~mask & (t == 0.0) & (k == 0)] = 1.0
    result[~mask & (t == 1.0) & (k == n)] = 1.0
    if np.any(mask):
        t_m = t[mask]
        log_val = log_c + k * np.log(t_m) + (n - k) * np.log(1.0 - t_m)
        result[mask] = np.exp(log_val)
    return result if result.shape != () else float(result)


def bernstein_basis_recursive(n, t):
    """
    使用递推关系计算所有 Bernstein 基函数 B_{n,0}(t), ..., B_{n,n}(t)

    递推公式：
        B_{0,0}(t) = 1
        B_{j,k}(t) = (1-t) * B_{j-1,k}(t) + t * B_{j-1,k-1}(t)

    这是种子项目 013_approx_bernstein 的核心递推算法的直接移植。
    """
    t = np.asarray(t, dtype=np.float64)
    t = np.clip(t, 0.0, 1.0)

    if n == 0:
        return np.ones((1,) + t.shape)

    # be[k] 对应 B_{current_degree, k}
    be = np.zeros((n + 1,) + t.shape, dtype=np.float64)
    be[0] = 1.0 - t
    be[1] = t

    for j in range(2, n + 1):
        be[j] = t * be[j - 1]
        for k in range(j - 1, 0, -1):
            be[k] = t * be[k - 1] + (1.0 - t) * be[k]
        be[0] = (1.0 - t) * be[0]

    return be


class BernsteinResistivityProfile:
    """
    使用 Bernstein 多项式参数化的电阻率随深度变化剖面

    ρ(z) = Σ_{k=0}^{n} c_k * B_{n,k}(z / z_max),  z ∈ [0, z_max]
    """

    def __init__(self, coefficients, z_max=10000.0):
        """
        Parameters
        ----------
        coefficients : array_like
            Bernstein 系数 c_0, ..., c_n
        z_max : float
            最大深度 [m]
        """
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.n = len(self.coefficients) - 1
        self.z_max = float(z_max)
        if self.z_max <= 0.0:
            raise ValueError("z_max 必须为正")
        if np.any(self.coefficients <= 0.0):
            raise ValueError("Bernstein 系数必须为正以保证电阻率为正")

    def evaluate(self, z):
        """
        计算深度 z 处的电阻率

        Parameters
        ----------
        z : float or ndarray
            深度 [m], 0 <= z <= z_max

        Returns
        -------
        float or ndarray
            电阻率 [Ω·m]
        """
        z = np.asarray(z, dtype=np.float64)
        if np.any(z < 0.0) or np.any(z > self.z_max):
            z = np.clip(z, 0.0, self.z_max)
        t = z / self.z_max
        basis = bernstein_basis_recursive(self.n, t)
        rho = np.dot(self.coefficients, basis)
        return rho

    def derivative(self, z):
        """
        计算电阻率对深度的导数 dρ/dz

        Bernstein 多项式的导数公式：
            d/dt B_{n,k}(t) = n * [B_{n-1,k-1}(t) - B_{n-1,k}(t)]

        因此：
            dρ/dz = (1/z_max) * Σ_{k=0}^{n} c_k * n * [B_{n-1,k-1}(t) - B_{n-1,k}(t)]
                  = (n/z_max) * Σ_{k=0}^{n-1} (c_{k+1} - c_k) * B_{n-1,k}(t)
        """
        z = np.asarray(z, dtype=np.float64)
        z = np.clip(z, 0.0, self.z_max)
        t = z / self.z_max
        if self.n == 0:
            return np.zeros_like(z)
        dc = np.diff(self.coefficients)
        basis = bernstein_basis_recursive(self.n - 1, t)
        drhodz = (self.n / self.z_max) * np.dot(dc, basis)
        return drhodz

    def roughness(self):
        """
        计算模型粗糙度（系数的二阶差分范数）

        用于正则化反演中的模型平滑约束。
        """
        if self.n < 2:
            return 0.0
        d2c = np.diff(self.coefficients, 2)
        return np.sum(d2c ** 2)

    def to_layer_model(self, n_layers):
        """
        将连续 Bernstein 剖面离散化为层状模型

        Parameters
        ----------
        n_layers : int
            离散层数

        Returns
        -------
        resistivities : ndarray
            各层电阻率 [Ω·m]
        thicknesses : ndarray
            各层厚度 [m]
        """
        if n_layers < 2:
            raise ValueError("层数至少为 2")
        z_interfaces = np.linspace(0.0, self.z_max, n_layers + 1)
        thicknesses = np.diff(z_interfaces)
        z_centers = (z_interfaces[:-1] + z_interfaces[1:]) / 2.0
        resistivities = self.evaluate(z_centers)
        # 保证电阻率为正
        resistivities = np.maximum(resistivities, 1e-6)
        return resistivities, thicknesses


class Bernstein2DResistivity:
    """
    二维 Bernstein 张量积基函数参数化电阻率分布

    ρ(y, z) = Σ_{i=0}^{n_y} Σ_{j=0}^{n_z} c_{ij} * B_{n_y,i}(y/y_max) * B_{n_z,j}(z/z_max)

    用于二维大地电磁反演中的电性结构参数化。
    """

    def __init__(self, coefficients, y_max=50000.0, z_max=10000.0):
        """
        Parameters
        ----------
        coefficients : ndarray, shape (n_y + 1, n_z + 1)
            二维 Bernstein 系数矩阵
        y_max, z_max : float
            横向和纵向最大范围 [m]
        """
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        if self.coefficients.ndim != 2:
            raise ValueError("系数必须是二维矩阵")
        self.ny, self.nz = self.coefficients.shape
        self.ny -= 1
        self.nz -= 1
        self.y_max = float(y_max)
        self.z_max = float(z_max)
        if np.any(self.coefficients <= 0.0):
            raise ValueError("系数必须为正")

    def evaluate(self, y, z):
        """
        计算 (y, z) 处的电阻率
        """
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        y = np.clip(y, 0.0, self.y_max)
        z = np.clip(z, 0.0, self.z_max)
        ty = y / self.y_max
        tz = z / self.z_max
        By = bernstein_basis_recursive(self.ny, ty)
        Bz = bernstein_basis_recursive(self.nz, tz)
        # ρ = Σ_{i,j} c_{ij} * By_i * Bz_j
        if y.ndim == 0 and z.ndim == 0:
            rho = np.dot(By.T, np.dot(self.coefficients, Bz))
        else:
            # 广播处理
            rho = np.einsum('i...,ij,j...->...', By, self.coefficients, Bz)
        return rho

    def roughness_yz(self):
        """
        计算二维粗糙度（y和z方向的二阶差分之和）
        """
        ry = 0.0
        rz = 0.0
        if self.ny >= 2:
            d2y = np.diff(self.coefficients, 2, axis=0)
            ry = np.sum(d2y ** 2)
        if self.nz >= 2:
            d2z = np.diff(self.coefficients, 2, axis=1)
            rz = np.sum(d2z ** 2)
        return ry + rz


if __name__ == "__main__":
    # 自检
    t = np.linspace(0, 1, 101)
    basis = bernstein_basis_recursive(3, t)
    print("Bernstein 基函数和:", np.sum(basis, axis=0)[:5])

    coeffs = np.array([100.0, 80.0, 50.0, 30.0, 20.0])
    profile = BernsteinResistivityProfile(coeffs, z_max=5000.0)
    z_test = np.array([0.0, 1250.0, 2500.0, 3750.0, 5000.0])
    rho_test = profile.evaluate(z_test)
    drho_test = profile.derivative(z_test)
    print(f"ρ(z) = {rho_test}")
    print(f"dρ/dz = {drho_test}")

    rho_layers, thick = profile.to_layer_model(4)
    print(f"离散化电阻率: {rho_layers}")
    print(f"厚度: {thick}")
