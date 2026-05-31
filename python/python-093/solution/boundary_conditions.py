#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boundary_conditions.py
水声传播抛物方程模型 — 边界条件处理

本模块处理 PE 求解中的三类边界：
- 海面（z=0）：压力释放边界（Dirichlet）
- 海底（z=h_b(r))：阻抗边界（Robin）或完全反射/吸收
- 深度截断（z=z_max）：吸收垫层（absorbing layer）

来源于：
- 905_pram（边界词 → 海底地形多边形边界编码）
- 807_nonlin_fixed_point（非线性迭代 → 海底阻抗自洽迭代）
- 1265_toms112（点是否在多边形内 → 域掩码裁剪）

核心物理与数学公式：
1. 海面压力释放边界：
   u(r, z=0) = 0   （Dirichlet）

2. 海底阻抗边界（Robin）：
   ∂u/∂z + γ_b · u = 0   在 z = h_b(r)
   其中 γ_b = i·k₀ · √(n_b² − cos²θ₀) 为海底导纳参数；
   n_b = c₀/c_b 为海底折射率；
   θ₀ 为掠射角。
   该边界条件通过非线性迭代（固定点/Newton）自洽求解 γ_b。

3. 吸收垫层（PML / sponge layer）：
   在 z ∈ [z_pml, z_max] 引入复坐标伸展：
   z̃ = z + i·σ(z)·(z − z_pml)
   其中 σ(z) = σ_max · [(z − z_pml)/L_pml]^p，p=2 或 3。
   波数修正：k̃ = k₀ · n(z) / (1 + i·σ'(z))。

4. 海底地形多边形编码（来自 pram 的边界词思想）：
   将 bathymetry 轮廓 h_b(r) 离散为有序多边形顶点序列：
   P = {(r_m, h_b(r_m))}_{m=0}^{M} ∪ {(r_M, 0), (r_0, 0)}
   利用 point_in_polygon 判断网格点是否在有效水域内。

5. 非线性固定点迭代（来自 807_nonlin_fixed_point）：
   对于非线性海底阻抗条件，设 γ = f(γ)，通过迭代：
   γ_{k+1} = f(γ_k)
   若 |γ_{k+1} − γ_k| < tol 则收敛；
   否则采用 Newton 修正加速：
   γ_{k+1} = γ_k − (γ_k − f(γ_k)) / (1 − f'(γ_k))。
"""

import numpy as np
from mesh_builder import point_in_polygon


class BoundaryConditionHandler:
    """
    管理抛物方程的三类边界条件。
    """

    def __init__(self, env, mesh):
        self.env = env
        self.mesh = mesh
        # PML 参数
        self.pml_fraction = 0.15
        self.pml_power = 2.0
        self.pml_sigma_max = 0.5  # 无量纲

    def apply_surface_bc(self, u):
        """
        海面压力释放边界：u(z=0) = 0。
        直接设置第一个深度节点为 0。
        """
        u[0] = 0.0
        return u

    def compute_seabed_admittance(self, theta_grazing, max_iter=20, tol=1e-10):
        """
        计算海底导纳参数 γ_b，使用固定点迭代 + Newton  safeguard。
        物理模型：
          γ_b = i·k₀ · √(n_b² − cos²θ)
        其中 n_b = c_w / c_b 为折射率比。
        迭代函数：g(γ) = i·k₀ · √(n_b² − 1 + (γ/k₀)²)
        （来自 Snell 定律自洽关系）。
        """
        k0 = self.env.k0
        n_b = self.env.c0 / self.env.seabed_cp
        cos_theta = np.cos(theta_grazing)

        def g(gamma):
            val = n_b ** 2 - cos_theta ** 2 + (gamma / k0) ** 2
            val = np.maximum(val, 0.0)
            return 1j * k0 * np.sqrt(val)

        gamma = 1j * k0 * np.sqrt(max(n_b ** 2 - cos_theta ** 2, 0.0))
        for it in range(max_iter):
            gamma_new = g(gamma)
            err = abs(gamma_new - gamma)
            if err < tol:
                return gamma_new
            # Newton 修正步
            dg = (g(gamma + 1e-8) - g(gamma - 1e-8)) / (2e-8)
            if abs(1 - dg) > 1e-12:
                gamma = gamma - (gamma - gamma_new) / (1.0 - dg)
            else:
                gamma = 0.5 * (gamma + gamma_new)
        return gamma

    def apply_seabed_bc_tridiagonal(self, a, b, c, u, m):
        """
        修改三对角矩阵系数以施加海底 Robin 边界。
        对于深度方向有限差分，海底处采用一阶近似：
        (u_N − u_{N−1})/Δz + γ_b·u_N = 0
        ⇒ u_N = u_{N−1} / (1 + γ_b·Δz)
        该关系通过修改三对角末行实现。

        参数 a, b, c 分别为下对角、主对角、上对角数组（长度 nz）。
        返回修改后的 (a, b, c)。
        """
        z_grid = self.mesh.z_grid
        h_b = self.mesh.seafloor_depth[m]
        # 找到最靠近海底的索引
        idx = np.searchsorted(z_grid, h_b, side='right') - 1
        if idx < 1:
            idx = len(z_grid) - 1
        dz = z_grid[idx] - z_grid[idx - 1]
        if dz < 1e-9:
            dz = 1.0
        # 假设小掠射角
        theta = 0.1  # rad
        gamma_b = self.compute_seabed_admittance(theta)
        # Robin: (u_idx - u_{idx-1})/dz + gamma_b * u_idx = 0
        # => u_idx = u_{idx-1} / (1 + gamma_b * dz)
        # 修改三对角第 idx 行：
        # 原方程包含 a[idx]*u_{idx-1} + b[idx]*u_idx + c[idx]*u_{idx+1}
        # 替换 u_idx 关系后，消去 u_idx，将方程降阶。
        # 简化处理：直接令 c[idx-1] = 0，b[idx] = 1 + gamma_b*dz，a[idx] = -1
        if idx < len(b):
            a[idx] = -1.0 / dz
            b[idx] = 1.0 / dz + gamma_b
            if idx + 1 < len(c):
                c[idx] = 0.0
        return a, b, c

    def pml_profile(self, z):
        """
        PML 吸收轮廓 σ(z)。
        在 z ≥ z_pml 时激活：
        σ(z) = σ_max · [(z − z_pml) / L_pml]^p
        """
        z = np.asarray(z, dtype=np.float64)
        z_pml = self.mesh.z_grid[-1] * (1.0 - self.pml_fraction)
        L_pml = self.mesh.z_grid[-1] - z_pml
        sigma = np.zeros_like(z, dtype=np.float64)
        mask = z > z_pml
        if np.any(mask):
            ratio = (z[mask] - z_pml) / max(L_pml, 1e-9)
            sigma[mask] = self.pml_sigma_max * (ratio ** self.pml_power)
        return sigma

    def pml_modified_wavenumber(self, z):
        """
        PML 区域内的复伸展波数修正。
        k_pml(z) = k₀ · n(z) / (1 + i·σ(z))
        注意：更精确的 PML 应对坐标微分算子进行修正，
        此处采用简化复波数吸收模型。
        """
        z = np.asarray(z, dtype=np.float64)
        sigma = self.pml_profile(z)
        k = self.env.wavenumber(z)
        return k / (1.0 + 1j * sigma)

    def bathymetry_polygon(self, r_extra=0.0):
        """
        将海底地形编码为闭合多边形（边界词思想）。
        顶点序列：沿海底轮廓从左到右，再沿海面返回。
        返回 (x_poly, y_poly)。
        """
        r = self.mesh.r_grid
        h = self.mesh.seafloor_depth
        # 海底轮廓
        x_poly = list(r) + [r[-1] + r_extra, r[0] - r_extra]
        y_poly = list(h) + [0.0, 0.0]
        return np.asarray(x_poly, dtype=np.float64), np.asarray(y_poly, dtype=np.float64)

    def mask_field_by_bathymetry(self, u, m):
        """
        将当前 range 步的场中超出海底的节点置零。
        """
        u = np.asarray(u, dtype=np.complex128)
        mask = self.mesh.node_mask[m, :]
        u_out = u.copy()
        u_out[~mask] = 0.0
        return u_out


def solve_tridiag(a, b, c, d):
    """
    求解三对角线性系统 T·x = d。
    a: 下对角 (长度 n，a[0] 未使用)
    b: 主对角 (长度 n)
    c: 上对角 (长度 n，c[n-1] 未使用)
    d: 右端项 (长度 n)
    使用 Thomas 算法（前向消去 + 回代），O(n) 复杂度。
    """
    n = len(b)
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    c = np.asarray(c, dtype=np.complex128)
    d = np.asarray(d, dtype=np.complex128)

    cp = np.zeros(n - 1, dtype=np.complex128)
    dp = np.zeros(n, dtype=np.complex128)
    x = np.zeros(n, dtype=np.complex128)

    # 前向消去
    dp[0] = d[0] / b[0]
    if n > 1:
        cp[0] = c[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if abs(denom) < 1e-20:
            denom = 1e-20
        if i < n - 1:
            cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    # 回代
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x
