# -*- coding: utf-8 -*-
"""
reinitialization.py
===================
水平集函数的重初始化（Reinitialization）模块，将水平集函数恢复为符号距离函数。

融合原始项目:
  - 603_jacobi: Jacobi 迭代法求解线性系统（用于重初始化 PDE 的隐式/半隐式迭代）

核心数学公式
------------
1. 重初始化 PDE (Sussman et al., 1994):
   ∂φ/∂τ + sign(φ_0)(|∇φ| - 1) = 0
   其中 τ 为伪时间，φ_0 为重初始化前的水平集值。

2. sign(φ_0) 的光滑化:
   S(φ_0) = φ_0 / √(φ_0² + |∇φ_0|² h²)
   其中 h = min(Δx, Δy)，避免了 sign 函数在零点的奇异性。

3. Godunov 型空间离散:
   |∇φ|² ≈ max( (a⁺)², (b⁻)² ) + max( (c⁺)², (d⁻)² )
   其中:
     a = (φ_{i,j} - φ_{i-1,j})/h ,  b = (φ_{i+1,j} - φ_{i,j})/h
     c = (φ_{i,j} - φ_{i,j-1})/h ,  d = (φ_{i,j+1} - φ_{i,j})/h
     a⁺ = max(a,0), b⁻ = min(b,0)

4. 伪时间步长约束:
   Δτ = 0.5 · h   （满足 CFL）

5. Jacobi 迭代视角:
   将重初始化方程改写为:
   φ^{k+1} = φ^k - Δτ · S(φ_0)(|∇φ^k| - 1)
   这本质上是 Jacobi 型不动点迭代，每次更新仅使用前一步值。
   对于隐式格式可视为:
   (I + Δτ · L) φ^{k+1} = φ^k + Δτ · S(φ_0)
   其中 L 为单调算子。

6. 收敛判据:
   |φ^{k+1} - φ^k|_∞ < tol
   或 |∇φ| 与 1 的偏差足够小。
"""

import numpy as np


class Reinitializer:
    """
    水平集重初始化器，采用 Godunov 迎风格式求解重初始化 PDE。
    """

    def __init__(self, levelset, max_iter=100, tol=1e-6):
        """
        参数:
            levelset : LevelSetFunction 实例
            max_iter : 最大伪时间迭代次数
            tol      : 收敛容差
        """
        self.ls = levelset
        self.max_iter = max_iter
        self.tol = tol
        self.dx = levelset.dx
        self.dy = levelset.dy

    def _smooth_sign(self, phi0):
        """
        光滑化 sign 函数。
        S(φ_0) = φ_0 / √(φ_0² + |∇φ_0|² h²)
        """
        nx, ny = self.ls.nx, self.ls.ny
        h = min(self.dx, self.dy)

        # 计算 |∇φ_0| 的二阶中心差分近似
        grad = np.zeros_like(phi0)
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                dxp = (phi0[i + 1, j] - phi0[i, j]) / self.dx
                dxm = (phi0[i, j] - phi0[i - 1, j]) / self.dx
                dyp = (phi0[i, j + 1] - phi0[i, j]) / self.dy
                dym = (phi0[i, j] - phi0[i, j - 1]) / self.dy
                grad[i, j] = np.sqrt(0.5 * (dxp ** 2 + dxm ** 2 + dyp ** 2 + dym ** 2))

        # 边界
        grad[0, :] = grad[1, :]
        grad[-1, :] = grad[-2, :]
        grad[:, 0] = grad[:, 1]
        grad[:, -1] = grad[:, -2]

        S = phi0 / np.sqrt(phi0 ** 2 + grad ** 2 * h ** 2 + 1e-12)
        return S

    def _godunov_gradient_magnitude(self, phi):
        """
        Godunov 型梯度模计算。
        |∇φ|² = max((a⁺)², (b⁻)²) + max((c⁺)², (d⁻)²)
        """
        nx, ny = phi.shape
        grad_mag = np.zeros_like(phi)

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                a = (phi[i, j] - phi[i - 1, j]) / self.dx
                b = (phi[i + 1, j] - phi[i, j]) / self.dx
                c = (phi[i, j] - phi[i, j - 1]) / self.dy
                d_val = (phi[i, j + 1] - phi[i, j]) / self.dy

                term_x = np.maximum(a, 0.0) ** 2 + np.minimum(b, 0.0) ** 2
                term_y = np.maximum(c, 0.0) ** 2 + np.minimum(d_val, 0.0) ** 2
                grad_mag[i, j] = np.sqrt(term_x + term_y)

        # 边界用内部值填充
        grad_mag[0, :] = grad_mag[1, :]
        grad_mag[-1, :] = grad_mag[-2, :]
        grad_mag[:, 0] = grad_mag[:, 1]
        grad_mag[:, -1] = grad_mag[:, -2]
        return grad_mag

    def reinitialize(self, phi0=None):
        """
        执行重初始化，将水平集恢复为符号距离函数。
        返回迭代次数和最终残差。
        """
        if phi0 is None:
            phi0 = self.ls.phi.copy()
        else:
            phi0 = np.asarray(phi0, dtype=np.float64)

        phi = phi0.copy()
        S = self._smooth_sign(phi0)
        h = min(self.dx, self.dy)
        dtau = 0.1 * h  # 更保守的伪时间步长，保证稳定性

        for it in range(self.max_iter):
            grad_mag = self._godunov_gradient_magnitude(phi)
            # 限制梯度模的更新量，防止数值爆炸
            phi_new = phi - dtau * S * np.clip(grad_mag - 1.0, -5.0, 5.0)

            # 边界零阶外推（Neumann）
            phi_new[0, :] = phi_new[1, :]
            phi_new[-1, :] = phi_new[-2, :]
            phi_new[:, 0] = phi_new[:, 1]
            phi_new[:, -1] = phi_new[:, -2]

            # 数值稳定性：限制 phi 的范围
            phi_max = np.max(np.abs(phi0)) * 2.0 + 1.0
            phi_new = np.clip(phi_new, -phi_max, phi_max)

            diff = np.max(np.abs(phi_new - phi))
            phi = phi_new.copy()

            if diff < self.tol:
                break

        self.ls.phi = phi
        return it + 1, diff

    def reinitialize_jacobi_style(self, phi0=None, omega=1.0):
        """
        Jacobi 型迭代重初始化（融入 jacobi 迭代思想）。
        每次更新所有网格点，使用前一步全局值（Jacobi 风格），
        可选超松弛因子 ω。

        迭代格式:
        φ^{k+1} = φ^k - ω · Δτ · S(φ_0)(|∇φ^k| - 1)

        这与线性系统的 Jacobi 迭代:
        x^{k+1} = D^{-1}(b - (L+U)x^k)
        在结构上等价，均为不动点迭代。
        """
        if phi0 is None:
            phi0 = self.ls.phi.copy()
        phi = phi0.copy()
        S = self._smooth_sign(phi0)
        h = min(self.dx, self.dy)
        dtau = 0.1 * h  # 保守步长

        for it in range(self.max_iter):
            grad_mag = self._godunov_gradient_magnitude(phi)
            phi_new = phi - omega * dtau * S * np.clip(grad_mag - 1.0, -5.0, 5.0)

            phi_new[0, :] = phi_new[1, :]
            phi_new[-1, :] = phi_new[-2, :]
            phi_new[:, 0] = phi_new[:, 1]
            phi_new[:, -1] = phi_new[:, -2]

            phi_max = np.max(np.abs(phi0)) * 2.0 + 1.0
            phi_new = np.clip(phi_new, -phi_max, phi_max)

            diff = np.max(np.abs(phi_new - phi))
            phi = phi_new.copy()
            if diff < self.tol:
                break

        self.ls.phi = phi
        return it + 1, diff

    def fast_marching_brute(self, phi0=None):
        """
        基于暴力搜索的快速行进法近似。
        对粗网格上的每个点，搜索所有过零点以计算精确距离。
        时间复杂度 O(N² M)，其中 N 为网格点数，M 为零点个数。
        适用于小规模验证计算。

        融入 distance_to_position_sphere 的距离变换思想:
        平面上的欧氏距离对应于球面距离在 R→∞ 时的极限行为。
        """
        if phi0 is None:
            phi0 = self.ls.phi.copy()

        nx, ny = phi0.shape
        x = self.ls.x
        y = self.ls.y

        # 提取零等值线点
        points = []
        for i in range(nx):
            for j in range(ny - 1):
                if phi0[i, j] * phi0[i, j + 1] < 0:
                    t = phi0[i, j] / (phi0[i, j] - phi0[i, j + 1])
                    points.append((x[i], y[j] + t * (y[j + 1] - y[j])))
        for i in range(nx - 1):
            for j in range(ny):
                if phi0[i, j] * phi0[i + 1, j] < 0:
                    t = phi0[i, j] / (phi0[i, j] - phi0[i + 1, j])
                    points.append((x[i] + t * (x[i + 1] - x[i]), y[j]))

        if len(points) == 0:
            return phi0.copy()

        points = np.array(points)
        phi_new = np.zeros_like(phi0)

        for i in range(nx):
            for j in range(ny):
                dists = np.sqrt((points[:, 0] - x[i]) ** 2 + (points[:, 1] - y[j]) ** 2)
                dmin = np.min(dists)
                phi_new[i, j] = dmin if phi0[i, j] >= 0 else -dmin

        self.ls.phi = phi_new
        return phi_new

    def check_sdf_property(self):
        """
        检查符号距离函数性质：|∇φ| 应接近 1。
        返回 |∇φ| - 1 的 L∞ 范数。
        """
        phi = self.ls.phi
        nx, ny = phi.shape
        grad_norm = np.zeros_like(phi)

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                dxp = (phi[i + 1, j] - phi[i, j]) / self.dx
                dxm = (phi[i, j] - phi[i - 1, j]) / self.dx
                dyp = (phi[i, j + 1] - phi[i, j]) / self.dy
                dym = (phi[i, j] - phi[i, j - 1]) / self.dy
                grad_norm[i, j] = np.sqrt(0.5 * (dxp ** 2 + dxm ** 2 + dyp ** 2 + dym ** 2))

        error = np.max(np.abs(grad_norm - 1.0))
        return error
