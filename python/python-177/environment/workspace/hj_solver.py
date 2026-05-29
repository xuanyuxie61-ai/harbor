# -*- coding: utf-8 -*-
"""
hj_solver.py
============
Hamilton-Jacobi 方程的高阶数值求解器。

融合原始项目:
  - 1289_traveling_wave_exact: 行波方程的精确解及其验证思想

核心数学公式
------------
1. Hamilton-Jacobi 方程（水平集形式）:
   ∂φ/∂t + H(∇φ) = 0
   其中 H(p,q) = V_n · √(p² + q²) 为 Hamiltonian。

2. 对于曲率驱动流 (Mean Curvature Flow):
   V_n = -ε κ
   方程变为: ∂φ/∂t = ε κ |∇φ|

3. 对于平流-扩散型界面演化:
   ∂φ/∂t + u·∇φ = γ κ |∇φ|
   其中 u 为速度场，γ 为表面张力系数。

4. 行波精确解验证（融入 traveling_wave_exact）:
   一维标量守恒律的粘性近似存在行波解:
   u(x,t) = u( (x - st)/ε )
   对于水平集方法，可构造一维自相似解:
   φ(x,t) = tanh( (x - x_0 - V t) / δ )
   该解在 δ → 0 时趋于符号距离函数。

5. WENO5 + TVD-RK3 半离散格式:
   dφ_{ij}/dt = L(φ) = -Ĥ(φ_x^+, φ_x^-, φ_y^+, φ_y^-)
   其中 Ĥ 为数值 Hamiltonian（Lax-Friedrichs 型）:
   Ĥ = H( (φ_x^+ + φ_x^-)/2, (φ_y^+ + φ_y^-)/2 )
       - α_x (φ_x^+ - φ_x^-)/2 - α_y (φ_y^+ - φ_y^-)/2
   α_x = max|∂H/∂p|, α_y = max|∂H/∂q|
"""

import numpy as np
from numerical_utils import weno5_derivative, tvd_rk3_step, central_diff_2nd


class HJSolver:
    """
    Hamilton-Jacobi 方程求解器，支持曲率驱动流、平流及外部力场。
    """

    def __init__(self, levelset, epsilon=0.01, gamma=0.0):
        """
        参数:
            levelset : LevelSetFunction 实例
            epsilon  : 曲率驱动系数
            gamma    : 表面张力系数
        """
        self.ls = levelset
        self.epsilon = epsilon
        self.gamma = gamma
        self.dx = levelset.dx
        self.dy = levelset.dy

    def _rhs_curvature_flow(self, phi):
        """
        曲率驱动流的右端项: L(φ) = ε κ |∇φ|
        使用 LevelSetFunction 的曲率计算。
        """
        ls_tmp = type(self.ls)(self.ls.nx, self.ls.ny,
                               self.ls.xlim, self.ls.ylim)
        ls_tmp.phi = phi.copy()
        kappa = ls_tmp.compute_curvature()
        _, _, grad_norm = ls_tmp.compute_gradient_norm()
        rhs = self.epsilon * kappa * grad_norm
        # 边界保持 Neumann
        rhs[0, :] = rhs[1, :]
        rhs[-1, :] = rhs[-2, :]
        rhs[:, 0] = rhs[:, 1]
        rhs[:, -1] = rhs[:, -2]
        return rhs

    def _rhs_advection(self, phi, u_field, v_field):
        """
        平流项右端项: L(φ) = -u·∇φ
        使用迎风格式或中心差分（视 CFL 条件而定）。
        """
        dx, dy = self.dx, self.dy
        phi_x = central_diff_2nd(phi, dx, axis=0)
        phi_y = central_diff_2nd(phi, dy, axis=1)
        rhs = -(u_field * phi_x + v_field * phi_y)
        return rhs

    def _rhs_combined(self, phi, u_field, v_field, forcing):
        """
        组合右端项:
        L(φ) = ε κ |∇φ| - u·∇φ + f_ext |∇φ|
        其中 forcing 为外部法向速度场 f_ext(x,y)。
        """
        # HOLE_2: 实现组合右端项的计算
        # 需要调用曲率计算与梯度模，并组合曲率驱动、平流与外部力场项
        raise NotImplementedError("HOLE_2: Combined RHS implementation missing")

    def step_rk3(self, dt, u_field=None, v_field=None, forcing=None):
        """
        执行一个 TVD-RK3 时间步。

        参数:
            dt      : 时间步长（需满足 CFL 条件）
            u_field, v_field : 速度场 (nx, ny)
            forcing : 外部法向速度 (nx, ny)
        """
        phi0 = self.ls.phi.copy()

        if u_field is None:
            u_field = np.zeros_like(phi0)
        if v_field is None:
            v_field = np.zeros_like(phi0)
        if forcing is None:
            forcing = np.zeros_like(phi0)

        def rhs_func(phi):
            return self._rhs_combined(phi, u_field, v_field, forcing)

        phi_new = tvd_rk3_step(phi0, dt, rhs_func)
        self.ls.phi = phi_new
        return self

    def compute_cfl_dt(self, u_field=None, v_field=None, forcing=None, cfl=0.5):
        """
        根据 CFL 条件计算最大允许时间步长。
        对 HJ 方程:
        dt ≤ cfl · min(dx, dy) / max(|V_n| + |u| + |v|)
        """
        phi0 = self.ls.phi.copy()
        if u_field is None:
            u_field = np.zeros_like(phi0)
        if v_field is None:
            v_field = np.zeros_like(phi0)
        if forcing is None:
            forcing = np.zeros_like(phi0)

        ls_tmp = type(self.ls)(self.ls.nx, self.ls.ny,
                               self.ls.xlim, self.ls.ylim)
        ls_tmp.phi = phi0.copy()
        kappa = ls_tmp.compute_curvature()
        Vn = np.abs(self.epsilon * kappa) + np.abs(forcing)
        vel_max = np.max(Vn + np.abs(u_field) + np.abs(v_field))
        if vel_max < 1e-14:
            vel_max = 1.0
        dt_max = cfl * min(self.dx, self.dy) / vel_max
        return dt_max

    @staticmethod
    def exact_tanh_1d(x, t, x0=0.0, V=0.5, delta=0.05):
        """
        一维粘性近似行波精确解（融入 traveling_wave_exact 思想）。
        φ(x,t) = tanh( (x - x0 - V t) / δ )
        该解满足:
        ∂φ/∂t + V ∂φ/∂x = ν ∂²φ/∂x²
        其中 ν = V δ。

        参数:
            x     : 空间坐标
            t     : 时间
            x0    : 初始位置
            V     : 波速
            delta : 界面厚度
        """
        return np.tanh((x - x0 - V * t) / delta)

    def compute_error_vs_exact(self, t, exact_func):
        """
        计算当前水平集与精确解之间的 L2 误差。
        用于数值验证。
        """
        X, Y = np.meshgrid(self.ls.x, self.ls.y, indexing='ij')
        phi_exact = exact_func(X, Y, t)
        diff = self.ls.phi - phi_exact
        error = np.sqrt(np.sum(diff ** 2) * self.dx * self.dy)
        return error


class ShearFlow:
    """
    剪切流速度场生成器，用于测试界面在复杂流场中的演化。
    """

    @staticmethod
    def simple_shear(X, Y, shear_rate=1.0):
        """
        简单剪切流:
        u = shear_rate · y
        v = 0
        """
        u = shear_rate * Y
        v = np.zeros_like(Y)
        return u, v

    @staticmethod
    def vortex_pair(X, Y, strength=1.0, center1=(-0.3, 0.0), center2=(0.3, 0.0)):
        """
        涡对速度场（点涡叠加）。
        对位于 (cx, cy) 的点涡，速度场为:
        u = -Γ (y - cy) / (2π r²)
        v =  Γ (x - cx) / (2π r²)
        """
        def point_vortex(cx, cy, Gamma):
            r2 = (X - cx) ** 2 + (Y - cy) ** 2
            r2 = np.maximum(r2, 1e-8)
            u = -Gamma * (Y - cy) / (2.0 * np.pi * r2)
            v = Gamma * (X - cx) / (2.0 * np.pi * r2)
            return u, v

        u1, v1 = point_vortex(center1[0], center1[1], strength)
        u2, v2 = point_vortex(center2[0], center2[1], -strength)
        return u1 + u2, v1 + v2

    @staticmethod
    def oscillatory_shear(X, Y, t, freq=2.0, amp=1.0):
        """
        振荡剪切流:
        u = amp · sin(2π f t) · y
        v = 0
        """
        u = amp * np.sin(2.0 * np.pi * freq * t) * Y
        v = np.zeros_like(Y)
        return u, v
