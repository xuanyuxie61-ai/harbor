"""
wave_pde.py
复合材料层合板中的应力波传播PDE求解。
原项目映射：
  - 020_artery_pde 的弹性波PDE结构（血管动脉壁振动方程）
    dw/dt = [u; v] 的耦合ODE系统思想
科学背景：
  层合板中的应力波传播可用修正的弹性波方程描述：
    ρ h ∂^2 u/∂t^2 = ∂N_x/∂x + ∂N_xy/∂y
    ρ h ∂^2 v/∂t^2 = ∂N_xy/∂x + ∂N_y/∂y
    ρ h ∂^2 w/∂t^2 = ∂^2M_x/∂x^2 + 2∂^2M_xy/∂x∂y + ∂^2M_y/∂y^2 + q(x,y,t)
  其中 N 为膜力，M 为弯矩，u,v,w 为面内与横向位移。
  简化为1D模型（沿纤维方向）：
    ρ A ∂^2u/∂t^2 = EA ∂^2u/∂x^2 - β ∂u/∂t + γ f(x,t)
  其中 β 为结构阻尼，γ 为外载系数。
  与 artery_pde 的方程形式类比：
    artery: dudt = v, dvdt = -α u - β v + γ x dp cos(ω t)
    wave:   dudt = v, dvdt = c^2 ∂^2u/∂x^2 - 2ζ ω_n v + f_ext/m
"""

import numpy as np
from scipy.integrate import odeint


class WavePropagation1D:
    """一维应力波在受损复合材料中的传播。"""

    def __init__(self, L, nx, E_func, rho, damping_ratio, forcing_params):
        """
        L: 杆长度
        nx: 空间离散点数
        E_func: 等效弹性模量函数 E(x)（可含损伤退化）
        rho: 密度
        damping_ratio: 阻尼比 ζ
        forcing_params: (amplitude, frequency, position)
        """
        self.L = float(L)
        self.nx = nx
        self.x = np.linspace(0.0, self.L, nx)
        self.dx = self.x[1] - self.x[0]
        self.E = np.array([E_func(xi) for xi in self.x])
        self.rho = float(rho)
        self.damping_ratio = float(damping_ratio)
        self.f_amp, self.f_omega, self.f_pos = forcing_params

        # 自然频率近似
        self.omega_n = np.pi / self.L * np.sqrt(np.mean(self.E) / self.rho)
        self.beta = 2.0 * self.damping_ratio * self.omega_n

    def deriv(self, w, t):
        """
        状态向量 w = [u_0, ..., u_{nx-1}, v_0, ..., v_{nx-1}]
        du/dt = v
        dv/dt = (E/ρ) d^2u/dx^2 - β v + f_ext/ρ
        使用中心差分：d^2u/dx^2 ≈ (u_{i-1} - 2u_i + u_{i+1}) / dx^2
        """
        nx = self.nx
        u = w[:nx]
        v = w[nx:]

        dudt = v.copy()
        dvdt = np.zeros(nx)

        # 空间二阶导数
        for i in range(1, nx - 1):
            d2u = (u[i - 1] - 2.0 * u[i] + u[i + 1]) / self.dx ** 2
            c2 = self.E[i] / self.rho
            dvdt[i] = c2 * d2u - self.beta * v[i]

        # 边界条件（固定端）
        dvdt[0] = 0.0
        dvdt[-1] = 0.0
        dudt[0] = 0.0
        dudt[-1] = 0.0

        # 外载（简谐激励）
        force = self.f_amp * np.cos(self.f_omega * t)
        # 在激励位置附近施加分布力
        for i in range(nx):
            dist = abs(self.x[i] - self.f_pos)
            if dist < 2.0 * self.dx:
                dvdt[i] += force / (self.rho * self.dx)

        return np.concatenate([dudt, dvdt])

    def solve(self, u0, v0, t_span):
        """求解时间历程。"""
        w0 = np.concatenate([u0, v0])
        sol = odeint(self.deriv, w0, t_span, rtol=1e-6, atol=1e-9)
        u_history = sol[:, :self.nx]
        v_history = sol[:, self.nx:]
        return u_history, v_history

    def compute_wave_speed(self):
        """计算波速 c = sqrt(E/ρ)。"""
        return np.sqrt(self.E / self.rho)

    def compute_attenuation_coefficient(self, frequency):
        """
        计算损伤引起的波衰减系数。
        对于粘弹性/损伤介质：α = ω^2 ζ / (2 c^3)
        """
        c = np.mean(self.compute_wave_speed())
        omega = 2.0 * np.pi * frequency
        alpha = omega ** 2 * self.damping_ratio / (2.0 * c ** 3)
        return alpha


def compute_stress_wave_reflection_coefficient(E1, E2, rho1, rho2):
    """
    计算应力波在两种材料界面的反射系数。
    R = (Z2 - Z1) / (Z2 + Z1)，其中 Z = ρ c 为声阻抗。
    """
    c1 = np.sqrt(E1 / rho1)
    c2 = np.sqrt(E2 / rho2)
    Z1 = rho1 * c1
    Z2 = rho2 * c2
    R = (Z2 - Z1) / (Z2 + Z1 + 1e-12)
    T = 2.0 * Z2 / (Z1 + Z2 + 1e-12)
    return R, T
