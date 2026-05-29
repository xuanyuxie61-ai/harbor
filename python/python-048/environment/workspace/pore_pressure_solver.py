"""
pore_pressure_solver.py
孔隙压力扩散与裂缝腔体流动求解模块

原项目映射:
    283_diffusion_pde -> 一维扩散方程离散
    343_euler         -> 常微分方程显式 Euler 时间推进
    142_cavity_flow_movie -> 裂缝腔体内部流速场计算

水力压裂中，注入流体在储层孔隙介质内遵循压力扩散方程，
同时在裂缝腔体内部存在低速黏性流动（Stokes/Hele-Shaw 近似）。
本模块将两者耦合，计算孔隙压力场演化及其对裂缝滑移的触发效应。

核心公式:
1. 孔隙压力扩散方程（线弹性多孔介质）:
   ∂p/∂t = D ∇²p + Q(x,t)
   其中 D = k / (μ φ c_t) 为水力扩散系数，
   k 为渗透率 (m²), μ 为流体黏度 (Pa·s),
   φ 为孔隙度, c_t 为综合压缩系数 (Pa^{-1})。

   一维形式（垂直于裂缝面方向）:
   ∂p/∂t = D * ∂²p/∂x² + (Q_0/φ) δ(x - x_s) H(t)

2. 显式 Euler 时间离散:
   p^{n+1} = p^n + Δt * [D * L(p^n) + S^n]
   其中 L(p) 为空间 Laplacian 的有限差分近似。

3. 裂缝腔体流速（Hele-Shaw 近似）:
   q = - (w² / 12μ) ∇_|| p
   w 为裂缝开度，∇_|| 为裂缝面内梯度。
   平均流速大小:
   |v| = (w² / 12μ) |∇_|| p|

4. 扩散方程空间离散（Neumann 零通量边界）:
   采用二阶中心差分:
   ∂²p/∂x² ≈ (p_{i-1} - 2p_i + p_{i+1}) / Δx²
   边界处设 ghost cell 满足 ∂p/∂x = 0，即 p_{-1} = p_0, p_{N} = p_{N-1}。

5. 压力触发的库仑破裂判据:
   τ_c = μ_f (σ_n - p) + c
   当剪应力 τ >= τ_c 时发生微地震事件。
"""

import numpy as np
from typing import Tuple, Callable


def laplacian_1d_neumann(u: np.ndarray, dx: float) -> np.ndarray:
    """
    在零 Neumann 边界条件下计算一维 Laplacian 的有限差分近似。

    公式:
        uxx[0] = (u[0] - 2u[0] + u[1]) / dx^2 = (u[1] - u[0]) / dx^2
                 （通过 ghost cell u_{-1}=u_0 得到）
        uxx[i] = (u[i-1] - 2u[i] + u[i+1]) / dx^2,  1 <= i <= N-2
        uxx[N-1] = (u[N-2] - u[N-1]) / dx^2
    """
    N = u.size
    if N < 2:
        return np.zeros_like(u)
    uxx = np.empty_like(u)
    uxx[0] = (u[1] - u[0]) / (dx * dx)
    uxx[1:N - 1] = (u[0:N - 2] - 2.0 * u[1:N - 1] + u[2:N]) / (dx * dx)
    uxx[N - 1] = (u[N - 2] - u[N - 1]) / (dx * dx)
    return uxx


def euler_integrate(dydt: Callable, tspan: Tuple[float, float],
                    y0: np.ndarray, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    显式 Euler 方法积分常微分方程组 dy/dt = f(t, y)。

    公式:
        y_{k+1} = y_k + Δt * f(t_k, y_k)
        t_{k+1} = t_k + Δt

    参数:
        dydt: 右端函数，接受 (t, y) 返回导数数组。
        tspan: (t0, tstop)。
        y0: 初始条件向量。
        n_steps: 时间步数。

    返回:
        t: (n_steps+1,) 时间序列。
        y: (n_steps+1, m) 状态序列。
    """
    y0 = np.asarray(y0, dtype=float)
    m = y0.size
    t0, tstop = tspan
    dt = (tstop - t0) / n_steps

    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1, m))
    t[0] = t0
    y[0, :] = y0

    for k in range(n_steps):
        t[k + 1] = t[k] + dt
        deriv = np.asarray(dydt(t[k], y[k, :]), dtype=float)
        if deriv.size != m:
            raise ValueError(f"dydt 返回维度 {deriv.size} 与状态维度 {m} 不匹配")
        y[k + 1, :] = y[k, :] + dt * deriv

    return t, y


def pore_pressure_diffusion_rhs(t: float, p: np.ndarray,
                                 D: float, dx: float,
                                 source_idx: int,
                                 source_rate: float) -> np.ndarray:
    """
    一维孔隙压力扩散方程的右端项。

    方程:
        dp/dt = D * d²p/dx² + Q_source
    """
    p = np.asarray(p, dtype=float)
    dpdt = D * laplacian_1d_neumann(p, dx)
    # 点源注入
    if 0 <= source_idx < p.size:
        dpdt[source_idx] += source_rate
    return dpdt


class PorePressureSolver:
    """
    孔隙压力扩散求解器，包含裂缝腔体流动耦合。
    """

    def __init__(self, x_min: float, x_max: float, nx: int,
                 D: float, source_rate: float, source_x: float):
        self.x = np.linspace(x_min, x_max, nx)
        self.dx = (x_max - x_min) / (nx - 1) if nx > 1 else 1.0
        self.D = D
        self.source_rate = source_rate
        self.source_idx = int(np.argmin(np.abs(self.x - source_x)))
        self.p0 = np.zeros(nx)

    def rhs(self, t: float, p: np.ndarray) -> np.ndarray:
        return pore_pressure_diffusion_rhs(t, p, self.D, self.dx,
                                            self.source_idx, self.source_rate)

    def solve(self, tspan: Tuple[float, float], n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用显式 Euler 求解压力演化。

        稳定性条件（CFL）:
            Δt <= dx² / (2D)
        """
        dt = (tspan[1] - tspan[0]) / n_steps
        cfl_limit = self.dx ** 2 / (2.0 * self.D) if self.D > 0 else np.inf
        if dt > cfl_limit:
            # 自动调整步数以满足稳定性
            n_steps = max(n_steps, int(np.ceil((tspan[1] - tspan[0]) / (0.9 * cfl_limit))))
        return euler_integrate(self.rhs, tspan, self.p0, n_steps)

    def cavity_flow_velocity(self, p: np.ndarray, w_aperture: float,
                              mu_fluid: float = 1.0e-3) -> np.ndarray:
        """
        计算裂缝腔体内的压力梯度驱动流速（Hele-Shaw 近似）。

        公式:
            q = - (w² / 12μ) ∇p
            |v| = |q| / w = (w / 12μ) |∇p|

        参数:
            p: 当前压力场 (Pa)。
            w_aperture: 裂缝平均开度 (m)。
            mu_fluid: 流体动力黏度 (Pa·s)，默认为水 1e-3。

        返回:
            v: 各网格点的流速大小 (m/s)。
        """
        if p.size != self.x.size:
            raise ValueError("压力场维度与空间网格不匹配")
        grad_p = np.zeros_like(p)
        nx = p.size
        if nx > 1:
            # 中心差分计算梯度
            grad_p[1:nx - 1] = (p[2:nx] - p[0:nx - 2]) / (2.0 * self.dx)
            grad_p[0] = (p[1] - p[0]) / self.dx
            grad_p[nx - 1] = (p[nx - 1] - p[nx - 2]) / self.dx
        # Hele-Shaw 平均流速: v = w²/(12μ) |∇p|
        coeff = (w_aperture ** 2) / (12.0 * mu_fluid)
        v = coeff * np.abs(grad_p)
        # 物理上限: 裂缝内流速不应超过局部声速的 1%（水约为 15 m/s）
        v_max_phys = 15.0
        return np.clip(v, 0.0, v_max_phys)

    def coulomb_failure_stress(self, p: np.ndarray,
                                sigma_n: float,
                                mu_fric: float = 0.6,
                                cohesion: float = 2.0e6) -> np.ndarray:
        """
        计算各网格点的库仑破裂应力（CFS）。

        公式:
            CFS = τ - τ_c = τ - [μ_f (σ_n - p) + c]
        当 CFS >= 0 时，介质达到破裂条件。

        参数:
            p: 孔隙压力 (Pa)。
            sigma_n: 有效正应力 (Pa)。
            mu_fric: 摩擦系数。
            cohesion: 黏聚力 (Pa)。

        返回:
            CFS 数组 (Pa)。
        """
        tau_c = mu_fric * (sigma_n - p) + cohesion
        # 假设远场剪应力恒定
        tau = mu_fric * sigma_n + cohesion  # 简化假设
        return tau - tau_c
