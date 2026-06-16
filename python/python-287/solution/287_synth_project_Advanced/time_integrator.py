# -*- coding: utf-8 -*-
"""
time_integrator.py
==================

时间积分器模块: ADI, Runge-Kutta, 以及 velocity-Verlet.

来源种子项目:
  - 1121_katyushapolye_PyADI-Fluid-Sim  -> ADI (Alternating Direction Implicit)
  - 861_pendulum_nonlinear_ode          -> RK4 时间步进
  - 744_md                              -> velocity-Verlet (分子动力学)

物理背景:
  Resistive MHD 方程是刚性 PDE 系统, 其中:
    - 扩散项 (eta * ∇²B) 要求隐式处理以避免 CFL 限制
    - 对流项 (v·∇)v 可以用显式方法
    - Alfven 波时间尺度 tau_A 远小于扩散时间尺度 tau_R

  ADI 方法 (Douglas-Gunn, 1964):
    将二维扩散方程 split 为两个一维问题:
      (1 - dt/2 * eta * d²/dx²) u* = (1 + dt/2 * eta * d²/dy²) u^n
      (1 - dt/2 * eta * d²/dy²) u^{n+1} = u*
    每一维只需解三对角 (或带状) 线性系统, 计算量 O(N).

  Runge-Kutta 方法 (经典 RK4):
    用于常微分方程约化系统 (如稳定性分析的 ODE 形式).

  Velocity-Verlet:
    用于追踪测试粒子在 MHD 场中的运动 (粒子-网格方法).
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Tuple


# -------------------------------------------------------------------
#  通用 ODE 接口 (与 861_pendulum_nonlinear_ode 对齐)
# -------------------------------------------------------------------
class ODERHS:
    """
    常微分方程右端项的抽象基类.
    类似 pendulum_nonlinear_deriv: dydt = f(t, y).
    """
    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# -------------------------------------------------------------------
#  RK4 积分器 (经典四阶 Runge-Kutta)
# -------------------------------------------------------------------
def rk4_step(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    经典 RK4 单步.

    算法:
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt/2 * k1)
        k3 = f(t + dt/2, y + dt/2 * k2)
        k4 = f(t + dt, y + dt * k3)
        y^{n+1} = y^n + (dt/6)(k1 + 2 k2 + 2 k3 + k4)

    参数:
        rhs: 右端项函数 f(t, y)
        t:   当前时间
        y:   当前状态
        dt:  时间步长

    返回:
        下一步状态 y^{n+1}
    """
    k1 = rhs(t, y)
    k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = rhs(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_integrate(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    t_span: Tuple[float, float],
    y0: np.ndarray,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RK4 积分全程.

    返回:
        (t_array, y_array): 时间数组 (n_steps+1,), 状态数组 (n_steps+1, len(y0))
    """
    t0, tf = t_span
    dt = (tf - t0) / n_steps
    t_arr = np.linspace(t0, tf, n_steps + 1)
    y_arr = np.zeros((n_steps + 1, y0.size))
    y_arr[0] = y0.copy()
    for n in range(n_steps):
        y_arr[n + 1] = rk4_step(rhs, t_arr[n], y_arr[n], dt)
    return t_arr, y_arr


# -------------------------------------------------------------------
#  三对角线性系统求解器 (Thomas 算法)
# -------------------------------------------------------------------
def solve_tridiagonal(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """
    Thomas 算法求解三对角线性系统 A x = rhs.

    矩阵 A 的对角线为 diag, 下次对角线为 lower, 上次对角线为 upper.

    算法:
        1. 向前消元 (forward sweep)
        2. 回代 (backward substitution)

    复杂度: O(N)
    """
    n = diag.size
    if lower.size != n - 1 or upper.size != n - 1 or rhs.size != n:
        raise ValueError("三对角系统维度不匹配")

    c_prime = np.zeros(n)
    d_prime = np.zeros(n)

    # 向前消元
    if abs(diag[0]) < 1.0e-30:
        raise ValueError("主元为零, 矩阵奇异")
    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]

    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c_prime[i - 1]
        if abs(denom) < 1.0e-30:
            denom = 1.0e-30  # 数值安全
        if i < n - 1:
            c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom

    # 回代
    x = np.zeros(n)
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


# -------------------------------------------------------------------
#  带状矩阵求解器 (五对角, 用于高阶差分)
# -------------------------------------------------------------------
def solve_pentadiagonal(
    lower2: np.ndarray,
    lower1: np.ndarray,
    diag: np.ndarray,
    upper1: np.ndarray,
    upper2: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """
    五对角线性系统求解 (用于 4 阶紧致差分格式).

    使用 LU 分解的带状形式.
    """
    n = diag.size
    # 转换为通用带状求解 (简单实现: 转为密集矩阵)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = diag[i]
        if i > 0:
            A[i, i - 1] = lower1[i - 1]
        if i > 1:
            A[i, i - 2] = lower2[i - 2]
        if i < n - 1:
            A[i, i + 1] = upper1[i]
        if i < n - 2:
            A[i, i + 2] = upper2[i]
    return np.linalg.solve(A, rhs)


# -------------------------------------------------------------------
#  ADI (Alternating Direction Implicit) 方法
# -------------------------------------------------------------------
class ADISolver:
    """
    Douglas-Gunn ADI 求解二维扩散方程:
        ∂u/∂t = eta * (∂²u/∂x² + ∂²u/∂y²)

    ADI 分裂:
        (1 - dt/2 * eta * D2x) u* = (1 + dt/2 * eta * D2y) u^n
        (1 - dt/2 * eta * D2y) u^{n+1} = u*

    每一维解一个三对角系统.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: np.ndarray,
        dy: np.ndarray,
        eta: float,
        bc_x: str = "periodic",
        bc_y: str = "dirichlet",
    ):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.eta = eta
        self.bc_x = bc_x
        self.bc_y = bc_y

        # 预计算 x 方向的三对角系数 (均匀网格)
        self._build_x_operator()
        self._build_y_operator()

    def _build_x_operator(self):
        """构造 x 方向的二阶差分算子 (均匀, 周期)."""
        dx = self.dx[0] if self.dx.size > 0 else 1.0
        self.sigma_x = self.eta / (dx * dx)

    def _build_y_operator(self):
        """构造 y 方向的二阶差分算子 (非均匀)."""
        # 对于非均匀网格, 二阶差分:
        # f''(y_i) ≈ 2/(dy_{i-1} + dy_i) * [
        #     (f_{i+1} - f_i)/dy_i - (f_i - f_{i-1})/dy_{i-1}
        # ]
        self.sigma_y = np.zeros(self.ny)
        for j in range(1, self.ny - 1):
            dy_m = self.dy[j - 1]
            dy_p = self.dy[j]
            self.sigma_y[j] = 2.0 / (dy_m + dy_p)

    def step(
        self,
        u: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        ADI 单步.

        参数:
            u: 当前场 (ny, nx)
            dt: 时间步长

        返回:
            更新后的场 u^{n+1} (ny, nx)
        """
        u_new = np.zeros_like(u)

        # 第一步: x 方向隐式, y 方向显式
        # 对于每个 y_j, 解一个 x 方向的三对角系统
        u_star = np.zeros_like(u)
        sigma_x_dt = self.eta * dt / 2.0
        for j in range(self.ny):
            # 显式 y 方向贡献: u^n + dt/2 * eta * d²u/dy²
            rhs_x = u[j].copy()
            if 1 <= j < self.ny - 1:
                dy_m = self.dy[j - 1]
                dy_p = self.dy[j]
                d2u_dy2 = self.sigma_y[j] * (
                    (u[j + 1] - u[j]) / dy_p - (u[j] - u[j - 1]) / dy_m
                )
                rhs_x += dt * self.eta * d2u_dy2

            # 隐式 x 方向: (I - dt/2 * eta * D2x) u* = rhs_x
            if self.bc_x == "periodic":
                u_star[j] = self._solve_periodic_1d(rhs_x, sigma_x_dt)
            else:
                u_star[j] = self._solve_dirichlet_1d(rhs_x, sigma_x_dt)

        # 第二步: y 方向隐式
        sigma_y_dt = self.eta * dt / 2.0
        for i in range(self.nx):
            rhs_y = u_star[:, i].copy()
            u_new[:, i] = self._solve_y_implicit(rhs_y, sigma_y_dt)

        return u_new

    def _solve_periodic_1d(self, rhs: np.ndarray, sigma_dt: float) -> np.ndarray:
        """周期边界一维隐式求解 (Sherman-Morrison)."""
        n = rhs.size
        # 主对角线: 1 + 2 sigma_dt, 次对角线: -sigma_dt
        diag = np.full(n, 1.0 + 2.0 * sigma_dt)
        off = np.full(n - 1, -sigma_dt)
        # 使用 Sherman-Morrison 处理周期性
        # (A + u v^T) x = b
        # 取 A 为三对角 (非周期), u = [1, 0, ..., 0, 1], v = [-1, 0, ..., 0, -1]
        # 简化: 直接用 scipy 或密集求解
        A = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
        A[0, -1] = -sigma_dt
        A[-1, 0] = -sigma_dt
        return np.linalg.solve(A, rhs)

    def _solve_dirichlet_1d(self, rhs: np.ndarray, sigma_dt: float) -> np.ndarray:
        """Dirichlet 边界一维隐式求解."""
        n = rhs.size
        diag = np.full(n, 1.0 + 2.0 * sigma_dt)
        lower = np.full(n - 1, -sigma_dt)
        upper = np.full(n - 1, -sigma_dt)
        return solve_tridiagonal(lower, diag, upper, rhs)

    def _solve_y_implicit(self, rhs: np.ndarray, sigma_dt: float) -> np.ndarray:
        """y 方向隐式求解 (非均匀网格)."""
        n = rhs.size
        # 构造三对角系统
        diag = np.ones(n)
        lower = np.zeros(n - 1)
        upper = np.zeros(n - 1)
        for j in range(1, n - 1):
            dy_m = self.dy[j - 1]
            dy_p = self.dy[j]
            coeff_m = 2.0 * sigma_dt / (dy_m * (dy_m + dy_p))
            coeff_p = 2.0 * sigma_dt / (dy_p * (dy_m + dy_p))
            diag[j] = 1.0 + coeff_m + coeff_p
            lower[j - 1] = -coeff_m
            upper[j] = -coeff_p
        # 边界: Dirichlet
        return solve_tridiagonal(lower, diag, upper, rhs)


# -------------------------------------------------------------------
#  Velocity-Verlet 积分器 (测试粒子追踪)
# -------------------------------------------------------------------
def velocity_verlet_step(
    pos: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    force_func: Callable[[np.ndarray], np.ndarray],
    mass: float,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Velocity-Verlet 单步 (来自 744_md).

    算法:
        x(t+dt) = x(t) + v(t) dt + 0.5 a(t) dt²
        a(t+dt) = F(x(t+dt)) / m
        v(t+dt) = v(t) + 0.5 (a(t) + a(t+dt)) dt

    参数:
        pos:  位置 (nd, np)
        vel:  速度 (nd, np)
        acc:  加速度 (nd, np)
        force_func: 力函数 F(pos) -> (nd, np)
        mass: 粒子质量
        dt:   时间步长

    返回:
        (pos_new, vel_new, acc_new)
    """
    pos_new = pos + vel * dt + 0.5 * acc * dt * dt
    force_new = force_func(pos_new)
    acc_new = force_new / mass
    vel_new = vel + 0.5 * dt * (acc + acc_new)
    return pos_new, vel_new, acc_new


# -------------------------------------------------------------------
#  CFL 条件
# -------------------------------------------------------------------
def cfl_time_step(
    dx_min: float,
    v_max: float,
    cfl_number: float = 0.5,
) -> float:
    """
    CFL 时间步长: dt = CFL * dx / v_max.

    对于 MHD, 最快波速为快磁声波:
        v_fast = sqrt(v_A^2 + c_s^2)
    """
    if v_max <= 0.0:
        return 1.0e10
    return cfl_number * dx_min / v_max


def diffusion_time_step(
    dx_min: float,
    eta: float,
    safety: float = 0.4,
) -> float:
    """
    扩散稳定性时间步长: dt = safety * dx² / eta.
    """
    if eta <= 0.0:
        return 1.0e10
    return safety * dx_min * dx_min / eta
