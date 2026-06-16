#!/usr/bin/env python3
"""
time_integrator.py
==================
时间积分器模块，融合种子项目:
  - [405] fem2d_heat_sparse: 后向 Euler 时间推进
  - [619] kepler_perturbed_ode: 哈密顿守恒检验, 辛积分

实现多种时间积分方法:
  - 显式: Forward Euler, SSP-RK2, SSP-RK3, RK4
  - 隐式: Backward Euler (后向 Euler), Crank-Nicolson
  - 辛方法: Störmer-Verlet (用于哈密顿系统)

稳定性与守恒性分析:
  - 显式方法: CFL 约束
  - 隐式方法: 无条件稳定 (A-稳定)
  - 辛方法: 保哈密顿量, 保辛结构
"""

import numpy as np
from scipy.sparse import eye as speye
from scipy.sparse.linalg import spsolve


class TimeIntegrator:
    """时间积分器基类"""

    def __init__(self, rhs_func, dt, t_start=0.0, t_end=1.0):
        """
        参数:
            rhs_func: 右端函数 f(t, u) → du/dt
            dt: 时间步长
            t_start, t_end: 时间区间
        """
        self.rhs_func = rhs_func
        self.dt = dt
        self.t_start = t_start
        self.t_end = t_end
        self.n_steps = int(np.ceil((t_end - t_start) / dt))
        self.t = t_start
        self.step_count = 0


class ForwardEuler(TimeIntegrator):
    """
    前向 Euler (1阶显式):
      u^{n+1} = u^n + Δt f(t^n, u^n)
    稳定性: |1 + Δt λ| ≤ 1 (对线性问题)
    """

    def step(self, t, u):
        """单步推进"""
        k1 = self.rhs_func(t, u)
        u_new = u + self.dt * k1
        return u_new, t + self.dt


class SSPRK2(TimeIntegrator):
    """
    SSP-RK2 (2阶强稳定保持 Runge-Kutta):
      u⁽¹⁾ = u^n + Δt f(t^n, u^n)
      u^{n+1} = (1/2) u^n + (1/2) [u⁽¹⁾ + Δt f(t^{n+1}, u⁽¹⁾)]
    SSP 系数: c = 1 (TVD 保持)
    """

    def step(self, t, u):
        k1 = self.rhs_func(t, u)
        u1 = u + self.dt * k1
        k2 = self.rhs_func(t + self.dt, u1)
        u_new = 0.5 * u + 0.5 * (u1 + self.dt * k2)
        return u_new, t + self.dt


class SSPRK3(TimeIntegrator):
    """
    SSP-RK3 (3阶强稳定保持 Runge-Kutta, Shu-Osher):
      u⁽¹⁾ = u^n + Δt f(t^n, u^n)
      u⁽²⁾ = (3/4) u^n + (1/4) [u⁽¹⁾ + Δt f(t^{n+1}, u⁽¹⁾)]
      u^{n+1} = (1/3) u^n + (2/3) [u⁽²⁾ + Δt f(t^{n+1/2}, u⁽²⁾)]
    SSP 系数: c = 1
    """

    def step(self, t, u):
        k1 = self.rhs_func(t, u)
        u1 = u + self.dt * k1
        k2 = self.rhs_func(t + self.dt, u1)
        u2 = 0.75 * u + 0.25 * (u1 + self.dt * k2)
        k3 = self.rhs_func(t + 0.5 * self.dt, u2)
        u_new = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + self.dt * k3)
        return u_new, t + self.dt


class ClassicRK4(TimeIntegrator):
    """
    经典 4 阶 Runge-Kutta:
      k₁ = f(t^n, u^n)
      k₂ = f(t^n + Δt/2, u^n + Δt k₁/2)
      k₃ = f(t^n + Δt/2, u^n + Δt k₂/2)
      k₄ = f(t^n + Δt, u^n + Δt k₃)
      u^{n+1} = u^n + (Δt/6)(k₁ + 2k₂ + 2k₃ + k₄)
    """

    def step(self, t, u):
        k1 = self.rhs_func(t, u)
        k2 = self.rhs_func(t + 0.5 * self.dt, u + 0.5 * self.dt * k1)
        k3 = self.rhs_func(t + 0.5 * self.dt, u + 0.5 * self.dt * k2)
        k4 = self.rhs_func(t + self.dt, u + self.dt * k3)
        u_new = u + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return u_new, t + self.dt


class BackwardEuler(TimeIntegrator):
    """
    后向 Euler (1阶隐式, 融合 [405] fem2d_heat_sparse):
      u^{n+1} = u^n + Δt f(t^{n+1}, u^{n+1})

    对线性系统 f(u) = A u:
      (I - Δt A) u^{n+1} = u^n
    需要求解线性系统。

    对非线性系统: 使用 Newton 迭代或 Picard 迭代。
    A-稳定: 无条件稳定 (对线性问题)。
    """

    def __init__(self, rhs_func, dt, t_start=0.0, t_end=1.0,
                 A_sparse=None, newton_tol=1e-10, newton_max_iter=20):
        """
        参数:
            rhs_func: 右端函数
            A_sparse: 若为线性系统, 传入稀疏矩阵 A
            newton_tol: Newton 迭代容差
            newton_max_iter: 最大迭代次数
        """
        super().__init__(rhs_func, dt, t_start, t_end)
        self.A_sparse = A_sparse
        self.newton_tol = newton_tol
        self.newton_max_iter = newton_max_iter
        self._factorized = None

    def step(self, t, u):
        if self.A_sparse is not None:
            # 线性隐式: (I - Δt A) u^{n+1} = u^n
            n = len(u)
            M = speye(n, format='csr') - self.dt * self.A_sparse
            u_new = spsolve(M, u)
        else:
            # 非线性: Picard 迭代 (简化)
            u_new = u.copy()
            t_new = t + self.dt
            for _ in range(self.newton_max_iter):
                f_val = self.rhs_func(t_new, u_new)
                u_next = u + self.dt * f_val
                if np.max(np.abs(u_next - u_new)) < self.newton_tol:
                    u_new = u_next
                    break
                u_new = u_next

        return u_new, t + self.dt


class CrankNicolson(TimeIntegrator):
    """
    Crank-Nicolson (2阶隐式):
      u^{n+1} = u^n + (Δt/2) [f(t^n, u^n) + f(t^{n+1}, u^{n+1})]
    对线性系统: (I - Δt/2 A) u^{n+1} = (I + Δt/2 A) u^n
    A-稳定, 但非 L-稳定 (可能产生振荡)。
    """

    def __init__(self, rhs_func, dt, t_start=0.0, t_end=1.0, A_sparse=None):
        super().__init__(rhs_func, dt, t_start, t_end)
        self.A_sparse = A_sparse

    def step(self, t, u):
        n = len(u)
        if self.A_sparse is not None:
            M_lhs = speye(n, format='csr') - 0.5 * self.dt * self.A_sparse
            M_rhs = speye(n, format='csr') + 0.5 * self.dt * self.A_sparse
            rhs = M_rhs @ u
            u_new = spsolve(M_lhs, rhs)
        else:
            f_n = self.rhs_func(t, u)
            u_pred = u + self.dt * f_n  # 初始猜测
            t_new = t + self.dt
            for _ in range(20):
                f_new = self.rhs_func(t_new, u_pred)
                u_next = u + 0.5 * self.dt * (f_n + f_new)
                if np.max(np.abs(u_next - u_pred)) < 1e-10:
                    break
                u_pred = u_next
            u_new = u_pred
        return u_new, t + self.dt


class StormerVerlet(TimeIntegrator):
    """
    Störmer-Verlet 辛积分器 (融合 [619] kepler_perturbed_ode)。
    用于哈密顿系统:
      H(q, p) = T(p) + V(q)
    方程:
      dq/dt = ∂H/∂p = p/m
      dp/dt = -∂H/∂q = -∇V(q)

    辛格式 (Kick-Drift-Kick):
      p^{n+1/2} = p^n - (Δt/2) ∇V(q^n)
      q^{n+1} = q^n + Δt p^{n+1/2}/m
      p^{n+1} = p^{n+1/2} - (Δt/2) ∇V(q^{n+1})

    保性质: 保辛结构, 近保能量 (长期积分误差有界)。
    """

    def __init__(self, grad_V, mass, dt, t_start=0.0, t_end=1.0):
        """
        参数:
            grad_V: 势能梯度 ∇V(q) → force
            mass: 质量
            dt: 时间步长
        """
        super().__init__(None, dt, t_start, t_end)
        self.grad_V = grad_V
        self.mass = mass

    def step(self, t, state):
        """
        参数:
            state: (q, p) 位置与动量
        """
        q, p = state[0].copy(), state[1].copy()
        # Kick
        force = -self.grad_V(q)
        p_half = p + 0.5 * self.dt * force
        # Drift
        q_new = q + self.dt * p_half / self.mass
        # Kick
        force_new = -self.grad_V(q_new)
        p_new = p_half + 0.5 * self.dt * force_new
        return (q_new, p_new), t + self.dt


class AdaptiveTimeController:
    """
    自适应时间步长控制器。
    基于 PID 控制策略调整 Δt。

    PID 控制器:
      Δt_{n+1} = Δt_n × (tol / err_n)^{k_I} × (err_{n-1} / err_n)^{k_P}
    通常 k_I = 0.7/order, k_P = 0.4/order ( Gustafsson )
    """

    def __init__(self, initial_dt, order=3, tol=1e-4, dt_min=1e-14, dt_max=1.0,
                 safety=0.9):
        self.dt = initial_dt
        self.order = order
        self.tol = tol
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.safety = safety
        self.prev_error = None

    def update(self, error):
        """
        根据误差更新 Δt。
        返回: (new_dt, accepted)
        """
        if error < 1e-30:
            return min(self.dt * 2.0, self.dt_max), True

        err_ratio = self.tol / error

        if self.prev_error is not None and self.prev_error > 1e-30:
            # PID 控制
            k_I = 0.7 / self.order
            k_P = 0.4 / self.order
            factor = self.safety * (err_ratio ** k_I) * \
                     ((self.prev_error / error) ** k_P)
        else:
            # I 控制
            k_I = 1.0 / (self.order + 1)
            factor = self.safety * (err_ratio ** k_I)

        factor = np.clip(factor, 0.2, 5.0)
        new_dt = self.dt * factor
        new_dt = np.clip(new_dt, self.dt_min, self.dt_max)

        accepted = error <= self.tol
        self.prev_error = error if accepted else self.prev_error

        return new_dt, accepted


def runge_kutta_fehlberg_45(rhs_func, t, u, dt, rtol=1e-6, atol=1e-8):
    """
    RKF45 (Runge-Kutta-Fehlberg 4(5)) 自适应步长方法。
    使用 4 阶和 5 阶两个估计的差来估计误差。

    Butcher 表:
      0    |
      1/4  | 1/4
      3/8  | 3/32       9/32
      12/13| 1932/2197  -7200/2197  7296/2197
      1    | 439/216    -8          3680/513    -845/4104
      1/2  | -8/27      2           -3544/2565  1859/4104  -11/40
      ─────┼──────────────────────────────────────────────────
      5阶  | 16/135     0           6656/12825  28561/56430 -9/50   2/55
      4阶  | 25/216     0           1408/2565   2197/4104   -1/5    0
    """
    k1 = dt * rhs_func(t, u)
    k2 = dt * rhs_func(t + dt / 4, u + k1 / 4)
    k3 = dt * rhs_func(t + 3 * dt / 8, u + 3 * k1 / 32 + 9 * k2 / 32)
    k4 = dt * rhs_func(t + 12 * dt / 13,
                        u + 1932 * k1 / 2197 - 7200 * k2 / 2197 + 7296 * k3 / 2197)
    k5 = dt * rhs_func(t + dt,
                        u + 439 * k1 / 216 - 8 * k2 + 3680 * k3 / 513 - 845 * k4 / 4104)
    k6 = dt * rhs_func(t + dt / 2,
                        u - 8 * k1 / 27 + 2 * k2 - 3544 * k3 / 2565 + 1859 * k4 / 4104 - 11 * k5 / 40)

    # 5阶解
    u5 = u + 16 * k1 / 135 + 6656 * k3 / 12825 + 28561 * k4 / 56430 - 9 * k5 / 50 + 2 * k6 / 55
    # 4阶解
    u4 = u + 25 * k1 / 216 + 1408 * k3 / 2565 + 2197 * k4 / 4104 - k5 / 5

    # 误差估计
    error = np.max(np.abs(u5 - u4))
    scale = atol + rtol * np.max(np.abs(u))
    error_norm = error / max(scale, 1e-30)

    return u5, error_norm


def integrate_ode(rhs_func, u0, t_span, dt, method='ssprk3', **kwargs):
    """
    统一 ODE 积分接口。

    参数:
        rhs_func: du/dt = f(t, u)
        u0: 初始条件
        t_span: (t_start, t_end)
        dt: 初始时间步长
        method: 积分方法名
    返回:
        dict: {t: [...], u: [...], n_steps: int}
    """
    t_start, t_end = t_span

    methods = {
        'euler': ForwardEuler,
        'ssprk2': SSPRK2,
        'ssprk3': SSPRK3,
        'rk4': ClassicRK4,
    }

    if method not in methods:
        raise ValueError(f"未知方法: {method}, 可选: {list(methods.keys())}")

    integrator = methods[method](rhs_func, dt, t_start, t_end)
    t_arr = [t_start]
    u_arr = [u0.copy()]
    t = t_start
    u = u0.copy()

    max_steps = int(1e7)
    for step in range(max_steps):
        if t >= t_end - 1e-14 * abs(t_end):
            break
        dt_actual = min(integrator.dt, t_end - t)
        integrator.dt = dt_actual
        u_new, t_new = integrator.step(t, u)
        u = u_new
        t = t_new
        t_arr.append(t)
        u_arr.append(u.copy())

    return {
        't': np.array(t_arr),
        'u': u_arr,
        'n_steps': len(t_arr) - 1
    }
