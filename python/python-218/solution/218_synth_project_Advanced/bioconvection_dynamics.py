"""
bioconvection_dynamics.py
=========================
生物对流 Lorenz 系统与时间依赖变分不等式的耦合动力学.

数学背景
--------
Avramenko et al. (2023) 将生物对流不稳定性约化为 Lorenz 型 ODE:
    dx/dt = Sc (y - x)
    dy/dt = R_a x + x z - y
    dz/dt = -x y - b z

其中:
    Sc = Schmidt 数 (动量扩散/质量扩散)
    R_a = 生物对流 Rayleigh 数
    b = 几何参数 (与波数相关)

时间依赖变分不等式 (TDVI):
    求 u(t) ∈ K 使得
        ⟨u'(t) + A(u(t)) - f(t), v - u(t)⟩ ≥ 0,  ∀ v ∈ K

    等价于演化互补问题:
        u(t) ≥ 0,  F(u(t), t) ≥ 0,  u(t)^T F(u(t), t) = 0

本模块的创新:
    将 Lorenz 系统的混沌动力学作为 TDVI 的参数驱动,
    研究 VI 解在混沌参数下的响应:
        1. 解路径的分形结构
        2. Lyapunov 指数对互补条件的影响
        3. 混沌同步与 VI 稳定性的关系

关键公式
--------
Lyapunov 指数 (最大):
    λ_1 = lim_{t→∞} (1/t) log ||δx(t)|| / ||δx(0)||

当 λ_1 > 0 时, 系统对初始条件敏感 (混沌).

作者: DA synthesis project
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Callable


class BioconvectionParameters:
    """
    生物对流系统的物理参数.

    Lorenz 约化:
        dx/dt = Sc (y - x)
        dy/dt = R_a x + x z - y
        dz/dt = -x y - b z

    典型参数值:
        Sc = 1000 (微生物悬浮液)
        R_a ∈ [20, 100] (临界 Rayleigh 数附近)
        b = 8/3 (标准 Lorenz 值)

    动力学行为:
        R_a < R_c: 稳定不动点 (无对流)
        R_c < R_a < R_H: 稳定极限环 (周期对流)
        R_a > R_H: 混沌 (湍流对流)
    """

    def __init__(
        self,
        Sc: float = 100.0,
        Ra: float = 50.0,
        b: float = 8.0 / 3.0,
    ):
        self.Sc = Sc
        self.Ra = Ra
        self.b = b

    def critical_rayleigh(self) -> float:
        """
        临界 Rayleigh 数 (Hopf 分岔点).

        R_c = Sc (Sc + b + 3) / (Sc - b - 1)
        (当 Sc > b + 1 时)
        """
        if self.Sc <= self.b + 1:
            return np.inf
        return self.Sc * (self.Sc + self.b + 3.0) / (self.Sc - self.b - 1.0)

    def is_chaotic(self) -> bool:
        """判断参数是否处于混沌区域."""
        R_c = self.critical_rayleigh()
        return self.Ra > 1.5 * R_c if np.isfinite(R_c) else False

    def fixed_points(self) -> list:
        """
        计算不动点.

        平凡不动点: (0, 0, 0)
        非平凡不动点 (当 Ra > R_c):
            C± = (±√(b(Ra/Sc - 1)), ±√(b(Ra/Sc - 1)), Ra/Sc - 1)
        """
        fps = [(0.0, 0.0, 0.0)]
        if self.Ra > self.Sc:
            z_star = self.Ra / self.Sc - 1.0
            xy_star = np.sqrt(self.b * z_star) if z_star > 0 else 0.0
            fps.append((xy_star, xy_star, z_star))
            fps.append((-xy_star, -xy_star, z_star))
        return fps


class BioconvectionODE:
    """
    生物对流 Lorenz 系统的 ODE 右端函数.

    状态向量: (x, y, z) ∈ R³
    参数: (Sc, R_a, b)
    """

    def __init__(self, params: BioconvectionParameters):
        self.params = params

    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        右端函数 f(t, [x, y, z]).

        dx/dt = Sc (y - x)
        dy/dt = R_a x + x z - y
        dz/dt = -x y - b z
        """
        x, y, z = state[0], state[1], state[2]
        Sc, Ra, b = self.params.Sc, self.params.Ra, self.params.b

        dxdt = Sc * (y - x)
        dydt = Ra * x + x * z - y
        dzdt = -x * y - b * z

        return np.array([dxdt, dydt, dzdt])

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """
        Jacobian 矩阵.

        J = [-Sc   Sc   0  ]
            [Ra+z  -1   x  ]
            [-y    -x   -b ]
        """
        x, y, z = state[0], state[1], state[2]
        Sc, Ra, b = self.params.Sc, self.params.Ra, self.params.b

        J = np.array([
            [-Sc,  Sc,  0.0],
            [Ra + z, -1.0, x],
            [-y, -x, -b],
        ])
        return J


class RK4Integrator:
    """
    经典四阶 Runge-Kutta 积分器.

    k_1 = h f(t_n, y_n)
    k_2 = h f(t_n + h/2, y_n + k_1/2)
    k_3 = h f(t_n + h/2, y_n + k_2/2)
    k_4 = h f(t_n + h, y_n + k_3)
    y_{n+1} = y_n + (k_1 + 2k_2 + 2k_3 + k_4) / 6

    局部截断误差: O(h^5)
    全局误差: O(h^4)
    """

    def __init__(self, rhs: Callable, dt: float = 0.01):
        self.rhs = rhs
        self.dt = dt

    def step(self, t: float, y: np.ndarray) -> np.ndarray:
        """执行一步 RK4."""
        h = self.dt
        k1 = h * self.rhs(t, y)
        k2 = h * self.rhs(t + h/2, y + k1/2)
        k3 = h * self.rhs(t + h/2, y + k2/2)
        k4 = h * self.rhs(t + h, y + k3)
        return y + (k1 + 2*k2 + 2*k3 + k4) / 6.0

    def integrate(self, t_span: Tuple[float, float], y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        在 [t0, tf] 上积分.

        Returns
        -------
        t_vals : ndarray (N+1,)
        y_vals : ndarray (N+1, n)
        """
        t0, tf = t_span
        n_steps = int((tf - t0) / self.dt) + 1
        t_vals = np.linspace(t0, tf, n_steps)
        y_vals = np.zeros((n_steps, len(y0)))
        y_vals[0] = y0.copy()

        for i in range(1, n_steps):
            y_vals[i] = self.step(t_vals[i-1], y_vals[i-1])

        return t_vals, y_vals


class TimeDependentVI:
    """
    时间依赖变分不等式 (TDVI) 的时间离散化.

    隐式 Euler 离散:
        求 u^{n+1} ∈ K 使得
            ⟨(u^{n+1} - u^n)/Δt + A(u^{n+1}) - f^{n+1}, v - u^{n+1}⟩ ≥ 0

    等价于一步 VI:
        求 u ∈ K 使得
            ⟨u + Δt(A(u) - f^{n+1}) - u^n, v - u⟩ ≥ 0

    即 G(u) = u + Δt(A(u) - f^{n+1}) - u^n, 解 VI(K, G).
    """

    def __init__(
        self,
        A_func: Callable[[np.ndarray], np.ndarray],
        K_projection: Callable[[np.ndarray], np.ndarray],
        dt: float = 0.01,
        n: int = 10,
    ):
        self.A_func = A_func
        self.K_projection = K_projection
        self.dt = dt
        self.n = n

    def build_step_F(self, u_n: np.ndarray, f_next: np.ndarray) -> Callable:
        """构建单步 VI 的映射 G."""
        def G(u: np.ndarray) -> np.ndarray:
            return u + self.dt * (self.A_func(u) - f_next) - u_n
        return G

    def project_step(self, u_n: np.ndarray, f_next: np.ndarray) -> np.ndarray:
        """
        投影法求解单步 TDVI.

        u^{n+1} = Π_K(u^n - Δt(A(u^n) - f^{n+1}))
        (一阶显式投影)
        """
        A_u = self.A_func(u_n)
        u_pred = u_n - self.dt * (A_u - f_next)
        return self.K_projection(u_pred)


class LyapunovAnalyzer:
    """
    Lyapunov 指数的数值估计.

    最大 Lyapunov 指数:
        λ_1 ≈ (1/T) Σ_{k} log(||J_k δ_k|| / ||δ_k||)

    其中 J_k 为第 k 步的 Jacobian, δ_k 为扰动向量 (每步归一化).

    Lyapunov 谱 (三维系统):
        λ_1 + λ_2 + λ_3 = trace(J̄)  (平均散度)
    对 Lorenz 系统: λ_1 + λ_2 + λ_3 = -(Sc + 1 + b) < 0
    (相空间体积收缩)
    """

    def __init__(self, ode: BioconvectionODE, dt: float = 0.01):
        self.ode = ode
        self.dt = dt
        self.integrator = RK4Integrator(ode.rhs, dt)

    def estimate_max_lyapunov(
        self,
        y0: np.ndarray,
        t_total: float,
        renorm_interval: int = 10,
    ) -> float:
        """
        估计最大 Lyapunov 指数.

        算法 (Benettin et al. 1980):
            1. 积分主轨迹 y(t)
            2. 每 renorm_interval 步, 计算扰动增长并归一化
            3. 平均 log 增长率
        """
        n_steps = int(t_total / self.dt)
        y = y0.copy()
        delta = np.array([1e-8, 0.0, 0.0])

        lyap_sum = 0.0
        n_renorms = 0

        for step in range(n_steps):
            # 积分主轨迹和扰动轨迹
            t = step * self.dt
            y_new = self.integrator.step(t, y)
            y_delta = y + delta
            y_delta_new = self.integrator.step(t, y_delta)

            # 扰动增长
            delta_new = y_delta_new - y_new
            delta_norm = np.linalg.norm(delta_new)

            if (step + 1) % renorm_interval == 0 and delta_norm > 1e-30:
                lyap_sum += np.log(delta_norm / np.linalg.norm(delta))
                delta = delta_new * (1e-8 / delta_norm)
                n_renorms += 1
            else:
                delta = delta_new

            y = y_new

        if n_renorms == 0:
            return 0.0

        T_total = n_renorms * renorm_interval * self.dt
        return float(lyap_sum / T_total)
