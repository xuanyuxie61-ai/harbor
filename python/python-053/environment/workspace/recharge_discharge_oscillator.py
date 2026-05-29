"""
recharge_discharge_oscillator.py
=================================
基于 two_body_ode (1369_two_body_ode) 的常微分方程数值积分框架，
实现 Jin (1997) 的 recharge-discharge oscillator 模型——ENSO 动力学的
核心理论框架之一。

科学背景
--------
Jin (1997) 将 ENSO 视为一个 recharge-discharge 振荡器：
- 暖水体积 (WWV) 在热带太平洋西部积累（充电阶段）；
- 当温跃层深度梯度达到临界值时，Bjerknes 正反馈触发 El Niño（放电阶段）；
- 放电后，Rossby 波在西边界反射并携带冷信号返回，使系统恢复。

该模型将复杂的海洋-大气相互作用简化为两个耦合的 ODE，
类似于二体问题中将引力作用简化为中心力场。

核心公式
--------
1. Recharge-Discharge Oscillator 方程组：

   d h_W / dt = -r * h_W - α * T_E
   d T_E / dt = R * h_W - ε * (T_E)³ + γ * T_E

   其中：
   - h_W : 西太平洋温跃层深度异常（充放电变量）
   - T_E : 东太平洋 SST 异常（ENSO 指数代理）
   - r   : WWV 的阻尼率（Rossby 波耗散）
   - α   : 温跃层-风应力耦合系数
   - R   : Bjerknes 反馈强度
   - ε   : 非线性饱和系数
   - γ   : 线性增长率

2. 将方程写为向量形式 u' = f(u)，采用 RK4 积分：

   k1 = f(u_n)
   k2 = f(u_n + Δt/2 * k1)
   k3 = f(u_n + Δt/2 * k2)
   k4 = f(u_n + Δt * k3)

   u_{n+1} = u_n + Δt/6 * (k1 + 2k2 + 2k3 + k4)

3. 系统的平衡点与稳定性：
   令 dh_W/dt = 0, dT_E/dt = 0，得：
   
   h_W* = -α * T_E* / r
   R * (-α * T_E* / r) - ε * (T_E*)³ + γ * T_E* = 0

   即：T_E* * [ -Rα/r - ε*(T_E*)² + γ ] = 0

   非平凡平衡点存在条件：γ > Rα/r
   此时系统存在极限环振荡（ENSO 周期）。

4. 振荡周期近似：
   T_osc ≈ 2π / √(Rα/r - γ²/4)   (当 γ²/4 < Rα/r 时)

5. 类比二体问题：
   h_W ↔ 角动量（充放电的"惯性"）
   T_E ↔ 径向距离（ENSO 强度）
   R, α ↔ 引力耦合常数
"""

import numpy as np
from typing import Tuple, Callable, Optional


def rdo_derivatives(t: float, u: np.ndarray,
                    r: float = 0.25,
                    alpha: float = 0.5,
                    R: float = 1.0,
                    epsilon: float = 0.3,
                    gamma: float = 0.4,
                    seasonal_forcing: Optional[Callable] = None) -> np.ndarray:
    """
    Recharge-Discharge Oscillator 的右端项。

    参数
    ----
    t : float
        时间（年）。
    u : np.ndarray, shape (2,)
        状态向量 [h_W, T_E]。
    r, alpha, R, epsilon, gamma : float
        模型参数。
    seasonal_forcing : callable, optional
        季节循环强迫 F(t)。

    返回
    ----
    up : np.ndarray, shape (2,)
        导数 [dh_W/dt, dT_E/dt]。
    """
    h_w, t_e = u[0], u[1]

    # 主方程
    dh_dt = -r * h_w - alpha * t_e
    dT_dt = R * h_w - epsilon * (t_e ** 3) + gamma * t_e

    if seasonal_forcing is not None:
        forcing = seasonal_forcing(t)
        dT_dt += forcing

    return np.array([dh_dt, dT_dt])


def seasonal_cycle(t: float, amplitude: float = 0.1, phase: float = 0.0) -> float:
    """
    季节循环强迫项。

    公式：F(t) = A * cos(2π * t - φ)

    物理意义：北半球冬季（t ≈ 0.75 年）风应力最强，
    春季（t ≈ 0.25 年）最弱，形成 ENSO 季节锁相。
    """
    return amplitude * np.cos(2.0 * np.pi * t - phase)


def rk4_integrate(f: Callable, u0: np.ndarray, t0: float, tf: float,
                  n_steps: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    四阶 Runge-Kutta 数值积分。

    参数
    ----
    f : callable
        右端项函数 f(t, u, **kwargs)。
    u0 : np.ndarray
        初始状态。
    t0, tf : float
        起始和终止时间。
    n_steps : int
        时间步数。
    **kwargs : dict
        传递给 f 的额外参数。

    返回
    ----
    t : np.ndarray, shape (n_steps+1,)
        时间序列。
    u : np.ndarray, shape (n_steps+1, len(u0))
        状态轨迹。
    """
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    dt = (tf - t0) / n_steps
    dim = u0.shape[0]
    t = np.linspace(t0, tf, n_steps + 1)
    u = np.zeros((n_steps + 1, dim), dtype=float)
    u[0] = u0

    for i in range(n_steps):
        k1 = f(t[i], u[i], **kwargs)
        k2 = f(t[i] + dt / 2.0, u[i] + dt / 2.0 * k1, **kwargs)
        k3 = f(t[i] + dt / 2.0, u[i] + dt / 2.0 * k2, **kwargs)
        k4 = f(t[i] + dt, u[i] + dt * k3, **kwargs)

        u[i + 1] = u[i] + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # 数值鲁棒性：截断异常值
        if np.any(np.isnan(u[i + 1])) or np.any(np.isinf(u[i + 1])):
            u[i + 1] = u[i]

    return t, u


def solve_rdo(years: float = 20.0,
              n_steps: int = 20000,
              h_w0: float = 0.5,
              t_e0: float = 0.3,
              r: float = 0.25,
              alpha: float = 0.5,
              R: float = 1.0,
              epsilon: float = 0.3,
              gamma: float = 0.4,
              seasonal_amp: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解 recharge-discharge oscillator 模型。

    参数
    ----
    years : float
        模拟年数。
    n_steps : int
        时间步数。
    h_w0, t_e0 : float
        初始条件。
    r, alpha, R, epsilon, gamma : float
        模型参数。
    seasonal_amp : float
        季节强迫振幅。

    返回
    ----
    t : np.ndarray
        时间序列（年）。
    u : np.ndarray, shape (n_steps+1, 2)
        状态轨迹 [h_W, T_E]。
    """
    u0 = np.array([h_w0, t_e0])

    def forcing(t):
        return seasonal_cycle(t, amplitude=seasonal_amp)

    def f(t, u):
        return rdo_derivatives(t, u, r=r, alpha=alpha, R=R,
                               epsilon=epsilon, gamma=gamma,
                               seasonal_forcing=forcing)

    t, u = rk4_integrate(f, u0, 0.0, years, n_steps)
    return t, u[:, 1]


def find_equilibrium(r: float, alpha: float, R: float,
                     epsilon: float, gamma: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    寻找系统的平衡点。

    公式：
    T_E* [ -Rα/r - ε*(T_E*)² + γ ] = 0

    返回
    ----
    trivial_eq : np.ndarray
        平凡平衡点 (0, 0)。
    nontrivial_eqs : np.ndarray or None
        非平凡平衡点（若存在）。
    """
    trivial = np.array([0.0, 0.0])

    # 非平凡解：-Rα/r - ε*T² + γ = 0
    # T² = (γ - Rα/r) / ε
    discriminant = gamma - R * alpha / r

    if discriminant <= 0:
        return trivial, None

    t_star = np.sqrt(discriminant / epsilon)
    h_star = -alpha * t_star / r

    nontrivial = np.array([
        [h_star, t_star],
        [h_star, -t_star]
    ])

    return trivial, nontrivial


def oscillation_period_approx(r: float, alpha: float, R: float,
                              epsilon: float, gamma: float) -> float:
    """
    计算振荡周期的解析近似。

    公式：
    T_osc ≈ 2π / √(Rα/r - γ²/4)
    """
    omega_sq = R * alpha / r - gamma ** 2 / 4.0
    if omega_sq <= 0:
        return float('inf')
    return 2.0 * np.pi / np.sqrt(omega_sq)


def classify_dynamics(r: float, alpha: float, R: float,
                      epsilon: float, gamma: float) -> str:
    """
    根据参数对系统动力学进行分类。

    分类标准：
    - 稳定结点：γ < Rα/r 且 γ²/4 > Rα/r
    - 阻尼振荡：γ < Rα/r 且 γ²/4 < Rα/r
    - 极限环：γ > Rα/r
    - 混沌（强非线性）：ε 很大时可能出现
    """
    ratio = R * alpha / r
    if gamma < ratio:
        if gamma ** 2 / 4.0 > ratio:
            return "stable_node"
        else:
            return "damped_oscillation"
    else:
        if epsilon > 0.5:
            return "strongly_nonlinear_limit_cycle"
        return "limit_cycle"
