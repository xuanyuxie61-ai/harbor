"""
state_dynamics.py
=================

航天器轨迹动力学模块。

融合种子项目:
  - 1115_sandyherho_inerOsci : 惯性振荡 ODE 求解器 (隐式/显式格式)

在 Pontryagin 最优控制中, 该模块定义:
  1. 状态向量 x = (r, v, m, theta, gamma)
     - r: 地心距 [m]
     - v: 速度 [m/s]
     - m: 质量 [kg]
     - theta: 经度 [rad]
     - gamma: 飞行路径角 [rad]
  2. 控制向量 u = (T, alpha)
     - T: 推力 [N]
     - alpha: 推力方向角 [rad]
  3. 状态方程 dx/dt = f(x, u, t)
  4. 隐式/显式时间积分格式

数学公式:
---------
1. 运动方程 (2D 平面, 无J2摄动):
     dr/dt      = v sin(gamma)
     dv/dt      = (T cos(alpha) - D) / m - mu/r^2 sin(gamma)
     dm/dt      = -T / (I_sp g_0)
     dtheta/dt  = (v cos(gamma)) / r
     dgamma/dt  = (T sin(alpha) + L) / (m v)
                  + (v/r - mu/(v r^2)) cos(gamma)

2. 阻力:  D = 0.5 rho(h) v^2 C_D A
3. 升力:  L = 0 (简化为无升力体)
4. 大气密度: rho(h) = rho_0 exp(-h / H)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from scientific_constants import (
    EARTH,
    SpacecraftParameters,
    atmospheric_density,
)


# ===========================================================================
# 1. 状态与控制数据结构
# ===========================================================================
@dataclass
class State:
    """状态向量 x = (r, v, m, theta, gamma)."""
    r: float      # 地心距 [m]
    v: float      # 速度 [m/s]
    m: float      # 质量 [kg]
    theta: float  # 经度 [rad]
    gamma: float  # 飞行路径角 [rad]

    def as_list(self) -> List[float]:
        return [self.r, self.v, self.m, self.theta, self.gamma]

    @staticmethod
    def from_list(x: List[float]) -> "State":
        if len(x) != 5:
            raise ValueError(f"状态向量长度必须为 5, 实际 {len(x)}")
        return State(*x)

    def check_physical_bounds(self) -> Tuple[bool, str]:
        """检查物理合理性."""
        if self.r <= 0:
            return False, f"地心距 r={self.r} 必须为正"
        if self.v < 0:
            return False, f"速度 v={self.v} 不能为负"
        if self.m <= 0:
            return False, f"质量 m={self.m} 必须为正"
        if abs(self.gamma) > math.pi / 2:
            return False, f"飞行路径角 gamma={self.gamma} 超出 [-pi/2, pi/2]"
        return True, "OK"


@dataclass
class Control:
    """控制向量 u = (T, alpha)."""
    thrust: float   # 推力 [N]
    alpha: float    # 推力方向角 [rad]

    def as_list(self) -> List[float]:
        return [self.thrust, self.alpha]

    def check_bounds(self, sc: SpacecraftParameters) -> Tuple[bool, str]:
        """检查控制约束."""
        if not (sc.thrust_min <= self.thrust <= sc.thrust_max):
            return False, f"推力 T={self.thrust} 超出 [{sc.thrust_min}, {sc.thrust_max}]"
        if abs(self.alpha) > math.pi:
            return False, f"方向角 alpha={self.alpha} 超出 [-pi, pi]"
        return True, "OK"


# ===========================================================================
# 2. 右端函数 f(x, u, t)
# ===========================================================================
def state_derivative(
    x: State, u: Control, sc: SpacecraftParameters, t: float = 0.0
) -> List[float]:
    """计算状态导数 dx/dt = f(x, u, t).

    运动方程 (2D 平面, 无升力, 指数大气):
        dr/dt      = v sin(gamma)
        dv/dt      = (T cos(alpha) - D) / m - mu/r^2 sin(gamma)
        dm/dt      = -T / (I_sp g_0)
        dtheta/dt  = (v cos(gamma)) / r
        dgamma/dt  = (T sin(alpha)) / (m v) + (v/r - mu/(v r^2)) cos(gamma)

    Parameters
    ----------
    x : State
        当前状态.
    u : Control
        当前控制.
    sc : SpacecraftParameters
        航天器参数.
    t : float
        当前时间 (用于时变系统, 此处未使用).

    Returns
    -------
    List[float]
        状态导数 [dr/dt, dv/dt, dm/dt, dtheta/dt, dgamma/dt].
    """
    r, v, m, theta, gamma = x.r, x.v, x.m, x.theta, x.gamma
    T, alpha = u.thrust, u.alpha

    # 边界保护
    if r <= EARTH.radius_mean:
        r = EARTH.radius_mean * 1.001  # 避免地面碰撞
    if v < 1.0:
        v = 1.0  # 避免除以零
    if m < sc.dry_mass:
        m = sc.dry_mass  # 燃料耗尽

    # 大气密度与阻力
    h = r - EARTH.radius_mean
    rho = atmospheric_density(h)
    D = 0.5 * rho * v * v * sc.drag_coeff * sc.ref_area

    # 重力参数
    mu = EARTH.mu
    g0 = EARTH.g0

    # 状态导数
    dr_dt = v * math.sin(gamma)
    dv_dt = (T * math.cos(alpha) - D) / m - (mu / (r * r)) * math.sin(gamma)
    dm_dt = -T / (sc.I_sp_vacuum * g0)
    dtheta_dt = (v * math.cos(gamma)) / r
    dgamma_dt = (
        (T * math.sin(alpha)) / (m * v)
        + (v / r - mu / (v * r * r)) * math.cos(gamma)
    )

    return [dr_dt, dv_dt, dm_dt, dtheta_dt, dgamma_dt]


# ===========================================================================
# 3. 时间积分格式 (来自 1115_inerOsci)
# ===========================================================================
def euler_step(
    x: State, u: Control, sc: SpacecraftParameters, dt: float, t: float = 0.0
) -> State:
    """显式 Euler 格式 (一阶).

    x_{n+1} = x_n + dt * f(x_n, u_n, t_n)
    """
    f = state_derivative(x, u, sc, t)
    x_new = [xi + dt * fi for xi, fi in zip(x.as_list(), f)]
    return State.from_list(x_new)


def rk4_step(
    x: State, u: Control, sc: SpacecraftParameters, dt: float, t: float = 0.0
) -> State:
    """经典 Runge-Kutta 4 阶格式.

    k1 = f(x_n, u_n, t_n)
    k2 = f(x_n + dt/2 k1, u_n, t_n + dt/2)
    k3 = f(x_n + dt/2 k2, u_n, t_n + dt/2)
    k4 = f(x_n + dt k3, u_n, t_n + dt)
    x_{n+1} = x_n + (dt/6)(k1 + 2k2 + 2k3 + k4)
    """
    x_list = x.as_list()

    def f_from_list(xl: List[float]) -> List[float]:
        return state_derivative(State.from_list(xl), u, sc, t)

    k1 = f_from_list(x_list)
    k2 = f_from_list([xi + 0.5 * dt * ki for xi, ki in zip(x_list, k1)])
    k3 = f_from_list([xi + 0.5 * dt * ki for xi, ki in zip(x_list, k2)])
    k4 = f_from_list([xi + dt * ki for xi, ki in zip(x_list, k3)])

    x_new = [
        xi + (dt / 6.0) * (k1i + 2.0 * k2i + 2.0 * k3i + k4i)
        for xi, k1i, k2i, k3i, k4i in zip(x_list, k1, k2, k3, k4)
    ]
    return State.from_list(x_new)


def implicit_trapezoidal_step(
    x: State, u: Control, sc: SpacecraftParameters, dt: float, t: float = 0.0,
    n_newton: int = 3
) -> State:
    """隐式梯形格式 (二阶 A-稳定).

    x_{n+1} = x_n + (dt/2) * (f(x_n, u_n) + f(x_{n+1}, u_n))

    通过 Newton 迭代求解隐式方程.
    """
    f0 = state_derivative(x, u, sc, t)
    x_pred = [xi + dt * fi for xi, fi in zip(x.as_list(), f0)]  # Euler 预测

    # Newton 迭代
    x_curr = x_pred
    for _ in range(n_newton):
        f_curr = state_derivative(State.from_list(x_curr), u, sc, t + dt)
        x_next = [
            xn + 0.5 * dt * (f0i + f_curri)
            for xn, f0i, f_curri in zip(x.as_list(), f0, f_curr)
        ]
        if max(abs(a - b) for a, b in zip(x_curr, x_next)) < 1e-10:
            break
        x_curr = x_next

    return State.from_list(x_curr)


# ===========================================================================
# 4. 轨迹积分器
# ===========================================================================
def integrate_trajectory(
    x0: State,
    controls: List[Control],
    sc: SpacecraftParameters,
    dt: float,
    method: str = "rk4",
) -> List[State]:
    """积分完整轨迹.

    Parameters
    ----------
    x0 : State
        初始状态.
    controls : List[Control]
        控制序列 (长度 = 时间步数).
    sc : SpacecraftParameters
        航天器参数.
    dt : float
        时间步长 [s].
    method : str
        积分方法: "euler", "rk4", "implicit".

    Returns
    -------
    List[State]
        轨迹 (长度 = len(controls) + 1).
    """
    if method not in ("euler", "rk4", "implicit"):
        raise ValueError(f"未知积分方法: {method}")

    trajectory = [x0]
    x_curr = x0
    for k, u in enumerate(controls):
        t = k * dt
        if method == "euler":
            x_next = euler_step(x_curr, u, sc, dt, t)
        elif method == "rk4":
            x_next = rk4_step(x_curr, u, sc, dt, t)
        else:
            x_next = implicit_trapezoidal_step(x_curr, u, sc, dt, t)
        trajectory.append(x_next)
        x_curr = x_next

    return trajectory


# ===========================================================================
# 自检
# ===========================================================================
def self_check() -> None:
    """动力学模块自检."""
    sc = SpacecraftParameters()
    x0 = State(
        r=EARTH.radius_mean + 200e3,  # 200 km 高度
        v=7800.0,
        m=sc.total_mass,
        theta=0.0,
        gamma=0.0,
    )
    u = Control(thrust=10000.0, alpha=0.1)
    dt = 10.0

    print("[State Dynamics] 初始状态:")
    print(f"  r={x0.r/1e3:.1f} km, v={x0.v:.1f} m/s, m={x0.m:.1f} kg")

    # Euler
    x1_euler = euler_step(x0, u, sc, dt)
    print(f"[Euler] t+{dt}s: r={x1_euler.r/1e3:.2f} km, v={x1_euler.v:.2f} m/s")

    # RK4
    x1_rk4 = rk4_step(x0, u, sc, dt)
    print(f"[RK4]   t+{dt}s: r={x1_rk4.r/1e3:.2f} km, v={x1_rk4.v:.2f} m/s")

    # Implicit
    x1_imp = implicit_trapezoidal_step(x0, u, sc, dt)
    print(f"[Imp]   t+{dt}s: r={x1_imp.r/1e3:.2f} km, v={x1_imp.v:.2f} m/s")

    # 物理边界检查
    valid, msg = x1_rk4.check_physical_bounds()
    print(f"[Bounds] {msg} (valid={valid})")


if __name__ == "__main__":
    self_check()
