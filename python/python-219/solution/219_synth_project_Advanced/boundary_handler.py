"""
boundary_handler.py
===================

边界条件处理与数值鲁棒性模块。

在最优控制中, 边界处理涉及:
  1. 状态约束 (如地面碰撞避免, 质量下限)
  2. 控制约束 (推力上下限, 方向角约束)
  3. 终端约束 (终端状态目标)
  4. 数值溢出保护

数学公式:
---------
1. 状态投影 (违反约束时):
     x_proj = argmin_{y in X} ||y - x||

2. 控制饱和:
     u_sat = clamp(u, u_min, u_max)

3. 障碍函数 (内点法):
     B(x) = -mu * sum_i log(g_i(x))
     其中 g_i(x) >= 0 为不等式约束

4. 惩罚函数:
     P(x) = rho * sum_i max(0, -g_i(x))^2
"""

from __future__ import annotations

import math
from typing import List, Tuple

from scientific_constants import EARTH, SpacecraftParameters
from state_dynamics import Control, State


# ===========================================================================
# 1. 状态边界检查与投影
# ===========================================================================
def project_state_to_feasible(
    x: State, sc: SpacecraftParameters
) -> State:
    """将状态投影到可行域.

    约束:
        r >= R_E + h_min  (最低轨道高度)
        v >= v_min         (最小速度)
        m >= m_dry         (干质量下限)
        |gamma| <= pi/2    (飞行路径角约束)
    """
    r = max(x.r, EARTH.radius_mean + 100e3)  # 最低 100 km
    v = max(x.v, 100.0)  # 最小 100 m/s
    m = max(x.m, sc.dry_mass)
    gamma = max(-math.pi / 2, min(math.pi / 2, x.gamma))
    theta = x.theta  # theta 无约束

    return State(r=r, v=v, m=m, theta=theta, gamma=gamma)


def check_state_bounds(
    x: State, sc: SpacecraftParameters
) -> Tuple[bool, str]:
    """检查状态是否满足物理约束."""
    h_min = 100e3
    if x.r < EARTH.radius_mean + h_min:
        return False, f"高度过低: h = {(x.r - EARTH.radius_mean)/1e3:.1f} km < {h_min/1e3:.1f} km"
    if x.v < 0:
        return False, f"速度为负: v = {x.v:.1f} m/s"
    if x.m < sc.dry_mass:
        return False, f"质量低于干质量: m = {x.m:.1f} < {sc.dry_mass:.1f} kg"
    if abs(x.gamma) > math.pi / 2:
        return False, f"飞行路径角过大: gamma = {x.gamma:.4f} rad"
    return True, "OK"


# ===========================================================================
# 2. 控制约束与饱和
# ===========================================================================
def saturate_control(
    u: Control, sc: SpacecraftParameters
) -> Control:
    """控制饱和 (投影到约束集).

    T in [T_min, T_max]
    alpha in [-pi, pi]
    """
    T_sat = max(sc.thrust_min, min(sc.thrust_max, u.thrust))
    alpha_sat = max(-math.pi, min(math.pi, u.alpha))
    return Control(thrust=T_sat, alpha=alpha_sat)


def check_control_bounds(
    u: Control, sc: SpacecraftParameters
) -> Tuple[bool, str]:
    """检查控制是否满足约束."""
    if not (sc.thrust_min <= u.thrust <= sc.thrust_max):
        return False, f"推力超出范围: T = {u.thrust:.1f} N"
    if abs(u.alpha) > math.pi:
        return False, f"方向角超出范围: alpha = {u.alpha:.4f} rad"
    return True, "OK"


# ===========================================================================
# 3. 障碍函数与惩罚函数
# ===========================================================================
def barrier_function(
    x: State, sc: SpacecraftParameters, mu: float = 1.0
) -> float:
    """对数障碍函数 (内点法).

    B(x) = -mu * sum_i log(g_i(x))
    其中 g_i(x) >= 0 为不等式约束.

    约束:
        g_1 = r - (R_E + h_min) >= 0
        g_2 = m - m_dry >= 0
        g_3 = pi/2 - |gamma| >= 0
    """
    h_min = 100e3
    eps = 1e-10

    g1 = x.r - (EARTH.radius_mean + h_min)
    g2 = x.m - sc.dry_mass
    g3 = math.pi / 2 - abs(x.gamma)

    B = 0.0
    if g1 > eps:
        B -= mu * math.log(g1)
    if g2 > eps:
        B -= mu * math.log(g2)
    if g3 > eps:
        B -= mu * math.log(g3)

    return B


def penalty_function(
    x: State, sc: SpacecraftParameters, rho: float = 1e6
) -> float:
    """二次惩罚函数 (外点法).

    P(x) = rho * sum_i max(0, -g_i(x))^2
    """
    h_min = 100e3

    g1 = x.r - (EARTH.radius_mean + h_min)
    g2 = x.m - sc.dry_mass
    g3 = math.pi / 2 - abs(x.gamma)

    P = 0.0
    if g1 < 0:
        P += rho * g1 * g1
    if g2 < 0:
        P += rho * g2 * g2
    if g3 < 0:
        P += rho * g3 * g3

    return P


# ===========================================================================
# 4. 数值鲁棒性保护
# ===========================================================================
def safe_divide(a: float, b: float, eps: float = 1e-15) -> float:
    """安全除法 (避免除以零)."""
    if abs(b) < eps:
        return a / eps if b >= 0 else -a / eps
    return a / b


def safe_sqrt(x: float) -> float:
    """安全平方根 (避免负数)."""
    return math.sqrt(max(0.0, x))


def safe_log(x: float, eps: float = 1e-15) -> float:
    """安全对数 (避免零或负数)."""
    return math.log(max(eps, x))


def safe_exp(x: float, limit: float = 500.0) -> float:
    """安全指数 (避免溢出)."""
    x_clamp = max(-limit, min(limit, x))
    return math.exp(x_clamp)


def clamp(x: float, lo: float, hi: float) -> float:
    """数值钳制."""
    return max(lo, min(hi, x))


# ===========================================================================
# 自检
# ===========================================================================
def self_check() -> None:
    """边界处理模块自检."""
    sc = SpacecraftParameters()

    # 状态投影
    x_bad = State(
        r=EARTH.radius_mean + 50e3,  # 过低
        v=-100.0,  # 负速度
        m=400.0,  # 低于干质量
        theta=0.0,
        gamma=1.8,  # 过大
    )
    x_proj = project_state_to_feasible(x_bad, sc)
    valid, msg = check_state_bounds(x_proj, sc)
    print(f"[State Projection] valid={valid}, msg={msg}")
    print(f"  原始: r={x_bad.r/1e3:.1f} km, v={x_bad.v:.1f}, m={x_bad.m:.1f}, gamma={x_bad.gamma:.3f}")
    print(f"  投影: r={x_proj.r/1e3:.1f} km, v={x_proj.v:.1f}, m={x_proj.m:.1f}, gamma={x_proj.gamma:.3f}")

    # 控制饱和
    u_bad = Control(thrust=50000.0, alpha=4.0)
    u_sat = saturate_control(u_bad, sc)
    valid, msg = check_control_bounds(u_sat, sc)
    print(f"[Control Saturation] valid={valid}, T={u_sat.thrust:.1f}, alpha={u_sat.alpha:.3f}")

    # 障碍函数
    x_ok = State(r=EARTH.radius_mean + 300e3, v=7700.0, m=1800.0, theta=0.0, gamma=0.1)
    B = barrier_function(x_ok, sc, mu=1.0)
    P = penalty_function(x_ok, sc, rho=1e6)
    print(f"[Barrier/Penalty] B={B:.4e}, P={P:.4e}")

    # 数值安全
    print(f"[Safe Ops] 1/0 = {safe_divide(1.0, 0.0):.3e}")
    print(f"[Safe Ops] sqrt(-1) = {safe_sqrt(-1.0):.3e}")
    print(f"[Safe Ops] exp(1000) = {safe_exp(1000.0):.3e}")


if __name__ == "__main__":
    self_check()
