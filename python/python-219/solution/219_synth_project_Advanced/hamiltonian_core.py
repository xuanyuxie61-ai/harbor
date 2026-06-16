"""
hamiltonian_core.py
===================

Pontryagin 极大值原理的 Hamilton 系统核心模块。

本模块实现最优控制的 Hamilton 函数、伴随方程、最优性条件。
是整个框架的数学核心。

数学公式:
---------
1. Hamilton 函数:
     H(x, u, lambda, t) = L(x, u, t) + lambda^T f(x, u, t)
     其中 L 为 Lagrange 被积函数 (运行代价),
          f 为状态方程右端,
          lambda 为伴随变量 (costate).

2. Pontryagin 极大值原理 (PMP):
     最优控制 u* 满足:
       H(x*, u*, lambda*, t) = max_{u in U} H(x*, u, lambda*, t)

3. 伴随方程:
     dlambda/dt = -partial H / partial x

4. 横截条件 (终端约束):
     lambda(T) = partial Phi / partial x(T)
     其中 Phi 为终端代价.

5. 控制最优性 (对无约束控制):
     partial H / partial u = 0

6. 对航天器轨迹:
     H = lambda_r * v sin(gamma)
       + lambda_v * [(T cos(alpha) - D)/m - mu/r^2 sin(gamma)]
       + lambda_m * [-T / (I_sp g_0)]
       + lambda_theta * [v cos(gamma) / r]
       + lambda_gamma * [(T sin(alpha))/(m v) + (v/r - mu/(v r^2)) cos(gamma)]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from scientific_constants import EARTH, SpacecraftParameters, atmospheric_density
from state_dynamics import Control, State, state_derivative


# ===========================================================================
# 1. Costate 数据结构
# ===========================================================================
@dataclass
class Costate:
    """伴随变量 lambda = (lr, lv, lm, ltheta, lgamma)."""
    lr: float      # 对应 r
    lv: float      # 对应 v
    lm: float      # 对应 m
    ltheta: float  # 对应 theta
    lgamma: float  # 对应 gamma

    def as_list(self) -> List[float]:
        return [self.lr, self.lv, self.lm, self.ltheta, self.lgamma]

    @staticmethod
    def from_list(lam: List[float]) -> "Costate":
        if len(lam) != 5:
            raise ValueError(f"costate 向量长度必须为 5, 实际 {len(lam)}")
        return Costate(*lam)

    def norm(self) -> float:
        return math.sqrt(sum(c * c for c in self.as_list()))


# ===========================================================================
# 2. Hamilton 函数计算
# ===========================================================================
def running_cost(x: State, u: Control, sc: SpacecraftParameters) -> float:
    """运行代价 L(x, u, t).

    选择燃料最优指标:
        L = T / T_max  (归一化推力, 最小化燃料消耗)

    或能量最优:
        L = 0.5 * T^2 / T_max^2  (平滑控制)
    """
    return 0.5 * (u.thrust / sc.thrust_max) ** 2


def hamiltonian(
    x: State,
    u: Control,
    lam: Costate,
    sc: SpacecraftParameters,
    t: float = 0.0,
) -> float:
    """计算 Hamilton 函数 H = L + lambda^T f.

    Parameters
    ----------
    x : State
        状态.
    u : Control
        控制.
    lam : Costate
        伴随变量.
    sc : SpacecraftParameters
        航天器参数.
    t : float
        时间.

    Returns
    -------
    float
        Hamilton 函数值.
    """
    L = running_cost(x, u, sc)
    f = state_derivative(x, u, sc, t)
    lam_list = lam.as_list()
    dot_product = sum(l * fi for l, fi in zip(lam_list, f))
    return L + dot_product


# ===========================================================================
# 3. 伴随方程 dlambda/dt = -dH/dx
# ===========================================================================
def costate_derivative(
    x: State,
    u: Control,
    lam: Costate,
    sc: SpacecraftParameters,
    t: float = 0.0,
    eps: float = 1e-6,
) -> List[float]:
    """计算伴随导数 dlambda/dt = -partial H / partial x.

    采用中心差分近似:
        dH/dx_i ≈ (H(x + eps e_i) - H(x - eps e_i)) / (2 eps)

    Parameters
    ----------
    x, u, lam, sc, t : 同 hamiltonian
    eps : float
        差分步长.

    Returns
    -------
    List[float]
        [dlr/dt, dlv/dt, dlm/dt, dltheta/dt, dlgamma/dt].
    """
    x_list = x.as_list()
    n = len(x_list)
    dH_dx = [0.0] * n

    for i in range(n):
        x_plus = list(x_list)
        x_minus = list(x_list)
        x_plus[i] += eps
        x_minus[i] -= eps

        H_plus = hamiltonian(State.from_list(x_plus), u, lam, sc, t)
        H_minus = hamiltonian(State.from_list(x_minus), u, lam, sc, t)
        dH_dx[i] = (H_plus - H_minus) / (2.0 * eps)

    return [-d for d in dH_dx]


# ===========================================================================
# 4. 最优控制计算
# ===========================================================================
def optimal_thrust_angle(
    x: State, lam: Costate, sc: SpacecraftParameters
) -> float:
    """计算最优推力方向角 alpha*.

    由 dH/dalpha = 0:
        dH/dalpha = lambda_v * (-T sin(alpha)) / m
                   + lambda_gamma * (T cos(alpha)) / (m v)
                  = 0
        => tan(alpha*) = lambda_gamma / (lambda_v * v)
        => alpha* = atan2(lambda_gamma, lambda_v * v)

    注意: 这是无约束情况; 若有 alpha 约束, 需要投影.
    """
    v = max(x.v, 1.0)
    num = lam.lgamma
    den = lam.lv * v
    return math.atan2(num, den)


def optimal_thrust_magnitude(
    x: State,
    lam: Costate,
    sc: SpacecraftParameters,
    alpha: float,
) -> float:
    """计算最优推力大小 T* (bang-bang 或奇异).

    由 dH/dT = 0 (对连续控制):
        dH/dT = T / T_max^2
              + lambda_v * cos(alpha) / m
              + lambda_m * (-1 / (I_sp g_0))
              + lambda_gamma * sin(alpha) / (m v)
        = 0
        => T* = -T_max^2 * [lambda_v cos(alpha)/m - lambda_m/(I_sp g_0)
                             + lambda_gamma sin(alpha)/(m v)]

    然后投影到 [T_min, T_max].
    """
    v = max(x.v, 1.0)
    m = max(x.m, sc.dry_mass)
    g0 = EARTH.g0

    switching = (
        lam.lv * math.cos(alpha) / m
        - lam.lm / (sc.I_sp_vacuum * g0)
        + lam.lgamma * math.sin(alpha) / (m * v)
    )
    T_star = -(sc.thrust_max ** 2) * switching

    # 投影到约束
    T_star = max(sc.thrust_min, min(sc.thrust_max, T_star))
    return T_star


def optimal_control(
    x: State, lam: Costate, sc: SpacecraftParameters
) -> Control:
    """计算最优控制 u* = (T*, alpha*)."""
    alpha_star = optimal_thrust_angle(x, lam, sc)
    T_star = optimal_thrust_magnitude(x, lam, sc, alpha_star)
    return Control(thrust=T_star, alpha=alpha_star)


# ===========================================================================
# 5. 终端代价与横截条件
# ===========================================================================
def terminal_cost(x_final: State, x_target: State) -> float:
    """终端代价 Phi (最小化终端状态误差).

    Phi = 0.5 * (w_r (r_f - r_t)^2 + w_v (v_f - v_t)^2
                 + w_m (m_f - m_t)^2 + w_gamma (gamma_f - gamma_t)^2)
    """
    w_r = 1.0 / (100e3) ** 2     # 100 km 尺度
    w_v = 1.0 / (100.0) ** 2     # 100 m/s 尺度
    w_m = 1.0 / (100.0) ** 2     # 100 kg 尺度
    w_gamma = 1.0 / (0.01) ** 2  # 0.01 rad 尺度

    return 0.5 * (
        w_r * (x_final.r - x_target.r) ** 2
        + w_v * (x_final.v - x_target.v) ** 2
        + w_m * (x_final.m - x_target.m) ** 2
        + w_gamma * (x_final.gamma - x_target.gamma) ** 2
    )


def terminal_costate(
    x_final: State, x_target: State
) -> Costate:
    """终端伴随变量 lambda(T) = dPhi/dx(T).

    由终端代价的梯度给出.
    """
    w_r = 1.0 / (100e3) ** 2
    w_v = 1.0 / (100.0) ** 2
    w_m = 1.0 / (100.0) ** 2
    w_gamma = 1.0 / (0.01) ** 2

    return Costate(
        lr=w_r * (x_final.r - x_target.r),
        lv=w_v * (x_final.v - x_target.v),
        lm=w_m * (x_final.m - x_target.m),
        ltheta=0.0,  # theta 无终端约束
        lgamma=w_gamma * (x_final.gamma - x_target.gamma),
    )


# ===========================================================================
# 6. Hamilton 守恒检验
# ===========================================================================
def hamiltonian_conservation_check(
    trajectory_states: List[State],
    trajectory_costates: List[Costate],
    trajectory_controls: List[Control],
    sc: SpacecraftParameters,
    dt: float,
) -> List[float]:
    """检验 Hamilton 函数沿轨迹的守恒性.

    对自治系统 (H 不显含 t), 最优轨迹上 H = const.

    Returns
    -------
    List[float]
        各时间步的 H 值.
    """
    H_values = []
    for x, lam, u in zip(
        trajectory_states, trajectory_costates, trajectory_controls
    ):
        H = hamiltonian(x, u, lam, sc)
        H_values.append(H)
    return H_values


# ===========================================================================
# 自检
# ===========================================================================
def self_check() -> None:
    """Hamilton 模块自检."""
    sc = SpacecraftParameters()
    x = State(
        r=EARTH.radius_mean + 300e3,
        v=7700.0,
        m=1800.0,
        theta=0.0,
        gamma=0.0,
    )
    u = Control(thrust=15000.0, alpha=0.1)
    lam = Costate(lr=1e-6, lv=1e-3, lm=-1e-4, ltheta=0.0, lgamma=1e-2)

    H = hamiltonian(x, u, lam, sc)
    print(f"[Hamiltonian] H = {H:.6e}")

    dlam = costate_derivative(x, u, lam, sc)
    print(f"[Costate Derivative] dlam/dt = {[f'{d:.3e}' for d in dlam]}")

    u_star = optimal_control(x, lam, sc)
    print(f"[Optimal Control] T* = {u_star.thrust:.1f} N, alpha* = {u_star.alpha:.4f} rad")

    x_target = State(
        r=EARTH.radius_mean + 400e3,
        v=7670.0,
        m=1500.0,
        theta=0.5,
        gamma=0.0,
    )
    Phi = terminal_cost(x, x_target)
    lam_T = terminal_costate(x, x_target)
    print(f"[Terminal] Phi = {Phi:.6e}, |lam_T| = {lam_T.norm():.6e}")


if __name__ == "__main__":
    self_check()
