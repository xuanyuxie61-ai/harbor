# -*- coding: utf-8 -*-
"""
binary_inspiral.py
==================
PROJECT_254 — 计算天体物理：双中子星并合与 kilonova 辐射转移

双中子星旋进轨道的 2.5PN (后牛顿) 演化.

运动方程
--------
在质心系中, 相对位矢 r = x_1 - x_2, 相对速度 v = dr/dt::

    dv/dt = - G M / r^2 * n_hat  +  F_1PN + F_2PN + F_2.5PN

其中 n_hat = r / |r|. 各阶修正 (Blanchet 2014)::

    F_1PN = (G M / r^2) * (v^2/c^2 * A_1 + G M/(r c^2) * A_2) * n_hat
            + (G M / r^2) * (v/c^2 * B) * v_hat

    F_2.5PN 为辐射反作用力, 导致轨道衰减::

        F_2.5PN = (8/5) * (G M)^3 * eta / (r^3 c^5) *
                  [ (17/3 * G M/r + 3 v^2) * v_hat * n_hat
                    - v * (3 G M/r * n_hat + v * v_hat) ]

轨道能量::

    E = - G m_1 m_2 / (2 a)  *  [1 - (7-eta)/4 * G M/(a c^2) + ...]

引力波频率::

    f_GW = 2 * f_orb = (1/pi) * sqrt(G M / a^3)

引力波啁啾质量::

    M_c = (m_1 m_2)^{3/5} / (m_1 + m_2)^{1/5} = M * eta^{3/5}

映射种子项目
-----------
- 1369 (two_body_ode)      → 二体 ODE 右端项的结构
- 860  (pendulum_nonlinear)→ 精确积分思想: 用 Jacobi 椭圆函数类比
                              PN 轨道的进动相位
"""

from __future__ import annotations
import math
from typing import Tuple, List

from physical_constants import C_LIGHT, G_GRAV, M_SUN


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
def default_binary_params() -> dict:
    """GW170817 样式的默认双中子星参数."""
    return {
        "m1_g": 1.46 * M_SUN,          # 主星质量 [g]
        "m2_g": 1.27 * M_SUN,          # 次星质量 [g]
        "eccentricity": 0.0,            # 初始偏心率 (准圆轨道)
        "initial_separation_cm": 4.0e7, # 初始间距 [cm] (~400 km)
        "initial_phase_rad": 0.0,       # 初始轨道相位
    }


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def symmetric_mass_ratio(m1: float, m2: float) -> float:
    """对称质量比  eta = m1 m2 / (m1+m2)^2,  取值 (0, 1/4]."""
    M = m1 + m2
    return m1 * m2 / (M * M)


def chirp_mass(m1: float, m2: float) -> float:
    """啁啾质量  M_c = (m1 m2)^{3/5} / (m1+m2)^{1/5}."""
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def orbital_frequency_newtonian(M: float, a: float) -> float:
    """牛顿轨道角频率  Omega = sqrt(G M / a^3)  [rad/s]."""
    return math.sqrt(G_GRAV * M / a ** 3)


def gw_frequency_from_orbital(f_orb: float) -> float:
    """引力波频率 (主导 quadrupole 模式)  f_GW = 2 f_orb."""
    return 2.0 * f_orb


def pn_expansion_parameter(M: float, r: float) -> float:
    """PN 展开参数  x = (G M Omega / c^3)^{2/3} ~ G M / (r c^2)."""
    return G_GRAV * M / (r * C_LIGHT ** 2)


# ---------------------------------------------------------------------------
# 右端项: 2.5PN 加速度
# ---------------------------------------------------------------------------
def acceleration_2_5pn(
    rx: float, ry: float, vx: float, vy: float,
    m1: float, m2: float
) -> Tuple[float, float]:
    """计算 2.5PN 精度的相对加速度 (ax, ay) [cm/s^2].

    状态向量: (rx, ry, vx, vy)
    采用 Blanchet-Damour 形式, 只保留到 O(c^{-5}) 辐射反作用项
    与 O(c^{-2}) 的 1PN 项. 2PN 项省略以控制复杂度.

    Parameters
    ----------
    rx, ry : float  相对位矢分量 [cm]
    vx, vy : float  相对速度分量 [cm/s]
    m1, m2 : float  两星质量 [g]

    Returns
    -------
    ax, ay : Tuple[float, float]  加速度 [cm/s^2]
    """
    M = m1 + m2
    eta = symmetric_mass_ratio(m1, m2)
    r = math.sqrt(rx * rx + ry * ry)
    r = max(r, 1.0e5)  # 防止奇点
    v2 = vx * vx + vy * vy
    vdotr = vx * rx + vy * ry
    n_x = rx / r
    n_y = ry / r

    # 牛顿项
    GM_r2 = G_GRAV * M / (r * r)
    aN_x = -GM_r2 * n_x
    aN_y = -GM_r2 * n_y

    # 1PN 修正 (简化形式, Damour & Deruelle 1985)
    c2 = C_LIGHT * C_LIGHT
    xPN = G_GRAV * M / (r * c2)  # PN 参数
    v2_c2 = v2 / c2

    # 1PN 径向系数
    A_1pn = (1.0 + eta * (-1.0)) * v2_c2 + (3.0 + eta * 0.5) * xPN \
            - 1.5 * (vdotr / r) ** 2 / c2
    # 1PN 速度系数
    B_1pn = -(4.0 - 2.0 * eta) * vdotr / (r * c2)

    a1PN_x = GM_r2 * (A_1pn * n_x + B_1pn * vx)
    a1PN_y = GM_r2 * (A_1pn * n_y + B_1pn * vy)

    # 2.5PN 辐射反作用 (leading-order radiation reaction, Burke-Thorne)
    c5 = c2 * c2 * C_LIGHT
    coeff_25 = (8.0 / 5.0) * G_GRAV ** 2 * M ** 2 * eta / (r ** 3 * c5)
    factor_radial = (17.0 / 3.0 * G_GRAV * M / r + 3.0 * v2)
    F_25_n = coeff_25 * (factor_radial * vdotr / r
                         - G_GRAV * M / r * vdotr / r)
    F_25_v = coeff_25 * (3.0 * G_GRAV * M / r * vdotr / r + v2)
    a25PN_x = F_25_n * n_x - F_25_v * vx
    a25PN_y = F_25_n * n_y - F_25_v * vy

    return aN_x + a1PN_x + a25PN_x, aN_y + a1PN_y + a25PN_y


# ---------------------------------------------------------------------------
# 四阶 Runge-Kutta 步进
# ---------------------------------------------------------------------------
def rk4_step(state: Tuple[float, float, float, float],
             dt: float, m1: float, m2: float
             ) -> Tuple[float, float, float, float]:
    """单步 RK4 积分.

    state = (rx, ry, vx, vy)
    """
    def rhs(s):
        rx, ry, vx, vy = s
        ax, ay = acceleration_2_5pn(rx, ry, vx, vy, m1, m2)
        return (vx, vy, ax, ay)

    k1 = rhs(state)
    s2 = tuple(state[i] + 0.5 * dt * k1[i] for i in range(4))
    k2 = rhs(s2)
    s3 = tuple(state[i] + 0.5 * dt * k2[i] for i in range(4))
    k3 = rhs(s3)
    s4 = tuple(state[i] + dt * k3[i] for i in range(4))
    k4 = rhs(s4)
    return tuple(
        state[i] + (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
        for i in range(4)
    )


# ---------------------------------------------------------------------------
# 轨道演化
# ---------------------------------------------------------------------------
def evolve_orbit(
    m1: float, m2: float,
    r0_cm: float,
    dt_s: float,
    n_steps: int,
    stop_at_contact: bool = True,
) -> dict:
    """演化双中子星轨道并返回轨迹.

    Parameters
    ----------
    m1, m2      : float  质量 [g]
    r0_cm       : float  初始间距 [cm]
    dt_s        : float  时间步长 [s]
    n_steps     : int    最大步数
    stop_at_contact : bool  在接触时停止 (r = R1 + R2)

    Returns
    -------
    dict  包含 time, rx, ry, vx, vy, r, f_GW, E_orb
    """
    R_NS = 1.2e6  # 中子星半径 [cm]
    r_contact = 2.0 * R_NS

    # 初始圆轨道速度
    M = m1 + m2
    v0 = math.sqrt(G_GRAV * M / r0_cm)
    state = (r0_cm, 0.0, 0.0, v0)

    traj = {
        "time": [], "rx": [], "ry": [], "vx": [], "vy": [],
        "r": [], "f_GW": [], "E_orb": [],
    }

    for step in range(n_steps):
        t = step * dt_s
        rx, ry, vx, vy = state
        r = math.sqrt(rx * rx + ry * ry)
        v = math.sqrt(vx * vx + vy * vy)

        # 轨道能量 (牛顿 + 1PN 近似)
        E_newt = 0.5 * v * v - G_GRAV * M / r
        E_orb = m1 * m2 / M * E_newt

        # 引力波频率
        Omega = math.sqrt(G_GRAV * M / max(r, r_contact) ** 3)
        f_gw = Omega / math.pi  # f_GW = 2 * f_orb = Omega / pi

        traj["time"].append(t)
        traj["rx"].append(rx)
        traj["ry"].append(ry)
        traj["vx"].append(vx)
        traj["vy"].append(vy)
        traj["r"].append(r)
        traj["f_GW"].append(f_gw)
        traj["E_orb"].append(E_orb)

        # 接触停止
        if stop_at_contact and r <= r_contact:
            break

        # RK4 步进
        state = rk4_step(state, dt_s, m1, m2)

    return traj


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _self_check() -> bool:
    """验证对称质量比、啁啾质量和牛顿轨道."""
    m1, m2 = 1.4 * M_SUN, 1.4 * M_SUN
    eta = symmetric_mass_ratio(m1, m2)
    assert abs(eta - 0.25) < 1.0e-10, "Equal-mass eta should be 0.25"
    Mc = chirp_mass(m1, m2)
    assert abs(Mc / M_SUN - 1.2188) < 0.005, "Chirp mass mismatch"
    # 牛顿圆轨道
    r0 = 4.0e7
    Omega = orbital_frequency_newtonian(m1 + m2, r0)
    P_orb = 2.0 * math.pi / Omega
    assert 0.001 < P_orb < 0.1, "Orbital period unphysical"
    return True


if __name__ == "__main__":
    _self_check()
    print("binary_inspiral self-check passed.")
    params = default_binary_params()
    traj = evolve_orbit(
        m1=params["m1_g"], m2=params["m2_g"],
        r0_cm=params["initial_separation_cm"],
        dt_s=1.0e-5, n_steps=2000,
    )
    print(f"  steps integrated : {len(traj['time'])}")
    print(f"  initial r        : {traj['r'][0]:.3e} cm")
    print(f"  final r          : {traj['r'][-1]:.3e} cm")
    print(f"  peak f_GW        : {max(traj['f_GW']):.2f} Hz")
