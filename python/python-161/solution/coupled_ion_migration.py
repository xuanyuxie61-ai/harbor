"""
coupled_ion_migration.py
基于种子项目 345_exm (ode1 Euler solver + predprey ODE system)
改造为钙钛矿太阳能电池中离子迁移-载流子耦合动力学求解器。

钙钛矿材料中存在可移动离子（如碘空位 V_I^+、甲胺离子 MA^+），
在外加电场和光照下发生迁移，导致：
  1. 能带弯曲随时间演变（滞后效应 / 电流-电压迟滞）
  2. 局部电场屏蔽，影响载流子分离效率

本模块将离子浓度 n_ion 与载流子浓度 (n_e, n_h) 建模为耦合 ODE 系统，
类似 predprey 的捕食者-猎物耦合结构：
  - 离子迁移改变电场（“猎物”环境）
  - 载流子复合受离子屏蔽电场影响（“捕食者”响应）

核心公式：
  1. 离子迁移方程（简化 ODE 模型）：
       d n_ion / dt = -∇·J_ion + G_ion - R_ion
       J_ion = q μ_ion n_ion E - D_ion ∇n_ion
  2. 载流子耦合方程：
       d n_e / dt = G - R(n_e, n_h, E_eff)
       d n_h / dt = G - R(n_e, n_h, E_eff)
     其中 E_eff = E_ext - E_ion 为有效电场。
  3. Euler 方法（ode1）：
       y_{k+1} = y_k + h f(t_k, y_k)
  4. 耦合系统中的“相互作用项”（类比 predprey）：
       ion_feedback = -k_1 * n_ion * n_e  （离子俘获电子）
       carrier_feedback = k_2 * n_e * n_h   （双分子复合）
"""

import numpy as np
from typing import Callable, Tuple


def ode1_euler(
    f: Callable[[float, np.ndarray], np.ndarray],
    tspan: Tuple[float, float],
    y0: np.ndarray,
    n_steps: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Euler 显式 ODE 求解器（对应原项目 ode1）。

    Parameters
    ----------
    f : callable(t, y) -> dy/dt
    tspan : (t0, tf)
    y0 : (m,) array
    n_steps : int

    Returns
    -------
    t : (n_steps+1,) array
    y : (n_steps+1, m) array
    """
    t0, tf = tspan
    if n_steps <= 0:
        raise ValueError("n_steps 必须为正")
    h = (tf - t0) / n_steps
    m = len(y0)
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, m))
    y[0, :] = y0

    for k in range(n_steps):
        ydot = f(t[k], y[k, :])
        # 数值鲁棒性检查
        ydot = np.where(np.isfinite(ydot), ydot, 0.0)
        y[k + 1, :] = y[k, :] + h * ydot
        # 浓度非负约束
        y[k + 1, :] = np.maximum(y[k + 1, :], 0.0)

    return t, y


def coupled_ion_carrier_system(
    t: float,
    y: np.ndarray,
    mu_ion: float,
    D_ion: float,
    mu_e: float,
    mu_h: float,
    E_ext: float,
    G_light: float,
    k_rec: float,
    k_ion_trap: float,
    n_ion_eq: float,
) -> np.ndarray:
    """
    离子-载流子耦合 ODE 的右端函数。

    状态向量 y = [n_ion, n_e, n_h]

    Parameters
    ----------
    E_ext : float
        外加电场 [V/cm]
    G_light : float
        光生载流子率 [cm^{-3} s^{-1}]
    k_rec : float
        双分子复合系数 [cm^3/s]
    k_ion_trap : float
        离子对电子的俘获系数 [cm^3/s]
    n_ion_eq : float
        离子平衡浓度 [cm^{-3}]
    """
    n_ion, n_e, n_h = y[0], y[1], y[2]

    # 离子屏蔽电场（简化：线性屏蔽）
    E_ion = 1e-15 * (n_ion - n_ion_eq)  # V/cm
    E_eff = E_ext - E_ion

    # 离子电流密度（一维简化）
    J_ion = q * mu_ion * n_ion * E_eff - q * D_ion * (n_ion - n_ion_eq) / 1e-4
    # 离子连续性（简化为一空间点）
    d_n_ion_dt = -J_ion / (q * 1e-4)  # 归一化

    # 载流子产生与复合
    # 有效电场影响分离效率
    separation_eff = min(abs(E_eff) / (abs(E_ext) + 1e-10), 1.0)
    G_eff = G_light * separation_eff

    # 复合：辐射 + SRH（离子陷阱贡献）
    R_total = k_rec * n_e * n_h + k_ion_trap * n_ion * n_e

    d_n_e_dt = G_eff - R_total
    d_n_h_dt = G_eff - R_total

    return np.array([d_n_ion_dt, d_n_e_dt, d_n_h_dt])


# 物理常数
q = 1.602176634e-19  # C


def solve_hysteresis_cycle(
    voltage_sweep: np.ndarray,
    time_per_step: float = 1e-3,
    mu_ion: float = 1e-10,  # cm^2/(V·s) 离子迁移率（很慢）
    thickness: float = 5e-5,  # cm
    n_ion0: float = 1e16,
    n_e0: float = 1e10,
    n_h0: float = 1e10,
    G_light: float = 1e21,
    k_rec: float = 1e-10,
    k_ion_trap: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    模拟一个 I-V 扫描迟滞循环。

    Parameters
    ----------
    voltage_sweep : (n_v,) array
        电压扫描序列 [V]
    time_per_step : float
        每步停留时间 [s]

    Returns
    -------
    V : (n_v,) array
        电压
    J : (n_v,) array
        电流密度 [mA/cm^2]
    n_ion_t : (n_v,) array
        离子浓度随时间演变
    E_ion_t : (n_v,) array
        离子屏蔽电场
    """
    n_v = len(voltage_sweep)
    if n_v < 2:
        raise ValueError("电压扫描点必须 ≥ 2")

    V = voltage_sweep
    J = np.zeros(n_v)
    n_ion_t = np.zeros(n_v)
    E_ion_t = np.zeros(n_v)

    y = np.array([n_ion0, n_e0, n_h0], dtype=float)
    n_ion_eq = n_ion0

    for i in range(n_v):
        V_step = V[i]
        E_ext = V_step / thickness  # V/cm

        # 定义当前步的 ODE 右端
        def f(t, yy):
            return coupled_ion_carrier_system(
                t, yy, mu_ion, 1e-12, 20.0, 10.0,
                E_ext, G_light, k_rec, k_ion_trap, n_ion_eq,
            )

        # 使用 ode1 (Euler) 推进
        _, y_hist = ode1_euler(f, (0.0, time_per_step), y, n_steps=50)
        y = y_hist[-1, :]

        n_ion, n_e, n_h = y
        # 数值鲁棒性：确保载流子浓度不会归零
        n_e = max(n_e, 1e12)
        n_h = max(n_h, 1e12)
        n_ion_t[i] = n_ion
        E_ion = 1e-15 * (n_ion - n_ion_eq)
        E_ion_t[i] = E_ion
        E_eff = E_ext - E_ion

        # 电流密度：J = q (μ_e n_e + μ_h n_h) E_eff + q D dn/dx (忽略扩散)
        J_ohm = q * (20.0 * n_e + 10.0 * n_h) * abs(E_eff)  # A/cm^2
        J[i] = J_ohm * 1e3  # mA/cm^2

    return V, J, n_ion_t, E_ion_t


def predprey_style_ion_dynamics(
    tspan: Tuple[float, float] = (0.0, 100.0),
    n_steps: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    借用 predprey 的耦合结构，构建简化的离子-载流子振荡模型。

    模型：
      dV_I/dt = a V_I - b V_I n_e          (离子增长，受电子抑制)
      dn_e/dt = -c n_e + d V_I n_e          (电子衰减，受离子促进)
    其中 V_I 为碘空位浓度，n_e 为电子浓度。
    """
    a, b, c, d = 0.1, 0.02, 0.15, 0.01
    y0 = np.array([20.0, 10.0])  # 归一化浓度

    def f(t, y):
        V_I, n_e = y
        # 数值鲁棒性：防止爆炸
        V_I = min(V_I, 1e6)
        n_e = min(n_e, 1e6)
        dV = a * V_I - b * V_I * n_e
        dne = -c * n_e + d * V_I * n_e
        return np.array([dV, dne])

    t, y_hist = ode1_euler(f, tspan, y0, n_steps)
    return t, y_hist[:, 0], y_hist[:, 1]


if __name__ == "__main__":
    # 测试 predprey 风格振荡
    t, V_I, n_e = predprey_style_ion_dynamics()
    print(f"碘空位浓度范围: [{V_I.min():.2f}, {V_I.max():.2f}]")
    print(f"电子浓度范围: [{n_e.min():.2f}, {n_e.max():.2f}]")

    # 测试 I-V 迟滞
    V_fwd = np.linspace(0.0, 1.0, 21)
    V_rev = np.linspace(1.0, 0.0, 21)
    V_full = np.concatenate([V_fwd, V_rev])
    V, J, n_ion_t, E_ion_t = solve_hysteresis_cycle(V_full, time_per_step=5e-4)
    print(f"最大电流密度: {J.max():.3f} mA/cm^2")
    print(f"离子浓度变化: {n_ion_t.min():.3e} -> {n_ion_t.max():.3e}")
