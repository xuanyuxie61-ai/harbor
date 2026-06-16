# -*- coding: utf-8 -*-
"""
electron_trajectory_rk4.py
==========================
快电子相空间轨迹 RK4 积分.

核心算法 (来自 830_ode_rk4):
----------------------------
经典 4 阶 Runge-Kutta 方法:
    k1 = f(t_n, y_n)
    k2 = f(t_n + h/2, y_n + h k1/2)
    k3 = f(t_n + h/2, y_n + h k2/2)
    k4 = f(t_n + h, y_n + h k3)
    y_{n+1} = y_n + (h/6)(k1 + 2k2 + 2k3 + k4)

ODE 系统 (来自 019_arneodo_ode, 018_arenstorf_ode):
----------------------------------------------------
快电子简化相空间 ODE (x, p_x, E_kin):
    dx/dt     = v_x = p_x / (γ m_e)
    dp_x/dt   = -e E_x - ν_ei p_x + F_stop(E)
    dE_kin/dt = -ν_loss · E_kin - P_rad(E)

其中:
    γ = 1 + E_kin / (m_e c^2)
    ν_ei = Spitzer 碰撞频率
    F_stop = Bethe stopping force
    P_rad = 韧致辐射损失

守恒量监测 (来自 018_arenstorf_ode/arenstorf_conserved):
    H = γ m_e c^2 + e φ(x) - ∫ F_stop dx
    应近似守恒 (忽略碰撞耗散).

参数管理 (来自 019_arneodo_ode/arneodo_parameters):
    默认参数 + 用户覆盖模式.
"""

import math

from plasma_parameters import (
    ELECTRON_MASS, ELECTRON_CHARGE, SPEED_OF_LIGHT,
    EV_TO_JOULE, relativistic_gamma,
    spitzer_collisional_frequency, bethe_stopping_power,
    CORONA_ELECTRON_DENSITY, CORONA_ELECTRON_TEMPERATURE,
    CORONA_ION_CHARGE_Z,
)


# ============================================================
# ODE 参数管理 (来自 019_arneodo_ode/arneodo_parameters)
# ============================================================
class ElectronTrajectoryParams:
    """
    电子轨迹 ODE 参数集合.

    设计模式来自 019_arneodo_ode 的参数管理:
    支持默认值 + 用户覆盖.
    """

    def __init__(self, **kwargs):
        # 默认值
        self.n_e = kwargs.get("n_e", CORONA_ELECTRON_DENSITY)
        self.T_e_ev = kwargs.get("T_e_ev", CORONA_ELECTRON_TEMPERATURE)
        self.Z_eff = kwargs.get("Z_eff", CORONA_ION_CHARGE_Z)
        self.E_field = kwargs.get("E_field", 0.0)  # 外加电场 [V/m]
        self.include_stopping = kwargs.get("include_stopping", True)
        self.include_radiation = kwargs.get("include_radiation", False)
        self.include_collisional = kwargs.get("include_collisional", True)

    def nu_ei(self):
        """Spitzer 碰撞频率."""
        return spitzer_collisional_frequency(
            self.n_e, self.T_e_ev, self.Z_eff)

    def dedx_bethe(self, energy_ev):
        """Bethe stopping power [eV/m]."""
        if not self.include_stopping:
            return 0.0
        return bethe_stopping_power(energy_ev, self.n_e, self.Z_eff)

    def rad_loss_rate(self, energy_ev):
        """
        韧致辐射功率 (简化):
            P_rad = (2 e^2 c γ^2 β^2) / (3 m_e c^2) · (Z n_e e^2 / (4π ε_0 m_e c^2))
        单位: [eV/s]
        """
        if not self.include_radiation:
            return 0.0
        gamma = relativistic_gamma(energy_ev)
        beta2 = 1.0 - 1.0 / (gamma ** 2)
        # 经典辐射功率 (简化 Larmor 公式)
        r_e = ELECTRON_CHARGE ** 2 / (4.0 * math.pi
                                       * 8.854e-12 * ELECTRON_MASS
                                       * SPEED_OF_LIGHT ** 2)
        P_L = (2.0 / 3.0) * r_e * SPEED_OF_LIGHT * beta2 * gamma ** 2
        # 乘以散射率
        n_Z = self.n_e * self.Z_eff
        sigma_T = (8.0 / 3.0) * math.pi * r_e ** 2
        P_rad_W = P_L * n_Z * sigma_T
        return P_rad_W / ELECTRON_CHARGE  # 转换到 eV/s


# ============================================================
# ODE 右端函数 (来自 019_arneodo_ode/arneodo_deriv)
# ============================================================
def electron_ode_rhs(t, state, params):
    """
    快电子相空间 ODE 右端函数.

    状态向量 state = [x, p_x, E_kin]:
        x     : 位置 [m]
        p_x   : x 方向动量 [kg·m/s]
        E_kin : 动能 [eV]

    方程:
        dx/dt     = p_x / (γ m_e)
        dp_x/dt   = -e E_x - ν_ei · p_x - (dE/dx)_Bethe · sign(v_x)
        dE_kin/dt = -(dE/dx)_Bethe · |v_x| - P_rad

    参数:
        t     : 时间
        state : [x, p_x, E_kin]
        params: ElectronTrajectoryParams 实例
    返回:
        [dx/dt, dp_x/dt, dE_kin/dt]
    """
    x, p_x, E_kin = state

    # Lorentz 因子
    m_e_c2_eV = ELECTRON_MASS * SPEED_OF_LIGHT ** 2 / ELECTRON_CHARGE
    gamma = 1.0 + E_kin / m_e_c2_eV
    gamma = max(gamma, 1.0)

    # 速度
    v_x = p_x / (gamma * ELECTRON_MASS)

    # 碰撞阻尼
    nu_ei = params.nu_ei() if params.include_collisional else 0.0

    # Bethe stopping
    dedx = params.dedx_bethe(max(E_kin, 1.0e3))  # 保护低能

    # 外力 (电场)
    F_ext = -ELECTRON_CHARGE * params.E_field

    # 碰撞阻尼力
    F_coll = -nu_ei * p_x

    # Stopping force (方向与速度相反)
    sign_v = 1.0 if v_x >= 0 else -1.0
    F_stop = -dedx * ELECTRON_CHARGE * sign_v  # 转换到 SI

    # ODE
    dxdt = v_x
    dpdt = F_ext + F_coll + F_stop
    # 能量损失 = 功率
    dEdt = -(dedx * abs(v_x) + params.rad_loss_rate(max(E_kin, 1.0e3)))

    return [dxdt, dpdt, dEdt]


# ============================================================
# RK4 积分器 (来自 830_ode_rk4)
# ============================================================
def rk4_step(f, t, y, dt, params):
    """
    单步 RK4 (来自 830_ode_rk4/ode_rk4).

    参数:
        f     : 右端函数 f(t, y, params)
        t     : 当前时间
        y     : 当前状态
        dt    : 时间步长
        params: ODE 参数
    返回:
        y_new: 下一步状态
    """
    k1 = f(t, y, params)
    y2 = [y[i] + 0.5 * dt * k1[i] for i in range(len(y))]
    k2 = f(t + 0.5 * dt, y2, params)
    y3 = [y[i] + 0.5 * dt * k2[i] for i in range(len(y))]
    k3 = f(t + 0.5 * dt, y3, params)
    y4 = [y[i] + dt * k3[i] for i in range(len(y))]
    k4 = f(t + dt, y4, params)

    y_new = [y[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i]
                                    + 2.0 * k3[i] + k4[i])
             for i in range(len(y))]
    return y_new


# ============================================================
# 守恒量计算 (来自 018_arenstorf_ode/arenstorf_conserved)
# ============================================================
def electron_hamiltonian(state, params):
    """
    电子广义能量 (Hamiltonian, 无耗散时守恒):
        H = γ m_e c^2 + e φ(x)
    其中 γ = 1 + E_kin / (m_e c^2).

    在均匀电场下: φ(x) = -E_x · x

    参数:
        state : [x, p_x, E_kin]
        params: 参数
    返回:
        H [J]
    """
    x, p_x, E_kin = state
    m_e_c2_J = ELECTRON_MASS * SPEED_OF_LIGHT ** 2
    gamma = 1.0 + E_kin * ELECTRON_CHARGE / m_e_c2_J
    # 动能贡献
    E_kin_J = E_kin * ELECTRON_CHARGE
    # 势能贡献
    phi = -params.E_field * x
    H = E_kin_J + ELECTRON_CHARGE * phi
    return H


# ============================================================
# 完整轨迹积分
# ============================================================
def integrate_single_electron(E0_ev, x0, vx0, t_end, n_steps, params):
    """
    积分单个电子轨迹.

    参数:
        E0_ev  : 初始动能 [eV]
        x0     : 初始位置 [m]
        vx0    : 初始速度 [m/s]
        t_end  : 终止时间 [s]
        n_steps: 时间步数
        params : 轨迹参数
    返回:
        dict: 包含 't', 'x', 'p', 'E_kin', 'hamiltonian'
    """
    dt = t_end / max(n_steps, 1)

    # 初始状态
    gamma0 = relativistic_gamma(E0_ev)
    p0 = gamma0 * ELECTRON_MASS * vx0
    state = [x0, p0, E0_ev]

    t_vals = [0.0]
    x_vals = [x0]
    p_vals = [p0]
    E_vals = [E0_ev]
    H_vals = [electron_hamiltonian(state, params)]

    for n in range(n_steps):
        t = (n + 1) * dt
        state = rk4_step(electron_ode_rhs, t - dt, state, dt, params)
        # 保护非负能量
        state[2] = max(state[2], 0.0)

        t_vals.append(t)
        x_vals.append(state[0])
        p_vals.append(state[1])
        E_vals.append(state[2])
        H_vals.append(electron_hamiltonian(state, params))

    return {
        "t": t_vals,
        "x": x_vals,
        "p": p_vals,
        "E_kin": E_vals,
        "hamiltonian": H_vals,
        "E_final_ev": E_vals[-1],
        "x_final_m": x_vals[-1],
        "energy_loss_fraction": 1.0 - E_vals[-1] / max(E_vals[0], 1.0),
        "hamiltonian_drift": abs(H_vals[-1] - H_vals[0])
                             / max(abs(H_vals[0]), 1.0e-30),
    }


def integrate_beam_electrons(beam, t_end, n_steps, params):
    """
    积分整个电子束的轨迹.

    参数:
        beam   : FastElectronBeam 实例
        t_end  : 终止时间
        n_steps: 时间步数
        params : 轨迹参数
    返回:
        list[dict]: 每个电子的轨迹结果
    """
    results = []
    for particle in beam.particles:
        E0 = particle["energy_ev"]
        x0 = particle["x"]
        vx0 = particle["vz"]  # 主传播方向
        res = integrate_single_electron(E0, x0, vx0, t_end, n_steps, params)
        res["particle_id"] = particle["id"]
        res["weight"] = particle["weight_j"]
        results.append(res)
    return results


def print_trajectory_summary(results):
    """打印轨迹积分摘要."""
    print("\n" + "=" * 72)
    print("快电子轨迹 RK4 积分结果")
    print("=" * 72)
    if not results:
        print("  (无轨迹数据)")
        print("=" * 72)
        return

    n = len(results)
    avg_loss = sum(r["energy_loss_fraction"] for r in results) / n
    avg_drift = sum(r["hamiltonian_drift"] for r in results) / n
    avg_x_final = sum(r["x_final_m"] for r in results) / n
    avg_E_final = sum(r["E_final_ev"] for r in results) / n

    print("  电子数       : {}".format(n))
    print("  平均能量损失 : {:.4f} %".format(avg_loss * 100))
    print("  平均穿透距离 : {:.4e} m".format(avg_x_final))
    print("  平均最终能量 : {:.4e} eV".format(avg_E_final))
    print("  Hamiltonian 漂移: {:.4e} (相对)".format(avg_drift))

    # 展示前 3 条轨迹
    for i, r in enumerate(results[:3]):
        print("  电子 #{}: E0={:.2e} eV -> Ef={:.2e} eV, "
              "x_f={:.2e} m, ΔH/H={:.2e}".format(
                  r["particle_id"],
                  r["E_kin"][0], r["E_final_ev"],
                  r["x_final_m"], r["hamiltonian_drift"]))
    print("=" * 72)
