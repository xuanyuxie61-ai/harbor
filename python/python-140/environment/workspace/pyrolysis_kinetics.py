"""
pyrolysis_kinetics.py
生物质热解反应动力学模块
实现复杂多组分热解反应网络的动力学求解，包含 Arrhenius 反应速率、
多步反应机理、以及刚性 ODE 系统的多种数值积分方法。
原项目映射:
  - 829_ode_midpoint_system (中点法求解 ODE 系统)
  - 1036_rk4 (经典四阶 Runge-Kutta 方法)
  - 317_doughnut_ode (环面几何上的非线性 ODE 系统)
"""

import numpy as np
from utils import safe_exp, check_bounds, finite_diff_jacobian


class BiomassPyrolysisKinetics:
    """
    生物质热解多组分反应动力学模型。
    
    采用分布式活化能模型（DAEM）与三组分并行反应模型（Broido-Shafizadeh）的耦合：
    
    反应网络:
        Biomass (B) --k1--> Active Cellulose (A)
        A --k2--> Volatiles (V) + Char (C)
        A --k3--> Tar (T) + Gas (G)
        Hemicellulose (H) --k4--> V + C
        Lignin (L) --k5--> V + C
    
    各反应速率遵循 Arrhenius 定律:
        k_i = A_i * exp(-E_{a,i} / (R * T))
    
    其中:
        A_i: 指前因子 [s^{-1}]
        E_{a,i}: 活化能 [J/mol]
        R: 理想气体常数 8.314 [J/(mol·K)]
        T: 温度 [K]
    """

    def __init__(self, R_gas=8.314):
        self.R = R_gas
        # 典型生物质热解参数 (文献值)
        # 纤维素
        self.A1 = 2.8e19    # s^{-1}
        self.Ea1 = 242.4e3  # J/mol
        self.A2 = 3.27e14
        self.Ea2 = 196.5e3
        self.A3 = 1.30e10
        self.Ea3 = 150.5e3
        # 半纤维素
        self.A4 = 2.1e16
        self.Ea4 = 186.7e3
        # 木质素
        self.A5 = 1.05e15
        self.Ea5 = 179.8e3
        # 初始质量分数
        self.y0 = np.array([0.40, 0.30, 0.30, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        # [B, H, L, A, V, C, T+G]

    def reaction_rates(self, T):
        """
        计算各反应速率常数 k_i。
        公式: k_i = A_i * exp(-E_{a,i} / (R * T))
        """
        # TODO: 实现 Arrhenius 反应速率计算
        raise NotImplementedError("Hole 1: 请补全 reaction_rates 方法")

    def deriv(self, t, y, T_func):
        """
        计算热解动力学 ODE 的右端项 dy/dt。
        
        状态向量 y = [y_B, y_H, y_L, y_A, y_V, y_C, y_TG]
        
        dy_B/dt = -k1(T) * y_B
        dy_H/dt = -k4(T) * y_H
        dy_L/dt = -k5(T) * y_L
        dy_A/dt =  k1(T) * y_B - (k2(T) + k3(T)) * y_A
        dy_V/dt =  k2(T) * y_A * alpha_V2 + k4(T) * y_H * alpha_V4 + k5(T) * y_L * alpha_V5
        dy_C/dt =  k2(T) * y_A * alpha_C2 + k4(T) * y_H * alpha_C4 + k5(T) * y_L * alpha_C5
        dy_TG/dt = k3(T) * y_A
        
        alpha: 产物分配系数
        """
        y = np.asarray(y, dtype=np.float64)
        # 静默截断微小负值，避免大量数值噪声警告
        y = np.clip(y, 0.0, 1.0)
        # 归一化保证质量守恒
        y_sum = np.sum(y[:3]) + y[3] + y[4] + y[5] + y[6]
        if y_sum > 1e-10 and abs(y_sum - 1.0) > 1e-6:
            y = y / y_sum

        T = T_func(t)
        k = self.reaction_rates(T)
        k1, k2, k3, k4, k5 = k[0], k[1], k[2], k[3], k[4]

        # 产物分配系数
        alpha_V2, alpha_C2 = 0.75, 0.25
        alpha_V4, alpha_C4 = 0.65, 0.35
        alpha_V5, alpha_C5 = 0.55, 0.45

        dydt = np.zeros(7, dtype=np.float64)
        dydt[0] = -k1 * y[0]                          # B
        dydt[1] = -k4 * y[1]                          # H
        dydt[2] = -k5 * y[2]                          # L
        dydt[3] =  k1 * y[0] - (k2 + k3) * y[3]       # A
        dydt[4] =  k2 * y[3] * alpha_V2 + k4 * y[1] * alpha_V4 + k5 * y[2] * alpha_V5  # V
        dydt[5] =  k2 * y[3] * alpha_C2 + k4 * y[1] * alpha_C4 + k5 * y[2] * alpha_C5  # C
        dydt[6] =  k3 * y[3]                          # TG
        return dydt

    def solve_rk4(self, tspan, y0, n_steps, T_func):
        """
        经典四阶 Runge-Kutta 方法求解热解动力学 ODE。
        映射自 rk4.m。
        """
        y0 = np.asarray(y0, dtype=np.float64)
        t0, tf = tspan
        dt = (tf - t0) / n_steps
        m = len(y0)
        t = np.zeros(n_steps + 1, dtype=np.float64)
        y = np.zeros((n_steps + 1, m), dtype=np.float64)
        t[0] = t0
        y[0, :] = y0

        for i in range(n_steps):
            ti = t[i]
            yi = y[i, :]
            f1 = self.deriv(ti, yi, T_func)
            f2 = self.deriv(ti + dt / 2.0, yi + dt * f1 / 2.0, T_func)
            f3 = self.deriv(ti + dt / 2.0, yi + dt * f2 / 2.0, T_func)
            f4 = self.deriv(ti + dt, yi + dt * f3, T_func)
            t[i + 1] = ti + dt
            y[i + 1, :] = yi + dt * (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0
            # 非负截断
            y[i + 1, :] = np.maximum(y[i + 1, :], 0.0)
            # 质量守恒归一化
            total = np.sum(y[i + 1, :])
            if total > 1e-10:
                y[i + 1, :] /= total
        return t, y

    def solve_midpoint(self, tspan, y0, n_steps, T_func, theta=0.5, max_iter=10):
        """
        隐式中点法（Midpoint Method）求解刚性 ODE 系统。
        映射自 ode_midpoint_system.m。
        
        中点格式:
            Y_{n+θ} = Y_n + θ * h * f(t_{n+θ}, Y_{n+θ})   (隐式迭代求解)
            Y_{n+1} = (1/θ) * Y_{n+θ} + (1 - 1/θ) * Y_n
        """
        y0 = np.asarray(y0, dtype=np.float64)
        t0, tf = tspan
        dt = (tf - t0) / n_steps
        m = len(y0)
        t = np.linspace(t0, tf, n_steps + 1)
        y = np.zeros((n_steps + 1, m), dtype=np.float64)
        y[0, :] = y0

        for i in range(n_steps):
            tm = t[i] + theta * dt
            ym = y[i, :].copy()
            # 不动点迭代求解隐式中点方程
            for _ in range(max_iter):
                fm = self.deriv(tm, ym, T_func)
                ym_new = y[i, :] + theta * dt * fm
                if np.linalg.norm(ym_new - ym) < 1e-12:
                    ym = ym_new
                    break
                ym = ym_new
            y[i + 1, :] = (1.0 / theta) * ym + (1.0 - 1.0 / theta) * y[i, :]
            y[i + 1, :] = np.maximum(y[i + 1, :], 0.0)
            total = np.sum(y[i + 1, :])
            if total > 1e-10:
                y[i + 1, :] /= total
        return t, y


def doughnut_pyrolysis_flow(t, y, m_param=1.0, n_param=0.5):
    """
    环面反应器内的热解流体动力学模型。
    映射自 doughnut_ode.m 的环面非线性动力学。
    
    在环面坐标系下，反应物流动的非线性耦合方程:
        dy1/dt = -m * y2 + n * y1 * y3   (方位角流动)
        dy2/dt =  m * y1 + n * y2 * y3   (极向流动)
        dy3/dt =  0.5 * n * (1 - y1^2 - y2^2 + y3^2)  (轴向压力/浓度)
    
    物理意义: y1, y2 为反应物流量在环面大/小半径方向的投影，
             y3 为反应进度或局部温度扰动。
    """
    y = np.asarray(y, dtype=np.float64)
    dy1dt = -m_param * y[1] + n_param * y[0] * y[2]
    dy2dt =  m_param * y[0] + n_param * y[1] * y[2]
    dy3dt =  0.5 * n_param * (1.0 - y[0] ** 2 - y[1] ** 2 + y[2] ** 2)
    return np.array([dy1dt, dy2dt, dy3dt], dtype=np.float64)


def solve_doughnut_flow_rk4(tspan, y0, n_steps, m_param=1.0, n_param=0.5):
    """
    使用 RK4 求解环面反应器流动方程。
    """
    y0 = np.asarray(y0, dtype=np.float64)
    t0, tf = tspan
    dt = (tf - t0) / n_steps
    m = len(y0)
    t = np.zeros(n_steps + 1, dtype=np.float64)
    y = np.zeros((n_steps + 1, m), dtype=np.float64)
    t[0] = t0
    y[0, :] = y0

    for i in range(n_steps):
        f1 = doughnut_pyrolysis_flow(t[i], y[i, :], m_param, n_param)
        f2 = doughnut_pyrolysis_flow(t[i] + dt / 2.0, y[i, :] + dt * f1 / 2.0, m_param, n_param)
        f3 = doughnut_pyrolysis_flow(t[i] + dt / 2.0, y[i, :] + dt * f2 / 2.0, m_param, n_param)
        f4 = doughnut_pyrolysis_flow(t[i] + dt, y[i, :] + dt * f3, m_param, n_param)
        t[i + 1] = t[i] + dt
        y[i + 1, :] = y[i, :] + dt * (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0
    return t, y
