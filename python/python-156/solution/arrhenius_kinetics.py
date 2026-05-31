"""
arrhenius_kinetics.py
=====================
基于逻辑斯蒂（Logistic）方程框架的Arrhenius反应动力学模块。

核心算法源自 logistic_ode (Project 702)，并改造用于描述
燃烧反应中的化学动力学演化。

原始 Logistic 方程：
    dY/dt = r Y (1 - Y/K)

在燃烧科学中，这对应于反应物的消耗过程：
    dY_F/dt = -A exp(-E_a/RT) ρ Y_F Y_O

通过类比，定义"有效增长率"：
    r_eff = A exp(-E_a/RT) ρ Y_O

和"有效承载量"：
    K_eff = Y_F0 (初始燃料质量分数)

则燃料消耗方程可写为：
    dY_F/dt = -r_eff Y_F (1 - Y_F/K_eff)  (当 Y_O 充足时)

对于多步反应机理，引入进展变量 c：
    dc/dt = k_f (1 - c)^n_f - k_b c^n_b

其中 k_f, k_b 分别为正/逆反应速率常数：
    k_f = A_f T^{n_f} exp(-E_{a,f} / RT)
    k_b = A_b T^{n_b} exp(-E_{a,b} / RT)

化学平衡常数（van't Hoff 方程）：
    K_c = (p / RT)^{Σν} * exp(-ΔG° / RT)

其中 ΔG° = ΔH° - T ΔS° 为标准吉布斯自由能变。

本模块实现：
1. 一步总包 Arrhenius 反应速率计算；
2. 多步反应机理的进展变量演化；
3. 绝热火焰温度迭代计算。
"""

import numpy as np

# 甲烷-空气燃烧的简化动力学参数
ARRHENIUS_PREEXP = 2.0e11       # m³/(kg·s)
ACTIVATION_ENERGY = 1.26e8      # J/mol
TEMPERATURE_EXPONENT = 0.0      # 温度指数

# 热力学参数
CP_MIX = 1200.0                 # J/(kg·K)
HEAT_RELEASE_CH4 = 5.0e7        # J/kg


def arrhenius_rate_constant(T, A=ARRHENIUS_PREEXP, Ea=ACTIVATION_ENERGY, n=0.0):
    """
    计算 Arrhenius 速率常数 k(T)。

    公式：
        k(T) = A T^n exp(-E_a / (R_u T))

    Parameters
    ----------
    T : float or ndarray
        温度，单位 K，必须为正数。
    A : float
        指前因子。
    Ea : float
        活化能，单位 J/mol。
    n : float
        温度指数。

    Returns
    -------
    k : float or ndarray
        速率常数。
    """
    R_u = 8.314462618
    T = np.maximum(T, 100.0)

    exponent = -Ea / (R_u * T)
    # 防止指数溢出
    exponent = np.clip(exponent, -700.0, 700.0)

    k = A * (T ** n) * np.exp(exponent)
    return k


def reaction_progress_ode(t, c, T, A_f=None, Ea_f=None, A_b=None, Ea_b=None):
    """
    进展变量 c 的 ODE 右端函数（类似 logistic_deriv）。

    dc/dt = k_f (1 - c)^{n_f} - k_b c^{n_b}

    Parameters
    ----------
    t : float
        时间（伪时间，用于反应进度）。
    c : float
        进展变量，范围 [0, 1]。
    T : float
        温度。
    A_f, Ea_f : float
        正反应参数。
    A_b, Ea_b : float
        逆反应参数。

    Returns
    -------
    dcdt : float
        进展变量变化率。
    """
    c = np.clip(c, 0.0, 1.0)

    if A_f is None:
        A_f = ARRHENIUS_PREEXP
    if Ea_f is None:
        Ea_f = ACTIVATION_ENERGY
    if A_b is None:
        A_b = ARRHENIUS_PREEXP * 0.01
    if Ea_b is None:
        Ea_b = ACTIVATION_ENERGY * 0.5

    k_f = arrhenius_rate_constant(T, A_f, Ea_f)
    k_b = arrhenius_rate_constant(T, A_b, Ea_b)

    n_f = 1.0
    n_b = 1.0

    dcdt = k_f * ((1.0 - c) ** n_f) - k_b * (c ** n_b)
    return dcdt


def integrate_progress_variable(T, dt=1.0e-6, n_steps=10000,
                                A_f=None, Ea_f=None, A_b=None, Ea_b=None):
    """
    使用显式欧拉法积分进展变量方程。

    Parameters
    ----------
    T : float
        温度。
    dt : float
        时间步长。
    n_steps : int
        积分步数。

    Returns
    -------
    c_history : ndarray
        进展变量演化历史。
    t_history : ndarray
        时间历史。
    """
    c = 0.0
    c_history = np.zeros(n_steps + 1)
    t_history = np.zeros(n_steps + 1)

    for i in range(n_steps):
        c_history[i] = c
        t_history[i] = i * dt

        dcdt = reaction_progress_ode(t_history[i], c, T, A_f, Ea_f, A_b, Ea_b)
        c_new = c + dt * dcdt
        c = np.clip(c_new, 0.0, 1.0)

    c_history[-1] = c
    t_history[-1] = n_steps * dt

    return c_history, t_history


def adiabatic_flame_temperature(Y_F0, Y_O0, T0, Q=HEAT_RELEASE_CH4, cp=CP_MIX):
    """
    计算绝热火焰温度。

    公式：
        T_ad = T0 + Q * min(Y_F0, Y_O0 / s) / cp

    其中 s 为化学计量质量比。

    Parameters
    ----------
    Y_F0 : float
        初始燃料质量分数。
    Y_O0 : float
        初始氧化剂质量分数。
    T0 : float
        初始温度。
    Q : float
        燃烧热。
    cp : float
        比热容。

    Returns
    -------
    T_ad : float
        绝热火焰温度。
    phi : float
        当量比。
    """
    s = 17.16  # CH4-air 化学计量比

    Y_F0 = np.clip(Y_F0, 0.0, 1.0)
    Y_O0 = np.clip(Y_O0, 0.0, 1.0)

    phi = (Y_F0 / Y_O0) / (1.0 / s) if Y_O0 > 1.0e-12 else 0.0

    Y_F_burnt = max(0.0, Y_F0 - min(Y_F0, Y_O0 / s))
    heat_released = Q * (Y_F0 - Y_F_burnt)

    T_ad = T0 + heat_released / cp
    T_ad = min(T_ad, 5000.0)  # 物理上界

    return T_ad, phi
