# -*- coding: utf-8 -*-
"""
ode_integrator.py
温度演化与结构动力学ODE积分器

本模块实现中子星crust层冷却与相变过程的ODE积分.
融入算法:
- unstable_ode (1374): 不稳定系统的精确解与数值处理
- tough_ode (1283): 刚性ODE的导数计算与隐式方法

核心物理公式:
1. 中子星冷却方程 (modified Urca过程):
   C_V dT/dt = -epsilon_nu + epsilon_crust
   
   其中:
   C_V = (pi^2/2) N(0) k_B^2 T  (电子比热, 简并费米气体)
   epsilon_nu = (4.13e27) * (m*/m)^3 * (x_p)^{1/3} * T_8^8  (MeV/fm^3/s)
   epsilon_crust = 核反应加热率
   
2. 相变动力学 (Ginzburg-Landau):
   dpsi/dt = -gamma * deltaF/dpsi + xi(t)
   
   psi: 序参量 (密度差)
   F = integral [a(T) psi^2 + b psi^4 + c (nabla psi)^2] dV
   
3. 不稳定ODE模型 (来自1374_unstable_ode):
   y' = A(mu) * y
   A = [[mu, 1/mu], [-1/mu, mu]]
   特征值: lambda = mu +/- i/mu
   精确解: y1(t) = exp(mu*t)(cos(t/mu) - mu^2 sin(t/mu))
   
4. 刚性ODE (来自1283_tough_ode):
   y1' = 2*t*y2^{0.2}*y4
   y2' = 10*t*exp(5*(y2-1))*y4
   y3' = 2*t*y4
   y4' = -2*t*log(y1)
"""

import numpy as np
from scipy.integrate import solve_ivp

# 物理常数
K_B = 8.617333262e-11  # MeV/K
HBARC = 197.3269804    # MeV·fm


def unstable_exact(t, mu):
    """
    不稳定ODE的精确解 (来自1374_unstable_ode).
    
    系统:
    y1' = mu*y1 + (1/mu)*y2
    y2' = -(1/mu)*y1 + mu*y2
    
    初始条件: y(0) = [1, 0]
    
    精确解:
    y1(t) = exp(mu*t)*(cos(t/mu) - mu^2*sin(t/mu))
    y2(t) = exp(mu*t)*(mu*sin(t/mu) + cos(t/mu)) / mu
    
    输入:
        t: 时间
        mu: 不稳定参数
    输出:
        y1, y2: 精确解
    """
    if mu == 0.0:
        raise ValueError("mu不能为零")
    exp_term = np.exp(mu * t)
    cos_term = np.cos(t / mu)
    sin_term = np.sin(t / mu)
    y1 = exp_term * (cos_term - mu**2 * sin_term)
    y2 = exp_term * (sin_term / mu + mu * cos_term)
    return y1, y2


def unstable_deriv(t, y, mu):
    """
    不稳定ODE的导数 (来自1374_unstable_ode).
    
    输入:
        t: 时间
        y: [y1, y2]
        mu: 参数
    输出:
        dydt: [dy1/dt, dy2/dt]
    """
    u, v = y
    if mu == 0.0:
        return np.array([0.0, 0.0])
    dudt = mu * u + (1.0 / mu) * v
    dvdt = -(1.0 / mu) * u + mu * v
    return np.array([dudt, dvdt])


def tough_deriv(t, y):
    """
    刚性ODE的导数 (来自1283_tough_ode).
    
    输入:
        t: 时间
        y: [y1, y2, y3, y4]
    输出:
        dydt: [dy1/dt, dy2/dt, dy3/dt, dy4/dt]
    """
    y1, y2, y3, y4 = y
    # 边界处理
    if y1 <= 0.0:
        y1 = 1e-15
    if y2 <= 0.0:
        y2 = 1e-15

    dy1dt = 2.0 * t * (y2**0.2) * y4
    dy2dt = 10.0 * t * np.exp(5.0 * (y2 - 1.0)) * y4
    dy3dt = 2.0 * t * y4
    dy4dt = -2.0 * t * np.log(y1)
    return np.array([dy1dt, dy2dt, dy3dt, dy4dt])


def neutrino_luminosity(temperature, rho, proton_fraction, m_star=1.0):
    """
    中微子发光度 (modified Urca过程).
    
    公式 (Friman & Maxwell 1979):
    epsilon_nu = 4.13e27 * (m*/m)^3 * (x_p)^{1/3} * (T/1e8 K)^8  (erg/cm^3/s)
    
    转换为 MeV/fm^3/s:
    1 erg = 6.2415e5 MeV
    1 cm^3 = 1e39 fm^3
    
    输入:
        temperature: 温度 (MeV)
        rho: 核子数密度 (fm^{-3})
        proton_fraction: 质子分数
        m_star: 有效质量比
    输出:
        epsilon_nu: 中微子能量损失率 (MeV/fm^3/s)
    """
    if temperature <= 0.0 or rho <= 0.0:
        return 0.0

    T_8 = temperature / (K_B * 1e8)  # 以1e8 K为单位
    x_p = proton_fraction

    # 系数 (简化)
    coeff = 4.13e27  # erg/cm^3/s
    # 单位转换
    coeff_MeV = coeff * 6.2415e5 / 1e39  # MeV/fm^3/s

    epsilon = coeff_MeV * (m_star**3) * (x_p**(1.0/3.0)) * (T_8**8)
    return epsilon


def heat_capacity_degenerate(rho, proton_fraction, temperature):
    """
    简并费米气体的比热.
    
    公式:
    C_V = (pi^2/2) * N(0) * k_B^2 * T
    N(0) = 3 * rho_p / (2 * epsilon_F)  (质子态密度)
    epsilon_F = k_F^2 / (2*m_p)
    
    输入:
        rho: 核子密度
        proton_fraction: 质子分数
        temperature: 温度
    输出:
        C_V: 体积比热 (MeV/fm^3/K)
    """
    if temperature <= 0.0 or rho <= 0.0:
        return 0.0

    rho_p = rho * proton_fraction
    m_p = 938.272  # MeV
    k_fp = (3.0 * np.pi**2 * rho_p)**(1.0/3.0)
    epsilon_F = k_fp**2 / (2.0 * m_p)

    if epsilon_F <= 0.0:
        return 0.0

    N0 = 3.0 * rho_p / (2.0 * epsilon_F)
    C_V = (np.pi**2 / 2.0) * N0 * K_B**2 * temperature
    return max(0.0, C_V)


def crust_cooling_ode(t, y, rho, proton_fraction, m_star=1.0,
                      heating_rate=0.0):
    """
    中子星crust冷却ODE.
    
    状态变量:
    y[0] = T: 温度 (MeV)
    y[1] = Q_heat: 累积加热能量 (MeV/fm^3)
    
    dT/dt = (-epsilon_nu + heating_rate) / C_V
    dQ_heat/dt = heating_rate
    
    输入:
        t: 时间 (s)
        y: [T, Q_heat]
        rho: 密度
        proton_fraction: 质子分数
        m_star: 有效质量比
        heating_rate: 外部加热率 (MeV/fm^3/s)
    输出:
        dydt: [dT/dt, dQ_heat/dt]
    """
    T = y[0]
    if T <= 0.0:
        return np.array([0.0, 0.0])

    C_V = heat_capacity_degenerate(rho, proton_fraction, T)
    if C_V <= 1e-30:
        return np.array([0.0, heating_rate])

    eps_nu = neutrino_luminosity(T, rho, proton_fraction, m_star)

    dTdt = (-eps_nu + heating_rate) / C_V
    dQdt = heating_rate

    # 边界: 温度不能为负
    if T + dTdt < 0.0:
        dTdt = -T * 0.1

    return np.array([dTdt, dQdt])


def solve_crust_cooling(t_span, T0, rho, proton_fraction,
                        m_star=1.0, heating_rate=0.0, method='RK45'):
    """
    求解中子星crust冷却方程.
    
    输入:
        t_span: [t_start, t_end] (秒)
        T0: 初始温度 (MeV)
        rho: 密度
        proton_fraction: 质子分数
        m_star: 有效质量比
        heating_rate: 加热率
        method: ODE求解方法
    输出:
        sol: scipy.integrate.solve_ivp结果
    """
    y0 = [T0, 0.0]

    def deriv(t, y):
        return crust_cooling_ode(t, y, rho, proton_fraction, m_star, heating_rate)

    sol = solve_ivp(deriv, t_span, y0, method=method, max_step=t_span[1]/100.0,
                    dense_output=True, rtol=1e-6, atol=1e-9)
    return sol


def phase_transition_kinetics(t, psi, T, T_c, gamma, a0, b, c):
    """
    Ginzburg-Landau相变动力学.
    
    序参量演化:
    dpsi/dt = -gamma * dF/dpsi
    
    自由能密度:
    f = a(T) * psi^2 + b * psi^4 + c * (nabla psi)^2
    a(T) = a0 * (T - T_c) / T_c
    
    输入:
        t: 时间
        psi: 序参量 (密度调制幅度)
        T: 温度
        T_c: 临界温度
        gamma: 弛豫系数
        a0, b, c: GL系数
    输出:
        dpsi/dt
    """
    if T_c <= 0.0:
        return 0.0
    a_T = a0 * (T - T_c) / T_c
    dF_dpsi = 2.0 * a_T * psi + 4.0 * b * psi**3
    dpsi_dt = -gamma * dF_dpsi
    return dpsi_dt


def solve_unstable_system(t_span, y0, mu, n_points=200):
    """
    求解不稳定ODE系统 (来自1374_unstable_ode).
    
    用于模拟中子星crust层中微扰动的指数增长.
    
    输入:
        t_span: [t0, t1]
        y0: 初始条件
        mu: 不稳定参数
        n_points: 输出点数
    输出:
        t, y_exact, y_numerical
    """
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    # 精确解
    y1_exact, y2_exact = unstable_exact(t_eval, mu)

    # 数值解
    sol = solve_ivp(lambda t, y: unstable_deriv(t, y, mu),
                    t_span, y0, t_eval=t_eval, method='RK45',
                    rtol=1e-9, atol=1e-12)

    return t_eval, np.array([y1_exact, y2_exact]), sol.y


def solve_tough_system(t_span, y0, method='RK45'):
    """
    求解刚性ODE系统 (来自1283_tough_ode).
    
    用于测试中子星crust层复杂反应网络的数值稳定性.
    
    输入:
        t_span: [t0, t1]
        y0: 初始条件
        method: 求解方法
    输出:
        sol: scipy.integrate.solve_ivp结果
    """
    sol = solve_ivp(tough_deriv, t_span, y0, method=method,
                    dense_output=True, rtol=1e-8, atol=1e-10)
    return sol


if __name__ == '__main__':
    # 自测试
    t = np.linspace(0, 1, 10)
    y1e, y2e = unstable_exact(t, mu=0.5)
    print(f"unstable exact: y1[0]={y1e[0]:.4f}, y1[-1]={y1e[-1]:.4f}")

    sol = solve_crust_cooling([0, 1e5], T0=1.0, rho=0.1, proton_fraction=0.3)
    print(f"crust cooling: T_final={sol.y[0,-1]:.6f} MeV")
