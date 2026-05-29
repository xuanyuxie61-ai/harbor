"""
epidemic_dynamics.py
流行病-信息耦合传播动力学模块

基于以下种子项目融合:
- 1023_rigid_body_ode: 刚体旋转欧拉方程
- 707_mackey_glass_dde: Mackey-Glass延迟微分方程
"""

import numpy as np
from typing import Tuple, List, Optional, Callable


def seaihr_ode_rhs(t: float, y: np.ndarray, params: dict) -> np.ndarray:
    """
    SEAIHR (Susceptible-Exposed-Asymptomatic-Infectious-Hospitalized-Recovered)
    流行病模型右端项。

    状态变量:
        y = [S, E, A, I, H, R, D]^T

    动力学方程:
        dS/dt = -beta * S * (I + eta_A * A) / N + omega * R
        dE/dt = beta * S * (I + eta_A * A) / N - sigma * E
        dA/dt = (1 - p_sym) * sigma * E - gamma_A * A
        dI/dt = p_sym * sigma * E - (gamma_I + alpha_H) * I
        dH/dt = alpha_H * I - (gamma_H + mu) * H
        dR/dt = gamma_A * A + gamma_I * I + gamma_H * H - omega * R
        dD/dt = mu * H

    参数:
        N: 总人口
        beta: 有效接触率
        eta_A: 无症状相对传染性
        sigma: 潜伏期倒数
        p_sym: 出现症状概率
        gamma_A: 无症状恢复率
        gamma_I: 有症状恢复率
        alpha_H: 住院率
        gamma_H: 住院恢复率
        mu: 住院死亡率
        omega: 免疫丧失率
    """
    # TODO Hole_1: 实现SEAIHR流行病模型右端项
    # 需要根据params中的参数和状态向量y计算dSdt, dEdt, dAdt, dIdt, dHdt, dRdt, dDdt
    # 并考虑边界保护与非负约束、总人口守恒性检查
    raise NotImplementedError("Hole_1: seaihr_ode_rhs 尚未实现")


def coupled_info_epidemic_rhs(t: float, y: np.ndarray,
                              history_func: Callable,
                              params: dict) -> np.ndarray:
    """
    耦合信息-流行病延迟动力学右端项。

    引入信息传播反馈 I_info(t)，基于历史感染数据，
    通过延迟微分方程影响接触率 beta(t):

        beta(t) = beta_0 * exp(-k_beta * I_info(t - tau))

    信息动力学 (Mackey-Glass型):
        dI_info/dt = (beta_info * I_past) / (1 + I_past^n_info) - gamma_info * I_info

    其中 I_past = I(t - tau_info) + A(t - tau_info)。

    组合状态:
        y = [S, E, A, I, H, R, D, I_info]^T
    """
    n_epi = 7
    y_epi = y[:n_epi]
    I_info = max(y[n_epi], 0.0)

    # 延迟历史
    tau = params.get('tau', 5.0)
    t_delayed = max(0.0, t - tau)
    y_delayed = history_func(t_delayed)

    I_past = max(y_delayed[3] + y_delayed[2], 0.0)  # I + A 延迟值

    # 信息影响接触率
    beta_0 = params['beta_0']
    k_beta = params.get('k_beta', 0.5)
    beta_eff = beta_0 * np.exp(-k_beta * I_info)

    # 更新参数
    params_local = params.copy()
    params_local['beta'] = max(beta_eff, 0.0)

    # 流行病动力学
    dydt_epi = seaihr_ode_rhs(t, y_epi, params_local)

    # 信息动力学 (Mackey-Glass型)
    beta_info = params.get('beta_info', 2.0)
    n_info = params.get('n_info', 9.65)
    gamma_info = params.get('gamma_info', 1.0)

    # 保护分母
    denom = 1.0 + I_past**n_info
    if denom < 1e-10:
        denom = 1e-10

    dI_info_dt = beta_info * I_past / denom - gamma_info * I_info

    return np.concatenate([dydt_epi, [dI_info_dt]])


def rk4_integrate(rhs_func: Callable,
                  y0: np.ndarray,
                  t_span: Tuple[float, float],
                  dt: float,
                  params: dict,
                  history_func: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    4阶Runge-Kutta积分器。

    经典RK4格式:
        k1 = f(t_n, y_n)
        k2 = f(t_n + dt/2, y_n + dt/2 * k1)
        k3 = f(t_n + dt/2, y_n + dt/2 * k2)
        k4 = f(t_n + dt, y_n + dt * k3)
        y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    局部截断误差: O(dt^5)
    全局误差: O(dt^4)
    """
    t0, tf = t_span
    n_steps = max(1, int(np.ceil((tf - t0) / dt)))
    dt_actual = (tf - t0) / n_steps

    t_history = np.linspace(t0, tf, n_steps + 1)
    y_history = np.zeros((n_steps + 1, len(y0)), dtype=np.float64)
    y_history[0, :] = y0

    y = y0.copy()

    for n in range(n_steps):
        t_n = t_history[n]

        if history_func is not None:
            def wrapped_rhs(t, y):
                return rhs_func(t, y, history_func, params)
        else:
            def wrapped_rhs(t, y):
                return rhs_func(t, y, params)

        k1 = wrapped_rhs(t_n, y)
        k2 = wrapped_rhs(t_n + 0.5 * dt_actual, y + 0.5 * dt_actual * k1)
        k3 = wrapped_rhs(t_n + 0.5 * dt_actual, y + 0.5 * dt_actual * k2)
        k4 = wrapped_rhs(t_n + dt_actual, y + dt_actual * k3)

        y = y + (dt_actual / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # 非负约束
        y = np.maximum(y, 0.0)

        y_history[n + 1, :] = y

    return t_history, y_history


def dde_rk4_integrate(rhs_func: Callable,
                      y0: np.ndarray,
                      t_span: Tuple[float, float],
                      dt: float,
                      params: dict,
                      history_func: Callable) -> Tuple[np.ndarray, np.ndarray]:
    """
    带延迟的4阶Runge-Kutta积分器。

    使用历史插值来处理延迟项。
    """
    t0, tf = t_span
    n_steps = max(1, int(np.ceil((tf - t0) / dt)))
    dt_actual = (tf - t0) / n_steps

    t_history = np.linspace(t0, tf, n_steps + 1)
    y_history = np.zeros((n_steps + 1, len(y0)), dtype=np.float64)
    y_history[0, :] = y0

    y = y0.copy()

    for n in range(n_steps):
        t_n = t_history[n]

        # 当前历史插值函数
        def current_history(t_query):
            if t_query <= t0:
                return history_func(t_query)
            # 线性插值
            idx = int((t_query - t0) / dt_actual)
            idx = min(idx, n)
            frac = (t_query - t0) / dt_actual - idx
            frac = max(0.0, min(1.0, frac))
            if idx + 1 <= n:
                return (1 - frac) * y_history[idx, :] + frac * y_history[idx + 1, :]
            return y_history[idx, :]

        def wrapped_rhs(t, y_state):
            return rhs_func(t, y_state, current_history, params)

        k1 = wrapped_rhs(t_n, y)
        k2 = wrapped_rhs(t_n + 0.5 * dt_actual, y + 0.5 * dt_actual * k1)
        k3 = wrapped_rhs(t_n + 0.5 * dt_actual, y + 0.5 * dt_actual * k2)
        k4 = wrapped_rhs(t_n + dt_actual, y + dt_actual * k3)

        y = y + (dt_actual / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        y = np.maximum(y, 0.0)
        y_history[n + 1, :] = y

    return t_history, y_history


def compute_reproduction_number(params: dict) -> float:
    """
    计算基本再生数 R_0 (SEAIHR模型)。

    公式推导:
        R_0 = R_0^{sym} + R_0^{asym}

    其中:
        R_0^{sym} = beta * p_sym / (gamma_I + alpha_H)
        R_0^{asym} = beta * eta_A * (1 - p_sym) / gamma_A
    """
    # TODO Hole_2: 实现基本再生数R_0的计算
    # 需要根据SEAIHR模型结构推导R_0解析公式
    # R_0 = R_0^{sym} + R_0^{asym}
    raise NotImplementedError("Hole_2: compute_reproduction_number 尚未实现")
