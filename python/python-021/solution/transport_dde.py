"""
transport_dde.py
等离子体能量/粒子输运的延迟微分方程模型。

核心物理模型：
  托卡马克中的能量约束表现出显著的滞后反馈特征：
  加热功率扰动不会立即体现在温度响应上，而是通过湍流输运、
  磁流体波传播等过程延迟 τ_E / 3 ~ 0.1-0.5 s。

  借鉴 Mackey-Glass 延迟微分方程的数学结构：

      dx(t)/dt = β · x(t - τ) / (1 + x(t - τ)^n) - γ · x(t)

  将其物理化为等离子体能量密度 W(t) [J/m³] 的演化方程：

      dW(t)/dt = P_heat · β · W(t - τ) / (W_0^n + W(t - τ)^n)
                 - γ · W(t) / τ_E
                 - P_brem(t) - P_sync(t)

  其中：
    - W(t)       : 等离子体热能密度 [J/m³]
    - τ          : 能量输运延迟 [s]
    - β, γ, n    : Mackey-Glass 型非线性反馈参数
    - P_heat     : 外部加热功率密度 [W/m³]
    - τ_E        : 能量约束时间 [s]
    - P_brem     : 轫致辐射损失 [W/m³]
    - P_sync     : 同步辐射损失 [W/m³]

  非线性项 W(t-τ)^n / (W_0^n + W(t-τ)^n) 模拟了输运垒（barrier）
  的阈值行为：当能量密度超过阈值时，输运通道被部分抑制，
  导致温度剖面出现内部输运垒 (ITB)。

数值方法：
  采用四阶 Runge-Kutta 方法结合线性插值处理延迟项。
  历史数组存储 t ∈ [t0 - τ, t0] 期间的初值，
  通过线性插值获取 t - τ 时刻的延迟状态。
"""

import numpy as np
from parameters import get_transport_params


def interpolate_history(t_query, t_hist, y_hist):
    """
    线性插值获取历史值。

    参数
    ------
    t_query : float
        查询时间点。
    t_hist : ndarray
        历史时间数组（单调递增）。
    y_hist : ndarray
        历史状态数组。

    返回
    ------
    y_query : float
        插值结果。若 t_query 超出范围，返回最近边界值。
    """
    if t_query <= t_hist[0]:
        return float(y_hist[0])
    if t_query >= t_hist[-1]:
        return float(y_hist[-1])
    # 二分查找
    idx = np.searchsorted(t_hist, t_query)
    if idx == 0:
        return float(y_hist[0])
    t1, t2 = t_hist[idx - 1], t_hist[idx]
    y1, y2 = y_hist[idx - 1], y_hist[idx]
    if abs(t2 - t1) < 1e-15:
        return float(y1)
    return float(y1 + (y2 - y1) * (t_query - t1) / (t2 - t1))


def transport_dde_rhs(t, W, W_delayed, gamma, beta, n, W0,
                       P_heat, tau_E, P_loss_coeff):
    """
    延迟输运方程右端项。

    公式
    ----
        dW/dt = P_heat · β · W_τ^n / (W0^n + W_τ^n) - γ W / τ_E - C_loss W

    参数
    ------
    t : float
    W : float
        当前能量密度。
    W_delayed : float
        延迟能量密度 W(t - τ)。
    gamma, beta, n : float
        Mackey-Glass 参数。
    W0 : float
        阈值能量密度 [J/m³]。
    P_heat : float
        加热功率密度 [W/m³]。
    tau_E : float
        能量约束时间 [s]。
    P_loss_coeff : float
        辐射损失系数。

    返回
    ------
    dWdt : float
    """
    W = float(W)
    W_delayed = float(W_delayed)
    if W_delayed < 0.0:
        W_delayed = 0.0

    # 非线性反馈（ITB 阈值行为）
    denom = W0 ** n + W_delayed ** n
    if denom < 1e-30:
        feedback = 0.0
    else:
        feedback = beta * (W_delayed ** n) / denom

    dWdt = P_heat * feedback - gamma * W / tau_E - P_loss_coeff * W
    return dWdt


def rk4_dde_step(t, W, h, tau, gamma, beta, n, W0,
                 P_heat, tau_E, P_loss_coeff, t_hist, W_hist):
    """
    单步四阶 RK4 积分（含延迟项）。

    参数
    ------
    t, W : float
        当前时间与状态。
    h : float
        步长。
    tau : float
        延迟时间。
    其他 : 输运参数。
    t_hist, W_hist : ndarray
        历史记录。

    返回
    ------
    W_new : float
        下一步状态。
    """
    def rhs_now(tn, Wn, Wd):
        return transport_dde_rhs(tn, Wn, Wd, gamma, beta, n, W0,
                                 P_heat, tau_E, P_loss_coeff)

    Wd1 = interpolate_history(t - tau, t_hist, W_hist)
    k1 = h * rhs_now(t, W, Wd1)

    Wd2 = interpolate_history(t + 0.5 * h - tau, t_hist, W_hist)
    k2 = h * rhs_now(t + 0.5 * h, W + 0.5 * k1, Wd2)

    Wd3 = interpolate_history(t + 0.5 * h - tau, t_hist, W_hist)
    k3 = h * rhs_now(t + 0.5 * h, W + 0.5 * k2, Wd3)

    Wd4 = interpolate_history(t + h - tau, t_hist, W_hist)
    k4 = h * rhs_now(t + h, W + k3, Wd4)

    return W + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def simulate_energy_transport(transport_params=None, n_steps=2000):
    """
    模拟托卡马克等离子体能量输运的延迟反馈动力学。

    参数
    ------
    transport_params : dict or None
    n_steps : int
        积分总步数。

    返回
    ------
    t_arr : ndarray
    W_arr : ndarray
        能量密度历史 [J/m³]。
    P_loss_arr : ndarray
        损失功率密度历史 [W/m³]。
    info : dict
        包含振荡频率、混沌度量等信息。
    """
    if transport_params is None:
        transport_params = get_transport_params()

    gamma = transport_params["gamma"]
    beta = transport_params["beta"]
    n = transport_params["n"]
    tau = transport_params["tau"]
    t0 = transport_params["t0"]
    y0 = transport_params["y0"]
    tstop = transport_params["tstop"]

    # 物理参数
    W0 = 5.0e4          # 阈值能量密度 [J/m³]
    P_heat = 2.0e5      # 加热功率密度 [W/m³]
    tau_E = 1.5         # 能量约束时间 [s]
    P_loss_coeff = 0.05 # 辐射损失系数 [1/s]

    h = (tstop - t0) / n_steps
    if h <= 0:
        raise ValueError("步长必须为正")

    # 初始化历史：恒定初始条件
    n_hist = max(int(np.ceil(tau / h)) + 10, 100)
    t_hist = np.linspace(t0 - tau, t0, n_hist)
    W_hist = np.full(n_hist, float(y0[0]))

    t_arr = np.zeros(n_steps + 1)
    W_arr = np.zeros(n_steps + 1)
    t_arr[0] = t0
    W_arr[0] = float(y0[0])

    for step in range(n_steps):
        t_curr = t_arr[step]
        W_curr = W_arr[step]

        W_next = rk4_dde_step(
            t_curr, W_curr, h, tau, gamma, beta, n, W0,
            P_heat, tau_E, P_loss_coeff, t_hist, W_hist
        )

        # 非负截断
        W_next = max(W_next, 0.0)

        W_arr[step + 1] = W_next
        t_arr[step + 1] = t_curr + h

        # 更新历史
        t_hist = np.append(t_hist[1:], t_curr + h)
        W_hist = np.append(W_hist[1:], W_next)

    P_loss_arr = P_loss_coeff * W_arr

    # 后处理：计算 Lyapunov 指数近似（混沌度量）
    # 采用 Wolf 算法简化版：
    if len(W_arr) > 100:
        dW = np.diff(W_arr)
        lyap_approx = np.mean(np.log(np.abs(dW[1:] / (dW[:-1] + 1e-20)) + 1e-20))
    else:
        lyap_approx = 0.0

    info = {
        "delay_tau": tau,
        "mackey_glass_n": n,
        "lyapunov_approx": lyap_approx,
        "mean_energy_density": float(np.mean(W_arr)),
        "max_energy_density": float(np.max(W_arr)),
    }
    return t_arr, W_arr, P_loss_arr, info


def compute_confinement_time_scaling(I_p, B_t, n_e20, P_loss, R, a, kappa, M=2.5):
    """
    ITER89-P 能量约束时间缩放律。

    公式
    ----
        τ_E = 0.048 · I_p^{0.85} · B_t^{0.2} · n_e20^{0.1}
              · P_loss^{-0.5} · R^{1.5} · a^{0.3} · κ^{0.5} · M^{0.5}   [s]

    参数
    ------
    I_p : float
        等离子体电流 [MA]。
    B_t : float
        环向磁场 [T]。
    n_e20 : float
        线平均电子密度 [10^{20} m^{-3}]。
    P_loss : float
        损失功率 [MW]。
    R, a : float
        大半径与小半径 [m]。
    kappa : float
        拉长比。
    M : float
        平均离子质量 [amu]。

    返回
    ------
    tau_E : float
        能量约束时间 [s]。
    """
    if P_loss <= 0:
        P_loss = 1e-6
    tau_E = (0.048 * (I_p ** 0.85) * (B_t ** 0.2) *
             (n_e20 ** 0.1) * (P_loss ** (-0.5)) *
             (R ** 1.5) * (a ** 0.3) * (kappa ** 0.5) * (M ** 0.5))
    return tau_E


def compute_particle_diffusivity(q, R0, a, nu_ei, rho_i):
    """
    新经典离子热导率 / 粒子扩散系数（简化香蕉区表达式）。

    公式
    ----
        D_neo = q² · ν_ei · ρ_i² · ε^{-1.5}   [m²/s]   (简化)

    其中 ε = a / R0 为逆纵横比。

    参数
    ------
    q : float
        安全因子。
    R0 : float
        大半径 [m]。
    a : float
        小半径 [m]。
    nu_ei : float
        碰撞频率 [Hz]。
    rho_i : float
        离子拉莫尔半径 [m]。

    返回
    ------
    D_neo : float
        新经典扩散系数 [m²/s]。
    """
    epsilon = a / (R0 + 1e-20)
    epsilon_safe = max(epsilon, 1e-6)
    return (q ** 2) * nu_ei * (rho_i ** 2) / (epsilon_safe ** 1.5)
