"""
dynamics_model.py
资产价格耦合动力学与随机微分方程数值求解模块。

融入的原项目核心算法：
- 1138_spring_double_ode: 双弹簧耦合ODE系统
- 1286_trapezoidal: 梯形隐式ODE求解器

科学背景：
金融市场中的资产价格并非独立运动，而是存在复杂的耦合关系：
动量效应（momentum）与均值回归（mean reversion）如同弹簧-质量系统
中的惯性力与恢复力。本项目将双弹簧ODE模型推广至高维资产系统，
并引入随机扰动项，构建耦合随机微分方程（CSDE）模型；
采用梯形隐式格式进行数值积分，保证在大时间步长下的数值稳定性。

核心数学模型：
1. 耦合SDE系统（推广的双弹簧模型）：
    对资产 i，设其价格偏离均衡水平的幅度为 u_i，动量为 v_i，则
        du_i = v_i dt,
        dv_i = [ -k_1 u_i + Σ_j k_{2,ij} (u_j - u_i) - γ v_i ] dt + σ_i dW_i(t)。
    其中 k_1 为个体均值回归强度，k_{2,ij} 为资产间耦合强度，
    γ 为阻尼系数，W_i(t) 为维纳过程。

2. 梯形隐式格式（Crank-Nicolson-like）：
    对确定性部分采用梯形法则，对随机部分采用Euler-Maruyama：
        Y_{n+1} = Y_n + 0.5 Δt [ f(t_n, Y_n) + f(t_{n+1}, Y_{n+1}) ] + g(t_n, Y_n) ΔW_n。
    每步需解非线性方程组，采用不动点迭代。
"""

import numpy as np


def coupled_market_dynamics(y: np.ndarray, t: float,
                             k1: float, K2: np.ndarray,
                             gamma: float, m: np.ndarray) -> np.ndarray:
    """
    高维耦合市场动力学系统的右端项。

    状态向量 y = [u_1, v_1, u_2, v_2, ..., u_n, v_n]^T，其中
    u_i 为资产 i 的价格偏离，v_i 为速度（动量）。

    微分方程：
        du_i/dt = v_i,
        dv_i/dt = ( -k1 * u_i + Σ_j K2_{ij} (u_j - u_i) - gamma * v_i ) / m_i。

    参数
    ----------
    y : np.ndarray, shape (2*n,)
        状态向量。
    t : float
        时间（未显式使用，保留接口一致性）。
    k1 : float
        个体均值回归强度。
    K2 : np.ndarray, shape (n, n)
        资产间耦合强度矩阵。
    gamma : float
        阻尼系数。
    m : np.ndarray, shape (n,)
        等效质量（惯性系数）。

    返回
    -------
    np.ndarray, shape (2*n,)
        时间导数 dy/dt。
    """
    n = len(m)
    if len(y) != 2 * n:
        raise ValueError("coupled_market_dynamics: 状态向量维度必须是 2n。")
    u = y[0::2]
    v = y[1::2]
    dydt = np.zeros_like(y)
    # === HOLE 3 BEGIN ===
    # TODO: 实现高维耦合市场动力学右端项
    # 科学知识：推广双弹簧ODE至高维资产耦合系统
    # 微分方程组：
    #   du_i/dt = v_i,
    #   dv_i/dt = ( -k1 * u_i + Σ_j K2_{ij} (u_j - u_i) - γ * v_i ) / m_i
    # 其中 u = y[0::2], v = y[1::2]
    # 可用变量：u, v, k1, K2, gamma, m, n, dydt
    # 需计算耦合项 coupling = K2 @ u - (K2.sum(axis=1)) * u
    # 并组装 dudt, dvdt 到 dydt[0::2], dydt[1::2]
    # === HOLE 3 END ===
    raise NotImplementedError("Hole 3: 耦合市场动力学核心方程待实现")


def trapezoidal_sde_solver(f, g, tspan: tuple, y0: np.ndarray,
                            n_steps: int, rng: np.random.Generator = None) -> tuple:
    """
    梯形隐式格式求解随机微分方程。

    对 SDE  dY = f(t, Y) dt + g(t, Y) dW，
    离散格式为：
        Y_{n+1} = Y_n + 0.5 Δt [ f(t_n, Y_n) + f(t_{n+1}, Y_{n+1}) ]
                          + g(t_n, Y_n) √Δt Z_n，
    其中 Z_n ~ N(0, I)。

    每步对隐式部分采用不动点迭代求解。

    参数
    ----------
    f : callable
        漂移项函数 f(t, y)。
    g : callable
        扩散项函数 g(t, y)。
    tspan : tuple (t0, tf)
        时间区间。
    y0 : np.ndarray
        初始条件。
    n_steps : int
        时间步数。
    rng : np.random.Generator
        随机数生成器。

    返回
    -------
    t : np.ndarray
        时间网格。
    y : np.ndarray, shape (n_steps+1, len(y0))
        数值解。
    """
    if rng is None:
        rng = np.random.default_rng()
    t0, tf = tspan
    dt = (tf - t0) / n_steps
    m = len(y0)
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, m))
    y[0, :] = y0

    for i in range(n_steps):
        ti = t[i]
        yi = y[i, :]
        fi = f(ti, yi)
        gi = g(ti, yi)
        dW = np.sqrt(dt) * rng.standard_normal(m)
        explicit_part = yi + dt * fi + gi * dW

        # 不动点迭代求解隐式方程
        yn = explicit_part.copy()
        for _ in range(20):
            fn = f(ti + dt, yn)
            yn_new = yi + 0.5 * dt * (fi + fn) + gi * dW
            if np.linalg.norm(yn_new - yn) < 1e-12:
                break
            yn = yn_new
        y[i + 1, :] = yn
    return t, y


def simulate_contagion(n_assets: int, T: float = 1.0, dt: float = 0.01,
                        k1: float = 2.0, gamma: float = 0.5,
                        sigma_noise: float = 0.3,
                        rng: np.random.Generator = None) -> tuple:
    """
    模拟资产价格传染动力学。

    模型设定：
    - 资产1在 t = T/2 时受到冲击（u_1 突变）。
    - 通过耦合矩阵 K2，冲击传播到其他资产。
    - 观察各资产的响应幅度与恢复时间。

    参数
    ----------
    n_assets : int
        资产数量。
    T : float
        总模拟时间。
    dt : float
        时间步长。
    k1 : float
        均值回归强度。
    gamma : float
        阻尼系数。
    sigma_noise : float
        噪声强度。
    rng : np.random.Generator
        随机数生成器。

    返回
    -------
    t : np.ndarray
        时间网格。
    y : np.ndarray
        状态轨迹。
    max_deviation : np.ndarray
        各资产的最大偏离幅度。
    """
    if rng is None:
        rng = np.random.default_rng()
    n_steps = int(np.round(T / dt))
    m = np.ones(n_assets)  # 等效质量
    # 构造耦合矩阵：资产1为中心节点
    K2 = np.zeros((n_assets, n_assets))
    for i in range(1, n_assets):
        K2[0, i] = 1.0
        K2[i, 0] = 1.0
    # 随机扰动耦合强度
    K2 += 0.2 * rng.random((n_assets, n_assets))
    K2 = 0.5 * (K2 + K2.T)
    np.fill_diagonal(K2, 0.0)

    y0 = np.zeros(2 * n_assets)
    # 在时间中点对资产1施加冲击

    def f(t, y):
        return coupled_market_dynamics(y, t, k1, K2, gamma, m)

    def g(t, y):
        noise = np.zeros(2 * n_assets)
        noise[1::2] = sigma_noise
        return noise

    t, y = trapezoidal_sde_solver(f, g, (0.0, T), y0, n_steps, rng)

    # 施加冲击
    shock_step = n_steps // 2
    y[shock_step:, 0] += 0.5 * np.exp(-k1 * (t[shock_step:] - t[shock_step]))

    u = y[:, 0::2]
    max_deviation = np.max(np.abs(u), axis=0)
    return t, y, max_deviation


def trapezoidal_ode_solver(f, tspan: tuple, y0: np.ndarray,
                            n_steps: int) -> tuple:
    """
    确定性梯形法ODE求解器（无随机项）。

    格式：
        y_{n+1} = y_n + 0.5 Δt [ f(t_n, y_n) + f(t_{n+1}, y_{n+1}) ]。

    截断误差为 O(Δt^3)，A-稳定，适合刚性ODE。
    """
    t0, tf = tspan
    dt = (tf - t0) / n_steps
    m = len(y0)
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, m))
    y[0, :] = y0
    for i in range(n_steps):
        ti = t[i]
        yi = y[i, :]
        fi = f(ti, yi)
        yn = yi + dt * fi
        for _ in range(20):
            fn = f(ti + dt, yn)
            yn_new = yi + 0.5 * dt * (fi + fn)
            if np.linalg.norm(yn_new - yn) < 1e-12:
                break
            yn = yn_new
        y[i + 1, :] = yn
    return t, y
