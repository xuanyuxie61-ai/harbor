"""
脑血流动力学 — 血管重构动力学模块

整合 sir_ode（SIR 传染病模型）与 predator_prey_ode（捕食者-猎物模型），
构建脑血管网络内皮细胞增殖与凋亡、以及血氧运输的耦合动力学系统。

科学背景:
- 血管网络可视为多室系统：
    S = 含氧血量（Susceptible）
    I = 正在释放氧的血量（Infected/Active）
    R = 已释放氧的血量（Recovered/Depleted）
  对应 SIR 方程:
    dS/dt = -α S I / N + γ R
    dI/dt =  α S I / N - β I
    dR/dt =            β I - γ R

- 内皮细胞增殖(E)与凋亡(A)构成 Lotka-Volterra 型竞争:
    dE/dt = α_E E - β_E E A    (增殖受凋亡抑制)
    dA/dt = -γ_A A + δ_A E A   (凋亡随内皮密度增加)

- 剪切应力诱导血管重构:
    dR_v/dt = k_τ (|τ - τ_ref| / τ_ref)^n_sign(τ - τ_ref) - k_R R_v
    其中 τ = 4 μ Q / (π r^3) 为壁面剪切应力。
"""

import numpy as np
from scipy.integrate import solve_ivp


def sir_deriv(t, y, alpha, beta, gamma):
    """
    SIR 型血氧室模型右端项。
    y = [S, I, R]
    """
    S, I, R = y
    N = S + I + R
    if N < 1e-14:
        return np.zeros(3)
    dSdt = -alpha * S * I / N + gamma * R
    dIdt = alpha * S * I / N - beta * I
    dRdt = beta * I - gamma * R
    return np.array([dSdt, dIdt, dRdt])


def predator_prey_deriv(t, y, alpha, beta, gamma, delta):
    """
    Lotka-Volterra 型内皮细胞动力学。
    y = [E, A]  (E=增殖内皮细胞, A=凋亡因子)
    """
    E, A = y
    if E < 0:
        E = 0.0
    if A < 0:
        A = 0.0
    dEdt = alpha * E - beta * E * A
    dAdt = -gamma * A + delta * E * A
    return np.array([dEdt, dAdt])


def coupled_vascular_remodeling(t_span, y0, params):
    """
    耦合血管重构 ODE 系统:
        y = [S, I, R, E, A, r]

    其中:
        S, I, R: 血氧三室模型
        E: 内皮细胞密度 (normalized)
        A: 凋亡信号浓度
        r: 血管半径 (受剪切应力调控)

    参数字典 params:
        alpha_sir, beta_sir, gamma_sir: SIR 参数
        alpha_pp, beta_pp, gamma_pp, delta_pp: 捕食者-猎物参数
        tau_ref: 参考剪切应力 [Pa]
        k_tau: 剪切应力敏感系数
        k_R: 半径恢复系数
        mu: 血液粘度 [Pa·s]
        Q0: 基准流量 [m³/s]
    """
    alpha_s = params['alpha_sir']
    beta_s = params['beta_sir']
    gamma_s = params['gamma_sir']
    alpha_pp = params['alpha_pp']
    beta_pp = params['beta_pp']
    gamma_pp = params['gamma_pp']
    delta_pp = params['delta_pp']
    tau_ref = params['tau_ref']
    k_tau = params['k_tau']
    k_R = params['k_R']
    mu = params['mu']
    Q0 = params['Q0']

    def deriv(t, y):
        # HOLE_1: 实现耦合血管重构ODE右端项
        # 状态向量 y = [S, I, R, E, A, r]
        # 需构建以下耦合动力学:
        #   1) SIR 型血氧室模型 (dS/dt, dIdt, dRdt)
        #   2) Lotka-Volterra 内皮细胞竞争 (dEdt, dAdt)
        #   3) 壁面剪切应力 τ = 4μQ/(πr³) 诱导的半径演化 (drdt)
        #   4) 半径对供氧效率的反馈: eff = clip((r/r0)^4, 0.01, 1.0)
        raise NotImplementedError("HOLE_1: 耦合血管重构ODE系统待实现")

    sol = solve_ivp(deriv, t_span, y0, method='RK45', dense_output=True,
                    max_step=(t_span[1] - t_span[0]) / 500)
    return sol


def murray_branching_law(r0, theta, n_branches=2):
    """
    Murray 血管分支定律:
        r0^3 = r1^3 + r2^3 + ... + rn^3
    且对于对称分支，r1 = r2 = r0 / n^(1/3)。

    能量最小化条件同时给出分支角度关系:
        cos(θ1) = (r0^4 + r1^4 - r2^4) / (2 r0^2 r1^2)
    """
    if r0 <= 0:
        return np.zeros(n_branches)
    r_child = r0 / (n_branches ** (1.0 / 3.0))
    return np.full(n_branches, r_child)


def wall_shear_stress(radius, Q, mu=3.5e-3):
    """
    圆管壁面剪切应力:
        τ_w = 4 μ Q / (π r^3)
    """
    if radius <= 1e-14:
        return 0.0
    return 4.0 * mu * Q / (np.pi * radius ** 3)
