"""
collective_ode.py
=================
原子核集体运动非线性 ODE 系统模块

本模块实现 Bohr-Mottelson 集体运动模型的耦合非线性常微分方程组，
描述原子核四极形变参数 β₂(t)、γ(t) 以及集体转动角动量 R(t) 的时间演化。

物理模型：
1. 四极振动-转动耦合哈密顿量：
   H_coll = Σ_{i=1}^3 [π_i² / (2B_i) + (1/2) C_i α_i²]
   其中 α_i 为实验室系四极张量分量，与 (β, γ) 的关系为：
   α_1 = β cosγ
   α_2 = β sinγ / √2
   α_3 = -β sinγ / √2

2. 非线性耦合项（来源于核子-集体运动耦合）：
   dβ/dt = π_β / B_β - κ_β β (β² - β₀²)
   dγ/dt = π_γ / (B_γ β²) - κ_γ γ
   dπ_β/dt = -C_β (β - β_eq) - ∂V_pair/∂β
   dπ_γ/dt = -C_γ γ - ∂V_pair/∂γ

3. 转动自由度（沿体坐标系 3 轴）：
   dR_3/dt = -λ_R R_3 + F_ext(t)
   dφ/dt = R_3 / I_3

其中 V_pair 为配对关联能，I_3 为绕 3 轴的转动惯量。
"""

import numpy as np
from math import sqrt, sin, cos, pi


class CollectiveHamiltonian:
    """
    原子核集体运动哈密顿量参数容器。
    """
    def __init__(self, mass_number=100, beta_eq=0.2, gamma_eq=0.0):
        self.A = mass_address = mass_number
        self.beta_eq = beta_eq
        self.gamma_eq = gamma_eq

        # 质量参数 (B_β, B_γ) 单位：MeV⁻¹ fm⁻⁴
        # 经验公式：B_β ≈ B_γ ≈ (3/8π) A M R₀²
        self.B_beta = 0.06 * self.A ** (5.0 / 3.0)
        self.B_gamma = self.B_beta

        # 刚度参数 (C_β, C_γ) 单位：MeV fm⁻²
        # 经验值：C_β ~ 50–200 MeV，与壳层结构相关
        self.C_beta = 80.0 + 0.3 * self.A
        self.C_gamma = 60.0 + 0.2 * self.A

        # 非线性耦合强度
        self.kappa_beta = 5.0   # MeV fm⁻²
        self.kappa_gamma = 3.0  # MeV fm⁻²

        # 转动惯量 (单位：MeV⁻¹)
        # 刚体转动惯量：I_rigid = (2/5) M A R²
        self.I3 = 0.02 * self.A ** (5.0 / 3.0)

        # 配对关联参数
        self.G_pair = 25.0 / self.A  # MeV
        self.delta_pair = 12.0 / sqrt(self.A)  # MeV

        # 阻尼系数
        self.lambda_beta = 0.5
        self.lambda_gamma = 0.5
        self.lambda_R = 0.3

    def pairing_energy(self, beta, gamma):
        """
        配对关联能作为形变参数的函数。

        近似：E_pair(β, γ) = -Δ² / G · [1 - c_β (β - β_eq)² - c_γ γ²]
        """
        return -self.delta_pair ** 2 / self.G_pair * (
            1.0 - 0.5 * (beta - self.beta_eq) ** 2 - 0.3 * gamma ** 2
        )

    def potential_energy(self, beta, gamma):
        """
        集体势能面：
        V(β, γ) = (1/2) C_β (β - β_eq)² + (1/2) C_γ γ² + E_pair(β, γ)
                  + V_cubic(β, γ)
        """
        V_harmonic = 0.5 * self.C_beta * (beta - self.beta_eq) ** 2
        V_harmonic += 0.5 * self.C_gamma * gamma ** 2
        V_pair = self.pairing_energy(beta, gamma)
        # 立方非谐项（模拟 γ-不稳定势）
        V_cubic = -2.0 * beta ** 3 * cos(3.0 * gamma)
        return V_harmonic + V_pair + V_cubic


def collective_derivatives(t, state, ham):
    """
    集体运动状态方程的右端项。

    状态向量：state = [β, γ, π_β, π_γ, R_3, φ]

    方程组：
    dβ/dt   = π_β / B_β
    dγ/dt   = π_γ / (B_γ β²)    (注意 β=0 时的奇异性)
    dπ_β/dt = -C_β(β - β_eq) - ∂V_pair/∂β - λ_β π_β
    dπ_γ/dt = -C_γ γ - ∂V_pair/∂γ - λ_γ π_γ
    dR_3/dt = -λ_R R_3 + F_ext(t)
    dφ/dt   = R_3 / I_3

    参数
    ----
    t : float
        时间 (MeV⁻¹·ħ，自然单位)
    state : ndarray, shape (6,)
        [beta, gamma, pi_beta, pi_gamma, R3, phi]
    ham : CollectiveHamiltonian
        哈密顿量参数

    返回
    ----
    dstate : ndarray, shape (6,)
        状态导数
    """
    beta, gamma, pi_beta, pi_gamma, R3, phi = state

    # 防止 β 过小导致数值奇异性
    beta_safe = max(abs(beta), 1e-6)

    d_beta = pi_beta / ham.B_beta
    d_gamma = pi_gamma / (ham.B_gamma * beta_safe ** 2)

    # 势能梯度
    dV_dbeta = ham.C_beta * (beta - ham.beta_eq)
    dV_dbeta += (2.0 * ham.delta_pair ** 2 / ham.G_pair *
                 0.5 * (beta - ham.beta_eq))
    dV_dbeta += -6.0 * beta_safe ** 2 * cos(3.0 * gamma)  # 立方项梯度

    dV_dgamma = ham.C_gamma * gamma
    dV_dgamma += (2.0 * ham.delta_pair ** 2 / ham.G_pair * 0.3 * gamma)
    dV_dgamma += 6.0 * beta_safe ** 3 * sin(3.0 * gamma)

    d_pi_beta = -dV_dbeta - ham.lambda_beta * pi_beta
    d_pi_gamma = -dV_dgamma - ham.lambda_gamma * pi_gamma

    # 外力矩（模拟碰撞或电磁激发）
    F_ext = 0.1 * sin(0.5 * t) * np.exp(-0.01 * t)
    d_R3 = -ham.lambda_R * R3 + F_ext
    d_phi = R3 / ham.I3

    return np.array([d_beta, d_gamma, d_pi_beta, d_pi_gamma, d_R3, d_phi])


def rk4_step(f, t, y, h, *args):
    """
    经典四阶 Runge-Kutta 单步积分。

    y_{n+1} = y_n + (h/6)(k1 + 2k2 + 2k3 + k4)
    """
    k1 = f(t, y, *args)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1, *args)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2, *args)
    k4 = f(t + h, y + h * k3, *args)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_collective_motion(ham, t_span, n_steps, y0=None):
    """
    求解集体运动 ODE 系统。

    参数
    ----
    ham : CollectiveHamiltonian
    t_span : tuple
        (t_min, t_max)
    n_steps : int
        时间步数
    y0 : ndarray, optional
        初始状态，默认 [beta_eq, 0, 0, 0, 0, 0]

    返回
    ----
    t_array : ndarray
        时间序列
    y_array : ndarray, shape (n_steps+1, 6)
        状态历史
    energy_array : ndarray
        总集体能量随时间演化
    """
    t_min, t_max = t_span
    h = (t_max - t_min) / n_steps
    t_array = np.linspace(t_min, t_max, n_steps + 1)

    if y0 is None:
        y0 = np.array([ham.beta_eq, ham.gamma_eq, 0.0, 0.0, 0.0, 0.0])

    y_array = np.zeros((n_steps + 1, 6))
    y_array[0] = y0
    energy_array = np.zeros(n_steps + 1)

    # 初始能量
    beta, gamma, pi_beta, pi_gamma, R3, phi = y0
    E0 = (pi_beta ** 2 / (2.0 * ham.B_beta) +
          pi_gamma ** 2 / (2.0 * ham.B_gamma * max(beta, 1e-6) ** 2) +
          ham.potential_energy(beta, gamma) +
          0.5 * R3 ** 2 / ham.I3)
    energy_array[0] = E0

    for n in range(n_steps):
        y_array[n + 1] = rk4_step(collective_derivatives,
                                   t_array[n], y_array[n], h, ham)
        beta, gamma, pi_beta, pi_gamma, R3, phi = y_array[n + 1]
        E = (pi_beta ** 2 / (2.0 * ham.B_beta) +
             pi_gamma ** 2 / (2.0 * ham.B_gamma * max(beta, 1e-6) ** 2) +
             ham.potential_energy(beta, gamma) +
             0.5 * R3 ** 2 / ham.I3)
        energy_array[n + 1] = E

    return t_array, y_array, energy_array


def adiabatic_invariant(y_array, ham, dt):
    """
    计算绝热不变量（作用量变量）以检验数值守恒性。

    对于 β 振动：I_β = (1/2π) ∮ π_β dβ
    采用离散近似：I_β ≈ Σ |π_β| |Δβ| / (2π N_cycles)
    """
    beta = y_array[:, 0]
    pi_beta = y_array[:, 2]

    # 计算每个周期的积分
    crossings = 0
    invariant = 0.0
    for i in range(1, len(beta)):
        if beta[i] * beta[i - 1] < 0:
            crossings += 1
        invariant += abs(pi_beta[i]) * abs(beta[i] - beta[i - 1])

    if crossings > 0:
        invariant /= (2.0 * pi * crossings)
    return invariant
