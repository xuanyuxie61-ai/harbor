"""
neural_mass_ode.py — 神经群体振荡器与 SIR-like 神经状态动力学
===============================================================
融合 sawtooth_ode（锯齿波驱动振荡器）与 SIR_ode（SIR 流行病模型）思想，
构建描述皮层局部场电位（LFP）生成机制的神经元群体模型。

核心模型：
1. Wilson-Cowan 型兴奋-抑制（E-I）神经质量模型，加入周期性锯齿波驱动
   模拟脑电 theta/gamma 节律的外部调制。
2. 神经元三态室模型（Active-Inactive-Refractory, AIR）
   基于 SIR 流行病模型改造：神经元在活跃态(A)、静息态(Q)、不应态(R)之间转换。

数学方程：
---
**E-I 振荡器（微尺度）：**

令 E(t), I(t) 分别为兴奋性与抑制性群体平均激活率，外部输入为锯齿波 s(t)：

    dE/dt = -E + S_e( a_ee * E - a_ei * I + P_e + k_e * s(t) )
    dI/dt = -I + S_i( a_ie * E - a_ii * I + P_i + k_i * s(t) )

其中 S_e, S_i 为 sigmoid 激活函数：

    S(x) = 1 / (1 + exp(-(x - theta)/sigma))

s(t) 为角频率 ω 的锯齿波：

    s(t) = A_s * ( mod(t + π/ω, 2π/ω) - π/ω )

---
**AIR 神经元状态室模型（扩展 SIR）：**

设 N 为神经元总数，A(t), Q(t), R(t) 分别为活跃、静息、不应态数量：

    dA/dt =  α * f_conn(E) * Q * A / N  - β * A  + γ * R
    dQ/dt = -α * f_conn(E) * Q * A / N  + β * A  - δ * Q
    dR/dt =  δ * Q - γ * R

其中 f_conn(E) = sigmoid(E) 表示连接强度随兴奋水平调制。
总数量守恒：A + Q + R = N。

---
**LFP 近似：**

局部场电位与突触后电流总和成正比：

    LFP(t) ≈ k_E * E(t) - k_I * I(t) + η * sqrt(dt) * ξ(t)

其中 ξ(t) 为高斯白噪声，η 为噪声强度。
"""

import numpy as np
from utils import sigmoid_activation, sawtooth_wave, rk4_step


class EIOscillator:
    """
    兴奋-抑制（E-I）神经质量振荡器，带锯齿波外部驱动。
    """

    def __init__(self,
                 a_ee=12.0, a_ei=4.0, a_ie=13.0, a_ii=11.0,
                 P_e=2.5, P_i=0.0,
                 theta_e=2.8, theta_i=4.0,
                 sigma_e=1.0, sigma_i=1.0,
                 k_e=1.5, k_i=0.5,
                 omega=2.0 * np.pi * 6.0,  # 6 Hz theta 节律
                 sawtooth_amp=1.0):
        self.a_ee = a_ee
        self.a_ei = a_ei
        self.a_ie = a_ie
        self.a_ii = a_ii
        self.P_e = P_e
        self.P_i = P_i
        self.theta_e = theta_e
        self.theta_i = theta_i
        self.sigma_e = sigma_e
        self.sigma_i = sigma_i
        self.k_e = k_e
        self.k_i = k_i
        self.omega = omega
        self.sawtooth_amp = sawtooth_amp

    def _dynamics(self, t, state):
        # TODO_HOLE_1: implement E-I neural mass dynamics
        # Given state = [E, I], compute dE/dt and dI/dt for the Wilson-Cowan model:
        #   dE/dt = -E + S_e(a_ee * E - a_ei * I + P_e + k_e * s(t))
        #   dI/dt = -I + S_i(a_ie * E - a_ii * I + P_i + k_i * s(t))
        # where s(t) is the sawtooth wave and S_* is the sigmoid activation.
        # Must return np.array([dE, dI], dtype=float).
        pass

    def simulate(self, E0=0.1, I0=0.05, t_span=(0.0, 5.0), dt=0.001):
        """
        返回时间序列 t, 状态矩阵 state[:, [E, I]]
        """
        t0, tf = t_span
        n_steps = int(np.ceil((tf - t0) / dt))
        t = np.linspace(t0, tf, n_steps + 1)
        state = np.zeros((n_steps + 1, 2), dtype=float)
        state[0] = [E0, I0]
        for i in range(n_steps):
            state[i + 1] = rk4_step(self._dynamics, t[i], state[i], dt)
        return t, state

    def compute_lfp(self, state, k_E=1.0, k_I=1.5, noise_std=0.02, dt=0.001):
        """
        由 E, I 状态计算 LFP 信号：LFP = k_E * E - k_I * I + noise
        """
        E = state[:, 0]
        I = state[:, 1]
        lfp = k_E * E - k_I * I
        if noise_std > 0:
            lfp += noise_std * np.sqrt(dt) * np.random.randn(len(lfp))
        return lfp


class AIRPopulationDynamics:
    """
    Active-Inactive-Refractory (AIR) 神经元群体动力学。
    基于 SIR 模型改造，将感染态映射为神经元活跃态，
    引入兴奋性输入 E(t) 调制连接强度。
    """

    def __init__(self, N=1000, alpha=0.3, beta=0.1, gamma=0.05, delta=0.02):
        """
        N      : 神经元总数
        alpha  : 静息→活跃转换率（受连接调制）
        beta   : 活跃→静息转换率（自然衰减）
        gamma  : 不应→活跃恢复率
        delta  : 静息→不应转换率
        """
        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def _dynamics(self, t, state, E_input_func):
        A, Q, R = state
        # 确保非负
        A = max(A, 0.0)
        Q = max(Q, 0.0)
        R = max(R, 0.0)
        N = self.N
        # 兴奋性输入调制连接概率
        E_t = E_input_func(t)
        conn_mod = sigmoid_activation(E_t, theta=0.5, sigma=0.3)
        # AIR 动力学
        dA = (self.alpha * conn_mod * Q * A / N
              - self.beta * A
              + self.gamma * R)
        dQ = (-self.alpha * conn_mod * Q * A / N
              + self.beta * A
              - self.delta * Q)
        dR = (self.delta * Q
              - self.gamma * R)
        return np.array([dA, dQ, dR], dtype=float)

    def simulate(self, E_input_func, A0=10, Q0=None, R0=0,
                 t_span=(0.0, 10.0), dt=0.01):
        """
        E_input_func : callable(t) -> float，外部兴奋性输入
        """
        if Q0 is None:
            Q0 = self.N - A0 - R0
        t0, tf = t_span
        n_steps = int(np.ceil((tf - t0) / dt))
        t = np.linspace(t0, tf, n_steps + 1)
        state = np.zeros((n_steps + 1, 3), dtype=float)
        state[0] = [float(A0), float(Q0), float(R0)]
        for i in range(n_steps):
            state[i + 1] = rk4_step(
                lambda ti, si: self._dynamics(ti, si, E_input_func),
                t[i], state[i], dt)
            # 强制守恒：A + Q + R = N
            s = state[i + 1]
            s = np.maximum(s, 0.0)
            total = np.sum(s)
            if total > 0:
                s = s * (self.N / total)
            state[i + 1] = s
        return t, state


class MultiPopulationArray:
    """
    多通道 E-I 振荡器阵列，模拟多电极记录的局部神经群体。
    """

    def __init__(self, n_channels=8, coupling_matrix=None, **ei_kwargs):
        self.n_channels = n_channels
        self.oscillators = [EIOscillator(**ei_kwargs) for _ in range(n_channels)]
        if coupling_matrix is None:
            # 弱近邻耦合
            C = np.eye(n_channels) * 0.0
            for i in range(n_channels - 1):
                C[i, i + 1] = 0.1
                C[i + 1, i] = 0.1
            coupling_matrix = C
        self.C = np.asarray(coupling_matrix, dtype=float)
        # 确保对角为零
        np.fill_diagonal(self.C, 0.0)

    def simulate(self, initial_states=None, t_span=(0.0, 3.0), dt=0.001):
        """
        各通道独立演化，加入线性耦合项。
        状态扩展为 [E_0, I_0, E_1, I_1, ..., E_{n-1}, I_{n-1}]
        耦合只作用于 E 变量：dE_i/dt += sum_j C_{ij} * (E_j - E_i)
        """
        n = self.n_channels
        if initial_states is None:
            initial_states = np.random.rand(n, 2) * 0.1
        t0, tf = t_span
        n_steps = int(np.ceil((tf - t0) / dt))
        t = np.linspace(t0, tf, n_steps + 1)
        state = np.zeros((n_steps + 1, 2 * n), dtype=float)
        state[0] = np.asarray(initial_states, dtype=float).flatten()

        def full_dynamics(ti, y):
            dydt = np.zeros_like(y)
            E_vals = y[0::2]
            for ch in range(n):
                idx_e = 2 * ch
                idx_i = 2 * ch + 1
                # 局部 E-I 动力学
                local = self.oscillators[ch]._dynamics(ti, [y[idx_e], y[idx_i]])
                # 通道间耦合（扩散耦合）
                coupling = np.sum(self.C[ch, :] * (E_vals - y[idx_e]))
                dydt[idx_e] = local[0] + coupling
                dydt[idx_i] = local[1]
            return dydt

        for i in range(n_steps):
            state[i + 1] = rk4_step(full_dynamics, t[i], state[i], dt)
        return t, state

    def extract_lfp_channels(self, state, k_E=1.0, k_I=1.5, noise_std=0.02, dt=0.001):
        """
        从全状态矩阵提取各通道 LFP。
        返回 shape (n_channels, n_timepoints)
        """
        n = self.n_channels
        n_t = state.shape[0]
        lfp = np.zeros((n, n_t), dtype=float)
        for ch in range(n):
            E = state[:, 2 * ch]
            I = state[:, 2 * ch + 1]
            lfp[ch] = k_E * E - k_I * I
        if noise_std > 0:
            lfp += noise_std * np.sqrt(dt) * np.random.randn(n, n_t)
        return lfp
