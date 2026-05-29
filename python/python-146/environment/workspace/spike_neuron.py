"""
spike_neuron.py
脉冲神经元动力学模块

基于 Hodgkin-Huxley 型生物物理模型描述神经元膜电位演化，
融合 control_bio_homework (RK4 数值积分) 与 grazing_ode (生物种群 ODE 思想)。

核心科学公式：
  膜电位方程:
    C_m dV/dt = -g_Na m^3 h (V - E_Na) - g_K n^4 (V - E_K) - g_L (V - E_L) + I_syn + I_ext

  门控变量方程 (x in {m, h, n}):
    dx/dt = alpha_x(V)(1 - x) - beta_x(V) x

  其中 alpha_x, beta_x 为电压依赖的速率函数:
    alpha_m(V) = 0.1 (V + 40) / (1 - exp(-(V + 40)/10))
    beta_m(V)  = 4.0 exp(-(V + 65)/18)
    alpha_h(V) = 0.07 exp(-(V + 65)/20)
    beta_h(V)  = 1.0 / (1 + exp(-(V + 35)/10))
    alpha_n(V) = 0.01 (V + 55) / (1 - exp(-(V + 55)/10))
    beta_n(V)  = 0.125 exp(-(V + 65)/80)

  发放条件:
    V(t) >= V_th  =>  发放脉冲, V <- V_reset,  refractory counter <- tau_ref / dt
"""

import numpy as np


class HHNeuron:
    """
    Hodgkin-Huxley 神经元模型。
    具备边界检查与数值鲁棒性处理。
    """

    # 物理常数 (单位: mV, ms, mS/cm^2, uF/cm^2)
    C_M = 1.0
    G_NA = 120.0
    G_K = 36.0
    G_L = 0.3
    E_NA = 50.0
    E_K = -77.0
    E_L = -54.387
    V_REST = -65.0
    V_TH = -50.0
    V_RESET = -65.0
    TAU_REF = 2.0  # ms

    def __init__(self, dt=0.01):
        """
        初始化神经元状态。
        dt: 时间步长 (ms), 必须为正数。
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if dt > 0.1:
            raise ValueError("dt too large for HH stability (require dt <= 0.1 ms).")
        self.dt = dt
        self.V = self.V_REST
        self.m = self._alpha_m(self.V_REST) / (self._alpha_m(self.V_REST) + self._beta_m(self.V_REST))
        self.h = self._alpha_h(self.V_REST) / (self._alpha_h(self.V_REST) + self._beta_h(self.V_REST))
        self.n = self._alpha_n(self.V_REST) / (self._alpha_n(self.V_REST) + self._beta_n(self.V_REST))
        self.refractory = 0  # 不应期计数器
        self.spike_times = []

    # ------------------------------------------------------------------
    # 电压依赖的速率函数 (带数值鲁棒性边界处理)
    # ------------------------------------------------------------------
    @staticmethod
    def _alpha_m(V):
        # alpha_m = 0.1*(V+40) / (1 - exp(-(V+40)/10))
        # 当 V -> -40 时，使用泰勒展开避免 0/0
        eps = 1e-6
        denom = 1.0 - np.exp(-(V + 40.0) / 10.0)
        if np.abs(denom) < eps:
            return 1.0  # 极限值 = 0.1 * 10 = 1.0
        return 0.1 * (V + 40.0) / denom

    @staticmethod
    def _beta_m(V):
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    @staticmethod
    def _alpha_h(V):
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    @staticmethod
    def _beta_h(V):
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    @staticmethod
    def _alpha_n(V):
        # alpha_n = 0.01*(V+55) / (1 - exp(-(V+55)/10))
        eps = 1e-6
        denom = 1.0 - np.exp(-(V + 55.0) / 10.0)
        if np.abs(denom) < eps:
            return 0.1  # 极限值 = 0.01 * 10 = 0.1
        return 0.01 * (V + 55.0) / denom

    @staticmethod
    def _beta_n(V):
        return 0.125 * np.exp(-(V + 65.0) / 80.0)

    # ------------------------------------------------------------------
    # 门控变量导数
    # ------------------------------------------------------------------
    def _gate_derivatives(self, V, m, h, n):
        dmdt = self._alpha_m(V) * (1.0 - m) - self._beta_m(V) * m
        dhdt = self._alpha_h(V) * (1.0 - h) - self._beta_h(V) * h
        dndt = self._alpha_n(V) * (1.0 - n) - self._beta_n(V) * n
        return np.array([dmdt, dhdt, dndt])

    # ------------------------------------------------------------------
    # 膜电位导数
    # ------------------------------------------------------------------
    def _dVdt(self, V, m, h, n, I_syn, I_ext):
        """
        TODO (Hole 1): 实现 Hodgkin-Huxley 膜电位导数。
        需要根据 HH 方程计算:
          C_m dV/dt = -g_Na m^3 h (V - E_Na) - g_K n^4 (V - E_K) - g_L (V - E_L) + I_syn + I_ext
        注意: 本文件的电导参数需与 signal_reconstruction.py 中线性化特征值公式保持一致。
        """
        raise NotImplementedError("Hole 1: 请补全 HH 膜电位导数公式")

    # ------------------------------------------------------------------
    # RK4 单步推进 (融合 control_bio_homework 的 RK4 思想)
    # ------------------------------------------------------------------
    def step(self, t, I_syn=0.0, I_ext=0.0):
        """
        使用经典四阶 Runge-Kutta 方法推进一个时间步。
        t: 当前时间 (仅用于记录发放时刻)
        I_syn: 突触输入电流 (uA/cm^2)
        I_ext: 外部注入电流 (uA/cm^2)
        """
        # 不应期处理
        if self.refractory > 0:
            self.refractory -= 1
            self.V = self.V_RESET
            return False

        dt = self.dt
        V0, m0, h0, n0 = self.V, self.m, self.h, self.n

        # k1
        dV1 = self._dVdt(V0, m0, h0, n0, I_syn, I_ext)
        dg1 = self._gate_derivatives(V0, m0, h0, n0)

        # k2
        V2 = V0 + 0.5 * dt * dV1
        m2 = m0 + 0.5 * dt * dg1[0]
        h2 = h0 + 0.5 * dt * dg1[1]
        n2 = n0 + 0.5 * dt * dg1[2]
        dV2 = self._dVdt(V2, m2, h2, n2, I_syn, I_ext)
        dg2 = self._gate_derivatives(V2, m2, h2, n2)

        # k3
        V3 = V0 + 0.5 * dt * dV2
        m3 = m0 + 0.5 * dt * dg2[0]
        h3 = h0 + 0.5 * dt * dg2[1]
        n3 = n0 + 0.5 * dt * dg2[2]
        dV3 = self._dVdt(V3, m3, h3, n3, I_syn, I_ext)
        dg3 = self._gate_derivatives(V3, m3, h3, n3)

        # k4
        V4 = V0 + dt * dV3
        m4 = m0 + dt * dg3[0]
        h4 = h0 + dt * dg3[1]
        n4 = n0 + dt * dg3[2]
        dV4 = self._dVdt(V4, m4, h4, n4, I_syn, I_ext)
        dg4 = self._gate_derivatives(V4, m4, h4, n4)

        # 更新
        self.V = V0 + dt / 6.0 * (dV1 + 2.0 * dV2 + 2.0 * dV3 + dV4)
        self.m = m0 + dt / 6.0 * (dg1[0] + 2.0 * dg2[0] + 2.0 * dg3[0] + dg4[0])
        self.h = h0 + dt / 6.0 * (dg1[1] + 2.0 * dg2[1] + 2.0 * dg3[1] + dg4[1])
        self.n = n0 + dt / 6.0 * (dg1[2] + 2.0 * dg2[2] + 2.0 * dg3[2] + dg4[2])

        # 边界截断，保持门控变量在 [0, 1]
        self.m = np.clip(self.m, 0.0, 1.0)
        self.h = np.clip(self.h, 0.0, 1.0)
        self.n = np.clip(self.n, 0.0, 1.0)

        # 发放检测
        if self.V >= self.V_TH:
            self.V = self.V_RESET
            self.refractory = int(np.round(self.TAU_REF / dt))
            self.spike_times.append(t)
            return True
        return False


class NeuronPopulation:
    """
    神经元群体模型，融合 grazing_ode 的捕食者-猎物（兴奋-抑制）耦合思想。
    构建脉冲耦合的兴奋性-抑制性神经网络。
    """

    def __init__(self, N_exc, N_inh, dt=0.01, p_conn=0.2):
        self.N_exc = N_exc
        self.N_inh = N_inh
        self.N = N_exc + N_inh
        self.dt = dt
        self.neurons = [HHNeuron(dt) for _ in range(self.N)]
        self.W = self._build_weight_matrix(p_conn)
        self.spike_record = [[] for _ in range(self.N)]

    def _build_weight_matrix(self, p_conn):
        """
        构建随机连接权重矩阵。
        W[i, j] 表示神经元 j -> i 的突触权重。
        兴奋性连接为正，抑制性连接为负。
        """
        np.random.seed(42)
        W = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue
                if np.random.rand() < p_conn:
                    if j < self.N_exc:
                        W[i, j] = np.random.uniform(0.5, 2.0)
                    else:
                        W[i, j] = np.random.uniform(-2.0, -0.5)
        return W

    def compute_synaptic_current(self, spike_vector):
        """
        计算突触输入电流。
        spike_vector: 长度为 N 的 0/1 向量，表示各神经元是否在本步发放。
        I_syn_i = sum_j W_{ij} * spike_j
        """
        return self.W @ spike_vector

    def simulate(self, T_total, I_ext_per_neuron=None):
        """
        运行群体仿真。
        T_total: 总仿真时间 (ms)
        I_ext_per_neuron: 每个神经元的外部电流 (uA/cm^2)
        """
        n_steps = int(np.round(T_total / self.dt))
        if I_ext_per_neuron is None:
            I_ext_per_neuron = np.zeros(self.N)
        if len(I_ext_per_neuron) != self.N:
            raise ValueError("I_ext_per_neuron length must match N.")

        voltage_trace = np.zeros((self.N, n_steps))
        spike_raster = np.zeros((self.N, n_steps), dtype=int)

        for step_idx in range(n_steps):
            t = step_idx * self.dt
            # 收集上一步的脉冲
            spike_vec = spike_raster[:, step_idx - 1] if step_idx > 0 else np.zeros(self.N)
            I_syn = self.compute_synaptic_current(spike_vec)

            for i in range(self.N):
                fired = self.neurons[i].step(t, I_syn[i], I_ext_per_neuron[i])
                voltage_trace[i, step_idx] = self.neurons[i].V
                if fired:
                    spike_raster[i, step_idx] = 1
                    self.spike_record[i].append(t)

        return voltage_trace, spike_raster


def demo_single_neuron():
    """单神经元 demo：恒定电流注入下的脉冲发放。"""
    neuron = HHNeuron(dt=0.01)
    T = 50.0
    n_steps = int(T / neuron.dt)
    V_trace = np.zeros(n_steps)
    I_ext = 10.0  # uA/cm^2
    for k in range(n_steps):
        t = k * neuron.dt
        neuron.step(t, I_ext=I_ext)
        V_trace[k] = neuron.V
    return V_trace, neuron.spike_times


def demo_population():
    """神经元群体 demo。"""
    pop = NeuronPopulation(N_exc=10, N_inh=5, dt=0.01, p_conn=0.3)
    I_ext = np.concatenate([
        np.random.uniform(5.0, 12.0, pop.N_exc),
        np.random.uniform(3.0, 8.0, pop.N_inh)
    ])
    voltage_trace, spike_raster = pop.simulate(T_total=30.0, I_ext_per_neuron=I_ext)
    return voltage_trace, spike_raster, pop.spike_record
