"""
neural_membrane.py
==================
神经元膜电位动力学模块

基于种子项目:
  - 100_blood_pressure_ode: ODE 参数化模型思想（血流动力学 ODE）
  - 619_kepler_perturbed_ode: 扰动 ODE 数值积分思想

科学背景:
  螺旋神经节神经元(SGN)在电刺激下的膜电位变化
  可用修正的 Hodgkin-Huxley 型 ODE 描述。

  简化 FitzHugh-Nagumo 模型:
      dV_m/dt = (1/τ_m) [ V_m - V_m³/3 - W + I_stim(t) ]
      dW/dt  = ε (V_m + a - b W)

  其中:
    - V_m: 膜电位 (mV)
    - W: 恢复变量
    - τ_m: 膜时间常数 (~0.1 ms)
    - ε: 恢复时间尺度参数
    - I_stim(t): 刺激电流 (μA/cm²)

  发放条件: 当 V_m 超过阈值 V_th 时记为一次动作电位。

  更精细的模型使用 Goldman-Hodgkin-Katz 方程:
      I_ion = g_Na m³h (V_m - E_Na) + g_K n⁴ (V_m - E_K) + g_L (V_m - E_L)
"""

import numpy as np
from scipy.integrate import solve_ivp


class SimplifiedSGNModel:
    """
    螺旋神经节神经元简化模型 (FitzHugh-Nagumo 型)。
    """

    def __init__(self, tau_m=0.1, epsilon=0.08, a=0.7, b=0.8,
                 V_rest=-65.0, V_thresh=-40.0):
        """
        Parameters
        ----------
        tau_m : float
            膜时间常数 (ms)
        epsilon : float
            恢复变量时间尺度
        a, b : float
            FHN 模型参数
        V_rest : float
            静息电位 (mV)
        V_thresh : float
            发放阈值 (mV)
        """
        self.tau_m = float(tau_m)
        self.epsilon = float(epsilon)
        self.a = float(a)
        self.b = float(b)
        self.V_rest = float(V_rest)
        self.V_thresh = float(V_thresh)

    def derivatives(self, t, y, stimulus_func):
        """
        ODE 右端项。

        Parameters
        ----------
        t : float
            时间 (ms)
        y : ndarray, shape (2,)
            [V_m, W]
        stimulus_func : callable
            I_stim(t) -> float

        Returns
        -------
        dydt : ndarray, shape (2,)
        """
        V_m, W = y
        # 将 V_m 归一化到 FHN 标准变量 (约 -2.5 到 2.5)
        v = (V_m - self.V_rest) / 25.0
        I_stim = stimulus_func(t) / 25.0  # 归一化刺激

        dvdt = (v - v**3 / 3.0 - W + I_stim) / self.tau_m
        dWdt = self.epsilon * (v + self.a - self.b * W)

        dVdt = dvdt * 25.0
        return np.array([dVdt, dWdt])

    def simulate(self, t_span, y0, stimulus_func, method='RK45',
                 max_step=0.01):
        """
        数值积分模拟膜电位时间演化。

        Parameters
        ----------
        t_span : tuple
            (t0, tf) in ms
        y0 : ndarray
            初始状态 [V_m0, W0]
        stimulus_func : callable
        method : str
            积分方法
        max_step : float
            最大步长

        Returns
        -------
        sol : OdeSolution
            scipy 解对象
        spike_times : list
            动作电位发放时刻
        """
        y0 = np.asarray(y0, dtype=float)
        if len(y0) != 2:
            raise ValueError("y0 长度必须为 2")

        # 事件检测: 膜电位上穿阈值
        def threshold_cross(t, y):
            return y[0] - self.V_thresh
        threshold_cross.terminal = False
        threshold_cross.direction = 1

        sol = solve_ivp(
            lambda t, y: self.derivatives(t, y, stimulus_func),
            t_span, y0, method=method, max_step=max_step,
            events=threshold_cross, dense_output=True
        )

        spike_times = sol.t_events[0].tolist() if sol.t_events[0] is not None else []
        return sol, spike_times


class DetailedSGNModel:
    """
    详细 SGN 膜模型 (Hodgkin-Huxley 风格)。

    膜方程:
        C_m dV_m/dt = -g_Na m³h (V_m - E_Na)
                        - g_K n⁴ (V_m - E_K)
                        - g_L (V_m - E_L)
                        + I_stim(t)

    门控变量动力学:
        dx/dt = α_x(V) (1-x) - β_x(V) x,   x ∈ {m, h, n}

    典型参数 (哺乳动物 SGN，37°C):
        C_m = 1.0 μF/cm²
        g_Na = 120 mS/cm², E_Na = +50 mV
        g_K  =  36 mS/cm², E_K  = -77 mV
        g_L  = 0.3 mS/cm², E_L  = -54.4 mV
    """

    def __init__(self, C_m=1.0, g_Na=120.0, g_K=36.0, g_L=0.3,
                 E_Na=50.0, E_K=-77.0, E_L=-54.4, T=310.15):
        """
        Parameters
        ----------
        C_m : float
            膜电容 (μF/cm²)
        g_Na, g_K, g_L : float
            最大电导 (mS/cm²)
        E_Na, E_K, E_L : float
            平衡电位 (mV)
        T : float
            温度 (K)
        """
        self.C_m = float(C_m)
        self.g_Na = float(g_Na)
        self.g_K = float(g_K)
        self.g_L = float(g_L)
        self.E_Na = float(E_Na)
        self.E_K = float(E_K)
        self.E_L = float(E_L)
        self.T = float(T)
        # Q10 温度修正因子 (假设 6.3)
        self.q10 = 6.3
        self.T_ref = 310.15  # 参考温度 37°C
        self.phi = self.q10 ** ((self.T - self.T_ref) / 10.0)

    def alpha_m(self, V):
        """Na⁺ 激活门控速率 (ms⁻¹)。"""
        V = np.asarray(V, dtype=float)
        return np.where(
            np.abs(V + 40.0) < 1e-6,
            1.0,
            0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
        )

    def beta_m(self, V):
        """Na⁺ 激活门控失活速率 (ms⁻¹)。"""
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    def alpha_h(self, V):
        """Na⁺ 失活门控激活速率 (ms⁻¹)。"""
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    def beta_h(self, V):
        """Na⁺ 失活门控失活速率 (ms⁻¹)。"""
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    def alpha_n(self, V):
        """K⁺ 门控激活速率 (ms⁻¹)。"""
        V = np.asarray(V, dtype=float)
        return np.where(
            np.abs(V + 55.0) < 1e-6,
            0.1,
            0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
        )

    def beta_n(self, V):
        """K⁺ 门控失活速率 (ms⁻¹)。"""
        return 0.125 * np.exp(-(V + 65.0) / 80.0)

    def derivatives(self, t, y, stimulus_func):
        """
        ODE 右端项。

        y = [V_m, m, h, n]
        """
        V_m, m, h, n = y
        phi = self.phi

        I_Na = self.g_Na * (m**3) * h * (V_m - self.E_Na)
        I_K = self.g_K * (n**4) * (V_m - self.E_K)
        I_L = self.g_L * (V_m - self.E_L)
        I_stim = stimulus_func(t)

        dVdt = (-I_Na - I_K - I_L + I_stim) / self.C_m
        dmdt = phi * (self.alpha_m(V_m) * (1.0 - m) - self.beta_m(V_m) * m)
        dhdt = phi * (self.alpha_h(V_m) * (1.0 - h) - self.beta_h(V_m) * h)
        dndt = phi * (self.alpha_n(V_m) * (1.0 - n) - self.beta_n(V_m) * n)

        return np.array([dVdt, dmdt, dhdt, dndt])

    def simulate(self, t_span, stimulus_func, V_rest=None, method='RK45',
                 max_step=0.005):
        """
        模拟详细 HH 模型。

        Parameters
        ----------
        t_span : tuple
            (t0, tf) in ms
        stimulus_func : callable
            I_stim(t) in μA/cm²
        V_rest : float or None
            若 None，则使用 E_L
        method : str
        max_step : float

        Returns
        -------
        sol : OdeSolution
        spike_times : list
        """
        if V_rest is None:
            V_rest = self.E_L

        # 稳态门控变量作为初值
        m0 = self.alpha_m(V_rest) / (self.alpha_m(V_rest) + self.beta_m(V_rest))
        h0 = self.alpha_h(V_rest) / (self.alpha_h(V_rest) + self.beta_h(V_rest))
        n0 = self.alpha_n(V_rest) / (self.alpha_n(V_rest) + self.beta_n(V_rest))
        y0 = np.array([V_rest, m0, h0, n0])

        def event_spike(t, y):
            return y[0] - 0.0  # 0 mV 阈值
        event_spike.terminal = False
        event_spike.direction = 1

        sol = solve_ivp(
            lambda t, y: self.derivatives(t, y, stimulus_func),
            t_span, y0, method=method, max_step=max_step,
            events=event_spike, dense_output=True
        )

        spike_times = sol.t_events[0].tolist() if sol.t_events[0] is not None else []
        return sol, spike_times


def biphasic_pulse(t, amplitude, phase_width_ms, interphase_gap_ms=0.05):
    """
    双相脉冲刺激电流波形。

    临床人工耳蜗使用双相脉冲避免电荷累积:
        0 <= t < T_p:    I = +A  (阴相)
        T_p <= t < T_p+T_g: I = 0  (间隙)
        T_p+T_g <= t < 2T_p+T_g: I = -A (阳相)

    Parameters
    ----------
    t : float
        时间 (ms)
    amplitude : float
        阴相幅度 (μA/cm²)
    phase_width_ms : float
        每相宽度 (ms)
    interphase_gap_ms : float
        相间间隙 (ms)

    Returns
    -------
    I : float
    """
    T = 2.0 * phase_width_ms + interphase_gap_ms
    t_mod = t % T
    if t_mod < phase_width_ms:
        return amplitude
    elif t_mod < phase_width_ms + interphase_gap_ms:
        return 0.0
    elif t_mod < T:
        return -amplitude
    return 0.0
