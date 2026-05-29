"""
thermoacoustic_oscillator.py - 热声耦合振荡器模型
================================================
基于非线性振荡器的火箭发动机燃烧不稳定预测。

原项目映射:
- 840_oscillator_ode -> 非线性振荡器描述压力脉动的自激振荡

科学背景:
=========
燃烧不稳定性的时域演化可用非线性振荡器模型描述。

简化的热声振荡器方程 (Rayleigh准则的非线性推广):
    d²p'/dt² + 2·α·dp'/dt + ω_0²·p' = γ·f(p', dp'/dt)

其中:
    p': 压力脉动
    α: 阻尼系数 (包含声学阻尼和增益)
    ω_0: 声学共振频率
    γ: 非线性耦合强度
    f: 火焰响应非线性函数

对于自激振荡系统，当Rayleigh增益超过声学阻尼时:
    α_eff = α_acoustic - α_gain < 0  =>  不稳定

极限环振幅由非线性饱和决定:
    |p'|_max ~ √(-α_eff / β)
    
    其中 β 为Van der Pol型非线性饱和系数

典型的非线性振荡器形式 (修正的Van der Pol):
    d²x/dt² - μ·(1 - x²)·dx/dt + ω²·x = 0

其中 μ 控制线性增长率 (μ > 0 不稳定)。
"""

import numpy as np
from utils import safe_divide, robust_sqrt, check_finite_array


class ThermoacousticOscillator:
    """
    热声耦合非线性振荡器。
    
    状态变量:
        y[0] = p'  (压力脉动)
        y[1] = dp'/dt  (压力脉动速率)
    
    状态方程:
        dy[0]/dt = y[1]
        dy[1]/dt = -2·α·y[1] - ω²·y[0] + γ·F(y[0], y[1])
    
    其中 F 为火焰响应函数，包含线性和非线性部分:
        F(p', dp'/dt) = n·dp'/dt - β·p'²·dp'/dt
    
    整理后:
        dy[1]/dt = -(ω²)·y[0] + (-2α + γ·n)·y[1] - γ·β·y[0]²·y[1]
    """
    
    def __init__(self,
                 natural_frequency_hz: float = 500.0,
                 acoustic_damping: float = 50.0,  # 1/s
                 flame_gain_coefficient: float = 80.0,  # 1/s
                 nonlinear_saturation: float = 1.0e8,  # 1/(Pa^2·s)
                 coupling_strength: float = 1.0,
                 initial_pressure_disturbance_pa: float = 100.0,
                 initial_velocity_disturbance_pa_s: float = 0.0):
        
        self.omega = 2.0 * np.pi * natural_frequency_hz
        self.alpha_acoustic = acoustic_damping
        self.n = flame_gain_coefficient / coupling_strength
        self.beta = nonlinear_saturation
        self.gamma = coupling_strength
        
        # 有效阻尼
        self.alpha_eff = self.alpha_acoustic - self.gamma * self.n
        
        self.y0 = np.array([
            initial_pressure_disturbance_pa,
            initial_velocity_disturbance_pa_s
        ])
        
        # 线性稳定性判断
        self.is_unstable = self.alpha_eff < 0
    
    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        计算状态导数 dy/dt。
        
        参数:
            t: 时间
            y: 状态向量 [p', dp'/dt]
        
        返回:
            dy/dt = [dp'/dt, d²p'/dt²]
        """
        p = y[0]
        dpdt = y[1]
        
        # 非线性火焰响应 (加入饱和保护)
        p_sat = np.clip(p, -1e6, 1e6)
        dpdt_sat = np.clip(dpdt, -1e9, 1e9)
        F = self.n * dpdt_sat - self.beta * p_sat ** 2 * dpdt_sat
        F = np.clip(F, -1e12, 1e12)
        
        # 加速度
        d2pdt2 = -self.omega ** 2 * p_sat - 2.0 * self.alpha_acoustic * dpdt_sat + self.gamma * F
        
        return np.array([dpdt, d2pdt2])
    
    def rk4_integrate(self, t_span: tuple, n_steps: int = 10000) -> dict:
        """
        使用四阶Runge-Kutta积分求解振荡器。
        
        RK4算法:
            k1 = h·f(t_n, y_n)
            k2 = h·f(t_n + h/2, y_n + k1/2)
            k3 = h·f(t_n + h/2, y_n + k2/2)
            k4 = h·f(t_n + h, y_n + k3)
            y_{n+1} = y_n + (k1 + 2k2 + 2k3 + k4)/6
        """
        t0, tf = t_span
        dt = (tf - t0) / n_steps
        
        t = np.linspace(t0, tf, n_steps + 1)
        y = np.zeros((n_steps + 1, 2))
        y[0] = self.y0
        
        for i in range(n_steps):
            k1 = dt * self.derivatives(t[i], y[i])
            k2 = dt * self.derivatives(t[i] + 0.5 * dt, y[i] + 0.5 * k1)
            k3 = dt * self.derivatives(t[i] + 0.5 * dt, y[i] + 0.5 * k2)
            k4 = dt * self.derivatives(t[i] + dt, y[i] + k3)
            
            y[i + 1] = y[i] + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            
            # 数值稳定性保护
            if not np.all(np.isfinite(y[i+1])):
                y[i+1] = y[i]
        
        return {
            "t": t,
            "pressure": y[:, 0],
            "pressure_rate": y[:, 1],
            "amplitude_envelope": self._compute_amplitude_envelope(y[:, 0], dt)
        }
    
    def _compute_amplitude_envelope(self, signal: np.ndarray, dt: float) -> np.ndarray:
        """
        使用Hilbert变换思想计算信号包络 (简化版)。
        
        使用滑动窗口峰值检测。
        """
        n = len(signal)
        envelope = np.zeros(n)
        window = int(0.5 * 2.0 * np.pi / (self.omega * dt))  # 半周期窗口
        window = max(window, 3)
        
        for i in range(n):
            i0 = max(0, i - window)
            i1 = min(n, i + window + 1)
            envelope[i] = np.max(np.abs(signal[i0:i1]))
        
        return envelope
    
    def compute_growth_rate(self, t: np.ndarray, amplitude: np.ndarray) -> float:
        """
        从振幅包络计算线性增长率。
        
        假设指数增长/衰减:
            A(t) = A_0·exp(σ·t)
            => ln(A) = ln(A_0) + σ·t
        
        返回:
            σ: 增长率 (1/s), σ > 0 不稳定
        """
        # 取对数并线性拟合
        amp_safe = np.maximum(np.abs(amplitude), 1e-10)
        amp_safe = np.clip(amp_safe, 1e-10, 1e10)
        log_amp = np.log(amp_safe)
        
        # 只使用前半段 (线性阶段)
        n_use = len(t) // 3
        if n_use < 3:
            return 0.0
        
        # 最小二乘拟合
        t_use = t[:n_use]
        log_use = log_amp[:n_use]
        
        # 检查数据有效性
        if not np.all(np.isfinite(log_use)):
            return 0.0
        
        A = np.vstack([t_use, np.ones(len(t_use))]).T
        try:
            sigma, _ = np.linalg.lstsq(A, log_use, rcond=None)[0]
        except Exception:
            sigma = 0.0
        
        sigma = float(np.clip(sigma, -1e6, 1e6))
        return sigma
    
    def limit_cycle_amplitude(self) -> float:
        """
        估计极限环振幅 (稳态振荡幅度)。
        
        对于Van der Pol型振荡器:
            d²x/dt² - μ·(1 - x²/A²)·dx/dt + ω²·x = 0
        
        极限环振幅:
            A_lim = √(-4·α_eff / (γ·β))
        
        当 α_eff < 0 (不稳定) 时成立。
        """
        # TODO(Hole_2): 实现极限环振幅估计
        # 科学知识: Van der Pol型非线性振荡器的极限环振幅公式
        # 当有效阻尼 α_eff < 0 (不稳定) 时, 系统存在极限环
        # 极限环振幅 A_lim 由非线性饱和项与线性增长率的平衡决定
        # 对于修正Van der Pol方程: d²x/dt² - μ(1-x²/A²)dx/dt + ω²x = 0
        # 需要推导并返回正确的 A_lim 表达式
        # 提示: 考虑 α_eff, γ (耦合强度), β (饱和系数) 之间的关系
        return 0.0
    
    def compute_oscillation_metrics(self, t: np.ndarray, pressure: np.ndarray) -> dict:
        """
        计算振荡特征指标。
        
        指标:
            1. 峰值压力脉动 (Peak-to-Peak)
            2. RMS压力脉动
            3. 主频 (通过零交叉计数)
            4. 增长/衰减率
            5. 极限环判断
        """
        p = pressure
        
        # 峰值
        p_max = np.max(p)
        p_min = np.min(p)
        p_p2p = p_max - p_min
        
        # RMS (去除直流分量)
        p_ac = p - np.mean(p)
        p_rms = np.sqrt(np.mean(p_ac ** 2))
        
        # 零交叉频率估计
        zero_crossings = np.where(np.diff(np.sign(p_ac)))[0]
        if len(zero_crossings) >= 2:
            T_est = 2.0 * (t[zero_crossings[-1]] - t[zero_crossings[0]]) / len(zero_crossings)
            f_est = safe_divide(1.0, T_est, default=0.0)
        else:
            f_est = self.omega / (2.0 * np.pi)
        
        # 包络增长率
        envelope = self._compute_amplitude_envelope(p, t[1] - t[0])
        growth_rate = self.compute_growth_rate(t, envelope)
        
        # 极限环判断
        A_lim = self.limit_cycle_amplitude()
        in_limit_cycle = abs(p_rms - A_lim / np.sqrt(2)) < 0.1 * A_lim if A_lim > 0 else False
        
        return {
            "peak_to_peak_pa": float(p_p2p),
            "rms_pa": float(p_rms),
            "estimated_frequency_hz": float(f_est),
            "growth_rate_1_per_s": float(growth_rate),
            "limit_cycle_amplitude_pa": float(A_lim),
            "in_limit_cycle": bool(in_limit_cycle)
        }


class MultiModeThermoacousticSystem:
    """
    多模态热声耦合系统。
    
    耦合多个声学模态的非线性动力学:
        d²p_n/dt² + 2·α_n·dp_n/dt + ω_n²·p_n = Σ_m γ_{nm}·F_m(p, dp/dt)
    
    其中耦合矩阵 γ_{nm} 描述模态间的能量传递。
    """
    
    def __init__(self,
                 mode_frequencies: np.ndarray,
                 damping_rates: np.ndarray,
                 coupling_matrix: np.ndarray = None):
        
        self.n_modes = len(mode_frequencies)
        self.omega = 2.0 * np.pi * np.array(mode_frequencies)
        self.alpha = np.array(damping_rates)
        
        if coupling_matrix is None:
            self.gamma = np.eye(self.n_modes) * 0.5
        else:
            self.gamma = np.array(coupling_matrix)
    
    def derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        状态向量: [p_1, v_1, p_2, v_2, ..., p_N, v_N]
        其中 v_n = dp_n/dt
        """
        n = self.n_modes
        dydt = np.zeros(2 * n)
        
        for i in range(n):
            p_i = state[2 * i]
            v_i = state[2 * i + 1]
            
            # 耦合火焰响应
            flame_response = 0.0
            for j in range(n):
                p_j = state[2 * j]
                v_j = state[2 * j + 1]
                # 简化耦合: 速度反馈
                flame_response += self.gamma[i, j] * v_j * (1.0 - 1e-8 * p_j ** 2)
            
            dydt[2 * i] = v_i
            dydt[2 * i + 1] = -self.omega[i] ** 2 * p_i - 2.0 * self.alpha[i] * v_i + flame_response
        
        return dydt
    
    def integrate(self, t_span: tuple = (0, 0.05), n_steps: int = 20000,
                  initial_conditions: np.ndarray = None) -> dict:
        """
        积分多模态系统。
        """
        if initial_conditions is None:
            y0 = np.zeros(2 * self.n_modes)
            y0[0] = 100.0  # 第一模态初始扰动
        else:
            y0 = np.array(initial_conditions)
        
        t0, tf = t_span
        dt = (tf - t0) / n_steps
        
        t = np.linspace(t0, tf, n_steps + 1)
        y = np.zeros((n_steps + 1, 2 * self.n_modes))
        y[0] = y0
        
        for i in range(n_steps):
            k1 = dt * self.derivatives(t[i], y[i])
            k2 = dt * self.derivatives(t[i] + 0.5 * dt, y[i] + 0.5 * k1)
            k3 = dt * self.derivatives(t[i] + 0.5 * dt, y[i] + 0.5 * k2)
            k4 = dt * self.derivatives(t[i] + dt, y[i] + k3)
            
            y[i + 1] = y[i] + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            
            if not np.all(np.isfinite(y[i+1])):
                y[i+1] = y[i]
        
        return {
            "t": t,
            "state": y,
            "mode_pressures": [y[:, 2*i] for i in range(self.n_modes)]
        }


if __name__ == "__main__":
    # 单模态振荡器
    osc = ThermoacousticOscillator(
        natural_frequency_hz=500.0,
        acoustic_damping=50.0,
        flame_gain_coefficient=80.0,
        nonlinear_saturation=1e8
    )
    print(f"Effective damping: {osc.alpha_eff:.2f} 1/s")
    print(f"Linear stability: {'UNSTABLE' if osc.is_unstable else 'STABLE'}")
    print(f"Limit cycle amplitude: {osc.limit_cycle_amplitude():.2f} Pa")
    
    result = osc.rk4_integrate((0, 0.05), n_steps=20000)
    metrics = osc.compute_oscillation_metrics(result["t"], result["pressure"])
    print(f"\nOscillation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    # 多模态系统
    freqs = np.array([500.0, 1500.0, 2500.0])
    damping = np.array([50.0, 150.0, 250.0])
    multi = MultiModeThermoacousticSystem(freqs, damping)
    multi_result = multi.integrate()
    print(f"\nMulti-mode system integrated.")
    print(f"Mode 1 final amplitude: {np.max(np.abs(multi_result['mode_pressures'][0][-1000:])):.2f} Pa")
