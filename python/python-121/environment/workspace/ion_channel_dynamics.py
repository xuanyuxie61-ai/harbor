"""
ion_channel_dynamics.py
心肌离子通道动力学与随机噪声模块

融入原项目:
- 1152_squircle_ode: Squircle ODE系统的守恒量分析思想
- 201_colored_noise: 1/f^α有色噪声生成
- 369_fd2d_predator_prey: 反应项的有限差分离散化思想

功能:
1. Hodgkin-Huxley类型心肌离子通道门控动力学
2. 1/f^α有色噪声模拟离子通道随机开放
3. Aliev-Panfilov简化模型用于组织电生理模拟

核心科学模型:
- 心肌细胞膜电位方程
- 离子通道门控变量（m, h, n, d, f, x, etc.）
- 随机离子通道动力学
"""

import numpy as np
from math import exp, sqrt, pi, cos, sin


# ============================================================================
# 有色噪声生成（源自 201_colored_noise/f_alpha）
# ============================================================================

def generate_colored_noise(n_samples, q_d, alpha):
    """
    生成 1/f^α 有色噪声序列
    
    算法（基于Kasdin方法）:
    1. 生成滤波器系数 h_k:
       h_0 = 1
       h_k = h_{k-1} * (α/2 + k - 2) / (k - 1), k ≥ 1
    
    2. 生成白噪声 w_k ~ N(0, q_d)
    
    3. 通过频域卷积生成有色噪声:
       X = IFFT(FFT(h) * FFT(w))
    
    功率谱密度:
    S(f) ∝ 1 / |f|^α
    
    参数:
        n_samples: 样本数
        q_d: 噪声方差
        alpha: 谱指数 (0: 白噪声, 1: 粉红噪声, 2: 布朗噪声)
    返回:
        noise: (n_samples,) 有色噪声序列
    """
    if n_samples <= 0 or q_d < 0 or alpha < 0:
        return np.zeros(n_samples)
    
    q_d_sqrt = sqrt(q_d)
    
    # 生成滤波器系数 h_k
    hfa = np.zeros(2 * n_samples)
    hfa[0] = 1.0
    for i in range(1, n_samples):
        hfa[i] = hfa[i - 1] * (0.5 * alpha + (i - 2)) / (i - 1)
    
    # 白噪声
    wfa = np.zeros(2 * n_samples)
    wfa[:n_samples] = np.random.randn(n_samples) * q_d_sqrt
    
    # FFT卷积
    H = np.fft.fft(hfa)
    W = np.fft.fft(wfa)
    X = np.fft.ifft(H * W)
    
    # 取实部并截取前n_samples
    noise = np.real(X[:n_samples])
    
    return noise


def generate_ion_channel_noise(n_channels, dt, T, D_ion=0.01, alpha=1.5):
    """
    生成离子通道随机噪声
    
    离子通道的随机开放/关闭可以用Langevin方程描述:
    dξ(t) = -γξ(t)dt + √(2D) dW(t)
    
    其中:
    - γ: 衰减速率
    - D: 扩散系数
    - W(t): Wiener过程
    
    参数:
        n_channels: 通道类型数
        dt: 时间步长
        T: 总时间
        D_ion: 离子扩散系数
        alpha: 噪声谱指数
    返回:
        noise_array: (n_time_steps, n_channels) 噪声数组
    """
    n_steps = int(T / dt) + 1
    noise_array = np.zeros((n_steps, n_channels))
    
    for c in range(n_channels):
        # 为每种离子通道生成独立的1/f^α噪声
        q_d = D_ion * dt
        noise = generate_colored_noise(n_steps, q_d, alpha)
        noise_array[:, c] = noise
    
    return noise_array


# ============================================================================
# Hodgkin-Huxley类型门控动力学
# ============================================================================

def gate_alpha_beta(v, gate_type):
    """
    计算离子通道门控变量的速率常数 α 和 β
    
    基于Beeler-Reuter或Luo-Rudy模型的心肌细胞参数
    
    门控方程:
    dx/dt = α_x(V) * (1 - x) - β_x(V) * x
    
    稳态值: x_∞ = α_x / (α_x + β_x)
    时间常数: τ_x = 1 / (α_x + β_x)
    
    参数:
        v: 膜电位 (mV)
        gate_type: 门控类型 ('m', 'h', 'j', 'd', 'f', 'x')
    返回:
        alpha, beta: 速率常数
    """
    v = float(v)
    
    if gate_type == 'm':  # Na+ 激活门
        alpha = 0.32 * (v + 47.13) / (1.0 - exp(-0.1 * (v + 47.13)))
        if abs(v + 47.13) < 1e-6:
            alpha = 3.2
        beta = 0.08 * exp(-v / 11.0)
    
    elif gate_type == 'h':  # Na+ 失活门
        alpha = 0.135 * exp(-(v + 80.0) / 6.8)
        beta = 3.56 / (1.0 + exp(-0.1 * (v + 40.0))) + 0.0075
    
    elif gate_type == 'j':  # Na+ 慢失活门
        alpha = (-1.2714e5 * exp(0.2444 * v) - 3.474e-5 * exp(-0.04391 * v)) * (v + 37.78) / (1.0 + exp(0.311 * (v + 79.23)))
        if abs(v + 37.78) < 1e-3:
            alpha = 0.0
        beta = 0.1212 * exp(-0.01052 * v) / (1.0 + exp(-0.1378 * (v + 40.14)))
    
    elif gate_type == 'd':  # Ca2+ L型通道激活门
        alpha = 0.095 * exp(-(v - 5.0) / 13.0) / (1.0 + exp(-(v - 5.0) / 13.0))
        beta = 0.07 * exp(-(v + 44.0) / 20.0) / (1.0 + exp((v + 44.0) / 20.0))
    
    elif gate_type == 'f':  # Ca2+ L型通道失活门
        alpha = 0.012 * exp(-(v + 28.0) / 30.0) / (1.0 + exp((v + 28.0) / 30.0))
        beta = 0.0065 * exp(-(v + 30.0) / 40.0) / (1.0 + exp(-(v + 30.0) / 40.0))
    
    elif gate_type == 'x':  # 延迟整流K+通道
        alpha = 0.0005 * exp(0.083 * (v + 50.0)) / (1.0 + exp(0.057 * (v + 50.0)))
        beta = 0.0013 * exp(-0.06 * (v + 20.0)) / (1.0 + exp(-0.04 * (v + 20.0)))
    
    else:
        alpha = 0.0
        beta = 1.0
    
    # 边界处理
    alpha = max(0.0, alpha)
    beta = max(0.0, beta)
    
    return alpha, beta


def update_gate(gate, v, gate_type, dt):
    """
    更新门控变量（前向欧拉）
    
    dx/dt = α(1-x) - βx
    x_{n+1} = x_n + dt * [α(1-x_n) - βx_n]
            = x_n + dt * (α - (α+β)*x_n)
    
    或使用解析解:
    x_{n+1} = x_∞ + (x_n - x_∞) * exp(-dt/τ)
    """
    alpha, beta = gate_alpha_beta(v, gate_type)
    tau = 1.0 / (alpha + beta + 1e-12)
    x_inf = alpha / (alpha + beta + 1e-12)
    
    # 解析积分（更稳定）
    gate_new = x_inf + (gate - x_inf) * exp(-dt / tau)
    
    # 边界截断
    gate_new = max(0.0, min(1.0, gate_new))
    
    return gate_new


# ============================================================================
# 离子电流计算
# ============================================================================

def compute_ionic_currents(v, gates, ion_noise=None):
    """
    计算心肌细胞离子电流
    
    总离子电流:
    I_ion = I_Na + I_Ca + I_K + I_K1 + I_Kp + I_b
    
    各电流公式:
    I_Na = G_Na * m³ * h * j * (V - E_Na)
    I_Ca = G_Ca * d * f * (V - E_Ca)
    I_K = G_K * x * x_i * (V - E_K)
    
    参数:
        v: 膜电位 (mV)
        gates: 门控变量字典 {'m':..., 'h':..., 'j':..., 'd':..., 'f':..., 'x':...}
        ion_noise: 离子通道噪声（可选）
    返回:
        currents: 各电流字典
        I_total: 总离子电流
    """
    # 标准 reversal potentials (mV)
    E_Na = 54.4
    E_Ca = 130.0  # 或基于Nernst方程动态计算
    E_K = -87.0
    
    # 最大电导 (mS/cm²)
    G_Na = 23.0
    G_Ca = 0.09
    G_K = 0.282  # 延迟整流
    G_K1 = 0.6047  # 内向整流
    G_Kp = 0.0183  # 平台期K+电流
    G_b = 0.03921  # 背景电流
    
    m = gates.get('m', 0.0)
    h = gates.get('h', 0.0)
    j = gates.get('j', 0.0)
    d = gates.get('d', 0.0)
    f = gates.get('f', 0.0)
    x = gates.get('x', 0.0)
    
    # Na+ 电流
    I_Na = G_Na * (m ** 3) * h * j * (v - E_Na)
    
    # Ca2+ L型电流
    I_Ca = G_Ca * d * f * (v - E_Ca)
    
    # K+ 延迟整流电流
    # xi (激活门，简化处理)
    xi = 1.0 / (1.0 + exp((v - 56.26) / 32.1))
    I_K = G_K * x * xi * (v - E_K)
    
    # K+ 内向整流电流
    alpha_K1 = 1.02 / (1.0 + exp(0.2385 * (v - E_K - 59.215)))
    beta_K1 = (0.49124 * exp(0.08032 * (v - E_K + 5.476)) + exp(0.06175 * (v - E_K - 594.31))) / (1.0 + exp(-0.5143 * (v - E_K + 4.753)))
    x_K1 = alpha_K1 / (alpha_K1 + beta_K1 + 1e-12)
    I_K1 = G_K1 * x_K1 * (v - E_K)
    
    # 平台期K+电流
    x_Kp = 1.0 / (1.0 + exp((7.488 - v) / 5.98))
    I_Kp = G_Kp * x_Kp * (v - E_K)
    
    # 背景电流
    E_b = -59.87
    I_b = G_b * (v - E_b)
    
    # 添加离子通道噪声
    noise_factor = 1.0
    if ion_noise is not None:
        noise_factor = 1.0 + 0.05 * ion_noise
    
    I_total = (I_Na + I_Ca + I_K + I_K1 + I_Kp + I_b) * noise_factor
    
    currents = {
        'I_Na': I_Na,
        'I_Ca': I_Ca,
        'I_K': I_K,
        'I_K1': I_K1,
        'I_Kp': I_Kp,
        'I_b': I_b,
        'I_total': I_total
    }
    
    return currents, I_total


# ============================================================================
# Aliev-Panfilov 简化模型（组织层面）
# ============================================================================

def aliev_panfilov_reaction(u, v, a=0.1, k=8.0, mu1=0.2, mu2=0.3, eps=0.002):
    """
    Aliev-Panfilov 反应项
    
    模型方程:
    ∂u/∂t = D∇²u - k*u*(u-a)*(u-1) - u*v
    ∂v/∂t = ε(u) * (-v - k*u*(u-a-1))
    
    其中:
    - u: 无量纲化膜电位 (0 = 静息, 1 = 去极化)
    - v: 恢复变量
    - a: 阈值参数 (≈0.1)
    - k: 非线性强度
    - ε(u) = ε0 + μ1*v/(u+μ2): 恢复变量速率
    
    反应项 f(u,v) = -k*u*(u-a)*(u-1) - u*v
    反应项 g(u,v) = ε(u) * (-v - k*u*(u-a-1))
    
    参数:
        u: 膜电位场
        v: 恢复变量场
        a, k, mu1, mu2: 模型参数
        eps: 基础恢复速率
    返回:
        f: du/dt 反应项
        g: dv/dt 反应项
    """
    # TODO: Hole 1 - 实现 Aliev-Panfilov 反应项
    # 需要计算:
    #   f(u,v) = -k*u*(u-a)*(u-1) - u*v
    #   eps(u) = eps + mu1*v/(u+mu2)
    #   g(u,v) = eps(u) * (-v - k*u*(u-a-1))
    # 注意边界处理和对 u, v 的 clip
    raise NotImplementedError("Hole 1: aliev_panfilov_reaction 待实现")


def single_cell_ap_model(t_max=500.0, dt=0.01, stim_period=300.0):
    """
    单细胞动作电位模拟
    
    膜电位方程:
    C_m * dV/dt = -I_ion + I_stim
    
    其中:
    - C_m = 1 μF/cm²: 膜电容
    - I_stim: 刺激电流
    
    参数:
        t_max: 最大模拟时间 (ms)
        dt: 时间步长 (ms)
        stim_period: 刺激周期 (ms)
    返回:
        t: 时间数组
        v: 膜电位轨迹
        gates: 门控变量历史
    """
    C_m = 1.0  # μF/cm²
    n_steps = int(t_max / dt) + 1
    
    t = np.linspace(0, t_max, n_steps)
    v = np.zeros(n_steps)
    
    # 初始条件
    v[0] = -86.2  # mV (静息电位)
    gates = {
        'm': 0.0,
        'h': 1.0,
        'j': 1.0,
        'd': 0.0,
        'f': 1.0,
        'x': 0.0
    }
    
    # 记录门控变量历史
    gate_history = {k: np.zeros(n_steps) for k in gates}
    for k in gates:
        gate_history[k][0] = gates[k]
    
    for i in range(1, n_steps):
        # 刺激电流（起搏刺激）
        I_stim = 0.0
        if (t[i] % stim_period) < 2.0:  # 2ms刺激
            I_stim = -80.0  # μA/cm²
        
        # 更新门控变量
        for g_name in gates:
            gates[g_name] = update_gate(gates[g_name], v[i - 1], g_name, dt)
            gate_history[g_name][i] = gates[g_name]
        
        # 计算离子电流
        _, I_ion = compute_ionic_currents(v[i - 1], gates)
        
        # 更新膜电位（前向欧拉）
        dv = (-I_ion + I_stim) / C_m
        v[i] = v[i - 1] + dt * dv
        
        # 数值稳定性处理
        v[i] = max(-100.0, min(60.0, v[i]))
    
    return t, v, gate_history


# ============================================================================
# Squircle ODE守恒量思想（源自 1152_squircle_ode）
# ============================================================================

def squircle_ode_integrate(y0, t_span, s=4.0, n_steps=1000):
    """
    Squircle ODE系统积分
    
    方程:
    du/dt = v^{s-1}
    dv/dt = -u^{s-1}
    
    守恒量:
    H(u,v) = (u^s + v^s) / s = const
    
    该ODE系统可用于测试数值积分器的守恒性质
    在心脏模型中，类似的守恒量出现在某些简化模型中
    
    参数:
        y0: (u0, v0) 初始条件
        t_span: (t0, t1) 时间范围
        s: squircle参数
        n_steps: 步数
    返回:
        t, u, v: 时间序列
        H: 守恒量历史
    """
    t0, t1 = t_span
    dt = (t1 - t0) / n_steps
    
    t = np.linspace(t0, t1, n_steps + 1)
    u = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)
    H = np.zeros(n_steps + 1)
    
    u[0], v[0] = y0
    H[0] = (abs(u[0]) ** s + abs(v[0]) ** s) / s
    
    for i in range(n_steps):
        # 隐式中点法（保辛）
        # 简化为显式欧拉用于测试
        ui, vi = u[i], v[i]
        
        # 防止除零
        u_pow = np.sign(ui) * (abs(ui) ** (s - 1)) if ui != 0 else 0.0
        v_pow = np.sign(vi) * (abs(vi) ** (s - 1)) if vi != 0 else 0.0
        
        u[i + 1] = ui + dt * v_pow
        v[i + 1] = vi - dt * u_pow
        
        H[i + 1] = (abs(u[i + 1]) ** s + abs(v[i + 1]) ** s) / s
    
    return t, u, v, H
