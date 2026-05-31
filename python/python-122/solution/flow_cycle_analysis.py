"""
脑血流动力学 — 血流周期性与循环检测模块

基于 cycle_brent（Brent 循环检测算法），检测脑血管网络中的周期性血流振荡、
心率谐波与病理状态（如脑血管痉挛）引发的异常周期模式。

科学背景:
- 正常脑血流具有心搏周期性（约 1 Hz），可用迭代映射 x_{n+1} = f(x_n) 建模。
- 在病理状态（如蛛网膜下腔出血后的血管痉挛）中，血流可能呈现异常周期或
  准周期振荡。
- Brent 算法可在 O(μ + λ) 时间内检测迭代序列中的周期，其中 μ 为进入周期
  前的步数，λ 为周期长度。
- 脑血管自动调节可建模为离散动力学系统:
    CBF_{n+1} = f(CBF_n, MAP_n, PaCO2_n)
  其中 CBF 为脑血流量，MAP 为平均动脉压，PaCO2 为动脉血二氧化碳分压。
"""

import numpy as np


def cycle_brent(f, x0):
    """
    Brent 循环检测算法。
    对于迭代映射 x_{n+1} = f(x_n)，检测首次进入循环的位置 μ 与周期长度 λ。

    算法步骤:
        1. 初始化: power=1, λ=1, tortoise=x0, hare=f(x0)
        2. while tortoise != hare:
               if power == λ: tortoise = hare; power *= 2; λ = 0
               hare = f(hare); λ += 1
        3. 寻找 μ: tortoise = x0; hare = x0; 将 hare 前进 λ 步
        4. 同步前进 tortoise 与 hare，直到相遇

    返回: (lam, mu)
    """
    power = 1
    lam = 1
    tortoise = x0
    hare = f(x0)

    while tortoise != hare:
        if power == lam:
            tortoise = hare
            power *= 2
            lam = 0
        hare = f(hare)
        lam += 1

    mu = 0
    tortoise = x0
    hare = x0
    for _ in range(lam):
        hare = f(hare)

    while tortoise != hare:
        tortoise = f(tortoise)
        hare = f(hare)
        mu += 1

    return lam, mu


def cerebrovascular_autoregulation_map(cbf, params):
    """
    脑血管自动调节离散映射。
    建模为:
        CBF_{n+1} = CBF_n + k1 * (MAP - MAP_ss) / MAP_ss - k2 * (CBF_n - CBF_ss) / CBF_ss
                  - k3 * sin(2π f_heart n dt)

    参数:
        cbf: 当前脑血流量 [mL/100g/min]
        params: 参数字典
    """
    MAP = params['MAP']
    MAP_ss = params['MAP_ss']
    CBF_ss = params['CBF_ss']
    k1 = params.get('k1', 0.1)
    k2 = params.get('k2', 0.2)
    k3 = params.get('k3', 2.0)
    f_heart = params.get('f_heart', 1.17)  # Hz (70 bpm)
    dt = params.get('dt', 0.01)
    n = params.get('step', 0)

    delta = k1 * (MAP - MAP_ss) / (MAP_ss + 1e-14) - k2 * (cbf - CBF_ss) / (CBF_ss + 1e-14)
    cardiac = -k3 * np.sin(2.0 * np.pi * f_heart * n * dt)
    cbf_new = cbf + delta + cardiac
    return max(cbf_new, 0.0)


def detect_hemodynamic_cycles(cbf_series, params):
    """
    检测脑血流时间序列中的周期性。
    将时间序列映射为离散状态后使用 Brent 算法检测循环。
    """
    # 将连续 CBF 值离散化为整数状态（保留 2 位小数）
    states = np.round(np.asarray(cbf_series, dtype=float) * 100).astype(int)

    if len(states) < 2:
        return None, None, []

    # 构建有限状态转移
    unique_states = list(sorted(set(states)))
    state_to_idx = {s: i for i, s in enumerate(unique_states)}
    n_unique = len(unique_states)

    # 构建转移函数表
    transitions = {}
    for i in range(len(states) - 1):
        s = states[i]
        s_next = states[i + 1]
        if s not in transitions:
            transitions[s] = s_next

    def f(x):
        return transitions.get(x, x)

    if len(transitions) < 2:
        return None, None, states.tolist()

    x0 = states[0]
    try:
        lam, mu = cycle_brent(f, x0)
    except (RuntimeError, RecursionError):
        lam, mu = None, None

    return lam, mu, states.tolist()


def analyze_frequency_content(signal, dt):
    """
    使用 FFT 分析血流信号的频谱成分。
    返回主要频率与对应幅值。
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    if n < 2:
        return np.array([]), np.array([])
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=dt)
    amps = np.abs(fft_vals) / n
    return freqs, amps


def classify_flow_regime(lam, mu, freqs, amps, dt):
    """
    基于周期检测结果与频谱分析对血流状态分类:
        - normal: 检测到心搏周期 (~1 Hz)
        - arrhythmic: 无显著周期
        - pathological: 检测到异常低频或高频振荡
    """
    if freqs is None or len(freqs) == 0:
        return 'insufficient_data'

    # 寻找主频
    if len(amps) > 1:
        dominant_idx = np.argmax(amps[1:]) + 1  # 忽略直流分量
        dominant_freq = freqs[dominant_idx]
        dominant_amp = amps[dominant_idx]
    else:
        dominant_freq = 0.0
        dominant_amp = 0.0

    if lam is None or mu is None:
        if 0.8 <= dominant_freq <= 1.5 and dominant_amp > 0.05 * np.max(amps):
            return 'normal_cardiac'
        return 'arrhythmic'

    period = lam * dt
    if 0.6 <= 1.0 / (period + 1e-14) <= 1.5:
        return 'normal_cardiac'
    elif 1.0 / (period + 1e-14) < 0.1:
        return 'pathological_slow_oscillation'
    elif 1.0 / (period + 1e-14) > 3.0:
        return 'pathological_fast_oscillation'
    return 'uncertain'
