"""
超声阵列波束形成与动态聚焦模块

基于多通道超声阵列信号处理，实现高分辨率波束形成与
自适应动态聚焦，是超声层析成像的核心信号处理单元。

物理模型:
对于 N 阵元线性阵列，第 n 个阵元接收到的回波信号为:
    s_n(t) = A·p(t - τ_n)
其中 τ_n = 2r_n/c 为往返延迟，r_n 为散射点到第n个阵元的距离。

延迟叠加波束形成（Delay-and-Sum, DAS）:
    b(t,θ) = Σ w_n · s_n(t - Δτ_n(θ))
其中 Δτ_n 为聚焦时延，w_n 为阵元加权（如Hanning窗、Hamming窗）。

动态聚焦（Dynamic Focusing）:
在接收过程中，随着深度增加动态调整聚焦延迟，
实现全深度范围内的最佳分辨率。

阵元加权（Apodization）:
- 矩形窗: w_n = 1
- Hanning窗: w_n = 0.5·(1 - cos(2πn/(N-1)))
- Hamming窗: w_n = 0.54 - 0.46·cos(2πn/(N-1))
- 切比雪夫窗: 等旁瓣电平

波束宽度（-6dB）:
    θ_{3dB} ≈ 0.886·λ / (N·d)
其中 λ 为波长，d 为阵元间距，N 为阵元数。

栅瓣条件:
    d > λ/2 时出现栅瓣（空间混叠）
"""

import numpy as np
from typing import Tuple, Optional


SOUND_SPEED_TISSUE = 1540.0  # m/s，生物软组织平均声速


def array_steering_vector(n_elements: int, element_spacing: float,
                          angle: float, frequency: float,
                          c: float = SOUND_SPEED_TISSUE) -> np.ndarray:
    """计算阵列的导向矢量（steering vector）。
    
    对于线性阵列，第n个阵元相对于参考点的相位延迟:
        φ_n = -2πf/c · n·d·sin(θ)
    
    导向矢量:
        a(θ) = [exp(jφ_0), exp(jφ_1), ..., exp(jφ_{N-1})]ᵀ
    
    参数:
        n_elements: 阵元数
        element_spacing: 阵元间距 (m)
        angle: 波束指向角 (rad)
        frequency: 频率 (Hz)
        c: 声速 (m/s)
    
    返回:
        steering_vector: (n_elements,) 复数导向矢量
    """
    k = 2.0 * np.pi * frequency / c
    n = np.arange(n_elements)
    phases = -k * n * element_spacing * np.sin(angle)
    return np.exp(1j * phases)


def hanning_window(n_elements: int) -> np.ndarray:
    """Hanning窗（余弦平方窗）。
    
    公式: w[n] = 0.5·(1 - cos(2πn/(N-1)))
    
    旁瓣衰减: -31 dB
    主瓣宽度: 2.0 × (λ/L)
    """
    n = np.arange(n_elements)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (n_elements - 1)))


def hamming_window(n_elements: int) -> np.ndarray:
    """Hamming窗。
    
    公式: w[n] = 0.54 - 0.46·cos(2πn/(N-1))
    
    旁瓣衰减: -41 dB
    主瓣宽度: 2.2 × (λ/L)
    """
    n = np.arange(n_elements)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (n_elements - 1))


def chebyshev_window(n_elements: int, sidelobe_level: float = -40.0) -> np.ndarray:
    """切比雪夫窗（Dolph-Chebyshev）。
    
    在阵列波束形成中，切比雪夫窗提供等旁瓣电平，
    最大旁瓣电平由参数 sidelobe_level (dB) 控制。
    
    数学上基于切比雪夫多项式 T_{N-1}(x)。
    """
    # 简化实现：使用近似方法
    # 更精确的实现需要求解多项式零点
    beta = np.cosh(np.arccosh(10**(-sidelobe_level / 20.0)) / (n_elements - 1))
    
    n = np.arange(n_elements)
    x = np.cos(np.pi * n / (n_elements - 1))
    
    # 近似切比雪夫窗
    window = np.zeros(n_elements)
    for i in range(n_elements):
        if abs(x[i]) <= 1.0 / beta:
            window[i] = np.cos((n_elements - 1) * np.arccos(beta * x[i]))
        else:
            window[i] = np.cosh((n_elements - 1) * np.arccosh(beta * abs(x[i])))
    
    window = window / np.max(window)
    return window


def delay_and_sum_beamforming(channel_data: np.ndarray,
                               sampling_rate: float,
                               element_positions: np.ndarray,
                               focus_depths: np.ndarray,
                               steering_angle: float = 0.0,
                               window_type: str = 'hanning',
                               c: float = SOUND_SPEED_TISSUE) -> np.ndarray:
    """延迟叠加波束形成（DAS）。

    算法流程:
    1. 对每个聚焦深度 z，计算各阵元到聚焦点的传播距离
    2. 计算对应的时间延迟 τ_n = r_n / c
    3. 对通道信号进行插值和延迟对齐
    4. 加权求和

    参数:
        channel_data: (n_elements, n_samples) 阵元接收信号
        sampling_rate: 采样率 (Hz)
        element_positions: (n_elements,) 阵元x坐标 (m)
        focus_depths: (n_depths,) 聚焦深度数组 (m)
        steering_angle: 波束偏转角 (rad)
        window_type: 窗函数类型
        c: 声速 (m/s)

    返回:
        beamformed: (n_depths,) 波束形成后的信号幅值
    """
    n_elements, n_samples = channel_data.shape
    n_depths = len(focus_depths)
    dt = 1.0 / sampling_rate

    # 阵元加权
    if window_type == 'hanning':
        weights = hanning_window(n_elements)
    elif window_type == 'hamming':
        weights = hamming_window(n_elements)
    elif window_type == 'chebyshev':
        weights = chebyshev_window(n_elements)
    else:
        weights = np.ones(n_elements)

    beamformed = np.zeros(n_depths, dtype=complex)
    time_axis = np.arange(n_samples) * dt

    for d_idx, z in enumerate(focus_depths):
        # 聚焦点坐标（考虑偏转）
        focus_x = z * np.sin(steering_angle)
        focus_z = z * np.cos(steering_angle)

        summed_signal = np.zeros(n_samples, dtype=complex)

        for elem_idx in range(n_elements):
            # 阵元到聚焦点的距离
            dx = focus_x - element_positions[elem_idx]
            dz = focus_z
            distance = np.sqrt(dx**2 + dz**2)

            # 单程传播时间（发射和接收各一次）
            delay = 2.0 * distance / c

            # 线性插值延迟
            delay_samples = delay / dt
            int_delay = int(delay_samples)
            frac_delay = delay_samples - int_delay

            # 边界检查
            if int_delay >= n_samples - 1:
                continue

            # 线性插值对齐
            shifted = np.zeros(n_samples, dtype=complex)
            for t_idx in range(n_samples):
                src_idx = t_idx + int_delay
                if src_idx < n_samples - 1:
                    shifted[t_idx] = ((1.0 - frac_delay) * channel_data[elem_idx, src_idx] +
                                      frac_delay * channel_data[elem_idx, src_idx + 1])
                elif src_idx < n_samples:
                    shifted[t_idx] = channel_data[elem_idx, src_idx]

            summed_signal += weights[elem_idx] * shifted

        # 取深度点对应时间（往返时间 ≈ 2z/c）的幅值
        time_idx = int(2.0 * z / (c * dt))
        time_idx = min(time_idx, n_samples - 1)
        beamformed[d_idx] = summed_signal[time_idx]

    return np.abs(beamformed)


def transmit_focus_delay(n_elements: int, element_spacing: float,
                         focus_depth: float, steering_angle: float = 0.0,
                         c: float = SOUND_SPEED_TISSUE) -> np.ndarray:
    """计算发射聚焦延迟。

    各阵元发射时间延迟（以使波前在聚焦点同时到达）:
        τ_n = (r_max - r_n) / c
    其中 r_n = √(z² + (x_n - x_focus)²) 为阵元到聚焦点距离。

    参数:
        n_elements: 阵元数
        element_spacing: 阵元间距 (m)
        focus_depth: 聚焦深度 (m)
        steering_angle: 偏转角 (rad)
        c: 声速 (m/s)

    返回:
        delays: (n_elements,) 发射延迟 (s)
    """
    x_elements = (np.arange(n_elements) - (n_elements - 1) / 2.0) * element_spacing
    focus_x = focus_depth * np.sin(steering_angle)
    focus_z = focus_depth * np.cos(steering_angle)

    distances = np.sqrt((x_elements - focus_x)**2 + focus_z**2)
    max_distance = np.max(distances)
    delays = (max_distance - distances) / c

    return delays


def beam_pattern(n_elements: int, element_spacing: float,
                 frequency: float, angles: np.ndarray,
                 window_type: str = 'hanning',
                 c: float = SOUND_SPEED_TISSUE) -> np.ndarray:
    """计算阵列的波束方向图（Beam Pattern）。

    归一化波束方向图:
        B(θ) = |Σ w_n · exp(j·2πn·d/λ·(sin(θ) - sin(θ₀)))| / max|B(θ)|

    参数:
        n_elements: 阵元数
        element_spacing: 阵元间距 (m)
        frequency: 频率 (Hz)
        angles: 角度数组 (rad)
        window_type: 窗函数类型
        c: 声速 (m/s)

    返回:
        pattern: (len(angles),) 归一化波束方向图 (dB)
    """
    wavelength = c / frequency
    k = 2.0 * np.pi / wavelength

    if window_type == 'hanning':
        weights = hanning_window(n_elements)
    elif window_type == 'hamming':
        weights = hamming_window(n_elements)
    elif window_type == 'chebyshev':
        weights = chebyshev_window(n_elements)
    else:
        weights = np.ones(n_elements)

    pattern = np.zeros(len(angles))
    n = np.arange(n_elements)

    for i, theta in enumerate(angles):
        phases = k * n * element_spacing * np.sin(theta)
        sv = np.exp(1j * phases)
        pattern[i] = np.abs(np.sum(weights * sv))

    # 归一化并转为dB
    pattern = pattern / np.max(pattern)
    pattern_db = 20.0 * np.log10(pattern + 1e-10)
    pattern_db = np.clip(pattern_db, -80.0, 0.0)

    return pattern_db


def simulate_array_response(n_elements: int = 64,
                            element_spacing: float = 0.3e-3,
                            frequency: float = 5e6,
                            sampling_rate: float = 40e6,
                            n_samples: int = 2048,
                            scatterer_depths: np.ndarray = None,
                            scatterer_amplitudes: np.ndarray = None,
                            c: float = SOUND_SPEED_TISSUE) -> Tuple[np.ndarray, np.ndarray]:
    """模拟超声阵列对散射点的响应。

    参数:
        n_elements: 阵元数
        element_spacing: 阵元间距 (m)
        frequency: 中心频率 (Hz)
        sampling_rate: 采样率 (Hz)
        n_samples: 每通道采样数
        scatterer_depths: 散射点深度 (m)
        scatterer_amplitudes: 散射点幅值
        c: 声速 (m/s)

    返回:
        channel_data: (n_elements, n_samples) 通道数据
        element_positions: (n_elements,) 阵元位置
    """
    element_positions = (np.arange(n_elements) - (n_elements - 1) / 2.0) * element_spacing
    dt = 1.0 / sampling_rate
    time_axis = np.arange(n_samples) * dt

    if scatterer_depths is None:
        scatterer_depths = np.array([0.02, 0.035, 0.05])
    if scatterer_amplitudes is None:
        scatterer_amplitudes = np.array([1.0, 0.7, 0.4])

    channel_data = np.zeros((n_elements, n_samples), dtype=complex)

    # 脉冲波形（高斯调制正弦波）
    pulse_width = 2.0 / frequency
    envelope = np.exp(-(time_axis - pulse_width)**2 / (2.0 * (pulse_width / 4.0)**2))
    pulse = envelope * np.sin(2.0 * np.pi * frequency * time_axis)

    for elem_idx in range(n_elements):
        for scat_idx, (depth, amp) in enumerate(zip(scatterer_depths, scatterer_amplitudes)):
            # 阵元到散射点的距离
            dx = element_positions[elem_idx]
            distance = np.sqrt(dx**2 + depth**2)
            delay = 2.0 * distance / c

            # 延迟叠加
            delayed_pulse = np.zeros(n_samples)
            for t_idx, t in enumerate(time_axis):
                t_delayed = t - delay
                if t_delayed >= 0 and t_delayed < time_axis[-1]:
                    # 线性插值
                    src_idx = int(t_delayed / dt)
                    frac = t_delayed / dt - src_idx
                    if src_idx < n_samples - 1:
                        delayed_pulse[t_idx] = ((1.0 - frac) * pulse[src_idx] +
                                                frac * pulse[src_idx + 1])

            # 衰减（距离衰减和频率相关衰减）
            attenuation = amp / (distance + 1e-6)
            channel_data[elem_idx] += attenuation * delayed_pulse

    # 添加噪声
    noise_level = 0.05 * np.max(np.abs(channel_data))
    channel_data += noise_level * np.random.randn(n_elements, n_samples)

    return channel_data, element_positions
