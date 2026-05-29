"""
noise_model.py
噪声模型与概率统计模块
（对应种子项目 119_brownian_motion_simulation, 284_digital_dice）

在光纤通信系统中，放大自发辐射（ASE）噪声、瑞利散射和热噪声
都会劣化信号质量。本模块提供：
  1. 基于布朗运动的ASE噪声模型
  2. 光子统计的概率模型（玻色-爱因斯坦分布）

核心物理公式：
  ASE噪声功率谱密度:
    ρ_ASE = n_sp (G - 1) h ν
  其中n_sp为自发辐射因子，G为增益，h为普朗克常数，ν为光频率。

  布朗运动模型（Wiener过程）:
    W(t+Δt) = W(t) + √(2D Δt) ξ,  ξ ~ N(0,1)
  对应光纤中的相位噪声:
    φ(z+Δz) = φ(z) + √(2 α_{coh} Δz) ξ

  光子数分布（玻色-爱因斯坦统计）:
    P(n) = (⟨n⟩^n) / (⟨n⟩+1)^{n+1}

  Parrondo悖论启发:
    两个独立的"输家"过程（高损耗+高噪声）在交替作用下
    可能产生净增益——对应光纤中的拉曼放大过程。
"""

import numpy as np


def brownian_motion_simulation(m, n_steps, d, t_total, seed=None):
    """
    模拟m维布朗运动（对应种子项目 119_brownian_motion_simulation）。

    参数:
        m: int, 空间维数
        n_steps: int, 时间步数
        d: float, 扩散系数
        t_total: float, 总时间
        seed: int or None

    返回:
        x: ndarray shape (m, n_steps), 位置轨迹
    """
    if seed is not None:
        np.random.seed(seed)
    if n_steps < 2:
        raise ValueError("brownian_motion_simulation: n_steps must be >= 2")
    if t_total <= 0 or d < 0:
        raise ValueError("brownian_motion_simulation: invalid physical parameters")

    dt = t_total / (n_steps - 1)
    x = np.zeros((m, n_steps))

    s = np.sqrt(2.0 * m * d * dt)
    if m == 1:
        dx = s * np.random.randn(1, n_steps - 1)
    else:
        a = np.random.randn(m, n_steps - 1)
        norms = np.sqrt(np.sum(a ** 2, axis=0))
        norms = np.where(norms < 1e-15, 1.0, norms)
        v = s / norms
        dx = a * v[None, :]

    x[:, 1:] = np.cumsum(dx, axis=1)
    return x


def generate_ase_noise(t_grid, n_sp, G, h_nu, bw, seed=None):
    """
    生成ASE噪声的复包络（基于布朗运动模型）。

    参数:
        t_grid: ndarray, 时间网格
        n_sp: float, 自发辐射因子
        G: float, 增益 (linear)
        h_nu: float, 光子能量 (J)
        bw: float, 带宽 (Hz)
        seed: int or None

    返回:
        noise: ndarray (complex), 噪声包络
    """
    if t_grid.size < 2:
        raise ValueError("generate_ase_noise: invalid time grid")
    if G <= 1.0:
        return np.zeros_like(t_grid, dtype=complex)

    if seed is not None:
        np.random.seed(seed)

    dt = t_grid[1] - t_grid[0]
    # ASE功率谱密度
    psd = n_sp * (G - 1.0) * h_nu  # W/Hz
    # 总噪声功率
    P_noise = psd * bw
    # 对应到每个时间样本的噪声方差
    sigma = np.sqrt(P_noise * dt)

    # 复高斯噪声（实部和虚部独立）
    noise = sigma * (np.random.randn(t_grid.size) + 1j * np.random.randn(t_grid.size))
    return noise


def bose_einstein_distribution(n_avg, n_max=50):
    """
    计算玻色-爱因斯坦分布的概率质量函数。

    P(n) = n_avg^n / (n_avg + 1)^{n+1}

    参数:
        n_avg: float, 平均光子数
        n_max: int, 截断最大光子数

    返回:
        probs: ndarray shape (n_max+1,), 概率分布
        n: ndarray, 对应的光子数
    """
    if n_avg <= 0:
        raise ValueError("bose_einstein_distribution: n_avg must be positive")

    n = np.arange(n_max + 1)
    # 使用log-space计算避免溢出
    log_probs = n * np.log(n_avg) - (n + 1) * np.log(n_avg + 1.0)
    probs = np.exp(log_probs)
    probs = probs / np.sum(probs)
    return probs, n


def photon_number_fluctuation(t_grid, pulse_power, wavelength=1550e-9, n_avg_per_mode=1.0):
    """
    基于光子统计计算脉冲的量子涨落。

    参数:
        t_grid: ndarray, 时间网格 (s)
        pulse_power: ndarray, 瞬时功率 (W)
        wavelength: float, 波长 (m)
        n_avg_per_mode: float, 每模式平均光子数

    返回:
        fluctuation: ndarray, 功率涨落
    """
    if t_grid.size < 2 or pulse_power.size != t_grid.size:
        return np.zeros_like(t_grid)

    h = 6.62607015e-34  # Planck constant J·s
    c = 2.99792458e8
    nu = c / wavelength
    dt = t_grid[1] - t_grid[0]

    # 每个时间bin中的平均光子数
    n_photon = pulse_power * dt / (h * nu)
    # 玻色-爱因斯坦统计的方差
    var_photon = n_photon * (1.0 + n_photon / n_avg_per_mode)
    # 相对涨落
    rel_fluct = np.sqrt(var_photon) / np.maximum(n_photon, 1.0)
    rel_fluct = np.clip(rel_fluct, 0.0, 1.0)

    return rel_fluct * pulse_power


def parrondo_inspired_noise_coupling(t_grid, A_signal, epsilon=0.005):
    """
    受Parrondo悖论启发的噪声耦合模型（对应种子项目 284_digital_dice）。

    物理诠释：光纤中的两个"不利"过程（色散展宽+非线性相移）
    在特定交替模式下可能产生脉冲压缩效果（类似孤子形成）。

    参数:
        t_grid: ndarray
        A_signal: ndarray (complex), 输入信号
        epsilon: float, 扰动参数

    返回:
        A_out: ndarray (complex), 耦合后的信号
    """
    if t_grid.size < 2 or A_signal.size != t_grid.size:
        return A_signal.copy()

    dt = t_grid[1] - t_grid[0]
    n = t_grid.size
    A_out = A_signal.copy()

    # 模拟两种"游戏"过程交替作用
    for step in range(min(20, n // 2)):
        # 游戏A: 随机相位扰动（对应色散引起的随机相位）
        phase_A = np.sqrt(dt) * (np.random.randn(n) * (0.5 - epsilon))
        A_out *= np.exp(1j * phase_A)

        # 游戏B: 状态相关的非线性相移
        power = np.abs(A_out) ** 2
        median_power = np.median(power)
        # 高功率区域受到更强的非线性相移
        if median_power > 1e-30:
            state = (power > median_power).astype(float)
        else:
            state = np.zeros_like(power)

        phase_B_low = np.sqrt(dt) * (np.random.randn(n) * (0.1 - epsilon))
        phase_B_high = np.sqrt(dt) * (np.random.randn(n) * (0.75 - epsilon))
        phase_B = state * phase_B_high + (1.0 - state) * phase_B_low
        A_out *= np.exp(1j * phase_B)

    return A_out
