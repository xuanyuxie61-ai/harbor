"""
spectral_analysis.py
等离子体湍流与 MHD 模的傅里叶谱分析。

核心物理模型：
  托卡马克中的密度/温度涨落 δn(r,θ,t) 可分解为傅里叶模：

      δn(r,θ,t) = Σ_{m,n} Ã_{mn}(r,t) exp( i (m θ - n φ - ω_{mn} t) )

  其中 m 为极向模数，n 为环向模数，ω_{mn} 为复频率（实部为频率，
  虚部为增长率）。

  本模块采用快速傅里叶变换（FFT）分析模拟或实验信号，
  基于 Conte-de Boor 的素因子 FFT 算法思想，但利用 NumPy 的
  高效 FFT 实现。保留原项目的核心算法结构：
    - 信号分解为长度 N 的数组
    - 计算 DFT：X_k = Σ_{j=0}^{N-1} x_j exp(-2πi j k / N)
    - 通过功率谱密度识别主导模数

  MHD 稳定性判据：
    若某 (m,n) 模的谱功率 P_{mn} 随时间指数增长：
        P_{mn}(t) ∝ exp(2 γ_{mn} t)
    则该模不稳定，增长率 γ_{mn} 由对数线性拟合得到。

  剪切阿尔芬波色散关系（参考）：
        ω² = k_∥² v_A² / (1 + k_⊥² ρ_s²)
    其中 v_A = B / √(μ₀ ρ_m) 为阿尔芬速度，
    k_∥ = (m - n q) / (q R₀) 为平行波数。
"""

import numpy as np
from parameters import N_FFT


def compute_fft_spectrum(signal, dt=1.0e-3):
    """
    计算信号的 FFT 功率谱密度。

    公式
    ----
        X_k = Σ_{j=0}^{N-1} x_j exp(-2πi j k / N)
        P_k = |X_k|² / N
        f_k = k / (N dt)   for k = 0, ..., N/2

    参数
    ------
    signal : ndarray
        时域信号。
    dt : float
        采样间隔 [s]。

    返回
    ------
    freqs : ndarray
        正频率分量 [Hz]。
    power : ndarray
        功率谱密度。
    """
    signal = np.asarray(signal, dtype=float)
    N = len(signal)
    if N < 2:
        return np.array([0.0]), np.array([0.0])

    # 零填充到 2 的幂次或 N_FFT
    N_fft = max(N, N_FFT)
    N_fft = 1 << (N_fft - 1).bit_length()  # 向上取到 2 的幂

    signal_pad = np.zeros(N_fft)
    signal_pad[:N] = signal

    # FFT
    X = np.fft.fft(signal_pad)
    power = np.abs(X) ** 2 / N_fft
    freqs = np.fft.fftfreq(N_fft, d=dt)

    # 仅返回正频率
    positive = freqs >= 0
    return freqs[positive], power[positive]


def compute_wavenumber_spectrum(phi_signal, m_max=32, n_max=16):
    """
    计算二维 (m,n) 模数功率谱。

    参数
    ------
    phi_signal : ndarray, shape (n_theta, n_phi, n_t)
        环向角、极向角、时间三维信号。
    m_max, n_max : int
        最大模数。

    返回
    ------
    P_mn : ndarray, shape (m_max+1, n_max+1)
        (m,n) 模功率。
    gamma_mn : ndarray, shape (m_max+1, n_max+1)
        近似增长率 [1/s]。
    """
    phi = np.asarray(phi_signal, dtype=float)
    if phi.ndim != 3:
        raise ValueError("phi_signal 必须为 3 维数组 (theta, phi, t)")
    n_theta, n_phi, n_t = phi.shape

    P_mn = np.zeros((m_max + 1, n_max + 1))
    gamma_mn = np.zeros((m_max + 1, n_max + 1))

    for m in range(m_max + 1):
        for n in range(n_max + 1):
            # 提取 (m,n) 模的复振幅
            A = np.zeros(n_t, dtype=complex)
            for it in range(n_t):
                phase = np.outer(np.arange(n_theta), m) * 2j * np.pi / n_theta
                phase += np.outer(np.arange(n_phi), n) * 2j * np.pi / n_phi
                # 这里简化处理：直接 2D FFT 切片
                fft2 = np.fft.fft2(phi[:, :, it])
                if m < fft2.shape[0] and n < fft2.shape[1]:
                    A[it] = fft2[m, n]

            power_t = np.abs(A) ** 2
            P_mn[m, n] = np.mean(power_t)

            # 增长率拟合：ln(P) = 2γ t + const
            if n_t > 10 and np.mean(power_t) > 1e-30:
                logP = np.log(power_t + 1e-30)
                t_idx = np.arange(n_t)
                # 线性回归
                slope = np.cov(t_idx, logP)[0, 1] / np.var(t_idx)
                gamma_mn[m, n] = 0.5 * slope

    return P_mn, gamma_mn


def alfvén_dispersion(k_parallel, k_perp, B, rho_m):
    """
    剪切阿尔芬波色散关系。

    公式
    ----
        ω² = k_∥² v_A² / (1 + k_⊥² ρ_s²)
        v_A = B / √(μ₀ ρ_m)

    参数
    ------
    k_parallel : float or ndarray
        平行波数 [1/m]。
    k_perp : float or ndarray
        垂直波数 [1/m]。
    B : float
        磁场强度 [T]。
    rho_m : float
        质量密度 [kg/m³]。

    返回
    ------
    omega : float or ndarray
        阿尔芬频率 [rad/s]。
    v_A : float
        阿尔芬速度 [m/s]。
    """
    from parameters import MU0
    v_A = B / np.sqrt(MU0 * rho_m + 1e-30)
    k_par = np.asarray(k_parallel)
    k_perp = np.asarray(k_perp)
    rho_s = 1.0e-3  # 简化离子声拉莫尔半径 [m]
    omega = np.abs(k_par) * v_A / np.sqrt(1.0 + (k_perp * rho_s) ** 2)
    return omega, v_A


def compute_growth_rate_from_spectrum(power_history, dt=1.0e-3):
    """
    从功率历史计算线性增长率 γ。

    公式
    ----
        P(t) = P_0 exp(2 γ t)
        γ = 0.5 · d(ln P)/dt

    参数
    ------
    power_history : ndarray
        某模的功率随时间演化。
    dt : float
        时间步长。

    返回
    ------
    gamma : float
        增长率 [1/s]。
    r_squared : float
        线性拟合的决定系数。
    """
    P = np.asarray(power_history, dtype=float)
    P = np.maximum(P, 1e-30)
    logP = np.log(P)
    t = np.arange(len(P)) * dt

    if len(t) < 2:
        return 0.0, 0.0

    # 线性回归
    slope, intercept = np.polyfit(t, logP, 1)
    gamma = 0.5 * slope

    # R²
    y_mean = np.mean(logP)
    ss_tot = np.sum((logP - y_mean) ** 2)
    ss_res = np.sum((logP - (slope * t + intercept)) ** 2)
    r_squared = 1.0 - ss_res / (ss_tot + 1e-30)

    return gamma, r_squared


def detect_unstable_modes(P_mn, gamma_mn, power_threshold=1.0e-3, gamma_threshold=0.0):
    """
    检测不稳定的 (m,n) 模。

    参数
    ------
    P_mn, gamma_mn : ndarray
    power_threshold : float
        相对功率阈值。
    gamma_threshold : float
        增长率阈值 [1/s]。

    返回
    ------
    unstable_modes : list of tuple
        [(m, n, gamma, P), ...]
    """
    P_max = np.max(P_mn)
    if P_max < 1e-30:
        return []

    unstable = []
    m_max, n_max = P_mn.shape[0] - 1, P_mn.shape[1] - 1
    for m in range(m_max + 1):
        for n in range(n_max + 1):
            if P_mn[m, n] / P_max > power_threshold and gamma_mn[m, n] > gamma_threshold:
                unstable.append((m, n, float(gamma_mn[m, n]), float(P_mn[m, n])))

    # 按增长率排序
    unstable.sort(key=lambda x: x[2], reverse=True)
    return unstable


def generate_turbulent_signal(n_t=2048, dt=1.0e-4, seed=42):
    """
    生成模拟的等离子体湍流信号（多模叠加）。

    信号模型
    --------
        s(t) = Σ_k A_k exp(γ_k t) sin(2π f_k t + φ_k) + η(t)

    其中 η(t) 为高斯白噪声，模拟湍流背景。

    参数
    ------
    n_t : int
        采样点数。
    dt : float
        采样间隔。
    seed : int
        随机种子。

    返回
    ------
    signal : ndarray
    true_params : dict
        真实模参数。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_t) * dt

    # 三个不稳定 MHD 模
    modes = [
        {"A": 1.0, "f": 5.0e3, "gamma": 1.0e2, "phi": 0.0},    # (m=2,n=1)
        {"A": 0.6, "f": 12.0e3, "gamma": 5.0e1, "phi": 1.2},   # (m=3,n=2)
        {"A": 0.3, "f": 25.0e3, "gamma": -2.0e1, "phi": 2.5},  # 阻尼模
    ]

    signal = np.zeros(n_t)
    for mode in modes:
        envelope = mode["A"] * np.exp(mode["gamma"] * t)
        signal += envelope * np.sin(2.0 * np.pi * mode["f"] * t + mode["phi"])

    # 高斯白噪声
    noise = rng.normal(0, 0.1, n_t)
    signal += noise

    true_params = {"modes": modes, "dt": dt, "noise_std": 0.1}
    return signal, true_params
