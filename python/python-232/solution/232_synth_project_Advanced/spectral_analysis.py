"""
spectral_analysis.py — 散射振幅谱分析与噪声滤波 (PROJECT_232)

融合种子项目:
  - 870_pink_noise: 功率谱密度与互相关 (correlation, cross_corr, ranh)
  - 1052_mctools_cdft_ncrystal: FFT与HDRFT算法 (takfft, takconv, FunctionXY)

核心物理:
  散射振幅的谱分析用于:
    1. 共振态识别 (Breit-Wigner 峰)
    2. 背景噪声的功率谱分析
    3. 非微扰 QCD 效应的频域特征

  功率谱密度:
    S(f) = |F(ω)|² = |∫ f(t) e^{-iωt} dt|²

  散射振幅的傅里叶变换:
    f̃(b) = ∫₀^∞ dq q J₀(qb) f(q)
  其中 b 为碰撞参数 (impact parameter)，
  J₀ 为零阶 Bessel 函数。

  HDRFT (High-order Discrete Resolvent Fourier Transform):
    用于从虚时间关联函数 γ(τ) 恢复实时动态结构因子 S(q,ω)
    来自 1052_mctools_cdft_ncrystal 的核心算法。
"""
import numpy as np
from constants import PI, TWOPI, EPS_MACH


# 物理常数 (局部定义避免循环导入)
const_hbar_local = 4.13566769692386e-15 * 0.5 / PI
const_boltzmann_local = 8.6173303e-5


# ===================================================================
# 功率谱密度 (融合 870_pink_noise)
# ===================================================================

def power_spectrum_density(signal, dt=1.0, window='hanning'):
    """
    信号的功率谱密度 (PSD)

    PSD(f) = (1/N) |FFT{x(n) · w(n)}|²

    其中 w(n) 为窗函数 (减少频谱泄漏)。

    物理应用: 分析散射截面随能量变化的功率谱，
    识别共振结构的特征频率。

    Parameters
    ----------
    signal : ndarray
        时域/能量域信号
    dt : float
        采样间距
    window : str
        窗函数类型 ('hanning', 'kaiser', 'none')

    Returns
    -------
    freq : ndarray
        频率轴
    psd : ndarray
        功率谱密度
    """
    N = len(signal)
    if N < 2:
        return np.array([0.0]), np.array([0.0])

    # 窗函数
    if window == 'hanning':
        w = np.hanning(N)
    elif window == 'kaiser':
        w = np.kaiser(N, 20)
    else:
        w = np.ones(N)

    windowed = signal * w
    # FFT
    fft_vals = np.fft.fftshift(np.fft.fft(np.fft.fftshift(windowed))) * dt / TWOPI
    freq = np.fft.fftshift(np.fft.fftfreq(N, dt)) * TWOPI

    psd = np.abs(fft_vals) ** 2 / N
    return freq, psd


def cross_correlation(sig1, sig2, dt=1.0):
    """
    两个信号的互相关函数

    C_{12}(τ) = ∫ f₁(t) f₂(t+τ) dt
              = IFFT[F₁*(ω) · F₂(ω)]

    物理应用: 比较不同分波振幅的相位关系，
    或分析理论预测与实验数据的吻合度。

    Parameters
    ----------
    sig1, sig2 : ndarray
        两个信号
    dt : float
        采样间距

    Returns
    -------
    lag : ndarray
        延迟轴
    correlation : ndarray
        互相关函数
    """
    N = min(len(sig1), len(sig2))
    if N < 2:
        return np.array([0.0]), np.array([0.0])

    f1 = np.fft.fft(sig1[:N])
    f2 = np.fft.fft(sig2[:N])
    corr = np.fft.ifft(f1.conj() * f2) * dt
    corr = np.fft.fftshift(corr)

    lag = np.arange(-N // 2, N // 2) * dt
    return lag, np.real(corr)


def pink_noise_spectrum(n_freq, beta=1.0):
    """
    生成 1/f^β 噪声的功率谱

    S(f) ∝ 1/f^β

    β=0: 白噪声
    β=1: 粉红噪声 (1/f 噪声)
    β=2: 布朗噪声 (随机游走)

    物理应用: 高能散射中，非微扰 QCD 背景
    可能表现出 1/f 型功率谱。

    生成方法 (Voss-McCartney 算法):
      1. 在八度频率上生成独立随机序列
      2. 叠加得到 1/f 谱

    Parameters
    ----------
    n_freq : int
        频率点数
    beta : float
        谱指数 (default: 1.0)

    Returns
    -------
    freq : ndarray
        频率轴
    amplitude : ndarray
        噪声振幅
    """
    if n_freq < 2:
        return np.array([1.0]), np.array([1.0])

    freq = np.arange(1, n_freq + 1, dtype=np.float64)
    # 功率谱 ∝ 1/f^β
    psd = 1.0 / np.maximum(freq ** beta, EPS_MACH)
    # 振幅 = √PSD
    amplitude = np.sqrt(psd)
    # 随机相位
    phase = np.random.uniform(0, TWOPI, n_freq)
    spectrum = amplitude * np.exp(1j * phase)
    return freq, np.real(spectrum)


# ===================================================================
# HDRFT 算法 (融合 1052_mctools_cdft_ncrystal)
# ===================================================================

class FunctionXY:
    """
    一维函数 y=f(x) 的数据结构

    来自 1052_mctools_cdft_ncrystal/FunctionXY.py
    支持 FFT 卷积、缩放、裁剪等操作。

    Attributes
    ----------
    x : ndarray
        x 坐标
    y : ndarray
        y 值 (可以是复数)
    """

    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.complex128)
        if len(self.x) != len(self.y):
            raise ValueError("FunctionXY: x and y must have same length")

    def get_delta_x(self):
        """获取 x 间距"""
        if len(self.x) < 2:
            return 1.0
        return self.x[1] - self.x[0]

    def crop(self, x_min, x_max):
        """裁剪到 [x_min, x_max]"""
        mask = (self.x >= x_min) & (self.x <= x_max)
        self.x = self.x[mask]
        self.y = self.y[mask]

    def scale_y(self, factor):
        """缩放 y 值"""
        self.y *= factor

    def accumulate(self, other):
        """累加另一个函数 (需要相同 x 网格)"""
        if len(self.x) == len(other.x):
            self.y += other.y

    def normalise(self):
        """归一化使积分=1"""
        area = np.trapz(np.abs(self.y), self.x)
        if area > EPS_MACH:
            self.y /= area


def hdrft_spectrum(time_grid, gamma_function, order=20,
                    first_order_cutoff=0.9):
    """
    HDRFT (High-order Discrete Resolvent Fourier Transform)

    从虚时间关联函数 γ(τ) 计算动态结构因子 S(ω):

    基本思路:
      γ(τ) = ∫ S(ω) e^{-ωτ} dω  (Laplace 变换)
      S(ω) = (1/2π) ∫ γ(τ) e^{iωτ} dτ  (逆 Laplace)

    但直接数值反演不稳定。HDRFT 通过高阶展开实现稳定反演:
      γ(τ) = exp(-Γ(τ))
      Γ(τ) = Σ_n (r₀)^n / n! · g_n(τ)

    其中 g_n 是 g₁ 的 n 重卷积:
      g_n = g₁ * g₁ * ... * g₁  (n 次)

    最终:
      S(ω) = e^{x₀} · Σ_n (r₀)^n / n! · G_n(ω)
    其中 G_n(ω) 是 g_n(τ) 的 FFT。

    Parameters
    ----------
    time_grid : ndarray
        虚时间网格 τ
    gamma_function : ndarray
        虚时间关联函数 γ(τ)
    order : int
        HDRFT 展开阶数
    first_order_cutoff : float
        一阶截止频率

    Returns
    -------
    omega : ndarray
        频率轴
    S_omega : ndarray
        动态结构因子 S(ω)
    """
    dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 1.0
    N = len(time_grid)

    # 频率轴
    omega = np.fft.fftshift(np.fft.fftfreq(N, dt)) * TWOPI

    # 计算 Γ(τ) = -ln γ(τ)
    gamma_safe = np.maximum(np.abs(gamma_function), EPS_MACH)
    exponent = -np.log(gamma_safe)

    # r₀ = Γ(0)
    r0 = exponent[0] if exponent[0] > 0 else 1.0
    ft_shifted = exponent - r0

    # g₀ = 1, g₁ = (Γ - r₀) / r₀
    f0 = np.ones(N)
    f1 = ft_shifted / max(r0, EPS_MACH)

    # FFT of g₀ and g₁
    g0_fft = np.fft.fftshift(np.fft.fft(np.fft.fftshift(f0))) * dt / TWOPI
    g1_fft = np.fft.fftshift(np.fft.fft(np.fft.fftshift(f1))) * dt / TWOPI

    # 高阶展开
    total_fft = g0_fft.copy()
    gn_fft = g1_fft.copy()
    coef = r0

    for n in range(1, order + 1):
        # 卷积: G_n = G_{n-1} ⊛ G_1
        gn_fft = np.convolve(gn_fft, g1_fft, mode='same') * dt / TWOPI
        coef *= r0 / n
        total_fft += coef * gn_fft

    S_omega = np.abs(total_fft) * np.exp(r0)

    # 归一化
    area = np.trapz(S_omega, omega) if len(omega) > 1 else 1.0
    if area > EPS_MACH:
        S_omega /= area

    return omega, S_omega


# ===================================================================
# 散射振幅的谱分解
# ===================================================================

def partial_wave_spectral_decomposition(delta_l_array, sqrt_s_grid):
    """
    分波振幅的谱分解

    将相移 δ_l(√s) 分解为:
      δ_l(√s) = δ_bg(√s) + Σ_r δ_res,r(√s)

    其中 δ_bg 为平滑背景，δ_res 为共振贡献。

    通过功率谱分析识别共振:
      - 共振在功率谱中表现为尖峰
      - 背景贡献集中在低频

    Parameters
    ----------
    delta_l_array : ndarray, shape (n_l, n_s)
        各分波在各能量点的相移
    sqrt_s_grid : ndarray
        √s 网格

    Returns
    -------
    background : ndarray
        背景相移 (低通滤波后)
    resonances : list of dict
        共振参数列表 [{sqrt_s, width, strength}, ...]
    """
    n_l, n_s = delta_l_array.shape
    ds = sqrt_s_grid[1] - sqrt_s_grid[0] if n_s > 1 else 1.0

    background = np.zeros_like(delta_l_array)
    resonances = []

    for l in range(n_l):
        signal = delta_l_array[l]
        freq, psd = power_spectrum_density(signal, ds)

        # 低通滤波 (背景)
        cutoff_idx = max(len(freq) // 10, 1)
        fft_vals = np.fft.fft(signal)
        fft_filtered = fft_vals.copy()
        # 保留低频分量
        fft_filtered[cutoff_idx:-cutoff_idx] = 0.0
        background[l] = np.real(np.fft.ifft(fft_filtered))

        # 识别共振峰 (PSD 中的尖峰)
        psd_smooth = np.convolve(psd, np.ones(5) / 5, mode='same')
        for i in range(1, len(psd_smooth) - 1):
            if psd_smooth[i] > psd_smooth[i - 1] and psd_smooth[i] > psd_smooth[i + 1]:
                if psd_smooth[i] > np.median(psd_smooth) * 3:
                    resonances.append({
                        'partial_wave': l,
                        'frequency': freq[i] if i < len(freq) else 0.0,
                        'strength': psd_smooth[i]
                    })

    return background, resonances


def impact_parameter_representation(f_theta, q_grid, b_grid):
    """
    散射振幅的碰撞参数表示

    f̃(b) = ∫₀^∞ dq q J₀(qb) f(q) / (2π)

    其中:
      b = 碰撞参数 (impact parameter)
      q = 动量转移
      J₀ = 零阶 Bessel 函数

    物理意义: f̃(b) 描述在碰撞参数 b 处的散射振幅。
    黑盘极限: |f̃(b)|² ≤ 1 (幺正性约束)

    Parameters
    ----------
    f_theta : ndarray
        散射振幅 f(q)
    q_grid : ndarray
        动量转移网格
    b_grid : ndarray
        碰撞参数网格

    Returns
    -------
    f_tilde_b : ndarray
        碰撞参数空间振幅
    """
    f_tilde_b = np.zeros(len(b_grid), dtype=np.complex128)
    for i, b in enumerate(b_grid):
        # J₀(qb) 使用 numpy 或 scipy
        # 简化: J₀(z) ≈ cos(z - π/4) · √(2/(πz)) 对于大 z
        qb = q_grid * b
        J0 = np.zeros_like(qb)
        nonzero = qb > 1e-6
        # 使用 Taylor 级数近似 J₀
        for j in range(len(qb)):
            if not nonzero[j]:
                J0[j] = 1.0
            else:
                z = qb[j]
                # 级数: J₀(z) = Σ (-1)^k (z/2)^{2k} / (k!)²
                J0_val = 0.0
                term = 1.0
                for k in range(20):
                    J0_val += term
                    term *= -(z / 2.0) ** 2 / ((k + 1) ** 2)
                    if abs(term) < EPS_MACH:
                        break
                J0[j] = J0_val

        integrand = q_grid * J0 * f_theta
        f_tilde_b[i] = np.trapz(integrand, q_grid) / TWOPI

    return f_tilde_b
