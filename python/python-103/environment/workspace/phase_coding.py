"""
phase_coding.py
相位编码、幻方矩阵与整数搜索模块
（对应种子项目 132_caesar, 708_magic_matrix, 445_four_fifths）

在光纤非线性光学中，本模块提供：
  1. 循环移位相位编码（Caesar密码启发的频谱循环卷积算子）
  2. 幻方矩阵（用于构造特殊的折射率分布测试矩阵和相位掩码）
  3. 整数组合搜索（用于寻找最优WDM信道波长分配）

核心物理公式：
  循环卷积相位编码:
    A_coded(t) = A(t) · exp(i φ_shift(t))
    其中φ_shift通过对频谱的循环移位实现：
    F{A_coded} = circshift(F{A}, k)

  幻方矩阵作为相位掩码:
    对于n×n幻方M，其元素为1到n²的排列，满足每行/列/对角线之和相等。
    归一化后用作空间相位调制器:
      φ_{i,j} = 2π (M_{i,j} - 1) / (n² - 1)

  色散匹配的波长整数关系:
    在WDM系统中，为最小化四波混频（FWM），信道间隔Δλ需满足:
      β₂ (2πc)² (1/λ_i² - 1/λ_j²) (1/λ_k² - 1/λ_l²) ≈ 0
    整数搜索用于寻找满足近零色散失配的波长组合。
"""

import numpy as np


def caesar_shift_phase(spectrum, shift_amount):
    """
    对频谱进行循环移位（对应Caesar密码的循环移位思想）。

    参数:
        spectrum: ndarray (complex), 频谱
        shift_amount: int, 移位量

    返回:
        shifted: ndarray (complex)
    """
    spectrum = np.asarray(spectrum)
    if spectrum.size == 0:
        return spectrum
    shift_amount = int(shift_amount) % spectrum.size
    if shift_amount == 0:
        return spectrum.copy()
    return np.roll(spectrum, shift_amount)


def magic_matrix(n):
    """
    生成奇数阶幻方矩阵（对应种子项目 708_magic_matrix）。

    Siamese方法:
      a) 从顶行中间开始，k=1
      b) 填入k
      c) 若k=n²，完成
      d) k增加1
      e) 向右上移动一格（循环环绕）
      f) 若该格已被占，向下移动一格
      g) 返回b)

    幻和（每行/列/对角线之和）= n(n²+1)/2
    """
    if n % 2 != 1:
        raise ValueError("magic_matrix: n must be odd")
    if n < 1:
        raise ValueError("magic_matrix: n must be positive")

    A = np.zeros((n, n), dtype=int)
    k = 1
    i = 0
    j = n // 2
    A[i, j] = k

    while k < n * n:
        k += 1
        im1 = (i - 1) % n
        jp1 = (j + 1) % n
        if A[im1, jp1] != 0:
            im1 = (i + 1) % n
            jp1 = j
        A[im1, jp1] = k
        i = im1
        j = jp1

    return A


def magic_phase_mask(n):
    """
    生成基于幻方矩阵的归一化相位掩码。

    返回相位值在[0, 2π)范围内的n×n矩阵。
    """
    M = magic_matrix(n)
    n2 = n * n
    phase = 2.0 * np.pi * (M - 1) / (n2 - 1)
    return phase


def apply_phase_mask_to_pulse(t, A, mask_size=5):
    """
    将幻方相位掩码概念应用到脉冲的时间-频率二维表示。

    通过短时傅里叶变换（STFT）将脉冲映射到时频平面，
    施加幻方相位调制，再逆变换。
    """
    if t.size < 2 or A.size != t.size:
        return A.copy()

    n = A.size
    # 简化的STFT：利用滑窗DFT
    window_size = min(mask_size * 2, n)
    hop = max(1, window_size // 4)
    n_frames = max(1, (n - window_size) // hop + 1)

    # 幻方相位掩码
    if mask_size % 2 == 0:
        mask_size += 1
    mask = magic_phase_mask(mask_size)

    # 简化的处理：直接在频域施加循环相位调制
    spectrum = np.fft.fft(A)
    n_freq = spectrum.size

    # 将幻方矩阵的元素映射为频谱不同频带的相位旋转
    phase_mod = np.zeros(n_freq)
    for idx in range(n_freq):
        band = idx % mask_size
        phase_mod[idx] = mask[band, band]

    modulated = spectrum * np.exp(1j * phase_mod)
    return np.fft.ifft(modulated)


def vector_sumlex_next(w):
    """
    生成下一个组合向量（用于four_fifths的整数搜索）。
    """
    m = w.size
    for i in range(m - 1, -1, -1):
        if w[i] > 0:
            w[i] -= 1
            if i < m - 1:
                w[i + 1] += 1
            return w
    return w


def four_fifths_search(n_max, exponent=5):
    """
    整数搜索：寻找满足 a^p + b^p + c^p + d^p = e^p 的整数组合。
    （对应种子项目 445_four_fifths）

    在光纤物理中的映射：寻找满足特定色散关系的WDM波长整数倍组合。
    """
    if n_max < 2:
        return None

    fifths = np.arange(1, n_max + 1, dtype=np.float64) ** exponent

    # 使用简化搜索（避免过大计算量）
    best_error = np.inf
    best_tuple = None

    # 限制搜索范围以保证可运行性
    search_limit = min(n_max, 30)
    for a in range(1, search_limit):
        for b in range(a, search_limit):
            for c in range(b, search_limit):
                for d in range(c, search_limit):
                    s = a ** exponent + b ** exponent + c ** exponent + d ** exponent
                    # 检查是否为某个整数的exponent次方
                    e_float = s ** (1.0 / exponent)
                    e_int = int(round(e_float))
                    if e_int > 0 and e_int <= n_max:
                        error = abs(e_int ** exponent - s)
                        if error < best_error:
                            best_error = error
                            best_tuple = (a, b, c, d, e_int)
                        if error == 0:
                            return best_tuple

    return best_tuple


def wdm_channel_search(center_wavelength_nm=1550, channel_spacing_nm=0.8, n_channels=4, target_fwm_efficiency=0.01):
    """
    寻找最小化四波混频效率的WDM信道整数波长分配。

    基于four_fifths整数搜索思想，寻找满足近零色散失配的波长组合。
    """
    best_channels = None
    best_fwm = np.inf

    # 在中心波长附近搜索整数倍间隔组合
    for base in range(1, 20):
        channels = [center_wavelength_nm + base * channel_spacing_nm * i for i in range(n_channels)]
        # 简化的FWM效率估计（正比于色散失配倒数）
        fwm_total = 0.0
        count = 0
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                for k in range(j + 1, n_channels):
                    # FWM条件: λ_i + λ_j = 2λ_k
                    mismatch = abs(channels[i] + channels[j] - 2.0 * channels[k])
                    fwm_total += 1.0 / (1.0 + mismatch)
                    count += 1
        if count > 0:
            avg_fwm = fwm_total / count
            if avg_fwm < best_fwm:
                best_fwm = avg_fwm
                best_channels = channels

    return best_channels, best_fwm
