"""
fractal_analysis.py
================================================================================
湍流分形分析模块 —— 基于种子项目 711_mandelbrot_area（迭代与面积估计思想）

湍流具有多尺度、间歇性的分形特征。本模块通过盒计数法（box-counting）
估计速度场等值面的分形维数，用于表征湍流的间歇性强度。

核心物理公式
--------------------------------------------------------------------------------
盒计数维数：
    D_f = lim_{ε→0} log N(ε) / log(1/ε)

其中 N(ε) 是覆盖集合所需边长为 ε 的盒子数。

对于湍流速度场，Kolmogorov 理论预测等耗散面维数 D ≈ 2.5–2.7，
而实际观测值因间歇性修正为 D ≈ 2.6–2.8（Meneveau & Sreenivasan, 1987）。

结构函数与分形维数的关系：
    D_p = 3 - ζ_p / p

其中 ζ_p 为 p 阶结构函数的标度指数，Kolmogorov 理论给出 ζ_p = p/3。

Richardson 级联的奇异性谱：
    f(α) = min_p [3 - D + p(α - ζ_p')]

其中 α 为 Holder 指数，描述局部奇异性强度。
"""

import numpy as np


def box_counting(field, threshold=None, max_level=6):
    """
    对二维标量场进行盒计数，估计分形维数。

    参数
    ----------
    field : np.ndarray, shape (nx, ny)
        标量场
    threshold : float, optional
        等值阈值，默认使用中位数
    max_level : int
        最大盒计数层级

    返回
    -------
    D_f : float
        盒计数维数估计
    scales, counts : np.ndarray
        尺度和计数
    """
    if field.ndim != 2:
        raise ValueError("box_counting: 目前仅支持二维场")

    if threshold is None:
        threshold = np.median(field)

    # 二值化：超过阈值的点
    binary = (field > threshold).astype(int)

    nx, ny = binary.shape
    min_dim = min(nx, ny)
    max_level = min(max_level, int(np.log2(min_dim)))

    scales = []
    counts = []

    for level in range(1, max_level + 1):
        box_size = 2 ** level
        n_boxes_x = nx // box_size
        n_boxes_y = ny // box_size

        if n_boxes_x < 1 or n_boxes_y < 1:
            break

        count = 0
        for i in range(n_boxes_x):
            for j in range(n_boxes_y):
                ix0 = i * box_size
                ix1 = (i + 1) * box_size
                jy0 = j * box_size
                jy1 = (j + 1) * box_size

                if np.any(binary[ix0:ix1, jy0:jy1] > 0):
                    count += 1

        epsilon = 1.0 / n_boxes_x
        scales.append(epsilon)
        counts.append(count)

    scales = np.array(scales, dtype=np.float64)
    counts = np.array(counts, dtype=np.float64)

    # 线性拟合 log(N) ~ D * log(1/epsilon)
    if len(scales) >= 2:
        log_eps = np.log(1.0 / scales)
        log_N = np.log(np.maximum(counts, 1))

        # 最小二乘
        A = np.vstack([log_eps, np.ones_like(log_eps)]).T
        D_f, _ = np.linalg.lstsq(A, log_N, rcond=None)[0]
    else:
        D_f = 0.0

    return D_f, scales, counts


def compute_intermittency_factor(field, window_size=8):
    """
    计算湍流间歇性因子（基于局部耗散的变异系数）。

    参数
    ----------
    field : np.ndarray
        二维标量场（如耗散率）
    window_size : int
        局部平均窗口

    返回
    -------
    mu : float
        间歇性因子
    """
    try:
        from scipy.ndimage import uniform_filter
    except ImportError:
        def uniform_filter(arr, size, mode='nearest'):
            from scipy.ndimage import uniform_filter as uf
            return uf(arr, size=size, mode=mode)

    local_mean = uniform_filter(field, size=window_size, mode='nearest')
    local_mean_safe = np.where(np.abs(local_mean) < 1e-15, 1e-15, local_mean)

    # 局部变异系数
    local_var = uniform_filter((field - local_mean)**2, size=window_size, mode='nearest')
    cv = np.sqrt(np.clip(local_var, 0, None)) / np.abs(local_mean_safe)
    cv = np.clip(cv, 0.0, 100.0)

    # 间歇性因子定义为 CV 的空间平均
    mu = float(np.mean(cv))
    return mu


def richardson_cascade_spectrum(k, epsilon, C=1.5, mu=0.25):
    """
    考虑间歇性修正的 Kolmogorov 能谱（She-Leveque 模型）。

    参数
    ----------
    k : np.ndarray
        波数
    epsilon : float
        耗散率
    C : float
        Kolmogorov 常数
    mu : float
        间歇性参数

    返回
    -------
    E : np.ndarray
        一维能谱
    """
    # Kolmogorov 尺度（假设 nu = 1.5e-5 m²/s，空气）
    nu = 1.5e-5
    eta = max((nu**3 / max(epsilon, 1e-12)) ** 0.25, 1e-10)

    # 惯性子区：E(k) = C ε^{2/3} k^{-5/3}
    k_safe = np.where(k <= 0, 1e-10, k)
    E = C * (max(epsilon, 1e-12) ** (2.0 / 3.0)) * (k_safe ** (-5.0 / 3.0))

    # 间歇性修正 (k η)^{-μ/9}
    corr = (k_safe * eta) ** (-mu / 9.0)
    corr = np.clip(corr, 1e-30, 1e30)
    E = E * corr

    # 指数截断（耗散区）
    arg = -5.0 * (k_safe * eta) ** (4.0 / 3.0)
    arg = np.clip(arg, -700, 700)  # 防止 exp 溢出
    E = E * np.exp(arg)

    E = np.where(k <= 0, 0.0, E)
    return E
