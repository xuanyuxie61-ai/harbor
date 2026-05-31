# -*- coding: utf-8 -*-
"""
foreground_edges.py
前景污染模拟与边界检测

核心物理：
    CMB 天图受到银河系同步辐射、热尘埃等前景污染。
    本模块：
    1. 生成合成“双C形”前景数据（模拟两种紧密嵌套的前景成分
       在特征空间中的分布），来自 double_c_data 的核心思想。
    2. 对温度图进行边缘检测，利用多项式拟合法定位前景边界。
       在间断点邻域内拟合分段多项式，通过比较左右导数确定跳变位置。

融合种子项目 314_double_c_data（双C形聚类生成）与
325_edge（分段不连续函数边界检测）。
"""

import numpy as np
from typing import List, Tuple


# ---------------------------------------------------------------------------
# 双C形前景数据生成
# ---------------------------------------------------------------------------
def generate_double_c_foreground(n1: int = 300, n2: int = 300,
                                  seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    在二维参数空间中生成两个嵌套C形前景成分。
    成分1：右向C，r∈[2,5], θ∈[π/2, 3π/2]
    成分2：左向C，r∈[2,5], θ∈[3π/2, 5π/2]，中心偏移 (0, 3.5)。
    返回 (x, y, label)。
    """
    rng = np.random.default_rng(seed)
    # 成分1
    r1 = rng.uniform(2.0, 5.0, n1)
    theta1 = rng.uniform(np.pi / 2.0, 3.0 * np.pi / 2.0, n1)
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)
    # 成分2
    r2 = rng.uniform(2.0, 5.0, n2)
    theta2 = rng.uniform(3.0 * np.pi / 2.0, 5.0 * np.pi / 2.0, n2)
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2) + 3.5
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    labels = np.concatenate([np.zeros(n1), np.ones(n2)])
    # 随机打乱
    perm = rng.permutation(len(x))
    return x[perm], y[perm], labels[perm]


def foreground_temperature_profile(theta: np.ndarray,
                                    amplitude: float = 100.0,
                                    width: float = 0.2) -> np.ndarray:
    """
    模拟银河平面附近的前景温度剖面：
        T(θ) = A exp[-(θ/θ_width)^2] + 背景噪声
    其中 θ 为相对于银道面的角距离。
    """
    signal = amplitude * np.exp(-(theta / width) ** 2)
    noise = np.random.randn(len(theta)) * 5.0
    return signal + noise


# ---------------------------------------------------------------------------
# 边缘检测：多项式拟合跳变检测
# ---------------------------------------------------------------------------
def detect_edges_1d(y: np.ndarray, x: np.ndarray = None,
                     window: int = 5, threshold: float = 0.5) -> List[int]:
    """
    一维信号边缘检测：在每个点周围取左右窗口，分别拟合线性多项式，
    若左右斜率差异超过阈值则标记为边缘。

    数学原理：
        左邻域 L = {x_{i-w}, ..., x_{i-1}}
        右邻域 R = {x_{i+1}, ..., x_{i+w}}
        分别最小二乘拟合 y = a_L x + b_L 与 y = a_R x + b_R。
        若 |a_L - a_R| > threshold * max(|a_L|, |a_R|, ε)，则标记边缘。
    """
    n = len(y)
    if x is None:
        x = np.arange(n, dtype=float)
    edges = []
    for i in range(window, n - window):
        # 左窗口
        xl = x[i - window:i]
        yl = y[i - window:i]
        # 右窗口
        xr = x[i + 1:i + window + 1]
        yr = y[i + 1:i + window + 1]
        # 线性最小二乘
        Al = np.vstack([xl, np.ones(len(xl))]).T
        Ar = np.vstack([xr, np.ones(len(xr))]).T
        try:
            a_l, _ = np.linalg.lstsq(Al, yl, rcond=None)[0]
            a_r, _ = np.linalg.lstsq(Ar, yr, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        denom = max(abs(a_l), abs(a_r), 1e-12)
        if abs(a_l - a_r) / denom > threshold:
            edges.append(i)
    return edges


def shepp_logan_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    二维 Shepp-Logan 幻影（10 个椭圆叠加），作为标准前景边界测试图。
    每个椭圆定义：
        ((x-xc)cosα + (y-yc)sinα)^2 / a^2 + ((x-xc)sinα - (y-yc)cosα)^2 / b^2 ≤ 1
    强度列表为医学标准值，这里缩放为 μK 量级。
    """
    # 标准 Shepp-Logan 椭圆参数 (A, a, b, xc, yc, α)
    ellipses = [
        (1.0, 0.69, 0.92, 0.0, 0.0, 0.0),
        (-0.98, 0.6624, 0.8740, 0.0, -0.0184, 0.0),
        (-0.02, 0.1100, 0.3100, 0.22, 0.0, -18.0),
        (-0.02, 0.1600, 0.4100, -0.22, 0.0, 18.0),
        (0.01, 0.2100, 0.2500, 0.0, 0.35, 0.0),
        (0.01, 0.0460, 0.0460, 0.0, 0.1, 0.0),
        (0.01, 0.0460, 0.0460, 0.0, -0.1, 0.0),
        (0.01, 0.0460, 0.0230, -0.08, -0.605, 0.0),
        (0.01, 0.0230, 0.0230, 0.0, -0.606, 0.0),
        (0.01, 0.0230, 0.0460, 0.06, -0.605, 0.0),
    ]
    z = np.zeros_like(x)
    for A, a, b, xc, yc, alpha_deg in ellipses:
        alpha = np.radians(alpha_deg)
        xt = (x - xc) * np.cos(alpha) + (y - yc) * np.sin(alpha)
        yt = (x - xc) * np.sin(alpha) - (y - yc) * np.cos(alpha)
        mask = (xt ** 2 / (a ** 2) + yt ** 2 / (b ** 2)) <= 1.0
        z += A * mask
    # 缩放到 μK 量级
    return z * 50.0


def gradient_edge_detector_2d(image: np.ndarray, threshold: float = 10.0) -> np.ndarray:
    """
    二维 Sobel 梯度边缘检测。
    计算离散梯度幅值：
        G_x = (I_{i+1,j} - I_{i-1,j}) / 2
        G_y = (I_{i,j+1} - I_{i,j-1}) / 2
        |∇I| = sqrt(G_x^2 + G_y^2)
    若 |∇I| > threshold 则标记为边缘。
    """
    ny, nx = image.shape
    edges = np.zeros((ny, nx), dtype=bool)
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            gx = 0.5 * (image[i + 1, j] - image[i - 1, j])
            gy = 0.5 * (image[i, j + 1] - image[i, j - 1])
            grad = np.sqrt(gx ** 2 + gy ** 2)
            if grad > threshold:
                edges[i, j] = True
    return edges


# ---------------------------------------------------------------------------
# 前景-背景分离评估
# ---------------------------------------------------------------------------
def compute_residual_rms(cmb_map: np.ndarray,
                          foreground_map: np.ndarray,
                          mask: np.ndarray) -> float:
    """
    计算掩膜区域内的前景残留 RMS：
        σ_res = sqrt( mean[ (T_CMB + T_fg)^2 ]_mask )
    """
    total = cmb_map + foreground_map
    vals = total[mask]
    if len(vals) == 0:
        return 0.0
    return float(np.sqrt(np.mean(vals ** 2)))
