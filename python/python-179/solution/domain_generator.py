"""
domain_generator.py
参数化计算域生成模块
====================
对应原项目 093_bird_egg（鸟类蛋形参数化公式）与 502_hand_data（手部轮廓数据获取），
融合生成具有生物形态特征的参数化一维/二维计算域，用于后续 FEM 离散化。
"""

import numpy as np
from typing import Tuple
from system_utils import robust_sqrt, clip_to_range


# ---------------------------------------------------------------------------
# 鸟类蛋形参数化公式（Narushin et al., 2021）
# ---------------------------------------------------------------------------

def chicken_egg_half_profile(B: float, L: float, w: float, x: np.ndarray) -> np.ndarray:
    """
    鸡形蛋的半高度轮廓（旋转对称截面）。

    公式
    ----
    给定最大宽度 B、长度 L、最大宽度偏移 w（从中心向钝端偏移量），
    对轴向坐标 x∈[-L/2, L/2]，半高度 y 满足

        y(x) = 0.5 * B * sqrt( (L² - 4x²) / (L² + 8wx + 4w²) )

    参数范围约束
    ------------
    B ∈ (0, L],  w ∈ [-L/4, L/4] 以保证分母正定。
    """
    x = np.asarray(x, dtype=float)
    B = float(B)
    L = float(L)
    w = float(w)
    # 边界保护
    B = clip_to_range(B, 1e-6, L)
    w = clip_to_range(w, -L / 4 + 1e-6, L / 4 - 1e-6)
    numerator = L * L - 4.0 * x * x
    denominator = L * L + 8.0 * w * x + 4.0 * w * w
    # 鲁棒处理
    numerator = np.maximum(numerator, 0.0)
    denominator = np.maximum(denominator, 1e-12)
    y = 0.5 * B * robust_sqrt(numerator / denominator)
    return y


def pyriform_egg_half_profile(B: float, L: float, w: float, x: np.ndarray) -> np.ndarray:
    """
    梨形蛋的半高度轮廓（更尖锐的尖端）。

    公式
    ----
        y(x) = 0.5 * B * sqrt( (L² - 4x²) * (L² + 2wL + 4x(L+2w)) /
                               (L⁴ + 4wL³ + 4w²L² + 16w²x² + 32w³x + 16w⁴) )
    """
    x = np.asarray(x, dtype=float)
    B = float(B)
    L = float(L)
    w = float(w)
    B = clip_to_range(B, 1e-6, L)
    w = clip_to_range(w, -L / 4 + 1e-6, L / 4 - 1e-6)
    term1 = L * L - 4.0 * x * x
    term2 = L * L + 2.0 * w * L + 4.0 * x * (L + 2.0 * w)
    denom = (L**4 + 4.0 * w * L**3 + 4.0 * w * w * L * L
             + 16.0 * w * w * x * x + 32.0 * w**3 * x + 16.0 * w**4)
    numerator = term1 * term2
    numerator = np.maximum(numerator, 0.0)
    denom = np.maximum(denom, 1e-12)
    y = 0.5 * B * robust_sqrt(numerator / denom)
    return y


def universal_egg_half_profile(B: float, L: float, w: float, D: float,
                                x: np.ndarray) -> np.ndarray:
    """
    通用蛋形公式：在鸡形基础上加入梨形修正。

    参数 D 为距尖端 1/4 长度处的直径。通过 Chebyshev 节点参数化 x 坐标，
    提高后续 FEM 插值的数值稳定性。
    """
    x = np.asarray(x, dtype=float)
    B = clip_to_range(float(B), 1e-6, float(L))
    L = float(L)
    w = clip_to_range(float(w), -L / 4 + 1e-6, L / 4 - 1e-6)
    D = clip_to_range(float(D), 1e-6, B)
    y_chicken = chicken_egg_half_profile(B, L, w, x)
    y_pyriform = pyriform_egg_half_profile(B, L, w, x)
    # 混合参数 λ = (B - D) / B，确保 D 控制尖端锐度
    lam = (B - D) / B
    lam = clip_to_range(lam, 0.0, 1.0)
    y = (1.0 - lam) * y_chicken + lam * y_pyriform
    return y


# ---------------------------------------------------------------------------
# Chebyshev 节点生成（对应原 093_bird_egg 的 cheby1space）
# ---------------------------------------------------------------------------

def chebyshev_nodes_1d(a: float, b: float, n: int) -> np.ndarray:
    """
    第一类 Chebyshev 节点：在 [a, b] 上生成 n 个点

        x_i = (a+b)/2 + (b-a)/2 * cos( (2i+1)π / (2n) ),  i=0,...,n-1

    该节点分布最小化多项式插值的 Lebesgue 常数，抑制 Runge 现象。
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    i = np.arange(n, dtype=float)
    t = np.cos((2.0 * i + 1.0) * np.pi / (2.0 * n))
    x = 0.5 * (a + b) + 0.5 * (b - a) * t
    return x


def generate_parametric_radial_domain(n: int = 64,
                                       B: float = 1.0,
                                       L: float = 2.0,
                                       w: float = 0.1,
                                       D: float = 0.6) -> np.ndarray:
    """
    生成一维参数化径向域坐标。
    以通用蛋形长度 L 为基准，在 [-L/2, L/2] 上取 Chebyshev 节点。
    返回的 x 坐标可直接用于后续 FEM  hat 函数离散化。
    """
    x = chebyshev_nodes_1d(-L / 2.0, L / 2.0, n)
    return x


def compute_radial_cross_section(x: np.ndarray,
                                  B: float = 1.0,
                                  L: float = 2.0,
                                  w: float = 0.1,
                                  D: float = 0.6) -> np.ndarray:
    """
    计算对应 x 坐标的径向截面半径 r(x) = y(x)。
    后续可将该一维半径函数绕轴旋转，得到二维轴对称域。
    """
    r = universal_egg_half_profile(B, L, w, D, x)
    return r


# ---------------------------------------------------------------------------
# 手部轮廓近似（对应原 502_hand_data）
# ---------------------------------------------------------------------------

def hand_outline_polygon(n_points: int = 100) -> np.ndarray:
    """
    生成一个合成手部轮廓的多边形顶点（n_points × 2）。
    使用参数化 Fourier 描述子近似人手形状，避免交互式 ginput。

    数学模型
    --------
    用有限 Fourier 级数描述闭合曲线
        z(θ) = Σ_{k=-K}^{K} c_k * e^{ikθ},  θ∈[0,2π]
    其中 c_k 为复 Fourier 系数。取前 K=8 项即可得到类手轮廓。
    该表示天然具有低秩结构：轮廓采样点矩阵 Z∈ℂ^{n×1} 的 Hankel 矩阵
    秩不超过 2K+1（Prony 方法）。
    """
    K = 8
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    # 预设 Fourier 系数（经验值，模拟手掌+五指凸起）
    c = np.zeros(2 * K + 1, dtype=complex)
    c[K] = 1.0 + 0.0j           # 直流分量
    c[K + 1] = 0.35 - 0.12j
    c[K - 1] = 0.35 + 0.12j
    c[K + 2] = -0.08 + 0.20j
    c[K - 2] = -0.08 - 0.20j
    c[K + 3] = 0.15 - 0.05j
    c[K - 3] = 0.15 + 0.05j
    c[K + 4] = -0.05 + 0.10j
    c[K - 4] = -0.05 - 0.10j
    c[K + 5] = 0.03 - 0.02j
    c[K - 5] = 0.03 + 0.02j
    # 计算轮廓
    z = np.zeros(n_points, dtype=complex)
    for k in range(-K, K + 1):
        z += c[K + k] * np.exp(1j * k * theta)
    # 转换为实坐标并中心归一化
    xy = np.column_stack((z.real, z.imag))
    xy -= xy.mean(axis=0)
    # 缩放至单位尺度
    max_norm = np.max(np.linalg.norm(xy, axis=1))
    if max_norm > 0:
        xy /= max_norm
    return xy


def hand_ellipse_fourier_approx(n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回手部轮廓的椭圆 Fourier 近似及其低秩主成分。
    返回 (xy, pcs)，其中 pcs 为前 3 个主成分方向（对应 SVD 的左奇异向量）。

    低秩解释
    --------
    对手部轮廓矩阵 X∈ℝ^{n×2} 执行 SVD：X = U Σ V^T。
    取截断秩 r=1 时，X ≈ σ1 u1 v1^T，即最佳秩-1 椭圆近似。
    """
    xy = hand_outline_polygon(n_points)
    Xc = xy - xy.mean(axis=0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Vt.T[:, :3]  # 前 3 个主方向
    return xy, pcs
