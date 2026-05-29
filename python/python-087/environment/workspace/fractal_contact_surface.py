"""
fractal_contact_surface.py
==========================
分形粗糙表面接触力学与间隙极小化

本模块将以下种子项目的核心算法融入结构力学：
  - 655_leaf_chaos   : 迭代函数系统 (IFS) 与混沌游戏 → 分形粗糙表面轮廓生成
  - 1218_test_min    : Brent 混合法（黄金分割 + 抛物插值）→ 接触间隙极小化

核心物理模型：
  - Weierstrass-Mandelbrot 分形表面：
        z(x) = Σ_{n=0}^{N-1} γ^{(D-2)n} · [cos(2π γ^n x + φ_n) / γ^{(2-D)n}]
    等价形式：
        z(x) = G^{D-1} · Σ_{n=0}^{N-1} γ^{(D-2)n} · cos(2π γ^n x / L_s + φ_n)
    其中 D 为分形维数 (1 < D < 2)，γ ≈ 1.5 为尺度因子，G 为粗糙度幅值。
  
  - Hertz-Mindlin 接触力模型（法向）：
        F_n = (4/3) E* √R · δ^{3/2}
    其中 δ 为接触压入深度，E* 为等效弹性模量，R 为等效曲率半径。
    
  - 对分形粗糙接触，采用 GW (Greenwood-Williamson) 扩展模型：
        微凸体高度分布 p(z) 由表面轮廓统计得到，
        总接触力 F = η A_n ∫_d^∞ (z - d)^{3/2} p(z) dz
    其中 η 为微凸体面密度，A_n 为名义接触面积，d 为分离距离。
"""

import numpy as np
from scipy.optimize import brent
from typing import Tuple, Optional, Callable


def ifs_fractal_surface_1d(length: float = 1.0, n_points: int = 1024,
                           D: float = 1.6, gamma: float = 1.5,
                           n_terms: int = 10, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 Weierstrass-Mandelbrot 函数生成分形粗糙表面一维轮廓。
    
    公式：
        z(x) = G^{D-1} · Σ_{n=0}^{N-1} γ^{(D-2)n} · cos(2π γ^n x / L + φ_n)
    
    为增加随机性，相位 φ_n 在 [0, 2π) 上均匀随机。
    
    参数
    ----
    length : 轮廓长度 L
    n_points : 采样点数
    D : 分形维数 (1, 2)
    gamma : 频率放大因子
    n_terms : Weierstrass 级数项数
    seed : 随机种子
    
    返回
    ----
    x : 位置坐标
    z : 表面高度
    """
    if not (1.0 < D < 2.0):
        raise ValueError("分形维数 D 必须在 (1, 2) 区间内")
    if gamma <= 1.0:
        raise ValueError("gamma 必须 > 1")
    if seed is not None:
        np.random.seed(seed)
    x = np.linspace(0, length, n_points)
    z = np.zeros_like(x)
    # 固定一个粗糙度幅值 G，使其归一化后的均方根约为 1e-4 * length
    G = length * 1e-4
    phases = np.random.uniform(0, 2 * np.pi, n_terms)
    for n in range(n_terms):
        amplitude = (G ** (D - 1)) * (gamma ** ((D - 2) * n))
        frequency = 2 * np.pi * (gamma ** n) / length
        z += amplitude * np.cos(frequency * x + phases[n])
    # 零均值
    z -= z.mean()
    return x, z


def ifs_fractal_surface_2d(size: float = 1.0, n_grid: int = 256,
                           D: float = 1.6, gamma: float = 1.5,
                           n_terms: int = 8, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    二维分形粗糙表面：
        z(x,y) = Σ_{n=0}^{N-1} γ^{(D-2)n} · [cos(2πγ^n(x+y)/L + φ_n)
                                              + cos(2πγ^n(x-y)/L + ψ_n)]
    该形式保证各向同性统计特性。
    """
    if not (1.0 < D < 2.0):
        raise ValueError("D 必须在 (1, 2)")
    if seed is not None:
        np.random.seed(seed)
    x = np.linspace(0, size, n_grid)
    y = np.linspace(0, size, n_grid)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    G = size * 1e-4
    for n in range(n_terms):
        amp = (G ** (D - 1)) * (gamma ** ((D - 2) * n))
        freq = 2 * np.pi * (gamma ** n) / size
        phi1 = np.random.uniform(0, 2 * np.pi)
        phi2 = np.random.uniform(0, 2 * np.pi)
        Z += amp * (np.cos(freq * (X + Y) + phi1) + np.cos(freq * (X - Y) + phi2))
    Z -= Z.mean()
    return X, Y, Z


def surface_statistics(z: np.ndarray) -> dict:
    """
    计算表面轮廓的统计参数，用于 Greenwood-Williamson 接触模型。
    
    返回
    ----
    dict : {
        "rms" : 均方根粗糙度 R_q = √(⟨z²⟩)
        "Ra"  : 算术平均粗糙度
        "skewness" : 偏度
        "kurtosis" : 峰度
        "peak_density" : 峰值密度（局部极大值数 / 长度）
    }
    """
    if z.size == 0:
        raise ValueError("空数组")
    rms = np.sqrt(np.mean(z ** 2))
    Ra = np.mean(np.abs(z - np.mean(z)))
    sigma = rms if rms > 1e-18 else 1.0
    skewness = np.mean((z - np.mean(z)) ** 3) / (sigma ** 3)
    kurtosis = np.mean((z - np.mean(z)) ** 4) / (sigma ** 4)
    # 局部极大值计数
    peaks = 0
    for i in range(1, len(z) - 1):
        if z[i] > z[i - 1] and z[i] > z[i + 1]:
            peaks += 1
    peak_density = peaks / len(z)
    return {
        "rms": float(rms),
        "Ra": float(Ra),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "peak_density": float(peak_density)
    }


def contact_gap_function(surface1_z: Callable[[np.ndarray], np.ndarray],
                         surface2_z: Callable[[np.ndarray], np.ndarray],
                         x: np.ndarray) -> np.ndarray:
    """
    计算两表面在对应位置 x 处的间隙函数：
        g(x) = z₂(x) - z₁(x)
    当 g(x) < 0 时发生穿透（接触）。
    """
    z1 = surface1_z(x)
    z2 = surface2_z(x)
    return z2 - z1


def minimize_contact_gap(surface1_z: Callable[[np.ndarray], np.ndarray],
                         surface2_z: Callable[[np.ndarray], np.ndarray],
                         x_min: float, x_max: float,
                         tol: float = 1e-9, maxiter: int = 100) -> Tuple[float, float]:
    """
    基于 1218_test_min 中 Brent 混合法的思想，在区间 [x_min, x_max] 上寻找
    使间隙函数 g(x) = z₂(x) - z₁(x) 最小的点 x*。
    
    这里调用 scipy.optimize.brent，其为 Brent 方法（黄金分割 + 反抛物插值）
    的稳健实现，超线性收敛（阶 ≈ 1.324）。
    
    返回
    ----
    x_star : 极小点
    g_min  : 最小间隙值
    """
    def gap_scalar(x):
        # brent 要求标量输入
        if np.isscalar(x):
            return float(surface2_z(np.array([x]))[0] - surface1_z(np.array([x]))[0])
        return surface2_z(x) - surface1_z(x)

    x_star = brent(gap_scalar, brack=(x_min, x_max), tol=tol, maxiter=maxiter)
    g_min = gap_scalar(x_star)
    return float(x_star), float(g_min)


def hertz_mindlin_normal_force(delta: float, E_star: float, R_eq: float) -> float:
    """
    Hertz-Mindlin 弹性接触法向力：
        F_n = (4/3) · E* · √R_eq · δ^{3/2}
    其中：
        1/E* = (1-ν₁²)/E₁ + (1-ν₂²)/E₂
    
    参数
    ----
    delta : 压入深度 [m]（必须 ≥ 0）
    E_star : 等效弹性模量 [Pa]
    R_eq : 等效曲率半径 [m]
    """
    if delta < 0:
        return 0.0
    if delta < 1e-18:
        return 0.0
    return (4.0 / 3.0) * E_star * np.sqrt(R_eq) * (delta ** 1.5)


def equivalent_modulus_radius(E1: float, nu1: float, E2: float, nu2: float,
                               R1: float, R2: float) -> Tuple[float, float]:
    """
    计算 Hertz 接触中的等效弹性模量 E* 与等效曲率半径 R_eq。
    
    公式：
        1/E* = (1-ν₁²)/E₁ + (1-ν₂²)/E₂
        1/R_eq = 1/R₁ + 1/R₂   (凸-凸接触)
    """
    E_star = 1.0 / ((1.0 - nu1 ** 2) / E1 + (1.0 - nu2 ** 2) / E2)
    R_eq = 1.0 / (1.0 / R1 + 1.0 / R2)
    return float(E_star), float(R_eq)


def gw_contact_force(stats: dict, separation: float, E_star: float,
                     R_eq: float, eta: float, A_n: float) -> float:
    """
    Greenwood-Williamson 统计接触模型总法向力。
    
    假设微凸体高度服从高斯分布 p(z) = (1/√(2π)σ) exp(-(z-μ)²/(2σ²))，
    则总力近似为：
        F = η A_n · (4/3) E* √R_eq · σ^{3/2} · ∫_{d/σ}^∞ (s - d/σ)^{3/2} φ(s) ds
    
    其中 φ(s) 为标准正态 PDF。此处用数值积分近似。
    """
    sigma = stats["rms"]
    if sigma <= 1e-18:
        return 0.0
    d_norm = separation / sigma
    # 积分区间 [max(d_norm, -5), 5]，使用 Simpson 法则
    s = np.linspace(max(d_norm, -5.0), 5.0, 1001)
    if len(s) < 2:
        return 0.0
    ds = s[1] - s[0]
    phi = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * s ** 2)
    integrand = np.maximum(s - d_norm, 0.0) ** 1.5 * phi
    integral = np.trapezoid(integrand, s)
    prefactor = eta * A_n * (4.0 / 3.0) * E_star * np.sqrt(R_eq) * (sigma ** 1.5)
    return float(prefactor * integral)
