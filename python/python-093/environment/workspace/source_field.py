#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
source_field.py
水声传播抛物方程模型 — 声源初始场生成

本模块生成抛物方程的初始声压场 u(0,z)，来源于：
- 301_disk01_monte_carlo（圆盘均匀采样 → 圆形声源孔径采样）
- 1082_sinc（sinc 函数 → 带限源场插值、高斯包络）
- 498_hammersley（准随机序列 → 声源参数高效采样）

核心物理公式：
1. 高斯束源（Gaussian beam starter）：
   u(0,z) = exp[−(z−z_s)² / (2·w₀²)] · exp(i·k₀·(z−z_s)² / (2·R_c))
   其中：
     z_s 为声源深度；
     w₀ 为束腰半径；
     R_c 为波前曲率半径（R_c→∞ 为平面波）；
     k₀ = ω/c₀ 为参考波数。

2. 格林函数 starter（更精确，适用于近场）：
   u(0,z) = H₀⁽¹⁾(k₀·ρ) · √(k₀/(2πi·ρ)) · exp(i·k₀·ρ)
   其中 ρ = |z − z_s|，H₀⁽¹⁾ 为零阶第一类 Hankel 函数。
   当 k₀ρ ≫ 1 时，渐近近似：
   H₀⁽¹⁾(k₀ρ) ≈ √(2/(π·k₀ρ)) · exp(i·(k₀ρ − π/4))。

3. 圆形声源孔径上的均匀采样（ disk01_monte_carlo 思想）：
   对于半径为 a 的圆形活塞声源，孔径上的点满足均匀分布：
   r = a·√u,  θ = 2πv,  u,v ∼ U(0,1)
   由此可计算方向性因子 D(θ) = 2·J₁(k·a·sinθ) / (k·a·sinθ)。

4. sinc 插值重建（带限场精确重构）：
   u(z) = Σ_m u(z_m) · sinc_n((z−z_m)/Δz)
   其中 sinc_n(x) = sin(πx)/(πx) 为 Shannon 采样定理的核函数。

5. Hammersley 准随机序列用于声源参数扫描：
   在多维参数空间 (f, z_s, w₀, R_c) 中均匀填充采样点，
   用于灵敏度分析和多频传播损失计算。
"""

import numpy as np
from special_functions import sincn_fun


# =============================================================================
# Hammersley 准随机序列（来自 498_hammersley）
# =============================================================================

def prime_list(n):
    """返回前 n 个素数（硬编码前 50 个）。"""
    primes = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
        31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
        127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
        179, 181, 191, 193, 197, 199, 211, 223, 227, 229
    ]
    return primes[:n]


def radical_inverse(i, base):
    """
    计算第 i 个 van der Corput 序列值（radical inverse）：
    i 用 base 进制表示为 d_k...d_1 d_0，则
    φ_base(i) = d_0/base + d_1/base² + ... + d_k/base^{k+1}
    """
    i = int(i)
    base = int(base)
    result = 0.0
    f = 1.0 / base
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result


def hammersley_sequence(i1, i2, m, n=None):
    """
    生成 Hammersley 低差异序列。
    第 i 个样本的第 1 维为 i/n（若 n 给定），其余维为素数基 radical inverse。
    返回形状为 (i2−i1, m) 的数组。
    """
    if n is None:
        n = i2
    primes = prime_list(m)
    seq = np.zeros((i2 - i1, m), dtype=np.float64)
    for idx, i in enumerate(range(i1, i2)):
        if m >= 1:
            seq[idx, 0] = i / n
        for j in range(1, m):
            seq[idx, j] = radical_inverse(i, primes[j - 1])
    return seq


# =============================================================================
# 圆盘均匀采样（来自 301_disk01_monte_carlo）
# =============================================================================

def disk_uniform_sample(n, radius=1.0, seed=None):
    """
    在半径为 radius 的圆盘内均匀随机采样 n 个点。
    算法：生成二维标准正态变量 (x,y)，归一化到单位圆，
    再乘以 √u（u∼U(0,1)）得到径向分布。
    返回 (n,2) 数组。
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    # 拒绝-free 方法：用正态分布 + 径向缩放
    g = rng.standard_normal((n, 2))
    norms = np.linalg.norm(g, axis=1)
    norms = np.maximum(norms, 1e-15)
    g = g / norms[:, None]
    u = rng.random(n)
    g *= (radius * np.sqrt(u))[:, None]
    return g


def disk_gaussian_sample(n, radius=1.0, seed=None):
    """高斯加权圆盘采样（用于方向性声源）。"""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    # Box-Muller 生成径向高斯分布，截断于 radius
    samples = []
    while len(samples) < n:
        u1, u2 = rng.random(2)
        r = radius * np.sqrt(-2.0 * np.log(u1 + 1e-15))
        theta = 2.0 * np.pi * u2
        if r <= radius:
            samples.append([r * np.cos(theta), r * np.sin(theta)])
    return np.asarray(samples[:n], dtype=np.float64)


# =============================================================================
# 声源初始场生成
# =============================================================================

def gaussian_starter(z, z_s, w0, k0, R_c=np.inf):
    """
    高斯束初始场。
    u(0,z) = exp[−(z−z_s)²/(2w₀²)] · exp[i·k₀·(z−z_s)²/(2R_c)]
    """
    z = np.asarray(z, dtype=np.float64)
    dz = z - z_s
    # === HOLE 1: Gaussian beam starter formula ===
    # TODO: Implement Gaussian beam starter field
    # u(0,z) = exp[-(z-z_s)^2/(2*w0^2)] * exp[i*k0*(z-z_s)^2/(2*R_c)]
    # When R_c is infinite (plane wave), the phase term should be 0.
    raise NotImplementedError("HOLE 1: Gaussian starter formula missing")
    return amplitude * np.exp(1j * phase)


def green_starter(z, z_s, k0):
    """
    Hankel 格林函数初始场（远场渐近）。
    u(0,z) ∝ H₀⁽¹⁾(k₀·|z−z_s|) 的远场近似。
    当 |z−z_s| 很小时截断避免奇点。
    """
    z = np.asarray(z, dtype=np.float64)
    rho = np.abs(z - z_s)
    rho = np.maximum(rho, 1e-6)
    # 远场渐近：H0^(1)(x) ~ sqrt(2/(πx)) * exp(i(x - π/4))
    amp = np.sqrt(2.0 / (np.pi * k0 * rho))
    phase = k0 * rho - np.pi / 4.0
    return amp * np.exp(1j * phase)


def directional_factor(theta, ka):
    """
    圆形活塞声源方向性因子：
    D(θ) = 2·J₁(ka·sinθ) / (ka·sinθ)
    其中 ka = k·a 为归一化孔径半径。
    """
    theta = np.asarray(theta, dtype=np.float64)
    x = ka * np.sin(theta)
    x = np.maximum(np.abs(x), 1e-15)
    # J1(x) 的一阶近似：J1(x) ≈ x/2 − x³/16 + ...
    # 使用 scipy 不可用时，采用级数展开
    j1 = _bessel_j1_series(x)
    return 2.0 * j1 / x


def _bessel_j1_series(x, max_terms=30):
    """J₁(x) 的级数展开：J₁(x) = Σ_{m=0} (−1)^m (x/2)^{2m+1} / [m!(m+1)!]"""
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x, dtype=np.float64)
    for m in range(max_terms):
        sign = (-1) ** m
        num = (x / 2.0) ** (2 * m + 1)
        den = np.exp(gammaln_approx(m + 1) + gammaln_approx(m + 2))
        term = sign * num / den
        result += term
        if np.all(np.abs(term) < 1e-15):
            break
    return result


def gammaln_approx(n):
    """整数 log-gamma 的 Stirling 近似（用于 J1 级数）。"""
    if n <= 1:
        return 0.0
    return (n - 0.5) * np.log(n) - n + 0.5 * np.log(2.0 * np.pi)


def sinc_interpolate(z_query, z_grid, u_grid):
    """
    利用归一化 sinc 函数进行带限插值：
    u(z) = Σ_m u(z_m) · sinc_n((z−z_m)/Δz)
    要求 z_grid 为均匀网格。
    """
    z_query = np.asarray(z_query, dtype=np.float64)
    z_grid = np.asarray(z_grid, dtype=np.float64)
    u_grid = np.asarray(u_grid, dtype=np.complex128)
    dz = z_grid[1] - z_grid[0]
    result = np.zeros_like(z_query, dtype=np.complex128)
    for m, z_m in enumerate(z_grid):
        result += u_grid[m] * sincn_fun((z_query - z_m) / dz)
    return result


def build_initial_field(z_grid, z_s, source_type='gaussian', **kwargs):
    """
    根据指定类型构建初始场。
    参数:
        z_grid: 深度网格 (m)
        z_s: 声源深度 (m)
        source_type: 'gaussian' | 'green' | 'directional'
        kwargs: w0, k0, R_c, ka 等
    """
    k0 = kwargs.get('k0', 2.0 * np.pi * 100.0 / 1500.0)
    if source_type == 'gaussian':
        w0 = kwargs.get('w0', 5.0)
        R_c = kwargs.get('R_c', np.inf)
        return gaussian_starter(z_grid, z_s, w0, k0, R_c)
    elif source_type == 'green':
        return green_starter(z_grid, z_s, k0)
    elif source_type == 'directional':
        # 简化为高斯束 + 方向性包络
        w0 = kwargs.get('w0', 5.0)
        ka = kwargs.get('ka', 10.0)
        u = gaussian_starter(z_grid, z_s, w0, k0)
        # 施加一个角度相关的方向性（简化模型）
        theta = np.arctan2(z_grid - z_s, 1.0)
        u *= directional_factor(theta, ka)
        return u
    else:
        raise ValueError(f"Unknown source_type: {source_type}")


def source_power_normalization(u, z_grid):
    """
    对初始场进行能量归一化：
    ∫ |u(z)|² dz = 1
    使用梯形法则积分。
    """
    z_grid = np.asarray(z_grid, dtype=np.float64)
    u = np.asarray(u, dtype=np.complex128)
    intensity = np.abs(u) ** 2
    power = np.trapezoid(intensity, z_grid)
    if power > 1e-15:
        return u / np.sqrt(power)
    return u
