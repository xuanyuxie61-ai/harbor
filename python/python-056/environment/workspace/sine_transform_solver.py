"""
sine_transform_solver.py
================================================================================
正弦变换快速PDE求解模块 (来源于 1085_sine_transform 项目)
================================================================================
本模块实现离散正弦变换 (DST) 及其反变换，用于快速求解带Dirichlet
边界条件的泊松方程和亥姆霍兹方程。在潮汐能提取问题中，正弦变换
用于高效求解海底地形修正的流速势以及涡轮尾流场的快速泊松求解器。

核心公式:
    离散正弦变换 (DST-I):
        S(k) = √(2/(N+1)) · Σ_{j=1}^{N} sin(π·k·j/(N+1)) · d(j)

    逆变换 (与正变换相同，对称):
        d(j) = √(2/(N+1)) · Σ_{k=1}^{N} sin(π·k·j/(N+1)) · S(k)

    泊松方程快速求解:
        ∂²u/∂x² = f(x),  u(0)=u(L)=0
        离散后:  D² u = f
        利用 DST 对角化:  Λ = diag(-4/h² sin²(πk/(2(N+1))))
        u = IDST( DST(f) / Λ )

    亥姆霍兹方程:
        ∇²u - κ²u = f
        对角元素:  Λ_k = -4/h² sin²(πk/(2(N+1))) - κ²
"""

import numpy as np
from typing import Tuple


def sine_transform_data(n: int, d: np.ndarray) -> np.ndarray:
    """
    对向量执行离散正弦变换 (DST-I)。

    参数:
        n: 数据点数
        d: 输入数据，长度 n

    返回:
        s: 正弦变换系数，长度 n
    """
    d = np.asarray(d, dtype=float).flatten()
    if d.size < n:
        raise ValueError("sine_transform_data: 输入数据长度不足")
    s = np.zeros(n)
    coeff = np.pi / (n + 1)
    scale = np.sqrt(2.0 / (n + 1))
    for k in range(1, n + 1):
        angles = coeff * k * np.arange(1, n + 1)
        s[k - 1] = np.sum(np.sin(angles) * d[:n])
    s *= scale
    return s


def inverse_sine_transform(n: int, s: np.ndarray) -> np.ndarray:
    """
    离散正弦变换的逆变换 (DST-I 是自逆的，相差比例因子)。

    参数:
        n: 数据点数
        s: 变换系数，长度 n

    返回:
        d: 原始数据
    """
    return sine_transform_data(n, s)


def dst_fast(d: np.ndarray) -> np.ndarray:
    """
    基于 FFT 的快速离散正弦变换 (DST-I)。

    算法:
        将 DST 映射到 DFT 计算:
            构造对称扩展序列 y = [0, d, 0, -d_rev]
            Y = FFT(y)
            S = -imag(Y[1:N+1]) / 2

    参数:
        d: 输入数据

    返回:
        DST-I 系数
    """
    d = np.asarray(d, dtype=float)
    n = d.size
    y = np.zeros(2 * n + 2)
    y[1:n + 1] = d
    y[n + 2:2 * n + 2] = -d[::-1]
    y_fft = np.fft.fft(y)
    s = -np.imag(y_fft[1:n + 1]) / 2.0
    # 归一化
    scale = np.sqrt(2.0 / (n + 1))
    return s * scale


def solve_poisson_1d(
    f: np.ndarray,
    L: float = 1.0,
) -> np.ndarray:
    """
    使用正弦变换求解一维泊松方程 u'' = f, u(0)=u(L)=0。

    公式:
        设 h = L/(N+1),  x_j = j·h
        离散 Laplacian 的特征值:
            λ_k = -4/h² · sin²(π·k/(2(N+1)))

    参数:
        f: 右端项数组，长度 N
        L: 区间长度

    返回:
        u: 解数组
    """
    f = np.asarray(f, dtype=float)
    N = f.size
    h = L / (N + 1)
    s_f = dst_fast(f)
    k = np.arange(1, N + 1)
    lam = -4.0 / (h * h) * np.sin(np.pi * k / (2.0 * (N + 1))) ** 2
    # 边界处理
    lam_safe = np.where(np.abs(lam) < 1e-14, 1.0, lam)
    s_u = s_f / lam_safe
    u = dst_fast(s_u)
    return u


def solve_helmholtz_1d(
    f: np.ndarray,
    kappa: float,
    L: float = 1.0,
) -> np.ndarray:
    """
    使用正弦变换求解一维亥姆霍兹方程 u'' - κ²u = f, u(0)=u(L)=0。

    公式:
        λ_k = -4/h² · sin²(π·k/(2(N+1))) - κ²

    参数:
        f: 右端项
        kappa: 波数参数
        L: 区间长度

    返回:
        u: 解数组
    """
    f = np.asarray(f, dtype=float)
    N = f.size
    h = L / (N + 1)
    s_f = dst_fast(f)
    k = np.arange(1, N + 1)
    lam = -4.0 / (h * h) * np.sin(np.pi * k / (2.0 * (N + 1))) ** 2 - kappa * kappa
    s_u = s_f / lam
    u = dst_fast(s_u)
    return u


def compute_wake_potential(
    thrust_distribution: np.ndarray,
    domain_length: float = 100.0,
    viscosity_scale: float = 0.01,
) -> np.ndarray:
    """
    计算涡轮尾流场的速度势修正。

    物理模型:
        将涡轮视为动量源项，尾流场满足修正的泊松方程:
            ∇²φ = -∇·(thrust / ρ)
        通过正弦变换快速求解。

    参数:
        thrust_distribution: 沿流向的推力分布 (N)
        domain_length: 计算域长度 (m)
        viscosity_scale: 粘性尺度

    返回:
        potential: 速度势
    """
    # 归一化推力分布
    rhs = -np.gradient(thrust_distribution, domain_length / len(thrust_distribution))
    return solve_poisson_1d(rhs, L=domain_length)
