#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shower_model.py
电磁簇射与强子化能量沉积模型

融合原项目:
- 437_flame_ode: 火焰增长ODE（类比簇射指数增长与饱和）
- 123_burgers_pde_etdrk4: Burgers方程ETD-RK4谱方法（强子化激波模拟）

在BSM信号分析中用于:
- 模拟电磁簇射的快速增长与饱和（flame ODE类比）
- 模拟强子化过程中的非线性能量沉积（Burgers PDE）
"""

import numpy as np
from typing import Tuple


def lambert_w_approx(z: float, max_iter: int = 50) -> float:
    """
    Lambert W 函数主分支 W_0(z) 的数值近似。

    W(z) 定义为 W(z) * exp(W(z)) = z。
    使用 Halley 迭代法求解：
        w_{n+1} = w_n - (w_n exp(w_n) - z) / ((w_n + 1) exp(w_n) - (w_n + 2)(w_n exp(w_n) - z) / (2w_n + 2))

    初始猜测:
        z > e 时:  w_0 = ln(z) - ln(ln(z))
        z ≤ e 时: w_0 = z / e

    Parameters
    ----------
    z : float
        自变量（z ≥ -1/e）
    max_iter : int
        最大迭代次数

    Returns
    -------
    float
        W_0(z) 的近似值
    """
    if z < -1.0 / np.e + 1e-10:
        return -1.0
    if z == 0.0:
        return 0.0

    # 初始猜测
    if z > np.e:
        w = np.log(z) - np.log(np.log(z))
    else:
        w = z / np.e

    for _ in range(max_iter):
        ew = np.exp(w)
        f = w * ew - z
        if abs(f) < 1e-12:
            break
        df = ew * (w + 1.0)
        ddf = ew * (w + 2.0)
        # Halley 迭代
        w = w - f / (df - f * ddf / (2.0 * df))

    return w


def flame_ode_solve(
    t_span: Tuple[float, float],
    y0: float,
    delta: float = 0.0001,
    n_steps: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解火焰增长 ODE，类比电磁簇射的发展。

    火焰 ODE:
        dy/dt = y² - y³ = y²(1 - y)

    在 BSM 探测器物理中的类比解释:
        - y: 归一化簇射能量（0 到 1）
        - t: 辐射长度 X/X_0（归一化深度）
        - y² 项: 簇射的指数增长（对产生 +  bremsstrahlung）
        - y³ 项: 饱和效应（电离损失占主导，低能电子被吸收）

    精确解（含 Lambert W 函数）:
        y(t) = 1 / (W(A * exp(A - t)) + 1)
        其中 A = 1/δ - 1

    Parameters
    ----------
    t_span : Tuple[float, float]
        时间（深度）范围
    y0 : float
        初始值
    delta : float
        初始火焰半径参数
    n_steps : int
        数值步数

    Returns
    -------
    t : np.ndarray
        时间网格
    y : np.ndarray
        解
    """
    t0, tf = t_span
    if tf <= t0:
        raise ValueError("t_span 必须满足 t0 < tf")
    if delta <= 0.0:
        raise ValueError("delta 必须为正")
    if y0 <= 0.0:
        y0 = delta

    dt = (tf - t0) / n_steps
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0

    # 使用隐式梯形法处理刚性
    A_param = 1.0 / delta - 1.0

    for i in range(n_steps):
        # 显式 Euler 预测
        y_pred = y[i] + dt * (y[i] ** 2 * (1.0 - y[i]))
        # 截断到 [0, 1]
        y_pred = np.clip(y_pred, 0.0, 1.0)
        # 隐式梯形校正（一次迭代）
        f_i = y[i] ** 2 * (1.0 - y[i])
        f_pred = y_pred ** 2 * (1.0 - y_pred)
        y[i + 1] = y[i] + 0.5 * dt * (f_i + f_pred)
        y[i + 1] = np.clip(y[i + 1], 0.0, 1.0)

    return t, y


def electromagnetic_shower_profile(
    depth_x0: np.ndarray,
    E0: float = 100.0,
    Ec: float = 0.008  # 8 MeV 临界能量（铅玻璃）
) -> np.ndarray:
    """
    使用火焰ODE类比计算电磁簇射纵向能量沉积剖面。

    标准簇射理论（Rossi 近似 B）:
        N(t) = (E0 / Ec) * exp(t) / (t + ln(E0/Ec))

    用火焰ODE模拟能量从初级电子向次级光子和电子的传递：
        - 初期: 指数增长（y² 项主导）
        - 极大值: t_max ≈ ln(E0/Ec) - 1
        - 后期: 衰减与饱和（y³ 项主导）

    Parameters
    ----------
    depth_x0 : np.ndarray
        深度 t = X / X_0（辐射长度）
    E0 : float
        初级电子能量 [GeV]
    Ec : float
        临界能量 [GeV]

    Returns
    -------
    np.ndarray
        归一化能量沉积剖面
    """
    depth_x0 = np.atleast_1d(depth_x0)
    if E0 <= 0.0 or Ec <= 0.0:
        return np.zeros_like(depth_x0)

    # 火焰ODE参数映射
    # δ ~ Ec/E0（初始能量比例）
    delta = min(Ec / E0, 0.99)
    tf = np.max(depth_x0)

    t_ode, y_ode = flame_ode_solve((0.0, tf), delta, delta, n_steps=5000)

    # 插值到请求的 depth
    profile = np.interp(depth_x0, t_ode, y_ode, left=0.0, right=0.0)

    # 乘以能量因子
    profile *= (E0 / Ec)

    return profile


def burgers_hadronization_pde(
    nx: int = 256,
    nt: int = 200,
    viscosity: float = 0.03,
    t_max: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 Burgers 方程模拟强子化过程中的能量密度激波传播。

    Burgers 方程:
        ∂u/∂t = -1/2 ∂(u²)/∂x + ν ∂²u/∂x²

    物理类比:
        - u(x,t): 能量密度
        - -1/2 ∂(u²)/∂x: 非线性输运（喷注碎裂中的能量再分配）
        - ν ∂²u/∂x²: 粘滞扩散（强子化中的软辐射）

    数值方法: ETD-RK4（指数时间差分 Runge-Kutta 4阶）
    基于 Kassam-Trefethen 方法，在 Fourier 空间处理线性粘性项，
    物理空间处理非线性对流项。

    Parameters
    ----------
    nx : int
        空间网格数（应为2的幂以利用 FFT）
    nt : int
        输出时间步数
    viscosity : float
        粘性系数 ν
    t_max : float
        最大模拟时间

    Returns
    -------
    x : np.ndarray
        空间坐标 [-π, π]
    tt : np.ndarray
        时间坐标
    uu : np.ndarray
        解矩阵 u(x,t)，形状 (nt, nx)
    """
    if nx < 4:
        raise ValueError("nx 必须 >= 4")

    # 空间网格
    x = np.linspace(-np.pi, np.pi, nx + 1)[:-1]

    # 初始条件: 高斯型能量团（模拟硬散射产生的部分子）
    u = np.exp(-10.0 * (np.sin(0.5 * x) ** 2))
    v = np.fft.fft(u)

    # 时间步
    dt = 0.4 / nx ** 2
    nmax = max(int(t_max / dt), nt * 10)
    jstep = max(nmax // (nt - 1), 1)

    # 波数
    k = np.concatenate([np.arange(nx // 2), [0], np.arange(-nx // 2 + 1, 0)])

    # Fourier 乘子（线性粘性项）
    L = 1.0j * viscosity * (k ** 2)
    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.0)

    # 围道积分求 ETD-RK4 系数
    m = 64
    r = np.exp(2.0j * np.pi * (np.arange(m) + 0.5) / m)

    LR = dt * L[:, None] + r[None, :]

    # 避免除零
    LR_safe = np.where(np.abs(LR) < 1e-14, 1e-14, LR)

    Q = dt * np.real(np.mean((np.exp(LR_safe / 2.0) - 1.0) / LR_safe, axis=1))
    f1 = dt * np.real(np.mean(
        (-4.0 - LR_safe + np.exp(LR_safe) * (4.0 - 3.0 * LR_safe + LR_safe ** 2))
        / LR_safe ** 3, axis=1))
    f2 = dt * np.real(np.mean(
        (2.0 + LR_safe + np.exp(LR_safe) * (-2.0 + LR_safe))
        / LR_safe ** 3, axis=1))
    f3 = dt * np.real(np.mean(
        (-4.0 - 3.0 * LR_safe - LR_safe ** 2 + np.exp(LR_safe) * (4.0 - LR_safe))
        / LR_safe ** 3, axis=1))

    # 非线性项乘子
    g = -0.5j * k

    uu = [u.copy()]
    tt = [0.0]

    for i in range(1, nmax + 1):
        t = i * dt

        Nv = g * np.fft.fft(np.real(np.fft.ifft(v)) ** 2)
        a = E2 * v + Q * Nv
        Na = g * np.fft.fft(np.real(np.fft.ifft(a)) ** 2)
        b = E2 * v + Q * Na
        Nb = g * np.fft.fft(np.real(np.fft.ifft(b)) ** 2)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = g * np.fft.fft(np.real(np.fft.ifft(c)) ** 2)

        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3

        if i % jstep == 0 and len(tt) < nt:
            u = np.real(np.fft.ifft(v))
            uu.append(u.copy())
            tt.append(t)

    uu = np.array(uu)
    tt = np.array(tt)

    return x, tt, uu


def hadronization_energy_spectrum(
    parton_energy: float,
    n_particles: int = 100,
    fragmentation_func: str = 'lund'
) -> np.ndarray:
    """
    计算强子化后的能量分布（基于 Lund 弦碎裂模型近似）。

    Lund 碎裂函数近似:
        f(z) = (1/z) * (1 - z)^a * exp(-b m_T^2 / z)

    简化数值实现: 使用 beta 分布近似碎裂函数
        P(z) ∝ z^{α-1} (1-z)^{β-1}

    其中 z = E_hadron / E_parton，α ≈ 0.3, β ≈ 1.5

    Parameters
    ----------
    parton_energy : float
        部分子能量 [GeV]
    n_particles : int
        产生的强子数
    fragmentation_func : str
        碎裂函数类型

    Returns
    -------
    np.ndarray
        各强子的能量 [GeV]
    """
    if parton_energy <= 0.0:
        return np.zeros(n_particles)

    # Beta 分布参数
    alpha = 0.3
    beta = 1.5

    # 使用近似 Gamma 分布采样
    # 若 scipy 不可用，使用 numpy 的 beta
    try:
        from numpy.random import default_rng
        rng = default_rng()
        z = rng.beta(alpha, beta, size=n_particles)
    except Exception:
        # 回退: 均匀分布 + 幂律变换
        u = np.random.uniform(0.0, 1.0, n_particles)
        z = u ** (1.0 / alpha) * (1.0 - u ** (1.0 / beta))
        z = np.clip(z, 0.01, 0.99)

    energies = z * parton_energy

    # 能量守恒修正
    total = np.sum(energies)
    if total > parton_energy:
        energies *= parton_energy / total

    return energies
