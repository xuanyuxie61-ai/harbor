"""
equatorial_wave_solver.py
=========================
基于 burgers_pde_etdrk4 (123_burgers_pde_etdrk4) 的指数时间差分 Runge-Kutta 4 阶
(ETDRK4) 谱方法，求解赤道海洋中的非线性平流-扩散方程，模拟 Kelvin 波与 Rossby 波
在 ENSO 充放电过程中的传播与耗散。

科学背景
--------
赤道波动动力学是 ENSO 的核心理论基础。根据 Matsuno (1966) 和 Gill (1980) 的理论，
赤道 β 平面上的线性波动方程存在三类解：Kelvin 波（向东传播、无纬度截断）、
Rossby 波（向西传播、存在纬度截断）和 Yanai 波（混合波）。

在 ENSO 的 recharge-discharge 框架中（Jin, 1997）：
- 风应力异常激发赤道 Kelvin 波，东传并改变东太平洋温跃层深度；
- 同时激发 off-equatorial Rossby 波，西传并在西边界反射为 Kelvin 波；
- 反射 Kelvin 波携带的温跃层信号在充放电时间尺度 τ_R ≈ (2n+1)π / (2β c) 后
  返回东太平洋，完成一个 ENSO 循环。

核心公式
--------
1. 赤道 β 平面下的非线性浅水方程（经向模态投影后）：
   
   ∂u/∂t + u * ∂u/∂x = -g' * ∂h/∂x + A_H * ∂²u/∂x² - r * u + τ^x / (ρ_0 * H)
   
   ∂h/∂t + ∂(h*u)/∂x = -H * ∂u/∂x + A_H * ∂²h/∂x² - ε * h

   其中 u 为纬向流速，h 为温跃层深度异常，g' 为约化重力，
   A_H 为水平涡扩散系数，r 为 Rayleigh 摩擦系数，
   ε 为 Newton 冷却系数，τ^x 为纬向风应力异常。

2. 为简化分析，考虑标量 Burgers 型方程作为波动振幅包络的近似：
   
   ∂u/∂t = - (1/2) * ∂(u²)/∂x + ν * ∂²u/∂x² + F(x, t)

   该方程在周期边界 [-π, π] 上通过 ETDRK4 谱方法求解。

3. ETDRK4 时间 stepping（Kassam & Trefethen, 2005）：
   在 Fourier 空间，线性算子 L = i * ν * k²，非线性项 N(u) 在物理空间计算。
   
   令 v = FFT(u)，则：
   
   v^{n+1} = E * v^n + N_v^n * f1 + 2*(N_a + N_b) * f2 + N_c * f3

   其中 E = exp(Δt * L)，系数 f1, f2, f3, Q 通过围道积分计算：
   
   Q  = Δt * Re[ mean( (exp(LR/2) - 1) / LR ) ]
   f1 = Δt * Re[ mean( (-4 - LR + exp(LR)*(4 - 3LR + LR²)) / LR³ ) ]
   f2 = Δt * Re[ mean( (2 + LR + exp(LR)*(-2 + LR)) / LR³ ) ]
   f3 = Δt * Re[ mean( (-4 - 3LR - LR² + exp(LR)*(4 - LR)) / LR³ ) ]

   LR = Δt * L(:, ones(m,1)) + r(ones(nx,1), :)
   r = exp(2πi * ((1:m) - 0.5) / m)  （单位圆上的 m 个根）

4. 相速度关系：
   Kelvin 波：c_K = √(g' * H)
   Rossby 波（n=1）：c_R = -c_K / 3
   充放电周期：T_R ≈ 2L / |c_R| + L / c_K ≈ 2–4 年
"""

import numpy as np
from typing import Tuple, Callable, Optional


def etdrk4_coefficients(nx: int, dt: float, vis: float, m: int = 64) -> Tuple:
    """
    预计算 ETDRK4 方法的系数。

    参数
    ----
    nx : int
        空间节点数（必须为偶数）。
    dt : float
        时间步长。
    vis : float
        粘性/扩散系数。
    m : int
        围道积分点数。

    返回
    ----
    k, L, E, E2, Q, f1, f2, f3, g : 各种 Fourier 乘子和系数。
    """
    if nx % 2 != 0:
        raise ValueError("nx must be even for FFT")
    if vis < 0:
        raise ValueError("viscosity must be non-negative")
    if dt <= 0:
        raise ValueError("dt must be positive")

    # 波数
    k = np.concatenate([
        np.arange(0, nx // 2),
        np.array([0]),
        np.arange(-nx // 2 + 1, 0)
    ])

    # 线性算子 L = i * vis * k²（Burgers 方程的扩散部分在 Fourier 空间）
    L = 1j * vis * k ** 2

    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.0)

    # 围道积分
    r = np.exp(2.0j * np.pi * (np.arange(1, m + 1) - 0.5) / m)

    # LR 矩阵: (nx, m)
    LR = dt * L[:, np.newaxis] + r[np.newaxis, :]

    # 避免除零：对于 LR=0 的情况，用极限值替换
    Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean(
        (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR ** 2)) / LR ** 3,
        axis=1
    ))
    f2 = dt * np.real(np.mean(
        (2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR ** 3,
        axis=1
    ))
    f3 = dt * np.real(np.mean(
        (-4.0 - 3.0 * LR - LR ** 2 + np.exp(LR) * (4.0 - LR)) / LR ** 3,
        axis=1
    ))

    # 非线性项的 Fourier 乘子: g = -0.5 * i * k
    g = -0.5j * k

    return k, L, E, E2, Q, f1, f2, f3, g


def solve_burgers_etdrk4(nx: int, nt: int, vis: float,
                         tmax: float = 1.0,
                         forcing: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 ETDRK4 谱方法求解 Burgers 方程。

    方程：
    u_t = -0.5 * d/dx(u²) + vis * u_{xx} + F(x, t)

    初始条件：
    u(x, 0) = exp(-10 * sin²(0.5 * x))

    参数
    ----
    nx : int
        空间节点数。
    nt : int
        输出时间步数。
    vis : float
        粘性系数。
    tmax : float
        总模拟时间。
    forcing : callable, optional
        外力项 F(x, t)。

    返回
    ----
    x : np.ndarray, shape (nx,)
        空间网格。
    tt : np.ndarray, shape (nt,)
        输出时间值。
    uu : np.ndarray, shape (nx, nt)
        时空解场。
    """
    if nx < 4:
        raise ValueError("nx must be at least 4")
    if nt < 2:
        raise ValueError("nt must be at least 2")

    # 空间网格
    x = np.linspace(-np.pi, np.pi, nx + 1)[:-1]

    # 初始条件
    u = np.exp(-10.0 * np.sin(0.5 * x) ** 2)
    v = np.fft.fft(u)

    # 时间步长（CFL 稳定性限制）
    dt = 0.4 / nx ** 2
    if vis > 0.01:
        dt = min(dt, 0.4 / (vis * nx ** 2))

    nmax = max(1, int(np.ceil(tmax / dt)))
    jstep = max(1, nmax // (nt - 1)) if nt > 1 else 1

    # 预计算 ETDRK4 系数
    _, _, E, E2, Q, f1, f2, f3, g = etdrk4_coefficients(nx, dt, vis)

    # 存储
    uu = np.zeros((nx, nt), dtype=float)
    tt = np.zeros(nt, dtype=float)
    uu[:, 0] = u
    tt[0] = 0.0

    out_idx = 1

    for i in range(1, nmax + 1):
        t = i * dt

        # 非线性项
        u_phys = np.real(np.fft.ifft(v))
        Nv = g * np.fft.fft(u_phys ** 2)

        if forcing is not None:
            f_vec = forcing(x, t)
            Nv += dt * np.fft.fft(f_vec) / nx  # 归一化

        a = E2 * v + Q * Nv
        Na = g * np.fft.fft(np.real(np.fft.ifft(a)) ** 2)

        b = E2 * v + Q * Na
        Nb = g * np.fft.fft(np.real(np.fft.ifft(b)) ** 2)

        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = g * np.fft.fft(np.real(np.fft.ifft(c)) ** 2)

        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3

        # 输出
        if out_idx < nt and i % jstep == 0:
            u_out = np.real(np.fft.ifft(v))
            uu[:, out_idx] = u_out
            tt[out_idx] = t
            out_idx += 1

    # 填充剩余输出点
    while out_idx < nt:
        uu[:, out_idx] = np.real(np.fft.ifft(v))
        tt[out_idx] = tmax
        out_idx += 1

    return x, tt, uu


def kelvin_wave_amplitude(x: np.ndarray, t: float,
                          c_k: float = 2.5,  # m/s
                          decay: float = 0.1,
                          width: float = 0.5) -> np.ndarray:
    """
    构造解析 Kelvin 波包振幅。

    公式：
    u_K(x, t) = A * exp(-(x - c_K * t)² / (2 * w²)) * exp(-decay * t)

    参数
    ----
    x : np.ndarray
        空间坐标（无量纲，或经度弧度）。
    t : float
        时间。
    c_k : float
        Kelvin 波相速度（m/s 或无量纲）。
    decay : float
        衰减率。
    width : float
        波包宽度。

    返回
    ----
    amplitude : np.ndarray
        波振幅。
    """
    return np.exp(-((x - c_k * t) ** 2) / (2.0 * width ** 2)) * np.exp(-decay * t)


def rossby_wave_amplitude(x: np.ndarray, t: float,
                          c_r: float = -0.8,
                          decay: float = 0.05,
                          width: float = 0.8) -> np.ndarray:
    """
    构造解析 Rossby 波包振幅（n=1 模态）。

    公式：
    u_R(x, t) = A * exp(-(x - c_R * t)² / (2 * w²)) * exp(-decay * t)

    注意 c_R < 0 表示向西传播。
    """
    return np.exp(-((x - c_r * t) ** 2) / (2.0 * width ** 2)) * np.exp(-decay * t)


def solve_coupled_wave_envelope(nx: int, nt: int,
                                vis: float = 0.03,
                                tmax: float = 5.0,
                                coupling_strength: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    求解耦合 Kelvin-Rossby 波包络方程，作为 ENSO 充放电动力学的简化模型。

    方程：
    u_t = -0.5 * d/dx(u²) + vis * u_{xx} + S(x, t)

    其中源项 S(x, t) 模拟风应力激发的 Kelvin 波与反射 Rossby 波的叠加：
    S(x, t) = coupling_strength * (kelvin_wave + rossby_wave)

    返回
    ----
    x, tt, uu : 时空解场。
    """
    def forcing(x_arr, t_val):
        k = kelvin_wave_amplitude(x_arr, t_val)
        r = rossby_wave_amplitude(x_arr, t_val)
        return coupling_strength * (k + r)

    return solve_burgers_etdrk4(nx, nt, vis, tmax, forcing)


def wave_energy(uu: np.ndarray, dx: float) -> np.ndarray:
    """
    计算波场的总能量时间序列。

    公式：
    E(t) = (1/2) * ∫ u(x,t)² dx

    参数
    ----
    uu : np.ndarray, shape (nx, nt)
        时空解场。
    dx : float
        空间步长。

    返回
    ----
    energy : np.ndarray, shape (nt,)
        能量时间序列。
    """
    return 0.5 * np.sum(uu ** 2, axis=0) * dx


def recharge_discharge_timescale(c_k: float, c_r: float, basin_width: float) -> float:
    """
    计算 ENSO recharge-discharge 时间尺度。

    公式（Jin, 1997）：
    τ_R = L / c_K + 2L / |c_R|

    其中 L 为太平洋 basin 宽度。

    参数
    ----
    c_k : float
        Kelvin 波速度 (m/s)。
    c_r : float
        Rossby 波速度 (m/s)。
    basin_width : float
         basin 宽度 (m)。

    返回
    ----
    tau_r : float
        充放电时间尺度 (s)。
    """
    if c_k <= 0 or abs(c_r) <= 0:
        raise ValueError("Wave speeds must be non-zero")
    tau_r = basin_width / c_k + 2.0 * basin_width / abs(c_r)
    return tau_r
