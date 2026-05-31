# -*- coding: utf-8 -*-
"""
wave_propagation.py
基于 ode_rk4（四阶 Runge-Kutta）与 rk23（2/3 阶自适应 Runge-Kutta），
实现电磁波在梯度有效折射率超表面中的传播模拟。

核心科学问题：
  超表面可等效为超薄梯度折射率层 n_eff(z)。沿传播方向 z 的波方程退化为
  一阶耦合 ODE 系统（慢变包络近似）：
      dE/dz = i k0 n_eff(z) E(z)
  对于双折射或各向异性超表面，可推广为耦合模式方程：
      dE_x/dz = i k0 (n_xx E_x + n_xy E_y)
      dE_y/dz = i k0 (n_yx E_x + n_yy E_y)

关键公式：
  1. 标量波传播（参考 ode_rk4）:
       k1 = f(t0, u0)
       k2 = f(t0 + h/2, u0 + h·k1/2)
       k3 = f(t0 + h/2, u0 + h·k2/2)
       k4 = f(t0 + h,   u0 + h·k3)
       u_{n+1} = u_n + h/6 (k1 + 2k2 + 2k3 + k4)
  2. RK23 误差估计（参考 rk23）:
       k1 = h·f(t_n, y_n)
       k2 = h·f(t_n + h,   y_n + k1)
       k3 = h·f(t_n + h/2, y_n + k1/4 + k2/4)
       y2 = y_n + (k1 + k2)/2          (2阶)
       y3 = y_n + (k1 + k2 + 4k3)/6    (3阶)
       e_{n+1} = |y3 - y2|
  3. 角谱传播（近轴近似）:
       E(kx, ky, z) = E(kx, ky, 0) · exp(i kz z)
       kz = √(k0² - kx² - ky²)   （倏逝波截断）
  4. 倏逝波边界处理:
       若 kx² + ky² > k0²，则 kz = i √(kx² + ky² - k0²)，
       对应指数衰减：exp(-|kz| z)
"""

import numpy as np


def rk4_integrate(f, t_span, y0, n_steps):
    """
    经典四阶 Runge-Kutta 积分器（参考 ode_rk4）。

    参数:
        f:        callable, f(t, y) -> ndarray
        t_span:   (t0, t1)
        y0:       初始条件 ndarray
        n_steps:  积分步数
    返回:
        t: 长度 n_steps+1 的时间数组
        y: 形状 (n_steps+1, len(y0)) 的解数组
    """
    t0, t1 = float(t_span[0]), float(t_span[1])
    y0 = np.asarray(y0, dtype=complex)
    dt = (t1 - t0) / n_steps
    t = np.linspace(t0, t1, n_steps + 1)
    y = np.zeros((n_steps + 1, y0.shape[0]), dtype=complex)
    y[0] = y0

    for i in range(n_steps):
        ti = t[i]
        ui = y[i]
        k1 = f(ti, ui)
        k2 = f(ti + dt / 2.0, ui + dt * k1 / 2.0)
        k3 = f(ti + dt / 2.0, ui + dt * k2 / 2.0)
        k4 = f(ti + dt, ui + dt * k3)
        y[i + 1] = ui + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return t, y


def rk23_integrate(f, t_span, y0, n_steps):
    """
    显式 RK23 积分器，带局部截断误差估计（参考 rk23）。

    返回:
        t, y, e: 时间、解、误差估计
    """
    t0, t1 = float(t_span[0]), float(t_span[1])
    y0 = np.asarray(y0, dtype=complex)
    dt = (t1 - t0) / n_steps
    t = np.linspace(t0, t1, n_steps + 1)
    m = y0.shape[0]
    y = np.zeros((n_steps + 1, m), dtype=complex)
    e = np.zeros((n_steps + 1, m), dtype=float)
    y[0] = y0
    e[0] = 0.0

    for i in range(n_steps):
        k1 = dt * f(t[i], y[i])
        k2 = dt * f(t[i] + dt, y[i] + k1)
        k3 = dt * f(t[i] + 0.5 * dt, y[i] + 0.25 * k1 + 0.25 * k2)
        y2 = y[i] + 0.5 * (k1 + k2)
        y3 = y[i] + (k1 + k2 + 4.0 * k3) / 6.0
        y[i + 1] = y3
        e[i + 1] = np.abs(y3 - y2)
    return t, y, e


def propagate_plane_wave_scalar(k0, z_span, n_eff_func, E0, n_steps=200):
    """
    利用 RK4 模拟平面波沿 z 通过梯度有效折射率层的传播。

    方程: dE/dz = i k0 n_eff(z) E(z)
    """
    E0 = complex(E0)

    def f(z, E):
        nz = n_eff_func(z)
        # 边界保护：折射率实部不小于 1（真空），虚部非正（无增益）
        if np.isreal(nz):
            nz = complex(max(np.real(nz), 1.0), min(np.imag(nz), 0.0))
        else:
            nz = complex(max(np.real(nz), 1.0), min(np.imag(nz), 0.0))
        return np.array([1j * k0 * nz * E[0]], dtype=complex)

    t, y = rk4_integrate(f, z_span, np.array([E0], dtype=complex), n_steps)
    return t, y[:, 0]


def propagate_coupled_modes(k0, z_span, n_matrix_func, E0_vec, n_steps=200):
    """
    利用 RK4 求解耦合模式方程（双折射/各向异性超表面）。

    dE/dz = i k0 N(z) E(z)
    其中 N(z) 为 2×2 有效折射率矩阵。
    """
    E0_vec = np.asarray(E0_vec, dtype=complex)
    if E0_vec.shape[0] != 2:
        raise ValueError("耦合模式仅支持 2 模式")

    def f(z, E):
        N = n_matrix_func(z)
        N = np.asarray(N, dtype=complex)
        # 数值保护：对角元实部 ≥ 1
        for i in range(2):
            N[i, i] = complex(max(np.real(N[i, i]), 1.0),
                              min(np.imag(N[i, i]), 0.0))
        return 1j * k0 * (N @ E)

    t, y = rk4_integrate(f, z_span, E0_vec, n_steps)
    return t, y


def angular_spectrum_propagate(field, k0, z, dx, dy):
    """
    基于角谱法（Angular Spectrum Method）的标量场传播。

    参数:
        field: 2-D complex array, 输入平面场分布
        k0:    真空波数 2π/λ
        z:     传播距离
        dx, dy: 采样间隔
    返回:
        propagated_field: 2-D complex array
    """
    field = np.asarray(field, dtype=complex)
    ny, nx = field.shape

    # FFT 频率轴
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)

    # 传播常数 kz
    kz2 = k0 ** 2 - KX ** 2 - KY ** 2
    # 倏逝波截断：实部取 0（不衰减也不增长），虚部取负（衰减）
    kz = np.zeros_like(kz2, dtype=complex)
    propagating = kz2 >= 0
    kz[propagating] = np.sqrt(kz2[propagating])
    evanescent = kz2 < 0
    kz[evanescent] = 1j * np.sqrt(-kz2[evanescent])

    # 角谱
    spectrum = np.fft.fft2(field)
    transfer = np.exp(1j * kz * z)
    propagated_spectrum = spectrum * transfer
    propagated_field = np.fft.ifft2(propagated_spectrum)
    return propagated_field


def effective_medium_profile(z, n_substrate, n_air, thickness,
                             profile_type='linear'):
    """
    生成梯度有效折射率剖面 n_eff(z)。
    可选类型: 'linear', 'quadratic', 'exponential'。
    """
    if profile_type == 'linear':
        return n_air + (n_substrate - n_air) * (z / thickness)
    elif profile_type == 'quadratic':
        t = z / thickness
        return n_air + (n_substrate - n_air) * (t ** 2)
    elif profile_type == 'exponential':
        return n_air + (n_substrate - n_air) * (1.0 - np.exp(-3.0 * z / thickness))
    else:
        # 默认线性
        return n_air + (n_substrate - n_air) * (z / thickness)
