"""
nonlinear_response.py
非线性响应函数与 Raman 增益辅助 ODE 系统

融合原项目:
  - 1152_squircle_ode: 非线性 ODE 的右端函数构造思想

科学背景:
  在广义非线性薛定谔方程（GNLSE）中，非线性响应函数 R(T) 包含瞬时电子
  响应（Kerr效应）和延迟的 Raman 响应:
      R(T) = (1 - f_R) * delta(T) + f_R * h_R(T)
  其中 f_R ~ 0.18 为 Raman 分数，h_R(T) 为归一化的 Raman 响应函数。

  对于石英光纤，h_R(T) 通常由以下多振子模型描述（Blow & Wood, 1989）:
      h_R(T) = (tau1^2 + tau2^2) / (tau1 * tau2^2) * exp(-T/tau2) * sin(T/tau1)   (T >= 0)
  其中 tau1 ~ 12.2 fs, tau2 ~ 32.0 fs。

  为了高效计算卷积积分，引入辅助变量（辅助 ODE 方法），将非线性响应
  的卷积转化为微分方程的积分，避免每次计算完整卷积。
"""

import numpy as np
from typing import Tuple


def raman_response_blow_wood(t: np.ndarray, tau1: float = 12.2e-15,
                              tau2: float = 32.0e-15) -> np.ndarray:
    """
    计算 Blow-Wood 模型的 Raman 响应函数 h_R(t)。

    公式:
        h_R(t) = A * exp(-t/tau2) * sin(t/tau1)    for t >= 0
        h_R(t) = 0                                  for t < 0
    其中 A = (tau1^2 + tau2^2) / (tau1 * tau2^2)，归一化条件:
        integral_0^inf h_R(t) dt = 1

    物理意义:
        tau1 对应光学声子频率 ~ 1/(2*pi*tau1) ~ 13.1 THz（石英拉曼频移）。
        tau2 为声子阻尼时间，决定拉曼增益谱的线宽。

    Parameters
    ----------
    t : np.ndarray
        时间延迟（s）。
    tau1 : float
        第一弛豫时间（s），默认 12.2 fs。
    tau2 : float
        第二弛豫时间（s），默认 32.0 fs。

    Returns
    -------
    np.ndarray
        Raman 响应函数值（与 t 同形状）。
    """
    if tau1 <= 0.0 or tau2 <= 0.0:
        raise ValueError("raman_response_blow_wood: tau1, tau2 must be > 0")
    A = (tau1 ** 2 + tau2 ** 2) / (tau1 * tau2 ** 2)
    h = np.zeros_like(t, dtype=float)
    mask = t >= 0.0
    h[mask] = A * np.exp(-t[mask] / tau2) * np.sin(t[mask] / tau1)
    return h


def raman_response_lin_agrawal(t: np.ndarray, tau1: float = 12.2e-15,
                                tau2: float = 32.0e-15) -> np.ndarray:
    """
    Lin & Agrawal 修正的 Raman 响应函数（单振子模型改进版）。

    公式:
        h_R(t) = (tau1^2 + tau2^2) / (tau1 * tau2^2) * exp(-t/tau2) * sin(t/tau1)

    与 Blow-Wood 一致，但参数可调整以匹配实验拉曼增益谱。
    拉曼增益系数 g_R 与 h_R 的傅里叶变换关系:
        g_R(omega) = (2 * omega_0 / (n_0 * c)) * f_R * Im{ tilde{h}_R(omega) }

    Parameters
    ----------
    t : np.ndarray
        时间（s）。
    tau1, tau2 : float
        弛豫参数。

    Returns
    -------
    np.ndarray
        Raman 响应。
    """
    return raman_response_blow_wood(t, tau1, tau2)


def nonlinear_response_full(t: np.ndarray, f_R: float = 0.18,
                            tau1: float = 12.2e-15,
                            tau2: float = 32.0e-15) -> np.ndarray:
    """
    完整的非线性响应函数 R(T)。

    公式:
        R(T) = (1 - f_R) * delta(T) + f_R * h_R(T)

    数值实现中，delta 函数在离散网格上不可表示，因此在 SSFM 的
    非线性步中分离处理:
        瞬时部分: (1 - f_R) * |A|^2 * A
        延迟部分: f_R * A * integral h_R(T') |A(T-T')|^2 dT'

    Parameters
    ----------
    t : np.ndarray
        时间延迟。
    f_R : float
        Raman 分数，默认 0.18。
    tau1, tau2 : float
        Raman 响应参数。

    Returns
    -------
    np.ndarray
        响应函数值。
    """
    if not (0.0 <= f_R <= 1.0):
        raise ValueError("nonlinear_response_full: f_R must be in [0, 1]")
    h = raman_response_blow_wood(t, tau1, tau2)
    # delta 函数部分在离散实现中单独处理
    return h


def raman_auxiliary_ode_rhs(y: np.ndarray, t: float,
                             tau1: float, tau2: float,
                             pump_term: float) -> np.ndarray:
    """
    Raman 辅助 ODE 系统的右端函数。

    为了将 Raman 卷积积分转化为 ODE，引入辅助变量:
        u(t) = integral_0^t h_R(t - t') * |A(t')|^2 dt'

    对 h_R(t) = A * exp(-t/tau2) * sin(t/tau1)，可证明 u(t) 满足二阶线性 ODE:
        d^2 u / dt^2 + (2/tau2) * du/dt + (1/tau1^2 + 1/tau2^2) * u
        = A * |A(t)|^2

    等价于一阶系统（状态变量 [u, v=du/dt]）:
        du/dt = v
        dv/dt = -(2/tau2) * v - (1/tau1^2 + 1/tau2^2) * u + A * pump_term

    此 ODE 形式来源于 squircle_ode 的非线性ODE构造思想，
    将复杂的积分-微分方程降维为局部ODE。

    Parameters
    ----------
    y : np.ndarray
        状态向量 [u, v]，形状 (2,)。
    t : float
        当前时间（形式参数，方程自治）。
    tau1, tau2 : float
        Raman 响应参数。
    pump_term : float
        泵浦项 |A(t)|^2。

    Returns
    -------
    np.ndarray
        导数 dy/dt，形状 (2,)。
    """
    A = (tau1 ** 2 + tau2 ** 2) / (tau1 * tau2 ** 2)
    u, v = y[0], y[1]
    omega0_sq = 1.0 / (tau1 ** 2) + 1.0 / (tau2 ** 2)
    damping = 2.0 / tau2
    dudt = v
    dvdt = -damping * v - omega0_sq * u + A * pump_term
    return np.array([dudt, dvdt], dtype=float)


def raman_response_convolution(A_power: np.ndarray, dt: float,
                                f_R: float = 0.18,
                                tau1: float = 12.2e-15,
                                tau2: float = 32.0e-15) -> np.ndarray:
    """
    通过离散卷积计算 Raman 响应项。

    公式:
        h_ram = f_R * dt * conv(h_R, |A|^2)

    使用 FFT 加速卷积计算:
        conv(a, b) = IFFT( FFT(a) * FFT(b) )

    Parameters
    ----------
    A_power : np.ndarray
        |A(T)|^2，时域数组。
    dt : float
        时间步长。
    f_R : float
        Raman 分数。
    tau1, tau2 : float
        Raman 响应参数。

    Returns
    -------
    np.ndarray
        Raman 卷积结果，与 A_power 同长度。
    """
    n = len(A_power)
    if n < 2:
        raise ValueError("raman_response_convolution: need at least 2 points")
    t = np.arange(n) * dt
    h = raman_response_blow_wood(t, tau1, tau2)
    # 使用 FFT 卷积（循环卷积，需补零至 2n-1）
    # TODO Hole 2: 实现基于 FFT 的 Raman 响应离散卷积
    # 公式: h_ram = f_R * dt * conv(h_R, |A|^2)
    # 使用 FFT 加速卷积计算（需补零至长度 2n-1 以避免循环卷积混叠）:
    #   conv(a,b) = IFFT( FFT(a, n_fft) * FFT(b, n_fft) )[:n]
    raise NotImplementedError("Hole 2: 请实现 raman_response_convolution 的 FFT 卷积")


def self_steepening_factor(omega: np.ndarray, omega0: float) -> np.ndarray:
    """
    计算自陡峭（self-steepening）算子的频域因子。

    GNLSE 中的自陡峭项来自非线性极化的时间导数:
        (1 + i/omega0 * d/dT) -> 频域: (1 + omega/omega0)

    公式:
        S(omega) = 1 + omega / omega0

    其中 omega 为相对频率（omega - omega0），在频域中 omega0 为载频。

    Parameters
    ----------
    omega : np.ndarray
        相对角频率数组（rad/s）。
    omega0 : float
        中心角频率（rad/s）。

    Returns
    -------
    np.ndarray
        自陡峭因子。
    """
    if omega0 <= 0.0:
        raise ValueError("self_steepening_factor: omega0 must be > 0")
    return 1.0 + omega / omega0


def shock_operator_time(A: np.ndarray, dt: float, omega0: float) -> np.ndarray:
    """
    在时间域近似计算冲击算子 (1 + i/omega0 * d/dT) 的作用。

    采用中心差分近似时间导数:
        dA/dT ~ (A[k+1] - A[k-1]) / (2*dt)

    Parameters
    ----------
    A : np.ndarray
        复振幅数组。
    dt : float
        时间步长。
    omega0 : float
        中心角频率。

    Returns
    -------
    np.ndarray
        (1 + i/omega0 * d/dT) A
    """
    n = len(A)
    dAdt = np.zeros_like(A, dtype=complex)
    if n >= 3:
        dAdt[1:-1] = (A[2:] - A[:-2]) / (2.0 * dt)
        # 边界采用前向/后向差分
        dAdt[0] = (A[1] - A[0]) / dt
        dAdt[-1] = (A[-1] - A[-2]) / dt
    elif n == 2:
        dAdt[0] = (A[1] - A[0]) / dt
        dAdt[1] = (A[1] - A[0]) / dt
    return A + (1j / omega0) * dAdt
