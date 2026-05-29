"""
gnlse_propagator.py
广义非线性薛定谔方程（GNLSE）的分步傅里叶传播器

融合原项目:
  - 1135_spiral_pde_movie: 反应扩散 PDE 的时间步进与边界处理
  - 016_arclength: 参数曲线的弧长计算与参数化变换

科学背景:
  超连续谱产生由 GNLSE 描述:
      dA/dz = -alpha/2 * A + hat{D}(A) + i*gamma * N(A)
  其中:
      hat{D} = sum_{m>=2} i^{m+1} * beta_m / m! * d^m/dT^m
      N(A) = (1 + i/omega0 * d/dT) * [ A(z,T) * integral R(T') |A(z,T-T')|^2 dT' ]

  分步傅里叶方法（Split-Step Fourier Method, SSFM）将线性色散步
  与非线性步分离:
      线性步（频域）:  dA/dz = (-alpha/2 + hat{D}) A
                      -> A(z+h,omega) = A(z,omega) * exp[ (-alpha/2 + D(omega)) * h ]
      非线性步（时域）: dA/dz = i*gamma * N(A)
                      -> 通常用 Runge-Kutta 或简单指数积分处理

  弧长参数化（arclength）用于自适应步长控制：当脉冲前沿陡峭化时，
  通过监测脉冲包络的弧长变化动态调整步长，保证数值稳定性。
"""

import numpy as np
from typing import Callable, Optional, Tuple


def dispersion_operator_fft(omega: np.ndarray, alpha: float,
                             beta_coeffs: np.ndarray) -> np.ndarray:
    """
    构建频域色散算子 D(omega)。

    公式:
        D(omega) = -alpha/2 + sum_{m=2}^{M} i^{m+1} * beta_m / m! * omega^m

    其中 omega 为相对角频率（omega - omega0），beta_coeffs[m] = beta_m / m!。

    Parameters
    ----------
    omega : np.ndarray
        相对角频率（rad/s 或 rad/ps，需与 beta 单位一致）。
    alpha : float
        损耗系数（1/m）。
    beta_coeffs : np.ndarray
        泰勒展开系数，beta_coeffs[m] = beta_m / m!。

    Returns
    -------
    np.ndarray
        复色散算子 D(omega)。
    """
    # TODO Hole 1: 实现频域色散算子 D(omega)
    # GNLSE 色散算子公式:
    #   D(omega) = -alpha/2 + sum_{m=2}^{M} i^{m+1} * beta_m / m! * omega^m
    # 注意: beta_coeffs[m] 已经包含 1/m! 因子，即 beta_coeffs[m] = beta_m / m!
    # omega 为相对角频率数组，alpha 为损耗系数
    raise NotImplementedError("Hole 1: 请实现 dispersion_operator_fft 的色散算子构建")


def linear_step_fft(A_freq: np.ndarray, dz: float,
                    D_omega: np.ndarray) -> np.ndarray:
    """
    在频域执行线性传播步。

    公式:
        A(z+dz, omega) = A(z, omega) * exp( D(omega) * dz )

    Parameters
    ----------
    A_freq : np.ndarray
        频域振幅（FFT 顺序）。
    dz : float
        传播步长（m）。
    D_omega : np.ndarray
        色散算子数组。

    Returns
    -------
    np.ndarray
        传播后的频域振幅。
    """
    return A_freq * np.exp(D_omega * dz)


def nonlinear_step_rk4(A_time: np.ndarray, dz: float, gamma: float,
                       omega0: float, dt: float,
                       raman_conv: np.ndarray,
                       use_shock: bool = True) -> np.ndarray:
    """
    使用时域 Runge-Kutta 4 阶方法执行非线性步。

    非线性方程:
        dA/dz = i*gamma * (1 + i/omega0 * d/dT) * [ A * ((1-f_R)*|A|^2 + f_R*conv) ]

    简记非线性算子 N(A)，则 RK4 步进:
        k1 = dz * N(A)
        k2 = dz * N(A + k1/2)
        k3 = dz * N(A + k2/2)
        k4 = dz * N(A + k3)
        A_new = A + (k1 + 2*k2 + 2*k3 + k4) / 6

    Parameters
    ----------
    A_time : np.ndarray
        时域复振幅。
    dz : float
        步长。
    gamma : float
        非线性系数（1/(W*m)）。
    omega0 : float
        中心角频率（rad/s）。
    dt : float
        时间步长（s）。
    raman_conv : np.ndarray
        Raman 卷积项 f_R * integral h_R |A|^2 dT'。
    use_shock : bool
        是否包含自陡峭算子。

    Returns
    -------
    np.ndarray
        更新后的时域振幅。
    """
    def rhs(A):
        pwr = np.abs(A) ** 2
        # 数值保护：限制功率避免溢出
        pwr = np.clip(pwr, 0.0, 1e12)
        raman_local = np.clip(raman_conv, -1e12, 1e12)
        nonlinear_term = A * ((1.0 - 0.18) * pwr + raman_local)
        if use_shock and omega0 > 0.0:
            # 近似冲击算子
            n = len(A)
            dAdt = np.zeros_like(A, dtype=complex)
            if n >= 3:
                dAdt[1:-1] = (A[2:] - A[:-2]) / (2.0 * dt)
                dAdt[0] = (A[1] - A[0]) / dt
                dAdt[-1] = (A[-1] - A[-2]) / dt
            nonlinear_term = nonlinear_term + (1j / omega0) * dAdt * ((1.0 - 0.18) * pwr + raman_local)
        result = 1j * gamma * nonlinear_term
        # 溢出保护
        result = np.where(np.isfinite(result), result, 0.0)
        return result

    k1 = rhs(A_time)
    k2 = rhs(A_time + 0.5 * k1 * dz)
    k3 = rhs(A_time + 0.5 * k2 * dz)
    k4 = rhs(A_time + k3 * dz)
    A_new = A_time + dz * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return A_new


def arclength_parameterization(y: np.ndarray, t: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    计算参数曲线的弧长并生成弧长参数化坐标。

    对于脉冲包络 A(t)，将其视为复平面上的曲线 (Re(A), Im(A))，
    弧长微元:
        ds = sqrt( (dRe/dt)^2 + (dIm/dt)^2 ) dt

    总弧长:
        S = integral_{t1}^{t2} sqrt( |dA/dt|^2 ) dt

    采用梯形法则数值积分:
        S ≈ dt * [ sum sqrt(|dA/dt|^2) - 0.5*first - 0.5*last ]

    该公式直接来源于 arclength_t 的实现。

    Parameters
    ----------
    y : np.ndarray
        复值函数数组（脉冲包络）。
    t : np.ndarray
        时间采样点。

    Returns
    -------
    tuple
        (total_arclength, s_param) 其中 s_param 为归一化弧长参数 [0,1]。
    """
    n = len(t)
    if n < 2:
        return 0.0, np.zeros_like(t)
    dt = t[1] - t[0]
    # 中心差分计算导数
    dydt = np.zeros_like(y, dtype=complex)
    if n >= 3:
        dydt[1:-1] = (y[2:] - y[:-2]) / (2.0 * dt)
        dydt[0] = (y[1] - y[0]) / dt
        dydt[-1] = (y[-1] - y[-2]) / dt
    else:
        dydt[0] = (y[1] - y[0]) / dt
    fx = np.abs(dydt)
    # 梯形积分
    s = dt * (np.sum(fx) - 0.5 * fx[0] - 0.5 * fx[-1])
    # 累积弧长参数
    cum = np.zeros(n)
    for i in range(1, n):
        cum[i] = cum[i - 1] + 0.5 * (fx[i - 1] + fx[i]) * dt
    s_param = cum / (s + 1e-20)
    return float(s), s_param


def adaptive_step_size_estimate(A_time: np.ndarray, t: np.ndarray,
                                 dz_current: float,
                                 z: float, z_target: float,
                                 min_dz: float = 1e-6,
                                 max_dz: float = 1e-2) -> float:
    """
    基于弧长变化率的自适应步长估计。

    策略:
        1. 计算当前脉冲包络的弧长 S。
        2. 若弧长变化剧烈（|dS/dz| 大），减小步长。
        3. 若接近传播终点，限制步长不超过剩余距离。

    公式:
        dz_new = dz_current * (S_target / (S + epsilon))^{0.5}

    其中 S_target 为经验目标弧长。

    Parameters
    ----------
    A_time : np.ndarray
        当前时域振幅。
    t : np.ndarray
        时间数组。
    dz_current : float
        当前步长。
    z : float
        当前传播距离。
    z_target : float
        目标传播距离。
    min_dz, max_dz : float
        步长上下界。

    Returns
    -------
    float
        建议的下一步长。
    """
    S, _ = arclength_parameterization(A_time, t)
    S_target = 1.0  # 归一化目标弧长
    ratio = S_target / (S + 1e-10)
    dz_new = dz_current * np.sqrt(np.clip(ratio, 0.1, 10.0))
    # 限制范围
    dz_new = np.clip(dz_new, min_dz, max_dz)
    # 不超出终点
    remaining = z_target - z
    if remaining > 0:
        dz_new = min(dz_new, remaining)
    return float(dz_new)


def ssfm_propagate(A0_time: np.ndarray, t: np.ndarray, z_target: float,
                   alpha: float, gamma: float, beta_coeffs: np.ndarray,
                   omega0: float, f_R: float = 0.18,
                   tau1: float = 12.2e-15, tau2: float = 32.0e-15,
                   dz_initial: float = 1e-4,
                   n_z_records: int = 100,
                   use_symmetrized: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用对称分步傅里叶法（SSFM）求解 GNLSE。

    对称 SSFM（Straight-Step 的改进）:
        A(z+dz/2) = exp( hat{D} * dz/2 ) * A(z)           [半线性步]
        A(z+dz)   = exp( i*gamma*N*dz ) * A(z+dz/2)       [完整非线性步]
        A(z+dz)   = exp( hat{D} * dz/2 ) * A(z+dz)        [半线性步]

    频域实现中，半线性步对应频域相位因子:
        A_freq <- A_freq * exp( D(omega) * dz/2 )

    参数:
        A0_time : 初始时域包络（复数组）
        t       : 时间窗口（s），等间距
        z_target: 总传播距离（m）
        alpha   : 损耗系数（1/m）
        gamma   : 非线性系数（1/(W*m)）
        beta_coeffs: 色散泰勒系数
        omega0  : 中心角频率（rad/s）
        f_R, tau1, tau2: Raman 参数
        dz_initial: 初始步长
        n_z_records: z方向记录点数
        use_symmetrized: 是否使用对称 SSFM

    返回:
        z_out   : 记录位置数组，形状 (n_z_records,)
        A_z     : 各位置上的时域振幅，形状 (n_z_records, n_t)
        spec_z  : 各位置上的功率谱，形状 (n_z_records, n_t)
    """
    n_t = len(t)
    dt = t[1] - t[0]
    # 频率轴
    omega = 2.0 * np.pi * np.fft.fftfreq(n_t, dt)
    # 色散算子
    D_omega = dispersion_operator_fft(omega, alpha, beta_coeffs)

    # 预分配输出
    z_out = np.linspace(0.0, z_target, n_z_records)
    A_z = np.zeros((n_z_records, n_t), dtype=complex)
    spec_z = np.zeros((n_z_records, n_t), dtype=float)

    A = A0_time.copy()
    z = 0.0
    dz = dz_initial
    record_idx = 0

    # 记录初始态
    A_z[0, :] = A
    spec_z[0, :] = np.abs(np.fft.fft(A)) ** 2

    while z < z_target and dz > 1e-12:
        # 确保不超出终点
        if z + dz > z_target:
            dz = z_target - z

        # 线性半步（频域）
        if use_symmetrized:
            A_freq = np.fft.fft(A)
            A_freq = A_freq * np.exp(D_omega * dz * 0.5)
            A = np.fft.ifft(A_freq)

        # Raman 卷积（基于当前振幅）
        pwr = np.abs(A) ** 2
        if f_R > 0.0:
            from nonlinear_response import raman_response_convolution
            raman_conv = raman_response_convolution(pwr, dt, f_R, tau1, tau2)
        else:
            raman_conv = np.zeros_like(pwr)

        # 非线性全步（时域 RK4）
        A = nonlinear_step_rk4(A, dz, gamma, omega0, dt, raman_conv, use_shock=True)

        # 线性半步（频域）
        if use_symmetrized:
            A_freq = np.fft.fft(A)
            A_freq = A_freq * np.exp(D_omega * dz * 0.5)
            A = np.fft.ifft(A_freq)
        else:
            # 非对称：完整线性步在非线性后
            A_freq = np.fft.fft(A)
            A_freq = A_freq * np.exp(D_omega * dz)
            A = np.fft.ifft(A_freq)

        z += dz

        # 记录
        if record_idx + 1 < n_z_records and z >= z_out[record_idx + 1]:
            record_idx += 1
            A_z[record_idx, :] = A
            spec_z[record_idx, :] = np.abs(np.fft.fft(A)) ** 2

        # 自适应步长（基于弧长）
        dz = adaptive_step_size_estimate(A, t, dz, z, z_target)

    # 补齐最后记录
    while record_idx + 1 < n_z_records:
        record_idx += 1
        A_z[record_idx, :] = A
        spec_z[record_idx, :] = np.abs(np.fft.fft(A)) ** 2

    return z_out, A_z, spec_z


def sech_pulse(t: np.ndarray, T0: float, P0: float,
               C: float = 0.0, omega_shift: float = 0.0) -> np.ndarray:
    """
    生成双曲正割型初始脉冲。

    公式:
        A(0,T) = sqrt(P0) * sech(T/T0) * exp( -i*C*T^2 / (2*T0^2) + i*omega_shift*T )

    其中:
        P0 为峰值功率（W）
        T0 为脉冲半宽（1/e 强度处为 T0 * arccosh(sqrt(2))）
        C  为啁啾参数（C>0 正常色散啁啾）

    Parameters
    ----------
    t : np.ndarray
        时间数组。
    T0 : float
        特征脉宽（s）。
    P0 : float
        峰值功率（W）。
    C : float
        啁啾参数。
    omega_shift : float
        频率偏移（rad/s）。

    Returns
    -------
    np.ndarray
        复振幅数组。
    """
    if T0 <= 0.0:
        raise ValueError("sech_pulse: T0 must be > 0")
    envelope = np.sqrt(P0) / np.cosh(t / T0)
    phase = -0.5 * C * (t / T0) ** 2 + omega_shift * t
    return envelope * np.exp(1j * phase)


def gaussian_pulse(t: np.ndarray, T0: float, P0: float,
                   C: float = 0.0) -> np.ndarray:
    """
    生成高斯型初始脉冲。

    公式:
        A(0,T) = sqrt(P0) * exp( - (1 + i*C) * T^2 / (2*T0^2) )

    FWHM 与 T0 的关系: FWHM = 2*sqrt(ln2) * T0 ~ 1.665 * T0。

    Parameters
    ----------
    t : np.ndarray
        时间数组。
    T0 : float
        1/e 半宽。
    P0 : float
        峰值功率。
    C : float
        啁啾参数。

    Returns
    -------
    np.ndarray
        复振幅。
    """
    if T0 <= 0.0:
        raise ValueError("gaussian_pulse: T0 must be > 0")
    return np.sqrt(P0) * np.exp(-(1.0 + 1j * C) * (t ** 2) / (2.0 * T0 ** 2))
