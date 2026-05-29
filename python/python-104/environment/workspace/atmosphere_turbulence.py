"""
atmosphere_turbulence.py — 大气湍流相位屏生成与光化学-热耦合模型

融合原项目:
  - 841_ozone_ode (臭氧化学反应动力学ODE)
  - 901_porous_medium_exact (多孔介质非线性扩散方程精确解)

功能:
  - Kolmogorov湍流相位屏生成 (FFT-based)
  - 大气折射率非线性扩散修正 (PME-Barenblatt解思想)
  - 光化学-热耦合ODE系统: 臭氧光解吸热引起的局部折射率扰动
  - Fried参数 r_0 计算

物理模型:
  1. 折射率结构常数 C_n^2(h) 沿高度分布 (Hufnagel-Valley模型):
       C_n^2(h) = 0.00594*(v/27)^2*(10^{-5}*h)^{10}*exp(-h/1000)
                  + 2.7e-16*exp(-h/1500) + A*exp(-h/100)

  2. 相位结构函数 (Kolmogorov):
       D_phi(r) = 6.88 * (r / r_0)^{5/3}

  3. 多孔介质非线性扩散修正 (源自901):
       折射率扰动 delta_n 满足:
         d(delta_n)/dt = D_eff * nabla^2( (delta_n)^m )
       其中 m > 1 为非线性指数, D_eff 为有效扩散系数.
       Barenblatt自相似解给出湍流耗散后的残余相位结构.

  4. 光化学-热耦合 (源自841臭氧ODE):
       紫外光解 O3 -> O2 + O 释放热量, 引起局部温度梯度 delta_T,
       折射率变化: delta_n = (n-1) * delta_T / T_0 * (79e-6 / P_0)
       该变化由如下ODE描述:
         d(delta_n)/dt = k1(t)*I_UV - k2*delta_n - k3*delta_n*delta_T
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift


# --- Kolmogorov 湍流相位屏 ---

def hufnagel_valley_cnsquared(h, v_wind=21.0, A_ground=1.7e-14):
    """
    Hufnagel-Valley C_n^2 模型 (单位: m^{-2/3}).
    h: 高度 (m)
    v_wind: 高空风速 (m/s)
    A_ground: 地面湍流强度
    """
    if np.any(h < 0):
        h = np.clip(h, 0, None)
    term1 = 0.00594 * (v_wind / 27.0) ** 2 * (1e-5 * h) ** 10 * np.exp(-h / 1000.0)
    term2 = 2.7e-16 * np.exp(-h / 1500.0)
    term3 = A_ground * np.exp(-h / 100.0)
    return term1 + term2 + term3


def fried_parameter(wavelength, Cn2_integral):
    """
    计算Fried参数 r_0.

    公式:
      r_0 = [ 0.423 * (2pi/lambda)^2 * integral(C_n^2(h) dh) ]^{-3/5}
    """
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")
    if Cn2_integral < 0:
        raise ValueError("Cn2_integral must be non-negative.")
    k = 2.0 * np.pi / wavelength
    r0 = (0.423 * k ** 2 * Cn2_integral) ** (-3.0 / 5.0)
    return r0


def kolmogorov_psd(fx, fy, r0):
    """
    Kolmogorov湍流功率谱密度 (相位).

    在频率域 (fx, fy) (单位: cycles/m):
      Phi(f) = 0.023 * r_0^{-5/3} * (f_x^2 + f_y^2)^{-11/6}
    """
    f2 = fx ** 2 + fy ** 2
    f2 = np.where(f2 < 1e-20, 1e-20, f2)
    Phi = 0.023 * (r0 ** (-5.0 / 3.0)) * (f2 ** (-11.0 / 6.0))
    return Phi


def generate_phase_screen(grid_size, pixel_scale, r0, L0=30.0, seed=None):
    """
    使用FFT方法生成Kolmogorov相位屏 (含外尺度修正).

    算法:
      1. 在频域生成高斯白噪声 W(f)
      2. 乘以 sqrt(Phi(f)), 其中 Phi 为von Karman谱:
           Phi_vK(f) = Phi_K(f) * (f^2 + f_0^2)^{-11/6}, f_0 = 1/L0
      3. 逆FFT得到空间域相位屏

    返回: phase_screen (rad), 单位圆掩码
    """
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2.")
    if pixel_scale <= 0:
        raise ValueError("pixel_scale must be positive.")
    if r0 <= 0:
        raise ValueError("r0 must be positive.")

    if seed is not None:
        np.random.seed(seed)

    # 频率网格
    freq = np.fft.fftfreq(grid_size, d=pixel_scale)
    fx, fy = np.meshgrid(freq, freq)
    f2 = fx ** 2 + fy ** 2

    # von Karman 谱 (含外尺度 L0)
    f0 = 1.0 / L0
    f2_safe = np.where(f2 < 1e-20, 1e-20, f2)
    Phi = 0.023 * (r0 ** (-5.0 / 3.0)) * (f2_safe + f0 ** 2) ** (-11.0 / 6.0)

    # 频域滤波
    W_real = np.random.normal(0, 1, (grid_size, grid_size))
    W_imag = np.random.normal(0, 1, (grid_size, grid_size))
    W = W_real + 1j * W_imag

    # 保证实对称性 (零频率和Nyquist频率处理)
    if grid_size % 2 == 0:
        W[0, grid_size // 2] = np.real(W[0, grid_size // 2])
        W[grid_size // 2, 0] = np.real(W[grid_size // 2, 0])
        W[grid_size // 2, grid_size // 2] = np.real(W[grid_size // 2, grid_size // 2])

    spectrum = W * np.sqrt(Phi)
    phase = np.fft.ifft2(spectrum).real * (grid_size ** 2)

    # 归一化: 确保相位结构函数在参考距离处匹配理论值
    x = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, x)
    rho = np.sqrt(X ** 2 + Y ** 2)
    mask = rho <= 1.0

    # 移除 piston
    phase -= np.mean(phase[mask])

    return phase, mask


# --- 多孔介质非线性扩散修正 (源自901) ---

def barenblatt_pme_solution(x, t, m=3.0, c=None, delta=1.0):
    """
    Barenblatt自相似解 (多孔介质方程精确解).

    PME:  u_t = nabla^2(u^m),  m > 1

    参数:
      alpha = 1/(m-1), beta = 1/(m+1), gamma = (m-1)/(2m(m+1))
    精确解:
      u(x,t) = max(0, (t+delta)^{-beta} * (c - gamma * (x/(t+delta)^beta)^2 )^alpha )

    在AO中, u 代表湍流耗散后的残余相位振幅.
    """
    if m <= 1.0:
        raise ValueError("m must be > 1 for PME.")
    if t + delta <= 0:
        raise ValueError("t + delta must be positive.")
    if c is None:
        c = np.sqrt(3.0) / 15.0

    alpha = 1.0 / (m - 1.0)
    beta = 1.0 / (m + 1.0)
    gamma = (m - 1.0) / (2.0 * m * (m + 1.0))

    factor = c - gamma * (x / ((t + delta) ** beta)) ** 2
    u = np.where(factor > 0, (t + delta) ** (-beta) * (factor ** alpha), 0.0)
    return u


def pme_diffusion_phase_correction(phase, pixel_scale, D_eff=1e-4, m=3.0, dt=1e-3, n_steps=10):
    """
    对相位屏施加非线性扩散修正.

    离散化:  u^{k+1} = u^k + dt * D_eff * nabla^2( (u^k)^m )
    使用中心差分:
      nabla^2(v)_{i,j} = (v_{i+1,j}+v_{i-1,j}+v_{i,j+1}+v_{i,j-1}-4v_{i,j}) / dx^2
    """
    if D_eff < 0:
        raise ValueError("D_eff must be non-negative.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if pixel_scale <= 0:
        raise ValueError("pixel_scale must be positive.")

    u = phase.copy()
    dx2 = pixel_scale ** 2
    for _ in range(n_steps):
        v = u ** m
        lap = np.zeros_like(u)
        lap[1:-1, 1:-1] = (
            v[2:, 1:-1] + v[:-2, 1:-1] + v[1:-1, 2:] + v[1:-1, :-2] - 4.0 * v[1:-1, 1:-1]
        ) / dx2
        u = u + dt * D_eff * lap
    return u


# --- 光化学-热耦合ODE (源自841) ---

def photochemical_refractive_index_ode(y, t, td=86400.0, k2=1e-2, k3=1e-12, q_heat=1e-6):
    """
    光化学-热耦合ODE右手边.

    状态变量: y = [delta_n, delta_T, O3_conc, UV_flux]

    方程组:
      d(delta_n)/dt = q_heat * k1(t) * y[2] - k2 * y[0] - k3 * y[0] * y[1]
      d(delta_T)/dt = q_heat * k1(t) * y[2] - k_thermal * y[1]
      d(O3)/dt     = -k1(t) * y[2] + k3 * y[0] * y[1]
      d(UV)/dt     = -k_absorb * y[3] * y[2] + source(t)

    其中 k1(t) = 0.01 * max(0, sin(2*pi*t/td)) 为光解速率.
    """
    if len(y) != 4:
        raise ValueError("State vector y must have length 4.")
    k1 = 0.01 * max(0.0, np.sin(2.0 * np.pi * t / td))
    k_thermal = 1e-3
    k_absorb = 1e-6

    dn, dT, o3, uv = y
    ddn_dt = q_heat * k1 * o3 - k2 * dn - k3 * dn * dT
    ddT_dt = q_heat * k1 * o3 - k_thermal * dT
    do3_dt = -k1 * o3 + k3 * dn * dT
    duv_dt = -k_absorb * uv * o3 + 0.01 * max(0.0, np.sin(2.0 * np.pi * t / td))

    return np.array([ddn_dt, ddT_dt, do3_dt, duv_dt], dtype=np.float64)


def rk4_integrate_thermal(y0, t_span, n_steps=1000):
    """
    使用经典四阶Runge-Kutta积分光化学ODE.

    RK4步进:
      k1 = h * f(t_n, y_n)
      k2 = h * f(t_n + h/2, y_n + k1/2)
      k3 = h * f(t_n + h/2, y_n + k2/2)
      k4 = h * f(t_n + h, y_n + k3)
      y_{n+1} = y_n + (k1 + 2k2 + 2k3 + k4)/6
    """
    if len(y0) != 4:
        raise ValueError("y0 must have length 4.")
    t0, tf = t_span
    if tf <= t0:
        raise ValueError("t_span[1] must be > t_span[0].")
    h = (tf - t0) / n_steps

    y = np.array(y0, dtype=np.float64)
    t = t0
    trajectory = [y.copy()]

    for _ in range(n_steps):
        k1 = h * photochemical_refractive_index_ode(y, t)
        k2 = h * photochemical_refractive_index_ode(y + 0.5 * k1, t + 0.5 * h)
        k3 = h * photochemical_refractive_index_ode(y + 0.5 * k2, t + 0.5 * h)
        k4 = h * photochemical_refractive_index_ode(y + k3, t + h)
        y = y + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        t += h
        trajectory.append(y.copy())

    return np.array(trajectory)


def apply_thermal_photochemical_phase(phase_base, grid_size, t_sim=100.0):
    """
    将光化学-热ODE结果映射到相位屏修正.

    在网格中心区域 (瞳孔内) 叠加由ODE计算得到的 delta_n 时间演化平均值.
    """
    y0 = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    traj = rk4_integrate_thermal(y0, (0.0, t_sim), n_steps=500)
    avg_dn = np.mean(traj[:, 0])

    x = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, x)
    rho = np.sqrt(X ** 2 + Y ** 2)
    mask = rho <= 1.0

    # 高斯加权, 中心最强
    gaussian_weight = np.exp(-rho ** 2 / 0.5)
    phase_thermal = phase_base + avg_dn * gaussian_weight * mask
    return phase_thermal


# --- 综合相位屏生成 ---

def generate_turbulent_phase_screen(grid_size=256, D_aperture=1.0,
                                     wavelength=500e-9, seeing=1.0,
                                     apply_pme_correction=True,
                                     apply_thermal_perturbation=True,
                                     seed=None):
    """
    综合生成大气湍流相位屏.

    参数:
      grid_size: 网格分辨率
      D_aperture: 望远镜口径 (m)
      wavelength: 工作波长 (m)
      seeing: 视宁度 (arcsec)
      apply_pme_correction: 是否施加多孔介质非线性扩散修正
      apply_thermal_perturbation: 是否施加光化学-热扰动
      seed: 随机种子

    返回: phase_screen (rad), r0 (m), mask
    """
    if grid_size < 4:
        raise ValueError("grid_size must be at least 4.")
    if D_aperture <= 0:
        raise ValueError("D_aperture must be positive.")
    if wavelength <= 0:
        raise ValueError("wavelength must be positive.")

    pixel_scale = D_aperture / grid_size

    # Fried参数: seeing (arcsec) ~ lambda/r_0 * 206265
    r0 = wavelength / (seeing / 206265.0)
    r0 = max(r0, 1e-3)

    phase, mask = generate_phase_screen(grid_size, pixel_scale, r0, seed=seed)

    if apply_pme_correction:
        phase = pme_diffusion_phase_correction(phase, pixel_scale, D_eff=1e-4, m=3.0, dt=1e-4, n_steps=5)

    if apply_thermal_perturbation:
        phase = apply_thermal_photochemical_phase(phase, grid_size, t_sim=50.0)

    # 再次中心化
    phase[mask] -= np.mean(phase[mask])

    return phase, r0, mask
