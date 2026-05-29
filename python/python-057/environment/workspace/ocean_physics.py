"""
ocean_physics.py
海洋物理核心参数与公式计算

包含：
- 海洋分层密度剖面
- 浮力频率 (Brunt-Väisälä frequency) N(z)
- Richardson 数计算
- 内波色散关系
- 湍流耗散率与混合效率
"""

import numpy as np

# 物理常数
G = 9.81          # 重力加速度 [m/s^2]
RHO0 = 1025.0     # 参考海水密度 [kg/m^3]
NU = 1.0e-6       # 分子粘性系数 [m^2/s]
KAPPA = 1.4e-7    # 分子热扩散系数 [m^2/s]
OMEGA_EARTH = 7.2921159e-5  # 地球自转角速度 [rad/s]
F_CORIOLIS = 1.0e-4  # 近似科里奥利参数 [rad/s] (中纬度)


def density_profile(z, rho0=RHO0, drho_dz=-0.01):
    """
    计算线性分层海洋密度剖面
    
    ρ(z) = ρ₀ + (dρ/dz) · z
    
    参数:
        z: 深度数组 [m]，负值表示向下
        rho0: 表层参考密度 [kg/m³]
        drho_dz: 密度垂向梯度 [kg/m⁴]
    
    返回:
        rho: 密度剖面 [kg/m³]
    """
    z = np.asarray(z)
    rho = rho0 + drho_dz * z
    # 边界处理：确保密度为正且物理合理
    rho = np.clip(rho, 1020.0, 1030.0)
    return rho


def buoyancy_frequency(z, rho0=RHO0, drho_dz=-0.01):
    """
    计算Brunt-Väisälä浮力频率 N(z)
    
    N² = -(g/ρ₀) · (dρ/dz)
    
    N = √[ (g/ρ₀) · |dρ/dz| ]
    
    参数:
        z: 深度数组 [m]
    
    返回:
        N: 浮力频率 [rad/s]
    """
    z = np.asarray(z)
    N2 = -(G / rho0) * drho_dz
    N2 = np.maximum(N2, 1.0e-8)  # 数值稳定性：最小N²
    N = np.sqrt(N2)
    return N


def richardson_number(dudz, dvdz, N):
    """
    计算梯度Richardson数
    
    Ri = N² / [ (∂u/∂z)² + (∂v/∂z)² ]
    
    参数:
        dudz: 水平速度u的垂向剪切 [1/s]
        dvdz: 水平速度v的垂向剪切 [1/s]
        N: 浮力频率 [rad/s]
    
    返回:
        Ri: Richardson数（标量或数组）
    """
    shear_squared = dudz**2 + dvdz**2
    N2 = N**2
    # 避免除零
    Ri = np.where(shear_squared > 1.0e-12,
                  N2 / shear_squared,
                  1.0e6)
    return Ri


def internal_wave_dispersion(kh, m, N, f=F_CORIOLIS):
    """
    线性内波色散关系
    
    ω² = (N² · k_h² + f² · m²) / (k_h² + m²)
    
    其中:
        k_h = √(k_x² + k_y²)  为水平波数
        m 为垂向波数
        N 为浮力频率
        f 为科里奥利参数
    
    参数:
        kh: 水平波数 [rad/m]
        m: 垂向波数 [rad/m]
        N: 浮力频率 [rad/s]
        f: 科里奥利参数 [rad/s]
    
    返回:
        omega: 角频率 [rad/s]
    """
    # HOLE 1: 实现线性内波色散关系
    # 提示: 计算 ω = sqrt( (N²·k_h² + f²·m²) / (k_h² + m²) )
    # 注意数值稳定性处理 (避免除零、开方负数)
    raise NotImplementedError("待实现: 内波色散关系")


def group_velocity(kh, m, N, f=F_CORIOLIS):
    """
    内波群速度
    
    c_gx = ∂ω/∂k_x = (k_x/ω) · (N² - f²) · m² / (k_h² + m²)²
    c_gz = ∂ω/∂m = -(m/ω) · (N² - f²) · k_h² / (k_h² + m²)²
    
    参数:
        kh: 水平波数 [rad/m]
        m: 垂向波数 [rad/m]
        N: 浮力频率 [rad/s]
        f: 科里奥利参数 [rad/s]
    
    返回:
        cgx, cgz: 水平与垂向群速度 [m/s]
    """
    kh = np.asarray(kh)
    m = np.asarray(m)
    omega = internal_wave_dispersion(kh, m, N, f)
    denom = (kh**2 + m**2)**2
    denom = np.where(denom < 1.0e-12, 1.0e-12, denom)
    
    factor = (N**2 - f**2) / denom
    cgx = (kh / omega) * factor * m**2
    cgz = -(m / omega) * factor * kh**2
    return cgx, cgz


def turbulent_dissipation_rate(Ri, shear_squared, nu=NU, mixing_efficiency=0.2):
    """
    湍流耗散率参数化（基于Osborn, 1980）
    
    ε = ν · |∂u/∂z|² · f(Ri)
    
    其中 f(Ri) 为Richardson数依赖的混合函数:
        f(Ri) = max(0, 1 - Ri/Ri_c)   (当 Ri < Ri_c)
        f(Ri) = 0                      (当 Ri ≥ Ri_c)
    
    参数:
        Ri: Richardson数
        shear_squared: 剪切平方 [(1/s)²]
        nu: 涡粘性系数 [m²/s]
        mixing_efficiency: 混合效率 Γ
    
    返回:
        epsilon: 湍流动能耗散率 [W/kg]
        Kz: 垂向涡扩散系数 [m²/s]
    """
    Ri = np.asarray(Ri)
    Ri_critical = 0.25  # Miles-Howard临界Richardson数
    
    # 混合函数
    mix_func = np.maximum(0.0, 1.0 - Ri / Ri_critical)
    epsilon = nu * shear_squared * mix_func
    
    # 垂向涡扩散系数 (Osborn, 1980)
    # K_z = Γ · ε / N²
    N2 = np.where(Ri > 1.0e-6, shear_squared * Ri, 1.0e-8)
    Kz = mixing_efficiency * epsilon / N2
    Kz = np.clip(Kz, 1.0e-7, 1.0e-1)  # 边界处理
    
    return epsilon, Kz


def breaking_criterion(amplitude, wavelength, N, depth):
    """
    内波破碎判据 (基于临界波陡)
    
    当波陡 ε = A · k_h 超过临界值时发生破碎:
        ε_c = O(0.1 ~ 0.3)
    
    参数:
        amplitude: 内波振幅 [m]
        wavelength: 波长 [m]
        N: 浮力频率 [rad/s]
        depth: 水深 [m]
    
    返回:
        is_breaking: 是否破碎 (布尔值)
        steepness: 波陡
        critical_steepness: 临界波陡
    """
    kh = 2.0 * np.pi / wavelength
    steepness = amplitude * kh
    
    # 临界波陡依赖于浮力频率和深度
    # ε_c ≈ 0.2 · (N² · depth / g)^(1/2)
    critical_steepness = 0.2 * np.sqrt(N**2 * depth / G)
    critical_steepness = np.clip(critical_steepness, 0.05, 0.5)
    
    is_breaking = steepness > critical_steepness
    return is_breaking, steepness, critical_steepness


def thope_internal_wave_spectrum(kh, N, f=F_CORIOLIS, E0=6.3e-5):
    """
    Garrett-Munk / Thorpe 内波能量谱参数化
    
    E(k_h) = E₀ · (k_h / k_*)^(-2) · f(ω)
    
    参数:
        kh: 水平波数 [rad/m]
        N: 浮力频率 [rad/s]
        f: 科里奥利参数 [rad/s]
        E0: 能量谱常数 [m²/s²]
    
    返回:
        spectrum: 能量谱密度 [m³/s²]
    """
    kh = np.asarray(kh)
    kh = np.where(kh < 1.0e-8, 1.0e-8, kh)
    
    k_star = 2.0 * np.pi / 1000.0  # 参考波数 [rad/m]
    
    # Garrett-Munk谱形状
    spectrum = E0 * (kh / k_star)**(-2.0)
    
    # 频率依赖的截断
    omega_min = f
    omega_max = N
    omega = internal_wave_dispersion(kh, 2.0 * np.pi / 200.0, N, f)
    
    freq_factor = np.ones_like(omega)
    freq_factor = np.where((omega < omega_min) | (omega > omega_max), 0.0, freq_factor)
    
    spectrum = spectrum * freq_factor
    return spectrum
