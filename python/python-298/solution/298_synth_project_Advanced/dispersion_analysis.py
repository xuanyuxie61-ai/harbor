"""
数值色散分析模块

分析 PIC 模拟中的数值色散关系:
- 有限差分离散化导致的数值修正
- 时间离散化 (leapfrog) 导致的数值色散
- 粒子离散化导致的数值噪声

核心色散关系:
    解析 (Bohm-Gross):
        omega^2 = omega_pe^2 + 3*k^2*v_th^2

    数值 (PIC):
        sin^2(omega*dt/2)/(dt/2)^2 = omega_pe^2 * [sin(k*dx/2)/(k*dx/2)]^2
                                      * S(k, omega)
    其中 S 为形状因子校正
"""

import numpy as np
from typing import Tuple, List, Dict
try:
    from plasma_constants import (PI, ELECTRON_CHARGE, ELECTRON_MASS,
                                 VACUUM_PERMITTIVITY, plasma_frequency_electron,
                                 debye_length, thermal_velocity)
except ImportError:
    from plasma_constants import (PI, ELECTRON_CHARGE, ELECTRON_MASS,
                                 VACUUM_PERMITTIVITY, plasma_frequency_electron,
                                 debye_length, thermal_velocity)
try:
    from high_order_fd import modified_wavenumber
except ImportError:
    from high_order_fd import modified_wavenumber


def numerical_dispersion_relation(k: float, omega_pe: float, v_th: float,
                                     dx: float, dt: float,
                                     fd_order: int = 2,
                                     shape_order: int = 1) -> complex:
    """
    PIC 数值色散关系求解

    对于 1D ES PIC (leapfrog + 高阶差分):
        sin^2(omega*dt/2) / (dt/2)^2 = omega_pe^2 * D_fd(k*dx) * S_shape(k*dx)

    其中:
        D_fd(k*dx) = (k_num/k)^2  有限差分校正
        S_shape    = [sin(k*dx/2)/(k*dx/2)]^(2*shape_order)  形状因子

    Parameters:
        k           : 波数
        omega_pe    : 等离子体频率
        v_th        : 热速度
        dx          : 网格间距
        dt          : 时间步
        fd_order    : 有限差分阶数
        shape_order : 形状因子阶数 (0=NGP, 1=CIC, 2=TSC)

    Returns:
        omega (复数): 正频率分支
    """
    # 修正波数 (有限差分)
    k_arr = np.array([k])
    km_sq = modified_wavenumber(k_arr, dx, fd_order, derivative=2)
    km_sq_val = float(np.real(km_sq[0]))

    # 形状因子
    kdx = k * dx
    if abs(kdx) < 1e-10:
        S_shape = 1.0
    else:
        S_shape = (np.sin(kdx / 2.0) / (kdx / 2.0))**(2 * shape_order)

    # 热修正 (Bohm-Gross)
    omega_bg_sq = omega_pe**2 + 3.0 * km_sq_val * (v_th / np.sqrt(2.0))**2

    # 数值色散: sin(omega*dt/2) = omega_eff * dt/2
    omega_eff_sq = omega_bg_sq * S_shape
    omega_dt_half = np.arcsin(min(np.sqrt(omega_eff_sq) * dt / 2.0, 1.0 - 1e-10))
    omega = 2.0 * omega_dt_half / dt

    return complex(omega, 0.0)


def numerical_dispersion_sweep(k_array: np.ndarray, omega_pe: float,
                                  v_th: float, dx: float, dt: float,
                                  fd_order: int = 2,
                                  shape_order: int = 1) -> np.ndarray:
    """
    对一组 k 值计算数值色散 omega(k)
    """
    omega_array = np.zeros(len(k_array), dtype=complex)
    for i, k in enumerate(k_array):
        omega_array[i] = numerical_dispersion_relation(k, omega_pe, v_th,
                                                          dx, dt, fd_order, shape_order)
    return omega_array


def phase_velocity_error(k_array: np.ndarray, omega_array: np.ndarray,
                            omega_pe: float, v_th: float) -> np.ndarray:
    """
    数值相速度误差:
        err = |v_ph_num - v_ph_exact| / |v_ph_exact|

    v_ph = omega / k
    """
    v_ph_num = np.real(omega_array) / (k_array + 1e-300)

    # 解析相速度 (Bohm-Gross)
    lambda_D = v_th / (np.sqrt(2.0) * omega_pe)
    omega_exact = omega_pe * np.sqrt(1.0 + 3.0 * (k_array * lambda_D)**2)
    v_ph_exact = omega_exact / (k_array + 1e-300)

    return np.abs(v_ph_num - v_ph_exact) / (np.abs(v_ph_exact) + 1e-300)


def group_velocity(k_array: np.ndarray, omega_array: np.ndarray,
                     dk: float = None) -> np.ndarray:
    """
    数值群速度:
        v_g = d(omega)/dk

    通过有限差分近似
    """
    if dk is None:
        dk = k_array[1] - k_array[0] if len(k_array) > 1 else 1.0

    v_g = np.zeros_like(k_array)
    omega_r = np.real(omega_array)
    for i in range(1, len(k_array) - 1):
        v_g[i] = (omega_r[i + 1] - omega_r[i - 1]) / (2.0 * dk)

    # 边界单侧
    if len(k_array) > 1:
        v_g[0] = (omega_r[1] - omega_r[0]) / dk
        v_g[-1] = (omega_r[-1] - omega_r[-2]) / dk

    return v_g


def landau_damping_comparison(k_array: np.ndarray, n_e: float,
                                  T_e: float, dx: float, dt: float,
                                  fd_order: int = 2) -> Dict[str, np.ndarray]:
    """
    比较数值色散与 Landau 阻尼解析结果

    Returns:
        dict with keys: 'k', 'omega_num_r', 'omega_num_i',
                        'omega_exact_r', 'gamma_landau'
    """
    omega_pe = plasma_frequency_electron(n_e)
    v_th = thermal_velocity(T_e, ELECTRON_MASS)

    # 数值
    omega_num = numerical_dispersion_sweep(k_array, omega_pe, v_th, dx, dt, fd_order)

    # Landau 解析
    from root_finder import landau_damping_rate
    omega_exact = np.array([landau_damping_rate(k, n_e, T_e) for k in k_array])

    return {
        'k': k_array,
        'omega_num_r': np.real(omega_num),
        'omega_num_i': np.imag(omega_num),
        'omega_exact_r': np.real(omega_exact),
        'gamma_landau': np.imag(omega_exact)
    }


def nyquist_constraint(dx: float) -> float:
    """
    Nyquist 波数约束:
        k_max = pi / dx
    """
    return PI / dx


def cfl_constraint(v_max: float, dx: float) -> float:
    """
    CFL 约束:
        dt <= dx / v_max
    """
    return dx / (v_max + 1e-300)


def finite_grid_instability_threshold(L: float, lambda_D: float) -> float:
    """
    有限网格不稳定性阈值:
        需要 L/lambda_D > ~10 以避免非物理模式
    """
    return L / (lambda_D + 1e-300)


def numerical_heating_rate(Np: int, N_grid: int, omega_pe: float,
                             dt: float) -> float:
    """
    数值加热率估计 (Dawson 1970):
        dW/dt ~ omega_pe * W / (N_p * (omega_pe * dt)^2)
               * (1/N_grid) * correction

    简化估计
    """
    if Np <= 0 or N_grid <= 0 or dt <= 0:
        return 0.0
    return omega_pe / (Np * (omega_pe * dt)**2 * N_grid)
