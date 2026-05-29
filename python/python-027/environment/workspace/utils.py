# -*- coding: utf-8 -*-
"""
utils.py
通用工具函数模块
基于种子项目 1419_xy_display (2D数据读取/处理) 重构

提供数据IO、数值稳定性处理、科学常数等辅助功能。
"""

import numpy as np


# 物理常数
PHYSICAL_CONSTANTS = {
    'epsilon_0': 8.854187817e-12,     # 真空介电常数 [F/m]
    'mu_0': 4.0e-7 * np.pi,           # 真空磁导率 [H/m]
    'e_charge': 1.602176634e-19,      # 元电荷 [C]
    'm_e': 9.1093837015e-31,          # 电子质量 [kg]
    'm_p': 1.67262192369e-27,         # 质子质量 [kg]
    'k_B': 1.380649e-23,              # Boltzmann常数 [J/K]
    'c': 299792458.0,                 # 光速 [m/s]
    'h_planck': 6.62607015e-34,       # Planck常数 [J*s]
}


def safe_exp(x, max_val=50.0, min_val=-50.0):
    """
    安全指数函数，防止溢出
    
    Parameters:
        x:      输入值
        max_val: 最大指数
        min_val: 最小指数
    
    Returns:
        exp(x) 或边界值
    """
    x_clipped = np.clip(x, min_val, max_val)
    return np.exp(x_clipped)


def safe_divide(a, b, default=0.0):
    """
    安全除法，避免除零
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        result = np.divide(a, b, out=np.full_like(np.asarray(a), default, dtype=float), where=np.abs(b) > 1.0e-30)
        return result
    else:
        if abs(b) < 1.0e-30:
            return default
        return a / b


def compute_coulomb_logarithm(T_e, n_e):
    """
    计算 Coulomb 对数
    
    公式:
        ln(Lambda) = 23.4 - 1.15*log10(n_e) + 3.45*log10(T_e)   (T_e > 10 eV)
        ln(Lambda) = 23.0 - 1.15*log10(n_e) + 3.45*log10(T_e)   (T_e < 10 eV)
    
    Parameters:
        T_e: 电子温度 [eV]
        n_e: 电子密度 [m^-3]
    
    Returns:
        ln_Lambda: Coulomb对数
    """
    if T_e <= 0 or n_e <= 0:
        return 15.0

    if T_e > 10.0:
        ln_L = 23.4 - 1.15 * np.log10(n_e) + 3.45 * np.log10(T_e)
    else:
        ln_L = 23.0 - 1.15 * np.log10(n_e) + 3.45 * np.log10(T_e)

    # 物理约束
    if ln_L < 5.0:
        ln_L = 5.0
    if ln_L > 25.0:
        ln_L = 25.0

    return ln_L


def compute_ion_gyroradius(T_i, B, m_i_amu, Z_i=1):
    """
    计算离子拉莫尔半径
    
    公式:
        rho_i = m_i * v_perp / (Z_i * e * B)
              = sqrt(2*m_i*k_B*T_i) / (Z_i*e*B)
    
    Parameters:
        T_i:     离子温度 [eV]
        B:       磁场强度 [T]
        m_i_amu: 离子质量数
        Z_i:     离子电荷数
    
    Returns:
        rho_i: 离子拉莫尔半径 [m]
    """
    e_c = PHYSICAL_CONSTANTS['e_charge']
    m_p = PHYSICAL_CONSTANTS['m_p']
    mi_kg = m_i_amu * m_p

    if B <= 0 or T_i <= 0 or Z_i <= 0:
        return np.nan

    rho_i = np.sqrt(2.0 * mi_kg * T_i * e_c) / (Z_i * e_c * B)
    return rho_i


def compute_magnetic_mirror_force(mu, B_grad):
    """
    计算磁镜力
    
    公式:
        F_mirror = -mu * grad_parallel(B)
    
    Parameters:
        mu:     磁矩 [J/T]
        B_grad: 磁场梯度 [T/m]
    
    Returns:
        F: 磁镜力 [N]
    """
    return -mu * B_grad


def write_data_file(filename, data, header=None):
    """
    将数据写入文本文件（基于 xy_display 的数据输出思想）
    
    Parameters:
        filename: 输出文件名
        data:     (N, M) 数据数组
        header:   可选的注释头
    """
    with open(filename, 'w') as f:
        if header is not None:
            f.write(f"# {header}\n")
        np.savetxt(f, data, fmt='%.6e')


def read_data_file(filename, skip_comments=True):
    """
    从文本文件读取数据（基于 xy_data_read 思想）
    
    Parameters:
        filename: 输入文件名
        skip_comments: 是否跳过注释行
    
    Returns:
        data: numpy数组
    """
    if skip_comments:
        return np.loadtxt(filename, comments='#')
    else:
        return np.loadtxt(filename)


def print_matrix_summary(mat, name="Matrix", max_display=5):
    """
    打印矩阵摘要信息
    """
    print(f"{name}: shape={mat.shape}, min={np.min(mat):.3e}, max={np.max(mat):.3e}, mean={np.mean(mat):.3e}")
    if mat.ndim == 2 and mat.shape[0] <= max_display and mat.shape[1] <= max_display:
        print(mat)


def check_bohm_criterion(v_i, c_s, tolerance=0.01):
    """
    验证Bohm判据
    
    Bohm判据: M = v_i / c_s >= 1
    
    Parameters:
        v_i: 离子速度 [m/s]
        c_s: 离子声速 [m/s]
        tolerance: 容差
    
    Returns:
        satisfied: bool
        mach_number: Mach数
    """
    if c_s <= 0:
        return False, 0.0
    M = v_i / c_s
    return M >= (1.0 - tolerance), M


def compute_sheath_heat_flux(n_se, T_e, T_i, gamma=7.0, Z_i=1):
    """
    计算鞘层热流密度
    
    公式:
        q_sheath = gamma * n_se * c_s * k_B * T_e
    
    其中 gamma 为鞘层传热系数:
        gamma = 2.5 * T_i/T_e + (1/2)*ln(2*pi*m_e/m_i*(1+T_i/T_e)) + 2
    
    Parameters:
        n_se:  鞘层边缘密度 [m^-3]
        T_e:   电子温度 [eV]
        T_i:   离子温度 [eV]
        gamma: 传热系数
        Z_i:   离子电荷数
    
    Returns:
        q: 热流密度 [W/m^2]
    """
    # TODO: 实现鞘层热流密度计算
    # 提示: 需先计算离子声速 c_s，再根据 gamma 计算热流密度 q
    #       注意与 parameters.py 中 ion_sound_speed 的物理一致性
    raise NotImplementedError("Hole_3: 请实现 compute_sheath_heat_flux 函数")



def convergence_diagnostics(residual_history, window=10):
    """
    分析收敛历史，判断数值稳定性
    
    Returns:
        converged: bool
        rate: 平均收敛速率
        oscillation: 振荡幅度指标
    """
    if len(residual_history) < window:
        return False, 0.0, 0.0

    recent = residual_history[-window:]
    rate = np.mean(np.diff(np.log(recent + 1.0e-30)))

    # 检测振荡
    diffs = np.diff(recent)
    sign_changes = np.sum(diffs[:-1] * diffs[1:] < 0)
    oscillation = sign_changes / max(len(diffs) - 1, 1)

    converged = recent[-1] < 1.0e-6 and oscillation < 0.5

    return converged, rate, oscillation


if __name__ == "__main__":
    print("utils.py 测试:")
    print(f"  Coulomb对数 = {compute_coulomb_logarithm(50.0, 1.0e19):.2f}")
    print(f"  离子拉莫尔半径 = {compute_ion_gyroradius(50.0, 5.3, 2.0):.3e} m")
    print(f"  鞘层热流 = {compute_sheath_heat_flux(5.0e18, 50.0, 50.0):.3e} W/m^2")
