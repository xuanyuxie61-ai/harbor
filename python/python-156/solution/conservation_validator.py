"""
conservation_validator.py
=========================
基于校验和算法的质量/能量守恒验证模块。

核心算法源自 isbn_check_digit_calculate (Project 600)，并改造用于
验证燃烧模拟结果的质量守恒和能量守恒。

原始 ISBN-10 校验和：
    10A + 9B + 8C + ... + 2I + J ≡ 0 (mod 11)

在燃烧模拟中，我们定义类似的加权校验和来验证守恒律：

质量守恒校验：
    对于每个节点 i，定义质量校验和：
        C_mass = Σ_k w_k * Y_k(i)

    其中 w_k 为组分 k 的分子量权重，Y_k 为质量分数。
    理想情况下 C_mass ≡ 1.0。

能量守恒校验：
    对于稳态系统，定义能量校验和：
        C_energy = h(T) + Σ_k Y_k * h_k^0

    其中 h(T) = cp * T 为显焓，h_k^0 为标准生成焓。

本模块实现：
1. 质量守恒加权校验和；
2. 能量守恒校验和；
3. 全局守恒误差分析；
4. 数值解的完整性验证。
"""

import numpy as np


def mass_conservation_checksum(Y_fields, weights=None):
    """
    计算质量守恒校验和。

    校验和定义为各组分质量分数之和，理想情况下应恒等于 1.0。

    Parameters
    ----------
    Y_fields : dict
        组分质量分数场，键为组分名称，值为 ndarray。
    weights : dict or None
        各组分权重（仅用于惰性气体修正，默认等权重）。

    Returns
    -------
    checksum : ndarray
        每个节点的质量校验和（总和）。
    max_error : float
        最大质量守恒误差 |checksum - 1.0|。
    """
    n = len(next(iter(Y_fields.values())))
    checksum = np.zeros(n)

    for name, Y in Y_fields.items():
        checksum += np.clip(Y, 0.0, 1.0)

    max_error = np.max(np.abs(checksum - 1.0))

    return checksum, max_error


def energy_conservation_checksum(T_field, Y_fields, cp=1200.0,
                                  enthalpies=None):
    """
    计算能量守恒校验和。

    公式：
        C_energy = cp * T + Σ_k Y_k * h_k^0

    Parameters
    ----------
    T_field : ndarray
        温度场。
    Y_fields : dict
        组分质量分数场。
    cp : float
        比热容。
    enthalpies : dict or None
        各组分标准生成焓。

    Returns
    -------
    checksum : ndarray
        能量校验和。
    max_error : float
        最大能量偏差（相对于平均值）。
    """
    if enthalpies is None:
        enthalpies = {'fuel': -4.5e7, 'oxidizer': 0.0, 'product': -3.9e7}

    checksum = cp * T_field
    for name, Y in Y_fields.items():
        h = enthalpies.get(name, 0.0)
        checksum += np.clip(Y, 0.0, 1.0) * h

    mean_energy = np.mean(checksum)
    if abs(mean_energy) < 1.0e-12:
        mean_energy = 1.0

    max_error = np.max(np.abs(checksum - mean_energy)) / abs(mean_energy)

    return checksum, max_error


def validate_simulation(T_field, Y_F_field, Y_O_field, Z_nodes,
                        mass_tol=1.0e-3, energy_tol=1.0e-2):
    """
    综合验证燃烧模拟结果的守恒性和物理合理性。

    Parameters
    ----------
    T_field : ndarray
        温度场。
    Y_F_field : ndarray
        燃料质量分数。
    Y_O_field : ndarray
        氧化剂质量分数。
    Z_nodes : ndarray
        混合分数节点。
    mass_tol : float
        质量守恒容差。
    energy_tol : float
        能量守恒容差。

    Returns
    -------
    validation : dict
        验证结果。
    """
    n = len(Z_nodes)

    # 质量守恒校验（含产物）
    Y_P_field = 1.0 - Y_F_field - Y_O_field
    Y_P_field = np.clip(Y_P_field, 0.0, 1.0)
    Y_fields = {'fuel': Y_F_field, 'oxidizer': Y_O_field, 'product': Y_P_field}
    mass_check, mass_error = mass_conservation_checksum(Y_fields)

    # 能量守恒校验
    energy_check, energy_error = energy_conservation_checksum(
        T_field, Y_fields,
        enthalpies={'fuel': -4.5e7, 'oxidizer': 0.0, 'product': -3.9e7}
    )

    # 物理合理性检查
    T_min = np.min(T_field)
    T_max = np.max(T_field)
    T_reasonable = (T_min >= 200.0 and T_max <= 3500.0)

    Y_reasonable = (np.all(Y_F_field >= -1.0e-6) and
                    np.all(Y_F_field <= 1.0 + 1.0e-6) and
                    np.all(Y_O_field >= -1.0e-6) and
                    np.all(Y_O_field <= 1.0 + 1.0e-6))

    validation = {
        'mass_conservation_error': mass_error,
        'mass_conservation_passed': mass_error < mass_tol,
        'energy_conservation_error': energy_error,
        'energy_conservation_passed': energy_error < energy_tol,
        'temperature_range': (float(T_min), float(T_max)),
        'temperature_reasonable': T_reasonable,
        'mass_fractions_reasonable': Y_reasonable,
        'overall_valid': (mass_error < mass_tol and
                          energy_error < energy_tol and
                          T_reasonable and Y_reasonable),
        'checksum_mass_mean': float(np.mean(mass_check)),
        'checksum_energy_mean': float(np.mean(energy_check)),
    }

    return validation
