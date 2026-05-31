#!/usr/bin/env python3
"""
synthetic_experiments.py
合成实验数据生成模块（源自 human_data 项目）

将原项目中基于人机交互的图像轮廓采集，迁移为燃料电池极化曲线的
参数化合成数据生成。生成包含活化损失、欧姆损失与浓差损失的
完整 I-V 特性曲线，用于验证模型的实验一致性。

物理模型（极化曲线）：
    V_cell = E_rev - η_act - η_ohm - η_conc

其中：
    E_rev = E_0 - (R·T / 2F) · ln(P_H2·P_O2^{1/2} / P_H2O)
    η_act = (R·T / α·F) · arcsinh(j / 2j_0)
    η_ohm = j · t_m / σ_m(λ)
    η_conc = (R·T / n·F) · ln(1 - j / j_L)
"""

import numpy as np


def generate_polarization_curve(params, n_points=100):
    """
    合成 PEMFC 极化曲线数据点（电流密度 vs. 电池电压）。
    对应原项目 human_data.m 中数据点采集的概念，但改为参数化科学生成。

    Parameters
    ----------
    params : dict
        物理参数
    n_points : int
        数据点数量

    Returns
    -------
    V_cell : ndarray
        电池电压 [V]
    I_cell : ndarray
        电流密度 [A/cm²]
    """
    T = params['T']
    R = params['R']
    F = params['F']
    alpha = params['alpha_a']
    j0 = params['j_0_ref']
    t_m = params['t_membrane']
    sigma_m = params['sigma_m_ref']
    lambda_eq = params['lambda_eq']

    # 可逆电位（Nernst 方程）
    P_H2 = 1.0   # atm
    P_O2 = 0.21  # atm
    P_H2O = 0.5  # atm（饱和蒸汽压近似）
    E_rev = params['E_0'] - (R * T / (2.0 * F)) * np.log(P_H2 * np.sqrt(P_O2) / P_H2O)

    # 电流密度范围（A/cm²）
    j_min = 1e-4
    j_max = 2.0  # A/cm²
    j = np.logspace(np.log10(j_min), np.log10(j_max), n_points)

    # 活化过电位（Tafel 形式 + 低电流修正）
    eta_act = (R * T) / (alpha * F) * np.arcsinh(j / (2.0 * j0))

    # 欧姆过电位（膜电阻）
    # 膜电导率随水含量变化
    sigma_lambda = sigma_m * (0.005139 * lambda_eq - 0.00326)
    sigma_lambda = max(sigma_lambda, 1e-3)
    eta_ohm = j * (t_m / sigma_lambda) * 1e-4  # 转换为 V（注意单位换算）

    # 浓差过电位（极限电流假设 j_L = 3 A/cm²）
    j_L = 3.0  # A/cm²
    j_ratio = np.clip(j / j_L, 0.0, 0.99)
    eta_conc = -(R * T / (4.0 * F)) * np.log(1.0 - j_ratio)

    # 总电压
    V_cell = E_rev - eta_act - eta_ohm - eta_conc
    V_cell = np.clip(V_cell, 0.0, E_rev)

    return V_cell, j


def generate_impedance_spectrum(params, n_freq=80):
    """
    合成电化学阻抗谱（EIS）数据。
    简化 Randles 电路模型：
        Z(ω) = R_ohm + R_ct / (1 + j·ω·R_ct·C_dl)
    """
    T = params['T']
    R_ohm = 0.05  # Ohm·cm²
    R_ct = 0.2    # Ohm·cm²
    C_dl = 0.02   # F/cm²

    freq = np.logspace(-3, 5, n_freq)  # Hz
    omega = 2.0 * np.pi * freq

    Z_real = R_ohm + R_ct / (1.0 + (omega * R_ct * C_dl) ** 2)
    Z_imag = -omega * R_ct ** 2 * C_dl / (1.0 + (omega * R_ct * C_dl) ** 2)

    return freq, Z_real, Z_imag


def generate_humidity_scan_data(params, n_rh=20):
    """
    生成不同入口相对湿度下的稳态性能扫描数据。
    模拟实验操作条件矩阵。
    """
    RH_range = np.linspace(0.3, 1.0, n_rh)
    T_range = np.linspace(313.15, 363.15, 5)

    data_matrix = np.zeros((n_rh, len(T_range)))
    for i, rh in enumerate(RH_range):
        for j, T in enumerate(T_range):
            # 水含量与相对湿度关系（Springer）
            lambda_w = 0.043 + 17.81 * rh - 39.85 * rh ** 2 + 36.0 * rh ** 3
            lambda_w = np.clip(lambda_w, 0.0, 22.0)

            # 估算最大功率密度（简化）
            sigma = 0.005139 * lambda_w - 0.00326
            sigma = max(sigma, 1e-3)
            P_max = 0.5 * sigma * (1.0 - T / 400.0)  # 简化模型
            data_matrix[i, j] = max(P_max, 0.0)

    return RH_range, T_range, data_matrix


if __name__ == '__main__':
    p = {
        'T': 353.15, 'R': 8.314, 'F': 96485.0, 'alpha_a': 0.5,
        'j_0_ref': 1e-3, 't_membrane': 50e-6, 'sigma_m_ref': 10.0,
        'lambda_eq': 14.0, 'E_0': 1.229
    }
    V, I = generate_polarization_curve(p)
    print("V range:", V.min(), V.max())
