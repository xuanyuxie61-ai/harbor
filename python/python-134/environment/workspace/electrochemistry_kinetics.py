#!/usr/bin/env python3
"""
electrochemistry_kinetics.py
电化学反应动力学模块（源自 biochemical_nonlinear_ode 项目）

将生化非线性 ODE 系统的质量作用定律与米氏动力学，迁移为 PEMFC
中的 Butler-Volmer 电化学动力学与多组分传质-反应耦合模型。

核心科学模型：
  - 阳极氢氧化反应 (HOR): H₂ → 2H⁺ + 2e⁻
  - 阴极氧还原反应 (ORR): O₂ + 4H⁺ + 4e⁻ → 2H₂O
  - Butler-Volmer 方程描述电极局部电流密度 j 与过电位 η 的关系
"""

import numpy as np


def compute_exchange_current_density(params):
    """
    计算温度修正后的交换电流密度 j_0。
    Arrhenius 形式：
        j_0 = j_0_ref * exp[ -E_act / (R·T) * (1 - T/T_ref) ]
    这里取简化形式。
    """
    T = params['T']
    j0_ref = params['j_0_ref']
    # 简化温度修正系数
    theta = np.exp(-5000.0 / params['R'] * (1.0 / T - 1.0 / 298.15))
    j0 = j0_ref * max(theta, 1e-6)
    return j0


def butler_volmer_kinetics(eta, params):
    """
    Butler-Volmer 动力学方程：
        j = j_0 [ exp(α_a · n_e · F · η / (R·T))
                - exp(-α_c · n_e · F · η / (R·T)) ]

    对应原项目 biochemical_nonlinear_ode 中的非线性反应速率形式，
    将 Michaelis-Menten 型饱和动力学映射为电化学指数动力学。

    Parameters
    ----------
    eta : ndarray
        过电位 [V]
    params : dict
        物理参数字典

    Returns
    -------
    j : ndarray
        局部电流密度 [A/m²]
    """
    T = params['T']
    F = params['F']
    R = params['R']
    alpha_a = params['alpha_a']
    alpha_c = params['alpha_c']
    j0 = compute_exchange_current_density(params)

    # 避免指数溢出
    arg_max = 500.0
    arg_fwd = np.clip(alpha_a * F * eta / (R * T), -arg_max, arg_max)
    arg_rev = np.clip(-alpha_c * F * eta / (R * T), -arg_max, arg_max)

    j = j0 * (np.exp(arg_fwd) - np.exp(arg_rev))
    return j


def reaction_source_terms(c, state, params):
    """
    计算多组分反应源项，对应原项目中 stoichiometric_matrix * rate_vector 的形式。

    状态向量 state = [c_H2, c_O2, c_Hp, lambda_w, T_loc]
    返回各组分的体积源项 (mol/(m³·s) 或相应单位)。
    """
    c_H2, c_O2, c_Hp, lambda_w, T_loc = state

    # 电化学反应速率（基于局部电流密度的等效体积源）
    j_local = butler_volmer_kinetics(c, params)  # 这里 c 传入的是过电位 eta

    # 实际计算中，将电流密度转换为摩尔通量：r = j / (n_e · F)
    n_e = 2.0  # 电子转移数（简化）
    r = j_local / (n_e * params['F'])

    # 化学计量矩阵 S (5 组分 × 2 反应)
    # 反应1 (HOR): H2 -> 2H+ + 2e-
    # 反应2 (ORR): O2 + 4H+ + 4e- -> 2H2O
    # 注意：这里 lambda_w 代表膜内水含量，与生成水耦合
    S = np.array([
        [-1.0,  0.0],   # H2
        [ 0.0, -1.0],   # O2
        [ 2.0, -4.0],   # H+
        [ 0.0,  2.0],   # H2O (耦合到 lambda)
        [ 0.0,  0.0],   # T (能量方程占位)
    ], dtype=float)

    rate_vec = np.array([r, r * 0.5], dtype=float)  # 简化：ORR 速率约为 HOR 的一半
    source = S @ rate_vec
    return source


def compute_conserved_quantities(state):
    """
    计算守恒量，对应原项目 biochemical_nonlinear_conserved.m。
    利用守恒矩阵 E 提取不变量。
    """
    c_H2, c_O2, c_Hp, lambda_w, T_loc = state
    # 构造守恒矩阵 E（行代表守恒关系）
    # 例如：总氢原子、总氧原子、电荷等
    E = np.array([
        [2.0, 0.0, 1.0, 2.0, 0.0],   # H 原子守恒
        [0.0, 2.0, 0.0, 1.0, 0.0],   # O 原子守恒
        [0.0, 0.0, 1.0, 0.0, 0.0],   # 电荷
    ], dtype=float)
    h = E @ np.array(state, dtype=float)
    return h


def compute_activation_overpotential(j_target, params, side='cathode'):
    """
    由目标电流密度反解活化过电位（Tafel 近似或完整 BV 反演）。
    用于极化曲线计算。
    """
    T = params['T']
    F = params['F']
    R = params['R']
    j0 = compute_exchange_current_density(params)

    if side == 'cathode':
        alpha = params['alpha_c']
        # Tafel 近似: η = (R·T / (α·F)) · ln(j / j0)
        eta = (R * T) / (alpha * F) * np.log(np.maximum(j_target, 1e-10) / j0)
    else:
        alpha = params['alpha_a']
        eta = (R * T) / (alpha * F) * np.arcsinh(j_target / (2.0 * j0))
    return eta


if __name__ == '__main__':
    p = {
        'T': 353.15, 'P': 1.5, 'R': 8.314, 'F': 96485.0,
        'alpha_a': 0.5, 'alpha_c': 0.5, 'j_0_ref': 1e-3
    }
    eta = np.linspace(-0.3, 0.3, 100)
    j = butler_volmer_kinetics(eta, p)
    print("Butler-Volmer max j:", np.max(np.abs(j)))
