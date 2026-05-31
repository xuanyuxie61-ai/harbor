"""
hyperelastic_law.py
超弹性本构模型模块

融合种子项目:
- 472_glycolysis_ode: 化学驱动ODE（化学-力学耦合）

科学应用: 软体机器人材料的超弹性本构关系
包含Neo-Hookean、Mooney-Rivlin模型及化学驱动耦合
"""

import numpy as np
from typing import Tuple


def neo_hookean_strain_energy(F: np.ndarray, mu: float, K_bulk: float) -> float:
    """
    Neo-Hookean超弹性应变能密度函数

    变形梯度 F (3x3)
    右Cauchy-Green张量: C = F^T * F
    第一不变量: I1 = tr(C)
    体积比: J = det(F)

    应变能密度:
        W = (mu/2)*(I1 - 3) - mu*ln(J) + (K_bulk/2)*(ln(J))^2

    参数:
        F: 变形梯度 (3x3)
        mu: 剪切模量 (Pa)
        K_bulk: 体积模量 (Pa)
    """
    if F.shape != (3, 3):
        raise ValueError("F must be 3x3")

    C = F.T @ F
    I1 = np.trace(C)
    J = np.linalg.det(F)
    if abs(J) < 1e-14:
        J = 1e-14

    lnJ = np.log(abs(J))
    W = 0.5 * mu * (I1 - 3.0) - mu * lnJ + 0.5 * K_bulk * lnJ ** 2
    return W


def neo_hookean_stress(F: np.ndarray, mu: float, K_bulk: float) -> np.ndarray:
    """
    Neo-Hookean第一Piola-Kirchhoff应力张量 P

    P = dW/dF = mu*(F - F^{-T}) + K_bulk*ln(J)*F^{-T}

    或Cauchy应力:
        sigma = (1/J) * P * F^T
              = (mu/J)*(F*F^T - I) + (K_bulk/J)*ln(J)*I
    """
    if F.shape != (3, 3):
        raise ValueError("F must be 3x3")

    J = np.linalg.det(F)
    if abs(J) < 1e-14:
        J = np.sign(J + 1e-14) * 1e-14

    F_inv_T = np.linalg.inv(F).T
    P = mu * (F - F_inv_T) + K_bulk * np.log(abs(J)) * F_inv_T
    return P


def mooney_rivlin_strain_energy(F: np.ndarray, C10: float, C01: float, K_bulk: float) -> float:
    """
    Mooney-Rivlin超弹性应变能密度函数

    应变不变量:
        I1 = tr(C)
        I2 = 0.5*(tr(C)^2 - tr(C^2))
        J = det(F)

    应变能:
        W = C10*(I1 - 3) + C01*(I2 - 3) + (K_bulk/2)*(ln(J))^2

    参数:
        C10, C01: Mooney-Rivlin材料常数 (Pa)
        K_bulk: 体积模量 (Pa)
    """
    if F.shape != (3, 3):
        raise ValueError("F must be 3x3")

    C = F.T @ F
    I1 = np.trace(C)
    I2 = 0.5 * (I1 ** 2 - np.trace(C @ C))
    J = np.linalg.det(F)
    if abs(J) < 1e-14:
        J = 1e-14

    lnJ = np.log(abs(J))
    W = C10 * (I1 - 3.0) + C01 * (I2 - 3.0) + 0.5 * K_bulk * lnJ ** 2
    return W


def mooney_rivlin_stress(F: np.ndarray, C10: float, C01: float, K_bulk: float) -> np.ndarray:
    """
    Mooney-Rivlin Cauchy应力张量

    sigma = (2/J) * [C10 + C01*I1]*B - (2*C01/J)*B^2 + [K_bulk*ln(J)/J]*I
    其中 B = F*F^T (左Cauchy-Green张量)
    """
    if F.shape != (3, 3):
        raise ValueError("F must be 3x3")

    J = np.linalg.det(F)
    if abs(J) < 1e-14:
        J = np.sign(J + 1e-14) * 1e-14

    B = F @ F.T
    I1 = np.trace(B)

    sigma = (2.0 / J) * (C10 + C01 * I1) * B - (2.0 * C01 / J) * (B @ B)
    sigma += (K_bulk * np.log(abs(J)) / J) * np.eye(3)
    return sigma


def soft_robot_1d_constitutive(epsilon: float, kappa: np.ndarray,
                               E: float, G: float, A: float,
                               Ixx: float, Iyy: float, J: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    1D软体机器人Cosserat杆线性化本构关系

    线应变 epsilon = v_3 - 1 (轴向)
    曲率 kappa = [kappa_1, kappa_2, kappa_3]^T (弯曲1, 弯曲2, 扭转)

    内力:
        n = [G*A_shear*(v1), G*A_shear*(v2), E*A*(v3-1)]^T
    内矩:
        m = [E*Ixx*kappa1, E*Iyy*kappa2, G*J*kappa3]^T

    简化返回轴向力和弯矩
    """
    # 轴向力
    N_axial = E * A * epsilon
    # 剪力（简化）
    V1 = G * A * kappa[0] * 0.0  # 小变形假设下剪力与曲率解耦
    V2 = G * A * kappa[1] * 0.0

    # 弯矩
    M1 = E * Ixx * kappa[0]
    M2 = E * Iyy * kappa[1]
    M3 = G * J * kappa[2]

    n = np.array([V1, V2, N_axial])
    m = np.array([M1, M2, M3])
    return n, m


def chemo_mechanical_coupling(y_chem: np.ndarray, epsilon: float,
                              E0: float, gamma: float, beta_chem: float) -> float:
    """
    化学-力学耦合模型 — 基于种子项目472_glycolysis_ode的化学ODE思想

    软体材料（如离子聚合物金属复合材料IPMC）的化学驱动:
    化学状态 y_chem = [c_ion, pH]^T 影响弹性模量:
        E(eff) = E0 * (1 + gamma * c_ion) * exp(-beta_chem * |pH - 7|)

    参数:
        y_chem: [离子浓度, pH值]
        epsilon: 机械应变
        E0: 基准弹性模量
        gamma: 离子浓度耦合系数
        beta_chem: pH敏感系数
    """
    c_ion = y_chem[0]
    pH = y_chem[1]

    # pH偏离中性时模量下降
    pH_factor = np.exp(-beta_chem * abs(pH - 7.0))
    # 离子浓度升高时模量上升（溶胀效应）
    ion_factor = 1.0 + gamma * c_ion

    E_eff = E0 * ion_factor * pH_factor
    # 保证正值和上下界
    E_eff = max(E0 * 0.1, min(E0 * 5.0, E_eff))
    return E_eff


def selkov_glycolysis_ode(t: float, y: np.ndarray, a: float = 0.08, b: float = 0.6) -> np.ndarray:
    """
    Selkov糖酵解ODE模型 — 直接来自种子项目472_glycolysis_ode
    用作软体机器人化学驱动的代谢动力学模型

    状态:
        y[0] = u: ATP浓度（或化学活化剂）
        y[1] = v: ADP浓度（或化学抑制剂）

    方程:
        du/dt = -u + a*v + u^2*v
        dv/dt =  b - a*v - u^2*v

    参数a,b控制振荡行为（Hopf分岔）
    """
    u, v = y
    dudt = -u + a * v + u * u * v
    dvdt = b - a * v - u * u * v
    return np.array([dudt, dvdt])


def tangent_stiffness_neo_hookean(F: np.ndarray, mu: float, K_bulk: float) -> np.ndarray:
    """
    Neo-Hookean材料的切线刚度张量（四阶张量的矩阵形式）

    材料切线模量:
        C_{ijkl} = mu * delta_{ik} * delta_{jl}
                 + (mu - K_bulk*ln(J)) * F^{-1}_{ji} * F^{-1}_{lk}
                 + K_bulk * F^{-1}_{ji} * F^{-1}_{lk}

    返回 6x6 Voigt形式矩阵
    """
    if F.shape != (3, 3):
        raise ValueError("F must be 3x3")

    J = np.linalg.det(F)
    if abs(J) < 1e-14:
        J = 1e-14

    F_inv = np.linalg.inv(F)
    lnJ = np.log(abs(J))

    # 简化Voigt形式 (6x6)
    C = np.zeros((6, 6))
    # 对角项
    for i in range(3):
        C[i, i] = mu + K_bulk
    for i in range(3, 6):
        C[i, i] = 0.5 * mu

    # 体积耦合项
    vol_term = K_bulk * lnJ - mu
    for i in range(3):
        for j in range(3):
            C[i, j] += vol_term

    return C
