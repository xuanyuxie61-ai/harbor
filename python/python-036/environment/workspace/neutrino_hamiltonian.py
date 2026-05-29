"""
neutrino_hamiltonian.py
中微子哈密顿量本征值分析与质量 Hierarchy 判定

本模块实现:
1. 真空中微子哈密顿量 H_vac = (1/2E) U M² U† 的构造
2. 物质中有效哈密顿量 H_mat = H_vac + V_mat
3. 广义本征值问题求解 (源自 chladni_figures 的稀疏本征值算法)
4. 质量 Hierarchy 判据计算
5. Gamma 函数近似 (源自 toms291/alogam, 用于 Fermi-Dirac 分布)

关键物理:
    - MSW (Mikheyev-Smirnov-Wolfenstein) 共振条件
    - 质量本征态在物质中的修正
    - 有效混合角随物质密度的演化
"""

import numpy as np
from constants import (
    GF, KM_TO_EV_INV,
    DELTA_M2_21, DELTA_M2_31, DELTA_M2_31_IH,
    THETA_12, THETA_23, THETA_13
)
from pmns_matrix import build_pmns_matrix, build_mass_matrix


def log_gamma_pike_hill(x):
    """
    计算 ln Γ(x), 基于 Pike & Hill, CACM 1966, Algorithm 291。

    公式 (Stirling 近似改进):
        ln Γ(x) ≈ (x - 0.5) ln(x) - x + 0.918938533204673
                  + Σ_k c_k / x^{2k+1}

    参数:
        x: 正实数

    返回:
        (value, ifault): ln Γ(x) 和错误码 (0=成功, 1=输入非法)
    """
    if x <= 0.0:
        return 0.0, 1

    y = float(x)
    if x < 7.0:
        f = 1.0
        z = y
        while z < 7.0:
            f *= z
            z += 1.0
        y = z
        f = -np.log(f)
    else:
        f = 0.0

    z = 1.0 / y / y
    value = f + (y - 0.5) * np.log(y) - y + 0.918938533204673
    # 渐近展开修正项
    value += (((
        -0.000595238095238 * z
        + 0.000793650793651) * z
        - 0.002777777777778) * z
        + 0.083333333333333) / y

    return value, 0


def fermi_dirac_distribution(energy, temperature, chemical_potential=0.0, eta=-1.0):
    """
    计算 Fermi-Dirac (η=-1) 或 Bose-Einstein (η=+1) 分布:
        f(E) = 1 / [exp((E - μ)/T) + η]

    对于中微子产生 (例如太阳核心), 使用 η = -1 (费米子)。
    参数:
        energy:             粒子能量 [任意单位]
        temperature:        温度 [与 energy 相同单位]
        chemical_potential: 化学势 [与 energy 相同单位]
        eta:                -1 为费米子, +1 为玻色子

    返回:
        f: 分布函数值
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    arg = (energy - chemical_potential) / temperature
    # 数值稳定性: 对大正 arg, exp -> inf -> f -> 0
    # 对大负 arg, exp -> 0 -> f -> 1/(1+η)
    if eta == -1:
        if arg > 700:
            return 0.0
        return 1.0 / (np.exp(arg) + 1.0)
    elif eta == 1:
        if arg > 700:
            return 0.0
        if arg < -700:
            return 0.0
        return 1.0 / (np.exp(arg) - 1.0)
    else:
        return 1.0 / (np.exp(arg) + eta)


def build_vacuum_hamiltonian(energy_gev, theta12=None, theta23=None,
                             theta13=None, delta_cp=None,
                             delta_m2_21=None, delta_m2_31=None,
                             hierarchy='normal'):
    """
    构造真空中的有效哈密顿量 H_vac [eV]。

    在超相对论极限下 (E >> m_i), 中微子传播由有效薛定谔方程描述:
        i d|ν_α⟩/dt = H_vac |ν_α⟩

    其中:
        H_vac = (1 / 2E) · U · diag(0, Δm²₂₁, Δm²₃₁) · U†

    参数:
        energy_gev:   中微子能量 [GeV]
        theta12, theta23, theta13: 混合角 [rad]
        delta_cp:     CP 相位 [rad]
        delta_m2_21, delta_m2_31: 质量平方差 [eV²]
        hierarchy:    'normal' 或 'inverted'

    返回:
        H: 3×3 厄米矩阵 [eV]
    """
    if energy_gev <= 0:
        raise ValueError("energy_gev must be positive")

    # === HOLE 2 ===
    # 请构造真空中的有效哈密顿量 H_vac [eV]
    # 提示:
    #   1. 调用 build_pmns_matrix 和 build_mass_matrix 获取 U 和 M2
    #   2. H_vac = (1 / 2E) · U · M2 · U†
    #   3. 注意能量单位转换: E [GeV] -> E [eV] = E_gev * 1e9
    #   4. U† 在 numpy 中表示为 U.conj().T
    # === END HOLE 2 ===
    raise NotImplementedError("HOLE 2: build_vacuum_hamiltonian 核心计算尚未实现")


def build_matter_hamiltonian(energy_gev, matter_potential_ev,
                             theta12=None, theta23=None,
                             theta13=None, delta_cp=None,
                             delta_m2_21=None, delta_m2_31=None,
                             hierarchy='normal'):
    """
    构造物质中的有效哈密顿量 H_mat [eV]。

    在物质中, 电子中微子通过与电子的带电流 (CC) forward scattering
    获得额外相位:
        H_mat = H_vac + V_CC · diag(1, 0, 0)

    其中 V_CC = sqrt(2) G_F N_e 为物质势。

    参数:
        energy_gev:          中微子能量 [GeV]
        matter_potential_ev: 物质势 V_CC [eV]
        ... (其他参数同 build_vacuum_hamiltonian)

    返回:
        H: 3×3 厄米矩阵 [eV]
    """
    H_vac = build_vacuum_hamiltonian(
        energy_gev, theta12, theta23, theta13, delta_cp,
        delta_m2_21, delta_m2_31, hierarchy
    )

    V = float(matter_potential_ev)
    V_mat = np.diag([V, 0.0, 0.0])

    H_mat = H_vac + V_mat
    return H_mat


def solve_hamiltonian_eigen(H):
    """
    求解哈密顿量的本征值和本征向量。

    物理意义:
        H |ν_i^m⟩ = E_i |ν_i^m⟩

    其中 |ν_i^m⟩ 为物质中的有效质量本征态。
    本征值对应有效质量平方差 (在超相对论极限下,
    E_i ≈ m_i^2 / 2E + p)。

    参数:
        H: 3×3 厄米矩阵

    返回:
        eigenvalues:  本征值数组 [eV], 按升序排列
        eigenvectors: 3×3 矩阵, 每列为对应本征向量
        U_matter:     物质中的有效混合矩阵
    """
    # 对于厄米矩阵, 使用 numpy.linalg.eigh (更稳定)
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # eigh 已按升序排列
    # 构建物质中的有效 PMNS 矩阵: U_mat = eigenvectors
    U_matter = eigenvectors

    return eigenvalues, eigenvectors, U_matter


def effective_mixing_angles_in_matter(energy_gev, matter_potential_ev,
                                       theta12=None, theta23=None,
                                       theta13=None, delta_cp=None,
                                       delta_m2_21=None, delta_m2_31=None,
                                       hierarchy='normal'):
    """
    计算物质中的有效混合角。

    对于双味子系统 (例如 e-τ 或 e-μ), MSW 共振条件给出:
        sin²(2θ₁₂^m) = sin²(2θ₁₂) / [ (cos(2θ₁₂) - A/Δm²₂₁)² + sin²(2θ₁₂) ]

    其中 A = 2√2 G_F N_e E = 2 E V_CC。

    参数:
        energy_gev:          中微子能量 [GeV]
        matter_potential_ev: 物质势 V_CC [eV]
        ... (其他参数)

    返回:
        dict: {'theta12_m': ..., 'theta23_m': ..., 'theta13_m': ...}
    """
    from constants import THETA_12, THETA_23, THETA_13

    t12 = THETA_12 if theta12 is None else theta12
    t23 = THETA_23 if theta23 is None else theta23
    t13 = THETA_13 if theta13 is None else theta13
    dm21 = DELTA_M2_21 if delta_m2_21 is None else delta_m2_21

    E_eV = energy_gev * 1e9
    V = matter_potential_ev

    # 12-扇区有效角 (双味近似)
    # A = 2 E V
    A = 2.0 * E_eV * V
    cos2t12 = np.cos(2.0 * t12)
    sin2t12 = np.sin(2.0 * t12)

    denom12 = (cos2t12 - A / dm21) ** 2 + sin2t12 ** 2
    sin2_2theta12_m = sin2t12 ** 2 / denom12

    # 避免 sqrt(负数)
    sin2_2theta12_m = max(0.0, min(1.0, sin2_2theta12_m))
    theta12_m = 0.5 * np.arcsin(np.sqrt(sin2_2theta12_m))

    # 23-扇区在物质中近似不变 (当 V 仅影响 e 味时)
    theta23_m = t23

    # 13-扇区物质修正 (较小)
    cos2t13 = np.cos(2.0 * t13)
    sin2t13 = np.sin(2.0 * t13)
    # 近似使用 Δm²_ee ≈ Δm²₃₁ - sin²(θ₁₂) Δm²₂₁
    dm31 = DELTA_M2_31 if hierarchy == 'normal' else DELTA_M2_31_IH
    dm_ee = np.abs(dm31 - (np.sin(t12) ** 2) * dm21)
    denom13 = (cos2t13 - A / dm_ee) ** 2 + sin2t13 ** 2
    sin2_2theta13_m = sin2t13 ** 2 / denom13
    sin2_2theta13_m = max(0.0, min(1.0, sin2_2theta13_m))
    theta13_m = 0.5 * np.arcsin(np.sqrt(sin2_2theta13_m))

    return {
        'theta12_m': float(theta12_m),
        'theta23_m': float(theta23_m),
        'theta13_m': float(theta13_m)
    }


def msw_resonance_density(energy_gev, theta=None, delta_m2=None):
    """
    计算 MSW 共振发生时的电子数密度 N_e^{res}。

    共振条件 (双味):
        2√2 G_F N_e^{res} E = Δm² cos(2θ)

    因此:
        N_e^{res} = Δm² cos(2θ) / (2√2 G_F E)

    参数:
        energy_gev: 中微子能量 [GeV]
        theta:      真空混合角 [rad], 默认 θ₁₂
        delta_m2:   质量平方差 [eV²], 默认 Δm²₂₁

    返回:
        N_e_res: 共振电子数密度 [cm^{-3}]
    """
    from constants import THETA_12, DELTA_M2_21

    t = THETA_12 if theta is None else theta
    dm2 = DELTA_M2_21 if delta_m2 is None else delta_m2
    E_eV = energy_gev * 1e9

    cos2t = np.cos(2.0 * t)
    numerator = dm2 * cos2t
    denominator = 2.0 * np.sqrt(2.0) * GF * E_eV

    # GF [GeV^{-2}], E [eV] -> 需要统一单位
    # GF = 1.166e-5 GeV^{-2} = 1.166e-5 / (1e9 eV)^{-2} = 1.166e-5 * 1e18 eV^{-2}
    GF_eV2 = GF * 1e18  # [eV^{-2}]
    denominator = 2.0 * np.sqrt(2.0) * GF_eV2 * E_eV

    ne_res_eV3 = numerator / denominator  # [eV^3]

    # eV^3 -> cm^{-3}: 1 eV = 1/(1.973e-5 cm^{-1}), 1 eV^3 = (1.973e-5)^3 cm^{-3}
    hbarc_cm = 1.973269804e-5  # eV * cm
    ne_res_cm3 = ne_res_eV3 * (hbarc_cm ** 3)

    return float(ne_res_cm3)


def hierarchy_discrimination_significance(delta_m2_31, sigma_dm31=0.03e-3):
    """
    计算质量 hierarchy 判别的统计显著性。

    基本思想:
        - NH: Δm²₃₁ > 0
        - IH: Δm²₃₁ < 0

    当 |Δm²₃₁| >> σ(Δm²₃₁) 时, 可通过 Δm²₃₁ 的符号直接判定 hierarchy。

    显著性 (σ 数):
        S = |Δm²₃₁| / σ(Δm²₃₁)

    参数:
        delta_m2_31: 观测到的 Δm²₃₁ [eV²]
        sigma_dm31:  Δm²₃₁ 的测量不确定度 [eV²]

    返回:
        significance: 以 σ 为单位的显著性
        hierarchy:    'normal', 'inverted', 或 'undetermined'
    """
    if sigma_dm31 <= 0:
        raise ValueError("sigma_dm31 must be positive")

    significance = abs(delta_m2_31) / sigma_dm31

    if delta_m2_31 > 0:
        hierarchy = 'normal'
    elif delta_m2_31 < 0:
        hierarchy = 'inverted'
    else:
        hierarchy = 'undetermined'

    return float(significance), hierarchy


def compute_oscillation_wavelengths(energy_gev, delta_m2_21=None, delta_m2_31=None):
    """
    计算中微子振荡波长。

    定义:
        L_{ij} = 4π E / Δm²_{ij} = 2.48 km × (E [GeV]) / (Δm² [eV²])

    参数:
        energy_gev: 中微子能量 [GeV]
        delta_m2_21: Δm²₂₁ [eV²]
        delta_m2_31: Δm²₃₁ [eV²]

    返回:
        dict: {'L_21': ..., 'L_31': ..., 'L_32': ...} [km]
    """
    dm21 = DELTA_M2_21 if delta_m2_21 is None else delta_m2_21
    dm31 = DELTA_M2_31 if delta_m2_31 is None else delta_m2_31
    dm32 = dm31 - dm21

    # L_{osc} [km] = 2.48 * E [GeV] / Δm² [eV²]
    factor = 2.48
    return {
        'L_21': factor * energy_gev / abs(dm21),
        'L_31': factor * energy_gev / abs(dm31),
        'L_32': factor * energy_gev / abs(dm32)
    }


def mass_sum_bounds(hierarchy='normal', m_lightest_eV=0.0):
    """
    计算三种中微子质量之和的上下界。

    已知:
        Δm²₂₁ = m²₂ - m²₁
        Δm²₃₁ = m²₃ - m²₁

    Normal Hierarchy:
        m₁ = m_lightest
        m₂ = √(m₁² + Δm²₂₁)
        m₃ = √(m₁² + Δm²₃₁)

    Inverted Hierarchy:
        m₃ = m_lightest
        m₁ = √(m₃² + |Δm²₃₁|)
        m₂ = √(m₃² + |Δm²₃₁| + Δm²₂₁)

    参数:
        hierarchy:      'normal' 或 'inverted'
        m_lightest_eV:  最轻中微子质量 [eV]

    返回:
        dict: {'m1', 'm2', 'm3', 'sum'} [eV]
    """
    if m_lightest_eV < 0:
        raise ValueError("m_lightest_eV must be non-negative")

    dm21 = DELTA_M2_21
    dm31 = DELTA_M2_31 if hierarchy == 'normal' else DELTA_M2_31_IH

    if hierarchy == 'normal':
        m1 = m_lightest_eV
        m2 = np.sqrt(m1 ** 2 + dm21)
        m3 = np.sqrt(m1 ** 2 + dm31)
    else:
        m3 = m_lightest_eV
        m1 = np.sqrt(m3 ** 2 + abs(dm31))
        m2 = np.sqrt(m3 ** 2 + abs(dm31) + dm21)

    return {
        'm1': float(m1),
        'm2': float(m2),
        'm3': float(m3),
        'sum': float(m1 + m2 + m3)
    }
