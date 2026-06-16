"""
中微子振荡物理核心模块
======================
实现三代中微子振荡的量子力学演化，包括真空和物质中的振荡（MSW效应）。

核心物理方程：
1. 味态与质量态的关系：|ν_α⟩ = Σ_i U_{αi} |ν_i⟩
2. PMNS混合矩阵：U = R23(θ23) × R13(θ13, δ) × R12(θ12)
3. 薛定谔型演化方程：i d/dx |ν(x)⟩ = H(x) |ν(x)⟩
4. 物质中的哈密顿量：H = U diag(0, Δm²₂₁/(2E), Δm²₃₁/(2E)) U† + diag(V(x), 0, 0)
5. 物质势：V(x) = √2 G_F N_e(x)
6. 振荡概率：P(να→νβ) = |⟨νβ|U(x,0)|να⟩|²

其中：
- U: 3×3 PMNS混合矩阵（复数）
- Δm²ij: 质量平方差 (eV²)
- E: 中微子能量 (MeV)
- G_F: 费米常数 = 1.1663787×10⁻⁵ GeV⁻²
- N_e: 电子数密度 (mol/cm³)
"""

import numpy as np
from typing import Tuple, Callable


# 物理常数
FERMI_CONSTANT = 1.1663787e-5  # GeV^-2
HBAR_C = 1.97327e-7  # eV·m (约化普朗克常数×光速)
SQRT2 = np.sqrt(2.0)


def pmns_matrix(theta12: float, theta13: float, theta23: float, delta_cp: float = 0.0) -> np.ndarray:
    """
    构造PMNS混合矩阵（复数3×3幺正矩阵）。

    数学表达：
    U = R23(θ23) × R13(θ13, δ) × R12(θ12)

    其中：
    R12 = [[c12, s12, 0], [-s12, c12, 0], [0, 0, 1]]
    R13 = [[c13, 0, s13*e^{-iδ}], [0, 1, 0], [-s13*e^{iδ}, 0, c13]]
    R23 = [[1, 0, 0], [0, c23, s23], [0, -s23, c23]]

    参数：
        theta12, theta13, theta23: 混合角（弧度）
        delta_cp: CP破坏相角（弧度）

    返回：
        U: 3×3复数幺正矩阵

    边界处理：
        - 角度限制在 [0, π/2] 范围内
        - δ限制在 [0, 2π) 范围内
    """
    # 边界检查
    theta12 = np.clip(theta12, 0, np.pi/2)
    theta13 = np.clip(theta13, 0, np.pi/2)
    theta23 = np.clip(theta23, 0, np.pi/2)
    delta_cp = delta_cp % (2 * np.pi)

    c12, s12 = np.cos(theta12), np.sin(theta12)
    c13, s13 = np.cos(theta13), np.sin(theta13)
    c23, s23 = np.cos(theta23), np.sin(theta23)

    # R12矩阵
    R12 = np.array([
        [c12, s12, 0],
        [-s12, c12, 0],
        [0, 0, 1]
    ], dtype=complex)

    # R13矩阵（包含CP相角）
    R13 = np.array([
        [c13, 0, s13 * np.exp(-1j * delta_cp)],
        [0, 1, 0],
        [-s13 * np.exp(1j * delta_cp), 0, c13]
    ], dtype=complex)

    # R23矩阵
    R23 = np.array([
        [1, 0, 0],
        [0, c23, s23],
        [0, -s23, c23]
    ], dtype=complex)

    # U = R23 × R13 × R12
    U = R23 @ R13 @ R12

    # 验证幺正性：U†U = I
    identity_check = U.conj().T @ U
    unitarity_error = np.linalg.norm(identity_check - np.eye(3))
    if unitarity_error > 1e-10:
        # 通过SVD投影到最近的幺正矩阵
        u_svd, _, vh_svd = np.linalg.svd(U)
        U = u_svd @ vh_svd

    return U


def matter_potential(electron_density: float, energy_MeV: float) -> float:
    """
    计算物质势 V(x) = √2 G_F N_e(x)。

    物理背景：
    当中微子在物质中传播时，与电子的相干前向散射产生有效势。
    对于νe，势为 V = √2 G_F N_e；对于νμ和ντ，势为0。

    单位转换：
    V [eV] = √2 × G_F [GeV⁻²] × N_e [mol/cm³] × N_A [mol⁻¹] × (ℏc)³ [eV³·m³] × 10³⁰ [cm³/m³]

    参数：
        electron_density: 电子数密度 (mol/cm³)
        energy_MeV: 中微子能量 (MeV)，用于单位归一化

    返回：
        V: 物质势 (eV)

    边界处理：
        - 电子密度必须非负
        - 能量必须为正
    """
    if electron_density < 0:
        raise ValueError(f"电子密度不能为负: {electron_density}")
    if energy_MeV <= 0:
        raise ValueError(f"中微子能量必须为正: {energy_MeV}")

    # 阿伏伽德罗常数
    N_A = 6.02214076e23  # mol^-1

    # 单位转换因子：GeV^-2 → eV^-1
    GeV_to_eV = 1e9
    GF_eV = FERMI_CONSTANT / (GeV_to_eV ** 2)  # eV^-2

    # N_e 从 mol/cm³ 转换为 1/m³
    N_e_m3 = electron_density * N_A * 1e6  # 1/m³

    # (ℏc)³ 用于体积转换
    hbar_c_m = HBAR_C * 1e-9  # m
    hbar_c_cube = hbar_c_m ** 3

    # V = √2 G_F N_e
    V = SQRT2 * GF_eV * N_e_m3 * hbar_c_cube

    # 归一化到能量尺度
    V_normalized = V / (energy_MeV * 1e6)  # 无量纲

    return V_normalized


def hamiltonian_vacuum(U: np.ndarray, delta_m2_21: float, delta_m2_31: float,
                       energy_eV: float) -> np.ndarray:
    """
    构造真空中的哈密顿量（质量基底下为对角矩阵）。

    物理方程：
    H_vac = U diag(0, Δm²₂₁/(2E), Δm²₃₁/(2E)) U†

    其中：
    - Δm²₂₁ = m²₂ - m²₁ (太阳质量平方差)
    - Δm²₃₁ = m²₃ - m²₁ (大气质量平方差)
    - E: 中微子能量

    参数：
        U: PMNS混合矩阵 (3×3)
        delta_m2_21: 质量平方差 Δm²₂₁ (eV²)
        delta_m2_31: 质量平方差 Δm²₃₁ (eV²)
        energy_eV: 中微子能量 (eV)

    返回：
        H: 3×3厄米特矩阵 (eV)

    边界处理：
        - 能量必须为正
        - 质量平方差可以为负（倒序质量层级）
    """
    if energy_eV <= 0:
        raise ValueError(f"能量必须为正: {energy_eV}")

    # 质量本征值（质量基底）
    m2_diag = np.array([0.0, delta_m2_21, delta_m2_31], dtype=float)

    # 哈密顿量在质量基底：diag(m²/(2E))
    H_mass = np.diag(m2_diag / (2.0 * energy_eV))

    # 转换到味基底：H_flavor = U H_mass U†
    H_flavor = U @ H_mass @ U.conj().T

    # 验证厄米特性
    hermitian_error = np.linalg.norm(H_flavor - H_flavor.conj().T)
    if hermitian_error > 1e-12:
        # 强制厄米化
        H_flavor = 0.5 * (H_flavor + H_flavor.conj().T)

    return H_flavor


def hamiltonian_matter(U: np.ndarray, delta_m2_21: float, delta_m2_31: float,
                       energy_eV: float, electron_density: float) -> np.ndarray:
    """
    构造物质中的哈密顿量（包含MSW效应）。

    物理方程：
    H_matter = H_vacuum + diag(V, 0, 0)

    其中 V = √2 G_F N_e 是物质势。

    参数：
        U: PMNS混合矩阵
        delta_m2_21, delta_m2_31: 质量平方差 (eV²)
        energy_eV: 中微子能量 (eV)
        electron_density: 电子数密度 (mol/cm³)

    返回：
        H: 3×3厄米特矩阵 (eV)
    """
    # 真空哈密顿量
    H_vac = hamiltonian_vacuum(U, delta_m2_21, delta_m2_31, energy_eV)

    # 物质势
    V = matter_potential(electron_density, energy_eV / 1e6)

    # 添加物质势到电子味态
    H_matter = H_vac.copy()
    H_matter[0, 0] += V

    return H_matter


def oscillation_probability(U_evolution: np.ndarray, alpha: int, beta: int) -> float:
    """
    计算振荡概率 P(να → νβ)。

    物理方程：
    P(να → νβ) = |⟨νβ|U(x,0)|να⟩|² = |U_evolution[β, α]|²

    参数：
        U_evolution: 演化算符 U(x, 0) (3×3复数矩阵)
        alpha: 初态味指标 (0=e, 1=μ, 2=τ)
        beta: 末态味指标 (0=e, 1=μ, 2=τ)

    返回：
        P: 振荡概率 [0, 1]

    边界处理：
        - 概率限制在 [0, 1] 范围内
        - 检查概率归一化：Σ_β P(να → νβ) = 1
    """
    if not (0 <= alpha <= 2 and 0 <= beta <= 2):
        raise ValueError(f"味指标必须在 [0, 2] 范围内: alpha={alpha}, beta={beta}")

    # P = |U[β, α]|²
    P = np.abs(U_evolution[beta, alpha]) ** 2

    # 边界限制
    P = np.clip(P, 0.0, 1.0)

    return P


def check_probability_normalization(U_evolution: np.ndarray, alpha: int,
                                    tolerance: float = 1e-10) -> bool:
    """
    检查概率归一化：Σ_β P(να → νβ) = 1。

    物理约束：
    由于U是幺正矩阵，概率必须归一化。

    参数：
        U_evolution: 演化算符 (3×3)
        alpha: 初态味指标
        tolerance: 容差

    返回：
        is_normalized: 是否归一化
    """
    total_prob = sum(oscillation_probability(U_evolution, alpha, beta) for beta in range(3))
    return abs(total_prob - 1.0) < tolerance


def effective_mixing_angles_matter(U: np.ndarray, delta_m2_21: float, delta_m2_31: float,
                                    energy_eV: float, electron_density: float) -> Tuple[float, float, float]:
    """
    计算物质中的有效混合角（MSW共振）。

    物理背景：
    在物质中，有效混合角和有效质量平方差与真空中不同。
    当 V(x) ≈ Δm² cos(2θ) / (2E) 时发生MSW共振。

    近似公式（双味混合）：
    sin²(2θ_m) = sin²(2θ) / [(cos(2θ) - 2EV/Δm²)² + sin²(2θ)]
    Δm²_m = Δm² √[(cos(2θ) - 2EV/Δm²)² + sin²(2θ)]

    参数：
        U: 真空PMNS矩阵
        delta_m2_21, delta_m2_31: 真空质量平方差
        energy_eV: 能量
        electron_density: 电子密度

    返回：
        theta12_m, theta13_m, theta23_m: 物质中的有效混合角 (弧度)
    """
    # 简化处理：使用真空混合角作为起点
    # 完整的物质效应需要对角化完整哈密顿量
    theta12_vac = np.arcsin(np.abs(U[0, 1]))
    theta13_vac = np.arcsin(np.abs(U[0, 2]))
    theta23_vac = np.arcsin(np.abs(U[1, 2]) / np.cos(theta13_vac))

    # 物质势
    V = matter_potential(electron_density, energy_eV / 1e6)

    # 太阳扇区的有效混合角（双味近似）
    if delta_m2_21 != 0:
        cos_2theta_12 = np.cos(2 * theta12_vac)
        sin_2theta_12 = np.sin(2 * theta12_vac)
        resonance_term_12 = 2 * energy_eV * V / delta_m2_21

        denominator_12 = np.sqrt((cos_2theta_12 - resonance_term_12)**2 + sin_2theta_12**2)
        sin2_2theta12_m = sin_2theta_12**2 / denominator_12**2
        theta12_m = 0.5 * np.arcsin(np.sqrt(np.clip(sin2_2theta12_m, 0, 1)))
    else:
        theta12_m = theta12_vac

    # 大气扇区的有效混合角
    if delta_m2_31 != 0:
        cos_2theta_13 = np.cos(2 * theta13_vac)
        sin_2theta_13 = np.sin(2 * theta13_vac)
        resonance_term_13 = 2 * energy_eV * V / delta_m2_31

        denominator_13 = np.sqrt((cos_2theta_13 - resonance_term_13)**2 + sin_2theta_13**2)
        sin2_2theta13_m = sin_2theta_13**2 / denominator_13**2
        theta13_m = 0.5 * np.arcsin(np.sqrt(np.clip(sin2_2theta13_m, 0, 1)))
    else:
        theta13_m = theta13_vac

    # θ23在物质中近似不变
    theta23_m = theta23_vac

    return theta12_m, theta13_m, theta23_m
