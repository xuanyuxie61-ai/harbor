"""
high_throughput_screen.py
=========================
高通量材料筛选模块。

科学背景:
  材料基因组计划 (MGI) 的核心是通过高通量计算快速评估大量候选材料。
  本模块整合所有前述模块, 对 LLZO 固态电解质进行多参数筛选:

    设计变量:
      - 掺杂浓度 x (Al, Ga, Ta, Nb)
      - 晶格应变 eps
      - 温度 T

    目标函数:
      - 离子电导率 sigma (最大化)
      - 电化学窗口 Delta_V (最大化)
      - 机械稳定性 (弹性模量)
      - 热稳定性 (分解温度)

    约束:
      - 热力学稳定性: E_hull < 25 meV/atom
      - 机械稳定性: Born 准则
      - 带隙 > 2 eV (电子绝缘)

  高通量工作流:
    1. 参数空间离散化 (稀疏网格, 源自 143)
    2. 并行评估每个参数点:
       a. 晶体结构编码 -> 描述符
       b. 离子输运模拟 (PNP 方程)
       c. 热传导分析
       d. 稳定性判断 (谱分析)
    3. 代理模型训练 (ML)
    4. Pareto 前沿识别

  Born 稳定性准则 (立方晶系):
    C11 > |C12|
    C11 + 2*C12 > 0
    C44 > 0

  弹性模量 (Voigt-Reuss-Hill 平均):
    K_V = (C11 + 2*C12) / 3
    G_V = (C11 - C12 + 3*C44) / 5
    K_R = K_V (立方)
    G_R = 5*(C11-C12)*C44 / (4*C44 + 3*(C11-C12))
    K_VRH = (K_V + K_R) / 2
    G_VRH = (G_V + G_R) / 2

  Debye 温度:
    Theta_D = (hbar/kB) * (6*pi^2*n)^{1/3} * v_m
    v_m = [1/3 * (2/v_t^3 + 1/v_l^3)]^{-1/3}
    v_l = sqrt((K + 4G/3) / rho)
    v_t = sqrt(G / rho)

  Pareto 前沿:
    解 x* 为 Pareto 最优 iff 不存在其他 x 使得:
      f_i(x) >= f_i(x*) for all i AND f_j(x) > f_j(x*) for some j
"""

import numpy as np
from material_constants import (
    KB, ELEMENTARY_CHARGE, LLZO_DIFFUSION_COEFF, LLZO_IONIC_CONDUCTIVITY,
    LLZO_ACTIVATION_ENERGY, LLZO_DENSITY, SMALL_NUMBER,
    DOPANT_CONCENTRATIONS, STRAIN_RANGE, TEMPERATURE_RANGE,
)
from ml_potential import GaussianKernelRegressor, generate_llzo_training_data, arrhenius_fit
from crystal_descriptor import composition_descriptor, lattice_volume, band_structure_descriptor
from stability_analysis import spectral_stability
from quadrature_rules import clenshaw_curtis_sparse_grid


# ============================================================================
# 弹性常数与机械稳定性
# ============================================================================
def elastic_constants_llzo(x_dopant, strain):
    """
    LLZO 的掺杂-应变依赖弹性常数 (经验模型)。

    立方晶系独立常数: C11, C12, C44

    未掺杂 LLZO (GPa):
      C11 ≈ 280, C12 ≈ 90, C44 ≈ 80

    掺杂效应:
      C11(x) = C11_0 * (1 - 0.5*x)
      C12(x) = C12_0 * (1 - 0.3*x)
      C44(x) = C44_0 * (1 + 0.2*x)

    应变效应:
      C_ij(eps) = C_ij * (1 - 3*eps) (线性软化)
    """
    C11_0 = 280e9  # Pa
    C12_0 = 90e9
    C44_0 = 80e9

    x = abs(x_dopant)
    C11 = C11_0 * (1.0 - 0.5*x) * (1.0 - 3.0*strain)
    C12 = C12_0 * (1.0 - 0.3*x) * (1.0 - 3.0*strain)
    C44 = C44_0 * (1.0 + 0.2*x) * (1.0 - 3.0*strain)

    return C11, C12, C44


def born_stability_check(C11, C12, C44):
    """
    立方晶系 Born 机械稳定性准则:

    C11 > |C12|
    C11 + 2*C12 > 0
    C44 > 0

    返回: (is_stable, violation_list)
    """
    violations = []
    if C11 <= abs(C12):
        violations.append(f"C11 ({C11/1e9:.1f} GPa) <= |C12| ({abs(C12)/1e9:.1f} GPa)")
    if C11 + 2*C12 <= 0:
        violations.append(f"C11+2*C12 = {(C11+2*C12)/1e9:.1f} GPa <= 0")
    if C44 <= 0:
        violations.append(f"C44 = {C44/1e9:.1f} GPa <= 0")
    return len(violations) == 0, violations


def vrh_moduli(C11, C12, C44):
    """
    Voigt-Reuss-Hill 平均弹性模量:

    K_V = (C11 + 2*C12) / 3
    G_V = (C11 - C12 + 3*C44) / 5
    K_R = K_V (立方晶系)
    G_R = 5*(C11-C12)*C44 / (4*C44 + 3*(C11-C12))

    返回: (K_VRH, G_VRH, E_VRH, nu_VRH)
    """
    K_V = (C11 + 2*C12) / 3.0
    G_V = (C11 - C12 + 3*C44) / 5.0
    K_R = K_V
    num_R = 5.0 * (C11 - C12) * C44
    den_R = 4*C44 + 3*(C11 - C12)
    G_R = num_R / max(den_R, SMALL_NUMBER)

    K_VRH = 0.5 * (K_V + K_R)
    G_VRH = 0.5 * (G_V + G_R)

    # Young's modulus: E = 9KG / (3K + G)
    E_VRH = 9.0 * K_VRH * G_VRH / max(3.0*K_VRH + G_VRH, SMALL_NUMBER)
    # Poisson ratio: nu = (3K - 2G) / (2*(3K + G))
    nu_VRH = (3.0*K_VRH - 2.0*G_VRH) / max(2.0*(3.0*K_VRH + G_VRH), SMALL_NUMBER)

    return K_VRH, G_VRH, E_VRH, nu_VRH


def debye_temperature(K_VRH, G_VRH, rho, n_atoms_per_volume, M_avg):
    """
    Debye 温度:
      Theta_D = (hbar/kB) * (6*pi^2*n)^{1/3} * v_m

    其中:
      v_l = sqrt((K + 4G/3) / rho)  (纵波速度)
      v_t = sqrt(G / rho)  (横波速度)
      v_m = [1/3 * (2/v_t^3 + 1/v_l^3)]^{-1/3}  (平均声速)
      n = n_atoms / V (数密度)
    """
    from material_constants import HBAR
    v_l = np.sqrt((K_VRH + 4.0*G_VRH/3.0) / max(rho, SMALL_NUMBER))
    v_t = np.sqrt(G_VRH / max(rho, SMALL_NUMBER))
    v_m = (1.0/3.0 * (2.0/max(v_t**3, SMALL_NUMBER) + 1.0/max(v_l**3, SMALL_NUMBER)))**(-1.0/3.0)
    n = n_atoms_per_volume
    Theta_D = (HBAR / KB) * (6.0 * np.pi**2 * n)**(1.0/3.0) * v_m
    return Theta_D


# ============================================================================
# 高通量工作流
# ============================================================================
def screening_parameter_grid(x_vals, strain_vals, T_vals):
    """
    构建参数网格 (全张量或稀疏)。

    返回:
        param_grid: [N_total, 3] 的数组
    """
    grid = []
    for x in x_vals:
        for eps in strain_vals:
            for T in T_vals:
                grid.append([x, eps, T])
    return np.array(grid)


def screening_sparse_grid(doping_range, strain_range, T_range, level=3):
    """
    使用 Smolyak 稀疏网格构建参数采样。

    将 [0,0.3] x [-0.02,0.02] x [250,350] 映射到 [-1,1]^3
    """
    pts_raw, wts = clenshaw_curtis_sparse_grid(3, level)

    # 映射: [-1,1] -> [a, b]
    def scale_to(x, a, b):
        return 0.5 * (x * (b - a) + (a + b))

    pts = np.zeros_like(pts_raw)
    pts[:, 0] = scale_to(pts_raw[:, 0], doping_range[0], doping_range[1])
    pts[:, 1] = scale_to(pts_raw[:, 1], strain_range[0], strain_range[1])
    pts[:, 2] = scale_to(pts_raw[:, 2], T_range[0], T_range[1])
    return pts, wts


def evaluate_candidate(x_dopant, strain, temperature):
    """
    评估单个候选材料。

    返回字典:
        - sigma: 离子电导率 [S/m]
        - E_a: 活化能 [eV]
        - C11, C12, C44: 弹性常数 [Pa]
        - K, G, E, nu: 弹性模量 [Pa]
        - mech_stable: 机械稳定性
        - debye_T: Debye 温度 [K]
        - score: 综合评分
    """
    # 离子输运
    kB_eV = KB / ELEMENTARY_CHARGE
    E_a0 = LLZO_ACTIVATION_ENERGY
    # 掺杂效应
    E_a = E_a0 - 0.5*x_dopant + 2.0*x_dopant**2 + 3.0*strain + 50.0*strain**2 - 10.0*x_dopant*strain
    E_a = max(E_a, 0.1)
    sigma_0 = LLZO_IONIC_CONDUCTIVITY * np.exp(E_a0 / (kB_eV * 300.0))
    sigma = sigma_0 * np.exp(-E_a / (kB_eV * temperature))

    # 弹性
    C11, C12, C44 = elastic_constants_llzo(x_dopant, strain)
    mech_stable, violations = born_stability_check(C11, C12, C44)
    K, G, E, nu = vrh_moduli(C11, C12, C44)

    # Debye 温度
    n_atoms = 480 / (12.97e-10)**3  # 近似
    Theta_D = debye_temperature(K, G, LLZO_DENSITY, n_atoms, 50.0)

    # 综合评分: log(sigma) + alpha * mech_stable + beta * Theta_D/500
    score = np.log10(max(sigma, SMALL_NUMBER)) + 2.0 * (1.0 if mech_stable else -5.0) + 0.3 * Theta_D / 500.0

    return {
        'x_dopant': x_dopant,
        'strain': strain,
        'temperature': temperature,
        'E_a_eV': E_a,
        'sigma_S_m': sigma,
        'C11_GPa': C11/1e9,
        'C12_GPa': C12/1e9,
        'C44_GPa': C44/1e9,
        'K_GPa': K/1e9,
        'G_GPa': G/1e9,
        'E_GPa': E/1e9,
        'nu': nu,
        'mech_stable': mech_stable,
        'Theta_D': Theta_D,
        'score': score,
    }


def run_high_throughput_screening(param_grid):
    """
    对参数网格中的每个候选材料进行评估。

    返回:
        results: list of dict
        pareto_front: Pareto 最优解的索引
    """
    results = []
    for i in range(len(param_grid)):
        x, eps, T = param_grid[i]
        res = evaluate_candidate(x, eps, T)
        results.append(res)
    return results


def pareto_front(objectives, maximize=True):
    """
    识别 Pareto 前沿 (多目标优化)。

    参数:
        objectives: [N, M] M 个目标
        maximize: 是否所有目标都最大化

    返回:
        is_pareto: [N] bool
    """
    N = len(objectives)
    if not maximize:
        objectives = -objectives
    is_pareto = np.ones(N, dtype=bool)
    for i in range(N):
        if not is_pareto[i]:
            continue
        for j in range(N):
            if i == j or not is_pareto[j]:
                continue
            # 检查 j 是否支配 i
            if np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]):
                is_pareto[i] = False
                break
    return is_pareto


def screening_summary(results):
    """
    生成筛选结果汇总。
    """
    sigmas = np.array([r['sigma_S_m'] for r in results])
    E_as = np.array([r['E_a_eV'] for r in results])
    scores = np.array([r['score'] for r in results])
    mech_stables = np.array([r['mech_stable'] for r in results])

    best_idx = np.argmax(scores)
    best = results[best_idx]

    summary = {
        'n_candidates': len(results),
        'n_mech_stable': int(np.sum(mech_stables)),
        'sigma_max': np.max(sigmas),
        'sigma_min': np.min(sigmas),
        'sigma_mean': np.mean(sigmas),
        'E_a_min': np.min(E_as),
        'E_a_max': np.max(E_as),
        'best_candidate': best,
        'best_index': int(best_idx),
    }
    return summary
