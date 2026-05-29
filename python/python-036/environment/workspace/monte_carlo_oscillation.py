"""
monte_carlo_oscillation.py
蒙特卡洛方法计算中微子振荡概率与参数不确定性分析

基于 cube_monte_carlo 和 square_monte_carlo 的核心思想:
    - 在多维参数空间中均匀随机采样
    - 使用大数定律估计积分值
    - 3D 采样用于能量-角度-基准线空间
    - 2D 采样用于 CP 相位-混合角参数空间

物理应用:
    1. 对 PMNS 参数不确定度进行传播分析
    2. 计算振荡概率在实验能谱上的平均值
    3. 评估质量 hierarchy 判别的置信度
"""

import numpy as np
from constants import (
    THETA_12, THETA_23, THETA_13, DELTA_CP,
    DELTA_M2_21, DELTA_M2_31, DELTA_M2_31_IH
)
from pmns_matrix import build_pmns_matrix
from neutrino_hamiltonian import build_vacuum_hamiltonian, solve_hamiltonian_eigen


def sample_unit_square_2d(n_samples, seed=None):
    """
    在单位正方形 [0,1]×[0,1] 内均匀随机采样。
    (源自 square_monte_carlo)

    参数:
        n_samples: 采样点数
        seed:      随机种子

    返回:
        samples: (n_samples, 2) 数组
    """
    rng = np.random.default_rng(seed)
    return rng.random((n_samples, 2))


def sample_unit_cube_3d(n_samples, seed=None):
    """
    在单位立方体 [0,1]³ 内均匀随机采样。
    (源自 cube_monte_carlo)

    参数:
        n_samples: 采样点数
        seed:      随机种子

    返回:
        samples: (n_samples, 3) 数组
    """
    rng = np.random.default_rng(seed)
    return rng.random((n_samples, 3))


def monomial_value(m, e, x):
    """
    计算单项式值 x[0]^e[0] * x[1]^e[1] * ... * x[m-1]^e[m-1]。
    (源自 cube_monte_carlo / square_monte_carlo)

    参数:
        m: 空间维度
        e: 指数向量
        x: 坐标点

    返回:
        value: 单项式值
    """
    value = 1.0
    for i in range(m):
        if e[i] < 0:
            raise ValueError("Exponents must be non-negative")
        if e[i] == 0:
            continue
        value *= x[i] ** e[i]
    return value


def cube_monomial_integral(e):
    """
    计算单项式在单位立方体 [0,1]^m 上的精确积分。
    (源自 cube_monte_carlo)

    公式:
        I = Π_i 1/(e_i + 1)
    """
    m = len(e)
    integral = 1.0
    for i in range(m):
        if e[i] < 0:
            raise ValueError("Exponents must be non-negative")
        integral /= (e[i] + 1)
    return integral


def monte_carlo_oscillation_probability(
        energy_range_gev, baseline_range_km,
        n_samples=50000, hierarchy='normal',
        param_uncertainties=None, seed=None
):
    """
    使用蒙特卡洛方法计算中微子振荡概率对参数不确定度的平均值。

    物理模型:
        P(ν_α → ν_β; E, L) = |⟨ν_β| e^{-i H L}|ν_α⟩|²

    由于实验测量的能量 E 和基线 L 都有展宽, 我们对 E 和 L 的分布
    进行随机采样, 计算概率的样本均值。

    参数:
        energy_range_gev:    (E_min, E_max) [GeV]
        baseline_range_km:   (L_min, L_max) [km]
        n_samples:           MC 采样点数
        hierarchy:           'normal' 或 'inverted'
        param_uncertainties: dict, 例如 {'theta13': 0.05, 'delta_cp': 0.3}
        seed:                随机种子

    返回:
        result: dict 包含:
            'P_ee_mean', 'P_ee_std': ν_e → ν_e 概率的均值和标准差
            'P_em_mean', 'P_em_std': ν_e → ν_μ 概率
            'P_et_mean', 'P_et_std': ν_e → ν_τ 概率
            'samples': 原始采样数据
    """
    rng = np.random.default_rng(seed)
    E_min, E_max = energy_range_gev
    L_min, L_max = baseline_range_km

    if E_min <= 0 or L_min < 0:
        raise ValueError("Energy and baseline ranges must be positive")

    # 采样 E 和 L
    E_samples = rng.uniform(E_min, E_max, n_samples)
    L_samples = rng.uniform(L_min, L_max, n_samples)

    # 采样 PMNS 参数 (高斯分布, 中心为标称值)
    if param_uncertainties is None:
        param_uncertainties = {}

    t12_samples = rng.normal(
        THETA_12, param_uncertainties.get('theta12', 0.0), n_samples
    )
    t23_samples = rng.normal(
        THETA_23, param_uncertainties.get('theta23', 0.0), n_samples
    )
    t13_samples = rng.normal(
        THETA_13, param_uncertainties.get('theta13', 0.0), n_samples
    )
    dcp_samples = rng.normal(
        DELTA_CP, param_uncertainties.get('delta_cp', 0.0), n_samples
    )

    # 确保混合角在物理范围内
    t12_samples = np.clip(t12_samples, 0.01, np.pi / 2 - 0.01)
    t23_samples = np.clip(t23_samples, 0.01, np.pi / 2 - 0.01)
    t13_samples = np.clip(t13_samples, 0.01, np.pi / 2 - 0.01)

    dm31 = DELTA_M2_31 if hierarchy == 'normal' else DELTA_M2_31_IH

    P_ee = np.zeros(n_samples, dtype=np.float64)
    P_em = np.zeros(n_samples, dtype=np.float64)
    P_et = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        E = E_samples[i]
        L = L_samples[i]

        U = build_pmns_matrix(
            t12_samples[i], t23_samples[i], t13_samples[i], dcp_samples[i]
        )
        M2 = np.diag([0.0, DELTA_M2_21, dm31])

        # H_vac [eV]
        H = (1.0 / (2.0 * E * 1e9)) * (U @ M2 @ U.conj().T)

        # 演化算符: U_prop = exp(-i H L)
        # L [km] -> L [eV^{-1}]: L_eV_inv = L_km * 5.0677e9
        L_ev_inv = L * 5.067730889e9

        # 矩阵指数
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        D = np.diag(np.exp(-1j * eigenvalues * L_ev_inv))
        U_prop = eigenvectors @ D @ eigenvectors.conj().T

        # 初始味态 |ν_e⟩ = (1, 0, 0)^T
        psi0 = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
        psi_L = U_prop @ psi0

        P_ee[i] = abs(psi_L[0]) ** 2
        P_em[i] = abs(psi_L[1]) ** 2
        P_et[i] = abs(psi_L[2]) ** 2

    return {
        'P_ee_mean': float(np.mean(P_ee)),
        'P_ee_std': float(np.std(P_ee)),
        'P_em_mean': float(np.mean(P_em)),
        'P_em_std': float(np.std(P_em)),
        'P_et_mean': float(np.mean(P_et)),
        'P_et_std': float(np.std(P_et)),
        'E_samples': E_samples,
        'L_samples': L_samples,
        'P_ee': P_ee,
        'P_em': P_em,
        'P_et': P_et
    }


def mc_hierarchy_significance(
        energy_gev, baseline_km,
        n_samples=20000, sigma_dm31=0.03e-3, seed=None
):
    """
    使用蒙特卡洛评估质量 hierarchy 判别的显著性。

    方法:
        1. 在 Δm²₃₁ 的测量不确定度范围内采样
        2. 对每个样本, 判断其符号
        3. 统计 NH 和 IH 的假设比例

    参数:
        energy_gev:   中微子能量 [GeV]
        baseline_km:  基线 [km]
        n_samples:    MC 采样数
        sigma_dm31:   Δm²₃₁ 的不确定度 [eV²]
        seed:         随机种子

    返回:
        dict: 包含显著性分析结果
    """
    rng = np.random.default_rng(seed)

    # 在 Δm²₃₁ 的正态分布假设下采样
    dm31_nh = rng.normal(DELTA_M2_31, sigma_dm31, n_samples)
    dm31_ih = rng.normal(DELTA_M2_31_IH, sigma_dm31, n_samples)

    # 计算 NH 假设下观测到正 Δm²₃₁ 的概率
    nh_correct = np.sum(dm31_nh > 0) / n_samples
    ih_correct = np.sum(dm31_ih < 0) / n_samples

    # 使用似然比
    # 在 NH 下, P(data|NH) / P(data|IH)
    # 对于正态分布, 似然比只依赖于 Δm²₃₁ 的符号

    return {
        'nh_correct_rate': float(nh_correct),
        'ih_correct_rate': float(ih_correct),
        'nh_confidence_sigma': float(
            np.sqrt(2.0) * abs(DELTA_M2_31) / sigma_dm31
        ),
        'ih_confidence_sigma': float(
            np.sqrt(2.0) * abs(DELTA_M2_31_IH) / sigma_dm31
        )
    }


def mc_integrate_oscillation_over_spectrum(
        energy_spectrum, weights, baseline_km,
        n_samples_per_bin=1000, hierarchy='normal', seed=None
):
    """
    在能量谱上积分振荡概率 (加权蒙特卡洛)。

    物理场景:
        反应堆中微子能谱是连续分布的,
        实验测量的是对所有能量的加权平均:
            ⟨P⟩ = ∫ dE Φ(E) P(E) / ∫ dE Φ(E)

    参数:
        energy_spectrum: (n_bins,) 能量 bin 中心 [GeV]
        weights:         (n_bins,) 能量 bin 权重 (通量)
        baseline_km:     基线 [km]
        n_samples_per_bin: 每 bin 采样数
        hierarchy:       'normal' 或 'inverted'
        seed:            随机种子

    返回:
        dict: 平均概率和误差估计
    """
    rng = np.random.default_rng(seed)
    n_bins = len(energy_spectrum)

    if len(weights) != n_bins:
        raise ValueError("energy_spectrum and weights must have same length")
    if np.sum(weights) <= 0:
        raise ValueError("weights must sum to positive value")

    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.sum(weights)

    dm31 = DELTA_M2_31 if hierarchy == 'normal' else DELTA_M2_31_IH
    U = build_pmns_matrix()
    M2 = np.diag([0.0, DELTA_M2_21, dm31])

    total_P_ee = 0.0
    total_P_em = 0.0
    total_P_et = 0.0

    for b in range(n_bins):
        E0 = energy_spectrum[b]
        if E0 <= 0:
            continue
        w = weights[b]

        # 在 bin 内均匀采样
        dE = 0.05 * E0  # 假设 bin 宽度为 5%
        E_samples = rng.uniform(E0 - dE, E0 + dE, n_samples_per_bin)
        E_samples = np.clip(E_samples, 0.001, None)

        for E in E_samples:
            H = (1.0 / (2.0 * E * 1e9)) * (U @ M2 @ U.conj().T)
            L_ev_inv = baseline_km * 5.067730889e9

            eigenvalues, eigenvectors = np.linalg.eigh(H)
            D = np.diag(np.exp(-1j * eigenvalues * L_ev_inv))
            U_prop = eigenvectors @ D @ eigenvectors.conj().T

            psi0 = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
            psi_L = U_prop @ psi0

            total_P_ee += w * abs(psi_L[0]) ** 2 / n_samples_per_bin
            total_P_em += w * abs(psi_L[1]) ** 2 / n_samples_per_bin
            total_P_et += w * abs(psi_L[2]) ** 2 / n_samples_per_bin

    return {
        'P_ee_avg': float(total_P_ee),
        'P_em_avg': float(total_P_em),
        'P_et_avg': float(total_P_et),
        'sum_prob': float(total_P_ee + total_P_em + total_P_et)
    }
