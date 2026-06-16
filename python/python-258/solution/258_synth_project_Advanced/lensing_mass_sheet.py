"""
lensing_mass_sheet.py — 质量片简并消除与参数变化分析
====================================================

来源种子: 1201_de-ranit_GPP-ModelParamVariation (参数变化实验设计),
          1165_bioinfo-tsukuba_GEMS-python-article-data (实验状态机/调度)
科学角色: 处理弱透镜质量重建中最基本的简并 —
         质量片变换 (Mass Sheet Transformation, MST),
         并实现宇宙学参数 Ω_m 变化对重建的影响分析。

质量片变换 (MST):
=================
给定透镜模型 (κ, γ), 一族变换后的模型:
    κ_λ = λκ + (1-λ)
    γ_λ = λγ
    (对所有 λ > 0)

性质:
    1. γ_λ 产生与 γ 完全相同的观测 (因为 g = γ/(1-κ) 不变):
       g_λ = γ_λ/(1-κ_λ) = λγ/(1-λκ-(1-λ)) = λγ/(λ(1-κ)) = γ/(1-κ) = g
    2. 但 κ_λ ≠ κ (除非 λ=1)
    3. 这意味着仅从弱剪切观测无法确定收敛场的绝对归一化

打破简并的方法:
    1. 强透镜约束 (Einstein 半径)
    2. 星系动力学 (速度弥散)
    3. X 射线观测 (气体质量)
    4. 宇宙学先验 (Σ_crit 依赖 Ω_m)

宇宙学参数依赖性:
    Σ_crit = (c²/(4πG)) × D_S/(D_L × D_LS)
    ∝ 1/D_A(z_L, z_S)

    D_A 依赖 Ω_m, Ω_Λ:
    D_A(z) = (c/H₀)/(1+z) × ∫₀ᶻ dz'/E(z')
    E(z) = √(Ω_m(1+z)³ + Ω_Λ)

    因此: κ ∝ 1/Σ_crit ∝ D_A(z_L, z_S)
    改变 Ω_m → 改变 D_A → 改变 κ 的归一化
"""

import numpy as np
from lensing_config import COSMO, GRID, RECON
from lensing_cosmology import (
    angular_diameter_distance, sigma_crit, E_z
)
from lensing_finite_diff import laplacian


def mass_sheet_transform(kappa, gamma1, gamma2, lam):
    """
    应用质量片变换 (MST)

    κ_λ = λκ + (1-λ)
    γ_λ = λγ

    物理意义:
    - λ > 1: 增加质量 (over-convergence)
    - λ < 1: 减少质量 (under-convergence)
    - λ = 1: 恒等变换
    - λ = 0: 纯质量片 (κ=1, γ=0)

    观测不变性:
        g_λ = γ_λ/(1-κ_λ) = λγ/(λ(1-κ)) = γ/(1-κ) = g

    Parameters
    ----------
    kappa : ndarray
        原始收敛场
    gamma1, gamma2 : ndarray
        原始剪切场
    lam : float
        MST 参数 (λ > 0)

    Returns
    -------
    kappa_new, gamma1_new, gamma2_new : ndarray
        变换后的场
    """
    if lam <= 0.0:
        raise ValueError(f"MST parameter λ must be positive, got {lam}")

    kappa_new = lam * kappa + (1.0 - lam)
    gamma1_new = lam * gamma1
    gamma2_new = lam * gamma2

    return kappa_new, gamma1_new, gamma2_new


def estimate_lambda_from_kappa_stats(kappa_recon, kappa_true=None, method='variance'):
    """
    从重建收敛场估计 MST 参数 λ

    方法 1 (方差匹配):
        如果知道真实 κ 的方差, 通过匹配方差估计 λ:
        Var(κ_λ) = λ² Var(κ)
        ⟹ λ = √(Var(κ_true) / Var(κ_recon))

    方法 2 (峰值统计):
        利用 NFW 轮廓的峰值与面积关系:
        κ_peak ∝ M_vir / (r_s² Σ_crit)
        结合质量-浓度关系约束 λ

    方法 3 (零均值约束):
        在宇宙学先验下, <κ> 应等于宇宙平均:
        <κ> = Ω_m/2 × ∫ dz n(z) ... ≈ 0.01-0.05
        λ = <κ>_cosmo / <κ>_recon

    Parameters
    ----------
    kappa_recon : ndarray
        重建的收敛场
    kappa_true : ndarray or None
        真实收敛场 (如果可用)
    method : str
        估计方法 ('variance', 'peak', 'mean')

    Returns
    -------
    lam_est : float
        估计的 λ
    """
    if method == 'variance' and kappa_true is not None:
        var_true = np.var(kappa_true)
        var_recon = np.var(kappa_recon)
        if var_recon > 1.0e-30:
            lam_est = np.sqrt(var_true / var_recon)
        else:
            lam_est = 1.0
    elif method == 'mean':
        # 宇宙学先验: <κ> ≈ 0.02 (典型值)
        kappa_cosmo = 0.02
        kappa_mean = np.mean(kappa_recon)
        if abs(kappa_mean) > 1.0e-10:
            lam_est = kappa_cosmo / kappa_mean
        else:
            lam_est = 1.0
    elif method == 'peak' and kappa_true is not None:
        peak_true = np.max(kappa_true)
        peak_recon = np.max(kappa_recon)
        if peak_recon > 1.0e-10:
            lam_est = peak_true / peak_recon
        else:
            lam_est = 1.0
    else:
        lam_est = 1.0

    return lam_est


def cosmo_dependence_omega_m(z_l=0.3, z_s=1.0, omega_m_range=None, n_points=20):
    """
    分析 Σ_crit 对 Ω_m 的依赖

    在 ΛCDM 中 (平坦宇宙, Ω_Λ = 1-Ω_m):
        Σ_crit(Ω_m) ∝ D_S / (D_L × D_LS)

    其中 D_A(z₁,z₂) = (c/H₀)/(1+z₂) ∫_{z₁}^{z₂} dz'/E(z')
    E(z) = √(Ω_m(1+z)³ + (1-Ω_m))

    预期行为:
    - Ω_m 增大 → H(z) 增大 → D_A 减小 → Σ_crit 变化
    - 在 z~0.5 附近, Σ_crit 对 Ω_m 的灵敏度约为 20-30%
      (当 Ω_m 从 0.2 变到 0.4)

    Parameters
    ----------
    z_l : float
        透镜红移
    z_s : float
        源红移
    omega_m_range : ndarray or None
        Ω_m 扫描范围
    n_points : int
        扫描点数

    Returns
    -------
    omega_m_values : ndarray
        Ω_m 值
    sigma_crit_values : ndarray
        Σ_crit(Ω_m) (M_sun/Mpc²)
    kappa_scale : ndarray
        κ 的相对标定 κ(Ω_m)/κ(Ω_m_fid)
    """
    if omega_m_range is None:
        omega_m_range = np.linspace(0.15, 0.50, n_points)

    sigma_crit_values = np.zeros(len(omega_m_range))
    Omega_L_fid = COSMO.Omega_L
    Omega_m_fid = COSMO.Omega_m

    for i, Om in enumerate(omega_m_range):
        # 临时修改宇宙学参数
        COSMO.Omega_m = Om
        COSMO.Omega_L = 1.0 - Om

        sc = sigma_crit(z_l, z_s)
        sigma_crit_values[i] = sc

    # 恢复原值
    COSMO.Omega_m = Omega_m_fid
    COSMO.Omega_L = Omega_L_fid

    # κ 的相对标定 (κ ∝ 1/Σ_crit)
    sc_fid = sigma_crit(z_l, z_s)
    kappa_scale = sc_fid / np.maximum(sigma_crit_values, 1.0e-10)

    return omega_m_range, sigma_crit_values, kappa_scale


def mass_reconstruction_with_omega_m_sweep(gamma1_obs, gamma2_obs, h,
                                            omega_m_values=None):
    """
    在不同 Ω_m 下进行质量重建, 评估宇宙学参数不确定性

    流程 (GEMS 实验状态机模式):
    1. 对每个 Ω_m:
       a. 重新计算 Σ_crit(Ω_m)
       b. 重标定剪切场: γ' = γ × (Σ_crit,fid/Σ_crit(Ω_m))
       c. 执行 KS 重建
       d. 记录收敛场统计

    这种参数扫描类似于 GPP-ModelParamVariation 中的
    年际参数变化实验设计。

    Parameters
    ----------
    gamma1_obs, gamma2_obs : ndarray
        观测剪切场
    h : float
        网格间距
    omega_m_values : ndarray or None
        Ω_m 扫描值

    Returns
    -------
    results : dict
        各 Ω_m 下的重建结果
    """
    if omega_m_values is None:
        omega_m_values = np.linspace(0.20, 0.40, 8)

    from lensing_kaiser_squires import kaiser_squires

    z_l = 0.3
    z_s = 1.0
    Omega_m_fid = COSMO.Omega_m
    Omega_L_fid = COSMO.Omega_L

    # 基准 Σ_crit
    sc_fid = sigma_crit(z_l, z_s)

    results = {
        'omega_m': [],
        'sigma_crit': [],
        'kappa_max': [],
        'kappa_mean': [],
        'kappa_var': [],
        'total_mass_proxy': [],
    }

    for Om in omega_m_values:
        COSMO.Omega_m = Om
        COSMO.Omega_L = 1.0 - Om

        sc = sigma_crit(z_l, z_s)

        # 重标定 (在 Σ_crit 变化时, 物理 κ 改变)
        # 但观测剪切是几何量, 不直接依赖 Ω_m
        # 这里模拟的是: 同一观测数据在不同宇宙学下的物理解释

        kappa_ks, _ = kaiser_squires(gamma1_obs, gamma2_obs, h, apply_filter=True)

        results['omega_m'].append(Om)
        results['sigma_crit'].append(sc)
        results['kappa_max'].append(np.max(kappa_ks))
        results['kappa_mean'].append(np.mean(kappa_ks))
        results['kappa_var'].append(np.var(kappa_ks))
        results['total_mass_proxy'].append(np.sum(kappa_ks))

    # 恢复
    COSMO.Omega_m = Omega_m_fid
    COSMO.Omega_L = Omega_L_fid

    for key in results:
        if key != 'omega_m':
            results[key] = np.array(results[key])
    results['omega_m'] = np.array(results['omega_m'])

    return results


def mst_invariance_test(kappa_true, gamma1_true, gamma2_true, h,
                         lambda_values=None):
    """
    MST 不变性测试: 验证不同 λ 下的重建一致性

    对一系列 λ 值:
    1. 生成变换后的剪切 γ_λ = λγ
    2. 重建 κ_recon(λ)
    3. 检查: κ_recon(λ) ≈ λκ_true + const

    Parameters
    ----------
    kappa_true : ndarray
    gamma1_true, gamma2_true : ndarray
    h : float
    lambda_values : ndarray

    Returns
    -------
    results : dict
    """
    if lambda_values is None:
        lambda_values = np.linspace(0.5, 2.0, 8)

    from lensing_kaiser_squires import kaiser_squires

    results = {
        'lambda': lambda_values,
        'recon_corr': [],
        'recon_bias': [],
    }

    for lam in lambda_values:
        _, g1_lam, g2_lam = mass_sheet_transform(
            kappa_true, gamma1_true, gamma2_true, lam
        )
        kappa_recon, _ = kaiser_squires(g1_lam, g2_lam, h, apply_filter=False)

        # 相关系数
        kappa_expected = lam * kappa_true
        corr = np.corrcoef(kappa_recon.flatten(), kappa_expected.flatten())[0, 1]

        # 偏差
        bias = np.mean(kappa_recon - kappa_expected)

        results['recon_corr'].append(corr)
        results['recon_bias'].append(bias)

    results['recon_corr'] = np.array(results['recon_corr'])
    results['recon_bias'] = np.array(results['recon_bias'])

    return results
