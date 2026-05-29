"""
uncertainty_analysis.py
================================================================================
海洋碳循环模型的不确定性量化与参数敏感性分析

融合项目：
    - 565_hypersphere_integrals : 超球面精确积分与均匀采样

核心科学问题：
    海洋碳循环模型包含大量参数（气体交换速率、生物泵效率、扩散系数、
    平衡常数等），这些参数存在观测不确定性。通过在高维参数空间球面上
    进行 Monte Carlo 采样，量化参数不确定性对模型输出（如 pH、碳库存）
    的影响。

科学背景：
    参数空间：设模型有 m 个独立参数 θ = (θ₁, ..., θₘ)。
    
    不确定性传播：
        对每个参数 θᵢ，给定基准值 θᵢ⁰ 和标准差 σᵢ。
        在标准化空间 ξᵢ = (θᵢ - θᵢ⁰) / σᵢ 中，考虑单位球面 S^{m-1} 上的
        扰动方向：
            θ = θ⁰ + r · ξ,   ||ξ|| = 1
    
    模型输出统计量：
        μ_Y = (1/N) Σ Y(θᵢ)
        σ_Y² = (1/N) Σ (Y(θᵢ) - μ_Y)²
    
    Sobol 一阶敏感性指数近似：
        Sᵢ ≈ Var(E[Y|θᵢ]) / Var(Y)
    
    超球面采样保证各方向均匀覆盖，避免网格采样在高维的"维度灾难"。

================================================================================
"""

import numpy as np


# =============================================================================
# 超球面采样与积分 (来自 hypersphere_integrals)
# =============================================================================

def hypersphere01_area(m):
    """
    单位超球面 S^{m-1} 的表面积。
    
    A_{m-1} = 2·π^(m/2) / Γ(m/2)
    
    参数:
        m : int, 空间维度 (m ≥ 1)
    """
    from math import gamma, pi
    if m < 1:
        raise ValueError("维度 m 必须 ≥ 1")
    return 2.0 * pi**(m / 2.0) / gamma(m / 2.0)


def hypersphere01_sample(m, n, seed=None):
    """
    在单位超球面 S^{m-1} 上均匀采样 n 个点。
    
    算法（Muller 方法）：
        1. 生成 m×n 标准正态随机变量 X ~ N(0,1)
        2. 归一化：ω = X / ||X||
    
    参数:
        m    : int, 维度
        n    : int, 采样数
        seed : int, 随机种子
    
    返回:
        ndarray, shape (m, n)
    """
    if seed is not None:
        np.random.seed(seed)
    x = np.random.randn(m, n)
    norms = np.linalg.norm(x, axis=0)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return x / norms


def hypersphere_monomial_integral(m, exponents):
    """
    计算单项式在超球面上的精确积分。
    
    ∫_{S^{m-1}} ∏ xᵢ^{eᵢ} dS = 
        0,                           若任一 eᵢ 为奇数
        2·∏Γ((eᵢ+1)/2) / Γ(½·Σ(eᵢ+1)),  否则
    
    参数:
        m          : int, 维度
        exponents  : array-like, 各变量幂次
    """
    from math import gamma
    exponents = np.asarray(exponents)
    
    if np.any(exponents % 2 != 0):
        return 0.0
    
    numerator = 1.0
    for e in exponents:
        numerator *= gamma((e + 1.0) / 2.0)
    
    denominator = gamma(0.5 * np.sum(exponents + 1.0))
    return 2.0 * numerator / denominator


# =============================================================================
# 参数不确定性传播
# =============================================================================

def propagate_uncertainty(model_func, base_params, param_scales,
                          n_samples=1000, seed=None):
    """
    通过超球面采样进行不确定性传播。
    
    参数:
        model_func   : callable, params -> float, 模型函数
        base_params  : ndarray, shape (m,), 基准参数
        param_scales : ndarray, shape (m,), 各参数扰动幅度
        n_samples    : int, Monte Carlo 采样数
        seed         : int
    
    返回:
        dict: {'mean', 'std', 'min', 'max', 'samples', 'params_samples'}
    """
    m = len(base_params)
    samples_dir = hypersphere01_sample(m, n_samples, seed)
    
    outputs = []
    params_samples = []
    
    for i in range(n_samples):
        perturbed = base_params + param_scales * samples_dir[:, i]
        # 保证参数物理上合理（正数）
        perturbed = np.maximum(perturbed, 1e-8)
        params_samples.append(perturbed)
        
        try:
            val = model_func(perturbed)
        except Exception:
            val = np.nan
        outputs.append(val)
    
    outputs = np.array(outputs)
    valid = outputs[~np.isnan(outputs)]
    
    return {
        'mean': np.mean(valid) if len(valid) > 0 else np.nan,
        'std': np.std(valid) if len(valid) > 0 else np.nan,
        'min': np.min(valid) if len(valid) > 0 else np.nan,
        'max': np.max(valid) if len(valid) > 0 else np.nan,
        'cv': np.std(valid) / np.mean(valid) if len(valid) > 0 and np.mean(valid) != 0 else np.nan,
        'samples': outputs,
        'params_samples': np.array(params_samples),
    }


def sobol_first_order_index(model_func, base_params, param_scales, param_index,
                            n_samples=1000, seed=None):
    """
    近似计算 Sobol 一阶敏感性指数。
    
    Sᵢ = Var(E[Y|θᵢ]) / Var(Y)
    
    使用超球面采样近似：
        固定 θᵢ 在其基准值，在其余参数的球面上采样，
        计算条件方差。重复对多个 θᵢ 值进行。
    
    参数:
        model_func   : callable
        base_params  : ndarray
        param_scales : ndarray
        param_index  : int, 要分析的参数索引
        n_samples    : int
    
    返回:
        float: Sᵢ 估计值
    """
    m = len(base_params)
    
    # 无条件方差
    result_total = propagate_uncertainty(model_func, base_params, param_scales,
                                         n_samples=n_samples, seed=seed)
    var_total = result_total['std']**2
    
    if var_total < 1e-15 or np.isnan(var_total):
        return 0.0
    
    # 条件方差：固定 param_index
    def conditional_model(other_params):
        full_params = base_params.copy()
        # other_params 是 m-1 维，需要插入固定值
        idx = 0
        for i in range(m):
            if i != param_index:
                full_params[i] = other_params[idx]
                idx += 1
        return model_func(full_params)
    
    # 在 m-1 维球面上采样
    other_base = np.delete(base_params, param_index)
    other_scales = np.delete(param_scales, param_index)
    
    result_cond = propagate_uncertainty(conditional_model, other_base, other_scales,
                                        n_samples=n_samples, seed=(seed+1 if seed else None))
    var_cond = result_cond['std']**2
    
    # Sᵢ ≈ 1 - Var_cond / Var_total
    S_i = 1.0 - var_cond / var_total
    S_i = max(0.0, min(1.0, S_i))
    return S_i


def full_sensitivity_analysis(model_func, base_params, param_names, param_scales,
                               n_samples=500, seed=None):
    """
    对所有参数进行全局敏感性分析。
    
    返回:
        dict: {'sobol_indices': {name: S_i}, 'uncertainty': 总体不确定度结果}
    """
    m = len(base_params)
    sobol = {}
    
    for i in range(m):
        S_i = sobol_first_order_index(model_func, base_params, param_scales, i,
                                      n_samples=n_samples, seed=seed)
        name = param_names[i] if i < len(param_names) else f"param_{i}"
        sobol[name] = S_i
    
    uncertainty = propagate_uncertainty(model_func, base_params, param_scales,
                                        n_samples=n_samples, seed=seed)
    
    return {
        'sobol_indices': sobol,
        'uncertainty': uncertainty,
    }


# =============================================================================
# 碳循环模型特定的不确定性分析
# =============================================================================

def carbon_cycle_uncertainty_analysis(DIC_surf, TA_surf, T, S,
                                       n_samples=500, seed=None):
    """
    分析碳酸盐化学计算中参数不确定性对 pH 和饱和度的影响。
    
    不确定参数：
        θ = [DIC, TA, T, S, K1_scale, K2_scale]
    
    基准值和相对不确定性：
        DIC: 2000 μmol/kg ± 2%
        TA:  2300 μmol/kg ± 1.5%
        T:   15°C ± 10%
        S:   35 psu ± 1%
        K1, K2: ± 5% (平衡常数的不确定性)
    """
    from carbonate_chemistry import solve_carbonate_system
    
    base_params = np.array([DIC_surf, TA_surf, T, S, 1.0, 1.0])
    param_scales = np.array([
        DIC_surf * 0.02,
        TA_surf * 0.015,
        1.5,
        0.35,
        0.05,
        0.05
    ])
    param_names = ['DIC', 'TA', 'T', 'S', 'K1_scale', 'K2_scale']
    
    def model_pH(params):
        D, A, Tp, Sp, k1s, k2s = params
        try:
            res = solve_carbonate_system(D * 1e-6, A * 1e-6, Tp, Sp)
            # 应用 K 的缩放
            res['pH'] += np.log10(k1s) * 0.1  # 近似
            return res['pH']
        except Exception:
            return np.nan
    
    def model_omega(params):
        D, A, Tp, Sp, k1s, k2s = params
        try:
            res = solve_carbonate_system(D * 1e-6, A * 1e-6, Tp, Sp)
            return res['Omega_aragonite']
        except Exception:
            return np.nan
    
    result_pH = propagate_uncertainty(model_pH, base_params, param_scales,
                                      n_samples=n_samples, seed=seed)
    result_omega = propagate_uncertainty(model_omega, base_params, param_scales,
                                         n_samples=n_samples, seed=(seed+1 if seed else None))
    
    # 计算各参数对 pH 的 Sobol 指数
    sobol_pH = {}
    for i in range(len(param_names)):
        S_i = sobol_first_order_index(model_pH, base_params, param_scales, i,
                                      n_samples=min(n_samples, 200), seed=seed)
        sobol_pH[param_names[i]] = S_i
    
    return {
        'pH_uncertainty': result_pH,
        'omega_uncertainty': result_omega,
        'sobol_pH': sobol_pH,
        'param_names': param_names,
    }
