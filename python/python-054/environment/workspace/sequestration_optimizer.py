"""
sequestration_optimizer.py
================================================================================
海洋碳封存效率优化 — 黄金分割搜索

融合项目：
    - 834_opt_golden : 黄金分割搜索一维优化

核心科学问题：
    确定海洋碳封存的 optimal 深度，使得：
    (1) CO₂ 水合物稳定性最大化
    (2) 再暴露到表层的时间尺度最大化
    (3) 对海洋酸化的局部影响最小化

    将问题简化为一个单峰目标函数：给定注入深度 z，计算综合封存效率指数。

科学背景：
    海洋碳封存（Ocean Carbon Sequestration）考虑将液态 CO₂ 注入深海。
    关键物理过程：
    
    1. CO₂ 水合物稳定性：
        水合物稳定存在的条件（Sloan, 1998）：
            T < T_hydrate(P) ≈ T₀ + a·ln(P/P₀)
        深海高压低温有利于水合物形成。
    
    2. 再暴露时间尺度：
        假设一维垂向平流-扩散，示踪物从深度 z 返回表层的特征时间：
            τ(z) ≈ z² / (2·K_z)  (扩散主导)
        或
            τ(z) ≈ z / w         (平流主导)
    
    3. 局部酸化影响：
        注入点附近 pH 下降程度与局部稀释率相关。
        稀释率 ∝ K_z^(3/2) / z² （考虑三维扩散）。
    
    综合目标函数（单峰假设）：
        E(z) = w₁·hydrate_stability(z) + w₂·(1 - exp(-τ(z)/τ₀))
               - w₃·acidification_impact(z)
    
    其中 w₁+w₂+w₃ = 1。

    黄金分割搜索：
        对于单峰函数 f(x) 在 [a,b] 上，每次迭代比较两个内点：
            x₁ = g·a + (1-g)·b
            x₂ = (1-g)·a + g·b
            g = (√5 - 1) / 2 ≈ 0.618
        保留包含极小值的子区间，收敛比 = g ≈ 0.618。
================================================================================
"""

import numpy as np


# =============================================================================
# 黄金分割搜索 (来自 opt_golden)
# =============================================================================

def golden_section_search(f, a, b, n_iterations=100, x_tol=1e-6):
    """
    黄金分割搜索求单峰函数 f(x) 在 [a,b] 上的最小值。
    
    算法：
        g = (√5 - 1) / 2 ≈ 0.618
        x1 = g·a + (1-g)·b
        x2 = (1-g)·a + g·b
        
        if f(x1) < f(x2): 保留 [a, x2]
        else:               保留 [x1, b]
    
    参数:
        f           : callable, f(x) -> float
        a, b        : float, 搜索区间
        n_iterations : int, 最大迭代次数
        x_tol       : float, x 区间容差
    
    返回:
        dict: {'x_opt': 最优 x, 'f_opt': 最优值, 'iterations': 迭代次数}
    """
    g = (np.sqrt(5.0) - 1.0) / 2.0  # 黄金分割比 ≈ 0.618
    
    x1 = g * a + (1.0 - g) * b
    x2 = (1.0 - g) * a + g * b
    f1 = f(x1)
    f2 = f(x2)
    
    for it in range(n_iterations):
        if abs(b - a) <= x_tol:
            break
        
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = g * a + (1.0 - g) * b
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = (1.0 - g) * a + g * b
            f2 = f(x2)
    
    x_opt = 0.5 * (a + b)
    f_opt = f(x_opt)
    
    return {
        'x_opt': x_opt,
        'f_opt': f_opt,
        'iterations': it + 1,
        'interval': (a, b),
    }


# =============================================================================
# 海洋碳封存物理模型
# =============================================================================

def hydrate_stability_depth(T_profile_func, P_profile_func, z):
    """
    计算在给定深度 z 处的 CO₂ 水合物稳定性指标。
    
    CO₂ 水合物平衡条件（Sloan, 1998 简化）：
        log₁₀(P_eq) = 1.847 + 0.0419·T_eq   (MPa, °C)
    
    若 P(z) > P_eq(T(z))，则水合物稳定。
    
    参数:
        T_profile_func : callable, z -> T(°C)
        P_profile_func : callable, z -> P(MPa)
        z              : float, 深度 (m)
    
    返回:
        float: 稳定性指标 (0-1)，1 表示完全稳定
    """
    T = T_profile_func(z)
    P = P_profile_func(z)
    
    # CO₂ 水合物平衡压力 (MPa)
    P_eq = 10.0**(1.847 + 0.0419 * T)
    
    stability = min(1.0, max(0.0, (P - P_eq) / (P_eq + 0.1)))
    return stability


def reexposure_time_scale(z, K_z=1e-4, w=0.0):
    """
    计算从深度 z 返回表层的特征时间尺度 (年)。
    
    扩散主导：τ = z² / (2·K_z)  (秒)
    平流主导：τ = z / w         (秒，若 w > 0)
    
    参数:
        z   : float, 深度 (m)
        K_z : float, 垂向扩散系数 (m²/s)
        w   : float, 上升流速度 (m/s)
    
    返回:
        float: 时间尺度 (年)
    """
    if z <= 0:
        return 0.0
    
    tau_diff = z**2 / (2.0 * K_z)
    tau_adv = z / w if w > 1e-10 else np.inf
    
    tau = min(tau_diff, tau_adv)
    tau_years = tau / (86400.0 * 365.25)
    return tau_years


def acidification_impact(z, injection_rate, K_z=1e-4, dilution_factor=1.0):
    """
    估算注入点附近的局部酸化影响指标。
    
    假设三维高斯稀释羽流：
        C(r) = Q / (4π·K_z·r) · exp(-w·r / (2·K_z))
    
    简化为与深度相关的无量纲影响指数：
        impact ∝ injection_rate / (K_z^(3/2) · z²) · dilution_factor
    
    参数:
        z               : float, 深度 (m)
        injection_rate  : float, 注入速率 (kg/s)
        K_z             : float, 扩散系数
        dilution_factor : float, 稀释增强因子
    
    返回:
        float: 无量纲酸化影响 (0-1)
    """
    if z <= 0:
        return 1.0
    
    impact = injection_rate / (K_z**1.5 * z**2) * dilution_factor
    # 归一化到 [0,1]
    impact_norm = min(1.0, impact / (impact + 1e-3))
    return impact_norm


def sequestration_efficiency(z, T_profile_func, P_profile_func,
                              K_z=1e-4, w=0.0,
                              w1=0.4, w2=0.4, w3=0.2,
                              injection_rate=1.0, tau0=1000.0):
    """
    综合封存效率目标函数（用于优化）。
    
    E(z) = w₁·hydrate_stability(z) + w₂·(1 - exp(-τ(z)/τ₀)) - w₃·acidification(z)
    
    返回负值因为 golden_section_search 求最小值。
    """
    if z <= 0 or z > 6000:
        return 1e6  # 惩罚
    
    h = hydrate_stability_depth(T_profile_func, P_profile_func, z)
    tau = reexposure_time_scale(z, K_z, w)
    acid = acidification_impact(z, injection_rate, K_z)
    
    E = w1 * h + w2 * (1.0 - np.exp(-tau / tau0)) - w3 * acid
    return -E  # 返回负值用于最小化


def optimize_sequestration_depth(z_min=500, z_max=4000,
                                  T_profile_func=None, P_profile_func=None,
                                  K_z=1e-4, w=0.0, n_iter=100):
    """
    使用黄金分割搜索优化碳封存深度。
    
    参数:
        z_min, z_max   : float, 搜索区间 (m)
        T_profile_func : callable, z -> T(°C), 默认使用典型深海温跃层
        P_profile_func : callable, z -> P(MPa), 默认使用静水压力
        K_z            : float, 扩散系数
        w              : float, 上升流速度
        n_iter         : int, 迭代次数
    
    返回:
        dict: 优化结果
    """
    if T_profile_func is None:
        def T_profile_func(z):
            # 典型深海温度剖面
            return max(2.0, 20.0 * np.exp(-z / 200.0))
    
    if P_profile_func is None:
        def P_profile_func(z):
            # 静水压力 + 大气压 (MPa)
            rho = 1025.0
            g = 9.81
            return 0.1013 + rho * g * z * 1e-6
    
    def objective(z):
        return sequestration_efficiency(z, T_profile_func, P_profile_func, K_z, w)
    
    result = golden_section_search(objective, z_min, z_max, n_iterations=n_iter, x_tol=1.0)
    z_opt = result['x_opt']
    
    # 计算各项指标在最优深度处的值
    h_opt = hydrate_stability_depth(T_profile_func, P_profile_func, z_opt)
    tau_opt = reexposure_time_scale(z_opt, K_z, w)
    acid_opt = acidification_impact(z_opt, 1.0, K_z)
    
    return {
        'optimal_depth_m': z_opt,
        'efficiency_score': -result['f_opt'],
        'iterations': result['iterations'],
        'hydrate_stability': h_opt,
        'reexposure_time_years': tau_opt,
        'acidification_impact': acid_opt,
    }


def multi_scenario_optimization(scenarios):
    """
    对多个封存情景进行优化对比。
    
    参数:
        scenarios : list of dict, 每个包含 'name', 'K_z', 'w', 'injection_rate'
    
    返回:
        list of dict: 各情景的优化结果
    """
    results = []
    for sc in scenarios:
        res = optimize_sequestration_depth(
            K_z=sc.get('K_z', 1e-4),
            w=sc.get('w', 0.0),
            n_iter=80
        )
        res['scenario_name'] = sc.get('name', 'unnamed')
        res['injection_rate'] = sc.get('injection_rate', 1.0)
        results.append(res)
    return results
