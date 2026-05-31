"""
catalyst_optimizer.py
催化剂负载分布优化模块

基于 opt_golden (834) 改造
用于优化 PEM 燃料电池阴极催化剂层中的 Pt 负载分布，
以在成本和性能之间取得最优平衡。

核心公式:
  目标泛函 (最小化总成本):
    J(L_pt) = w_1 * C_Pt(L_pt) + w_2 * P_loss(L_pt) + w_3 * D_penalty(L_pt)
  
  其中:
    C_Pt(L_pt) = m_Pt * Price_Pt  (Pt 材料成本)
    P_loss(L_pt) = P_max - P(L_pt)  (功率损失)
    D_penalty(L_pt) = max(0, L_pt - L_max)^2  (过量负载惩罚)
  
  性能-负载经验关系 (基于经验数据拟合):
    P(L_pt) = P_max * [1 - exp(-k * L_pt / L_ref)]
  
  使用黄金分割搜索 (Golden Section Search) 求解:
    L_pt^* = argmin J(L_pt),  L_pt in [L_min, L_max]
  
  黄金分割比例:
    phi = (sqrt(5) - 1) / 2  approx 0.618
"""

import numpy as np


GOLDEN_RATIO = (np.sqrt(5.0) - 1.0) / 2.0


def golden_section_search(f, a, b, n_max=100, x_tol=1e-8):
    """
    黄金分割搜索法求解单峰函数在 [a,b] 上的最小值。
    
    基于 opt_golden (834) 改造为 Python。
    
    参数:
        f: 目标函数
        a, b: 搜索区间端点
        n_max: 最大迭代次数
        x_tol: x 区间容差
    
    返回:
        a, b: 包含最小值的最终区间
        it: 实际迭代次数
        x_opt: 最优解估计
        f_opt: 最优值
    """
    if a >= b:
        raise ValueError("必须满足 a < b")
    if n_max < 1:
        raise ValueError("n_max 必须 >= 1")
    if x_tol <= 0:
        raise ValueError("x_tol 必须为正")
    
    x1 = GOLDEN_RATIO * a + (1.0 - GOLDEN_RATIO) * b
    x2 = (1.0 - GOLDEN_RATIO) * a + GOLDEN_RATIO * b
    
    f1 = f(x1)
    f2 = f(x2)
    
    it = 0
    for it in range(1, n_max + 1):
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = GOLDEN_RATIO * a + (1.0 - GOLDEN_RATIO) * b
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = (1.0 - GOLDEN_RATIO) * a + GOLDEN_RATIO * b
            f2 = f(x2)
        
        if abs(b - a) <= x_tol:
            break
    
    x_opt = 0.5 * (a + b)
    f_opt = f(x_opt)
    
    return a, b, it, x_opt, f_opt


def power_performance(L_pt, P_max=1.0, k_eff=5.0, L_ref=0.1):
    """
    催化剂负载与电池性能的半经验关系。
    
    公式:
        P(L_pt) = P_max * [1 - exp(-k_eff * L_pt / L_ref)]
    
    参数:
        L_pt: Pt 负载 [mg/cm^2]
        P_max: 最大归一化功率
        k_eff: 效率因子
        L_ref: 参考负载 [mg/cm^2]
    """
    if L_pt < 0:
        return 0.0
    
    exponent = -k_eff * L_pt / max(L_ref, 1e-10)
    exponent = np.clip(exponent, -50, 0)
    
    P = P_max * (1.0 - np.exp(exponent))
    return float(P)


def catalyst_cost(L_pt, area_active=250.0, price_pt=50.0):
    """
    计算 Pt 催化剂成本 [USD]。
    
    参数:
        L_pt: Pt 负载 [mg/cm^2]
        area_active: 活性面积 [cm^2]
        price_pt: Pt 单价 [USD/g]
    """
    if L_pt < 0 or area_active < 0:
        return 0.0
    
    # L_pt [mg/cm^2] * area [cm^2] = mg
    mass_mg = L_pt * area_active
    mass_g = mass_mg / 1000.0
    cost = mass_g * price_pt
    
    return float(cost)


def objective_function(L_pt, w_cost=0.3, w_power=0.5, w_penalty=0.2,
                        P_max=1.0, k_eff=5.0, L_ref=0.1,
                        area_active=250.0, price_pt=50.0, L_max=0.5):
    """
    催化剂负载优化的目标函数。
    
    J(L_pt) = w_c * C_Pt(L_pt)/C_ref + w_p * (1 - P(L_pt)/P_max) + w_d * D(L_pt)
    """
    if L_pt < 0:
        return 1e10
    
    # 归一化成本 (参考: 0.4 mg/cm^2 时的成本)
    C_ref = catalyst_cost(0.4, area_active, price_pt)
    C_pt = catalyst_cost(L_pt, area_active, price_pt)
    cost_term = C_pt / max(C_ref, 1e-10)
    
    # 功率损失
    P_val = power_performance(L_pt, P_max, k_eff, L_ref)
    power_term = 1.0 - P_val / max(P_max, 1e-10)
    
    # 过量负载惩罚
    penalty_term = max(0.0, L_pt - L_max) ** 2 * 100.0
    
    J = w_cost * cost_term + w_power * power_term + w_penalty * penalty_term
    
    return float(J)


def optimize_catalyst_loading(w_cost=0.3, w_power=0.5, w_penalty=0.2,
                               L_min=0.02, L_max=0.8):
    """
    优化催化剂负载分布。
    
    返回:
        L_opt: 最优 Pt 负载 [mg/cm^2]
        J_opt: 最优目标值
        info: 优化信息字典
    """
    def f(L):
        return objective_function(L, w_cost, w_power, w_penalty)
    
    a, b, it, L_opt, J_opt = golden_section_search(f, L_min, L_max, 
                                                     n_max=200, x_tol=1e-8)
    
    # 数值边界保护
    L_opt = np.clip(L_opt, L_min, L_max)
    
    info = {
        'interval': (a, b),
        'iterations': it,
        'power_at_opt': power_performance(L_opt),
        'cost_at_opt': catalyst_cost(L_opt),
        'objective_value': J_opt
    }
    
    return L_opt, J_opt, info


def sensitivity_analysis(L_opt, delta_L=0.01):
    """
    在最优解附近进行敏感性分析。
    
    计算目标函数对负载的局部敏感性 dJ/dL。
    """
    J_plus = objective_function(L_opt + delta_L)
    J_minus = objective_function(L_opt - delta_L)
    
    sensitivity = (J_plus - J_minus) / (2.0 * delta_L)
    
    return sensitivity


if __name__ == "__main__":
    L_opt, J_opt, info = optimize_catalyst_loading()
    print(f"最优 Pt 负载: {L_opt:.4f} mg/cm^2")
    print(f"最优目标值: {J_opt:.6f}")
    print(f"迭代次数: {info['iterations']}")
    sens = sensitivity_analysis(L_opt)
    print(f"敏感性: {sens:.6f}")
