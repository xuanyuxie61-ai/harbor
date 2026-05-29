"""
butler_volmer.py
电化学反应动力学 Butler-Volmer 方程求解模块

基于 zero_muller (1431) 与 wdk (1404) 改造
用于求解氢燃料电池阴极氧还原反应 (ORR) 的 Butler-Volmer 非线性方程。

核心公式:
  Butler-Volmer 方程:
    j = j_0 * [exp( (alpha_a * n * F * eta) / (R * T) ) 
               - exp( -(alpha_c * n * F * eta) / (R * T) )]
  
  交换电流密度 (Tafel 修正):
    j_0 = j_0_ref * (C_O2 / C_O2_ref)^gamma * exp[ -(E_a / R) * (1/T - 1/T_ref) ]
  
  过电位关系:
    eta = E - E_eq - j * R_ct
  
  需要求解关于 eta 的非线性方程: f(eta) = 0
"""

import numpy as np
import cmath


# 物理常数
FARADAY = 96485.33212      # C/mol, 法拉第常数
GAS_CONSTANT = 8.314462618  # J/(mol*K), 气体常数


def butler_volmer_current(eta, j0, alpha_a, alpha_c, n, T):
    """
    计算 Butler-Volmer 电流密度 [A/m^2]。
    
    参数:
        eta: 过电位 [V]
        j0: 交换电流密度 [A/m^2]
        alpha_a: 阳极传递系数
        alpha_c: 阴极传递系数
        n: 电子转移数
        T: 温度 [K]
    
    返回:
        j: 电流密度 [A/m^2]
    """
    if T <= 0:
        raise ValueError("温度 T 必须为正")
    if j0 < 0:
        raise ValueError("交换电流密度 j0 必须非负")
    
    rt_nf = GAS_CONSTANT * T / (n * FARADAY)
    
    # 边界处理: 防止指数溢出
    arg_a = alpha_a * eta / rt_nf
    arg_c = -alpha_c * eta / rt_nf
    
    arg_a = np.clip(arg_a, -700, 700)
    arg_c = np.clip(arg_c, -700, 700)
    
    j = j0 * (np.exp(arg_a) - np.exp(arg_c))
    
    # 数值鲁棒性: 当eta接近0时使用泰勒展开
    if abs(eta) < 1e-12:
        j = j0 * (alpha_a + alpha_c) * eta / rt_nf
    
    return j


def exchange_current_density(T, C_O2, j0_ref=1e-4, C_O2_ref=1.2, 
                             gamma=0.5, E_a=73200, T_ref=298.15):
    """
    计算温度与氧气浓度修正后的交换电流密度。
    
    公式:
        j_0 = j_0_ref * (C_O2/C_O2_ref)^gamma * exp[ -E_a/R * (1/T - 1/T_ref) ]
    """
    if T <= 0 or C_O2 <= 0:
        raise ValueError("T 和 C_O2 必须为正")
    
    conc_ratio = C_O2 / C_O2_ref
    if conc_ratio <= 0:
        raise ValueError("浓度比必须为正")
    
    thermal_factor = np.exp(-(E_a / GAS_CONSTANT) * (1.0 / T - 1.0 / T_ref))
    
    j0 = j0_ref * (conc_ratio ** gamma) * thermal_factor
    
    # 边界保护
    if not np.isfinite(j0) or j0 <= 0:
        j0 = j0_ref * 1e-6  # 最小值保护
    
    return j0


def solve_overpotential_muller(E, E_eq, R_ct, j0, alpha_a, alpha_c, n, T,
                                tol=1e-12, max_iter=100):
    """
    求解 Butler-Volmer 非线性方程。
    
    方程:
        f(eta) = eta - (E - E_eq) + R_ct * j0 * [exp(alpha_a * nF*eta/RT) 
                                                  - exp(-alpha_c * nF*eta/RT)] = 0
    
    基于 zero_muller (1431) 的复数迭代思想改造，
    实际使用阻尼牛顿法 (Damped Newton-Raphson) 保证收敛鲁棒性。
    """
    rt_nf = GAS_CONSTANT * T / (n * FARADAY)
    E_cell = E - E_eq
    beta_a = alpha_a / rt_nf
    beta_c = alpha_c / rt_nf
    
    def safe_exp(x):
        """安全指数函数，防止溢出。"""
        return np.exp(np.clip(x, -350, 350))
    
    def f(eta):
        """Butler-Volmer 方程残差。"""
        e_pos = safe_exp(beta_a * eta)
        e_neg = safe_exp(-beta_c * eta)
        return eta - E_cell + R_ct * j0 * (e_pos - e_neg)
    
    def df(eta):
        """Jacobian (导数)。"""
        e_pos = safe_exp(beta_a * eta)
        e_neg = safe_exp(-beta_c * eta)
        return 1.0 + R_ct * j0 * (beta_a * e_pos + beta_c * e_neg)
    
    # 使用 Muller 法的复数初始猜测策略，但内部用牛顿法
    # 初始猜测: E_cell 本身（零阶近似）
    eta = float(E_cell)
    
    # 如果初始猜测导致 Jacobian 过小，调整
    if abs(df(eta)) < 1e-12:
        eta = 0.0
    
    # TODO(Hole_1): 实现阻尼牛顿法迭代求解过电位非线性方程
    # 方程: f(eta) = eta - E_cell + R_ct * j0 * (exp(beta_a*eta) - exp(-beta_c*eta)) = 0
    # 需要实现迭代循环、阻尼步长限制、收敛判断和边界保护
    # 提示: 初始猜测 eta = E_cell; 使用 safe_exp 防止溢出; 阻尼步长不超过 0.1 V
    raise NotImplementedError("Hole_1: 请实现 solve_overpotential_muller 的迭代求解逻辑")



def solve_overpotential_wdk(E, E_eq, R_ct, j0, alpha_a, alpha_c, n, T,
                            tol=1e-12, max_iter=200):
    """
    使用 Weierstrass-Durand-Kerner (WDK) 方法求解过电位。
    
    将 Butler-Volmer 方程在参考点附近泰勒展开为多项式，
    然后使用 WDK 方法求多项式根。
    
    基于 wdk 改造。
    """
    rt_nf = GAS_CONSTANT * T / (n * FARADAY)
    E_cell = E - E_eq
    
    # 构建泰勒展开多项式（5阶近似）
    # f(eta) = eta - E_cell + R_ct*j0*[sum_k (alpha_a^k - (-alpha_c)^k) * (eta/rt_nf)^k / k! ]
    # 简化为多项式: c0 + c1*eta + c2*eta^2 + c3*eta^3 + c4*eta^4 + c5*eta^5 = 0
    
    coeffs = np.zeros(6, dtype=complex)
    coeffs[0] = -E_cell
    coeffs[1] = 1.0 + R_ct * j0 * (alpha_a + alpha_c) / rt_nf
    coeffs[2] = R_ct * j0 * (alpha_a**2 - alpha_c**2) / (2.0 * rt_nf**2)
    coeffs[3] = R_ct * j0 * (alpha_a**3 + alpha_c**3) / (6.0 * rt_nf**3)
    coeffs[4] = R_ct * j0 * (alpha_a**4 - alpha_c**4) / (24.0 * rt_nf**4)
    coeffs[5] = R_ct * j0 * (alpha_a**5 + alpha_c**5) / (120.0 * rt_nf**5)
    
    # 由于泰勒展开多项式可能不稳定，这里直接使用 Muller 法作为回退
    # WDK 在此物理问题中容易溢出，因此使用更鲁棒的复数 Muller 法
    return solve_overpotential_muller(E, E_eq, R_ct, j0, alpha_a, alpha_c, n, T)


def orr_kinetic_parameters(T=353.15):
    """
    返回 ORR 的典型动力学参数。
    
    参考值来源: Neyerlin et al., J. Electrochem. Soc. 153, A1955 (2006)
    """
    params = {
        'alpha_a': 0.5,          # 阳极传递系数
        'alpha_c': 0.5,          # 阴极传递系数
        'n': 4,                  # 电子转移数 (O2 + 4H+ + 4e- -> 2H2O)
        'j0_ref': 1e-4,          # A/m^2 (Pt表面积基准)
        'C_O2_ref': 1.2,         # mol/m^3
        'gamma': 0.5,            # 反应级数
        'E_a': 73200,            # J/mol
        'T_ref': 298.15,         # K
        'E_eq': 1.23 - 0.9e-3 * (T - 298.15),  # 温度修正的平衡电位 [V]
        'R_ct': 1e-4,            # 电荷转移电阻 [Ohm*m^2]
    }
    return params


if __name__ == "__main__":
    p = orr_kinetic_parameters()
    eta_m = solve_overpotential_muller(0.7, p['E_eq'], p['R_ct'], p['j0_ref'],
                                       p['alpha_a'], p['alpha_c'], p['n'], 353.15)
    eta_w = solve_overpotential_wdk(0.7, p['E_eq'], p['R_ct'], p['j0_ref'],
                                    p['alpha_a'], p['alpha_c'], p['n'], 353.15)
    print(f"Muller 法过电位: {eta_m:.6f} V")
    print(f"WDK 法过电位: {eta_w:.6f} V")
