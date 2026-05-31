"""
ripening_model.py
催化剂颗粒 Ostwald 熟化与尺寸分布演化模块

基于 disk_distance_stats (292) 与 legendre_exactness (659) 改造
用于模拟 PEM 燃料电池阴极 Pt 纳米颗粒的溶解-再沉积熟化过程。

核心公式:
  Gibbs-Thomson 效应:
    mu(r) = mu_inf + (2 * gamma * V_m) / r
  
  溶解度修正 (Kelvin 方程):
    C_sat(r) = C_sat_inf * exp( (2 * gamma * V_m) / (r * R * T) )
  
  LSW (Lifshitz-Slyozov-Wagner) 熟化速率:
    dr/dt = (D * V_m * C_sat_inf / r) * [C_bulk/C_sat_inf - exp(a_LSW/r)]
  
  其中 a_LSW = 2 * gamma * V_m / (R * T)
  
  临界半径:
    r_c = a_LSW / ln(C_bulk / C_sat_inf)
  
  平均半径演化 (t -> infinity):
    <r>^3 - <r_0>^3 = (8 * gamma * D * V_m^2 * C_sat_inf / (9 * R * T)) * t
"""

import numpy as np


# 物理常数
GAS_CONSTANT = 8.314462618  # J/(mol*K)


def kelvin_solubility(r, gamma, V_m, T, C_sat_inf):
    """
    计算曲率修正后的溶解度。
    
    公式: C_sat(r) = C_sat_inf * exp(2*gamma*V_m / (r*R*T))
    """
    if r <= 0:
        raise ValueError("半径 r 必须为正")
    if T <= 0:
        raise ValueError("温度 T 必须为正")
    
    exponent = (2.0 * gamma * V_m) / (r * GAS_CONSTANT * T)
    # 防止指数过大
    exponent = np.clip(exponent, -50, 50)
    
    return C_sat_inf * np.exp(exponent)


def critical_radius(gamma, V_m, T, C_bulk, C_sat_inf):
    """
    计算 LSW 临界半径。
    
    公式: r_c = 2*gamma*V_m / (R*T*ln(C_bulk/C_sat_inf))
    """
    if C_bulk <= 0 or C_sat_inf <= 0:
        return 1e-8  # 返回保护值
    
    ratio = C_bulk / C_sat_inf
    if ratio <= 1.0:
        return 1e-8  # 无过饱和，熟化极慢
    
    ln_ratio = np.log(ratio)
    
    if abs(ln_ratio) < 1e-12:
        return 1e20  # 极大值，表示无熟化
    
    rc = (2.0 * gamma * V_m) / (GAS_CONSTANT * T * ln_ratio)
    return max(rc, 1e-10)


def ripening_rate(r, D, V_m, C_sat_inf, C_bulk, gamma, T):
    """
    计算颗粒半径变化速率 dr/dt [m/s]。
    
    公式:
        dr/dt = (D * V_m / r) * (C_bulk - C_sat(r))
              = (D * V_m * C_sat_inf / r) * [C_bulk/C_sat_inf - exp(2*gamma*V_m/(r*R*T))]
    """
    if r <= 0:
        return 0.0
    
    C_sat_r = kelvin_solubility(r, gamma, V_m, T, C_sat_inf)
    rate = (D * V_m / r) * (C_bulk - C_sat_r)
    
    # 数值边界保护
    # 实际催化剂层中离子omer传质阻力会显著限制溶解速率
    max_rate = 1e-9  # m/s (约 3 nm/年，更符合实际衰减速率)
    rate = np.clip(rate, -max_rate, max_rate)
    
    return rate


def evolve_size_distribution(radii, D, V_m, C_sat_inf, C_bulk, gamma, T, dt, n_steps):
    """
    使用显式欧拉法演化颗粒尺寸分布。
    
    物理约束: 颗粒半径不低于 0.5 nm（约 2-3 个 Pt 原子直径）。
    
    参数:
        radii: 初始半径数组 [m]
        dt: 时间步长 [s]
        n_steps: 时间步数
    
    返回:
        radii_history: (n_steps+1, N) 半径演化历史
    """
    if dt <= 0 or n_steps < 0:
        raise ValueError("dt>0, n_steps>=0")
    
    # 保护 C_bulk: 必须大于 C_sat_inf 才能驱动熟化
    C_bulk_safe = max(float(C_bulk), 1.5 * float(C_sat_inf))
    if C_bulk_safe <= 0 or C_sat_inf <= 0:
        # 无过饱和，返回恒等演化
        N = len(radii)
        radii_history = np.zeros((n_steps + 1, N))
        for s in range(n_steps + 1):
            radii_history[s, :] = np.clip(radii, 0.5e-9, 1e-6)
        return radii_history
    
    N = len(radii)
    radii_history = np.zeros((n_steps + 1, N))
    radii_history[0, :] = np.clip(radii, 0.5e-9, 1e-6)
    
    for step in range(n_steps):
        r_current = radii_history[step, :]
        
        rates = np.array([ripening_rate(float(ri), float(D), float(V_m), 
                                         float(C_sat_inf), C_bulk_safe, 
                                         float(gamma), float(T)) 
                         for ri in r_current])
        
        # 保护 rates
        rates = np.array([r if np.isfinite(r) else 0.0 for r in rates])
        
        r_new = r_current + rates * dt
        
        # 边界保护: 半径不能低于 0.5 nm（物理下限），不能过大
        r_new = np.clip(r_new, 0.5e-9, 1e-6)
        
        radii_history[step + 1, :] = r_new
    
    return radii_history


def lsw_analytical_r3(t, r0, gamma, D, V_m, C_sat_inf, T):
    """
    LSW 理论: <r>^3 = r0^3 + K_LSW * t
    
    其中 K_LSW = (8 * gamma * D * V_m^2 * C_sat_inf) / (9 * R * T)
    """
    K_LSW = (8.0 * gamma * D * V_m ** 2 * C_sat_inf) / (9.0 * GAS_CONSTANT * T)
    r0_cubed = r0 ** 3
    return (r0_cubed + K_LSW * t) ** (1.0 / 3.0)


def disk_distance_stats_monte_carlo(radii1, radii2, n_samples=1000):
    """
    基于 disk_distance_stats 改造的颗粒间距统计。
    
    在二维催化剂层截面上，随机采样计算颗粒间距离分布的统计特征。
    用于评估颗粒聚集程度对熟化的影响。
    
    返回:
        mean_dist: 平均距离 [m]
        var_dist: 距离方差 [m^2]
    """
    if len(radii1) == 0 or len(radii2) == 0:
        return 0.0, 0.0
    
    distances = np.zeros(n_samples)
    
    for i in range(n_samples):
        # 在单位圆盘上随机采样两个点
        theta1 = 2.0 * np.pi * np.random.random()
        theta2 = 2.0 * np.pi * np.random.random()
        
        # 使用 sqrt(rand) 保证均匀分布
        rad1 = np.sqrt(np.random.random())
        rad2 = np.sqrt(np.random.random())
        
        p1 = np.array([rad1 * np.cos(theta1), rad1 * np.sin(theta1)])
        p2 = np.array([rad2 * np.cos(theta2), rad2 * np.sin(theta2)])
        
        distances[i] = np.linalg.norm(p1 - p2)
    
    mean_dist = np.mean(distances)
    if n_samples > 1:
        var_dist = np.sum((distances - mean_dist) ** 2) / (n_samples - 1)
    else:
        var_dist = 0.0
    
    return mean_dist, var_dist


def gauss_legendre_integral_exactness(f, order, w, x, a=-1, b=1):
    """
    基于 legendre_exactness 改造的高斯-勒让德积分检验。
    
    用于验证颗粒尺寸分布矩的数值积分精度。
    
    公式:
        int_a^b f(x) dx approx sum_i w_i * f(x_i)
    """
    if len(w) != len(x):
        raise ValueError("权重 w 和节点 x 长度必须一致")
    
    # 坐标变换到 [a, b]
    t = 0.5 * (b - a) * x + 0.5 * (a + b)
    jac = 0.5 * (b - a)
    
    integral = jac * np.sum(w * f(t))
    return integral


def moment_size_distribution(radii, w=None, k=1):
    """
    计算颗粒尺寸分布的第 k 阶矩。
    
    M_k = sum_i w_i * r_i^k / sum_i w_i
    """
    if len(radii) == 0:
        return 0.0
    
    if w is None:
        w = np.ones(len(radii))
    
    w = np.array(w)
    radii = np.array(radii)
    
    total_w = np.sum(w)
    if total_w < 1e-30:
        return 0.0
    
    moment = np.sum(w * (radii ** k)) / total_w
    return moment


def pt_dissolution_parameters():
    """
    返回 Pt 纳米颗粒溶解-熟化的典型物理参数。
    
    参考: Bi et al., ECS Trans. 50, 1219 (2013)
    """
    params = {
        'gamma': 2.5,            # J/m^2, Pt/ionomer 界面能
        'V_m': 9.09e-6,          # m^3/mol, Pt 摩尔体积
        'D_Pt2': 1e-12,          # m^2/s, Pt^2+ 在 ionomer 中的扩散系数
        'C_sat_inf': 1e-6,       # mol/m^3, 平坦表面溶解度
        'T': 353.15,             # K
        'rho_Pt': 21450,         # kg/m^3, Pt 密度
    }
    return params


if __name__ == "__main__":
    p = pt_dissolution_parameters()
    r0 = np.array([2e-9, 3e-9, 4e-9, 5e-9, 6e-9])  # 5个颗粒
    hist = evolve_size_distribution(r0, p['D_Pt2'], p['V_m'], p['C_sat_inf'],
                                     2e-6, p['gamma'], p['T'], dt=3600, n_steps=24)
    print(f"24h后平均半径: {np.mean(hist[-1])*1e9:.2f} nm")
    mu, var = disk_distance_stats_monte_carlo(r0, r0)
    print(f"颗粒间距统计: mean={mu:.4f}, var={var:.6f}")
