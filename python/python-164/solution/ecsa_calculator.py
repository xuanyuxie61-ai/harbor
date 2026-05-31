"""
ecsa_calculator.py
电化学活性表面积 (ECSA) 损失计算与稳定性分析模块

基于 power_method (902) 改造
用于评估 PEM 燃料电池催化剂衰减过程中的 ECSA 演化及系统稳定性。

核心公式:
  ECSA 定义:
    ECSA = A_Pt * m_Pt / L_Pt
    
  其中 A_Pt 为单位质量 Pt 的活性表面积 [m^2/g_Pt],
        m_Pt 为催化剂层中 Pt 总质量 [g],
        L_Pt 为 Pt 负载 [g/cm^2]。
  
  单位质量活性表面积:
    A_Pt = sum_i (4 * pi * r_i^2) / (sum_i (4/3 * pi * r_i^3 * rho_Pt))
    
  ECSA 损失动力学:
    d(ECSA)/dt = -k_1 * ECSA - k_2 * ECSA^2
    
  线性化后的稳定性分析: 雅可比矩阵最大特征值决定系统稳定性。
  
  使用幂法 (Power Method) 计算衰减雅可比矩阵的主导特征值:
    lambda^{(k+1)} = y^T * A * y / (y^T * y)
    y^{(k+1)} = A * y / ||A * y||
"""

import numpy as np


def power_method_eigenvalue(A, y0=None, it_max=100, tol=1e-10):
    """
    幂法计算实矩阵最大模特征值及对应特征向量。
    
    基于 power_method (902) 改造为 Python。
    
    参数:
        A: (n, n) 实矩阵
        y0: 初始猜测向量
        it_max: 最大迭代次数
        tol: 收敛容差
    
    返回:
        lambda_max: 主导特征值
        y: 对应特征向量
        it_num: 实际迭代次数
    """
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("A 必须是方阵")
    
    if y0 is None:
        y = np.random.random(n)
    else:
        y = np.array(y0, dtype=float)
    
    # 归一化
    norm_y = np.linalg.norm(y)
    if norm_y < 1e-30:
        y = np.ones(n)
        norm_y = np.linalg.norm(y)
    y = y / norm_y
    
    it_num = 0
    lambda_old = 0.0
    
    ay = A @ y
    lambda_val = float(y @ ay)
    y = ay / np.linalg.norm(ay)
    if lambda_val < 0:
        y = -y
    
    for it_num in range(1, it_max + 1):
        lambda_old = lambda_val
        y_old = y.copy()
        
        ay = A @ y
        lambda_val = float(y @ ay)
        y = ay / np.linalg.norm(ay)
        if lambda_val < 0:
            y = -y
        
        val_dif = abs(lambda_val - lambda_old)
        
        # 特征向量方向收敛判断
        cos_yy = float(y @ y_old)
        sin_yy = np.sqrt(max(0.0, (1.0 - cos_yy) * (1.0 + cos_yy)))
        
        if val_dif <= tol and sin_yy <= tol:
            break
    
    return lambda_val, y, it_num


def ecsa_from_size_distribution(radii, rho_pt=21450):
    """
    从颗粒尺寸分布计算单位质量活性表面积 [m^2/g_Pt]。
    
    公式:
        A_Pt = sum(4*pi*r_i^2) / sum((4/3)*pi*r_i^3*rho_pt) * 1000
    """
    if len(radii) == 0:
        return 0.0
    
    radii = np.array(radii)
    surface_area = np.sum(4.0 * np.pi * radii ** 2)
    volume = np.sum((4.0 / 3.0) * np.pi * radii ** 3)
    mass = volume * rho_pt  # kg
    
    if mass < 1e-30:
        return 0.0
    
    # 转换为 m^2/g
    ecsa_specific = surface_area / (mass * 1000.0)
    return float(ecsa_specific)


def ecsa_loss_kinetics(ECSA, t, k1=1e-6, k2=1e-10):
    """
    ECSA 损失动力学方程。
    
    公式: d(ECSA)/dt = -k1 * ECSA - k2 * ECSA^2
    
    解析解 (当 ECSA0 > 0):
        ECSA(t) = k1 * ECSA0 / [ (k1 + k2*ECSA0) * exp(k1*t) - k2*ECSA0 ]
    """
    if t < 0 or ECSA <= 0:
        return 0.0
    
    ECSA0 = ECSA
    denom = (k1 + k2 * ECSA0) * np.exp(k1 * t) - k2 * ECSA0
    
    if abs(denom) < 1e-30:
        return 0.0
    
    ecsa_t = k1 * ECSA0 / denom
    return float(max(ecsa_t, 0.0))


def build_stability_jacobian(n_species, rate_constants, interaction_matrix):
    """
    构建催化剂衰减系统的稳定性雅可比矩阵。
    
    参数:
        n_species: 物种/状态数
        rate_constants: 衰减速率常数列表
        interaction_matrix: 物种间相互作用矩阵
    
    返回:
        J: (n_species, n_species) 雅可比矩阵
    """
    if n_species <= 0:
        raise ValueError("物种数必须为正")
    
    J = np.zeros((n_species, n_species))
    
    for i in range(n_species):
        J[i, i] = -rate_constants[i] if i < len(rate_constants) else -1e-6
        for j in range(n_species):
            if i != j:
                J[i, j] = interaction_matrix[i, j] if interaction_matrix is not None else 0.0
    
    return J


def stability_analysis_max_eigenvalue(J, y0=None):
    """
    使用幂法分析系统稳定性。
    
    主导特征值的物理意义:
      - lambda_max < 0: 系统渐近稳定 (衰减最终会停止或达到稳态)
      - lambda_max > 0: 系统不稳定 (衰减会加速)
      - lambda_max = 0: 临界稳定
    
    返回:
        lambda_max: 主导特征值
        stability: 'stable', 'unstable', 'critical'
    """
    if J.shape[0] != J.shape[1]:
        raise ValueError("雅可比矩阵必须是方阵")
    
    lambda_max, _, _ = power_method_eigenvalue(J, y0=y0, it_max=200, tol=1e-12)
    
    if lambda_max < -1e-10:
        stability = 'stable'
    elif lambda_max > 1e-10:
        stability = 'unstable'
    else:
        stability = 'critical'
    
    return lambda_max, stability


def voltage_loss_from_ecsa(ECSA_ratio, b_tafel=0.06):
    """
    由 ECSA 损失引起的电压损失 [V]。
    
    公式:
        delta_V = b * log10(ECSA0 / ECSA) = -b * log10(ECSA_ratio)
    
    其中 b 为 Tafel 斜率 [V/decade]。
    """
    if ECSA_ratio <= 0:
        return 0.5  # 极大损失
    
    ratio = 1.0 / ECSA_ratio
    ratio = np.clip(ratio, 1.0, 1e6)
    
    dV = b_tafel * np.log10(ratio)
    return float(dV)


def total_ecsa_loss_model(t_hours, ECSA0, params=None):
    """
    综合 ECSA 损失模型。
    
    综合考虑:
      - 溶解-熟化损失 (t^1/3 律)
      - 碳腐蚀导致的 Pt 脱离 (指数衰减)
      - 毒化效应 (线性衰减)
    """
    if params is None:
        params = {
            'k_ripening': 0.05,     # 1/h^(1/3)
            'k_corrosion': 1e-4,    # 1/h
            'k_poisoning': 1e-5,    # 1/h
        }
    
    t = max(t_hours, 0.0)
    
    # 溶解熟化: ECSA ~ ECSA0 / (1 + k_r * t^(1/3))
    ripening_factor = 1.0 / (1.0 + params['k_ripening'] * (t ** (1.0 / 3.0)))
    
    # 碳腐蚀: 指数衰减
    corrosion_factor = np.exp(-params['k_corrosion'] * t)
    
    # 毒化: 线性衰减
    poisoning_factor = max(0.0, 1.0 - params['k_poisoning'] * t)
    
    ECSA_t = ECSA0 * ripening_factor * corrosion_factor * poisoning_factor
    
    return float(max(ECSA_t, 0.0))


if __name__ == "__main__":
    radii = np.array([2e-9, 3e-9, 4e-9, 5e-9])
    ecsa = ecsa_from_size_distribution(radii)
    print(f"比表面积: {ecsa:.2f} m^2/g_Pt")
    
    J = build_stability_jacobian(3, [1e-4, 2e-4, 5e-5], 
                                  np.array([[-1e-4, 1e-5, 0],
                                            [2e-5, -2e-4, 1e-5],
                                            [0, 3e-5, -5e-5]]))
    lam, stab = stability_analysis_max_eigenvalue(J)
    print(f"主导特征值: {lam:.6e}, 稳定性: {stab}")
