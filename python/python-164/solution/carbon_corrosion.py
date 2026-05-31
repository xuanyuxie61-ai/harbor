"""
carbon_corrosion.py
碳载体腐蚀动力学与前沿传播模块

基于 burgers_time_inviscid (126) 改造
用于模拟 PEM 燃料电池高电位运行下碳载体腐蚀引起的催化剂层结构退化。

核心公式:
  碳腐蚀电化学反应:
    C + 2H2O -> CO2 + 4H+ + 4e-
    E^0 = 0.207 V vs. RHE (标准电位)
  
  腐蚀电流密度 (Butler-Volmer 简化):
    j_corr = j_0_corr * exp( (alpha_corr * F * eta_corr) / (R * T) )
  
  碳质量损失率:
    dm_c/dt = - (M_c / (n_e * F)) * j_corr * A_c
  
  结构退化传播方程 (守恒律形式):
    dS_c/dt + d/dx( v_c * S_c ) = -k_corr * S_c * theta_pore
  
  其中 S_c 为碳比表面积, v_c 为腐蚀前沿速度,
  theta_pore 为孔隙率相关因子。
  
  该方程在形式上等价于一维守恒律:
    du/dt + df/dx = s(u)
  
  可采用 Godunov / Lax-Wendroff / MacCormack 格式求解。
"""

import numpy as np


FARADAY = 96485.33212
GAS_CONSTANT = 8.314462618
M_C = 12.011e-3  # kg/mol, 碳摩尔质量


def corrosion_current_density(E, E_corr_0=0.207, j0_corr=1e-8, 
                               alpha_corr=0.5, T=353.15):
    """
    计算碳腐蚀电流密度 [A/m^2]。
    
    简化 Tafel 关系:
        j_corr = j0_corr * exp( alpha_corr * F * (E - E_corr_0) / (R * T) )
    """
    eta_corr = E - E_corr_0
    
    if eta_corr <= 0:
        return 0.0
    
    exponent = alpha_corr * FARADAY * eta_corr / (GAS_CONSTANT * T)
    exponent = np.clip(exponent, -50, 50)
    
    j_corr = j0_corr * np.exp(exponent)
    
    # 边界保护
    max_j = 1e4  # A/m^2
    return float(np.clip(j_corr, 0.0, max_j))


def carbon_mass_loss_rate(j_corr, A_carbon):
    """
    计算碳质量损失速率 [kg/s]。
    
    公式: dm/dt = - M_c * j_corr * A_carbon / (4 * F)
    """
    if j_corr < 0 or A_carbon < 0:
        return 0.0
    
    rate = -M_C * j_corr * A_carbon / (4.0 * FARADAY)
    return rate


def numerical_flux_godunov(u_left, u_right, v=0.0):
    """
    Godunov 数值通量。
    
    对于线性对流方程 u_t + v * u_x = 0:
        F(uL, uR) = v * uL  (v > 0)
        F(uL, uR) = v * uR  (v < 0)
    """
    if v >= 0:
        return v * u_left
    else:
        return v * u_right


def solve_corrosion_propagation(u0, nx, nt, dx, dt, v_front, k_corr, theta_pore,
                                 method='godunov'):
    """
    求解碳腐蚀结构退化传播方程。
    
    方程: du/dt + v * du/dx = -k_corr * u * theta
    
    基于 burgers_time_inviscid 中的守恒律格式改造。
    
    参数:
        u0: 初始碳比表面积分布
        nx: 空间网格数
        nt: 时间步数
        dx: 空间步长 [m]
        dt: 时间步长 [s]
        v_front: 腐蚀前沿速度 [m/s]
        k_corr: 腐蚀速率常数 [1/s]
        theta_pore: 孔隙率因子
        method: 'godunov', 'lax_wendroff', 'maccormack'
    
    返回:
        U: (nt+1, nx) 演化历史
    """
    if nx < 2 or nt < 0 or dx <= 0 or dt <= 0:
        raise ValueError("网格参数无效")
    
    # CFL 条件检查
    cfl = abs(v_front) * dt / dx
    if cfl > 1.0:
        # 自动调整时间步长
        dt = 0.9 * dx / max(abs(v_front), 1e-10)
        print(f"警告: CFL={cfl:.2f}>1, 已自动调整 dt={dt:.4e}")
    
    U = np.zeros((nt + 1, nx))
    u = np.array(u0, dtype=float)
    
    if len(u) != nx:
        raise ValueError("u0 长度必须与 nx 一致")
    
    U[0, :] = u
    
    for n_step in range(nt):
        unew = np.zeros(nx)
        
        if method == 'godunov':
            # Godunov 格式
            unew[0] = u[0] - dt * (numerical_flux_godunov(u[0], u[1], v_front) 
                                    - numerical_flux_godunov(u[-1], u[0], v_front)) / dx \
                        - dt * k_corr * u[0] * theta_pore
            
            for i in range(1, nx - 1):
                flux_right = numerical_flux_godunov(u[i], u[i + 1], v_front)
                flux_left = numerical_flux_godunov(u[i - 1], u[i], v_front)
                unew[i] = u[i] - dt * (flux_right - flux_left) / dx \
                           - dt * k_corr * u[i] * theta_pore
            
            unew[nx - 1] = u[nx - 1] - dt * (numerical_flux_godunov(u[nx - 1], u[0], v_front)
                                                - numerical_flux_godunov(u[nx - 2], u[nx - 1], v_front)) / dx \
                            - dt * k_corr * u[nx - 1] * theta_pore
        
        elif method == 'lax_wendroff':
            # Lax-Wendroff 格式
            unew[0] = 0.5 * (u[1] + u[-1]) - 0.5 * dt / dx * (
                        v_front * u[1] - v_front * u[-1]) \
                        - dt * k_corr * u[0] * theta_pore
            
            for i in range(1, nx - 1):
                unew[i] = u[i] - 0.5 * dt / dx * (v_front * u[i + 1] - v_front * u[i - 1]) \
                           + 0.5 * (dt / dx) ** 2 * v_front ** 2 * (u[i + 1] - 2 * u[i] + u[i - 1]) \
                           - dt * k_corr * u[i] * theta_pore
            
            unew[nx - 1] = u[nx - 1] - 0.5 * dt / dx * (v_front * u[0] - v_front * u[nx - 2]) \
                            + 0.5 * (dt / dx) ** 2 * v_front ** 2 * (u[0] - 2 * u[nx - 1] + u[nx - 2]) \
                            - dt * k_corr * u[nx - 1] * theta_pore
        
        elif method == 'maccormack':
            # MacCormack 预测-校正格式
            us = np.zeros(nx)
            for i in range(nx - 1):
                us[i] = u[i] - dt / dx * (v_front * u[i + 1] - v_front * u[i]) \
                         - dt * k_corr * u[i] * theta_pore
            us[nx - 1] = u[nx - 1] - dt / dx * (v_front * u[0] - v_front * u[nx - 1]) \
                          - dt * k_corr * u[nx - 1] * theta_pore
            
            unew[0] = 0.5 * (u[0] + us[0]) - 0.5 * dt / dx * (
                        v_front * us[0] - v_front * us[-1]) \
                        - dt * k_corr * us[0] * theta_pore
            
            for i in range(1, nx):
                unew[i] = 0.5 * (u[i] + us[i]) - 0.5 * dt / dx * (
                            v_front * us[i] - v_front * us[i - 1]) \
                            - dt * k_corr * us[i] * theta_pore
        
        else:
            raise ValueError(f"未知方法: {method}")
        
        # 边界保护: 比表面积不能为负
        unew = np.clip(unew, 0.0, np.max(u0) * 1.5)
        
        u = unew
        U[n_step + 1, :] = u
    
    return U


def corrosion_front_velocity(E, T=353.15):
    """
    估计腐蚀前沿传播速度 [m/s]。
    
    基于经验关联式:
        v_c = A * exp(-Ea / (R * T)) * exp(alpha * F * E / (R * T))
    """
    A_prefactor = 1e-12  # m/s
    Ea = 80000  # J/mol
    alpha = 0.3
    
    v = A_prefactor * np.exp(-Ea / (GAS_CONSTANT * T)) \
        * np.exp(alpha * FARADAY * E / (GAS_CONSTANT * T))
    
    return float(np.clip(v, 0.0, 1e-6))


def structural_integrity_loss(S_c_current, S_c_initial):
    """
    计算结构完整性损失比例。
    """
    if S_c_initial <= 0:
        return 0.0
    
    loss = 1.0 - S_c_current / S_c_initial
    return float(np.clip(loss, 0.0, 1.0))


if __name__ == "__main__":
    nx = 51
    L = 10e-6  # 10 um CCL
    dx = L / (nx - 1)
    u0 = np.ones(nx) * 200.0  # 初始比表面积 200 m^2/g
    
    E = 1.0  # V
    v = corrosion_front_velocity(E)
    k = 1e-5  # 1/s
    theta = 0.4
    dt = 0.5 * dx / max(v, 1e-10)
    nt = 100
    
    U = solve_corrosion_propagation(u0, nx, nt, dx, dt, v, k, theta, method='godunov')
    print(f"碳腐蚀传播: 初始均值={np.mean(U[0]):.2f}, 最终均值={np.mean(U[-1]):.2f} m^2/g")
