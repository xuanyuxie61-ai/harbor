"""
carbon_transport.py
================================================================================
海洋碳输送的常微分方程与反应-扩散方程求解

融合项目：
    - 343_euler           : 显式前向欧拉法
    - 059_autocatalytic_ode : 自催化化学反应动力学

核心科学问题：
    模拟海洋中溶解无机碳 (DIC) 的时间演化，包含：
    (1) 物理输送（平流-扩散）
    (2) 生物泵（光合作用、呼吸、沉降）
    (3) 海-气交换
    (4) 碳酸盐化学平衡的快速动力学（自催化特征）

科学背景：
    一维垂直水柱中的 DIC 守恒方程：
        ∂DIC/∂t = -w·∂DIC/∂z + ∂/∂z(K_z·∂DIC/∂z) + J_bio(z,t) - J_airsea(t)·δ(z=0)
                  + J_remin(z,t)
    
    其中：
        w       : 上升流/下沉速度 (m/s)
        K_z     : 垂向扩散系数 (m²/s)
        J_bio   : 生物源汇项
        J_remin : 颗粒物再矿化
    
    生物泵参数化（基于 Michaelis-Menten 动力学）：
        J_bio = μ_max · exp(-z/z_euphotic) · DIC / (DIC + K_half)
    
    自催化碳酸盐动力学（快速时间尺度，源自 Gray-Scott 思想）：
        CO₂ + H₂O ⇌ H₂CO₃  (水合反应，催化酶：碳酸酐酶)
        
        在反应-扩散框架下，可将 pH 缓冲视为一种自催化反馈：
        当 [CO₃²⁻] 升高 → pH 升高 → 促进 CO₂ → HCO₃⁻ 转化 → 缓冲 pH 变化

================================================================================
"""

import numpy as np


# =============================================================================
# 显式前向欧拉法 (来自 euler 项目)
# =============================================================================

def euler_forward(dydt, t_span, y0, n_steps):
    """
    显式前向欧拉法积分常微分方程组 dy/dt = f(t, y)。
    
    离散格式：
        y_{n+1} = y_n + Δt · f(t_n, y_n)
        Δt = (t_stop - t_0) / n_steps
    
    参数:
        dydt    : callable, f(t, y) -> ndarray
        t_span  : tuple, (t0, t_stop)
        y0      : ndarray, 初始条件
        n_steps : int, 时间步数
    
    返回:
        t : ndarray, 时间序列
        y : ndarray, 解矩阵，shape (n_steps+1, len(y0))
    """
    t0, t_stop = t_span
    dt = (t_stop - t0) / n_steps
    m = len(y0)
    
    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1, m))
    
    t[0] = t0
    y[0, :] = y0
    
    for i in range(n_steps):
        y[i+1, :] = y[i, :] + dt * dydt(t[i], y[i, :])
        t[i+1] = t[i] + dt
    
    return t, y


# =============================================================================
# 自催化碳酸盐动力学 (来自 autocatalytic_ode 思想)
# =============================================================================

def autocatalytic_carbonate_deriv(t, state, alpha=0.002, beta=0.08, gamma=0.5):
    """
    自催化碳酸盐反应动力学系统的右端项。
    
    状态变量：
        w = [CO₂(aq)]     (溶解 CO₂)
        x = [HCO₃⁻]       (碳酸氢根)
        y = [CO₃²⁻]       (碳酸根)
        z = [CaCO₃(s)]    (固体碳酸钙，沉淀量)
    
    反应网络（简化的 Gray-Scott 型自催化）：
        dw/dt = -α·w                          (CO₂ 消耗/气交换)
        dx/dt =  α·w - β·x - x·y²             (HCO₃⁻ 产生与自催化消耗)
        dy/dt =  β·x + x·y² - γ·y             (CO₃²⁻ 自催化生成与沉降)
        dz/dt =  γ·y                          (CaCO₃ 沉淀)
    
    其中 x·y² 项体现自催化：CO₃²⁻ 浓度升高促进 HCO₃⁻ → CO₃²⁻ 转化。
    
    参数:
        state : ndarray, [w, x, y, z]
        alpha : float, CO₂ 消耗率 (1/天)
        beta  : float, 缓冲反应速率
        gamma : float, 沉降速率
    
    返回:
        ndarray, 时间导数 [dw/dt, dx/dt, dy/dt, dz/dt]
    """
    w, x, y, z = state
    
    # 边界处理：防止负浓度
    w = max(w, 0.0)
    x = max(x, 0.0)
    y = max(y, 0.0)
    z = max(z, 0.0)
    
    dwdt = -alpha * w
    dxdt = alpha * w - beta * x - x * y * y
    dydt = beta * x + x * y * y - gamma * y
    dzdt = gamma * y
    
    return np.array([dwdt, dxdt, dydt, dzdt])


# =============================================================================
# 一维垂向碳输送模型
# =============================================================================

def vertical_carbon_transport_model(
    z_grid, DIC_initial, T_profile, S_profile,
    dt_days=1.0, n_days=365, w=0.0, Kz=1e-4,
    mu_max=0.1, z_euphotic=50.0, K_half=10.0,
    pCO2_atm=410.0, u10=5.0,
    remin_rate=0.01, remin_depth_scale=1000.0
):
    """
    一维垂向水柱 DIC 输送模型。
    
    空间离散：有限体积法在均匀网格上
        ∂DIC_i/∂t = -(F_{i+1/2} - F_{i-1/2})/Δz + J_bio,i + J_remin,i
    
    通量 F = w·DIC - Kz·∂DIC/∂z
    
    边界条件：
        上边界 (z=0): 海-气 CO₂ 通量作为 Neumann 条件
        下边界 (z=-H): 零通量
    
    参数:
        z_grid        : ndarray, 深度网格 (m, 负值，0=表层)
        DIC_initial   : ndarray, 初始 DIC 分布 (μmol/kg)
        T_profile     : ndarray, 温度剖面 (°C)
        S_profile     : ndarray, 盐度剖面 (psu)
        dt_days       : float, 时间步长 (天)
        n_days        : int, 总积分天数
        w             : float, 垂向速度 (m/s, 正为上升)
        Kz            : float, 垂向扩散系数 (m²/s)
        mu_max        : float, 最大光合速率 (μmol/kg/day)
        z_euphotic    : float, 真光层深度 (m)
        K_half        : float, 半饱和常数 (μmol/kg)
        pCO2_atm      : float, 大气 pCO₂ (μatm)
        u10           : float, 风速 (m/s)
        remin_rate    : float, 再矿化速率 (1/day)
        remin_depth_scale : float, 再矿化特征深度 (m)
    
    返回:
        DIC_history : ndarray, shape (n_steps+1, nz)
        t_history   : ndarray, 时间 (天)
    """
    nz = len(z_grid)
    dz = np.diff(z_grid)
    if not np.allclose(dz, dz[0]):
        raise ValueError("当前实现要求均匀深度网格")
    dz = dz[0]
    
    # 转换时间步为秒
    dt = dt_days * 86400.0
    n_steps = int(n_days / dt_days)
    
    DIC = DIC_initial.copy().astype(float)
    DIC_history = np.zeros((n_steps + 1, nz))
    DIC_history[0, :] = DIC
    t_history = np.zeros(n_steps + 1)
    
    # 从 carbonate_chemistry 导入海-气通量计算
    from carbonate_chemistry import air_sea_co2_flux, solve_carbonate_system
    
    for step in range(n_steps):
        # [HOLE 2] 需要补全上边界海-气通量计算：
        #   1. 调用 solve_carbonate_system 计算表层 pCO2
        #      注意：DIC 输入单位为 μmol/kg，需转换为 mol/kg
        #      总碱度 TA 可近似为常数 2.3e-3 mol/kg
        #   2. 从返回结果中提取 pCO2（单位 μatm）
        #   3. 调用 air_sea_co2_flux 计算海-气通量
        #   4. 将通量从 mmol/m²/d 转换为 μmol/kg/day
        #      使用混合层深度 H_mld 和海水密度 rho
        #   提示：此处的单位链必须与 carbonate_chemistry.py 中 pCO2 的单位约定保持一致
        raise NotImplementedError("HOLE 2: 上边界海-气通量计算与单位转换待补全")
        
        DIC_new = DIC.copy()
        
        # 内部点：平流-扩散 + 生物源汇
        for i in range(1, nz - 1):
            # 平流项 (中心差分)
            adv = -w * (DIC[i+1] - DIC[i-1]) / (2.0 * abs(dz))
            # 扩散项
            diff = Kz * (DIC[i+1] - 2.0 * DIC[i] + DIC[i-1]) / (dz**2)
            # 生物源汇 (真光层内光合消耗 DIC)
            z_depth = abs(z_grid[i])
            J_bio = -mu_max * np.exp(-z_depth / z_euphotic) * DIC[i] / (DIC[i] + K_half)
            # 再矿化 (深层)
            J_remin = remin_rate * np.exp(-z_depth / remin_depth_scale) * (2000.0 - DIC[i])
            
            dDICdt = adv + diff + J_bio + J_remin
            DIC_new[i] = DIC[i] + dt_days * dDICdt
        
        # 上边界：包含海-气交换
        z_depth = abs(z_grid[0])
        J_bio0 = -mu_max * np.exp(-z_depth / z_euphotic) * DIC[0] / (DIC[0] + K_half)
        J_remin0 = remin_rate * np.exp(-z_depth / remin_depth_scale) * (2000.0 - DIC[0])
        adv0 = -w * (DIC[1] - DIC[0]) / abs(dz) if nz > 1 else 0.0
        diff0 = Kz * (DIC[1] - DIC[0]) / (dz**2) if nz > 1 else 0.0
        DIC_new[0] = DIC[0] + dt_days * (adv0 + diff0 + J_bio0 + J_remin0 + flux_top_conc)
        
        # 下边界：零通量
        DIC_new[-1] = DIC[-1]
        if nz > 1:
            adv_bot = -w * (DIC[-1] - DIC[-2]) / abs(dz)
            diff_bot = Kz * (DIC[-2] - DIC[-1]) / (dz**2)
            J_bio_bot = -mu_max * np.exp(-abs(z_grid[-1]) / z_euphotic) * DIC[-1] / (DIC[-1] + K_half)
            J_remin_bot = remin_rate * np.exp(-abs(z_grid[-1]) / remin_depth_scale) * (2000.0 - DIC[-1])
            DIC_new[-1] = DIC[-1] + dt_days * (adv_bot + diff_bot + J_bio_bot + J_remin_bot)
        
        # 非负约束
        DIC_new = np.maximum(DIC_new, 0.0)
        DIC = DIC_new
        DIC_history[step + 1, :] = DIC
        t_history[step + 1] = (step + 1) * dt_days
    
    return DIC_history, t_history


# =============================================================================
# 箱式碳循环模型（简化全球碳循环）
# =============================================================================

def box_carbon_cycle_model(t_span, y0, n_steps, 
                           k12=0.1, k21=0.05, k23=0.02, k32=0.01,
                           F_anthro=8.0, buffer_factor=10.0):
    """
    三箱碳循环模型（大气-表层海洋-深层海洋）。
    
    状态变量：
        y[0] = N1  (大气碳库, Pg C)
        y[1] = N2  (表层海洋碳库, Pg C)
        y[2] = N3  (深层海洋碳库, Pg C)
    
    动力学方程（Takahashi 风格）：
        dN1/dt = -k12·N1 + k21·N2 + F_anthro
        dN2/dt =  k12·N1 - k21·N2 - k23·N2 + k32·N3 - (N2-N2_0)/buffer_factor
        dN3/dt =  k23·N2 - k32·N3
    
    buffer factor (Revelle factor) 体现海洋对 CO₂ 吸收的缓冲能力：
        ∂ln(pCO₂)/∂ln(DIC) ≈ 10
    
    参数:
        t_span  : tuple, (t0, t_stop)，单位：年
        y0      : ndarray, [N1_0, N2_0, N3_0]
        n_steps : int
        k12, k21, k23, k32 : float, 箱间交换系数 (1/年)
        F_anthro : float, 人为 CO₂ 排放速率 (Pg C/年)
        buffer_factor : float, Revelle 缓冲因子
    
    返回:
        t, y (同 euler_forward 输出格式)
    """
    N2_0 = y0[1]
    
    def dydt(t, y):
        N1, N2, N3 = y
        # 边界保护
        N1 = max(N1, 0.0)
        N2 = max(N2, 0.0)
        N3 = max(N3, 0.0)
        
        dN1 = -k12 * N1 + k21 * N2 + F_anthro
        dN2 = k12 * N1 - k21 * N2 - k23 * N2 + k32 * N3 - (N2 - N2_0) / buffer_factor
        dN3 = k23 * N2 - k32 * N3
        return np.array([dN1, dN2, dN3])
    
    return euler_forward(dydt, t_span, y0, n_steps)


def compute_anthropogenic_carbon_inventory(DIC_pre, DIC_post, rho, thickness):
    """
    计算人为碳库存量 (mol C/m²)。
    
    C_anth = ∫ (DIC_post - DIC_pre) · ρ · dz
    
    参数:
        DIC_pre    : ndarray, 工业化前 DIC (μmol/kg)
        DIC_post   : ndarray, 现代 DIC (μmol/kg)
        rho        : float or ndarray, 海水密度 (kg/m³)
        thickness  : float or ndarray, 层厚 (m)
    
    返回:
        float: 单位面积人为碳库存 (mol C/m²)
    """
    delta_DIC = np.array(DIC_post) - np.array(DIC_pre)  # μmol/kg
    if np.isscalar(rho):
        rho = np.full_like(delta_DIC, float(rho))
    if np.isscalar(thickness):
        thickness = np.full_like(delta_DIC, float(thickness))
    
    # mol/m² = μmol/kg * kg/m³ * m * 1e-6
    inventory = np.sum(delta_DIC * rho * thickness * 1e-6)
    return inventory
