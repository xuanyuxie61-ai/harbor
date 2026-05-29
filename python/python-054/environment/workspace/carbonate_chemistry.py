"""
carbonate_chemistry.py
================================================================================
海洋碳酸盐化学系统 — 基于 Muller 复数根查找法求解质子平衡方程

核心科学问题：
    给定总溶解无机碳 (DIC) 与总碱度 (TA)，求解海水中的 [H⁺] 浓度，
    进而计算 pH、碳酸根饱和度 Ω、表层海水 pCO₂ 等关键海洋酸化指标。

科学背景：
    海水碳酸盐系统由以下平衡控制：
        CO₂ + H₂O ⇌ H₂CO₃ ⇌ H⁺ + HCO₃⁻ ⇌ 2H⁺ + CO₃²⁻
    
    定义：
        DIC = [CO₂*] + [HCO₃⁻] + [CO₃²⁻]
        TA  = [HCO₃⁻] + 2[CO₃²⁻] + [B(OH)₄⁻] + [OH⁻] + [HPO₄²⁻] + ... - [H⁺]
    
    各物种分数由平衡常数决定（Millero, 1995; Dickson, 1990）：
        α₀ = [H⁺]² / ([H⁺]² + K₁[H⁺] + K₁K₂)
        α₁ = K₁[H⁺] / ([H⁺]² + K₁[H⁺] + K₁K₂)
        α₂ = K₁K₂ / ([H⁺]² + K₁[H⁺] + K₁K₂)
    
    质子平衡方程（非线性，需数值求根）：
        f([H⁺]) = TA - DIC·(α₁ + 2α₂) - K_w/[H⁺] + [H⁺]
                  - B_T·K_B/(K_B + [H⁺]) = 0
    
    其中：
        K₁, K₂  : 碳酸第一、第二步解离常数 (mol/kg)
        K_w      : 水的离子积
        K_B      : 硼酸解离常数
        B_T      : 硼酸总浓度 ≈ 0.000416·S/35 (mol/kg)
        K_sp(CaCO₃) : 方解石/文石溶度积

核心算法（来自 zero_muller）：
    使用 Muller 二次插值法求解 f(x)=0。每次迭代用三个点拟合抛物线，
    取使 |f(x)| 更小的根作为新的近似值，收敛阶约为 1.84。
================================================================================
"""

import numpy as np


# =============================================================================
# 温度-盐度依赖的平衡常数计算 (Millero 1995, Dickson & Millero 1987)
# =============================================================================

def equilibrium_constants(T, S):
    """
    计算海水碳酸盐系统在给定温度 T (°C) 和盐度 S (psu) 下的平衡常数。
    
    公式来源：
        K1, K2 来自 Lueker et al. (2000) 对 Mehrbach et al. (1973) 的重新拟合：
            pK1 = 3633.86/Tk - 61.2172 + 9.6777·ln(Tk) - 0.011555·S + 0.0001152·S²
            pK2 = 471.78/Tk - 25.9290 + 3.16967·ln(Tk) - 0.01781·S + 0.0001122·S²
        
        Kw 来自 Millero (1995)：
            ln(Kw) = 148.9652 - 13847.26/Tk - 23.6521·ln(Tk)
                     + (-5.977 + 118.67/Tk + 1.0495·ln(Tk))·√S - 0.01615·S
        
        KB 来自 Dickson (1990)：
            ln(KB) = (-8966.90 - 2890.53·√S - 77.942·S + 1.728·S^1.5 - 0.0996·S²)/Tk
                     + 148.0248 + 137.194·√S + 1.62142·S
                     + (-24.4344 - 25.085·√S - 0.2474·S)·ln(Tk)
                     + 0.053105·√S·Tk
    
    参数:
        T : float, 温度 (°C)
        S : float, 盐度 (psu)
    
    返回:
        dict: 包含 K0, K1, K2, Kw, KB, Ksp_calcite, Ksp_aragonite
    """
    Tk = T + 273.15  # 开尔文温度
    
    # ---- K1 (碳酸第一解离常数) ----
    pK1 = 3633.86 / Tk - 61.2172 + 9.6777 * np.log(Tk) - 0.011555 * S + 0.0001152 * S**2
    K1 = 10.0**(-pK1)
    
    # ---- K2 (碳酸第二解离常数) ----
    # Lueker et al. (2000), refit of Mehrbach et al. (1973), total scale
    # 注意: 符号与某些文献转录版本不同,以下为 pyCO2SYS 验证过的正确形式
    pK2 = 471.78 / Tk + 25.929 - 3.16967 * np.log(Tk) - 0.01781 * S + 0.0001122 * S**2
    K2 = 10.0**(-pK2)
    
    # ---- K0 (CO2 溶解度, Weiss 1974) ----
    T100 = Tk / 100.0
    lnK0 = -60.2409 + 93.4517 / T100 + 23.3585 * np.log(T100) \
           + S * (0.023517 - 0.023656 * T100 + 0.0047036 * T100**2)
    K0 = np.exp(lnK0)
    
    # ---- Kw (水的离子积) ----
    lnKw = 148.9652 - 13847.26 / Tk - 23.6521 * np.log(Tk) \
           + (-5.977 + 118.67 / Tk + 1.0495 * np.log(Tk)) * np.sqrt(S) - 0.01615 * S
    Kw = np.exp(lnKw)
    
    # ---- KB (硼酸解离常数) ----
    sqrtS = np.sqrt(S)
    lnKB = (-8966.90 - 2890.53 * sqrtS - 77.942 * S + 1.728 * S**1.5 - 0.0996 * S**2) / Tk \
           + 148.0248 + 137.194 * sqrtS + 1.62142 * S \
           + (-24.4344 - 25.085 * sqrtS - 0.2474 * S) * np.log(Tk) \
           + 0.053105 * sqrtS * Tk
    KB = np.exp(lnKB)
    
    # ---- 硼酸总浓度 (Lee et al. 2010) ----
    BT = 0.0004326 * S / 35.0
    
    # ---- 溶度积 (Mucci 1983) ----
    # 方解石
    logKsp_calc = -171.9065 - 0.077993 * Tk + 2839.319 / Tk + 71.595 * np.log10(Tk) \
                  + (-0.77712 + 0.0028426 * Tk + 178.34 / Tk) * sqrtS \
                  - 0.07711 * S + 0.0041249 * S**1.5
    Ksp_calcite = 10.0**logKsp_calc
    
    # 文石
    logKsp_arag = -171.945 - 0.077993 * Tk + 2903.293 / Tk + 71.595 * np.log10(Tk) \
                  + (-0.068393 + 0.0017276 * Tk + 88.135 / Tk) * sqrtS \
                  - 0.10018 * S + 0.0059413 * S**1.5
    Ksp_aragonite = 10.0**logKsp_arag
    
    return {
        'K0': K0,
        'K1': K1,
        'K2': K2,
        'Kw': Kw,
        'KB': KB,
        'BT': BT,
        'Ksp_calcite': Ksp_calcite,
        'Ksp_aragonite': Ksp_aragonite,
    }


# =============================================================================
# Muller 根查找法 (来自 zero_muller 项目)
# =============================================================================

def zero_muller(func, x1, x2, x3, fatol=1e-12, xatol=1e-12, xrtol=1e-12, itmax=100):
    """
    Muller 二次插值法求解 f(x)=0，支持复数运算。
    
    算法推导：
        已知三点 (xold, fxold), (xmid, fxmid), (xnew, fxnew)，
        拟合二次多项式 P(x) = a·(x-xnew)² + b·(x-xnew) + c，其中 c = fxnew。
        
        由插值条件：
            a·(xold-xnew)² + b·(xold-xnew) + c = fxold
            a·(xmid-xnew)² + b·(xmid-xnew) + c = fxmid
        
        解得：
            a = [(xmid-xnew)(fxold-fxnew) - (xold-xnew)(fxmid-fxnew)] / denom
            b = [(xold-xnew)²(fxmid-fxnew) - (xmid-xnew)²(fxold-fxnew)] / denom
            c = fxnew
            denom = (xold-xnew)(xmid-xnew)(xold-xmid)
        
        二次公式给出两个候选根：
            x± = xnew + (-b ± √(b²-4ac)) / (2a)
        
        选择使 |f(x)| 更小的根。
    
    参数:
        func   : callable, f(x)
        x1, x2, x3 : complex or float, 三个初始点
        fatol  : float, |f(x)| 的绝对容差
        xatol  : float, x 增量的绝对容差
        xrtol  : float, x 增量的相对容差
        itmax  : int, 最大迭代次数
    
    返回:
        xnew   : 根的估计值
        fxnew  : 函数在根处的值
        it_num : 实际迭代次数
    """
    xold = complex(x1)
    xmid = complex(x2)
    xnew = complex(x3)
    fxold = complex(func(xold))
    fxmid = complex(func(xmid))
    fxnew = complex(func(xnew))
    
    for it_num in range(itmax):
        # 计算二次插值系数
        denom = (xold - xnew) * (xmid - xnew) * (xold - xmid)
        if abs(denom) < 1e-30:
            # 退化情形：退化为割线法
            if abs(fxmid - fxnew) > 1e-30:
                dx = -fxnew * (xmid - xnew) / (fxmid - fxnew)
            else:
                dx = complex(1e-6, 1e-6)
            x_plus = xnew + dx
            x_minus = xnew - dx
        else:
            a_coeff = ((xmid - xnew) * (fxold - fxnew) - (xold - xnew) * (fxmid - fxnew)) / denom
            b_coeff = ((xold - xnew)**2 * (fxmid - fxnew) - (xmid - xnew)**2 * (fxold - fxnew)) / denom
            c_coeff = fxnew
            
            discriminant = b_coeff**2 - 4.0 * a_coeff * c_coeff
            sqrt_disc = np.sqrt(discriminant)
            
            # 选择使分母更大的符号，避免抵消
            if abs(b_coeff + sqrt_disc) > abs(b_coeff - sqrt_disc):
                denom_plus = b_coeff + sqrt_disc
                denom_minus = b_coeff - sqrt_disc
            else:
                denom_plus = b_coeff - sqrt_disc
                denom_minus = b_coeff + sqrt_disc
            
            if abs(a_coeff) < 1e-30:
                # 几乎线性
                dx = -c_coeff / b_coeff if abs(b_coeff) > 1e-30 else complex(1e-6, 0)
                x_plus = xnew + dx
                x_minus = xnew - dx
            else:
                x_plus = xnew + (-2.0 * c_coeff) / denom_plus if abs(denom_plus) > 1e-30 else xnew
                x_minus = xnew + (-2.0 * c_coeff) / denom_minus if abs(denom_minus) > 1e-30 else xnew
        
        fx_plus = complex(func(x_plus))
        fx_minus = complex(func(x_minus))
        
        if abs(fx_plus) < abs(fx_minus):
            x_candidate = x_plus
            fx_candidate = fx_plus
        else:
            x_candidate = x_minus
            fx_candidate = fx_minus
        
        # 更新三点历史：保留离根最近的三个点
        points = [(abs(func(xold)), xold, fxold),
                  (abs(func(xmid)), xmid, fxmid),
                  (abs(fxnew), xnew, fxnew),
                  (abs(fx_candidate), x_candidate, fx_candidate)]
        points.sort(key=lambda t: t[0])
        
        xold = points[0][1]
        fxold = points[0][2]
        xmid = points[1][1]
        fxmid = points[1][2]
        xnew = points[2][1]
        fxnew = points[2][2]
        
        # 收敛检验
        dx_mag = abs(x_candidate - xnew)
        if abs(fxnew) <= fatol:
            return xnew, fxnew, it_num + 1
        if dx_mag <= xatol:
            return xnew, fxnew, it_num + 1
        if xnew != 0 and dx_mag <= xrtol * abs(xnew):
            return xnew, fxnew, it_num + 1
    
    return xnew, fxnew, itmax


# =============================================================================
# 碳酸盐系统求解器
# =============================================================================

def solve_carbonate_system(DIC, TA, T, S, Ca=0.01028):
    """
    求解海水碳酸盐系统，返回 [H⁺], pH, 各物种浓度, 饱和度状态, pCO₂。
    
    求解策略：
        1. 计算温度-盐度依赖的平衡常数
        2. 构建质子平衡残差函数 f([H⁺])
        3. 使用 Muller 法求根
        4. 通过后处理计算所有衍生量
    
    参数:
        DIC : float, 总溶解无机碳 (mol/kg)
        TA  : float, 总碱度 (mol/kg)
        T   : float, 温度 (°C)
        S   : float, 盐度 (psu)
        Ca  : float, 钙离子浓度 (mol/kg), 默认值约 10.28 mmol/kg
    
    返回:
        dict: 包含 [H⁺], pH, CO₂*, HCO₃⁻, CO₃²⁻, pCO₂, Ω_calcite, Ω_aragonite
    """
    # 边界检验
    if DIC <= 0 or TA <= 0:
        raise ValueError("DIC 和 TA 必须为正数")
    if not (0 <= T <= 40):
        raise ValueError("温度 T 超出合理海洋范围 [0, 40]°C")
    if not (0 <= S <= 45):
        raise ValueError("盐度 S 超出合理海洋范围 [0, 45] psu")
    
    K = equilibrium_constants(T, S)
    K1, K2, Kw, KB, BT = K['K1'], K['K2'], K['Kw'], K['KB'], K['BT']
    
    # 质子平衡残差函数
    def proton_residual(H):
        if isinstance(H, complex):
            H = float(H.real)
        else:
            H = float(H)
        if H <= 0:
            return 1e6
        H2 = H * H
        denom = H2 + K1 * H + K1 * K2
        if denom <= 0:
            return 1e6
        alpha1 = K1 * H / denom
        alpha2 = K1 * K2 / denom
        
        residual = TA - DIC * (alpha1 + 2.0 * alpha2) - Kw / H + H \
                   - BT * KB / (KB + H)
        return residual
    
    # Muller 法初始猜测：先用 pH ≈ 8.0 附近搜索
    H_guess = 10.0**(-8.0)
    # 提供三个初始点：稍酸、中性、稍碱
    x1 = H_guess * 0.1
    x2 = H_guess
    x3 = H_guess * 10.0
    
    H_root, fH, iters = zero_muller(proton_residual, x1, x2, x3,
                                     fatol=1e-14, xatol=1e-16, itmax=200)
    H = float(H_root.real)
    if H <= 0:
        H = 1e-8  # 回退
    
    # ---- 后处理：计算所有物种浓度 ----
    # [HOLE 1] 需要补全碳酸盐系统后处理：
    #   1. 计算各物种分数 alpha0, alpha1, alpha2
    #   2. 计算 CO2star, HCO3, CO3 浓度
    #   3. 计算 pH = -log10(H)
    #   4. 计算 pCO2（注意单位转换为 μatm）
    #   5. 计算碳酸钙饱和度 Omega_calcite 和 Omega_aragonite
    #   6. 按约定格式返回字典
    # 关键科学公式：
    #   alpha0 = H² / (H² + K1·H + K1·K2)
    #   alpha1 = K1·H / (H² + K1·H + K1·K2)
    #   alpha2 = K1·K2 / (H² + K1·H + K1·K2)
    #   pCO2 = (CO2star / K0) × 1e6
    #   Omega = [Ca²⁺][CO₃²⁻] / Ksp
    raise NotImplementedError("HOLE 1: 碳酸盐系统后处理与返回结构待补全")


def batch_solve_carbonate(DIC_arr, TA_arr, T_arr, S_arr, units='molkg'):
    """
    批量求解碳酸盐系统，支持数组输入。
    
    参数:
        DIC_arr, TA_arr, T_arr, S_arr : array-like，等长数组
        units : str, 'molkg' (mol/kg) 或 'umolkg' (μmol/kg)
    
    返回:
        list of dict: 每个元素为一个 solve_carbonate_system 的返回字典
    """
    n = len(DIC_arr)
    scale = 1e-6 if units == 'umolkg' else 1.0
    results = []
    for i in range(n):
        try:
            res = solve_carbonate_system(
                DIC_arr[i] * scale, TA_arr[i] * scale, T_arr[i], S_arr[i]
            )
        except ValueError as e:
            res = {'error': str(e), 'pH': np.nan, 'pCO2': np.nan,
                   'Omega_aragonite': np.nan}
        results.append(res)
    return results


def air_sea_co2_flux(pCO2_ocean, pCO2_atm, T, S, u10=5.0):
    """
    计算海-气 CO₂ 通量 (mmol C m⁻² d⁻¹)。
    
    使用 Wanninkhof (2014) 气体交换参数化：
        F = k_w · K0 · (pCO2_ocean - pCO2_atm)
        k_w = 0.251 · u10² · (Sc/660)^(-0.5)
        Sc = 2116.8 - 136.25·T + 4.7353·T² - 0.092307·T³ + 0.0007555·T⁴
    
    参数:
        pCO2_ocean : float, 表层海水 pCO₂ (μatm)
        pCO2_atm   : float, 大气 pCO₂ (μatm)
        T          : float, 温度 (°C)
        S          : float, 盐度 (psu)
        u10        : float, 10m 风速 (m/s)
    
    返回:
        float: 通量 (正值表示海洋→大气释放，负值表示大气→海洋吸收)
    """
    # Schmidt 数 (Wanninkhof 2014, 针对 CO₂ 在海水中的扩散)
    Sc = 2116.8 - 136.25 * T + 4.7353 * T**2 - 0.092307 * T**3 + 0.0007555 * T**4
    if Sc <= 0:
        Sc = 1.0
    
    # 气体传输速率 (cm/h)
    k_cm_h = 0.251 * u10**2 * (Sc / 660.0)**(-0.5)
    # 转换为 m/d
    k_m_d = k_cm_h * 0.24  # cm/h -> m/d (1 cm/h = 0.24 m/d)
    
    # CO2 溶解度 (mol/(m³·μatm))
    K = equilibrium_constants(T, S)
    K0 = K['K0']  # mol/(kg·atm)
    # 海水密度近似 (kg/m³)
    rho_sw = 1023.0 + 0.8 * (S - 35.0) - 0.4 * (T - 20.0)
    K0_molar = K0 * rho_sw / 1e6  # mol/(m³·μatm)
    
    flux = k_m_d * K0_molar * (pCO2_ocean - pCO2_atm)  # mmol/m²/d (因 pCO2 单位为 μatm)
    flux *= 1e3  # mol -> mmol
    return flux
