"""
核裂变势能面 (Potential Energy Surface, PES) 计算与极值分析
==============================================================
融合原始项目:
  - 898_polynomials: 多元多项式系统 (camera_f 等)
  - 1430_zero_laguerre: Laguerre 多项式求根方法
  - 1431_zero_muller: Muller 复数求根方法

科学背景:
---------
裂变势能面 V(q) 描述核系统在不同形变下的宏观能量，由
液滴模型 (Liquid Drop Model, LDM) 与壳修正 (Shell Correction) 组成:

  V(q) = E_LDM(q) + δE_shell(q) + δE_pair(q)

液滴模型能量:
  E_LDM = E_vol + E_surf + E_Coul + E_asym + E_rot

其中各项经典公式:
  E_vol   = -a_v A                    (体积能)
  E_surf  = a_s A^(2/3) B_s(q)        (表面能, B_s为表面积比)
  E_Coul  = a_c Z²/A^(1/3) B_C(q)     (库仑能, B_C为库仑形状因子)
  E_asym  = a_a (N-Z)²/A              (不对称能)

裂变势垒的经典表达式 (对于 β₂ 主导的一维近似):
  V(β₂) ≈ E₀ [ (β₂/β₂^f)² - 2(β₂/β₂^f)³ ]
其中 β₂^f 为鞍点形变。

壳修正能量通过 Strutinsky 方法计算:
  δE_shell = Σ_μ (ε_μ - ε̃_μ) n_μ
其中 ε_μ 为单粒子能级，ε̃_μ 为平滑化的能级，n_μ 为占据数。

本模块还包含势能面极值点（鞍点、极小值）的数值搜索，
这是计算裂变路径与裂变半寿命的关键步骤。
"""

import numpy as np
from typing import Callable, Tuple, Optional

# 液滴模型参数 (Myers-Swiatecki 参数集, 单位 MeV)
LDM_A_VOLUME = 15.4941
LDM_A_SURFACE = 17.9439
LDM_A_COULOMB = 0.7053
LDM_A_ASYMMETRY = 23.2

# 表面与库仑形状因子近似系数 (基于椭球近似)
def surface_area_ratio(beta2: float) -> float:
    """
    表面积比 B_s(β₂) ≈ 1 + (2/5)β₂² - (38/105)β₂³ + ...
    使用椭球表面积的精确表达式数值近似。
    """
    # 小形变展开，截断到 β₂⁴ 阶
    b2 = beta2
    b2sq = b2 * b2
    return 1.0 + 0.4 * b2sq - (38.0 / 105.0) * b2sq * b2 + 0.2 * b2sq * b2sq


def coulomb_shape_factor(beta2: float) -> float:
    """
    库仑形状因子 B_C(β₂) ≈ 1 - (1/5)β₂² - (4/105)β₂³ + ...
    """
    b2 = beta2
    b2sq = b2 * b2
    return 1.0 - 0.2 * b2sq - (4.0 / 105.0) * b2sq * b2 + 0.05 * b2sq * b2sq


def liquid_drop_energy(mass_number: int, charge_number: int, beta2: float) -> float:
    """
    计算一维近似下的液滴模型能量 (MeV).
    
    公式:
    E_LDM = -a_v A + a_s A^(2/3) B_s(β₂) + a_c Z²/A^(1/3) B_C(β₂) + a_a (A-2Z)²/A
    
    参数:
        mass_number: 质量数 A
        charge_number: 电荷数 Z
        beta2: 四极形变
    返回:
        总液滴能量 (MeV)
    """
    if mass_number <= 0 or charge_number <= 0 or charge_number > mass_number:
        raise ValueError("Invalid nuclear parameters")
    A = float(mass_number)
    Z = float(charge_number)
    N = A - Z
    
    Bs = surface_area_ratio(beta2)
    Bc = coulomb_shape_factor(beta2)
    
    E_vol = -LDM_A_VOLUME * A
    E_surf = LDM_A_SURFACE * (A ** (2.0 / 3.0)) * Bs
    E_coul = LDM_A_COULOMB * (Z ** 2) / (A ** (1.0 / 3.0)) * Bc
    E_asym = LDM_A_ASYMMETRY * (N - Z) ** 2 / A
    
    return E_vol + E_surf + E_coul + E_asym


def shell_correction_energy(beta2: float, beta3: float, mass_number: int) -> float:
    """
    壳修正能量的简化模型.
    
    采用高斯型壳修正近似 (Nilsson 模型简化):
    δE_shell = Σ_λ A_λ exp( - (β_λ - β_λ^shell)² / (2 σ_λ²) ) cos(ω_λ β_λ)
    
    其中振幅 A_λ 与能级密度相关，ω_λ 反映壳振荡周期。
    这里使用基于经验公式的近似。
    """
    # 主壳修正振幅 (~ 几个 MeV)
    A_shell = 5.0 * np.exp(-mass_number / 200.0)
    
    # β₂ 方向的壳振荡
    omega2 = 8.0  # 由主壳量子数决定
    phase2 = omega2 * beta2
    
    # β₃ 方向的壳修正通常较弱
    A3 = 0.3 * A_shell
    omega3 = 6.0
    phase3 = omega3 * beta3
    
    # 阻尼因子：大形变下壳修正衰减
    damping = np.exp(-0.5 * (beta2 ** 2 + beta3 ** 2))
    
    delta_E = A_shell * np.cos(phase2) * damping + A3 * np.cos(phase3) * damping
    return delta_E


def pairing_correction_energy(delta: float, delta_0: float) -> float:
    """
    BCS 配对修正能.
    
    公式:
    δE_pair = - (1/4) g(ε_F) Δ²
    其中 g(ε_F) 为费米面单粒子能级密度，Δ 为配对能隙。
    
    简化表达（以 Δ₀ 为参考）:
    δE_pair = - Δ² / Δ₀  (单位 MeV, 近似标度)
    """
    if delta < 0:
        delta = 0.0
    if delta_0 <= 0:
        delta_0 = 1.0
    return -(delta ** 2) / delta_0


def fission_barrier_height(mass_number: int, charge_number: int) -> float:
    """
    计算裂变势垒高度的经验公式 (MeV).
    
    基于液滴模型的 fissility parameter:
    x = E_Coul(球) / (2 E_surf(球)) = (a_c Z²/A^(1/3)) / (2 a_s A^(2/3))
    
    势垒高度近似:
    E_B ≈ E_surf(0) (1 - x)³ / (1 + x)   for 0 < x < 1
    """
    A = float(mass_number)
    Z = float(charge_number)
    x_fissility = (LDM_A_COULOMB * Z ** 2 / A ** (1.0 / 3.0)) / (
        2.0 * LDM_A_SURFACE * A ** (2.0 / 3.0)
    )
    x_fissility = np.clip(x_fissility, 0.0, 1.0)
    E_surf_0 = LDM_A_SURFACE * A ** (2.0 / 3.0)
    if x_fissility >= 1.0:
        return 0.0
    barrier = E_surf_0 * (1.0 - x_fissility) ** 3 / (1.0 + x_fissility)
    return barrier


def potential_energy(
    q: np.ndarray,
    mass_number: int,
    charge_number: int,
    delta_0: float = 1.5,
) -> float:
    """
    计算裂变势能面 V(q) 在给定集体坐标 q 处的值.
    
    参数:
        q: 集体坐标 [β₂, β₃, β₄, β₅, Δ]
        mass_number: 质量数 A
        charge_number: 电荷数 Z
        delta_0: 参考配对能隙 (MeV)
    返回:
        势能值 (MeV)，以球形基态为参考零点
    """
    if len(q) < 5:
        raise ValueError("q must contain at least 5 elements")
    beta2, beta3, beta4, beta5, delta_val = q[0], q[1], q[2], q[3], q[4]
    
    # 基态能量（球形，无配对激发）
    E_gs = liquid_drop_energy(mass_number, charge_number, 0.0)
    
    # 当前构型液滴能
    E_ldm = liquid_drop_energy(mass_number, charge_number, beta2)
    
    # 高阶形变修正（β₄, β₅ 的弹性恢复能）
    E_higher = 20.0 * (beta4 ** 2 + beta5 ** 2) * (mass_number ** (2.0 / 3.0))
    
    # 壳修正
    E_shell = shell_correction_energy(beta2, beta3, mass_number)
    
    # 配对修正
    E_pair = pairing_correction_energy(delta_val, delta_0)
    
    # 总势能（相对基态）
    V = (E_ldm - E_gs) + E_higher + E_shell + E_pair
    return float(V)


# ============================================================================
# 极值点搜索 (融合 zero_laguerre.m 与 zero_muller.m)
# ============================================================================

def potential_energy_1d(beta2: float, mass_number: int, charge_number: int) -> float:
    """一维势能函数（仅 β₂ 变化，其余为零），用于求根测试."""
    q = np.array([beta2, 0.0, 0.0, 0.0, 0.0])
    return potential_energy(q, mass_number, charge_number)


def zero_laguerre(
    f: Callable[[float], float],
    x0: float,
    degree: int = 4,
    abserr: float = 1e-10,
    kmax: int = 100,
) -> Tuple[float, int, int]:
    """
    Laguerre 多项式求根方法 (改编自 zero_laguerre.m).
    
    用于搜索势能面导数的零点（即极值点）。
    迭代公式:
    z = (f')² - (β+1) f f'' ,  β = 1/(degree-1)
    dx = -(β+1) f / [β f' + √z]
    
    参数:
        f: 目标函数
        x0: 初始猜测
        degree: 预估多项式次数（影响步长参数）
        abserr: 收敛容差
        kmax: 最大迭代次数
    返回:
        (根, 错误码, 迭代次数)
    """
    if degree < 2:
        degree = 2
    x = float(x0)
    ierror = 0
    k = 0
    beta = 1.0 / (degree - 1)
    
    h = 1e-6  # 数值微分步长
    
    while True:
        fx = f(x)
        if abs(fx) <= abserr:
            break
        
        # 数值一阶、二阶导数
        fp = (f(x + h) - f(x - h)) / (2.0 * h)
        fpp = (f(x + h) - 2.0 * fx + f(x - h)) / (h * h)
        
        k += 1
        if k > kmax:
            ierror = 2
            return x, ierror, k
        
        z = fp ** 2 - (beta + 1.0) * fx * fpp
        z = max(z, 0.0)
        bot = beta * fp + np.sqrt(z)
        
        if abs(bot) < 1e-15:
            ierror = 3
            return x, ierror, k
        
        dx = -(beta + 1.0) * fx / bot
        x = x + dx
        
        # 边界保护
        if not np.isfinite(x):
            ierror = 4
            return x, ierror, k
    
    return x, ierror, k


def zero_muller(
    f: Callable[[complex], complex],
    x1: complex,
    x2: complex,
    x3: complex,
    fatol: float = 1e-12,
    xatol: float = 1e-12,
    xrtol: float = 1e-12,
    itmax: int = 100,
) -> Tuple[complex, complex]:
    """
    Muller 复数求根方法 (改编自 zero_muller.m).
    
    通过三个点构造抛物线插值，选取使 |f| 更小的根作为下一步迭代点。
    适用于复平面上的势能解析延拓，寻找 saddle point 的复数解（
    这在量子隧穿计算中有重要意义）。
    
    二次插值公式:
    a = [(x_m-x_n)(f_o-f_n) - (x_o-x_n)(f_m-f_n)] / [(x_o-x_n)(x_m-x_n)(x_o-x_m)]
    b = ...
    c = f_n
    x_new = x_n - 2c / [b ± √(b²-4ac)]
    """
    xnew = complex(x1)
    xmid = complex(x2)
    xold = complex(x3)
    
    fxnew = f(xnew)
    fxmid = f(xmid)
    fxold = f(xold)
    
    if abs(fxnew) < fatol:
        return xnew, fxnew
    
    iterate = 0
    while True:
        # 保持 fxnew 为最小的函数值
        if abs(fxmid) <= abs(fxnew):
            xnew, xmid = xmid, xnew
            fxnew, fxmid = fxmid, fxnew
        
        xlast = xnew
        iterate += 1
        if iterate > itmax:
            break
        
        # Muller 差商
        a_num = (xmid - xnew) * (fxold - fxnew) - (xold - xnew) * (fxmid - fxnew)
        b_num = (xold - xnew) ** 2 * (fxmid - fxnew) - (xmid - xnew) ** 2 * (fxold - fxnew)
        c_num = (xold - xnew) * (xmid - xnew) * (xold - xmid) * fxnew
        
        denom = (xold - xnew) * (xmid - xnew) * (xold - xmid)
        if abs(denom) < 1e-20:
            break
        
        a = a_num / denom
        b = b_num / denom
        # c = c_num / denom  # 实际上使用标准二次公式形式
        
        # 标准 Muller 公式，使用更稳定的计算
        a_coef = a_num
        b_coef = b_num
        c_coef = c_num
        
        discrm = b_coef ** 2 - 4.0 * a_coef * c_coef
        
        if abs(a_coef) < 1e-20:
            break
        
        sqrt_disc = np.sqrt(discrm)
        xplus = xnew + (-b_coef + sqrt_disc) / (2.0 * a_coef)
        xminus = xnew + (-b_coef - sqrt_disc) / (2.0 * a_coef)
        
        fplus = f(xplus)
        fminus = f(xminus)
        
        if abs(fminus) < abs(fplus):
            xnew = xminus
            fxnew = fminus
        else:
            xnew = xplus
            fxnew = fplus
        
        fxold = fxmid
        fxmid = fxnew
        xold = xmid
        xmid = xlast
        
        # 收敛检验
        x_inc = xnew - xmid
        x_ave = abs(xnew + xmid + xold) / 3.0
        if abs(x_inc) <= xatol:
            break
        if abs(x_inc) <= xrtol * x_ave:
            break
        if abs(fxnew) <= fatol:
            break
    
    return xnew, fxnew


def find_saddle_point_1d(
    mass_number: int,
    charge_number: int,
    beta2_min: float = -0.3,
    beta2_max: float = 2.0,
) -> Tuple[float, float]:
    """
    在 β₂ 方向上搜索鞍点（一维近似）。
    
    策略:
    1. 粗网格扫描确定势能最大值区间
    2. 在该区间内使用 zero_laguerre 搜索 dV/dβ₂ = 0
    
    返回:
        (beta2_saddle, V_saddle)
    """
    n_scan = 200
    beta2_grid = np.linspace(beta2_min, beta2_max, n_scan)
    V_grid = np.array([potential_energy_1d(b, mass_number, charge_number) for b in beta2_grid])
    
    # 寻找势能极大值（鞍点近似）
    dV = np.diff(V_grid)
    sign_change = np.where((dV[:-1] > 0) & (dV[1:] < 0))[0]
    
    if len(sign_change) == 0:
        # _fallback: 取最大值点
        idx_max = np.argmax(V_grid)
        return float(beta2_grid[idx_max]), float(V_grid[idx_max])
    
    # 在第一个极大值附近精确定位
    idx = sign_change[0]
    x0 = float(beta2_grid[idx + 1])
    
    dV_func = lambda b: (potential_energy_1d(b + 1e-5, mass_number, charge_number) -
                         potential_energy_1d(b - 1e-5, mass_number, charge_number)) / (2e-5)
    
    beta_saddle, ierr, _ = zero_laguerre(dV_func, x0, degree=6, abserr=1e-8, kmax=200)
    
    if ierr != 0:
        # fallback
        beta_saddle = x0
    
    V_saddle = potential_energy_1d(beta_saddle, mass_number, charge_number)
    return float(beta_saddle), float(V_saddle)


def find_scission_point_1d(
    mass_number: int,
    charge_number: int,
    beta2_saddle: float,
    beta2_max: float = 3.0,
) -> Tuple[float, float]:
    """
    搜索断裂点 (scission point)：势能由下降转为趋于平坦的点。
    
    断裂判据: d²V/dβ₂² → 0 且 V 趋于两碎片库仑能之和
    近似方法: 寻找 |dV/dβ₂| < threshold 且 β₂ > β₂^saddle 的第一个点
    """
    n_scan = 300
    beta2_grid = np.linspace(beta2_saddle, beta2_max, n_scan)
    V_grid = np.array([potential_energy_1d(b, mass_number, charge_number) for b in beta2_grid])
    
    dV = np.gradient(V_grid, beta2_grid)
    
    # 寻找斜率接近零且位于鞍点之后的点
    threshold = 0.5  # MeV per unit beta2
    candidates = np.where((np.abs(dV) < threshold) & (beta2_grid > beta2_saddle + 0.2))[0]
    
    if len(candidates) == 0:
        idx = len(beta2_grid) - 1
    else:
        idx = candidates[0]
    
    return float(beta2_grid[idx]), float(V_grid[idx])
