"""
utils.py - 通用数值工具与科学常数库
========================================
为火箭发动机燃烧不稳定分析提供底层数值支持。
包含：CORDIC高精度三角函数、Gamma函数、科学常数、数值稳定性工具。

原项目映射:
- 219_cordic   -> CORDIC算法用于燃烧室几何高精度角度计算
- 179_circle_integrals -> Gamma函数与圆积分用于截面特性分析
"""

import numpy as np
from math import factorial

# ============================================================
# 科学常数 (SI单位制)
# ============================================================
R_UNIVERSAL = 8.314462618   # 通用气体常数, J/(mol·K)
GAMMA_AIR = 1.4             # 空气比热比
GAMMA_COMBUSTION = 1.2      # 燃烧产物比热比 (典型LOX/RP-1)
CP_COMBUSTION = 1800.0      # 燃烧产物定压比热, J/(kg·K)
RHO_OX = 1141.0             # 液氧密度, kg/m^3
RHO_FUEL = 807.0            # RP-1煤油密度, kg/m^3
T_ADIABATIC = 3600.0        # 绝热燃烧温度, K
PRESSURE_CHAMBER = 7.0e6    # 燃烧室压力, Pa (7 MPa typical)
SOUND_SPEED = 1200.0        # 燃烧室声速, m/s
DYNAMIC_VISCOSITY = 8.5e-5  # 燃烧产物动力粘度, Pa·s
LATENT_HEAT = 2.13e6        # 汽化潜热, J/kg
DIFFUSIVITY_THERMAL = 1.2e-4 # 热扩散系数, m^2/s
PRE_EXPONENTIAL = 1.8e10    # Arrhenius预指数因子, 1/s
ACTIVATION_ENERGY = 1.26e5  # 活化能, J/mol
STOICHIOMETRIC_RATIO = 2.56 # 氧燃质量比


# ============================================================
# CORDIC算法 - 高精度三角函数计算
# ============================================================
# 原项目: 219_cordic
# 应用于火箭发动机喷注角度、燃烧室锥角等几何参数的高精度计算
# CORDIC (COordinate Rotation DIgital Computer) 算法
# 通过一系列微旋转逼近目标角度，仅使用移位和加法

def cordic_angles_table():
    """
    生成CORDIC角度查找表: arctan(2^{-k}), k=0,1,2,...
    
    理论依据:
    每次迭代旋转角度 θ_k = arctan(2^{-k})
    旋转矩阵: R_k = [1, -σ_k·2^{-k}; σ_k·2^{-k}, 1]
    其中 σ_k ∈ {+1, -1} 决定旋转方向
    
    收敛性证明:
    对于任意 |θ| ≤ π/2，存在序列{σ_k}使得
    θ = Σ σ_k·arctan(2^{-k}) + 残差
    当k→∞时残差趋于0
    """
    angles = np.zeros(60)
    for k in range(60):
        angles[k] = np.arctan(2.0 ** (-k))
    return angles


def cordic_kprod_table():
    """
    生成CORDIC增益补偿因子查找表。
    
    K_n = Π_{i=0}^{n-1} 1/√(1 + 2^{-2i})
    
    由于每次旋转矩阵的行列式为 (1 + 2^{-2k})，
    向量长度会增长，需要累积补偿。
    当 n→∞ 时, K_∞ ≈ 0.60725293500888125617
    """
    kprod = np.zeros(33)
    k_running = 1.0
    for k in range(33):
        k_running *= 1.0 / np.sqrt(1.0 + (2.0 ** (-2 * k)))
        kprod[k] = k_running
    return kprod


_CORDIC_ANGLES = cordic_angles_table()
_CORDIC_KPROD = cordic_kprod_table()


def cordic_cos_sin(beta: float, n_iter: int = 40) -> tuple:
    """
    使用CORDIC算法计算 cos(β) 和 sin(β)。
    
    参数:
        beta: 角度, 弧度
        n_iter: CORDIC迭代次数, 默认40次保证双精度
    
    返回:
        (cos_beta, sin_beta)
    
    边界处理:
        - 自动将角度归约到 [-π, π]
        - 再归约到 [-π/2, π/2] 并记录象限符号
    
    工程应用:
        用于燃烧室喷注面板孔位角度的高精度计算，
        避免标准浮点三角函数在极端角度下的精度损失。
    """
    if not np.isfinite(beta):
        return np.nan, np.nan
    
    # 角度归约到 [-π, π]
    theta = beta % (2.0 * np.pi)
    if theta > np.pi:
        theta -= 2.0 * np.pi
    elif theta < -np.pi:
        theta += 2.0 * np.pi
    
    # 记录象限符号, 归约到 [-π/2, π/2]
    sign_factor = 1.0
    if theta < -0.5 * np.pi:
        theta += np.pi
        sign_factor = -1.0
    elif theta > 0.5 * np.pi:
        theta -= np.pi
        sign_factor = -1.0
    
    v = np.array([1.0, 0.0])
    poweroftwo = 1.0
    
    n_iter = max(1, min(n_iter, 60))
    
    for j in range(n_iter):
        sigma = 1.0 if theta >= 0.0 else -1.0
        factor = sigma * poweroftwo
        
        # 旋转矩阵乘法 (无乘法, 仅有移位和加减)
        v_new = np.array([
            v[0] - factor * v[1],
            factor * v[0] + v[1]
        ])
        v = v_new
        
        # 更新剩余角度
        angle = _CORDIC_ANGLES[j] if j < 60 else _CORDIC_ANGLES[59] / (2.0 ** (j - 59))
        theta -= sigma * angle
        poweroftwo *= 0.5
    
    # 增益补偿
    if n_iter > 0:
        idx = min(n_iter - 1, len(_CORDIC_KPROD) - 1)
        v = v * _CORDIC_KPROD[idx]
    
    v = sign_factor * v
    return float(v[0]), float(v[1])


def cordic_arctan2(y: float, x: float, n_iter: int = 40) -> float:
    """
    CORDIC算法计算 arctan2(y, x)，即向量(x,y)与x轴的夹角。
    
    工程应用:
        用于燃烧室截面周向位置的精确计算。
    """
    if x == 0.0 and y == 0.0:
        return 0.0
    
    # 向量模式CORDIC: 将向量旋转至x轴
    angle = 0.0
    poweroftwo = 1.0
    
    n_iter = max(1, min(n_iter, 60))
    
    for j in range(n_iter):
        if y > 0:
            sigma = 1.0
        else:
            sigma = -1.0
        
        factor = sigma * poweroftwo
        x_new = x + factor * y
        y_new = y - factor * x
        x, y = x_new, y_new
        
        angle += sigma * _CORDIC_ANGLES[j]
        poweroftwo *= 0.5
    
    return float(angle)


# ============================================================
# Gamma函数与特殊函数
# ============================================================
# 原项目: 179_circle_integrals
# 用于燃烧室圆形截面平均特性计算、统计分布

def gamma_function_half_integer(n: int) -> float:
    """
    计算半整数Gamma函数 Γ(n/2)。
    
    理论基础:
        Γ(1/2) = √π
        Γ(z+1) = z·Γ(z)  (递推关系)
        
    对于正整数k:
        Γ(k + 1/2) = (2k)!/(4^k·k!) · √π
        Γ(k) = (k-1)!
    
    工程应用:
        用于圆形截面燃烧室中液滴尺寸分布的统计矩计算，
        以及单位圆上多项式积分的解析计算。
    """
    if n <= 0:
        raise ValueError("n must be positive for gamma_function_half_integer")
    
    if n % 2 == 0:
        k = n // 2
        return float(factorial(k - 1))
    else:
        k = n // 2
        # Γ(k + 1/2) = (2k)! / (4^k * k!) * √π
        result = factorial(2 * k)
        result = result / (4.0 ** k * factorial(k))
        result = result * np.sqrt(np.pi)
        return float(result)


def circle_monomial_integral(e1: int, e2: int) -> float:
    """
    计算单位圆周上的单项式积分:
    
        I = ∮_{x^2+y^2=1} x^{e1} · y^{e2} ds
    
    解析解 (Davis & Rabinowitz, Methods of Numerical Integration, p.263):
    
        若 e1 或 e2 为奇数, I = 0
        否则:
            I = 2 · Γ((e1+1)/2) · Γ((e2+1)/2) / Γ((e1+e2+2)/2)
    
    参数:
        e1, e2: 非负整数指数
    
    工程应用:
        用于圆形燃烧室截面周向平均热流密度的矩计算。
    """
    if e1 < 0 or e2 < 0:
        raise ValueError("Exponents must be nonnegative")
    
    if (e1 % 2 == 1) or (e2 % 2 == 1):
        return 0.0
    
    # 使用Gamma函数的倍增公式和对数域计算避免溢出
    if e1 > 100 or e2 > 100:
        # 大指数情况使用对数Gamma
        log_I = np.log(2.0) + \
                _log_gamma((e1 + 1) * 0.5) + \
                _log_gamma((e2 + 1) * 0.5) - \
                _log_gamma((e1 + e2 + 2) * 0.5)
        return float(np.exp(log_I))
    
    integral = 2.0
    integral *= gamma_function_half_integer(e1 + 1)
    integral *= gamma_function_half_integer(e2 + 1)
    integral /= gamma_function_half_integer(e1 + e2 + 2)
    return float(integral)


def _log_gamma(z: float) -> float:
    """对数Gamma函数的Lanczos近似。"""
    if z <= 0:
        return np.inf
    # Lanczos系数
    p = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
         -176.61502916214059, 12.507343278686905, -0.13857109526572012,
         9.9843695780195716e-6, 1.5056327351493116e-7]
    x = z
    y = z
    if y < 0.5:
        return np.log(np.pi) - np.log(np.sin(np.pi * y)) - _log_gamma(1.0 - y)
    y -= 1.0
    a = 0.99999999999980993
    for i, pi in enumerate(p):
        a += pi / (y + i + 1)
    t = y + len(p) - 0.5
    return 0.5 * np.log(2.0 * np.pi) + np.log(a) + (y + 0.5) * np.log(t) - t


# ============================================================
# 数值稳定性工具
# ============================================================

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法，避免除以零。"""
    if abs(b) < 1e-300:
        return default
    return a / b


def robust_sqrt(x: float) -> float:
    """鲁棒平方根，处理负数输入。"""
    if x < 0:
        if x > -1e-14:
            return 0.0
        return np.nan
    return np.sqrt(x)


def check_finite_array(arr: np.ndarray, name: str = "array") -> None:
    """检查数组中是否存在非有限值。"""
    if not np.all(np.isfinite(arr)):
        bad_idx = np.where(~np.isfinite(arr))[0]
        raise ValueError(f"{name} contains non-finite values at indices: {bad_idx[:10]}")


# ============================================================
# 推进剂物性计算 (基于NASA CEA数据拟合)
# ============================================================

def combustion_temperature(pressure: float, mixture_ratio: float) -> float:
    """
    计算燃烧温度 T_c(P, r)。
    
    基于NASA CEA数据的简化模型:
        T_c = T_ad · (P/P_ref)^{0.03} · exp(-0.05·(r - r_st)^2 / r_st^2)
    
    参数:
        pressure: 压力, Pa
        mixture_ratio: 氧燃质量比
    
    返回:
        燃烧温度, K
    """
    if pressure <= 0:
        raise ValueError("Pressure must be positive")
    if mixture_ratio <= 0:
        raise ValueError("Mixture ratio must be positive")
    
    p_ratio = pressure / PRESSURE_CHAMBER
    r_dev = (mixture_ratio - STOICHIOMETRIC_RATIO) / STOICHIOMETRIC_RATIO
    
    T_c = T_ADIABATIC * (p_ratio ** 0.03) * np.exp(-0.05 * r_dev ** 2)
    return float(T_c)


def specific_impulse_ideal(pressure: float, expansion_ratio: float, gamma: float = GAMMA_COMBUSTION) -> float:
    """
    计算理想比冲 (Ideal Specific Impulse)。
    
    火箭方程:
        I_sp = (1/g_0) · √[2γ/(γ-1) · R·T_c · (1 - (P_e/P_c)^{(γ-1)/γ})]
    
    其中:
        g_0 = 9.80665 m/s^2 (标准重力)
        P_e/P_c = (1 + (γ-1)/2 · Ma_e^2)^{-γ/(γ-1)}
    
    参数:
        pressure: 燃烧室压力, Pa
        expansion_ratio: 喷管面积扩张比 A_e/A_t
        gamma: 比热比
    
    返回:
        比冲, s
    """
    g_0 = 9.80665
    R_specific = R_UNIVERSAL / 0.022  # 近似燃烧产物摩尔质量 ~22 g/mol
    T_c = combustion_temperature(pressure, STOICHIOMETRIC_RATIO)
    
    # 通过面积比反推出口马赫数 (等熵关系)
    # A/A* = (1/Ma) · [(2/(γ+1))(1+(γ-1)/2·Ma^2)]^{(γ+1)/(2(γ-1))}
    def area_ratio_func(Ma):
        if Ma <= 0:
            return 1e300
        term = (2.0 / (gamma + 1.0)) * (1.0 + (gamma - 1.0) / 2.0 * Ma ** 2)
        return (1.0 / Ma) * (term ** ((gamma + 1.0) / (2.0 * (gamma - 1.0))))
    
    # 二分法求马赫数
    Ma_low, Ma_high = 1.01, 20.0
    for _ in range(100):
        Ma_mid = 0.5 * (Ma_low + Ma_high)
        f_mid = area_ratio_func(Ma_mid)
        if f_mid > expansion_ratio:
            Ma_high = Ma_mid
        else:
            Ma_low = Ma_mid
        if Ma_high - Ma_low < 1e-8:
            break
    Ma_e = 0.5 * (Ma_low + Ma_high)
    
    # 计算压力比
    p_ratio = (1.0 + (gamma - 1.0) / 2.0 * Ma_e ** 2) ** (-gamma / (gamma - 1.0))
    
    # 计算比冲
    term = 1.0 - p_ratio ** ((gamma - 1.0) / gamma)
    v_e = np.sqrt(2.0 * gamma / (gamma - 1.0) * R_specific * T_c * term)
    I_sp = v_e / g_0
    
    return float(I_sp)


if __name__ == "__main__":
    # 自测试
    c, s = cordic_cos_sin(np.pi / 3.0)
    print(f"CORDIC cos(π/3)={c:.15f}, sin(π/3)={s:.15f}")
    print(f"NumPy  cos(π/3)={np.cos(np.pi/3):.15f}, sin(π/3)={np.sin(np.pi/3):.15f}")
    
    I = circle_monomial_integral(2, 2)
    print(f"Circle integral x^2·y^2 = {I:.10f}")
    
    print(f"Specific impulse = {specific_impulse_ideal(7e6, 20.0):.2f} s")
