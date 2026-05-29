"""
大气热力学核心模块 (Atmospheric Thermodynamics Core Module)

集成以下种子项目算法:
- 1270_toms443: Lambert W 函数，用于求解隐式饱和水汽压方程
- 1269_toms291: log-Gamma 函数，用于微物理统计分布
- 1427_zero_brent: Brent 法求根，用于确定 LCL/LFC/EL
- 807_nonlin_fixed_point: 不动点迭代，用于饱和调整

核心公式:
  Clausius-Clapeyron: es(T) = e0 * exp( Lv/Rv * (1/T0 - 1/T) )
  潜在温度: θ = T * (p0/p)^(R/cp)
  虚温: Tv = T * (1 + 0.608*q)
  CAPE = ∫_{z_LFC}^{z_EL} g * (Tv_parcel - Tv_env)/Tv_env dz
  CIN  = ∫_{0}^{z_LFC} g * (Tv_parcel - Tv_env)/Tv_env dz
"""

import numpy as np
from typing import Tuple, Callable

# ── 物理常数 ───────────────────────────────────────────────
_Rd = 287.05       # J/(kg·K), 干空气气体常数
_Rv = 461.51       # J/(kg·K), 水汽气体常数
_cp = 1004.0       # J/(kg·K), 干空气定压比热
_Lv = 2.501e6      # J/kg, 水汽化潜热
_g  = 9.80665      # m/s², 重力加速度
_p0 = 100000.0     # Pa, 参考气压
_T0 = 273.15       # K, 参考温度
_e0 = 611.2        # Pa, 参考饱和水汽压
_epsilon = _Rd / _Rv  # 0.622


# ═══════════════════════════════════════════════════════════
#  1. 特殊函数: Lambert W 与 log-Gamma
# ═══════════════════════════════════════════════════════════

def lambert_w(x: float, tol: float = 1e-12, max_iter: int = 50) -> float:
    """
    主分支 W_0(x) 的 Halley 迭代实现 (基于 toms443 思想).
    求解 W*exp(W) = x.
    用于 Clausius-Clapeyron 方程的隐式反解.
    """
    if x < -1.0 / np.e:
        return np.nan
    if x == 0.0:
        return 0.0
    if x < -0.35:
        # 靠近分支点展开
        p = np.sqrt(2.0 * (np.e * x + 1.0))
        w = -1.0 + p - p**2 / 3.0 + 11.0 * p**3 / 72.0
    elif x < 0.5:
        w = x * (1.0 + x * (1.0 + x * (3.0 / 2.0))) / (1.0 + x * (5.0 / 2.0))
    elif x < 2.0:
        # 过渡区, 使用级数展开避免 log(log(x)) 奇点
        w = 0.5 + 0.5 * x / (x + np.e)
    elif x < 10.0:
        lx = np.log(x)
        if lx > 0.01:
            w = lx - np.log(lx) + np.log(lx) / lx
        else:
            w = lx
    else:
        L1 = np.log(x)
        L2 = np.log(L1)
        w = L1 - L2 + L2 / L1 + L2 * (L2 - 2.0) / (2.0 * L1**2)

    for _ in range(max_iter):
        ew = np.exp(w)
        we = w * ew - x
        w1 = w + 1.0
        dw = we / (ew * w1 - (w + 2.0) * we / (2.0 * w1))
        w -= dw
        if abs(dw) < tol * (abs(w) + 1.0):
            break
    return w


def log_gamma(x: float) -> float:
    """
    log Γ(x) 的计算 (基于 toms291 / Pike-Hill 算法).
    用于微物理中的 Gamma 分布对数计算.
    """
    if x <= 0.0:
        return np.nan
    if x < 7.0:
        # 递升变换: Γ(x) = Γ(x+n) / [x(x+1)...(x+n-1)]
        f = 1.0
        y = x
        while y < 7.0:
            f *= y
            y += 1.0
        return log_gamma(y) - np.log(f)
    # Stirling 渐近展开
    z = 1.0 / x**2
    s = (1.0 / 12.0 - z * (1.0 / 360.0 - z * (1.0 / 1260.0 - z * (1.0 / 1680.0)))) / x
    return (x - 0.5) * np.log(x) - x + 0.5 * np.log(2.0 * np.pi) + s


# ═══════════════════════════════════════════════════════════
#  2. 饱和水汽压与逆问题
# ═══════════════════════════════════════════════════════════

def saturation_vapor_pressure(temperature: float) -> float:
    """
    基于 Clausius-Clapeyron 方程的饱和水汽压 (Pa).
    es(T) = e0 * exp[ Lv/Rv * (1/T0 - 1/T) ]
    """
    if temperature <= 0.0:
        return 0.0
    return _e0 * np.exp((_Lv / _Rv) * (1.0 / _T0 - 1.0 / temperature))


def saturation_vapor_pressure_lambert(temperature: float) -> float:
    """
    使用 Lambert W 函数的饱和水汽压替代形式 (用于隐式求解验证).
    从 es = e0 * exp(a - b/T) 导出, 其中 a = Lv/(Rv*T0), b = Lv/Rv.
    """
    a = _Lv / (_Rv * _T0)
    b = _Lv / _Rv
    if temperature <= 0.0:
        return 0.0
    arg = (a - b / temperature) * np.exp(a - b / temperature)
    if arg < -1.0 / np.e:
        return 0.0
    w = lambert_w(arg)
    return _e0 * w / (a - b / temperature) if abs(a - b / temperature) > 1e-12 else saturation_vapor_pressure(temperature)


def dewpoint_from_vapor_pressure(e: float) -> float:
    """
    由水汽压反求露点温度 (K), 使用 Lambert W 解析反演.
    从 e = e0 * exp[ Lv/Rv * (1/T0 - 1/Td) ] 解出 Td:
    Td = b / (a - W( a * e/e0 * exp(-a) * e^a ) ), 这里使用数值 Newton 迭代保证鲁棒.
    """
    if e <= 0.0:
        return _T0 - 50.0
    if e >= saturation_vapor_pressure(350.0):
        return 350.0
    a = _Lv / (_Rv * _T0)
    b = _Lv / _Rv
    # 解析近似
    Td_approx = b / (a - np.log(e / _e0))
    # 小修正 Newton
    for _ in range(5):
        f = saturation_vapor_pressure(Td_approx) - e
        df = _Lv / (_Rv * Td_approx**2) * saturation_vapor_pressure(Td_approx)
        if abs(df) < 1e-20:
            break
        dT = f / df
        Td_approx -= dT
        if abs(dT) < 1e-4:
            break
    return max(150.0, Td_approx)


# ═══════════════════════════════════════════════════════════
#  3. 潜在温度、虚温、比湿
# ═══════════════════════════════════════════════════════════

def potential_temperature(temperature: float, pressure: float) -> float:
    """
    潜在温度 (K): θ = T * (p0/p)^(R/cp)
    """
    if pressure <= 0.0:
        return temperature
    return temperature * (_p0 / pressure)**(_Rd / _cp)


def virtual_temperature(temperature, qv):
    """
    虚温 (K): Tv = T * (1 + 0.608*q)
    """
    qv_safe = np.asarray(qv)
    qv_safe = np.where(qv_safe > 0.0, qv_safe, 0.0)
    return np.asarray(temperature) * (1.0 + 0.608 * qv_safe)


def mixing_ratio_to_specific_humidity(w: float) -> float:
    """混合比转比湿: q = w/(1+w)"""
    return w / (1.0 + w)


def specific_humidity_from_t_rh(temperature: float, rh: float, pressure: float) -> float:
    """
    由温度、相对湿度、气压求比湿 (kg/kg).
    q = ε*es/(p - (1-ε)*es) * RH
    """
    es = saturation_vapor_pressure(temperature)
    q = _epsilon * es / (pressure - (1.0 - _epsilon) * es) * rh
    return max(0.0, min(q, 0.05))  # 边界截断


# ═══════════════════════════════════════════════════════════
#  4. Brent 求根法 (zero_brent) — 用于 LCL/LFC/EL
# ═══════════════════════════════════════════════════════════

def brent_zero(f: Callable[[float], float], a: float, b: float,
               tol: float = 1e-8, max_iter: int = 100) -> float:
    """
    Brent 法求 f(x)=0 在 [a,b] 内的根 (基于 1427_zero_brent).
    要求 f(a) 与 f(b) 异号.
    """
    fa = f(a)
    fb = f(b)
    if fa * fb > 0.0:
        # 边界条件处理: 尝试扩展区间
        for scale in [2.0, 5.0, 10.0, 50.0, 100.0]:
            mid = (a + b) / 2.0
            new_a = mid - scale * (b - a)
            new_b = mid + scale * (b - a)
            fa_new = f(new_a)
            fb_new = f(new_b)
            if fa_new * fb_new <= 0.0:
                a, b = new_a, new_b
                fa, fb = fa_new, fb_new
                break
        else:
            return np.nan

    c, fc = a, fa
    s = b
    for _ in range(max_iter):
        if fb * fc > 0.0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, fa = b, fb
            b, fb = c, fc
            c, fc = a, fa
        tol_act = 2.0 * np.finfo(float).eps * abs(b) + 0.5 * tol
        m = 0.5 * (c - b)
        if abs(m) <= tol_act or fb == 0.0:
            return b
        if abs(e) < tol_act or abs(fa) <= abs(fb):
            d = e = m
        else:
            s = fb / fa
            if a == c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            min1 = 3.0 * m * q - abs(tol_act * q)
            min2 = abs(e * q)
            if 2.0 * p < (min1 if min1 < min2 else min2):
                d = e = p / q
            else:
                d = e = m
        a = b
        fa = fb
        if abs(d) > tol_act:
            b += d
        else:
            b += tol_act if m > 0.0 else -tol_act
        fb = f(b)
    return b


# ═══════════════════════════════════════════════════════════
#  5. 不动点迭代 (nonlin_fixed_point) — 饱和调整
# ═══════════════════════════════════════════════════════════

def fixed_point_iteration(g: Callable[[float], float], x0: float,
                          tol: float = 1e-10, max_iter: int = 100) -> float:
    """
    不动点迭代 x_{k+1} = g(x_k), 带发散保护 (基于 807_nonlin_fixed_point).
    """
    x = x0
    for k in range(max_iter):
        try:
            x_new = g(x)
        except (OverflowError, ValueError, ZeroDivisionError):
            return x
        if not np.isfinite(x_new):
            # 边界拉回
            x_new = x * 0.5 + x0 * 0.5
        if abs(x_new - x) < tol * (abs(x) + 1.0):
            return x_new
        # Aitken Δ² 加速
        if k >= 2:
            x_new = x_new - (x_new - x)**2 / (x_new - 2.0 * x + x_prev2 + 1e-30)
        x_prev2 = x
        x = x_new
    return x


def saturation_adjustment(temperature: float, q_total: float, pressure: float,
                          dt: float = 1.0) -> Tuple[float, float]:
    """
    饱和调整: 给定总水比湿 q_total 和温度 T,
    通过不动点迭代求平衡态 (T, qv, ql).
    返回 (T_new, qv_new).

    热力学约束:
      q_total = qv + ql
      qv = q_sat(T) = ε*es(T)/[p-(1-ε)*es(T)]
      T_new = T + Lv/cp * (qv - q_total)   (若 q_total > q_sat)
    """
    def qsat(Tk):
        es = saturation_vapor_pressure(Tk)
        return _epsilon * es / (pressure - (1.0 - _epsilon) * es)

    qv = min(q_total, qsat(temperature))
    # 不动点映射: T = T_old + Lv/cp * (qsat(T) - q_total)
    def g(Tk):
        qs = qsat(Tk)
        if qs >= q_total:
            return temperature  # 未饱和, 保持原温
        return temperature + (_Lv / _cp) * (qs - q_total)

    if qsat(temperature) >= q_total:
        return temperature, q_total

    T_new = fixed_point_iteration(g, temperature, tol=1e-6, max_iter=50)
    qv_new = min(qsat(T_new), q_total)
    # 边界保护
    T_new = max(150.0, min(T_new, 350.0))
    qv_new = max(0.0, min(qv_new, q_total))
    return T_new, qv_new


# ═══════════════════════════════════════════════════════════
#  6. LCL, LFC, EL 与 CAPE/CIN 计算
# ═══════════════════════════════════════════════════════════

def lifting_condensation_level(p_sfc: float, T_sfc: float, qv_sfc: float) -> Tuple[float, float]:
    """
    计算抬升凝结高度 LCL (基于 Bolton 1980 公式).
    返回 (p_lcl, T_lcl).
    """
    Td = dewpoint_from_vapor_pressure(
        qv_sfc * p_sfc / (_epsilon + (1.0 - _epsilon) * qv_sfc)
    )
    # Bolton LCL 近似
    TL = 1.0 / (1.0 / (Td - 56.0) + np.log(T_sfc / Td) / 800.0) + 56.0
    p_lcl = p_sfc * (TL / T_sfc)**(1.0 / (_Rd / _cp))
    return max(10000.0, p_lcl), max(200.0, TL)


def parcel_temperature(pressure: float, p_sfc: float, T_sfc: float, qv_sfc: float) -> float:
    """
    湿绝热抬升气块温度 (K), 使用迭代法.
    从 sfc 沿 moist adiabat 抬升到 pressure.
    """
    p_lcl, T_lcl = lifting_condensation_level(p_sfc, T_sfc, qv_sfc)
    if pressure >= p_lcl:
        # 干绝热
        return T_sfc * (pressure / p_sfc)**(_Rd / _cp)
    # 湿绝热 (伪绝热近似)
    T = T_lcl
    p = p_lcl
    dp = -5000.0  # Pa, 向上步长
    while p > pressure and p > 10000.0:
        p_next = max(pressure, p + dp)
        es = saturation_vapor_pressure(T)
        qs = _epsilon * es / (p - (1.0 - _epsilon) * es)
        dT_dp = (1.0 / p) * (_Rd * T + _Lv * qs) / (_cp + (_Lv**2) * qs * _epsilon / (_Rv * T**2))
        T += dT_dp * (p_next - p)
        p = p_next
    return max(150.0, T)


def compute_cape_cin(pressure_levels: np.ndarray, T_env: np.ndarray,
                     qv_env: np.ndarray, p_sfc: float, T_sfc: float,
                     qv_sfc: float) -> Tuple[float, float, float, float, float]:
    """
    计算 CAPE, CIN, LCL, LFC, EL.
    返回 (CAPE, CIN, p_LCL, p_LFC, p_EL).
    """
    n = len(pressure_levels)
    # 虚温环境廓线
    Tv_env = virtual_temperature(T_env, qv_env)
    # 气块虚温
    T_parcel = np.array([parcel_temperature(p, p_sfc, T_sfc, qv_sfc) for p in pressure_levels])
    # 假设气块比湿守恒到 LCL, 之后为饱和比湿
    qv_parcel = np.zeros(n)
    p_lcl, _ = lifting_condensation_level(p_sfc, T_sfc, qv_sfc)
    for i, p in enumerate(pressure_levels):
        if p >= p_lcl:
            qv_parcel[i] = qv_sfc
        else:
            qv_parcel[i] = _epsilon * saturation_vapor_pressure(T_parcel[i]) / (
                p - (1.0 - _epsilon) * saturation_vapor_pressure(T_parcel[i])
            )
    Tv_parcel = virtual_temperature(T_parcel, qv_parcel)

    # === HOLE 1 START ===
    # 任务: 实现浮力计算、LFC/EL 定位、CAPE/CIN 数值积分
    # 科学背景:
    #   1. 浮力公式: buoyancy = g * (Tv_parcel - Tv_env) / Tv_env
    #   2. LFC (自由对流高度): 浮力首次 > 0 且在 LCL 以上的气压层
    #   3. EL (平衡高度): LFC 以上浮力最后 > 0 之后再次 < 0 的气压层
    #   4. Brent 法精化: 在 buoyancy=0 处求精确零 crossing
    #   5. CAPE = ∫_{LFC}^{EL} buoyancy dz  (正值累加)
    #   6. CIN  = ∫_{0}^{LFC} buoyancy dz  (负值累加)
    #   7. 静力近似: dz = -Rd*Tv/g * dp/p
    # TODO: implement buoyancy, LFC/EL, and CAPE/CIN integration
    raise NotImplementedError("HOLE 1: compute_cape_cin 的 buoyancy/CAPE/CIN 计算尚未实现")
    # === HOLE 1 END ===
