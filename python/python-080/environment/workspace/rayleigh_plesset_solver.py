"""
rayleigh_plesset_solver.py
扩展 Rayleigh-Plesset 与 Keller-Miksis 气泡动力学求解器

核心物理模型:
1. Rayleigh-Plesset 方程（不可压缩液体）:
   R * d²R/dt² + (3/2) * (dR/dt)² = (p_g - p_∞)/ρ - 4ν*(dR/dt)/R - 2σ/(ρR)

2. Keller-Miksis 方程（可压缩液体修正）:
   (1 - dR/dt/c) * R * d²R/dt² + (3/2 - dR/dt/(2c)) * (dR/dt)²
   = (1 + dR/dt/c) * (p_g - p_∞)/ρ + (R/(ρc)) * d(p_g)/dt

3. 气泡内气体状态方程（Van der Waals）:
   (p_g + a_vdw * n_g²/V²) * (V - n_g*b_vdw) = n_g * R_g * T_g

4. 热传导修正（Plesset-Zwick 近似）:
   dT_g/dt = - (3(γ-1)/R) * (dR/dt) * (T_g - T_∞) * sqrt(α_th/(πR²))

映射来源:
- 060_axon_ode: Hodgkin-Huxley ODE 框架 → 气泡壁运动 ODE 系统
- 1221_test_nonlin: 非线性方程测试问题 → 非线性耦合求解
- 095_bisection_integer: 整数二分搜索 → 临界空化核半径搜索
"""

import numpy as np
from scipy.integrate import solve_ivp
from utils import safe_divide, WATER_DENSITY, WATER_VISCOSITY, SURFACE_TENSION, SOUND_SPEED_WATER, VAPOR_PRESSURE, ATMOSPHERIC_PRESSURE

# =====================================================================
# 物理参数
# =====================================================================
GAMMA_ADIABATIC = 1.4       # 绝热指数
THERMAL_DIFFUSIVITY = 1.43e-7  # 热扩散系数 [m²/s]
VAN_DER_WAALS_A = 0.00365   # Van der Waals 常数 a [Pa·m⁶/mol²]
VAN_DER_WAALS_B = 4.27e-5   # Van der Waals 常数 b [m³/mol]
GAS_CONSTANT_MOLAR = 8.314  # [J/(mol·K)]
AMBIENT_TEMPERATURE = 293.15  # [K]


def van_der_waals_pressure(n_g, V, T_g):
    """
    Van der Waals 状态方程求气体压力。
    p = nRT/(V - nb) - an²/V²

    TODO: 实现 Van der Waals 状态方程
    """
    pass


def gas_pressure_adiabatic(R, R0, p_g0):
    """
    绝热气体压力: p_g = p_g0 * (R0/R)^(3γ)
    边界处理: R 接近零时返回极大值。
    """
    ratio = safe_divide(R0, R, default=1e6)
    return p_g0 * ratio ** (3.0 * GAMMA_ADIABATIC)


def plesset_zwick_heat_flux(R, dRdt, T_g):
    """
    Plesset-Zwick 热通量修正。
    q = - (3(γ-1)/R) * dR/dt * (T_g - T_∞) * sqrt(α_th / (π R²))
    """
    delta_T = T_g - AMBIENT_TEMPERATURE
    factor = -3.0 * (GAMMA_ADIABATIC - 1.0) / (R + 1e-15)
    factor *= dRdt * delta_T
    factor *= np.sqrt(THERMAL_DIFFUSIVITY / (np.pi * (R**2 + 1e-30)))
    return factor


def rayleigh_plesset_rhs(t, y, p_inf, sigma, rho, mu, R0, p_g0):
    """
    Rayleigh-Plesset 方程的一阶 ODE 系统。
    y = [R, dR/dt, T_g, n_g]
    dy/dt = [dR/dt, d²R/dt², dT_g/dt, dn_g/dt]

    对应 060_axon_ode 的 Hodgkin-Huxley ODE 框架，迁移到气泡壁运动。
    """
    R, dRdt, T_g, n_g = y
    if R <= 0:
        R = 1e-9

    V = (4.0 / 3.0) * np.pi * R**3
    p_g = van_der_waals_pressure(n_g, V, T_g)

    # Rayleigh-Plesset 方程右端项
    term1 = safe_divide(p_g - p_inf, rho)
    term2 = -4.0 * mu * dRdt / (rho * R)
    term3 = -2.0 * sigma / (rho * R)
    d2Rdt2 = safe_divide(term1 + term2 + term3 - 1.5 * dRdt**2, R)

    # 温度变化（热传导）
    dTgdt = plesset_zwick_heat_flux(R, dRdt, T_g)

    # 气体摩尔数守恒（简化模型：扩散项）
    D_gas = 2.0e-9  # 气体扩散系数 [m²/s]
    dn_gdt = -4.0 * np.pi * R**2 * D_gas * (p_g - p_g0) / (GAS_CONSTANT_MOLAR * T_g * R0 + 1e-30)

    return [dRdt, d2Rdt2, dTgdt, dn_gdt]


def keller_miksis_rhs(t, y, p_inf, sigma, rho, mu, c, R0, p_g0):
    """
    Keller-Miksis 可压缩修正方程的一阶 ODE 系统。
    y = [R, dR/dt, T_g, n_g]

    (1 - dR/dt/c) * R * d²R/dt² + (3/2 - dR/dt/(2c)) * (dR/dt)²
    = (1 + dR/dt/c) * (p_g - p_∞)/ρ + (R/(ρc)) * d(p_g)/dt
    """
    R, dRdt, T_g, n_g = y
    if R <= 0:
        R = 1e-9

    V = (4.0 / 3.0) * np.pi * R**3
    p_g = van_der_waals_pressure(n_g, V, T_g)

    # d(p_g)/dt 的数值近似
    dp_g_dt = 0.0  # 简化处理，在实际迭代中可由前一步值修正

    lhs_coeff = 1.0 - dRdt / (c + 1e-15)
    rhs_term1 = (1.0 + dRdt / (c + 1e-15)) * (p_g - p_inf) / rho
    rhs_term2 = (R / (rho * c + 1e-30)) * dp_g_dt
    rhs_total = rhs_term1 + rhs_term2
    nonlinear_term = (1.5 - dRdt / (2.0 * c + 1e-15)) * dRdt**2

    d2Rdt2 = safe_divide(rhs_total - nonlinear_term, lhs_coeff * R)

    dTgdt = plesset_zwick_heat_flux(R, dRdt, T_g)
    D_gas = 2.0e-9
    dn_gdt = -4.0 * np.pi * R**2 * D_gas * (p_g - p_g0) / (GAS_CONSTANT_MOLAR * T_g * R0 + 1e-30)

    return [dRdt, d2Rdt2, dTgdt, dn_gdt]


def solve_rayleigh_plesset(R0, p_g0, p_inf, t_span, method='RK45', use_keller_miksis=True):
    """
    求解扩展 Rayleigh-Plesset / Keller-Miksis 方程。

    参数:
        R0: 初始气泡半径 [m]
        p_g0: 初始气泡内气体压力 [Pa]
        p_inf: 远场液体压力 [Pa]
        t_span: 时间区间 [t0, tf] [s]
        method: 积分方法
        use_keller_miksis: 是否使用 Keller-Miksis 修正
    返回:
        sol: scipy.integrate.solve_ivp 解对象
    """
    y0 = [R0, 0.0, AMBIENT_TEMPERATURE, p_g0 * (4.0/3.0)*np.pi*R0**3 / (GAS_CONSTANT_MOLAR * AMBIENT_TEMPERATURE)]
    rho = WATER_DENSITY
    mu = WATER_VISCOSITY
    sigma = SURFACE_TENSION
    c = SOUND_SPEED_WATER

    if use_keller_miksis:
        rhs = lambda t, y: keller_miksis_rhs(t, y, p_inf, sigma, rho, mu, c, R0, p_g0)
    else:
        rhs = lambda t, y: rayleigh_plesset_rhs(t, y, p_inf, sigma, rho, mu, R0, p_g0)

    sol = solve_ivp(rhs, t_span, y0, method=method, dense_output=True, max_step=t_span[1]/1000.0)
    return sol


def critical_nucleation_radius_bisection(p_inf, p_v, sigma, rho, R_min=1e-9, R_max=1e-3, tol=1e-12):
    """
    使用二分法搜索临界空化核半径。
    临界条件: p_g - p_∞ + 2σ/R = 0，即 Blake 临界半径。

    对应 095_bisection_integer 的二分搜索思想，迁移到连续域。
    Blake 临界半径满足:
        R_crit = sqrt( 2σ / (3(p_∞ - p_v)) )

    这里用数值二分法验证并求解更一般的方程:
        f(R) = p_v*(R0/R)^(3γ) - p_∞ + 2σ/R = 0
    """
    R0 = 1e-6  # 参考半径

    def f(R):
        if R <= 0:
            return -1e20
        p_g = p_v * (R0 / R) ** (3.0 * GAMMA_ADIABATIC)
        return p_g - p_inf + 2.0 * sigma / R

    fa = f(R_min)
    fb = f(R_max)

    # 确保区间内有根
    if fa * fb > 0:
        # 解析解作为回退
        R_crit_analytical = np.sqrt(2.0 * sigma / (3.0 * max(p_inf - p_v, 1.0)))
        return R_crit_analytical

    # 二分法
    while abs(R_max - R_min) > tol:
        R_mid = (R_min + R_max) / 2.0
        fc = f(R_mid)
        if fc == 0:
            return R_mid
        if fa * fc < 0:
            R_max = R_mid
            fb = fc
        else:
            R_min = R_mid
            fa = fc

    return (R_min + R_max) / 2.0


def nonlinear_bubble_residue(y, p_inf, sigma, rho, mu, R0, p_g0):
    """
    构建气泡稳态的非线性残差函数 F(y) = 0。
    对应 1221_test_nonlin 的非线性测试问题框架。

    未知量 y = [R, dRdt, T_g, n_g]
    稳态条件:
      F1 = dR/dt = 0
      F2 = d²R/dt² = 0  → (p_g - p_∞)/ρ - 2σ/(ρR) = 0
      F3 = dT_g/dt = 0  → T_g = T_∞
      F4 = dn_g/dt = 0  → 气体扩散平衡
    """
    R, dRdt, T_g, n_g = y
    V = (4.0 / 3.0) * np.pi * R**3
    p_g = van_der_waals_pressure(n_g, V, T_g)

    F1 = dRdt
    F2 = safe_divide(p_g - p_inf, rho) - 2.0 * sigma / (rho * R)
    F3 = T_g - AMBIENT_TEMPERATURE
    F4 = p_g - p_g0  # 简化扩散平衡条件

    return np.array([F1, F2, F3, F4], dtype=float)


def nonlinear_bubble_jacobian(y, p_inf, sigma, rho, mu, R0, p_g0):
    """
    计算非线性残差函数的 Jacobian 矩阵（数值差分）。
    对应 1221_test_nonlin 的 Jacobian 测试问题。
    """
    n = len(y)
    J = np.zeros((n, n), dtype=float)
    h = 1e-8
    F0 = nonlinear_bubble_residue(y, p_inf, sigma, rho, mu, R0, p_g0)
    for j in range(n):
        yp = y.copy()
        yp[j] += h
        Fp = nonlinear_bubble_residue(yp, p_inf, sigma, rho, mu, R0, p_g0)
        J[:, j] = (Fp - F0) / h
    return J


def solve_steady_state_newton(p_inf, sigma, rho, mu, R0, p_g0, max_iter=50, tol=1e-12):
    """
    使用 Newton 法求解气泡稳态。
    y_{k+1} = y_k - J(y_k)^{-1} F(y_k)
    """
    y = np.array([R0, 0.0, AMBIENT_TEMPERATURE, p_g0 * (4.0/3.0)*np.pi*R0**3 / (GAS_CONSTANT_MOLAR * AMBIENT_TEMPERATURE)], dtype=float)

    for k in range(max_iter):
        F = nonlinear_bubble_residue(y, p_inf, sigma, rho, mu, R0, p_g0)
        if np.linalg.norm(F) < tol:
            break
        J = nonlinear_bubble_jacobian(y, p_inf, sigma, rho, mu, R0, p_g0)
        try:
            dy = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            # 使用伪逆处理奇异 Jacobian
            dy = -np.linalg.lstsq(J, F, rcond=None)[0]
        y = y + dy
        # 边界保护
        y[0] = max(y[0], 1e-9)
        y[2] = max(y[2], 1.0)
        y[3] = max(y[3], 1e-20)
    return y
