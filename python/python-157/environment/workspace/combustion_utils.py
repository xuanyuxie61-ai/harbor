"""
combustion_utils.py
通用工具与物理常数模块
融合来源：多项目参数管理思想（861_pendulum_nonlinear_ode, 315_double_well_ode）
"""
import numpy as np

# 物理常数
R_UNIVERSAL = 8.314462618  # J/(mol·K)，通用气体常数
ATM_PA = 101325.0          # Pa，标准大气压

# 爆轰相关默认参数
DEFAULT_GAMMA = 1.4        # 比热比
DEFAULT_Q = 2.5e6          # J/kg，单位质量化学反应释热
DEFAULT_E_A = 8.314e4      # J/mol，活化能
DEFAULT_A_PRE = 1.0e8      # 1/s，指前因子
DEFAULT_T_IGN = 1500.0     # K，点火温度阈值
DEFAULT_RHO_0 = 1.225      # kg/m^3，未燃气体密度
DEFAULT_P_0 = 101325.0     # Pa，未燃气体压强
DEFAULT_T_0 = 300.0        # K，未燃气体温度
DEFAULT_W_MOL = 0.029      # kg/mol，摩尔质量


def check_positive(value, name):
    """边界检查：确保值为正数"""
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive finite, got {value}")
    return value


def check_nonnegative(value, name):
    """边界检查：确保值为非负数"""
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be nonnegative finite, got {value}")
    return value


def check_interval(a, b, name_a="a", name_b="b"):
    """边界检查：确保 a < b"""
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError(f"Interval endpoints must be finite: {name_a}={a}, {name_b}={b}")
    if b <= a:
        raise ValueError(f"Require {name_a} < {name_b}, got {a} >= {b}")
    return a, b


def arrhenius_rate(T, A, Ea, R=R_UNIVERSAL):
    r"""
    Arrhenius 反应速率:
        k = A * exp(-Ea / (R * T))
    输入:
        T : 温度 [K]
        A : 指前因子
        Ea: 活化能 [J/mol]
        R : 通用气体常数
    输出:
        k : 反应速率常数
    """
    # TODO: 实现 Arrhenius 反应速率公式 k = A * exp(-Ea / (R * T))
    # 注意边界检查与指数溢出防护
    raise NotImplementedError("Hole_1: 请实现 Arrhenius 反应速率公式")


def specific_heat_ratio_cv_cp(gamma):
    r"""
    由比热比 gamma = Cp/Cv 计算定容比热:
        Cv = R / (gamma - 1)
        Cp = gamma * R / (gamma - 1)
    """
    check_positive(gamma, "gamma")
    if gamma <= 1.0:
        raise ValueError("gamma must be > 1 for ideal gas")
    cv = R_UNIVERSAL / (gamma - 1.0)
    cp = gamma * cv
    return cv, cp


def sound_speed(T, gamma, W_mol=DEFAULT_W_MOL):
    r"""
    理想气体声速:
        a = sqrt(gamma * R * T / W_mol)
    """
    check_positive(T, "Temperature T")
    check_positive(gamma, "gamma")
    check_positive(W_mol, "Molar mass W_mol")
    return np.sqrt(gamma * R_UNIVERSAL * T / W_mol)


def znd_progress_variable_derivative(lambda_var, T, A, Ea, n_order=1.0, R=R_UNIVERSAL):
    r"""
    ZND 模型中反应进度变量演化:
        dλ/dt = -k * (1 - λ)^n,   k = A * exp(-Ea/(R*T))
    这里 λ ∈ [0, 1]，0 表示未反应，1 表示完全反应。
    返回 dλ/dt（负值表示反应消耗未燃物）。
    """
    check_nonnegative(lambda_var, "lambda")
    if lambda_var > 1.0:
        lambda_var = 1.0
    check_positive(T, "Temperature T")
    k = arrhenius_rate(T, A, Ea, R)
    remain = max(1.0 - lambda_var, 0.0)
    if remain <= 0.0:
        return 0.0
    return -k * (remain ** n_order)


def rankine_hugoniot_pressure_ratio(M, gamma):
    r"""
    Rankine-Hugoniot 关系：激波前后压强比
        p2/p1 = 1 + 2*gamma/(gamma+1) * (M^2 - 1)
    """
    check_positive(M, "Mach number M")
    check_positive(gamma, "gamma")
    return 1.0 + 2.0 * gamma / (gamma + 1.0) * (M * M - 1.0)


def rankine_hugoniot_density_ratio(M, gamma):
    r"""
    Rankine-Hugoniot 关系：激波前后密度比
        rho2/rho1 = (gamma+1) * M^2 / ((gamma-1) * M^2 + 2)
    """
    check_positive(M, "Mach number M")
    check_positive(gamma, "gamma")
    return (gamma + 1.0) * M * M / ((gamma - 1.0) * M * M + 2.0)


def cj_detonation_velocity(gamma, Q, p0, rho0):
    r"""
    Chapman-Jouguet (CJ) 爆轰速度（简化模型）:
        D_CJ^2 = 2 * (gamma^2 - 1) * Q + (gamma * p0 / rho0)
    更精确地，对理想气体:
        D_CJ ≈ sqrt(2 * (gamma^2 - 1) * Q + a0^2)
    其中 a0 = sqrt(gamma * p0 / rho0) 为未燃气声速。
    """
    check_positive(gamma, "gamma")
    check_positive(Q, "Heat release Q")
    check_positive(p0, "Pressure p0")
    check_positive(rho0, "Density rho0")
    a0_sq = gamma * p0 / rho0
    D_cj_sq = 2.0 * (gamma * gamma - 1.0) * Q + a0_sq
    if D_cj_sq <= 0.0:
        raise ValueError("CJ detonation velocity squared is non-positive")
    return np.sqrt(D_cj_sq)


def von_neumann_spike_conditions(D, gamma, p0, rho0):
    r"""
    Von Neumann 尖峰条件（强爆轰近似）:
        将 CJ 速度 D 视为激波速度，计算激波后（Von Neumann 状态）参数:
        p_vN = p0 * (1 + 2*gamma/(gamma+1) * (M^2 - 1))
        rho_vN = rho0 * (gamma+1) * M^2 / ((gamma-1)*M^2 + 2)
        T_vN = p_vN / (rho_vN * R_specific)
    其中 M = D / a0 为激波马赫数。
    """
    check_positive(D, "Detonation velocity D")
    a0 = sound_speed_from_prho(p0, rho0, gamma)
    M = D / a0
    p_ratio = rankine_hugoniot_pressure_ratio(M, gamma)
    rho_ratio = rankine_hugoniot_density_ratio(M, gamma)
    p_vn = p0 * p_ratio
    rho_vn = rho0 * rho_ratio
    T_vn = p_vn / (rho_vn * (R_UNIVERSAL / DEFAULT_W_MOL))
    return p_vn, rho_vn, T_vn, M


def sound_speed_from_prho(p, rho, gamma):
    r"""
    由状态方程计算声速:
        a = sqrt(gamma * p / rho)
    """
    check_positive(p, "Pressure p")
    check_positive(rho, "Density rho")
    check_positive(gamma, "gamma")
    return np.sqrt(gamma * p / rho)


def temperature_from_energy(e, lambda_var, Q, cv):
    r"""
    由比内能与反应进度计算温度:
        e = cv * T + (1 - λ) * Q   =>   T = (e - (1-λ)*Q) / cv
    """
    check_positive(cv, "cv")
    T = (e - (1.0 - lambda_var) * Q) / cv
    if T < 0.0:
        # 数值鲁棒：允许极小负值时截断
        T = max(T, 1.0e-6)
    return T


def cholesky_factor(a):
    r"""
    对 2x2 对称正定矩阵进行 Cholesky 分解:
        A = L * L^T
    返回下三角矩阵 L。
    融合来源：331_ellipse_monte_carlo 的 r8po_fa 思想。
    """
    a = np.asarray(a, dtype=float)
    if a.shape != (2, 2):
        raise ValueError("Only 2x2 matrix supported")
    if not np.allclose(a, a.T):
        raise ValueError("Matrix must be symmetric")
    if a[0, 0] <= 0.0:
        raise ValueError("Matrix not positive definite")
    L = np.zeros((2, 2))
    L[0, 0] = np.sqrt(a[0, 0])
    L[1, 0] = a[1, 0] / L[0, 0]
    diag2 = a[1, 1] - L[1, 0] ** 2
    if diag2 <= 0.0:
        raise ValueError("Matrix not positive definite")
    L[1, 1] = np.sqrt(diag2)
    return L


def solve_lower_triangular(L, b):
    r"""
    解下三角方程组 L x = b。
    """
    x = np.zeros_like(b, dtype=float)
    x[0] = b[0] / L[0, 0]
    x[1] = (b[1] - L[1, 0] * x[0]) / L[1, 1]
    return x
