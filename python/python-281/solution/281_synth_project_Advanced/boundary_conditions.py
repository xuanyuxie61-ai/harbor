"""
boundary_conditions.py
======================
电化学边界条件模块。
融合项目: 892_polyiamonds (边界反射/旋转操作),
         1297_FormalCellular_SecureCofigurationSpace (形式化边界验证)

边界类型:
  1. 球心对称: ∂c/∂r|_{r=0} = 0 (Neumann)
  2. Butler-Volmer 表面反应: -D*∂c/∂r|_{r=R} = j/(n*F) (非线性 Robin)
  3. 恒流充放电: j = I/(A) = const
  4. 恒压充电: j 由 OCV - V_applied 决定

关键公式:
  Butler-Volmer 方程:
  j = j₀ * [exp(αₐ*F*η/(R*T)) - exp(-α_c*F*η/(R*T))]
  η = Φ_s - Φ_e - U(c_s)  (过电位)

  表面通量边界:
  -D*∂c/∂r|_{r=R} = j/(F)  (单电子反应)
"""

import math
import numpy as np
from electrode_constants import (
    FARADAY, R_GAS, T_REF, ALPHA_ANODE, ALPHA_CATHODE,
    I0_EXCHANGE, C_MAX, PARTICLE_RADIUS
)
from thermodynamic_models import redlich_kister_ocv


def butler_volmer_current(eta, j0=I0_EXCHANGE, T=T_REF,
                           alpha_a=ALPHA_ANODE, alpha_c=ALPHA_CATHODE):
    """
    Butler-Volmer 电化学反应电流密度。

    j = j₀ * [exp(αₐ*F*η/(R*T)) - exp(-α_c*F*η/(R*T))]

    其中:
    - j₀: 交换电流密度 [A/m²]
    - η: 过电位 [V], η = Φ_s - Φ_e - U(c_s)
    - αₐ, α_c: 阳极/阴极传递系数
    - F, R: Faraday 常数和气体常数
    - T: 温度 [K]

    数值安全: 指数参数限制在 [-500, 500]

    Parameters
    ----------
    eta : float or ndarray
        过电位 [V]
    j0 : float
        交换电流密度 [A/m²]
    T : float
        温度 [K]
    alpha_a, alpha_c : float
        传递系数

    Returns
    -------
    float or ndarray
        电流密度 [A/m²]
    """
    V_T = R_GAS * T / FARADAY  # 热电压

    if isinstance(eta, np.ndarray):
        arg_a = np.clip(alpha_a * eta / V_T, -500, 500)
        arg_c = np.clip(-alpha_c * eta / V_T, -500, 500)
        return j0 * (np.exp(arg_a) - np.exp(arg_c))
    else:
        arg_a = max(min(alpha_a * eta / V_T, 500.0), -500.0)
        arg_c = max(min(-alpha_c * eta / V_T, 500.0), -500.0)
        return j0 * (math.exp(arg_a) - math.exp(arg_c))


def butler_volmer_derivative(eta, j0=I0_EXCHANGE, T=T_REF,
                               alpha_a=ALPHA_ANODE, alpha_c=ALPHA_CATHODE):
    """
    Butler-Volmer 电流对过电位的导数: ∂j/∂η。

    ∂j/∂η = j₀ * [αₐ*F/(R*T)*exp(αₐ*F*η/(R*T))
                   + α_c*F/(R*T)*exp(-α_c*F*η/(R*T))]

    用于 Robin 边界条件的线性化。

    Parameters
    ----------
    eta : float
        过电位 [V]
    j0 : float
        交换电流密度
    T : float
        温度 [K]

    Returns
    -------
    float
        ∂j/∂η [A/(m²·V)]
    """
    V_T = R_GAS * T / FARADAY
    arg_a = max(min(alpha_a * eta / V_T, 500.0), -500.0)
    arg_c = max(min(-alpha_c * eta / V_T, 500.0), -500.0)

    dj_deta = j0 * (alpha_a / V_T * math.exp(arg_a)
                     + alpha_c / V_T * math.exp(arg_c))
    return dj_deta


def exchange_current_density(c_s, c_max=C_MAX, j0_ref=I0_EXCHANGE,
                               c_e=1000.0, c_e_ref=1000.0):
    """
    交换电流密度的浓度依赖关系。

    j₀ = j₀_ref * (c_s/c_max)^{αₐ} * (1 - c_s/c_max)^{α_c}
                    * (c_e/c_e_ref)^{αₐ}

    这反映了 Li 嵌入/脱出反应中, 反应速率与表面浓度和空位浓度的关系。

    Parameters
    ----------
    c_s : float
        颗粒表面 Li 浓度 [mol/m³]
    c_max : float
        最大嵌入浓度 [mol/m³]
    j0_ref : float
        参考交换电流密度 [A/m²]
    c_e : float
        电解液 Li 浓度 [mol/m³]
    c_e_ref : float
        参考电解液浓度 [mol/m³]

    Returns
    -------
    float
        交换电流密度 [A/m²]
    """
    x_s = max(1e-10, min(1.0 - 1e-10, c_s / c_max))
    # 浓度依赖
    conc_factor = (x_s ** ALPHA_ANODE) * ((1.0 - x_s) ** ALPHA_CATHODE)
    # 电解液依赖
    elec_factor = (c_e / c_e_ref) ** ALPHA_ANODE

    return j0_ref * conc_factor * elec_factor


def overpotential(Phi_s, Phi_e, c_surface, T=T_REF):
    """
    计算过电位 η。

    η = Φ_s - Φ_e - U(c_surface)

    其中 U(c) 为 OCV (平衡电位)。

    Parameters
    ----------
    Phi_s : float
        固相电位 [V]
    Phi_e : float
        液相电位 [V]
    c_surface : float
        表面浓度 [mol/m³]
    T : float
        温度 [K]

    Returns
    -------
    float
        过电位 [V]
    """
    x_s = max(1e-12, min(1.0 - 1e-12, c_surface / C_MAX))
    U_eq = redlich_kister_ocv(x_s)
    return Phi_s - Phi_e - U_eq


def constant_current_boundary(I_app, particle_radius=PARTICLE_RADIUS):
    """
    恒流充放电的表面通量。

    j = I_app / (4*π*R²)  [A/m²]
    ∂c/∂r|_{r=R} = -j / (F * D)  [mol/m⁴]

    注意: I_app > 0 为充电 (Li 脱出), I_app < 0 为放电 (Li 嵌入)

    Parameters
    ----------
    I_app : float
        施加电流 [A]
    particle_radius : float
        颗粒半径 [m]

    Returns
    -------
    float
        表面通量密度 [A/m²]
    """
    area = 4.0 * math.pi * particle_radius * particle_radius
    if area < 1e-30:
        raise ValueError("颗粒表面积不能为零")
    return I_app / area


def surface_flux_boundary(c_surface, c_next, D_surface, h, r_surface,
                            Phi_s=4.0, Phi_e=0.0, T=T_REF, j0=None):
    """
    Butler-Volmer 表面通量边界条件的通量值。

    -D*∂c/∂r|_{r=R} = j_BV / F

    使用 ghost point 方法实现:
    c_ghost = c_surface + 2*h * j_BV / (F * D_surface)

    Parameters
    ----------
    c_surface : float
        表面浓度 [mol/m³]
    c_next : float
        次表面浓度 [mol/m³]
    D_surface : float
        表面扩散系数 [m²/s]
    h : float
        网格间距 [m]
    r_surface : float
        表面径向坐标 [m]
    Phi_s : float
        固相电位 [V]
    Phi_e : float
        液相电位 [V]
    T : float
        温度 [K]
    j0 : float, optional
        交换电流密度 (默认浓度依赖)

    Returns
    -------
    flux : float
        表面 Li 通量 [mol/(m²·s)]
    j_current : float
        表面电流密度 [A/m²]
    eta : float
        过电位 [V]
    """
    if j0 is None:
        j0 = exchange_current_density(c_surface)

    eta = overpotential(Phi_s, Phi_e, c_surface, T)
    j_current = butler_volmer_current(eta, j0, T)
    flux = j_current / FARADAY  # mol/(m²·s)

    return flux, j_current, eta


def center_symmetry_boundary(c_interior, h):
    """
    球心对称边界条件: ∂c/∂r|_{r=0} = 0。

    使用 ghost point: c_ghost = c_{interior}
    即 c_0 (r=0) 的值由对称性确定。

    L'Hôpital 规则在 r=0 处:
    ∇²c|_{r=0} = 3 * ∂²c/∂r²|_{r=0}

    Parameters
    ----------
    c_interior : float
        第一个内部点的浓度
    h : float
        网格间距

    Returns
    -------
    c_center : float
        球心处的浓度 (等于 c_interior, 对称条件)
    """
    return c_interior  # Neumann ∂c/∂r = 0


def robin_boundary_coefficients(D_surface, dj_deta, h, r_surface, T=T_REF):
    """
    Robin 边界条件的线性化系数。

    将非线性 BV 边界线性化为:
    a*c_{N} + b*c_{N-1} = d

    其中:
    -D*(c_N - c_{N-1})/h = j₀/(F*V_T) * (η₀ + dη/dc * (c_N - c_N^k))

    Parameters
    ----------
    D_surface : float
        表面扩散系数
    dj_deta : float
        ∂j/∂η
    h : float
        网格间距
    r_surface : float
        表面径向坐标
    T : float
        温度

    Returns
    -------
    a_coeff : float
        c_N 的系数
    b_coeff : float
        c_{N-1} 的系数
    """
    # 扩散通量: -D*(c_N - c_{N-1})/h
    # 反应通量: j/F = (dj/dη * dη/dc) * δc / F
    # dη/dc = -dU/dc = -(dU/dx) * (1/c_max)

    # 简化: 扩散侧
    a_diff = -D_surface / h
    b_diff = D_surface / h

    # 反应侧 (Robin 耦合)
    # ∂(j/F)/∂c_surface ≈ -(dj/dη / F) * (dU/dx / c_max)
    # 近似为耦合系数
    reaction_coupling = dj_deta / (FARADAY * C_MAX)

    a_coeff = a_diff - reaction_coupling * h
    b_coeff = b_diff

    return a_coeff, b_coeff


def apply_boundary_conditions(c_field, D_func, r_grid, h, N,
                                Phi_s=4.0, Phi_e=0.0, T=T_REF,
                                mode='constant_current', I_app=0.0):
    """
    应用完整的边界条件到浓度场。

    返回修改后的浓度场和边界通量信息。

    Parameters
    ----------
    c_field : ndarray
        浓度场 (包括边界点)
    D_func : callable
        扩散系数函数
    r_grid : ndarray
        径向坐标
    h : float
        网格间距
    N : int
        总网格点数 (含边界)
    Phi_s, Phi_e : float
        固/液相电位
    T : float
        温度
    mode : str
        'constant_current' 或 'butler_volmer'
    I_app : float
        施加电流 (恒流模式)

    Returns
    -------
    c_updated : ndarray
        更新后的浓度场
    boundary_info : dict
        边界条件信息
    """
    c = c_field.copy()

    # ---- 球心边界 (r=0, 对称) ----
    c[0] = center_symmetry_boundary(c[1] if N > 1 else c[0], h)

    # ---- 表面边界 (r=R) ----
    c_surface = c[N-1]
    D_surface = D_func(c_surface, T)

    if mode == 'constant_current':
        # 恒流模式
        j_surface = constant_current_boundary(I_app)
        flux = j_surface / FARADAY
        eta = 0.0
        # Ghost point: c_ghost = c[N-1] + 2*h*flux/D_surface
        if D_surface > 1e-30:
            c_ghost = c_surface + 2.0 * h * flux / D_surface
            c_ghost = max(0.0, min(c_ghost, C_MAX))
            # 更新表面浓度 (一阶近似)
            c[N-1] = 0.5 * (c[N-2] + c_ghost) if N > 1 else c_surface
        else:
            c[N-1] = c_surface

    elif mode == 'butler_volmer':
        # Butler-Volmer 模式
        flux, j_current, eta = surface_flux_boundary(
            c_surface, c[N-2] if N > 1 else c_surface,
            D_surface, h, r_grid[N-1] if N > 0 else PARTICLE_RADIUS,
            Phi_s, Phi_e, T
        )
        if D_surface > 1e-30:
            c_ghost = c_surface + 2.0 * h * flux / D_surface
            c_ghost = max(0.0, min(c_ghost, C_MAX))
            c[N-1] = 0.5 * (c[N-2] + c_ghost) if N > 1 else c_surface
        else:
            c[N-1] = c_surface
        j_surface = j_current
    else:
        raise ValueError(f"未知边界模式: {mode}")

    # 全局浓度保护
    c = np.clip(c, 0.0, C_MAX * 0.999)

    boundary_info = {
        'surface_concentration': float(c_surface),
        'surface_flux': float(flux) if 'flux' in dir() else 0.0,
        'surface_current_density': float(j_surface) if 'j_surface' in dir() else 0.0,
        'overpotential': float(eta) if 'eta' in dir() else 0.0,
        'center_concentration': float(c[0]),
        'D_surface': float(D_surface)
    }

    return c, boundary_info


def mass_conservation_check(c_field, r_grid, h, N, c_initial_total=None):
    """
    质量守恒检查。

    球坐标中总 Li 量:
    M = 4π * ∫₀ᴿ c(r)*r² dr

    数值积分使用梯形法则:
    M ≈ 4π * Σ_i c_i * r_i² * h

    Parameters
    ----------
    c_field : ndarray
        浓度场
    r_grid : ndarray
        径向坐标
    h : float
        网格间距
    N : int
        网格点数
    c_initial_total : float, optional
        初始总 Li 量 (用于比较)

    Returns
    -------
    dict
        质量守恒信息
    """
    # 梯形积分: ∫ c*r² dr
    integrand = c_field * r_grid ** 2
    integral = np.trapz(integrand, dx=h) if N > 1 else integrand[0] * h
    total_li = 4.0 * math.pi * integral

    result = {
        'total_lithium_mol_per_m2': total_li,
        'mean_concentration': float(np.mean(c_field)),
        'max_concentration': float(np.max(c_field)),
        'min_concentration': float(np.min(c_field))
    }

    if c_initial_total is not None and abs(c_initial_total) > 1e-30:
        relative_change = abs(total_li - c_initial_total) / abs(c_initial_total)
        result['relative_mass_change'] = relative_change
        result['mass_conserved'] = relative_change < 0.01  # 1% 阈值

    return result
