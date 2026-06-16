"""
stability_analysis.py - 数值格式的von Neumann稳定性分析

本模块实现针对边界等离子体输运方程的数值稳定性分析：
  1. von Neumann稳定性分析（傅里叶模式分析）
  2. 紧致有限差分格式的放大因子计算
  3. 对流-扩散方程的稳定性区域
  4. DG方法的CFL限制分析
  5. 刚性系统的稳定性分析

公式体系：
  对于方程 u_t + a*u_x = D*u_xx
  半离散格式: du_j/dt = L_h * u_j
  von Neumann分析: u_j = G^k * exp(i*k*j*h)
  放大因子: G(k*h) = (1 + ...) / (1 + ...)
  稳定性条件: |G| <= 1 + O(dt)
"""

import numpy as np
from typing import Tuple, Dict
from legendre_fd import compact_fd_coefficients_1st, compact_fd_coefficients_2nd


# =============================================================================
# von Neumann稳定性分析 - 基本框架
# =============================================================================
def von_neumann_amplification_advection(scheme: str, k_h: np.ndarray,
                                          CFL: float = 0.5,
                                          order: int = 4) -> np.ndarray:
    """
    计算纯对流方程的放大因子 G(k*h)

    方程: u_t + a*u_x = 0

    对于半离散格式:
      du_j/dt = -a/h * sum(c_m * u_{j+m})
    代入 u_j = G * exp(i*j*k*h):
      G = exp(-i*a*dt/h * sum(c_m * exp(i*m*k*h)))

    支持的格式:
      'upwind1': 一阶迎风
      'central2': 二阶中心差分
      'central4': 四阶中心差分
      'compact4': 四阶紧致格式
      'compact6': 六阶紧致格式

    参数:
        scheme: 格式名称
        k_h: 无量纲波数 k*h ∈ [0, pi]
        CFL: CFL数 a*dt/h
        order: 紧致格式阶数
    返回:
        G: 放大因子（复数数组）
    """
    G = np.zeros(len(k_h), dtype=complex)

    if scheme == 'upwind1':
        # 一阶迎风: u_j^{n+1} = u_j^n - CFL*(u_j^n - u_{j-1}^n)
        # G = 1 - CFL*(1 - exp(-i*k*h))
        G = 1.0 - CFL * (1.0 - np.exp(-1j * k_h))

    elif scheme == 'central2':
        # 二阶中心差分 (Leapfrog)
        # G = 1 - i*CFL*sin(k*h)
        G = 1.0 - 1j * CFL * np.sin(k_h)

    elif scheme == 'central4':
        # 四阶中心差分
        # du/dx ≈ (-u_{j+2} + 8*u_{j+1} - 8*u_{j-1} + u_{j-2}) / (12*h)
        # G = 1 - i*CFL/12 * (8*sin(kh) - sin(2kh))
        G = 1.0 - 1j * CFL / 12.0 * (8.0 * np.sin(k_h) - np.sin(2.0 * k_h))

    elif scheme == 'compact4':
        # 四阶紧致格式
        alpha, a_coeff, b_coeff = compact_fd_coefficients_1st(order)
        # 隐式部分: (1 + 2*alpha*cos(kh)) * G_correction = ...
        # 显式部分: -i*CFL*(2*a*sin(kh) + 2*b*sin(2kh))
        numerator = 1.0 - 1j * CFL * (2.0 * a_coeff * np.sin(k_h) +
                                         2.0 * b_coeff * np.sin(2.0 * k_h))
        denominator = 1.0 + 2.0 * alpha * np.cos(k_h)
        G = numerator / (denominator + 1e-30)

    elif scheme == 'compact6':
        alpha, a_coeff, b_coeff = compact_fd_coefficients_1st(6)
        numerator = 1.0 - 1j * CFL * (2.0 * a_coeff * np.sin(k_h) +
                                         2.0 * b_coeff * np.sin(2.0 * k_h))
        denominator = 1.0 + 2.0 * alpha * np.cos(k_h)
        G = numerator / (denominator + 1e-30)

    return G


def von_neumann_amplification_diffusion(scheme: str, k_h: np.ndarray,
                                          diffusion_number: float = 0.3,
                                          order: int = 4) -> np.ndarray:
    """
    计算纯扩散方程的放大因子

    方程: u_t = D*u_xx

    定义扩散数: r = D*dt/h^2

    支持的格式:
      'forward_euler': 显式Euler
      'backward_euler': 隐式Euler
      'crank_nicolson': Crank-Nicolson
      'compact4': 四阶紧致

    参数:
        scheme: 格式名称
        k_h: 无量纲波数
        diffusion_number: 扩散数 r = D*dt/h^2
        order: 紧致格式阶数
    返回:
        G: 放大因子（复数）
    """
    r = diffusion_number
    G = np.zeros(len(k_h), dtype=complex)

    if scheme == 'forward_euler':
        # G = 1 - 2r*(1 - cos(kh))
        G = 1.0 - 2.0 * r * (1.0 - np.cos(k_h))

    elif scheme == 'backward_euler':
        # G = 1 / (1 + 2r*(1 - cos(kh)))
        G = 1.0 / (1.0 + 2.0 * r * (1.0 - np.cos(k_h)))

    elif scheme == 'crank_nicolson':
        # G = (1 - r*(1-cos(kh))) / (1 + r*(1-cos(kh)))
        num = 1.0 - r * (1.0 - np.cos(k_h))
        den = 1.0 + r * (1.0 - np.cos(k_h))
        G = num / (den + 1e-30)

    elif scheme == 'compact4':
        alpha, a_coeff, b_coeff = compact_fd_coefficients_2nd(order)
        # 半离散: du/dt = D * (a*(u_{j+1}-2u_j+u_{j-1})/h^2 + b*(u_{j+2}-2u_j+u_{j-2})/(4h^2))
        # 隐式: (1 + 2*alpha*cos(kh)) * du/dt = ...
        spatial = -2.0 * r * (a_coeff * (1.0 - np.cos(k_h)) +
                                b_coeff * (1.0 - np.cos(2.0 * k_h)) / 2.0)
        denominator = 1.0 + 2.0 * alpha * np.cos(k_h)
        # 使用Crank-Nicolson时间积分
        G = (1.0 + 0.5 * spatial / (denominator + 1e-30)) / \
            (1.0 - 0.5 * spatial / (denominator + 1e-30) + 1e-30)

    return G


def von_neumann_amplification_advection_diffusion(
        k_h: np.ndarray, CFL: float = 0.3,
        diffusion_number: float = 0.1,
        scheme: str = 'compact4') -> np.ndarray:
    """
    对流-扩散耦合方程的放大因子

    方程: u_t + a*u_x = D*u_xx

    参数:
        k_h: 无量纲波数
        CFL: CFL数
        diffusion_number: 扩散数
        scheme: 空间格式
    返回:
        G: 放大因子
    """
    # 对流部分
    G_adv = von_neumann_amplification_advection(scheme, k_h, CFL)

    # 扩散部分
    G_diff = von_neumann_amplification_diffusion('crank_nicolson', k_h, diffusion_number)

    # 组合（分裂近似）
    G = G_adv * G_diff

    return G


# =============================================================================
# 稳定性区域计算
# =============================================================================
def compute_stability_region_advection(scheme: str,
                                         CFL_range: np.ndarray,
                                         k_h_range: np.ndarray,
                                         order: int = 4) -> np.ndarray:
    """
    计算对流方程的稳定性区域

    参数:
        scheme: 格式名称
        CFL_range: CFL数范围
        k_h_range: 波数范围
        order: 格式阶数
    返回:
        stable: shape (len(CFL), len(k_h)) 布尔数组，True表示稳定
    """
    stable = np.zeros((len(CFL_range), len(k_h_range)), dtype=bool)

    for i, CFL in enumerate(CFL_range):
        G = von_neumann_amplification_advection(scheme, k_h_range, CFL, order)
        stable[i, :] = np.abs(G) <= 1.0 + 1e-10

    return stable


def find_max_CFL(scheme: str, order: int = 4, tol: float = 1e-4) -> float:
    """
    查找给定格式的最大稳定CFL数

    通过二分搜索找到使所有波数都满足 |G| <= 1 的最大CFL

    参数:
        scheme: 格式名称
        order: 格式阶数
        tol: 精度
    返回:
        CFL_max: 最大稳定CFL数
    """
    k_h = np.linspace(0.01, np.pi, 200)

    CFL_low = 0.0
    CFL_high = 2.0

    for _ in range(50):
        CFL_mid = (CFL_low + CFL_high) / 2.0
        G = von_neumann_amplification_advection(scheme, k_h, CFL_mid, order)
        max_G = np.max(np.abs(G))

        if max_G <= 1.0 + tol:
            CFL_low = CFL_mid
        else:
            CFL_high = CFL_mid

        if CFL_high - CFL_low < 1e-6:
            break

    return CFL_low


# =============================================================================
# DG方法的CFL分析
# =============================================================================
def dg_cfl_limit(N_order: int, scheme_type: str = 'upwind') -> float:
    """
    计算DG方法的CFL限制

    对于N阶DG方法:
      CFL_max ≈ 1 / (2N+1)  (Lax-Friedrichs通量)

    精确值依赖于数值通量和基函数

    参数:
        N_order: 多项式阶数
        scheme_type: 通量类型
    返回:
        CFL_max: 最大稳定CFL数
    """
    if scheme_type == 'upwind':
        # 迎风通量：CFL ≈ 1/(2N+1)
        return 1.0 / (2.0 * N_order + 1.0)
    elif scheme_type == 'central':
        # 中心通量：更严格的限制
        return 0.5 / (2.0 * N_order + 1.0)
    else:
        return 1.0 / (2.0 * N_order + 1.0)


# =============================================================================
# 刚性系统稳定性分析
# =============================================================================
def analyze_stiff_system(eigenvalues: np.ndarray, dt: float,
                           method: str = 'rk4') -> np.ndarray:
    """
    分析刚性ODE系统的数值稳定性

    对于 du/dt = L*u, 其中L的特征值为lambda_i
    数值方法稳定的条件是 |R(z)| <= 1, 其中 z = lambda*dt
    R(z) 是方法的稳定性函数

    各方法的稳定性函数:
      Euler显式: R(z) = 1 + z
      Euler隐式: R(z) = 1/(1-z)
      RK4: R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24
      CN: R(z) = (1+z/2)/(1-z/2)

    参数:
        eigenvalues: 系统矩阵的特征值
        dt: 时间步长
        method: 时间积分方法
    返回:
        amplification: 各特征值对应的放大因子
    """
    z = eigenvalues * dt

    if method == 'euler_explicit':
        R = 1.0 + z
    elif method == 'euler_implicit':
        R = 1.0 / (1.0 - z + 1e-30)
    elif method == 'rk4':
        R = 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0
    elif method == 'crank_nicolson':
        R = (1.0 + z / 2.0) / (1.0 - z / 2.0 + 1e-30)
    elif method == 'rk3_tvd':
        # TVD RK3 (Shu-Osher)
        R = 1.0 + z + z**2 / 2.0 + z**3 / 6.0  # 近似
    else:
        R = 1.0 + z

    return R


def plasma_transport_eigenvalues(n_points: int, D: float, h: float,
                                   a: float = 0.0) -> np.ndarray:
    """
    计算等离子体输运离散系统的特征值

    离散算子: L = D * D2 - a * D1
    其中 D1, D2 是一阶和二阶中心差分矩阵

    特征值: lambda_k = -D*(2/h^2)*(1-cos(k*pi*h)) - i*a*sin(k*pi*h)/h

    参数:
        n_points: 网格点数
        D: 扩散系数
        h: 网格间距
        a: 对流速度
    返回:
        eigenvalues: 复数特征值数组
    """
    k = np.arange(1, n_points)
    lambda_k = (-D * 2.0 / h**2 * (1.0 - np.cos(k * np.pi / n_points))
                - 1j * a / h * np.sin(k * np.pi / n_points))
    return lambda_k


# =============================================================================
# 完整稳定性报告
# =============================================================================
def generate_stability_report(
        advection_speed: float = 1e4,
        diffusion_coeff: float = 1.0,
        h: float = 0.01,
        N_order: int = 3) -> Dict:
    """
    生成完整的稳定性分析报告

    参数:
        advection_speed: 对流速度
        diffusion_coeff: 扩散系数
        h: 网格间距
        N_order: DG多项式阶数
    返回:
        report: 包含各项稳定性指标的字典
    """
    # CFL数
    CFL_adv = abs(advection_speed) / h
    diff_number = diffusion_coeff / h**2

    # DG CFL限制
    dg_cfl = dg_cfl_limit(N_order)
    dg_dt_max = dg_cfl * h / (abs(advection_speed) + 1e-30)

    # 特征值分析
    eigvals = plasma_transport_eigenvalues(32, diffusion_coeff, h, advection_speed)
    spectral_radius = np.max(np.abs(eigvals))

    # 刚性比
    real_parts = np.abs(np.real(eigvals))
    imag_parts = np.abs(np.imag(eigvals))
    if np.min(imag_parts + 1e-30) > 1e-30:
        stiffness_ratio = np.max(real_parts) / (np.min(imag_parts + 1e-30) + 1e-30)
    else:
        stiffness_ratio = float('inf')

    # 最大稳定时间步长
    dt_explicit = 2.0 / (spectral_radius + 1e-30)

    report = {
        'CFL_advection': CFL_adv,
        'diffusion_number': diff_number,
        'dg_cfl_limit': dg_cfl,
        'dg_max_dt': dg_dt_max,
        'spectral_radius': spectral_radius,
        'stiffness_ratio': stiffness_ratio,
        'max_explicit_dt': dt_explicit,
        'is_stiff': stiffness_ratio > 10.0,
        'recommended_method': 'implicit' if stiffness_ratio > 10.0 else 'rk4',
        'N_order': N_order,
        'n_modes': len(eigvals),
    }

    return report
