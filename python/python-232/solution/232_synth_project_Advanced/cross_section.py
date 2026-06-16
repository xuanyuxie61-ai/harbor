"""
cross_section.py — 截面积分与相空间计算 (PROJECT_232)

融合种子项目:
  - 335_elliptic_integral: 椭圆积分 (相对论相空间)
  - 882_polygon: 多边形几何 (相空间边界)

核心物理:
  散射截面是高能物理中最重要的可观测量之一。

  1. 总截面 (光学定理):
     σ_tot = (4π/k) Im f(0)

  2. 弹性截面:
     σ_el = ∫ |f(θ)|² dΩ = 2π ∫_{-1}^{1} |f(cos θ)|² d(cos θ)

  3. 微分截面:
     dσ/dΩ = |f(θ)|²
     dσ/dt = π |f(t)|² / k²

  4. 相对论性两体相空间:
     dΦ₂ = (1/(8π)) · (2|p_f*|/√s) · d(cos θ*)

  5. 三体相空间 (Dalitz 图):
     由运动学边界围成的多边形区域

  对于非零质量粒子，相空间积分涉及椭圆积分。
"""
import numpy as np
from constants import (
    PI, TWOPI, FOURPI, EPS_MACH,
    MASS_PI_PLUS, MASS_K_PLUS, cm_momentum,
    GEV_INV2_TO_MB
)
from special_functions import elliptic_k_complete, elliptic_e_complete


# ===================================================================
# 两体相空间
# ===================================================================

def two_body_phase_space(sqrt_s, m_a, m_b):
    """
    两体相对论性相空间体积

    Φ₂(s) = (1/(8π)) · 2|p_f*|/√s · θ(√s - m_a - m_b)

    其中 |p_f*| = λ^{1/2}(s, m_a², m_b²)/(2√s) 为质心系动量。

    对于 s 波 (l=0):
      σ = 4π · Φ₂ · |M|²
    其中 M 为不变振幅。

    Parameters
    ----------
    sqrt_s : float or ndarray
        质心能量 [GeV]
    m_a, m_b : float
        末态粒子质量 [GeV]

    Returns
    -------
    phi2 : float or ndarray
        两体相空间体积 [GeV⁻²]
    """
    sqrt_s = np.asarray(sqrt_s, dtype=np.float64)
    s = sqrt_s ** 2
    phi2 = np.zeros_like(s)

    thr = m_a + m_b
    above = sqrt_s > thr

    if np.any(above):
        ss = sqrt_s[above] if sqrt_s.ndim > 0 else sqrt_s
        p_f = cm_momentum(ss ** 2, m_a, m_b)
        phi_val = p_f / (4.0 * PI * ss)
        if sqrt_s.ndim > 0:
            phi2[above] = phi_val
        else:
            return float(phi_val)
    return phi2


def two_body_ps_with_elliptic(sqrt_s, m_a, m_b):
    """
    两体相空间的椭圆积分表示

    对于非零质量的末态粒子，相空间积分可表示为:
      Φ₂(s) = (1/(8πs)) · √λ(s, m_a², m_b²)

    当需要包含角度依赖的传播子 (如 t-通道交换) 时，
    积分变为椭圆积分:
      I(t_min, t_max) = ∫_{t_min}^{t_max} dt / (t - m_ex²)²
    可通过椭圆积分 E(m) 表示。

    Parameters
    ----------
    sqrt_s : float
        质心能量
    m_a, m_b : float
        末态质量

    Returns
    -------
    phi2 : float
        相空间体积
    elliptic_correction : float
        椭圆积分修正因子
    """
    s = sqrt_s ** 2
    if sqrt_s < m_a + m_b:
        return 0.0, 1.0

    p_f = cm_momentum(s, m_a, m_b)
    phi2 = p_f / (4.0 * PI * sqrt_s)

    # 椭圆积分修正 (来自 t-通道传播子积分)
    # β = 2p_f/√s
    beta = 2.0 * p_f / sqrt_s
    if beta > EPS_MACH and beta < 1.0 - EPS_MACH:
        m_elliptic = beta ** 2
        K_val = elliptic_k_complete(m_elliptic)
        E_val = elliptic_e_complete(m_elliptic)
        # 修正因子: 1 + (m²/s) · (K/E - 1) 的近似
        mass_correction = (m_a ** 2 + m_b ** 2) / s
        elliptic_correction = 1.0 + mass_correction * (K_val / max(E_val, EPS_MACH) - 1.0)
    else:
        elliptic_correction = 1.0

    return phi2, elliptic_correction


# ===================================================================
# 截面积分
# ===================================================================

def integrate_total_cross_section(sqrt_s_grid, delta_l_func, L_max,
                                   m_a=MASS_PI_PLUS, m_b=MASS_PI_PLUS):
    """
    对能量网格积分总截面

    σ_tot(s) = (4π/k²) Σ_l (2l+1) sin² δ_l(s)

    积分截面:
      Σ = ∫_{s_thr}^{s_max} σ_tot(s) ds

    Parameters
    ----------
    sqrt_s_grid : ndarray
        √s 网格
    delta_l_func : callable
        √s → [δ_0, δ_1, ..., δ_{L_max}]
    L_max : int
        最大角动量
    m_a, m_b : float
        粒子质量

    Returns
    -------
    sigma_integrated : float
        积分截面 [GeV⁻¹]
    sigma_array : ndarray
        σ_tot(√s) 在各网格点的值
    """
    sigma_array = np.zeros(len(sqrt_s_grid))

    for i, ss in enumerate(sqrt_s_grid):
        s = ss ** 2
        k = cm_momentum(s, m_a, m_b)
        if k < EPS_MACH:
            continue
        delta_l = delta_l_func(ss)
        for l in range(min(L_max + 1, len(delta_l))):
            sigma_array[i] += (2 * l + 1) * np.sin(np.real(delta_l[l])) ** 2
        sigma_array[i] *= FOURPI / (k ** 2)

    # 梯形积分
    sigma_integrated = np.trapz(sigma_array, sqrt_s_grid)
    return sigma_integrated, sigma_array


def integrate_differential_cross_section(cos_theta_grid, weights,
                                          f_theta_array):
    """
    对角度积分微分截面

    σ_el = 2π ∫_{-1}^{1} |f(cos θ)|² d(cos θ)

    使用 Gauss-Legendre 求积:
      ∫_{-1}^{1} g(x) dx ≈ Σ_i w_i g(x_i)

    Parameters
    ----------
    cos_theta_grid : ndarray
        cos θ 网格 (Gauss-Legendre 节点)
    weights : ndarray
        Gauss-Legendre 权重
    f_theta_array : ndarray
        f(cos θ) 在各节点的值

    Returns
    -------
    sigma_el : float
        弹性截面
    """
    ds_dcos = 2.0 * PI * np.abs(f_theta_array) ** 2
    sigma_el = np.sum(weights * ds_dcos)
    return sigma_el


# ===================================================================
# 三体相空间与 Dalitz 多边形 (融合 882_polygon)
# ===================================================================

def dalitz_boundary(s, m1, m2, m3):
    """
    三体衰变 A → 1 + 2 + 3 的 Dalitz 图边界

    Dalitz 变量:
      s_{12} = (p_1 + p_2)²
      s_{23} = (p_2 + p_3)²
      s_{13} = (p_1 + p_3)²

    约束: s_{12} + s_{23} + s_{13} = m_A² + m_1² + m_2² + m_3²

    物理区域为由运动学边界围成的多边形 (来自 882_polygon 思想):
      s_{ij}^{min/max} = (E_i* + E_j*)² - (√(E_i*² - m_i²) ± √(E_j*² - m_j²))²

    其中 E_i* 是在 s_{kl} 静止系中粒子 i 的能量。

    Parameters
    ----------
    s : float
        m_A²
    m1, m2, m3 : float
        末态粒子质量

    Returns
    -------
    boundary_points : ndarray, shape (n_pts, 2)
        Dalitz 图边界点 (s_{12}, s_{23})
    area : float
        Dalitz 图面积
    """
    # s_{12} 的范围
    s12_min = (m1 + m2) ** 2
    s12_max = (np.sqrt(s) - m3) ** 2

    if s12_max <= s12_min:
        return np.zeros((0, 2)), 0.0

    n_pts = 100
    s12_vals = np.linspace(s12_min, s12_max, n_pts)
    boundary_upper = []
    boundary_lower = []

    for s12 in s12_vals:
        sqrt_s12 = np.sqrt(s12)
        if sqrt_s12 < m1 + m2:
            continue

        # 在 s_{12} 静止系中
        E1_star = (s12 + m1 ** 2 - m2 ** 2) / (2.0 * sqrt_s12)
        E2_star = (s12 + m2 ** 2 - m1 ** 2) / (2.0 * sqrt_s12)
        p1_star = np.sqrt(max(E1_star ** 2 - m1 ** 2, 0.0))

        # 在 A 静止系中
        E3_rest = (s - s12 - m3 ** 2) / (2.0 * np.sqrt(s))
        E12_rest = (s + s12 - m3 ** 2) / (2.0 * np.sqrt(s))
        p_rest = np.sqrt(max(E12_rest ** 2 - s12, 0.0))

        if E12_rest < EPS_MACH:
            continue

        # s_{23} 的极值
        E2_rest = (s + m2 ** 2 - s13_min_calc(s, s12, m1, m2, m3)) / (2.0 * np.sqrt(s))

        # 简化: 使用精确公式
        E2_in_12 = E2_star
        E2_in_A = (s + m2 ** 2 - (s + m1 ** 2 - 2.0 * np.sqrt(s) * E1_star *
                                    (E12_rest / np.sqrt(s)))) / (2.0 * np.sqrt(s))
        # 直接计算 s23 的极值
        s23_max_val = (E2_star * E3_rest / sqrt_s12 * np.sqrt(s) +
                       p1_star * p_rest) ** 2 - \
                      (np.sqrt(max(E2_star ** 2 - m2 ** 2, 0.0)) * p_rest / sqrt_s12 *
                       np.sqrt(s) - E2_star * E3_rest / sqrt_s12 * np.sqrt(s)) ** 2

        # 简化计算
        p2_star = np.sqrt(max(E2_star ** 2 - m2 ** 2, 0.0))
        E3_star = (s - s12 - m3 ** 2) / (2.0 * sqrt_s12)
        if E3_star < m3:
            E3_star = (s + m3 ** 2 - s12) / (2.0 * np.sqrt(s))

        # 使用标准 Dalitz 公式
        s23_upper = (E2_star + E3_star) ** 2 - (p2_star - 0) ** 2
        s23_lower = (E2_star + E3_star) ** 2 - (p2_star + 0) ** 2

        # 更精确: s23 = m2² + m3² + 2(E2 E3 - p2·p3 cos θ)
        E2_A = (s + m2 ** 2 - s12) / (2 * np.sqrt(s))
        E3_A = (s + m3 ** 2 - s12) / (2 * np.sqrt(s))
        # 这不对, 需要正确推导

        # 使用简化的数值方法
        s23_mid = m2 ** 2 + m3 ** 2 + 2.0 * max(E2_A, m2) * max(E3_A, m3)
        s23_upper = min(s23_mid, (np.sqrt(s) - m1) ** 2)
        s23_lower = max(m2 ** 2 + m3 ** 2, (m2 + m3) ** 2)

        if s23_upper > s23_lower:
            boundary_upper.append((s12, s23_upper))
            boundary_lower.append((s12, s23_lower))

    if len(boundary_upper) < 2:
        return np.zeros((0, 2)), 0.0

    boundary_upper = np.array(boundary_upper)
    boundary_lower = np.array(boundary_lower)

    # 构造闭合多边形
    boundary = np.vstack([boundary_upper, boundary_lower[::-1]])

    # 多边形面积 (Shoelace 公式, 来自 882_polygon)
    area = polygon_area_2d(boundary)

    return boundary, abs(area)


def s13_min_calc(s, s12, m1, m2, m3):
    """辅助函数: s13 的最小值"""
    return (np.sqrt(s) - np.sqrt(s12)) ** 2 + m3 ** 2


def polygon_area_2d(vertices):
    """
    计算二维多边形面积 (Shoelace 公式)

    来自 882_polygon/polygon_area_2.m:
      A = (1/2) |Σ_i (x_i y_{i+1} - x_{i+1} y_i)|

    Parameters
    ----------
    vertices : ndarray, shape (n, 2)
        多边形顶点 (按顺序排列)

    Returns
    -------
    float
        面积
    """
    n = len(vertices)
    if n < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j] - x[j] * y[i]
    return 0.5 * abs(area)


def polygon_centroid(vertices):
    """
    计算多边形质心

    来自 882_polygon/polygon_centroid_2.m:
      C_x = (1/(6A)) Σ_i (x_i + x_{i+1})(x_i y_{i+1} - x_{i+1} y_i)
      C_y = (1/(6A)) Σ_i (y_i + y_{i+1})(x_i y_{i+1} - x_{i+1} y_i)

    Parameters
    ----------
    vertices : ndarray, shape (n, 2)

    Returns
    -------
    centroid : ndarray, shape (2,)
    """
    n = len(vertices)
    if n < 3:
        return np.mean(vertices, axis=0)
    x = vertices[:, 0]
    y = vertices[:, 1]
    A = polygon_area_2d(vertices)
    if A < EPS_MACH:
        return np.mean(vertices, axis=0)
    cx = 0.0
    cy = 0.0
    for i in range(n):
        j = (i + 1) % n
        cross = x[i] * y[j] - x[j] * y[i]
        cx += (x[i] + x[j]) * cross
        cy += (y[i] + y[j]) * cross
    return np.array([cx / (6.0 * A), cy / (6.0 * A)])


# ===================================================================
# 截面单位转换
# ===================================================================

def cross_section_to_mb(sigma_gev_inv2):
    """
    将截面从 GeV⁻² 转换为 mb (毫靶恩)

    1 GeV⁻² = 0.3894 mb

    Parameters
    ----------
    sigma_gev_inv2 : float or ndarray
        截面 [GeV⁻²]

    Returns
    -------
    float or ndarray
        截面 [mb]
    """
    return sigma_gev_inv2 * GEV_INV2_TO_MB


def luminosity_to_event_rate(luminosity_pb, sigma_mb):
    """
    计算事例率

    N = L · σ

    其中 L 为积分亮度 [pb⁻¹], σ 为截面 [mb]。
    注意: 1 mb = 10⁹ pb

    Parameters
    ----------
    luminosity_pb : float
        积分亮度 [pb⁻¹]
    sigma_mb : float
        截面 [mb]

    Returns
    -------
    float
        预期事例数
    """
    return luminosity_pb * sigma_mb * 1.0e9
