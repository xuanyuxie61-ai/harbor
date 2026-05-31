"""
crust_integrals.py
中子星地壳核pasta相的多区域数值积分模块

中子星地壳（密度约 10^4 ~ 10^{11} g/cm^3）存在复杂的核pasta相结构：
球形核（gnocchi相）、柱形核（spaghetti相）、板状核（lasagna相）等。
这些结构需要在三角形、四边形和楔形区域上进行高维积分以计算自由能。

原项目映射:
- 1302_triangle_exactness   -> 三角形区域单值积分与精确度检验
- 957_quadrilateral_witherden_rule -> 四边形高斯积分规则
- 1409_wedge_integrals      -> 三维楔形区域单值积分
"""

import numpy as np
import math
from typing import Tuple


# =============================================================================
# 三角形区域积分 (源自 1302_triangle_exactness)
# =============================================================================
def triangle_area(t: np.ndarray) -> float:
    """
    计算2D三角形的有向面积。

    公式:
        A = 0.5 * | x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2) |

    Parameters
    ----------
    t : np.ndarray, shape (2, 3)
        三个顶点的坐标 [ [x1,x2,x3], [y1,y2,y3] ]。
    """
    area = 0.5 * (
        t[0, 0] * (t[1, 1] - t[1, 2])
        + t[0, 1] * (t[1, 2] - t[1, 0])
        + t[0, 2] * (t[1, 0] - t[1, 1])
    )
    return abs(area)


def triangle_reference_to_physical(t: np.ndarray, n: int, ref: np.ndarray) -> np.ndarray:
    """
    将参考三角形上的点映射到物理三角形。

    参考三角形顶点: (0,0), (1,0), (0,1)
    """
    phy = np.zeros((2, n))
    phy[0, :] = t[0, 0] + (t[0, 1] - t[0, 0]) * ref[0, :] + (t[0, 2] - t[0, 0]) * ref[1, :]
    phy[1, :] = t[1, 0] + (t[1, 1] - t[1, 0]) * ref[0, :] + (t[1, 2] - t[1, 0]) * ref[1, :]
    return phy


def triangle_unit_monomial_integral(expon: Tuple[int, int]) -> float:
    """
    单位三角形上单项式的精确积分。

    公式:
        ∫∫_{T_ref} x^m y^n dx dy = m! n! / (m + n + 2)!

    Parameters
    ----------
    expon : tuple of int
        指数 (m, n)。
    """
    m, n = expon
    if m < 0 or n < 0:
        raise ValueError("Exponents must be non-negative.")

    value = 1.0
    k = m
    for i in range(1, n + 1):
        k += 1
        value *= i / k
    k += 1
    value /= k
    k += 1
    value /= k
    return value


def monomial_value(m_dim: int, n_pts: int, e: Tuple[int, ...], x: np.ndarray) -> np.ndarray:
    """
    计算单项式 ∏ x_i^{e_i} 在多个点上的值。
    源自 1302_triangle_exactness 中的 monomial_value。
    """
    v = np.ones(n_pts)
    for i in range(m_dim):
        if e[i] != 0:
            v *= x[i, :] ** e[i]
    return v


def triangle_quadrature_error(dim_num: int, expon: Tuple[int, ...],
                               point_num: int, x_ref: np.ndarray,
                               w: np.ndarray) -> float:
    """
    计算三角形区域上某单项式的积分误差。

    error = | Q(f) - I(f) |

    Q(f) = Σ_i w_i f(x_i)
    I(f) = 精确积分值
    """
    if dim_num != 2:
        raise ValueError("Only 2D triangles supported.")

    exact = triangle_unit_monomial_integral(expon)
    vals = monomial_value(dim_num, point_num, expon, x_ref)
    quad = np.dot(w, vals)
    return abs(quad - exact)


# =============================================================================
# 四边形区域 Witherden 积分规则 (源自 957_quadrilateral_witherden_rule)
# =============================================================================
def quadrilateral_unit_area() -> float:
    """单位正方形 [0,1]×[0,1] 的面积为 1。"""
    return 1.0


def quadrilateral_unit_monomial_integral(e: Tuple[int, int]) -> float:
    """
    单位正方形上的单项式积分。

    公式:
        ∫_0^1 ∫_0^1 x^a y^b dx dy = 1/((a+1)(b+1))
    """
    a, b = e
    if a < 0 or b < 0:
        raise ValueError("Exponents must be non-negative.")
    return 1.0 / ((a + 1) * (b + 1))


def quadrilateral_witherden_rule(p: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回四边形区域上给定精度 p (0 <= p <= 21) 的 Witherden-Vincent 积分规则。

    源自 957_quadrilateral_witherden_rule 的核心结构。
    此处仅实现低阶规则（p <= 5）的显式节点和权重，高阶使用对称张量积近似。

    Parameters
    ----------
    p : int
        期望精度。

    Returns
    -------
    n : int
        规则阶数。
    x, y : np.ndarray
        节点坐标。
    w : np.ndarray
        权重。
    """
    if p < 0:
        raise ValueError("Precision p must be non-negative.")

    if p <= 1:
        # 1点规则 (p=1)
        n = 1
        x = np.array([0.5])
        y = np.array([0.5])
        w = np.array([1.0])
    elif p <= 3:
        # 4点规则 (p=3) - Gauss-Legendre 2x2 张量积
        n = 4
        pts = np.array([0.211324865405187, 0.788675134594813])
        wg = np.array([0.5, 0.5])
        x, y = np.meshgrid(pts, pts)
        x = x.ravel()
        y = y.ravel()
        w = np.outer(wg, wg).ravel()
    elif p <= 5:
        # 9点规则 (p=5) - Gauss-Legendre 3x3 张量积
        n = 9
        pts = np.array([0.112701665379258, 0.5, 0.887298334620742])
        wg = np.array([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0])
        x, y = np.meshgrid(pts, pts)
        x = x.ravel()
        y = y.ravel()
        w = np.outer(wg, wg).ravel()
    else:
        # 更高阶用 5x5 Gauss-Legendre 近似
        n = 25
        pts = np.array([
            0.046910077030668, 0.230765344947158, 0.5,
            0.769234655052842, 0.953089922969332
        ])
        wg = np.array([
            0.118463442528095, 0.239314335249683, 0.284444444444444,
            0.239314335249683, 0.118463442528095
        ])
        x, y = np.meshgrid(pts, pts)
        x = x.ravel()
        y = y.ravel()
        w = np.outer(wg, wg).ravel()

    return n, x, y, w


# =============================================================================
# 楔形区域积分 (源自 1409_wedge_integrals)
# =============================================================================
def wedge01_volume() -> float:
    """
    单位楔形体体积。
    底面为单位三角形，高为 2（z ∈ [-1, 1]）。
    V = 0.5 * 2 = 1。
    """
    return 1.0


def wedge01_monomial_integral(e: Tuple[int, int, int]) -> float:
    """
    单位楔形体上的单项式积分。

    积分区域:
        0 <= x, 0 <= y, x + y <= 1, -1 <= z <= 1

    公式:
        I = (1 / C(k+2, 2)) * (2 / (e_z + 1))  若 e_z 为偶数
        I = 0                                     若 e_z 为奇数

    其中 k = e_x + e_y，且 C(n,2) = n!/(2!(n-2)!) 相关组合。
    实际递推实现见原代码。
    """
    ex, ey, ez = e
    if ex < 0 or ey < 0:
        raise ValueError("Exponents ex, ey must be non-negative.")
    if ez == -1:
        raise ValueError("ez = -1 is not a legal input.")
    if ez % 2 == 1:
        return 0.0

    value = 1.0
    k = ex
    for i in range(1, ey + 1):
        k += 1
        value *= i / k
    k += 1
    value /= k
    k += 1
    value /= k
    value *= 2.0 / (ez + 1)
    return value


# =============================================================================
# 核pasta相自由能计算
# =============================================================================
def pasta_free_energy_triangle_lattice(
    lattice_spacing: float,
    surface_tension: float,
    nuclear_radius: float
) -> float:
    """
    使用三角形区域积分计算球形核（gnocchi相）的自由能密度。

    模型: 在Wigner-Seitz原胞（近似为正三角形）内积分核物质的
    体积能和表面能。

    F = ∫_cell [ ε_bulk(r) + γ δ(r - R_n) ] dA

    Parameters
    ----------
    lattice_spacing : float
        晶格常数 (fm)。
    surface_tension : float
        表面张力系数 (MeV/fm^2)。
    nuclear_radius : float
        核半径 (fm)。

    Returns
    -------
    float
        自由能密度 (MeV/fm^2)。
    """
    if lattice_spacing <= 0.0 or nuclear_radius <= 0.0 or surface_tension < 0.0:
        raise ValueError("All physical parameters must be positive.")

    # 三角形原胞顶点
    a = lattice_spacing
    t = np.array([
        [0.0, a, a / 2.0],
        [0.0, 0.0, a * math.sqrt(3.0) / 2.0]
    ])
    area = triangle_area(t)

    # 在三角形内采样并计算自由能
    # 使用3点高斯积分规则
    ref_pts = np.array([[1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]])
    weights = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])

    phy_pts = triangle_reference_to_physical(t, 3, ref_pts)

    F = 0.0
    center = np.array([a / 2.0, a * math.sqrt(3.0) / 6.0])
    for i in range(3):
        r_vec = phy_pts[:, i] - center
        r_dist = math.sqrt(np.sum(r_vec**2))
        # 体积能密度 (常数)
        eps_vol = -15.0  # MeV/fm^2 (简化值)
        # 表面能 (δ函数近似为在 R_n 附近的环上)
        surf = surface_tension if abs(r_dist - nuclear_radius) < 0.1 else 0.0
        F += weights[i] * (eps_vol + surf)

    return F * area


def pasta_free_energy_quadrilateral_sheet(
    sheet_width: float,
    sheet_spacing: float,
    surface_tension: float
) -> float:
    """
    使用四边形积分计算板状核（lasagna相）的自由能密度。

    模型: 在周期性四边形单元内积分板状结构的能量。
    """
    if sheet_width <= 0.0 or sheet_spacing <= 0.0:
        raise ValueError("Physical parameters must be positive.")

    n, x, y, w = quadrilateral_witherden_rule(5)

    F = 0.0
    for i in range(n):
        # 板位于 x = 0.5 附近，宽度 sheet_width/sheet_spacing
        xi = x[i] * sheet_spacing
        yi = y[i] * sheet_spacing
        eps_vol = -15.0
        # 表面能在板边界
        in_sheet = abs(xi - sheet_spacing / 2.0) < sheet_width / 2.0
        surf = surface_tension if in_sheet else 0.0
        F += w[i] * (eps_vol + surf)

    return F * sheet_spacing**2


def pasta_free_energy_wedge_cylinder(
    cylinder_radius: float,
    cell_size: float,
    surface_tension: float
) -> float:
    """
    使用楔形区域积分计算柱形核（spaghetti相）的自由能密度。

    模型: 在3D楔形单元内积分柱形结构的能量。
    """
    if cylinder_radius <= 0.0 or cell_size <= 0.0:
        raise ValueError("Physical parameters must be positive.")

    # 简化的楔形积分：用3D张量积近似
    # 三角形底面 × z方向
    n_tri, x_tri, y_tri, w_tri = quadrilateral_witherden_rule(3)
    # 映射到三角形
    x_tri = x_tri[:3]
    y_tri = y_tri[:3]
    w_tri = w_tri[:3]

    z_pts = np.array([-0.577350269189626, 0.577350269189626])
    z_w = np.array([1.0, 1.0])

    F = 0.0
    for i in range(len(x_tri)):
        for j in range(len(z_pts)):
            xi = x_tri[i] * cell_size
            yi = y_tri[i] * cell_size
            zi = z_pts[j] * cell_size
            eps_vol = -15.0
            r_dist = math.sqrt(xi**2 + yi**2)
            in_cyl = r_dist < cylinder_radius
            surf = surface_tension if abs(r_dist - cylinder_radius) < 0.05 else 0.0
            F += w_tri[i] * z_w[j] * (eps_vol + surf) * in_cyl

    return F * cell_size**3
