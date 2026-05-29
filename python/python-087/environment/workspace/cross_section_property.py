"""
cross_section_property.py
=========================
蜂窝六边形截面几何特性与多维积分

本模块将以下种子项目的核心算法融入结构力学：
  - 526_hexagon_chaos : 正六边形几何、重心坐标、菱形分解 → 蜂窝微结构截面几何建模
  - 553_hyperball_integrals : M维超球积分、Gamma函数、均匀采样 → 截面几何特性数值积分

核心物理模型：
  - 蜂窝夹芯结构等效弹性模量（Gibson-Ashby 公式）：
        E*/E_s = C₁ · (t/l)³
    其中 t 为壁厚，l 为蜂窝边长，E_s 为基材弹性模量。
  
  - 截面惯性矩与面积矩通过域积分计算：
        I_y = ∫_A z² dA,    I_z = ∫_A y² dA
        J   = ∫_A (y² + z²) dA = I_y + I_z   (极惯性矩)
  
  - 对任意截面，采用 Monte Carlo 均匀采样估计几何积分：
        I_y ≈ (A_total / N) · Σ z_i²
    其中采样点均匀分布于截面域内。
"""

import numpy as np
from scipy.special import gamma as Gamma
from typing import Tuple, List


def regular_hexagon_vertices(R: float = 1.0) -> np.ndarray:
    """
    生成中心在原点、外接圆半径为 R 的正六边形顶点（逆时针）。
    顶点坐标：
        v_k = R · [cos(kπ/3), sin(kπ/3)],  k = 0, ..., 5
    """
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    verts = np.column_stack((R * np.cos(angles), R * np.sin(angles)))
    return verts


def hexagon_area(R: float = 1.0) -> float:
    """
    正六边形面积：A = (3√3 / 2) R²，其中 R 为外接圆半径。
    """
    return 1.5 * np.sqrt(3.0) * R ** 2


def hexagon_uniform_sample(R: float = 1.0, n_samples: int = 1000) -> np.ndarray:
    """
    基于 526_hexagon_chaos 的菱形分解思想，在正六边形内均匀随机采样。
    
    正六边形可分解为 3 个全等菱形。对每个菱形，由顶点 (0,0), v_k, v_{k+1}
    张成。均匀采样方法：
        p = λ₁ v_k + λ₂ v_{k+1},   λ₁, λ₂ ~ U(0,1), 并限制 λ₁+λ₂ ≤ 1
    实际采用三角形重心坐标：
        p = α·0 + β·v_k + γ·v_{k+1},  α+β+γ=1, α,β,γ∈[0,1]
    等价于 p = β·v_k + γ·v_{k+1}, β+γ ≤ 1。
    
    参数
    ----
    R : 外接圆半径
    n_samples : 采样点数
    
    返回
    ----
    samples : (n_samples, 2) 均匀采样点
    """
    verts = regular_hexagon_vertices(R)
    samples = np.zeros((n_samples, 2), dtype=np.float64)
    # 六边形由 6 个等边三角形组成，或 3 个菱形。这里用 6 个三角形 (中心, v_k, v_{k+1})
    triangles = []
    for k in range(6):
        triangles.append(np.array([[0.0, 0.0], verts[k], verts[(k + 1) % 6]]))
    areas = []
    for tri in triangles:
        # 三角形面积 = 0.5 |cross(v1-v0, v2-v0)|
        a = 0.5 * abs(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
        areas.append(a)
    areas = np.array(areas)
    probs = areas / areas.sum()
    for i in range(n_samples):
        # 按面积比例随机选三角形
        t_idx = np.searchsorted(np.cumsum(probs), np.random.rand())
        tri = triangles[t_idx]
        # 在三角形内均匀采样：sqrt(r1) 保证面积均匀
        r1, r2 = np.random.rand(2)
        sqrt_r1 = np.sqrt(r1)
        alpha = 1.0 - sqrt_r1
        beta = sqrt_r1 * (1.0 - r2)
        gamma = sqrt_r1 * r2
        samples[i] = alpha * tri[0] + beta * tri[1] + gamma * tri[2]
    return samples


def honeycomb_cell_geometry(cell_size: float, wall_thickness: float,
                            n_rings: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建蜂窝 lattice 的六边形单胞集合。
    以中心六边形为第 0 环，逐环扩展。
    
    返回
    ----
    centers : (n_cells, 2) 各六边形中心坐标
    vertices_list : (n_cells, 6, 2) 各六边形顶点
    """
    if cell_size <= 0 or wall_thickness <= 0:
        raise ValueError("cell_size 与 wall_thickness 必须为正")
    centers = []
    # 六边形密排方向间距
    dx = 1.5 * cell_size
    dy = np.sqrt(3.0) * cell_size
    centers.append(np.array([0.0, 0.0]))
    for ring in range(1, n_rings + 1):
        for k in range(6):
            angle = k * np.pi / 3.0
            for step in range(ring):
                # 沿六边形环方向生成中心
                cx = ring * dx * np.cos(angle) + step * dx * np.cos(angle + 2 * np.pi / 3.0)
                cy = ring * dy * np.sin(angle) + step * dy * np.sin(angle + 2 * np.pi / 3.0)
                centers.append(np.array([cx, cy]))
    centers = np.array(centers)
    # 去重（浮点容差）
    uniq = []
    for c in centers:
        if not any(np.linalg.norm(c - u) < 1e-9 for u in uniq):
            uniq.append(c)
    centers = np.array(uniq)
    verts_list = []
    for c in centers:
        verts = regular_hexagon_vertices(cell_size) + c
        verts_list.append(verts)
    verts_list = np.array(verts_list)
    return centers, verts_list


def hyperball_monomial_integral(dim: int, exponents: Tuple[int, ...],
                                radius: float = 1.0) -> float:
    """
    基于 553_hyperball_integrals 的 Folland 公式，计算 M 维超球内单项式精确积分：
        I = ∫_{||x||≤r} ∏ x_i^{e_i} dV
    
    若任一 e_i 为奇数，由对称性 I = 0。
    否则：
        I = 2 · r^{S} / S · ∏ Γ((e_i+1)/2) / Γ(S/2)
    其中 S = Σ (e_i + 1)。
    """
    exponents = tuple(int(e) for e in exponents)
    if len(exponents) != dim:
        raise ValueError("指数元组长度必须与维度一致")
    if any(e < 0 for e in exponents):
        raise ValueError("指数必须非负")
    if any(e % 2 == 1 for e in exponents):
        return 0.0
    S = sum(e + 1 for e in exponents)
    # 使用 Gamma 函数
    numerator = 1.0
    for e in exponents:
        numerator *= Gamma(0.5 * (e + 1))
    denominator = Gamma(0.5 * S)
    val = 2.0 * (radius ** S) / S * numerator / denominator
    return float(val)


def section_property_monte_carlo(polygon_vertices: np.ndarray,
                                  n_samples: int = 50000) -> dict:
    """
    对任意多边形截面，使用 Monte Carlo 均匀采样估计几何特性。
    
    采用 rejection sampling：在包围盒内均匀撒点，保留在多边形内的点，
    再用这些点估计面积、形心、惯性矩。
    
    数学公式：
        A      ≈ (N_in / N_total) · A_bbox
        Cy     ≈ (1/N_in) Σ y_i
        I_z    ≈ (A / N_in) Σ y_i²
        I_y    ≈ (A / N_in) Σ z_i²
        J      = I_y + I_z
    
    参数
    ----
    polygon_vertices : (n, 2) 多边形顶点（逆时针闭合或开放）
    n_samples : Monte Carlo 采样数
    
    返回
    ----
    props : 包含 'area', 'centroid_y', 'centroid_z', 'I_y', 'I_z', 'J' 的字典
    """
    poly = np.asarray(polygon_vertices, dtype=np.float64)
    if poly.shape[1] != 2:
        raise ValueError("顶点必须为二维")
    # 计算包围盒
    ymin, ymax = poly[:, 0].min(), poly[:, 0].max()
    zmin, zmax = poly[:, 1].min(), poly[:, 1].max()
    area_bbox = (ymax - ymin) * (zmax - zmin)
    if area_bbox <= 0:
        raise ValueError("包围盒面积为零")

    def point_in_polygon(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
        """Ray-casting 算法判断点是否在多边形内。"""
        n = len(poly)
        inside = np.zeros(len(points), dtype=bool)
        x, y = points[:, 0], points[:, 1]
        for i in range(n):
            j = (i + 1) % n
            xi, yi = poly[i]
            xj, yj = poly[j]
            # 检查边是否跨越水平线 y
            intersect = ((yi > y) != (yj > y)) & \
                        (x < (xj - xi) * (y - yi) / (yj - yi + 1e-18) + xi)
            inside ^= intersect
        return inside

    # Rejection sampling：批量生成
    batch = n_samples
    pts = np.random.uniform([ymin, zmin], [ymax, zmax], size=(batch, 2))
    mask = point_in_polygon(pts, poly)
    accepted = pts[mask]
    n_in = accepted.shape[0]
    if n_in == 0:
        raise RuntimeError("Monte Carlo 采样全部落在多边形外，请检查顶点顺序或范围")
    area_est = area_bbox * n_in / batch
    centroid = accepted.mean(axis=0)
    I_z = area_est * np.mean(accepted[:, 0] ** 2)
    I_y = area_est * np.mean(accepted[:, 1] ** 2)
    J = I_y + I_z
    return {
        "area": float(area_est),
        "centroid_y": float(centroid[0]),
        "centroid_z": float(centroid[1]),
        "I_y": float(I_y),
        "I_z": float(I_z),
        "J": float(J),
        "n_accepted": int(n_in)
    }


def equivalent_honeycomb_properties(cell_size: float, wall_thickness: float,
                                    E_s: float, rho_s: float) -> dict:
    """
    基于 Gibson-Ashby 理论计算正六蜂窝夹芯的等效工程常数。
    
    对规则六边形蜂窝（壁厚 t，边长 l）：
        相对密度    ρ*/ρ_s = (2/√3) · (t/l) · [1 - t/(2l)]
        面内模量    E₁*/E_s ≈ (4/√3) · (t/l)³ · [1 + 3(t/l)²]^{-1}
        面外剪切    G₁₃*/E_s ≈ (1/√3) · (t/l) · [1 + 3(t/l)²]^{-1}
    
    参数
    ----
    cell_size : 蜂窝边长 l [m]
    wall_thickness : 壁厚 t [m]
    E_s : 基材弹性模量 [Pa]
    rho_s : 基材密度 [kg/m³]
    
    返回
    ----
    dict : 等效密度、等效模量、等效剪切模量
    """
    l = cell_size
    t = wall_thickness
    if t >= l:
        raise ValueError("壁厚不能大于等于边长")
    ratio = t / l
    rho_star = rho_s * (2.0 / np.sqrt(3.0)) * ratio * (1.0 - 0.5 * ratio)
    E_star = E_s * (4.0 / np.sqrt(3.0)) * (ratio ** 3) / (1.0 + 3.0 * ratio ** 2)
    G_star = E_s * (1.0 / np.sqrt(3.0)) * ratio / (1.0 + 3.0 * ratio ** 2)
    return {
        "rho_star": float(rho_star),
        "E_star": float(E_star),
        "G_star": float(G_star),
        "relative_density": float(rho_star / rho_s)
    }
