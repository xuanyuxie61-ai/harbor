"""
海洋平台水动力系数与波浪荷载计算模块

基于种子项目：
  - 882_polygon：多边形几何、面积、形心、惯性矩、固角
  - 1080_simplex_integrals：单纯形精确积分

核心物理模型：
  1. 势流理论面板法（Boundary Element Method）：
       假设流体无粘、无旋、不可压缩，速度势 φ 满足 Laplace 方程：
           ∇²φ = 0
       边界条件：
           - 自由表面（线性化）：∂φ/∂z = (ω²/g) φ
           - 物面：∂φ/∂n = V_n
           - 底部：∂φ/∂n = 0
           - 远场：辐射条件

  2. 格林函数（三维自由表面）：
       G(x, ξ) = 1/|x - ξ| + 1/|x - ξ*|
       其中 ξ* 为 ξ 关于静水面的镜像点。

  3. 绕射问题（入射波被平台散射）：
       φ = φ_I + φ_D
       物面条件：∂φ_D/∂n = -∂φ_I/∂n
       入射波势（线性 Airy 波）：
           φ_I = (A g/ω) · cosh[k(z+h)]/cosh(kh) · sin(kx cosβ + ky sinβ - ωt)

  4. 辐射问题（平台强迫振荡产生波浪）：
       辐射势 φ_j 满足物面条件：∂φ_j/∂n = n_j  (j = 1..6)
       附加质量：A_ij(ω) = -ρ Re[ ∬_S φ_j n_i dS ]
       辐射阻尼：B_ij(ω) =  ρ ω Im[ ∬_S φ_j n_i dS ]

  5. 波浪力（Froude-Krylov + 绕射）：
       F_i = iωρ ∬_S (φ_I + φ_D) n_i dS

  6. Morison 方程（用于小构件 drag/inertia 力）：
       dF = 0.5 ρ C_d D |u - ξ˙| (u - ξ˙) dz + ρ C_m (πD²/4) (du/dt) dz
"""

import numpy as np
from typing import Tuple, List, Optional
from mesh_geometry import polygon_area_2d, polygon_centroid_2d, polygon_solid_angle_3d


# ======================================================================
# 1. 面板法基础：格林函数与影响系数
# ======================================================================

def green_function_3d(
    x: np.ndarray, xi: np.ndarray, use_image: bool = True
) -> float:
    """
    三维格林函数：G(x, ξ) = 1/|x - ξ| + 1/|x - ξ*|
    ξ* 为 ξ 关于 z=0 平面的镜像点 (x, y, -z)。
    """
    x = np.asarray(x, dtype=float)
    xi = np.asarray(xi, dtype=float)
    r = np.linalg.norm(x - xi)
    if r < 1e-12:
        return 0.0
    val = 1.0 / r
    if use_image:
        xi_star = xi.copy()
        xi_star[2] = -xi[2]
        r_star = np.linalg.norm(x - xi_star)
        if r_star > 1e-12:
            val += 1.0 / r_star
    return val


def panel_normal_3d(vertices: np.ndarray) -> np.ndarray:
    """
    计算三维多边形面板的单位法向量（右手定则）。
    取前三个顶点构成的两条边叉乘。
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape[0] < 3:
        return np.array([0.0, 0.0, 1.0])
    v1 = vertices[1] - vertices[0]
    v2 = vertices[2] - vertices[0]
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-15:
        return np.array([0.0, 0.0, 1.0])
    return n / norm


def panel_area_3d(vertices: np.ndarray) -> float:
    """
    计算三维平面多边形面积。
    投影到最佳平面后使用 Shoelace 公式。
    """
    vertices = np.asarray(vertices, dtype=float)
    n = panel_normal_3d(vertices)
    # 找到最大分量轴作为投影方向
    abs_n = np.abs(n)
    proj_axis = np.argmax(abs_n)
    # 投影到另外两个轴构成的平面
    coords = np.delete(vertices, proj_axis, axis=1)
    return abs(polygon_area_2d(coords))


def panel_centroid_3d(vertices: np.ndarray) -> np.ndarray:
    """计算三维多边形面板的形心。"""
    vertices = np.asarray(vertices, dtype=float)
    n = panel_normal_3d(vertices)
    abs_n = np.abs(n)
    proj_axis = np.argmax(abs_n)
    coords = np.delete(vertices, proj_axis, axis=1)
    c2d = polygon_centroid_2d(coords)
    # 恢复三维坐标（投影轴取平均）
    centroid = np.zeros(3)
    other_axes = [i for i in range(3) if i != proj_axis]
    centroid[other_axes[0]] = c2d[0]
    centroid[other_axes[1]] = c2d[1]
    centroid[proj_axis] = np.mean(vertices[:, proj_axis])
    return centroid


# ======================================================================
# 2. 水动力系数计算（简化面板法）
# ======================================================================

def compute_hydrodynamic_coefficients_panel_method(
    panels: List[np.ndarray],
    omega: float,
    rho: float = 1025.0,
    h: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用简化面板法计算附加质量 A(ω)、辐射阻尼 B(ω) 和波浪力传递函数。
    假设：
      - 面板尺寸远小于波长，使用点源近似。
      - 忽略自由表面格林函数的复杂积分，使用镜像源近似。
      - 辐射势在物面上近似为 φ_j ≈ n_j / (ik)，k 为波数。

    参数
    ----
    panels : 三维面板顶点列表，每个面板为 n×3 数组
    omega : 圆频率 (rad/s)
    rho : 流体密度
    h : 水深

    返回
    ----
    A : 6×6 附加质量矩阵
    B : 6×6 辐射阻尼矩阵
    F_tf : 6 维波浪力传递函数（单位波幅）
    """
    n_panels = len(panels)
    if n_panels == 0:
        return np.zeros((6, 6)), np.zeros((6, 6)), np.zeros(6)

    g = 9.80665
    k = omega * omega / g  # 深水近似

    # 计算每个面板的属性
    areas = np.zeros(n_panels)
    centroids = np.zeros((n_panels, 3))
    normals = np.zeros((n_panels, 3))
    for i, p in enumerate(panels):
        areas[i] = panel_area_3d(p)
        centroids[i] = panel_centroid_3d(p)
        normals[i] = panel_normal_3d(p)

    # 点源近似下的影响系数矩阵
    # 简化模型：φ_j(p) ≈ G(p, q_j) · σ_j · area_j
    # 源强 σ_j 满足：-2π σ_j + Σ_{k≠j} G(p_j, q_k) σ_k area_k = n_j·e_m
    # 这里使用简化对角近似

    # 附加质量（对角近似）：A_mm ≈ ρ Σ area_i · (n_i)_m² / k
    A = np.zeros((6, 6))
    B = np.zeros((6, 6))
    for i in range(n_panels):
        nvec = normals[i]
        area = areas[i]
        # 平移模式 (surge, sway, heave)
        for m in range(3):
            A[m, m] += rho * area * nvec[m] ** 2 / k
            B[m, m] += rho * omega * area * nvec[m] ** 2 / (k ** 2)
        # 旋转模式 (roll, pitch, yaw) — 使用 r × n
        r = centroids[i]
        rxn = np.cross(r, nvec)
        for m in range(3):
            A[m + 3, m + 3] += rho * area * rxn[m] ** 2 / k
            B[m + 3, m + 3] += rho * omega * area * rxn[m] ** 2 / (k ** 2)

    # 波浪力传递函数（Froude-Krylov 近似）
    F_tf = np.zeros(6)
    for i in range(n_panels):
        r = centroids[i]
        nvec = normals[i]
        area = areas[i]
        z = r[2]
        # 入射波压力幅值（单位波幅）
        p_amp = rho * g * np.cosh(k * (z + h)) / np.cosh(k * h)
        # 平移力
        F_tf[:3] += p_amp * area * nvec
        # 力矩
        F_tf[3:] += p_amp * area * np.cross(r, nvec)

    return A, B, F_tf


# ======================================================================
# 3. Morison 方程（立柱与撑杆水动力）
# ======================================================================

def morison_force_1d(
    u: float,
    u_dot: float,
    xi_dot: float,
    xi_ddot: float,
    D: float,
    C_d: float = 1.0,
    C_m: float = 2.0,
    rho: float = 1025.0,
    dz: float = 1.0,
) -> Tuple[float, float]:
    """
    一维 Morison 方程计算单位长度构件上的波浪力。
    dF_drag = 0.5 ρ C_d D |u - ξ˙| (u - ξ˙) dz
    dF_inertia = ρ C_m (πD²/4) (du/dt - ξ¨) dz
    返回 (drag_force, inertia_force)。
    """
    rel_vel = u - xi_dot
    drag = 0.5 * rho * C_d * D * abs(rel_vel) * rel_vel * dz
    inertia = rho * C_m * (np.pi * D * D / 4.0) * (u_dot - xi_ddot) * dz
    return drag, inertia


def morison_force_on_platform_column(
    wave_kinematics: dict,
    platform_motion: dict,
    column_diameter: float = 15.0,
    draft: float = 20.0,
    z_nodes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    计算波浪作用在平台立柱上的总 Morison 力（6-DOF）。
    wave_kinematics: {'u': 水平速度数组, 'u_dot': 水平加速度数组, 'z': 深度坐标}
    platform_motion: {'xi_dot': 平台速度, 'xi_ddot': 平台加速度}
    返回 [Fx, Fy, Fz, Mx, My, Mz]。
    """
    if z_nodes is None:
        z_nodes = np.linspace(-draft, 0.0, 41)
    dz = z_nodes[1] - z_nodes[0] if len(z_nodes) > 1 else 1.0
    F = np.zeros(6)
    u = wave_kinematics.get("u", np.zeros_like(z_nodes))
    u_dot = wave_kinematics.get("u_dot", np.zeros_like(z_nodes))
    xi_dot = platform_motion.get("xi_dot", 0.0)
    xi_ddot = platform_motion.get("xi_ddot", 0.0)
    for idx, z in enumerate(z_nodes):
        ui = u[idx] if idx < len(u) else 0.0
        udi = u_dot[idx] if idx < len(u_dot) else 0.0
        fd, fi = morison_force_1d(ui, udi, xi_dot, xi_ddot, column_diameter, dz=abs(dz))
        F[0] += fd + fi
        # 力矩
        F[4] += (fd + fi) * z
    return F


# ======================================================================
# 4. 半潜平台简化几何面板生成
# ======================================================================

def generate_semi_submersible_panels(
    col_spacing_x: float = 55.0,
    col_spacing_y: float = 40.0,
    col_diameter: float = 15.0,
    col_height: float = 20.0,
    pontoon_width: float = 10.0,
    pontoon_height: float = 8.0,
    n_azimuth: int = 16,
) -> List[np.ndarray]:
    """
    生成简化半潜式平台（四立柱 + 下浮体）的三维面板模型。
    每个立柱离散为 n_azimuth 个四边形面板，下浮体离散为长方体面板。
    返回面板顶点列表。
    """
    panels = []
    # 立柱位置
    col_positions = [
        (-col_spacing_x * 0.5, -col_spacing_y * 0.5, -col_height * 0.5),
        (-col_spacing_x * 0.5, col_spacing_y * 0.5, -col_height * 0.5),
        (col_spacing_x * 0.5, -col_spacing_y * 0.5, -col_height * 0.5),
        (col_spacing_x * 0.5, col_spacing_y * 0.5, -col_height * 0.5),
    ]
    # 生成立柱面板
    theta = np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False)
    dtheta = theta[1] - theta[0]
    for cx, cy, cz in col_positions:
        r = col_diameter * 0.5
        for t in theta:
            t2 = t + dtheta
            x1 = cx + r * np.cos(t)
            y1 = cy + r * np.sin(t)
            x2 = cx + r * np.cos(t2)
            y2 = cy + r * np.sin(t2)
            # 立柱侧面面板（四边形）
            panel = np.array(
                [
                    [x1, y1, cz - col_height * 0.5],
                    [x2, y2, cz - col_height * 0.5],
                    [x2, y2, cz + col_height * 0.5],
                    [x1, y1, cz + col_height * 0.5],
                ],
                dtype=float,
            )
            panels.append(panel)

    # 下浮体（pontoon）— 简化连接相邻立柱的矩形管道
    connections = [
        (0, 1),
        (2, 3),
        (0, 2),
        (1, 3),
    ]
    for i, j in connections:
        c1 = np.array(col_positions[i])
        c2 = np.array(col_positions[j])
        mid = 0.5 * (c1 + c2)
        length = np.linalg.norm(c2[:2] - c1[:2])
        direction = (c2[:2] - c1[:2]) / length
        normal = np.array([-direction[1], direction[0]])
        w = pontoon_width * 0.5
        h = pontoon_height * 0.5
        # 四个侧面面板
        corners = [
            mid[:2] + w * normal + np.array([0, 0]),
            mid[:2] - w * normal + np.array([0, 0]),
        ]
        for sign in [-1, 1]:
            z_bottom = mid[2] - h
            z_top = mid[2] + h
            for ci in range(2):
                c = corners[ci]
                c_next = corners[(ci + 1) % 2]
                panel = np.array(
                    [
                        [c[0], c[1], z_bottom],
                        [c_next[0], c_next[1], z_bottom],
                        [c_next[0], c_next[1], z_top],
                        [c[0], c[1], z_top],
                    ],
                    dtype=float,
                )
                panels.append(panel)
    return panels


# ======================================================================
# 5. 波浪运动学（Airy 线性波）
# ======================================================================

def airy_wave_kinematics(
    x: float,
    z: float,
    t: float,
    A: float,
    T: float,
    h: float,
    beta: float = 0.0,
) -> dict:
    """
    计算 Airy 线性波的波浪运动学量。
    返回字典：eta, u, w, u_dot, w_dot, p_dyn
    """
    # TODO: 实现 Airy 线性波运动学计算
    # 关键物理：
    #   深水色散关系：k = ω² / g
    #   速度势：φ = (Ag/ω) · cosh[k(z+h)]/cosh(kh) · sin(kx cosβ + ky sinβ - ωt)
    #   水平速度 u = ∂φ/∂x
    #   垂向速度 w = ∂φ/∂z
    #   动压力 p_dyn = -ρ ∂φ/∂t
    raise NotImplementedError("airy_wave_kinematics 需要实现")


rho = 1025.0  # 海水密度全局默认值
