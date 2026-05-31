"""
geometry_utils.py
=================
几何处理与边界条件工具。

融合种子项目：
  - 785_naca         : NACA 4位对称翼型生成
  - 1298_triangle_analyze : 三角形几何分析（面积、角度、外接圆、内切圆等）
  - 1294_tri_surface_to_obj : 三角面元表面数据处理

科学应用：
  在声学工程中，NACA翼型表面作为非定常流动的声学边界；
  三角形几何分析用于网格质量评估与自适应加密判据；
  三角面元处理用于3D声学散射体的表面离散化。
"""

import numpy as np


# =====================================================================
# NACA 翼型 (融合 785_naca/naca4_symmetric.m)
# =====================================================================

def naca4_symmetric(t, c, x):
    r"""
    NACA 4位对称翼型的厚度分布。

    .. math::
        y_t(x) = 5 t c \left[
        0.2969 \sqrt{\frac{x}{c}}
        - 0.1260 \frac{x}{c}
        - 0.3516 \left(\frac{x}{c}\right)^2
        + 0.2843 \left(\frac{x}{c}\right)^3
        - 0.1015 \left(\frac{x}{c}\right)^4
        \right]

    原始代码：785_naca/naca4_symmetric.m。

    Parameters
    ----------
    t : float
        最大相对厚度（如 0.12 表示 12%）。
    c : float
        弦长。
    x : float or np.ndarray
        沿弦长方向的坐标，:math:`0 \le x \le c`。

    Returns
    -------
    float or np.ndarray
        厚度 y(x)。
    """
    x = np.asarray(x, dtype=float)
    if np.any(x < -1e-9) or np.any(x > c + 1e-9):
        raise ValueError("x must be in [0, c].")
    x = np.clip(x, 0.0, c)
    xi = x / c
    y = 5.0 * t * c * (
        0.2969 * np.sqrt(xi)
        + ((((-0.1015) * xi + 0.2843) * xi - 0.3516) * xi - 0.1260) * xi
    )
    return y


def naca4_cambered(m, p, t, c, x):
    r"""
    NACA 4位弯度翼型。

    .. math::
        y_c(x) = \begin{cases}
        \frac{m}{p^2} (2 p \frac{x}{c} - (\frac{x}{c})^2), & 0 \le x \le p c \\
        \frac{m}{(1-p)^2} ((1-2p) + 2p \frac{x}{c} - (\frac{x}{c})^2), & p c \le x \le c
        \end{cases}

    Parameters
    ----------
    m : float
        最大弯度（弦长比例）。
    p : float
        最大弯度位置（弦长比例）。
    t : float
        最大相对厚度。
    c : float
        弦长。
    x : float or np.ndarray
        弦向坐标。

    Returns
    -------
    np.ndarray, shape (N, 2)
        上表面和下表面坐标 [x, y]。
    """
    x = np.asarray(x, dtype=float)
    if np.any(x < -1e-9) or np.any(x > c + 1e-9):
        raise ValueError("x must be in [0, c].")
    x = np.clip(x, 0.0, c)
    xi = x / c

    # 弯度线
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    mask = xi <= p
    if p > 0.0:
        yc[mask] = (m / p ** 2) * (2.0 * p * xi[mask] - xi[mask] ** 2)
        dyc_dx[mask] = (2.0 * m / p ** 2) * (p - xi[mask])
    if p < 1.0:
        yc[~mask] = (m / (1.0 - p) ** 2) * (
            (1.0 - 2.0 * p) + 2.0 * p * xi[~mask] - xi[~mask] ** 2)
        dyc_dx[~mask] = (2.0 * m / (1.0 - p) ** 2) * (p - xi[~mask])

    yt = naca4_symmetric(t, c, x)

    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    upper = np.column_stack((xu, yu))
    lower = np.column_stack((xl, yl))
    return upper, lower


def generate_naca_airfoil_points(t, c, n_points=100, m=0.0, p=0.0):
    """
    生成 NACA 翼型表面点序列。

    Parameters
    ----------
    t : float
        厚度比例。
    c : float
        弦长。
    n_points : int
        每面点数。
    m, p : float
        弯度参数（m=0 为对称翼型）。

    Returns
    -------
    np.ndarray
        有序的表面点 (从尾缘出发，沿上表面到前缘，再沿下表面回尾缘)。
    """
    # 在 x 上加密：前缘附近更密
    theta = np.linspace(0.0, np.pi, n_points)
    x = 0.5 * c * (1.0 - np.cos(theta))
    x = np.clip(x, 0.0, c)

    if m == 0.0:
        y = naca4_symmetric(t, c, x)
        upper = np.column_stack((x, y))
        lower = np.column_stack((x, -y))
    else:
        upper, lower = naca4_cambered(m, p, t, c, x)

    # 组合：尾缘 -> 前缘(上) -> 前缘(下) -> 尾缘
    surface = np.vstack([
        upper[::-1],
        lower[1:, :]
    ])
    return surface


# =====================================================================
# 三角形几何分析 (融合 1298_triangle_analyze)
# =====================================================================

def triangle_area(node_xy):
    """
    三角形有向面积。

    .. math::
        A = \frac{1}{2} |x_1(y_2 - y_3) + x_2(y_3 - y_1) + x_3(y_1 - y_2)|

    Parameters
    ----------
    node_xy : np.ndarray, shape (3, 2)
        三个顶点坐标。

    Returns
    -------
    float
        面积。
    """
    node_xy = np.asarray(node_xy, dtype=float)
    if node_xy.shape != (3, 2):
        raise ValueError("node_xy must have shape (3, 2).")
    area = 0.5 * abs(
        node_xy[0, 0] * (node_xy[1, 1] - node_xy[2, 1]) +
        node_xy[1, 0] * (node_xy[2, 1] - node_xy[0, 1]) +
        node_xy[2, 0] * (node_xy[0, 1] - node_xy[1, 1])
    )
    return area


def triangle_angles(node_xy):
    """
    三角形内角（弧度）。

    .. math::
        \cos A = \frac{b^2 + c^2 - a^2}{2bc}

    Parameters
    ----------
    node_xy : np.ndarray, shape (3, 2)

    Returns
    -------
    np.ndarray, shape (3,)
        三个内角。
    """
    node_xy = np.asarray(node_xy, dtype=float)
    # 边长
    a = np.linalg.norm(node_xy[1] - node_xy[2])
    b = np.linalg.norm(node_xy[0] - node_xy[2])
    c = np.linalg.norm(node_xy[0] - node_xy[1])

    angles = np.zeros(3, dtype=float)
    if a > 0.0 and b > 0.0 and c > 0.0:
        angles[0] = np.arccos(np.clip((b ** 2 + c ** 2 - a ** 2) / (2.0 * b * c), -1.0, 1.0))
        angles[1] = np.arccos(np.clip((a ** 2 + c ** 2 - b ** 2) / (2.0 * a * c), -1.0, 1.0))
        angles[2] = np.arccos(np.clip((a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b), -1.0, 1.0))
    return angles


def triangle_circumcircle(node_xy):
    """
    三角形外接圆半径与圆心。

    .. math::
        R = \frac{abc}{4A}

    Parameters
    ----------
    node_xy : np.ndarray, shape (3, 2)

    Returns
    -------
    float
        外接圆半径。
    np.ndarray, shape (2,)
        圆心坐标。
    """
    node_xy = np.asarray(node_xy, dtype=float)
    A = triangle_area(node_xy)
    a = np.linalg.norm(node_xy[1] - node_xy[2])
    b = np.linalg.norm(node_xy[0] - node_xy[2])
    c = np.linalg.norm(node_xy[0] - node_xy[1])
    if A <= 1e-14:
        return np.inf, node_xy[0].copy()
    R = a * b * c / (4.0 * A)

    # 圆心公式
    D = 2.0 * (
        node_xy[0, 0] * (node_xy[1, 1] - node_xy[2, 1]) +
        node_xy[1, 0] * (node_xy[2, 1] - node_xy[0, 1]) +
        node_xy[2, 0] * (node_xy[0, 1] - node_xy[1, 1])
    )
    if abs(D) < 1e-14:
        return np.inf, node_xy[0].copy()

    ux = (
        (node_xy[0, 0] ** 2 + node_xy[0, 1] ** 2) * (node_xy[1, 1] - node_xy[2, 1]) +
        (node_xy[1, 0] ** 2 + node_xy[1, 1] ** 2) * (node_xy[2, 1] - node_xy[0, 1]) +
        (node_xy[2, 0] ** 2 + node_xy[2, 1] ** 2) * (node_xy[0, 1] - node_xy[1, 1])
    ) / D
    uy = (
        (node_xy[0, 0] ** 2 + node_xy[0, 1] ** 2) * (node_xy[2, 0] - node_xy[1, 0]) +
        (node_xy[1, 0] ** 2 + node_xy[1, 1] ** 2) * (node_xy[0, 0] - node_xy[2, 0]) +
        (node_xy[2, 0] ** 2 + node_xy[2, 1] ** 2) * (node_xy[1, 0] - node_xy[0, 0])
    ) / D

    return R, np.array([ux, uy])


def triangle_incircle(node_xy):
    """
    三角形内切圆半径与圆心。

    .. math::
        r = \frac{2A}{a+b+c}

    Parameters
    ----------
    node_xy : np.ndarray, shape (3, 2)

    Returns
    -------
    float
        内切圆半径。
    np.ndarray, shape (2,)
        内心坐标。
    """
    node_xy = np.asarray(node_xy, dtype=float)
    A = triangle_area(node_xy)
    a = np.linalg.norm(node_xy[1] - node_xy[2])
    b = np.linalg.norm(node_xy[0] - node_xy[2])
    c = np.linalg.norm(node_xy[0] - node_xy[1])
    perim = a + b + c
    if perim <= 1e-14:
        return 0.0, np.mean(node_xy, axis=0)
    r = 2.0 * A / perim
    center = (a * node_xy[0] + b * node_xy[1] + c * node_xy[2]) / perim
    return r, center


def triangle_quality(node_xy):
    r"""
    三角形质量因子。

    .. math::
        q = 4 \sqrt{3} \frac{A}{a^2 + b^2 + c^2}

    等边三角形 q=1，退化三角形 q->0。

    Parameters
    ----------
    node_xy : np.ndarray, shape (3, 2)

    Returns
    -------
    float
        质量因子 [0, 1]。
    """
    node_xy = np.asarray(node_xy, dtype=float)
    A = triangle_area(node_xy)
    a = np.linalg.norm(node_xy[1] - node_xy[2])
    b = np.linalg.norm(node_xy[0] - node_xy[2])
    c = np.linalg.norm(node_xy[0] - node_xy[1])
    denom = a ** 2 + b ** 2 + c ** 2
    if denom <= 1e-14:
        return 0.0
    q = 4.0 * np.sqrt(3.0) * A / denom
    return float(np.clip(q, 0.0, 1.0))


# =====================================================================
# 三角面元表面处理 (融合 1294_tri_surface_to_obj)
# =====================================================================

def compute_surface_normals(nodes, triangles):
    """
    计算三角面元的法向量。

    Parameters
    ----------
    nodes : np.ndarray, shape (N_nodes, 3)
        节点坐标。
    triangles : np.ndarray, shape (N_tri, 3)
        三角形节点索引（0-based）。

    Returns
    -------
    np.ndarray, shape (N_tri, 3)
        单位法向量。
    """
    nodes = np.asarray(nodes, dtype=float)
    triangles = np.asarray(triangles, dtype=int)

    v0 = nodes[triangles[:, 0], :]
    v1 = nodes[triangles[:, 1], :]
    v2 = nodes[triangles[:, 2], :]

    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e1, e2)

    norms = np.linalg.norm(normals, axis=1)
    norms = np.where(norms < 1e-14, 1.0, norms)
    normals = normals / norms[:, np.newaxis]
    return normals


def tri_surface_area(nodes, triangles):
    """
    计算三角化表面的总面积。

    Parameters
    ----------
    nodes : np.ndarray, shape (N, 3)
    triangles : np.ndarray, shape (M, 3)

    Returns
    -------
    float
        总面积。
    """
    nodes = np.asarray(nodes, dtype=float)
    triangles = np.asarray(triangles, dtype=int)

    total = 0.0
    for tri in triangles:
        pts = nodes[tri, :2]  # 投影到 xy 平面或直接使用 3D
        a = np.linalg.norm(nodes[tri[1]] - nodes[tri[0]])
        b = np.linalg.norm(nodes[tri[2]] - nodes[tri[1]])
        c = np.linalg.norm(nodes[tri[0]] - nodes[tri[2]])
        s = 0.5 * (a + b + c)
        area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))
        total += area
    return total


class AcousticBoundary:
    """
    声学边界管理器：支持 NACA 翼型、三角化表面等。
    """

    def __init__(self, boundary_type='naca'):
        self.boundary_type = boundary_type
        self.surface_points = None
        self.triangles = None
        self.normals = None

    def generate_naca_boundary(self, t=0.12, c=1.0, n_points=200, m=0.0, p=0.0,
                                angle_of_attack=0.0):
        """
        生成 NACA 翼型边界。

        Parameters
        ----------
        t, c, n_points, m, p : 见 generate_naca_airfoil_points。
        angle_of_attack : float
            攻角（度）。
        """
        surf = generate_naca_airfoil_points(t, c, n_points, m, p)
        if angle_of_attack != 0.0:
            theta = np.radians(angle_of_attack)
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
            surf = (rot @ surf.T).T
        self.surface_points = surf
        return surf

    def reflect_acoustic_rays(self, ray_origins, ray_directions):
        """
        计算声学射线在边界上的反射（镜面反射近似）。

        .. math::
            d_{reflected} = d - 2 (d \cdot n) n

        Parameters
        ----------
        ray_origins : np.ndarray, shape (N, 2)
            射线起点。
        ray_directions : np.ndarray, shape (N, 2)
            射线方向（单位向量）。

        Returns
        -------
        np.ndarray
            反射后方向。
        """
        if self.surface_points is None:
            raise ValueError("Boundary not initialized.")

        # 简化为最近边界点的法向量
        reflected = np.zeros_like(ray_directions)
        for i in range(ray_origins.shape[0]):
            origin = ray_origins[i]
            d = ray_directions[i]
            # 找最近表面点
            dists = np.sum((self.surface_points - origin) ** 2, axis=1)
            idx = np.argmin(dists)
            pt = self.surface_points[idx]

            # 局部法向量（指向外部）
            if idx < len(self.surface_points) - 1:
                tangent = self.surface_points[idx + 1] - pt
            else:
                tangent = pt - self.surface_points[idx - 1]
            tangent = tangent / (np.linalg.norm(tangent) + 1e-14)
            normal = np.array([-tangent[1], tangent[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-14)

            # 确保法向量指向射线来向
            if np.dot(d, normal) > 0.0:
                normal = -normal

            reflected[i] = d - 2.0 * np.dot(d, normal) * normal
            reflected[i] = reflected[i] / (np.linalg.norm(reflected[i]) + 1e-14)

        return reflected

    def mesh_quality_statistics(self, tri_nodes):
        """
        对一组三角形进行质量统计。

        Parameters
        ----------
        tri_nodes : list of np.ndarray
            每个元素 shape (3, 2)。

        Returns
        -------
        dict
            质量统计信息。
        """
        qualities = []
        areas = []
        for tri in tri_nodes:
            q = triangle_quality(tri)
            a = triangle_area(tri)
            qualities.append(q)
            areas.append(a)

        qualities = np.array(qualities)
        areas = np.array(areas)

        return {
            'min_quality': float(np.min(qualities)),
            'max_quality': float(np.max(qualities)),
            'mean_quality': float(np.mean(qualities)),
            'min_area': float(np.min(areas)),
            'max_area': float(np.max(areas)),
            'total_area': float(np.sum(areas))
        }
