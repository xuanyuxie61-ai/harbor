"""
dalitz_geometry.py
------------------
三体 B 衰变 Dalitz 图的几何构造与 FEM 网格映射。
映射自种子项目 381_fem_to_triangle (FEM 节点/单元格式转换) 和
580_image_mesh2d (边界点 → 内部三角网格)。

物理背景:
  对于 B → h1 h2 h3 三体衰变, 定义子不变质量平方
      s_12 = (p_1 + p_2)^2,   s_13 = (p_1 + p_3)^2,   s_23 = (p_2 + p_3)^2
  满足能量-动量守恒
      s_12 + s_13 + s_23 = m_B^2 + m_1^2 + m_2^2 + m_3^2 := sigma_total

  在 (s_12, s_13) 平面上, 物理允许区域为一条闭合边界围成的"曲边三角形"。
  本模块:
    1) 由边界采样点生成曲边三角区域的 FEM 节点和单元 (仿 580);
    2) 在参考三角形与物理 Dalitz 区域之间做仿射映射 (仿 381);
    3) 计算边界上每一点的法向量, 供后续有限差分边界条件使用。

数学公式:
  对于 B(p_B) → p_1 + p_2 + p_3, 在 B 静止系中:
      E_i* = (m_B^2 + m_i^2 - s_jk) / (2 m_B),   {i,j,k} 为 {1,2,3} 的置换
  两粒子动量 (Källén 函数):
      |p_i*| = sqrt( lambda(s_jk, m_j^2, m_k^2) ) / (2 sqrt(s_jk))
  边界由 cos(theta_12) = ±1 给出:
      s_13^{±}(s_12) = m_1^2 + m_3^2
                       + (E_1* E_3* ∓ |p_1*| |p_3*|) * 2
  其中 E_i* 和 |p_i*| 均在 s_12 固定的子系统中计算。
"""

from __future__ import annotations
import math
from typing import List, Tuple

from b_physics_constants import kallen, kallen_sqrt


# -------------------------------------------------------------------------- #
#                          Dalitz 边界计算                                  #
# -------------------------------------------------------------------------- #
def dalitz_boundary(
        m_parent: float,
        m1: float,
        m2: float,
        m3: float,
        n_boundary: int = 180,
) -> List[Tuple[float, float]]:
    """
    返回 (s_12, s_13) 平面上的物理允许区域的边界点序列。
    沿 s_12 方向均匀采样 n_boundary 个点, 对每个点计算上、下边界 s_13^+, s_13^-。

    公式 (PDG Dalitz plot kinematics):
        s_12^{min} = (m_1 + m_2)^2,    s_12^{max} = (m_B - m_3)^2
        s_13^{±}(s_12) = m_1^2 + m_3^2
                         + 0.5 * ( (m_B^2 - m_2^2 - s_12)(s_12 + m_1^2 - m_2^2)/s_12
                                   ∓ Lambda^{1/2}(s_12, m_1^2, m_2^2)
                                     * Lambda^{1/2}(m_B^2, s_12, m_3^2) / s_12 )
    """
    if m_parent <= m1 + m2 + m3:
        raise ValueError("父粒子质量必须大于三体阈值")
    s12_min = (m1 + m2) ** 2
    s12_max = (m_parent - m3) ** 2
    if s12_max <= s12_min:
        raise ValueError("物理允许区间为空, 检查质量输入")

    mB2 = m_parent * m_parent
    m1_2 = m1 * m1
    m2_2 = m2 * m2
    m3_2 = m3 * m3

    pts_lower: List[Tuple[float, float]] = []
    pts_upper: List[Tuple[float, float]] = []

    for i in range(n_boundary + 1):
        s12 = s12_min + (s12_max - s12_min) * i / n_boundary
        lam12 = kallen(s12, m1_2, m2_2)
        lamB = kallen(mB2, s12, m3_2)
        if lam12 <= 0.0 or lamB <= 0.0 or s12 <= 0.0:
            continue
        num1 = (mB2 - m2_2 - s12) * (s12 + m1_2 - m2_2)
        num2 = math.sqrt(max(lam12, 0.0)) * math.sqrt(max(lamB, 0.0))
        denom = 2.0 * s12
        center = m1_2 + m3_2 + num1 / denom
        half_span = num2 / denom
        s13_lo = center - half_span
        s13_hi = center + half_span
        pts_lower.append((s12, s13_lo))
        pts_upper.append((s12, s13_hi))

    # 合并为闭合多边形: 下边界正向 + 上边界反向
    pts_upper.reverse()
    boundary = pts_lower + pts_upper
    return boundary


def dalitz_extent(
        m_parent: float, m1: float, m2: float, m3: float,
) -> Tuple[float, float, float, float]:
    """
    返回 (s_12^{min}, s_12^{max}, s_13^{min}, s_13^{max})
    用于构造外接矩形, 供有限差分网格生成使用。
    """
    s12_min = (m1 + m2) ** 2
    s12_max = (m_parent - m3) ** 2
    s13_min = (m1 + m3) ** 2
    s13_max = (m_parent - m2) ** 2
    return s12_min, s12_max, s13_min, s13_max


# -------------------------------------------------------------------------- #
#         参考三角形 ↔ 物理 Dalitz 区域的仿射映射 (仿 381_fem_to_triangle) #
# -------------------------------------------------------------------------- #
class DalitzAffineMap:
    """
    参考三角形 T_ref = { (0,0), (1,0), (0,1) } 到物理 Dalitz 三角形
    (以 s_12^{min}, s_12^{max}, s_13^{min} 为基准) 的仿射映射。

    设参考坐标 (xi, eta), 物理坐标 (s12, s13):
        s12(xi, eta) = s12_min + (s12_max - s12_min) * xi
        s13(xi, eta) = s13_min + (s13_max - s13_min) * eta
        Jacobian J = diag(s12_max - s12_min, s13_max - s13_min)
    对一般曲边 Dalitz 区域, 此映射为曲边三角形的外接矩形子集近似。
    """
    def __init__(self, m_parent: float, m1: float, m2: float, m3: float):
        self.s12_min, self.s12_max, self.s13_min, self.s13_max = \
            dalitz_extent(m_parent, m1, m2, m3)
        self.L12 = self.s12_max - self.s12_min
        self.L13 = self.s13_max - self.s13_min
        if self.L12 <= 0.0 or self.L13 <= 0.0:
            raise ValueError("Dalitz 区域退化, 无法构造仿射映射")
        self.Jacobian = self.L12 * self.L13   # |det(J)|
        self.Jacobian_inv = 1.0 / self.Jacobian

    def ref_to_dalitz(self, xi: float, eta: float) -> Tuple[float, float]:
        s12 = self.s12_min + self.L12 * xi
        s13 = self.s13_min + self.L13 * eta
        return s12, s13

    def dalitz_to_ref(self, s12: float, s13: float) -> Tuple[float, float]:
        xi  = (s12 - self.s12_min) / self.L12
        eta = (s13 - self.s13_min) / self.L13
        return xi, eta

    def in_physical_region(self, s12: float, s13: float,
                            m_parent: float, m1: float, m2: float, m3: float,
                            tol: float = 1.0e-10) -> bool:
        """判断 (s12, s13) 是否处于物理允许区域, 使用边界条件。"""
        s12_min = (m1 + m2) ** 2
        s12_max = (m_parent - m3) ** 2
        if s12 < s12_min - tol or s12 > s12_max + tol:
            return False
        mB2 = m_parent * m_parent
        m1_2, m2_2, m3_2 = m1 * m1, m2 * m2, m3 * m3
        lam12 = kallen(s12, m1_2, m2_2)
        lamB  = kallen(mB2, s12, m3_2)
        if lam12 <= 0.0 or lamB <= 0.0 or s12 <= 0.0:
            return False
        num1 = (mB2 - m2_2 - s12) * (s12 + m1_2 - m2_2)
        num2 = math.sqrt(max(lam12, 0.0)) * math.sqrt(max(lamB, 0.0))
        denom = 2.0 * s12
        s13_lo = m1_2 + m3_2 + (num1 - num2) / denom
        s13_hi = m1_2 + m3_2 + (num1 + num2) / denom
        return (s13 >= s13_lo - tol) and (s13 <= s13_hi + tol)


# -------------------------------------------------------------------------- #
#      Dalitz 区域的 FEM 三角网格 (仿 580_image_mesh2d)                    #
# -------------------------------------------------------------------------- #
def build_dalitz_fem_mesh(
        m_parent: float,
        m1: float,
        m2: float,
        m3: float,
        n_s12: int = 30,
        n_s13: int = 30,
) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int, int]]]:
    """
    在 Dalitz 物理区域上构造 FEM 三角形网格。
    算法 (仿 image_mesh2d):
      1. 在 (s12, s13) 外接矩形上生成 n_s12 × n_s13 的规则节点;
      2. 保留位于物理区域内的节点;
      3. 对保留节点作 Delaunay 风格的三角剖分: 对每对相邻 4 节点
         (i,j), (i+1,j), (i,j+1), (i+1,j+1) 生成两个三角形;
      4. 丢弃至少一个顶点在区域外的三角形。
    返回: (nodes, elements)
        nodes[i]     = (s12_i, s13_i)
        elements[k]  = (i0, i1, i2)  节点索引
    """
    s12_min, s12_max, s13_min, s13_max = dalitz_extent(m_parent, m1, m2, m3)
    h12 = (s12_max - s12_min) / max(n_s12, 1)
    h13 = (s13_max - s13_min) / max(n_s13, 1)

    # --- 第 1,2 步: 生成物理区域节点, 记录原网格索引 ---
    idx_map = {}     # (i_grid, j_grid) -> node_idx
    nodes: List[Tuple[float, float]] = []
    for i in range(n_s12 + 1):
        for j in range(n_s13 + 1):
            s12 = s12_min + i * h12
            s13 = s13_min + j * h13
            # 使用仿射映射辅助类做物理区域判断
            tmp = DalitzAffineMap(m_parent, m1, m2, m3)
            if tmp.in_physical_region(s12, s13, m_parent, m1, m2, m3):
                idx_map[(i, j)] = len(nodes)
                nodes.append((s12, s13))

    # --- 第 3 步: 构造三角形 ---
    elements: List[Tuple[int, int, int]] = []
    for i in range(n_s12):
        for j in range(n_s13):
            corners = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
            ids = [idx_map.get(c, -1) for c in corners]
            if all(k >= 0 for k in ids):
                i0, i1, i2, i3 = ids
                elements.append((i0, i1, i2))
                elements.append((i1, i3, i2))

    return nodes, elements


def triangle_area(
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        p2: Tuple[float, float],
) -> float:
    """三角形面积 (有向): 0.5 * |det([p1-p0, p2-p0])|。"""
    return 0.5 * abs(
        (p1[0] - p0[0]) * (p2[1] - p0[1])
        - (p2[0] - p0[0]) * (p1[1] - p0[1])
    )


def mesh_statistics(
        nodes: List[Tuple[float, float]],
        elements: List[Tuple[int, int, int]],
) -> dict:
    """返回 FEM 网格的统计信息: 节点数, 单元数, 最小/最大/平均面积。"""
    n_node = len(nodes)
    n_elem = len(elements)
    if n_elem == 0:
        return dict(n_node=n_node, n_elem=0,
                    min_area=0.0, max_area=0.0, mean_area=0.0)
    areas = []
    for (i0, i1, i2) in elements:
        a = triangle_area(nodes[i0], nodes[i1], nodes[i2])
        areas.append(a)
    return dict(
        n_node=n_node,
        n_elem=n_elem,
        min_area=min(areas),
        max_area=max(areas),
        mean_area=sum(areas) / len(areas),
    )


def boundary_normal_segments(
        boundary: List[Tuple[float, float]],
) -> List[Tuple[float, float, float, float, float, float]]:
    """
    返回边界多边形的各线段: (x0, y0, x1, y1, nx, ny)
    其中 (nx, ny) 为指向区域外侧的单位法向量。
    用于后续在有限差分格式中施加 Neumann/Robin 边界条件。
    """
    segs = []
    n = len(boundary)
    if n < 3:
        return segs
    for k in range(n):
        x0, y0 = boundary[k]
        x1, y1 = boundary[(k + 1) % n]
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        if L < 1.0e-30:
            continue
        # 外法向: 对逆时针多边形, 外法向为 (dy, -dx)/L
        nx, ny = dy / L, -dx / L
        segs.append((x0, y0, x1, y1, nx, ny))
    return segs
