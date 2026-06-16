"""
detector_mesh.py
================

来源:
  - 789_navier_stokes_mesh2d :  2D 非结构网格解析 (节点, 单元, 边界)
  - 1353_triangulation_t3_to_t4 :  T3 → T4 网格细化 (插入重心节点)

物理重构: 探测器几何网格
--------------------------------
高能物理探测器的读出单元通常按 (η, φ) 网格组织:
    η : 赝快度 (纵向位置)
    φ : 方位角 (横向位置)

典型分段:
    像素探测器: Δη × Δφ ≈ 0.01 × 0.01 (中央)
    条式探测器: Δη × Δφ ≈ 0.03 × 0.015
    量能器 (塔): Δη × Δφ ≈ 0.087 × 0.087 (CMS ECAL)

本模块:
1. 生成/解析 (η, φ) 探测器网格 (类似 mesh2d_extract)
2. 将线性 T3 三角形单元升级为 T4 (带重心) 单元,
   使得响应矩阵在 bin 内支持二次基函数 (bubble function)
3. 计算每个 bin 的几何中心与面积 (用于响应矩阵加权)

T3 → T4 升级的数学:
    给定三角形 T = (v1, v2, v3),
    重心 c = (v1 + v2 + v3) / 3.
    T4 = (v1, v2, v3, c).
    基函数:
        φ_1 = λ_1 (1 - 9 λ_2 λ_3)    (bubble 修正)
        ...
    使得在重心处有额外的自由度, 提高响应矩阵在 bin 内的分辨率.
"""

from __future__ import annotations
from typing import List, Tuple, Dict
import math


# ===========================================================================
#            2D 探测器网格生成与解析
# ===========================================================================

class DetectorMesh:
    """
    2D 探测器 (η, φ) 网格.

    属性:
        n_eta, n_phi    : 分段数
        eta_edges       : η 边界数组, 长度 n_eta + 1
        phi_edges       : φ 边界数组, 长度 n_phi + 1
        node_coords     : (N_nodes × 2) 节点坐标
        elements        : (N_elem × 4) 四边形单元 → 节点索引
        element_areas   : 每个单元的面积
        element_centers : 每个单元的几何中心 (eta_c, phi_c)
    """

    def __init__(
        self,
        eta_range: Tuple[float, float] = (-2.5, 2.5),
        phi_range: Tuple[float, float] = (-math.pi, math.pi),
        n_eta: int = 10,
        n_phi: int = 12,
    ):
        if n_eta < 1 or n_phi < 1:
            raise ValueError("n_eta, n_phi 必须 ≥ 1")
        if eta_range[0] >= eta_range[1] or phi_range[0] >= phi_range[1]:
            raise ValueError("range 必须左 < 右")

        self.n_eta = n_eta
        self.n_phi = n_phi
        self.eta_range = eta_range
        self.phi_range = phi_range

        # 边界
        self.eta_edges = [
            eta_range[0] + (eta_range[1] - eta_range[0]) * i / n_eta
            for i in range(n_eta + 1)
        ]
        self.phi_edges = [
            phi_range[0] + (phi_range[1] - phi_range[0]) * i / n_phi
            for i in range(n_phi + 1)
        ]

        # 节点
        self.node_coords: List[Tuple[float, float]] = []
        for i in range(n_eta + 1):
            for j in range(n_phi + 1):
                self.node_coords.append((self.eta_edges[i], self.phi_edges[j]))

        # 四边形单元 (每单元 4 节点)
        self.elements: List[Tuple[int, int, int, int]] = []
        for i in range(n_eta):
            for j in range(n_phi):
                n0 = i * (n_phi + 1) + j
                n1 = n0 + 1
                n2 = (i + 1) * (n_phi + 1) + j + 1
                n3 = (i + 1) * (n_phi + 1) + j
                self.elements.append((n0, n1, n2, n3))

        # 面积与中心
        self.element_areas: List[float] = []
        self.element_centers: List[Tuple[float, float]] = []
        for (n0, n1, n2, n3) in self.elements:
            c0 = self.node_coords[n0]
            c1 = self.node_coords[n1]
            c2 = self.node_coords[n2]
            c3 = self.node_coords[n3]
            # 中心 = 平均
            center = (
                0.25 * (c0[0] + c1[0] + c2[0] + c3[0]),
                0.25 * (c0[1] + c1[1] + c2[1] + c3[1]),
            )
            # 面积 (四边形, 对角线叉积 / 2)
            area = 0.5 * abs(
                (c2[0] - c0[0]) * (c3[1] - c1[1]) - (c3[0] - c1[0]) * (c2[1] - c0[1])
            )
            self.element_areas.append(area)
            self.element_centers.append(center)

    def n_elements(self) -> int:
        return len(self.elements)

    def n_nodes(self) -> int:
        return len(self.node_coords)

    def bin_index(self, eta: float, phi: float) -> int:
        """给定 (η, φ), 返回所在的单元索引 (-1 表示越界)."""
        if eta < self.eta_range[0] or eta > self.eta_range[1]:
            return -1
        if phi < self.phi_range[0] or phi > self.phi_range[1]:
            return -1
        i = int((eta - self.eta_range[0]) / (self.eta_range[1] - self.eta_range[0]) * self.n_eta)
        j = int((phi - self.phi_range[0]) / (self.phi_range[1] - self.phi_range[0]) * self.n_phi)
        i = min(i, self.n_eta - 1)
        j = min(j, self.n_phi - 1)
        return i * self.n_phi + j


# ===========================================================================
#          T3 → T4 升级 (三角形 → 带重心三角形)
# ===========================================================================

def triangulation_t3_to_t4(
    node_coords: List[Tuple[float, float]],
    triangles: List[Tuple[int, int, int]],
) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int, int, int]]]:
    """
    将线性三角形网格 (T3) 升级为 4 节点三角形网格 (T4).

    算法 (1353_triangulation_t3_to_t4 移植):
        1. 对每个三角形 T_k = (v_a, v_b, v_c):
             计算重心 c_k = (v_a + v_b + v_c) / 3
        2. 将 c_k 作为新节点追加到节点列表
        3. 新单元 T4_k = (v_a, v_b, v_c, c_k)

    返回 (new_node_coords, new_elements).

    物理意义:
        每个 bin 的重心代表响应矩阵在该 bin 内的 "二次采样点",
        允许在 bin 内用二次多项式重建能谱 (而非分段常数).
    """
    new_nodes = list(node_coords)
    new_elems = []
    for (a, b, c) in triangles:
        va = node_coords[a]
        vb = node_coords[b]
        vc = node_coords[c]
        centroid = (
            (va[0] + vb[0] + vc[0]) / 3.0,
            (va[1] + vb[1] + vc[1]) / 3.0,
        )
        new_idx = len(new_nodes)
        new_nodes.append(centroid)
        new_elems.append((a, b, c, new_idx))
    return new_nodes, new_elems


def quadrilateral_to_triangles(
    mesh: DetectorMesh,
) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int, int]]]:
    """
    将四边形网格剖分为三角形 (每个四边形 → 2 个三角形).

    返回 (node_coords, triangles).
    """
    triangles = []
    for (n0, n1, n2, n3) in mesh.elements:
        triangles.append((n0, n1, n2))
        triangles.append((n0, n2, n3))
    return mesh.node_coords, triangles


def mesh_enrichment_for_response(
    mesh: DetectorMesh,
) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int, int, int]]]:
    """
    为响应矩阵计算准备富化网格:
    1. 将四边形 → 三角形
    2. T3 → T4 (插入重心)

    返回 (enriched_nodes, t4_elements).
    """
    nodes, triangles = quadrilateral_to_triangles(mesh)
    return triangulation_t3_to_t4(nodes, triangles)


# ===========================================================================
#            边界条件处理 (周期性 φ, 截断 η)
# ===========================================================================

def apply_boundary_conditions(
    mesh: DetectorMesh,
    values: List[float],
    phi_periodic: bool = True,
) -> List[float]:
    """
    对网格上的值施加边界条件.

    φ 方向: 周期性 (φ = -π 与 φ = π 等价)
        ⇒ 第一列与最后一列的值应相等 (平均化)
    η 方向: 截断 (边界外无接受度)
        ⇒ 边界单元的值保持不变 (但不外推)

    返回施加边界条件后的值数组.
    """
    if len(values) != mesh.n_elements():
        raise ValueError("values 长度与单元数不匹配")
    out = list(values)
    if phi_periodic and mesh.n_phi > 1:
        # 对每个 η 行, 平均第一列与最后一列
        for i in range(mesh.n_eta):
            idx_first = i * mesh.n_phi + 0
            idx_last = i * mesh.n_phi + (mesh.n_phi - 1)
            avg = 0.5 * (out[idx_first] + out[idx_last])
            out[idx_first] = avg
            out[idx_last] = avg
    return out


# ===========================================================================
#            网格质量度量 (用于响应矩阵数值稳定性分析)
# ===========================================================================

def mesh_quality_aspect_ratio(mesh: DetectorMesh) -> List[float]:
    """
    计算每个四边形单元的纵横比:
        AR = max(Δη, Δφ) / min(Δη, Δφ)

    纵横比接近 1 表示单元接近正方形, 响应矩阵的条件数较小.
    """
    ratios = []
    for i in range(mesh.n_eta):
        deta = mesh.eta_edges[i + 1] - mesh.eta_edges[i]
        for j in range(mesh.n_phi):
            dphi = mesh.phi_edges[j + 1] - mesh.phi_edges[j]
            ar = max(deta, dphi) / (min(deta, dphi) + 1e-300)
            ratios.append(ar)
    return ratios


__all__ = [
    "DetectorMesh",
    "triangulation_t3_to_t4",
    "quadrilateral_to_triangles",
    "mesh_enrichment_for_response",
    "apply_boundary_conditions",
    "mesh_quality_aspect_ratio",
]
