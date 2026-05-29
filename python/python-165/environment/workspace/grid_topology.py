"""
grid_topology.py
电网拓扑几何建模与Delaunay三角剖分
融合种子项目：triangulation, circle_arc_grid, mesh2d_write, xyf_display
"""

import numpy as np
from typing import List, Tuple, Optional
from utils import circle_arc_grid, triangle_area_2d, triangle_angles


class GridTopology:
    """
    智能电网拓扑模型。

    基于二维平面 Delaunay 三角剖分构建电网几何拓扑，
    节点沿圆弧（ring feeder）与径向线分布，形成辐射-环网混合结构。
    """

    def __init__(self, nodes: np.ndarray, elements: Optional[np.ndarray] = None):
        self.nodes = np.array(nodes, dtype=np.float64)
        self.n_nodes = self.nodes.shape[0]
        if elements is None:
            self.elements = self.delaunay_triangulate()
        else:
            self.elements = np.array(elements, dtype=np.int32)
        self.n_elements = self.elements.shape[0]
        self.adjacency = self._build_adjacency()

    def delaunay_triangulate(self) -> np.ndarray:
        """
        增量式 Delaunay 三角剖分（基于种子项目 r8tris2 思想）。

        空外接圆准则：对于三角剖分 T 中的任意三角形 Δ，其外接圆内部
        不包含其他节点。数学表述为：对任意节点 p ∉ Δ，
            det( [x_p, y_p, x_p^2+y_p^2, 1]
                 [x_i, y_i, x_i^2+y_i^2, 1]
                 [x_j, y_j, x_j^2+y_j^2, 1]
                 [x_k, y_k, x_k^2+y_k^2, 1] ) > 0
        其中 (i,j,k) 为三角形顶点按逆时针排列。

        该剖分使三角形最小角最大化，从而在电网中保证相邻母线间
        连线夹角不会过小，降低线路交叉与短路风险。
        """
        pts = self.nodes
        n = pts.shape[0]
        if n < 3:
            return np.zeros((0, 3), dtype=np.int32)

        # 计算包围盒并添加超级三角形
        minx, miny = pts.min(axis=0)
        maxx, maxy = pts.max(axis=0)
        dx, dy = maxx - minx, maxy - miny
        dmax = max(dx, dy)
        midx, midy = (minx + maxx) * 0.5, (miny + maxy) * 0.5

        super_tri = np.array([
            [midx - 20 * dmax, midy - dmax],
            [midx + 20 * dmax, midy - dmax],
            [midx, midy + 20 * dmax]
        ], dtype=np.float64)

        triangles = [(0, 1, 2)]
        all_pts = np.vstack([pts, super_tri])

        def in_circumcircle(p, a, b, c):
            ax, ay = a[0] - p[0], a[1] - p[1]
            bx, by = b[0] - p[0], b[1] - p[1]
            cx, cy = c[0] - p[0], c[1] - p[1]
            det = (ax * ax + ay * ay) * (bx * cy - by * cx) \
                - (bx * bx + by * by) * (ax * cy - ay * cx) \
                + (cx * cx + cy * cy) * (ax * by - ay * bx)
            return det > 0

        for p_idx in range(n):
            p = all_pts[p_idx]
            bad_triangles = []
            for t in triangles:
                a, b, c = all_pts[t[0]], all_pts[t[1]], all_pts[t[2]]
                if in_circumcircle(p, a, b, c):
                    bad_triangles.append(t)

            polygon = []
            for t in bad_triangles:
                edges = [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]
                for e in edges:
                    shared = False
                    for ot in bad_triangles:
                        if t == ot:
                            continue
                        o_edges = [(ot[0], ot[1]), (ot[1], ot[2]), (ot[2], ot[0])]
                        if any((e[0] == oe[1] and e[1] == oe[0]) for oe in o_edges):
                            shared = True
                            break
                    if not shared:
                        polygon.append(e)

            # 安全移除 bad triangles
            bad_set = set(bad_triangles)
            triangles = [t for t in triangles if t not in bad_set]
            for e in polygon:
                triangles.append((e[0], e[1], p_idx))

        # 移除包含超级三角形顶点的三角形
        filtered = []
        super_idx = {n, n + 1, n + 2}
        for t in triangles:
            if not any(v in super_idx for v in t):
                filtered.append(t)
        if not filtered:
            return np.zeros((0, 3), dtype=np.int32)
        return np.array(filtered, dtype=np.int32)

    def _build_adjacency(self) -> List[List[int]]:
        """
        构建节点邻接表（融合 mesh2d_write 的连接性思想）。
        """
        adj = [[] for _ in range(self.n_nodes)]
        for tri in self.elements:
            for i in range(3):
                for j in range(i + 1, 3):
                    u, v = int(tri[i]), int(tri[j])
                    if v not in adj[u]:
                        adj[u].append(v)
                    if u not in adj[v]:
                        adj[v].append(u)
        return adj

    def get_edge_list(self) -> np.ndarray:
        """
        提取唯一的边列表，用于导纳矩阵构建。
        """
        edges = set()
        for tri in self.elements:
            for i in range(3):
                u, v = int(tri[i]), int(tri[(i + 1) % 3])
                if u == v:
                    continue
                edges.add((min(u, v), max(u, v)))
        return np.array(sorted(edges), dtype=np.int32)

    def compute_mesh_quality(self) -> dict:
        """
        计算网格质量指标（融合 triangle_analyze 思想）。

        1. 最小角：q_min = min(α,β,γ)
        2. 面积比：q_ar = 4√3·A / (a^2+b^2+c^2)
        3. 平均面积与面积方差

        在电网中，这些指标映射为：
        - 最小角大 → 线路夹角合理，电磁干扰小
        - 面积均匀 → 供电区域划分均衡，负荷分配均匀
        """
        if self.n_elements == 0:
            return {"min_angle_deg": 0.0, "area_ratio": 0.0,
                    "mean_area": 0.0, "std_area": 0.0}
        min_angles = []
        area_ratios = []
        areas = []
        for tri in self.elements:
            a = self.nodes[tri[0]]
            b = self.nodes[tri[1]]
            c = self.nodes[tri[2]]
            area = triangle_area_2d(a, b, c)
            areas.append(area)
            angles = triangle_angles(a, b, c)
            min_angles.append(np.degrees(angles.min()))
            lab = np.linalg.norm(b - a)
            lbc = np.linalg.norm(c - b)
            lca = np.linalg.norm(a - c)
            denom = lab**2 + lbc**2 + lca**2
            if denom > 1e-12:
                area_ratios.append(4.0 * np.sqrt(3.0) * area / denom)
            else:
                area_ratios.append(0.0)
        return {
            "min_angle_deg": float(np.min(min_angles)),
            "area_ratio": float(np.mean(area_ratios)),
            "mean_area": float(np.mean(areas)),
            "std_area": float(np.std(areas))
        }

    @staticmethod
    def generate_ring_radial_topology(n_ring: int, n_radial: int,
                                       r_inner: float, r_outer: float,
                                       rng_seed: int = 42) -> "GridTopology":
        """
        生成环形-径向混合配电网拓扑。

        内环半径 r_inner 布置 n_ring 个节点，外环 r_outer 布置 n_ring 个节点，
        并沿径向添加 n_radial 条射线，每条射线上均匀分布中间节点。
        为避免 Delaunay 三角剖分退化（共圆点集），引入微小随机扰动。

        此拓扑模拟实际城市配电网的"双环网+辐射支路"结构，
        兼顾供电可靠性与经济性。
        """
        rng = np.random.default_rng(rng_seed)
        nodes = []
        # 内环
        inner = circle_arc_grid(0.0, 0.0, r_inner, 0.0, 360.0, n_ring)
        inner = inner[:-1]
        nodes.extend(inner.tolist())
        # 外环
        outer = circle_arc_grid(0.0, 0.0, r_outer, 0.0, 360.0, n_ring)
        outer = outer[:-1]
        nodes.extend(outer.tolist())
        # 径向射线节点
        for i in range(n_ring):
            angle = 2.0 * np.pi * i / n_ring
            for j in range(1, n_radial + 1):
                t = j / (n_radial + 1)
                r = r_inner + t * (r_outer - r_inner)
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                nodes.append([x, y])
        pts = np.array(nodes, dtype=np.float64)
        # 添加微小随机扰动，破坏共圆性，保证 Delaunay 非退化
        pts += rng.normal(0.0, 0.05 * r_inner, pts.shape)
        return GridTopology(pts)
