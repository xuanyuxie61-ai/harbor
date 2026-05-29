"""
mesh_engine.py
================================================================================
电机有限元网格生成、读写与质量分析引擎

融合原项目:
  - 261_cvt_square_uniform     : CVT（重心Voronoi镶嵌）网格生成
  - 447_freefem_msh_io         : FreeFEM 2D网格文件(.msh)读写
  - 491_grid_display           : 网格数据结构与节点/单元/边界管理
  - 1337_triangulation_histogram: 三角剖分质量统计指标
  - 441_floyd                  : 网格节点连通性最短路径分析（磁路分析）

核心科学内容:
  1. CVT 迭代算法生成电机定子槽内高质量三角形网格
  2. 电机2D截面的结构化/非结构化混合网格（定子、转子、气隙分区）
  3. FreeFEM .msh 格式兼容读写
  4. 三角形单元质量指标: 纵横比、内角、面积、外接圆半径
  5. Floyd-Warshall 最短路径用于磁路长度估计
  6. 网格边界标识: 定子外圆、气隙内圆、转子内圆、槽口
================================================================================
"""

import numpy as np
from scipy.spatial import Delaunay


class CVTMeshGenerator:
    """
    基于 Centroidal Voronoi Tessellation (CVT) 的高质量网格生成器。

    CVT 条件: 生成点 p_i 必须为其 Voronoi 区域 V_i 的质心:

        p_i = ∫_{V_i} x ρ(x) dΩ / ∫_{V_i} ρ(x) dΩ

    对于均匀密度 ρ(x) = 1, 质心即为区域的几何中心。

    Lloyd 迭代算法:
        1. 给定初始生成点集 P = {p_1, ..., p_n}
        2. 计算 Delaunay 三角剖分
        3. 对每个生成点 p_i, 计算其 Voronoi 区域的质心 c_i
        4. 更新 p_i ← c_i
        5. 重复步骤 2-4 直到收敛

    收敛判据（能量单调递减）:
        E(P) = Σ_i ∫_{V_i} ||x - p_i||^2 ρ(x) dΩ

    融合原项目 261_cvt_square_uniform 的核心算法,
    扩展至环形/扇形区域的电机几何。
    """

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    def generate_in_annular_sector(
        self,
        r_inner: float,
        r_outer: float,
        theta_min: float,
        theta_max: float,
        n_points: int,
        n_samples_per_point: int = 1000,
        max_iter: int = 50,
        tol: float = 1.0e-6,
    ) -> tuple:
        """
        在环形扇区 [r_inner, r_outer] × [theta_min, theta_max] 内生成 CVT 点集。

        返回:
            points : (n_points, 2) 笛卡尔坐标
            triangles : (n_tri, 3) 三角形单元节点索引
        """
        if r_inner < 0 or r_outer <= r_inner:
            raise ValueError("半径参数无效")
        if theta_max <= theta_min:
            raise ValueError("角度范围无效")

        # 初始点: 在环形扇区内均匀随机分布
        points = self._sample_annular_sector(
            r_inner, r_outer, theta_min, theta_max, n_points
        )

        for it in range(max_iter):
            # Delaunay 三角剖分
            tri = Delaunay(points)
            triangles = tri.simplices

            # 近似 Voronoi 质心: 对每个生成点，用周围采样点的最近邻近似
            # 实际实现: 大量采样点 + 最近邻归类 + 取平均
            sample_num = n_samples_per_point * n_points
            samples = self._sample_annular_sector(
                r_inner, r_outer, theta_min, theta_max, sample_num
            )

            # 为每个采样点找到最近的生成点
            dists = np.linalg.norm(
                samples[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2
            )
            nearest = np.argmin(dists, axis=1)

            new_points = np.zeros_like(points)
            counts = np.zeros(n_points)
            for i in range(n_points):
                mask = nearest == i
                count = np.sum(mask)
                if count > 0:
                    new_points[i] = np.mean(samples[mask], axis=0)
                    counts[i] = count
                else:
                    # 孤立点保持原位（罕见）
                    new_points[i] = points[i]
                    counts[i] = 1

            # 检查收敛
            displacement = np.max(np.linalg.norm(new_points - points, axis=1))
            points = new_points
            if displacement < tol:
                break

        # 最终 Delaunay 剖分
        tri = Delaunay(points)
        triangles = tri.simplices
        return points, triangles

    def _sample_annular_sector(
        self, r_in: float, r_out: float, th_min: float, th_max: float, n: int
    ) -> np.ndarray:
        """在环形扇区内均匀随机采样（考虑面积权重 r dr dθ）."""
        # r^2 ~ U[r_in^2, r_out^2]
        r_sq = self.rng.uniform(r_in**2, r_out**2, size=n)
        r = np.sqrt(r_sq)
        theta = self.rng.uniform(th_min, th_max, size=n)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y))


class Mesh2D:
    """
    2D 有限元网格数据结构，管理节点、三角形单元、边界边与区域标签。

    融合原项目 491_grid_display 与 1337_triangulation_histogram 的
    网格数据处理能力。

    数据组织:
        nodes      : (n_node, 2)   节点坐标 [x, y]
        elements   : (n_elem, 3)   三角形单元，节点索引从0开始
        node_tags  : (n_node,)     节点区域标签
        elem_tags  : (n_elem,)     单元区域标签
        boundaries : list[(v1, v2, tag)] 边界边列表
    """

    # 区域标签常量
    TAG_AIR_GAP = 1
    TAG_STATOR_CORE = 2
    TAG_ROTOR_CORE = 3
    TAG_MAGNET = 4
    TAG_WINDING = 5
    TAG_SHAFT = 6
    TAG_OUTER_BOUNDARY = 10
    TAG_INNER_BOUNDARY = 11

    def __init__(self):
        self.nodes = np.zeros((0, 2), dtype=float)
        self.elements = np.zeros((0, 3), dtype=int)
        self.node_tags = np.zeros(0, dtype=int)
        self.elem_tags = np.zeros(0, dtype=int)
        self.boundaries = []

    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    def n_elements(self) -> int:
        return self.elements.shape[0]

    def build_from_points_triangles(self, points: np.ndarray, triangles: np.ndarray):
        """从点集和三角剖分构建网格."""
        self.nodes = np.asarray(points, dtype=float)
        self.elements = np.asarray(triangles, dtype=int)
        n_node = self.nodes.shape[0]
        n_elem = self.elements.shape[0]
        self.node_tags = np.zeros(n_node, dtype=int)
        self.elem_tags = np.zeros(n_elem, dtype=int)
        self._validate()

    def _validate(self):
        """验证网格拓扑一致性."""
        n_node = self.n_nodes()
        n_elem = self.n_elements()
        if n_elem > 0:
            min_idx = self.elements.min()
            max_idx = self.elements.max()
            if min_idx < 0 or max_idx >= n_node:
                raise ValueError(
                    f"单元节点索引越界: [{min_idx}, {max_idx}], 节点数={n_node}"
                )
        # 检查退化单元
        for e in range(n_elem):
            v = self.elements[e]
            if len(set(v)) != 3:
                raise ValueError(f"退化单元 {e}: 节点 {v}")

    def compute_element_areas(self) -> np.ndarray:
        """计算所有三角形单元的面积."""
        n = self.n_elements()
        areas = np.zeros(n)
        for e in range(n):
            v = self.elements[e]
            p1, p2, p3 = self.nodes[v[0]], self.nodes[v[1]], self.nodes[v[2]]
            areas[e] = 0.5 * abs(
                (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
            )
        return areas

    def compute_quality_metrics(self) -> dict:
        """
        计算三角剖分质量指标（融合原项目 1337_triangulation_histogram 的统计思想）.

        指标:
          - aspect_ratio : 外接圆半径 / (2 * 内切圆半径), 理想值 = 1.0
          - min_angle    : 最小内角 [度]
          - max_angle    : 最大内角 [度]
          - area_ratio   : 最大面积 / 最小面积
          - skewness     : (θ_max - θ_eq) / (180 - θ_eq), θ_eq = 60°
        """
        n = self.n_elements()
        aspect_ratios = np.zeros(n)
        min_angles = np.zeros(n)
        max_angles = np.zeros(n)
        areas = np.zeros(n)

        for e in range(n):
            v = self.elements[e]
            p1, p2, p3 = self.nodes[v[0]], self.nodes[v[1]], self.nodes[v[2]]

            # 边长
            a = np.linalg.norm(p2 - p3)
            b = np.linalg.norm(p1 - p3)
            c = np.linalg.norm(p1 - p2)

            # 面积
            area = 0.5 * abs(
                (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
            )
            areas[e] = area

            # 内角（余弦定理）
            if a > 0 and b > 0 and c > 0:
                ang_a = np.arccos(np.clip((b * b + c * c - a * a) / (2 * b * c), -1.0, 1.0))
                ang_b = np.arccos(np.clip((a * a + c * c - b * b) / (2 * a * c), -1.0, 1.0))
                ang_c = np.pi - ang_a - ang_b
                angles_deg = np.degrees([ang_a, ang_b, ang_c])
                min_angles[e] = np.min(angles_deg)
                max_angles[e] = np.max(angles_deg)

                # 外接圆半径 R = abc / (4A)
                R = a * b * c / (4.0 * area + 1.0e-30)
                # 内切圆半径 r = A / s, s = (a+b+c)/2
                s = 0.5 * (a + b + c)
                r_in = area / (s + 1.0e-30)
                aspect_ratios[e] = R / (2.0 * r_in + 1.0e-30)
            else:
                aspect_ratios[e] = np.inf
                min_angles[e] = 0.0
                max_angles[e] = 180.0

        # 全局统计
        valid = areas > 1.0e-14
        metrics = {
            "aspect_ratio_mean": float(np.mean(aspect_ratios[valid])) if np.any(valid) else np.inf,
            "aspect_ratio_max": float(np.max(aspect_ratios[valid])) if np.any(valid) else np.inf,
            "min_angle_mean": float(np.mean(min_angles[valid])) if np.any(valid) else 0.0,
            "min_angle_min": float(np.min(min_angles[valid])) if np.any(valid) else 0.0,
            "max_angle_max": float(np.max(max_angles[valid])) if np.any(valid) else 180.0,
            "area_min": float(np.min(areas[valid])) if np.any(valid) else 0.0,
            "area_max": float(np.max(areas[valid])) if np.any(valid) else 0.0,
            "area_ratio": (
                float(np.max(areas[valid]) / (np.min(areas[valid]) + 1.0e-30))
                if np.any(valid)
                else np.inf
            ),
            "n_degenerate": int(np.sum(~valid)),
        }
        return metrics

    def tag_elements_by_region(self, region_funcs: dict):
        """
        根据区域判断函数为单元打标签.

        参数:
            region_funcs : dict {tag: func(x,y) -> bool}
        """
        n = self.n_elements()
        self.elem_tags = np.zeros(n, dtype=int)
        centroids = np.zeros((n, 2))
        for e in range(n):
            v = self.elements[e]
            centroids[e] = np.mean(self.nodes[v], axis=0)

        for tag, func in region_funcs.items():
            mask = np.array([func(c[0], c[1]) for c in centroids])
            self.elem_tags[mask] = tag

    def build_connectivity_matrix(self) -> np.ndarray:
        """
        构建节点连通性矩阵（邻接矩阵），用于 Floyd-Warshall 最短路径分析。

        返回:
            adj : (n_node, n_node) 邻接矩阵，adj[i,j] = 节点i到j的欧氏距离
        """
        n = self.n_nodes()
        adj = np.full((n, n), np.inf)
        np.fill_diagonal(adj, 0.0)

        for e in range(self.n_elements()):
            v = self.elements[e]
            for i in range(3):
                for j in range(i + 1, 3):
                    vi, vj = v[i], v[j]
                    dist = np.linalg.norm(self.nodes[vi] - self.nodes[vj])
                    adj[vi, vj] = min(adj[vi, vj], dist)
                    adj[vj, vi] = min(adj[vj, vi], dist)
        return adj

    def floyd_warshall_magnetic_path(self, source_idx: int) -> np.ndarray:
        """
        使用 Floyd-Warshall 算法计算从源节点到所有其他节点的最短磁路长度。

        融合原项目 441_floyd 的核心算法。

        磁路长度估计假设磁通沿最短几何路径传播，忽略材料磁阻差异。
        更精确模型应将距离加权为磁阻: d_mag = μ_0 / μ * d_geom.

        算法:
            初始化: D^{(0)}[i,j] = dist(i,j) 或 ∞
            迭代:   D^{(k)}[i,j] = min( D^{(k-1)}[i,j], D^{(k-1)}[i,k] + D^{(k-1)}[k,j] )
            结果:   D^{(n)} 为全对最短路径
        """
        adj = self.build_connectivity_matrix()
        n = adj.shape[0]
        D = adj.copy()

        for k in range(n):
            for j in range(n):
                # 向量化内层循环以提高效率
                d_ik = D[:, k]
                d_kj = D[k, j]
                d_ij = D[:, j]
                D[:, j] = np.minimum(d_ij, d_ik + d_kj)

        return D[source_idx, :]

    def extract_boundary_edges(self) -> list:
        """
        提取网格边界边（只属于一个三角形的边）.

        返回:
            boundaries : [(v1, v2, tag), ...]
        """
        edge_count = {}
        for e in range(self.n_elements()):
            v = self.elements[e]
            edges = [(min(v[i], v[j]), max(v[i], v[j])) for i, j in [(0, 1), (1, 2), (2, 0)]]
            for ed in edges:
                edge_count[ed] = edge_count.get(ed, 0) + 1

        self.boundaries = []
        for ed, count in edge_count.items():
            if count == 1:
                self.boundaries.append((ed[0], ed[1], self.TAG_OUTER_BOUNDARY))
        return self.boundaries

    def write_msh(self, filename: str):
        """
        写入 FreeFEM .msh 格式文件（融合原项目 447_freefem_msh_io 的 write 功能）.

        格式:
            第1行: v_num  t_num  e_num
            顶点行: x  y  tag
            三角形行: v1  v2  v3  tag
            边界边行: v1  v2  tag
        """
        self.extract_boundary_edges()
        v_num = self.n_nodes()
        t_num = self.n_elements()
        e_num = len(self.boundaries)

        with open(filename, "w") as f:
            f.write(f"{v_num}  {t_num}  {e_num}\n")
            for i in range(v_num):
                tag = int(self.node_tags[i]) if i < len(self.node_tags) else 0
                f.write(f"{self.nodes[i, 0]:.15g}  {self.nodes[i, 1]:.15g}  {tag}\n")
            for e in range(t_num):
                tag = int(self.elem_tags[e]) if e < len(self.elem_tags) else 0
                v = self.elements[e]
                f.write(f"{v[0]+1}  {v[1]+1}  {v[2]+1}  {tag}\n")
            for ed in self.boundaries:
                f.write(f"{ed[0]+1}  {ed[1]+1}  {ed[2]}\n")

    @classmethod
    def read_msh(cls, filename: str):
        """读取 FreeFEM .msh 格式文件（融合原项目 447_freefem_msh_io 的 read 功能）."""
        mesh = cls()
        with open(filename, "r") as f:
            parts = f.readline().strip().split()
            if len(parts) < 3:
                raise ValueError("MSH文件头格式错误")
            v_num, t_num, e_num = map(int, parts)

            nodes = []
            node_tags = []
            for _ in range(v_num):
                line = f.readline().strip()
                while line == "":
                    line = f.readline().strip()
                vals = line.split()
                nodes.append([float(vals[0]), float(vals[1])])
                node_tags.append(int(vals[2]) if len(vals) > 2 else 0)

            elements = []
            elem_tags = []
            for _ in range(t_num):
                line = f.readline().strip()
                while line == "":
                    line = f.readline().strip()
                vals = line.split()
                elements.append([int(vals[0]) - 1, int(vals[1]) - 1, int(vals[2]) - 1])
                elem_tags.append(int(vals[3]) if len(vals) > 3 else 0)

            boundaries = []
            for _ in range(e_num):
                line = f.readline().strip()
                while line == "":
                    line = f.readline().strip()
                vals = line.split()
                boundaries.append((int(vals[0]) - 1, int(vals[1]) - 1, int(vals[2])))

        mesh.nodes = np.array(nodes, dtype=float)
        mesh.elements = np.array(elements, dtype=int)
        mesh.node_tags = np.array(node_tags, dtype=int)
        mesh.elem_tags = np.array(elem_tags, dtype=int)
        mesh.boundaries = boundaries
        mesh._validate()
        return mesh


def build_simple_pmsm_mesh(
    R_so: float = 0.1,
    R_si: float = 0.08,
    R_ro: float = 0.075,
    R_ri: float = 0.05,
    n_stator: int = 200,
    n_rotor: int = 150,
    n_airgap: int = 80,
    n_slots: int = 6,
    seed: int = 42,
) -> Mesh2D:
    """
    构建简化的永磁同步电机（PMSM）2D截面网格。

    几何分区:
      - 定子铁芯 : R_si < r < R_so
      - 气隙     : R_ro < r < R_si
      - 转子铁芯 : R_ri < r < R_ro
      - 永磁体   : 附着在转子表面
      - 轴       : r < R_ri
    """
    cvt = CVTMeshGenerator(seed=seed)
    all_points = []
    all_triangles = []
    point_offset = 0

    # 定子铁芯（扇区分割，避免槽的细节）
    dtheta = 2.0 * np.pi / n_slots
    for k in range(n_slots):
        th1 = k * dtheta + 0.02
        th2 = (k + 1) * dtheta - 0.02
        pts, tri = cvt.generate_in_annular_sector(
            R_si, R_so, th1, th2, n_stator // n_slots, max_iter=20
        )
        all_points.append(pts)
        all_triangles.append(tri + point_offset)
        point_offset += pts.shape[0]

    # 转子铁芯
    pts, tri = cvt.generate_in_annular_sector(
        R_ri, R_ro, 0.0, 2.0 * np.pi, n_rotor, max_iter=20
    )
    all_points.append(pts)
    all_triangles.append(tri + point_offset)
    point_offset += pts.shape[0]

    # 气隙（环形）
    pts, tri = cvt.generate_in_annular_sector(
        R_ro, R_si, 0.0, 2.0 * np.pi, n_airgap, max_iter=20
    )
    all_points.append(pts)
    all_triangles.append(tri + point_offset)
    point_offset += pts.shape[0]

    # 合并
    nodes = np.vstack(all_points)
    elements = np.vstack(all_triangles)

    mesh = Mesh2D()
    mesh.build_from_points_triangles(nodes, elements)

    # 区域标签
    def in_stator(x, y):
        r = np.sqrt(x * x + y * y)
        return R_si < r < R_so

    def in_rotor(x, y):
        r = np.sqrt(x * x + y * y)
        return R_ri < r < R_ro

    def in_airgap(x, y):
        r = np.sqrt(x * x + y * y)
        return R_ro < r < R_si

    def in_shaft(x, y):
        r = np.sqrt(x * x + y * y)
        return r < R_ri

    mesh.tag_elements_by_region({
        mesh.TAG_STATOR_CORE: in_stator,
        mesh.TAG_ROTOR_CORE: in_rotor,
        mesh.TAG_AIR_GAP: in_airgap,
        mesh.TAG_SHAFT: in_shaft,
    })

    return mesh
