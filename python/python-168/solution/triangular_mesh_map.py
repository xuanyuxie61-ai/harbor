"""
triangular_mesh_map.py
三角网格地图表示、Delaunay剖分与边界边检测

核心数学模型：
1. Delaunay 三角剖分准则：
   空外接圆条件：对于三角形 T = (p_i, p_j, p_k)，
   其外接圆内部不含其他点集 P 中的点
   
2. 外接圆判定（行列式法）：
   | x_i  y_i  x_i^2+y_i^2  1 |
   | x_j  y_j  x_j^2+y_j^2  1 |
   | x_k  y_k  x_k^2+y_k^2  1 | > 0  (逆时针，点在圆内)
   | x_t  y_t  x_t^2+y_t^2  1 |

3. 边界边检测：
   若边 (v_i, v_j) 仅属于一个三角形，则为边界边
   
4. 三角形面积（有向）：
   A = 0.5 * | (x_j-x_i)(y_k-y_i) - (x_k-x_i)(y_j-y_i) |

5. 法向量估计（融合 polygonal_surface_display 思想）：
   对于三维点云升维后的三角面片，
   n = (p_j - p_i) × (p_k - p_i) / ||...||
"""

import numpy as np


class TriangularMeshMap:
    """
    基于三角网格的稠密地图表示
    """

    def __init__(self):
        self.vertices = np.zeros((0, 2), dtype=np.float64)
        self.triangles = np.zeros((0, 3), dtype=np.int64)
        self.boundary_edges = np.zeros((0, 2), dtype=np.int64)
        self.normals = np.zeros((0, 3), dtype=np.float64)
        self.vertex_uncertainties = np.array([])

    def from_point_cloud(self, points, max_tri_area=None, max_points=800):
        """
        从点云构建三角网格地图（Delaunay剖分）
        
        Parameters
        ----------
        points : ndarray, shape (N, 2)
            二维点云
        max_tri_area : float or None
            最大三角形面积阈值（用于自适应细分）
        max_points : int
            最大点数限制（超出则均匀降采样）
        """
        points = np.asarray(points, dtype=np.float64)
        if points.shape[0] < 3:
            self.vertices = points.copy()
            return

        # 均匀降采样控制规模
        if points.shape[0] > max_points:
            step = int(np.ceil(points.shape[0] / max_points))
            points = points[::step]

        # 去重并添加微小扰动避免完全共线/共点导致的退化
        points = np.unique(np.round(points, decimals=8), axis=0)
        # 添加微小随机扰动消除退化
        noise = np.random.randn(*points.shape) * 1e-8
        points = points + noise
        self.vertices = points.copy()

        # Delaunay 三角剖分（使用 scipy.spatial.Delaunay 保证正确性）
        try:
            from scipy.spatial import Delaunay
            delaunay = Delaunay(points)
            self.triangles = delaunay.simplices.astype(np.int64)
        except Exception:
            # 回退到自定义 Bowyer-Watson（无 scipy 时）
            self.triangles = self._bowyer_watson(points)

        # 移除退化三角形（面积过小或角度接近0/180度）
        self._remove_degenerate_triangles()

        # 检测边界边
        self.boundary_edges = self._detect_boundary_edges()

        # 估计法向量（升维到3D后计算）
        self._estimate_normals()

        # 自适应细分（若面积过大）
        if max_tri_area is not None and max_tri_area > 0:
            self._adaptive_refinement(max_tri_area)

        return self

    def _bowyer_watson(self, points):
        """
        Bowyer-Watson 算法实现 Delaunay 三角剖分
        
        算法步骤：
        1. 构建一个足够大的超级三角形包含所有点
        2. 逐点插入：
           a. 找到所有外接圆包含该点的三角形（坏三角形）
           b. 构建坏三角形边界的多边形洞
           c. 连接新点到洞的每个边界边，形成新三角形
        3. 删除与超级三角形共享顶点的三角形
        """
        n = points.shape[0]
        if n < 3:
            return np.zeros((0, 3), dtype=np.int64)

        # 构建超级三角形
        minx, miny = np.min(points, axis=0)
        maxx, maxy = np.max(points, axis=0)
        dx = maxx - minx
        dy = maxy - miny
        dmax = max(dx, dy)
        if dmax < 1e-12:
            dmax = 1.0

        midx = (minx + maxx) * 0.5
        midy = (miny + maxy) * 0.5

        # 超级三角形顶点
        super_p1 = np.array([midx - 20 * dmax, midy - 10 * dmax])
        super_p2 = np.array([midx + 20 * dmax, midy - 10 * dmax])
        super_p3 = np.array([midx, midy + 20 * dmax])

        verts = np.vstack([points, super_p1, super_p2, super_p3])
        super_idx = [n, n + 1, n + 2]

        # 初始化三角形列表
        triangles = [[super_idx[0], super_idx[1], super_idx[2]]]

        for p_idx in range(n):
            px, py = verts[p_idx]
            bad_triangles = []
            polygon_edges = []

            for tri in triangles:
                i, j, k = tri
                if self._in_circumcircle(verts[i], verts[j], verts[k], verts[p_idx]):
                    bad_triangles.append(tri)

            # 找到坏三角形的边界边
            edge_count = {}
            for tri in bad_triangles:
                edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
                for e in edges:
                    e_sorted = tuple(sorted(e))
                    edge_count[e_sorted] = edge_count.get(e_sorted, 0) + 1

            # 边界边只出现一次
            boundary = [e for e, cnt in edge_count.items() if cnt == 1]

            # 删除坏三角形
            triangles = [t for t in triangles if t not in bad_triangles]

            # 添加新三角形
            for e in boundary:
                triangles.append([e[0], e[1], p_idx])

        # 删除包含超级三角形顶点的三角形
        final_triangles = []
        for tri in triangles:
            if not any(v in super_idx for v in tri):
                # 保证逆时针方向
                tri = self._orient_ccw(tri, verts)
                final_triangles.append(tri)

        return np.array(final_triangles, dtype=np.int64)

    @staticmethod
    def _in_circumcircle(a, b, c, p):
        """
        判断点 p 是否在三角形 abc 的外接圆内
        
        计算行列式：
        | ax  ay  ax^2+ay^2  1 |
        | bx  by  bx^2+by^2  1 |
        | cx  cy  cx^2+cy^2  1 |
        | px  py  px^2+py^2  1 |
        
        行列式 > 0 表示 p 在圆内（abc为逆时针）
        """
        ax, ay = a
        bx, by = b
        cx, cy = c
        px, py = p

        matrix = np.array([
            [ax, ay, ax * ax + ay * ay, 1.0],
            [bx, by, bx * bx + by * by, 1.0],
            [cx, cy, cx * cx + cy * cy, 1.0],
            [px, py, px * px + py * py, 1.0]
        ], dtype=np.float64)

        det = np.linalg.det(matrix)
        return det > 0

    @staticmethod
    def _orient_ccw(tri, verts):
        """确保三角形顶点为逆时针排列"""
        i, j, k = tri
        area = (verts[j, 0] - verts[i, 0]) * (verts[k, 1] - verts[i, 1]) \
             - (verts[k, 0] - verts[i, 0]) * (verts[j, 1] - verts[i, 1])
        if area < 0:
            return [i, k, j]
        return [i, j, k]

    def _detect_boundary_edges(self):
        """
        检测三角网格的边界边
        
        算法：每条有向边统计出现次数，出现一次的为边界边
        然后沿边界链式连接形成闭合边界环
        
        融合自 triangulation_boundary_edges 项目
        """
        if self.triangles.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.int64)

        edge_count = {}
        for tri in self.triangles:
            edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
            for e in edges:
                e_sorted = tuple(sorted(e))
                edge_count[e_sorted] = edge_count.get(e_sorted, 0) + 1

        # 边界边只出现一次
        boundary = [list(e) for e, cnt in edge_count.items() if cnt == 1]
        if not boundary:
            return np.zeros((0, 2), dtype=np.int64)

        # 链式排序边界边
        sorted_boundary = self._chain_boundary_edges(boundary)
        return np.array(sorted_boundary, dtype=np.int64)

    @staticmethod
    def _chain_boundary_edges(edges):
        """
        将边界边链式连接成有序边界环
        
        融合自 triangulation_boundary_edges 的核心算法：
        1. 找到一条可用边作为起始
        2. 搜索下一条与之相连的可用边
        3. 重复直到回到起点或没有可用边
        """
        if not edges:
            return []

        edges = [list(e) for e in edges]
        used = [False] * len(edges)
        result = []

        current = 0
        used[current] = True
        result.append(edges[current])
        n0 = edges[current][0]
        n2 = edges[current][1]

        while True:
            found = False
            for i, e in enumerate(edges):
                if not used[i]:
                    if e[0] == n2:
                        used[i] = True
                        result.append(e)
                        n2 = e[1]
                        found = True
                        break
                    elif e[1] == n2:
                        used[i] = True
                        result.append([e[1], e[0]])
                        n2 = e[0]
                        found = True
                        break
            if not found:
                break
            if n2 == n0:
                break

        return result

    def _estimate_normals(self):
        """
        估计三角面片法向量
        
        将二维三角剖分升维到三维（z=0平面），计算面法向
        n = (v1 - v0) × (v2 - v0)
        
        融合自 polygonal_surface_display 的曲面法向思想
        """
        if self.triangles.shape[0] == 0:
            self.normals = np.zeros((0, 3), dtype=np.float64)
            return

        normals = np.zeros((self.triangles.shape[0], 3), dtype=np.float64)
        for idx, tri in enumerate(self.triangles):
            p0 = np.array([self.vertices[tri[0], 0],
                           self.vertices[tri[0], 1], 0.0])
            p1 = np.array([self.vertices[tri[1], 0],
                           self.vertices[tri[1], 1], 0.0])
            p2 = np.array([self.vertices[tri[2], 0],
                           self.vertices[tri[2], 1], 0.0])

            v1 = p1 - p0
            v2 = p2 - p0
            n = np.cross(v1, v2)
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
            else:
                n = np.array([0.0, 0.0, 1.0])
            normals[idx] = n

        self.normals = normals

    def _adaptive_refinement(self, max_area):
        """
        自适应细分：将面积过大的三角形细分为4个小三角形
        
        通过在各边中点插入新顶点，每个原三角形生成4个子三角形
        """
        if self.triangles.shape[0] == 0:
            return

        new_vertices = list(self.vertices)
        new_triangles = []
        vertex_map = {}

        def get_midpoint_idx(vi, vj):
            key = tuple(sorted((vi, vj)))
            if key not in vertex_map:
                mid = (self.vertices[vi] + self.vertices[vj]) * 0.5
                vertex_map[key] = len(new_vertices)
                new_vertices.append(mid)
            return vertex_map[key]

        for tri in self.triangles:
            i, j, k = tri
            p_i = self.vertices[i]
            p_j = self.vertices[j]
            p_k = self.vertices[k]

            # 计算面积
            area = 0.5 * abs((p_j[0] - p_i[0]) * (p_k[1] - p_i[1])
                           - (p_k[0] - p_i[0]) * (p_j[1] - p_i[1]))

            if area > max_area:
                m_ij = get_midpoint_idx(i, j)
                m_jk = get_midpoint_idx(j, k)
                m_ki = get_midpoint_idx(k, i)
                new_triangles.append([i, m_ij, m_ki])
                new_triangles.append([m_ij, j, m_jk])
                new_triangles.append([m_ki, m_jk, k])
                new_triangles.append([m_ij, m_jk, m_ki])
            else:
                new_triangles.append([i, j, k])

        self.vertices = np.array(new_vertices, dtype=np.float64)
        self.triangles = np.array(new_triangles, dtype=np.int64)
        # 重新计算边界和法向
        self.boundary_edges = self._detect_boundary_edges()
        self._estimate_normals()

    def _remove_degenerate_triangles(self):
        """移除退化三角形（面积过小或角度极端）"""
        if self.triangles.shape[0] == 0:
            return
        valid = []
        for tri in self.triangles:
            p = self.vertices[tri]
            area = 0.5 * abs((p[1,0]-p[0,0])*(p[2,1]-p[0,1]) - (p[2,0]-p[0,0])*(p[1,1]-p[0,1]))
            if area < 1e-14:
                continue
            # 检查边长
            a = np.linalg.norm(p[1]-p[0])
            b = np.linalg.norm(p[2]-p[1])
            c = np.linalg.norm(p[0]-p[2])
            if a < 1e-8 or b < 1e-8 or c < 1e-8:
                continue
            # 检查角度（避免接近180度的扁平三角形）
            cos_a = np.clip((b*b + c*c - a*a)/(2*b*c), -1, 1)
            cos_b = np.clip((a*a + c*c - b*b)/(2*a*c), -1, 1)
            cos_c = np.clip((a*a + b*b - c*c)/(2*a*b), -1, 1)
            angles = [np.arccos(cos_a), np.arccos(cos_b), np.arccos(cos_c)]
            if max(angles) > np.pi - 0.01:
                continue
            valid.append(tri)
        self.triangles = np.array(valid, dtype=np.int64)

    def compute_mesh_quality(self):
        """
        计算网格质量指标
        
        Returns
        -------
        min_angle : float
            最小内角（度）
        max_aspect_ratio : float
            最大长宽比
        """
        if self.triangles.shape[0] == 0:
            return 0.0, 0.0

        min_angles = []
        aspect_ratios = []

        for tri in self.triangles:
            p = self.vertices[tri]
            # 三边长度
            a = np.linalg.norm(p[1] - p[0])
            b = np.linalg.norm(p[2] - p[1])
            c_len = np.linalg.norm(p[0] - p[2])

            if a < 1e-8 or b < 1e-8 or c_len < 1e-8:
                continue

            # 用余弦定理求角度
            cos_alpha = (b * b + c_len * c_len - a * a) / (2.0 * b * c_len)
            cos_beta = (a * a + c_len * c_len - b * b) / (2.0 * a * c_len)
            cos_gamma = (a * a + b * b - c_len * c_len) / (2.0 * a * b)
            cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
            cos_beta = np.clip(cos_beta, -1.0, 1.0)
            cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
            angles = [np.arccos(cos_alpha), np.arccos(cos_beta), np.arccos(cos_gamma)]
            min_angles.append(np.min(angles))

            # 面积
            area = 0.5 * abs((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1])
                           - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1]))
            s = (a + b + c_len) * 0.5
            if area > 1e-12 and s > 1e-12:
                r_in = area / s
                r_circ = a * b * c_len / (4.0 * area)
                ar = r_circ / (2.0 * r_in) if r_in > 1e-12 else 1.0
                aspect_ratios.append(ar)

        min_angle_deg = np.degrees(np.min(min_angles)) if min_angles else 0.0
        max_ar = np.max(aspect_ratios) if aspect_ratios else 1.0
        return min_angle_deg, max_ar

    def export_to_stl_like(self):
        """
        导出类STL格式的三角面片数据
        
        融合自 stla_to_tri_surface_fast 的三角网格导出思想
        
        Returns
        -------
        nodes : ndarray
        faces : ndarray
        normals : ndarray
        """
        return self.vertices.copy(), self.triangles.copy(), self.normals.copy()
