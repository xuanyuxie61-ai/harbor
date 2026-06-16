"""
畴结构网格拓扑模块
===================
对应种子项目: 755_mesh_etoe (单元邻接表 → 铁电畴邻接图)

物理背景:
    多铁性材料中, 极化矢量在不同空间区域取不同方向, 形成畴结构.
    畴壁是相邻畴之间的过渡区域, 其能量和动力学行为决定材料的宏观性质.

    本模块将 2D 计算域分割为三角网格, 构建单元邻接表 (ETOE),
    用于:
    1. 识别畴壁位置 (相邻单元极化方向差异大)
    2. 计算畴壁能量密度
    3. 分析畴结构的拓扑特征 (畴壁交叉、涡旋等)

ETOE 算法 (对应种子项目 755):
    输入: ETOV 表 (n_elements × 3), 每个单元的三个顶点编号
    输出: ETOE 表 (n_elements × 3), 每条边的邻接单元编号

    步骤:
    1. 提取所有边: 每个三角形有 3 条边
    2. 排序边记录: (min(v1,v2), max(v1,v2), side, element)
    3. 扫描排序后的列表, 找到重复边 → 确定邻接关系
    4. 唯一边 → 边界

拓扑不变量:
    Euler 特征数: χ = V - E + F (顶点数 - 边数 + 面数)
    对于 2D 三角网格: χ = 2 - 2g (g 为亏格)
"""

import numpy as np


# ============================================================
# 三角网格生成与 ETOE 构建
# ============================================================

class DomainMeshTopology:
    """
    铁电/铁磁畴结构的网格拓扑分析器.

    将 2D 规则网格转换为三角剖分, 并构建邻接关系.
    用于识别畴壁、计算拓扑荷、分析畴结构演化.
    """

    def __init__(self, nx, ny, Lx, Ly):
        """
        参数:
            nx, ny: 规则网格点数
            Lx, Ly: 物理域尺寸
        """
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)

        # 生成三角剖分
        self.etov = self._build_triangulation()
        self.n_elements = self.etov.shape[0]

        # 构建邻接表
        self.etoe = self._build_etoe()

        # 边界列表
        self.boundary_edges = self._find_boundary()

    def _build_triangulation(self):
        """
        将规则 (nx × ny) 网格三角剖分.

        每个矩形单元 (i,j)-(i+1,j)-(i+1,j+1)-(i,j+1)
        分成 2 个三角形:
            T1: (i,j), (i+1,j), (i+1,j+1)
            T2: (i,j), (i+1,j+1), (i,j+1)

        顶点编号约定: vertex_id = j * nx + i

        返回:
            etov: shape (n_triangles, 3), 每个三角形的顶点编号
        """
        triangles = []
        for j in range(self.ny - 1):
            for i in range(self.nx - 1):
                v00 = j * self.nx + i
                v10 = j * self.nx + (i + 1)
                v01 = (j + 1) * self.nx + i
                v11 = (j + 1) * self.nx + (i + 1)

                # 三角形 1: 下三角
                triangles.append([v00, v10, v11])
                # 三角形 2: 上三角
                triangles.append([v00, v11, v01])

        return np.array(triangles, dtype=int)

    def _build_etoe(self):
        """
        构建单元-单元邻接表 (对应种子项目 755_mesh_etoe).

        算法:
            1. 对每个三角形, 提取 3 条边
            2. 每条边表示为 (min(v1,v2), max(v1,v2), side, element)
            3. 按 (v_min, v_max) 排序
            4. 扫描排序列表, 找重复边

        返回:
            etoe: shape (n_triangles, 3), 邻接单元编号 (-1 表示边界)
        """
        n_tri = self.n_elements
        e_order = 3  # 每个三角形 3 条边

        # 构建边记录
        edge_records = []
        for elem_id in range(n_tri):
            verts = self.etov[elem_id]
            for side in range(e_order):
                v1 = verts[side]
                v2 = verts[(side + 1) % e_order]
                v_min = min(v1, v2)
                v_max = max(v1, v2)
                edge_records.append((v_min, v_max, side, elem_id))

        # 按 (v_min, v_max) 排序
        edge_records.sort(key=lambda x: (x[0], x[1]))

        # 初始化 ETOE 为 -1 (边界)
        etoe = -np.ones((n_tri, e_order), dtype=int)

        # 扫描找重复边
        i = 0
        n_edges = len(edge_records)
        while i < n_edges - 1:
            rec1 = edge_records[i]
            rec2 = edge_records[i + 1]

            if rec1[0] == rec2[0] and rec1[1] == rec2[1]:
                # 找到邻接对
                side1, elem1 = rec1[2], rec1[3]
                side2, elem2 = rec2[2], rec2[3]
                etoe[elem1, side1] = elem2
                etoe[elem2, side2] = elem1
                i += 2
            else:
                i += 1

        return etoe

    def _find_boundary(self):
        """
        找到所有边界边.

        返回:
            boundary: list of (element_id, side_id) 元组
        """
        boundary = []
        n_tri, e_order = self.etoe.shape
        for elem_id in range(n_tri):
            for side in range(e_order):
                if self.etoe[elem_id, side] == -1:
                    boundary.append((elem_id, side))
        return boundary

    # ============================================================
    # 畴壁识别与分析
    # ============================================================

    def identify_domain_walls(self, P_field, threshold=0.3):
        """
        识别畴壁位置.

        畴壁定义: 相邻单元的极化方向差异超过阈值.

        判据:
            若 cos(θ) = P⃗ᵢ·P⃗ⱼ/(|P⃗ᵢ|·|P⃗ⱼ|) < threshold
            则单元 i 和 j 之间存在畴壁

        参数:
            P_field: 极化场, shape (nx, ny, 3)
            threshold: 畴壁判定阈值 (0-1)

        返回:
            wall_mask: bool 数组, shape (n_elements,)
                True 表示该单元位于畴壁附近
        """
        wall_mask = np.zeros(self.n_elements, dtype=bool)

        # 计算每个单元的局部极化 (从网格场插值)
        elem_P = self._interpolate_field_to_elements(P_field)

        # 归一化
        norms = np.linalg.norm(elem_P, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-30)
        elem_P_hat = elem_P / norms

        # 检查邻接关系
        for elem_id in range(self.n_elements):
            for side in range(3):
                neighbor = self.etoe[elem_id, side]
                if neighbor == -1:
                    continue
                cos_theta = np.dot(elem_P_hat[elem_id], elem_P_hat[neighbor])
                if cos_theta < threshold:
                    wall_mask[elem_id] = True
                    wall_mask[neighbor] = True
                    break

        return wall_mask

    def _interpolate_field_to_elements(self, field):
        """
        将网格场插值到三角形单元中心.

        参数:
            field: shape (nx, ny, n_comp) 或 (nx, ny)

        返回:
            elem_field: shape (n_elements, n_comp)
        """
        if field.ndim == 2:
            field_3d = field[:, :, np.newaxis]
        else:
            field_3d = field

        n_comp = field_3d.shape[2]
        elem_field = np.zeros((self.n_elements, n_comp))

        for elem_id in range(self.n_elements):
            verts = self.etov[elem_id]
            for vi in verts:
                j = vi // self.nx
                i = vi % self.nx
                # 边界截断
                i = min(i, self.nx - 1)
                j = min(j, self.ny - 1)
                elem_field[elem_id] += field_3d[i, j]
            elem_field[elem_id] /= 3.0

        return elem_field

    # ============================================================
    # 拓扑荷计算
    # ============================================================

    def compute_topological_charge(self, P_field):
        """
        计算每个单元的局部拓扑荷密度.

        拓扑荷密度 (Pontryagin 密度):
            q = (1/4π) · P⃗ · (∂P⃗/∂x × ∂P⃗/∂y)

        对于归一化极化场 n⃗ = P⃗/|P⃗|:
            Q = ∫∫ q dx dy ∈ ℤ

        总拓扑荷 Q 表征畴结构的拓扑类型:
            Q = 0: 平凡 (单畴或反平行畴)
            Q = ±1: 涡旋/反涡旋
            Q = ±2: 双涡旋

        参数:
            P_field: 极化场, shape (nx, ny, 3)

        返回:
            Q_total: 总拓扑荷 (标量)
            q_density: 拓扑荷密度, shape (nx, ny)
        """
        # 归一化
        norms = np.linalg.norm(P_field, axis=2, keepdims=True)
        norms = np.maximum(norms, 1e-30)
        n_field = P_field / norms

        # 计算空间导数 (中心差分)
        dn_dx = np.zeros_like(n_field)
        dn_dy = np.zeros_like(n_field)

        dn_dx[1:-1] = (n_field[2:] - n_field[:-2]) / (2.0 * self.dx)
        dn_dx[0] = (n_field[1] - n_field[0]) / self.dx
        dn_dx[-1] = (n_field[-1] - n_field[-2]) / self.dx

        dn_dy[:, 1:-1] = (n_field[:, 2:] - n_field[:, :-2]) / (2.0 * self.dy)
        dn_dy[:, 0] = (n_field[:, 1] - n_field[:, 0]) / self.dy
        dn_dy[:, -1] = (n_field[:, -1] - n_field[:, -2]) / self.dy

        # Pontryagin 密度: q = n⃗·(∂n⃗/∂x × ∂n⃗/∂y) / (4π)
        cross_product = np.cross(dn_dx, dn_dy)
        q_density = np.sum(n_field * cross_product, axis=2) / (4.0 * np.pi)

        Q_total = np.sum(q_density) * self.dx * self.dy

        return Q_total, q_density

    # ============================================================
    # 统计量
    # ============================================================

    def mesh_statistics(self):
        """
        输出网格拓扑统计信息.

        包括: Euler 特征数, 边界边数, 平均邻接数等.
        """
        n_vertices = self.nx * self.ny
        n_edges_total = 0
        counted = set()
        for elem_id in range(self.n_elements):
            verts = self.etov[elem_id]
            for side in range(3):
                v1 = verts[side]
                v2 = verts[(side + 1) % 3]
                edge = (min(v1, v2), max(v1, v2))
                if edge not in counted:
                    counted.add(edge)
                    n_edges_total += 1

        n_faces = self.n_elements
        euler_char = n_vertices - n_edges_total + n_faces

        stats = {
            'n_vertices': n_vertices,
            'n_edges': n_edges_total,
            'n_triangles': n_faces,
            'euler_characteristic': euler_char,
            'n_boundary_edges': len(self.boundary_edges),
            'n_internal_edges': n_edges_total - len(self.boundary_edges),
        }

        return stats
