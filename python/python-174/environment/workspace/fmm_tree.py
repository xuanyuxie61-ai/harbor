"""
fmm_tree.py
FMM八叉树空间划分模块

融合种子项目:
- 242_cvt_4_movie (CVT聚类思想, 用于自适应空间划分)
- 952_quadrilateral (四边形几何计算)
- 185_circles (圆形区域边界)
- 758_mesh2d_to_medit (网格数据管理)

科学背景:
FMM依赖于层次化的空间树结构(八叉树)来组织粒子。
树中每个节点代表一个立方体区域, 包含若干粒子。
通过CVT(Centroidal Voronoi Tessellation)思想优化粒子聚类,
使得每个叶子节点内的粒子分布更均匀。

核心算法:
1. 八叉树构建:
    - 给定边界框 [xmin, xmax] x [ymin, ymax] x [zmin, zmax]
    - 若节点内粒子数 > max_particles, 则沿x,y,z中点分割为8个子节点
    - 递归直至满足条件或达到最大深度

2. 邻居列表 (Well-Separated Pairs):
    两个节点A和B是"well-separated"(充分分离)的, 当:
        dist(center_A, center_B) > s * max(radius_A, radius_B)
    其中 s >= 2 为分离参数 (通常取 s=2 或 s=3)

3. CVT优化聚类 (简化版):
    对叶子节点内的粒子, 迭代更新:
        a. 计算节点内粒子质心
        b. 若质心与节点中心偏差过大, 微调节点中心
    这改善了多极展开的收敛性

核心公式:
    - 区域半径: r = sqrt(3)/2 * side_length (对角线的一半)
    - 分离条件: d > s * r
    - 粒子到质心的二阶矩: I = sum_i q_i * |x_i - c|^2
"""

import numpy as np


class OctreeNode:
    """八叉树节点"""

    def __init__(self, center, half_size, depth=0, max_depth=8, max_particles=10, order=4):
        """
        参数:
            center: ndarray (3,), 节点中心
            half_size: float, 半边长
            depth: int, 当前深度
            max_depth: int, 最大深度
            max_particles: int, 最大粒子数
            order: int, 展开阶数
        """
        self.center = np.asarray(center, dtype=float)
        self.half_size = float(half_size)
        self.depth = depth
        self.max_depth = max_depth
        self.max_particles = max_particles
        self.order = order
        self.particle_indices = []
        self.children = None  # 8个子节点或None
        self.radius = np.sqrt(3.0) * half_size  # 外接球半径
        self.parent = None

        # FMM展开
        self.multipole_moments_real = []
        self.multipole_moments_imag = []
        self.local_coeffs_real = []
        self.local_coeffs_imag = []
        for l in range(order + 1):
            self.multipole_moments_real.append(np.zeros(l + 1))
            self.multipole_moments_imag.append(np.zeros(l + 1))
            self.local_coeffs_real.append(np.zeros(l + 1))
            self.local_coeffs_imag.append(np.zeros(l + 1))

    def is_leaf(self):
        return self.children is None

    def contains(self, point):
        """检查点是否在节点区域内"""
        point = np.asarray(point)
        return np.all(np.abs(point - self.center) <= self.half_size + 1e-12)

    def insert(self, point_idx, point, all_points=None):
        """
        插入粒子
        
        若节点已满且未达到最大深度, 则分裂
        """
        if self.is_leaf():
            self.particle_indices.append(point_idx)
            if (len(self.particle_indices) > self.max_particles
                    and self.depth < self.max_depth):
                if all_points is None:
                    all_points = point  # fallback
                self._split(all_points)
        else:
            # 插入到对应的子节点
            child_idx = self._child_index(point)
            self.children[child_idx].insert(point_idx, point, all_points)

    def _child_index(self, point):
        """确定点属于哪个子节点 (0-7)"""
        dx = point[0] >= self.center[0]
        dy = point[1] >= self.center[1]
        dz = point[2] >= self.center[2]
        return (int(dx) << 2) | (int(dy) << 1) | int(dz)

    def _split(self, points):
        """分裂节点为8个子节点, 并重新分配粒子"""
        if self.children is not None:
            return
        h = self.half_size * 0.5
        offsets = [
            [-h, -h, -h], [-h, -h,  h], [-h,  h, -h], [-h,  h,  h],
            [ h, -h, -h], [ h, -h,  h], [ h,  h, -h], [ h,  h,  h]
        ]
        self.children = []
        for off in offsets:
            child = OctreeNode(
                self.center + np.array(off),
                h,
                self.depth + 1,
                self.max_depth,
                self.max_particles,
                self.order
            )
            child.parent = self
            self.children.append(child)

        # 重新分配粒子
        temp_indices = self.particle_indices.copy()
        self.particle_indices = []
        for idx in temp_indices:
            child_idx = self._child_index(points[idx])
            self.children[child_idx].particle_indices.append(idx)

    def refine_cvt(self, points, charges, max_iter=5):
        """
        CVT优化: 调整节点中心使其接近质心 (融合242_cvt_4_movie)
        
        算法:
            for iter in range(max_iter):
                centroid = sum_i q_i * x_i / sum_i q_i
                center = 0.5 * center + 0.5 * centroid  (阻尼更新)
        
        参数:
            points: ndarray (N, 3)
            charges: ndarray (N,)
        """
        if len(self.particle_indices) == 0:
            return
        local_points = points[self.particle_indices]
        local_charges = np.abs(charges[self.particle_indices])
        total_c = np.sum(local_charges)
        if total_c < 1e-15:
            return
        for _ in range(max_iter):
            centroid = np.sum(local_charges[:, None] * local_points, axis=0) / total_c
            self.center = 0.5 * self.center + 0.5 * centroid

    def collect_leaves(self):
        """收集所有叶子节点"""
        if self.is_leaf():
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.collect_leaves())
        return leaves

    def get_all_nodes(self):
        """获取所有节点 (先序遍历)"""
        nodes = [self]
        if self.children is not None:
            for child in self.children:
                nodes.extend(child.get_all_nodes())
        return nodes

    def bounding_quadrilateral_area(self):
        """
        计算节点边界四边形在xy平面的投影面积 (融合952_quadrilateral)
        
        节点边界为立方体, 在xy平面的投影为矩形 (也是四边形)
        """
        hs = self.half_size
        quad = np.array([
            [self.center[0] - hs, self.center[1] - hs],
            [self.center[0] + hs, self.center[1] - hs],
            [self.center[0] + hs, self.center[1] + hs],
            [self.center[0] - hs, self.center[1] + hs]
        ])
        # 矩形面积 = 底 * 高
        area = (2 * hs) * (2 * hs)
        return area, quad

    def well_separated_from(self, other, separation_param=2.0):
        """
        检查两个节点是否充分分离
        
        条件:
            dist(center_A, center_B) > separation_param * max(radius_A, radius_B)
        """
        dist = np.linalg.norm(self.center - other.center)
        return dist > separation_param * max(self.radius, other.radius)

    def is_adjacent(self, other):
        """检查两个节点是否相邻 (边界接触或重叠)"""
        # 若两个立方体边界距离 <= 0, 则相邻
        dist_per_dim = np.abs(self.center - other.center)
        sum_half = self.half_size + other.half_size
        return np.all(dist_per_dim <= sum_half + 1e-12)

    def get_neighbors(self, all_nodes):
        """获取所有相邻节点"""
        neighbors = []
        for node in all_nodes:
            if node is not self and self.is_adjacent(node):
                neighbors.append(node)
        return neighbors

    def get_interaction_list(self, all_nodes):
        """
        获取交互列表:
        - 父节点的邻居的子节点中, 既非当前节点邻居也非当前节点本身的节点
        """
        if self.parent is None:
            return []
        interaction = []
        parent_neighbors = self.parent.get_neighbors(all_nodes)
        for pn in parent_neighbors:
            if pn.is_leaf():
                if pn is not self and not self.is_adjacent(pn) and pn not in interaction:
                    interaction.append(pn)
            else:
                for child in pn.children:
                    if child is not self and not self.is_adjacent(child) and child not in interaction:
                        interaction.append(child)
        return interaction


class FMMOctree:
    """FMM八叉树"""

    def __init__(self, points, charges, max_depth=6, max_particles=20, order=4, separation_param=2.0):
        """
        参数:
            points: ndarray (N, 3)
            charges: ndarray (N,)
            max_depth: int
            max_particles: int
            order: int, 展开阶数
            separation_param: float, 分离参数
        """
        self.points = np.asarray(points, dtype=float)
        self.charges = np.asarray(charges, dtype=float)
        self.N = self.points.shape[0]
        self.max_depth = max_depth
        self.max_particles = max_particles
        self.order = order
        self.separation_param = separation_param

        # 计算边界框
        min_coord = np.min(self.points, axis=0)
        max_coord = np.max(self.points, axis=0)
        center = 0.5 * (min_coord + max_coord)
        size = np.max(max_coord - min_coord)
        if size < 1e-10:
            size = 1.0
        half_size = size * 0.5 * 1.01  # 稍微放大以避免边界问题

        self.root = OctreeNode(center, half_size, depth=0, max_depth=max_depth,
                               max_particles=max_particles, order=order)

        # 插入所有粒子
        for i in range(self.N):
            self.root.insert(i, self.points[i], self.points)

        # 重新分配分裂节点的粒子 (因为_split时需要原始位置)
        self._redistribute_particles(self.root)

        # CVT优化
        leaves = self.root.collect_leaves()
        for leaf in leaves:
            leaf.refine_cvt(self.points, self.charges)

    def _redistribute_particles(self, node):
        """递归重新分配粒子到叶子节点"""
        if node.is_leaf():
            return
        # 收集所有待重新分配的粒子
        particles_to_redistribute = []
        # 从子节点收集 (如果有)
        for child in node.children:
            particles_to_redistribute.extend(child.particle_indices)
            child.particle_indices = []
        # 加上当前节点的 (分裂时清空的)
        # 但node.particle_indices在split后已清空
        # 需要将原始粒子从根往下重新分配
        # 这里采用更简单的方式: 从根节点持有所有索引开始
        pass  # 实际上在insert时已正确分配, 除了split清空的需要处理

    def _build_downward(self, node):
        """自顶向下构建: 分裂后重新分配"""
        if node.is_leaf():
            return
        # 若节点有粒子但已分裂, 需要重新分配 (实际不应发生)
        if len(node.particle_indices) > 0:
            temp = node.particle_indices.copy()
            node.particle_indices = []
            for idx in temp:
                child_idx = node._child_index(self.points[idx])
                node.children[child_idx].particle_indices.append(idx)
        for child in node.children:
            self._build_downward(child)

    def rebuild(self):
        """重建树结构 (在CVT优化后)"""
        # 简化: 重新构建
        min_coord = np.min(self.points, axis=0)
        max_coord = np.max(self.points, axis=0)
        center = 0.5 * (min_coord + max_coord)
        size = np.max(max_coord - min_coord)
        if size < 1e-10:
            size = 1.0
        half_size = size * 0.5 * 1.01

        self.root = OctreeNode(center, half_size, depth=0, max_depth=self.max_depth,
                               max_particles=self.max_particles, order=self.order)
        for i in range(self.N):
            self.root.insert(i, self.points[i])
        self._build_downward(self.root)

    def get_all_nodes(self):
        return self.root.get_all_nodes()

    def get_leaves(self):
        return self.root.collect_leaves()

    def compute_moments_upward(self):
        """自底向上计算多极矩"""
        # TODO(Hole_2): 实现自底向上的多极矩计算
        # 分为两个阶段:
        # 1. P2M: 对每个叶子节点, 使用 MultipoleExpansion.add_particles 计算多极矩
        #    并将结果存储到 leaf.multipole_moments_real / imag
        # 2. M2M: 自底向上遍历内部节点, 对每个子节点调用 m2m_translate
        #    将子节点多极矩转换到父节点中心并累加
        # 注意: 空叶子节点可跳过; 结果数组格式为列表, 每个元素是长度为(l+1)的ndarray
        raise NotImplementedError("Hole_2: 请实现compute_moments_upward")
