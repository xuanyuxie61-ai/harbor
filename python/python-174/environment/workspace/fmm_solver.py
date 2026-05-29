"""
fmm_solver.py
FMM核心求解器

科学背景:
本模块实现完整的FMM计算流程:
    1. 构建八叉树
    2. 自底向上计算多极矩 (P2M -> M2M)
    3. 自顶向下计算局部展开 (M2L -> L2L)
    4. 叶子节点直接计算近场相互作用
    5. 汇总远场(局部展开)和近场(直接求和)贡献

复杂度分析:
    - 直接求和: O(N^2)
    - FMM: O(N * L^2) + O(N * log N) (树构建)
    当 L 为常数时, FMM达到线性复杂度 O(N)

误差控制:
    - 展开阶数 L 控制截断误差
    - 分离参数 s 控制M2L精度
    - 最大粒子数/深度控制划分粒度
"""

import numpy as np
from fmm_tree import FMMOctree
from multipole_expansion import MultipoleExpansion
from local_expansion import LocalExpansion
from translation_operators import m2m_translate, m2l_translate, l2l_translate
from nbody_kernel import coulomb_potential_direct


class FMMSolver:
    """快速多极子方法求解器"""

    def __init__(self, points, charges, order=4, max_depth=6, max_particles=20, separation_param=2.0):
        """
        参数:
            points: ndarray (N, 3)
            charges: ndarray (N,)
            order: int, 展开阶数
            max_depth: int, 最大树深度
            max_particles: int, 叶子最大粒子数
            separation_param: float, 分离参数
        """
        self.points = np.asarray(points, dtype=float)
        self.charges = np.asarray(charges, dtype=float)
        self.N = self.points.shape[0]
        self.order = order
        self.max_depth = max_depth
        self.max_particles = max_particles
        self.separation_param = separation_param

        # 构建树
        self.tree = FMMOctree(self.points, self.charges, max_depth, max_particles, order, separation_param)

        # 计算多极矩
        self.tree.compute_moments_upward()

        # 计算局部展开
        self._compute_local_expansions()

    def _compute_local_expansions(self):
        """自顶向下计算局部展开"""
        all_nodes = self.tree.get_all_nodes()
        # 初始化所有节点的局部展开
        for node in all_nodes:
            node.local_coeffs_real = []
            node.local_coeffs_imag = []
            for l in range(node.order + 1):
                node.local_coeffs_real.append(np.zeros(l + 1))
                node.local_coeffs_imag.append(np.zeros(l + 1))

        # 自顶向下
        def downward(node):
            if node.parent is not None:
                # L2L: 从父节点下传
                child_real, child_imag = l2l_translate(
                    node.parent.local_coeffs_real,
                    node.parent.local_coeffs_imag,
                    node.parent.center,
                    node.center,
                    node.order
                )
                for l in range(node.order + 1):
                    m_len = min(len(child_real[l]), len(node.local_coeffs_real[l]))
                    node.local_coeffs_real[l][:m_len] += child_real[l][:m_len]
                    node.local_coeffs_imag[l][:m_len] += child_imag[l][:m_len]

            # M2L: 从交互列表转换
            interaction_list = node.get_interaction_list(all_nodes)
            for src in interaction_list:
                if (len(src.particle_indices) == 0 and src.is_leaf()
                        and np.sum(np.abs(src.multipole_moments_real[0])) < 1e-15):
                    continue
                l_real, l_imag = m2l_translate(
                    src.multipole_moments_real,
                    src.multipole_moments_imag,
                    src.center,
                    node.center,
                    node.order
                )
                for l in range(node.order + 1):
                    m_len = min(len(l_real[l]), len(node.local_coeffs_real[l]))
                    node.local_coeffs_real[l][:m_len] += l_real[l][:m_len]
                    node.local_coeffs_imag[l][:m_len] += l_imag[l][:m_len]

            if not node.is_leaf():
                for child in node.children:
                    downward(child)

        downward(self.tree.root)

    def compute_potential(self):
        """
        计算所有粒子处的势能
        
        采用稳健的近场直接求和 + 远场多极展开策略:
        - 对于相邻叶子节点: 直接P2P求和
        - 对于非相邻叶子节点: 使用多极展开直接评估 (M2P)
        
        返回:
            ndarray (N,), 势能
        """
        potential = np.zeros(self.N)
        leaves = self.tree.get_leaves()
        all_nodes = self.tree.get_all_nodes()

        for leaf in leaves:
            if len(leaf.particle_indices) == 0:
                continue

            # 确定邻居叶子
            neighbors = leaf.get_neighbors(all_nodes)
            neighbor_leaves = set()
            for nb in neighbors:
                if nb.is_leaf():
                    neighbor_leaves.add(id(nb))
                else:
                    for lf in nb.collect_leaves():
                        neighbor_leaves.add(id(lf))
            neighbor_leaves.add(id(leaf))

            # 收集邻居粒子索引
            neighbor_indices = []
            for lf in leaves:
                if id(lf) in neighbor_leaves:
                    neighbor_indices.extend(lf.particle_indices)
            neighbor_indices = list(set(neighbor_indices))
            neighbor_pts = self.points[neighbor_indices]
            neighbor_chg = self.charges[neighbor_indices]

            # 对于每个粒子, 计算近场+远场
            for idx in leaf.particle_indices:
                pt = self.points[idx]
                phi = 0.0

                # 近场: 直接求和 (邻居叶子)
                diff = pt - neighbor_pts
                dist = np.linalg.norm(diff, axis=1)
                mask = (dist > 1e-12) & (neighbor_chg != 0.0)
                if np.any(mask):
                    phi += np.sum(neighbor_chg[mask] / dist[mask])

                # TODO(Hole_3): 实现远场多极展开势能评估
                # 遍历所有非邻居的叶子节点, 使用多极展开(M2P)计算远场贡献
                # 步骤:
                # 1. 跳过 neighbor_leaves 中的叶子和空叶子
                # 2. 构造 MultipoleExpansion 对象, 将 leaf 的多极矩赋值给它
                # 3. 调用 evaluate_potential 在粒子位置 pt 处评估势能
                # 4. 累加到 phi
                # 注意: 多极矩数据存储在 other_leaf.multipole_moments_real / imag 中
                raise NotImplementedError("Hole_3: 请实现远场多极展开势能评估")

        return potential

    def compute_force(self):
        """
        计算所有粒子处的受力 (简化版)
        
        返回:
            ndarray (N, 3)
        """
        force = np.zeros((self.N, 3))
        leaves = self.tree.get_leaves()
        all_nodes = self.tree.get_all_nodes()

        for leaf in leaves:
            if len(leaf.particle_indices) == 0:
                continue
            neighbors = leaf.get_neighbors(all_nodes)
            neighbor_indices = []
            for nb in neighbors:
                if nb.is_leaf():
                    neighbor_indices.extend(nb.particle_indices)
                else:
                    for lf in nb.collect_leaves():
                        neighbor_indices.extend(lf.particle_indices)
            neighbor_indices.extend(leaf.particle_indices)
            neighbor_indices = list(set(neighbor_indices))

            if len(neighbor_indices) == 0:
                continue

            neighbor_pts = self.points[neighbor_indices]
            neighbor_chg = self.charges[neighbor_indices]

            for idx in leaf.particle_indices:
                pt = self.points[idx]
                diff = pt - neighbor_pts
                dist = np.linalg.norm(diff, axis=1)
                mask = dist > 1e-12
                if np.any(mask):
                    inv_r3 = 1.0 / (dist[mask] ** 3)
                    f = np.sum((neighbor_chg[mask] * inv_r3)[:, None] * diff[mask], axis=0)
                    force[idx] += f

        return force

    def get_tree_statistics(self):
        """获取树统计信息"""
        all_nodes = self.tree.get_all_nodes()
        leaves = self.tree.get_leaves()
        depths = [node.depth for node in all_nodes]
        n_particles_per_leaf = [len(leaf.particle_indices) for leaf in leaves]
        return {
            "total_nodes": len(all_nodes),
            "total_leaves": len(leaves),
            "max_depth": max(depths) if depths else 0,
            "avg_particles_per_leaf": float(np.mean(n_particles_per_leaf)) if n_particles_per_leaf else 0.0,
            "min_particles_per_leaf": int(np.min(n_particles_per_leaf)) if n_particles_per_leaf else 0,
            "max_particles_per_leaf": int(np.max(n_particles_per_leaf)) if n_particles_per_leaf else 0,
        }
