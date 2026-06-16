"""
adaptive_mesh.py — 异质结自适应网格生成
==========================================
核心科学问题: 在异质结界面附近生成自适应加密网格,
以精确解析势能剖面的陡峭梯度 (界面处 V(z) 的快速变化).

融合种子项目:
  - 757_mesh2d: 基于四叉树的自适应网格生成
  - 1350_triangulation_refine: 三角形网格细分策略
  - 754_mesh_display: 网格I/O和拓扑操作 (去掉可视化)
"""

import numpy as np


class AdaptiveMesh1D:
    """
    一维自适应网格生成器.

    使用等分布原理 (equidistribution) 生成自适应网格:
        integral from z_{i-1} to z_i of w(z) dz = constant

    其中 w(z) 是监视函数 (monitor function), 通常是:
        w(z) = 1 + alpha * |d²V/dz²|^(1/2)
    或
        w(z) = 1 + alpha * |dV/dz|^p

    物理意义: 在势能变化剧烈的地方 (异质结界面)
    自动加密网格.
    """

    def __init__(self, z_min, z_max, N_init, potential_func=None,
                 refinement_ratio=2.0, max_refinement_level=5,
                 tolerance=1e-6):
        """
        参数
        ----
        z_min, z_max : float
            计算域边界 [m]
        N_init : int
            初始均匀网格点数
        potential_func : callable or None
            势能函数 V(z), 用于计算监视函数
        refinement_ratio : float
            最大相邻网格比
        max_refinement_level : int
            最大细分层数
        tolerance : float
            收敛容差
        """
        self.z_min = z_min
        self.z_max = z_max
        self.N_init = N_init
        self.potential_func = potential_func
        self.refinement_ratio = refinement_ratio
        self.max_level = max_refinement_level
        self.tolerance = tolerance

        # 初始均匀网格
        self.z_nodes = np.linspace(z_min, z_max, N_init)
        self.level = 0

        # 如果有势能函数, 进行自适应加密
        if potential_func is not None:
            self._adapt_mesh()

    def _compute_monitor_function(self, z):
        """
        监视函数 (monitor function):
            w(z) = 1 + alpha * |V''(z)|^{1/2}

        自适应理论依据:
            对于方程 -d/dz(p(z) dψ/dz) + V(z)ψ = Eψ,
            等分布网格使得截断误差均匀分布.
        """
        if self.potential_func is None:
            return np.ones_like(z)

        dz = z[1] - z[0] if len(z) > 1 else 1e-15
        V = np.array([self.potential_func(zi) for zi in z])

        # 二阶差分近似 V''(z)
        d2Vdz2 = np.zeros_like(z)
        if len(z) >= 3:
            for i in range(1, len(z) - 1):
                dz_local = (z[i + 1] - z[i - 1]) / 2.0
                dz_local = max(dz_local, 1e-20)
                d2Vdz2[i] = (V[i + 1] - 2 * V[i] + V[i - 1]) / dz_local**2

        # 监视函数参数
        V_range = np.max(np.abs(V)) if np.max(np.abs(V)) > 0 else 1.0
        alpha = 1.0 / max(V_range, 1e-10)

        w = 1.0 + alpha * np.sqrt(np.abs(d2Vdz2))
        return w

    def _adapt_mesh(self):
        """
        等分布自适应网格生成.

        算法:
        1. 计算监视函数 w(z)
        2. 计算累积分布 theta(z) = integral_0^z w(s) ds
        3. 等分 theta: theta_i = i * theta_total / N
        4. 反解 z_i = theta^{-1}(theta_i)
        """
        for level in range(self.max_level):
            self.level = level

            # 计算监视函数
            w = self._compute_monitor_function(self.z_nodes)

            # 累积积分 (梯形法则)
            dz = np.diff(self.z_nodes)
            w_avg = 0.5 * (w[:-1] + w[1:])
            theta = np.zeros(len(self.z_nodes))
            theta[1:] = np.cumsum(w_avg * dz)
            theta_total = theta[-1]

            if theta_total < 1e-15:
                break

            # 等分 theta
            N = len(self.z_nodes)
            theta_uniform = np.linspace(0, theta_total, N)

            # 反插值: theta → z
            z_new = np.interp(theta_uniform, theta, self.z_nodes)

            # 检查收敛
            max_diff = np.max(np.abs(z_new - self.z_nodes))
            self.z_nodes = z_new

            if max_diff < self.tolerance * (self.z_max - self.z_min):
                break

            # 检查相邻网格比
            dz_new = np.diff(self.z_nodes)
            max_ratio = np.max(dz_new[1:] / dz_new[:-1]) if len(dz_new) > 1 else 1.0
            if max_ratio > self.refinement_ratio:
                # 平滑网格
                self.z_nodes = self._smooth_mesh(self.z_nodes, iterations=5)

    def _smooth_mesh(self, z, iterations=3):
        """
        网格平滑 (Laplace 平滑 + 约束):
            z_i^{new} = (z_{i-1} + z_{i+1}) / 2
        保持边界不变.
        """
        z_smooth = z.copy()
        for _ in range(iterations):
            for i in range(1, len(z_smooth) - 1):
                z_smooth[i] = 0.5 * (z_smooth[i - 1] + z_smooth[i + 1])
        z_smooth[0] = z[0]
        z_smooth[-1] = z[-1]
        return z_smooth

    def refine_by_bisection(self, indices):
        """
        二分法局部加密 (融合 1350_triangulation_refine 思想):
        在指定区间插入中点.

        参数
        ----
        indices : list of int
            需要加密的区间索引
        """
        z_new = [self.z_nodes[0]]
        for i in range(len(self.z_nodes) - 1):
            z_new.append(self.z_nodes[i + 1])
            if i in indices:
                z_mid = 0.5 * (self.z_nodes[i] + self.z_nodes[i + 1])
                z_new.append(z_mid)
        self.z_nodes = np.sort(np.array(z_new))

    def refine_near_interface(self, z_interface, width):
        """
        在界面附近加密网格.

        参数
        ----
        z_interface : float
            界面位置 [m]
        width : float
            加密区域半宽度 [m]
        """
        indices = []
        for i in range(len(self.z_nodes) - 1):
            z_mid = 0.5 * (self.z_nodes[i] + self.z_nodes[i + 1])
            if abs(z_mid - z_interface) < width:
                indices.append(i)

        if indices:
            self.refine_by_bisection(indices)

    def quadtree_refine(self, level):
        """
        四叉树风格均匀加密 (融合 757_mesh2d 的 quadtree 思想):
        每层将每个区间二分.

        参数
        ----
        level : int
            加密层数
        """
        for _ in range(level):
            z_new = []
            for i in range(len(self.z_nodes)):
                z_new.append(self.z_nodes[i])
                if i < len(self.z_nodes) - 1:
                    z_mid = 0.5 * (self.z_nodes[i] + self.z_nodes[i + 1])
                    z_new.append(z_mid)
            self.z_nodes = np.array(z_new)

    def get_grid_spacing(self):
        """返回网格间距数组"""
        return np.diff(self.z_nodes)

    def get_quality_metric(self):
        """
        网格质量度量 (均匀性):
            Q = min(dz) / max(dz)
        理想均匀网格 Q = 1.
        """
        dz = self.get_grid_spacing()
        if len(dz) == 0:
            return 1.0
        return np.min(dz) / max(np.max(dz), 1e-20)

    def jacobian(self, xi):
        """
        计算映射 Jacobian: dz/dxi
        其中 xi ∈ [0,1] 是参考坐标.

        对于自适应网格:
            dz/dxi = (z_max - z_min) / w(z(xi)) / <1/w>
        """
        z = np.interp(xi, np.linspace(0, 1, len(self.z_nodes)), self.z_nodes)
        # 数值微分
        dz_dxi = np.gradient(z, xi)
        return dz_dxi

    def __len__(self):
        return len(self.z_nodes)


class HeteroStructureMesh:
    """
    异质结专用多层网格.

    处理多层异质结结构 (如 hBN/MoS2/WSe2/hBN)
    每层独立网格, 界面处网格连续.
    """

    def __init__(self, layers, thicknesses, points_per_layer=50,
                 interface_refinement=3):
        """
        参数
        ----
        layers : list of Material2D
            材料层列表
        thicknesses : list of float
            每层厚度 [m]
        points_per_layer : int
            每层初始网格点数
        interface_refinement : int
            界面处加密次数
        """
        self.layers = layers
        self.thicknesses = thicknesses
        self.n_layers = len(layers)

        assert len(layers) == len(thicknesses)

        # 计算界面位置
        self.interfaces = np.cumsum([0] + list(thicknesses))
        self.z_total = np.sum(thicknesses)

        # 生成网格
        self.z_nodes = self._generate_layer_mesh(
            points_per_layer, interface_refinement)

    def _generate_layer_mesh(self, ppl, refinement):
        """生成多层网格"""
        z_all = []
        for i_layer in range(self.n_layers):
            z_start = self.interfaces[i_layer]
            z_end = self.interfaces[i_layer + 1]

            # 层内均匀网格
            z_layer = np.linspace(z_start, z_end, ppl)
            z_all.extend(z_layer[:-1])  # 避免重复界面点

        z_all.append(self.interfaces[-1])
        z_nodes = np.array(z_all)

        # 界面加密
        for _ in range(refinement):
            z_refined = list(z_nodes)
            for z_int in self.interfaces[1:-1]:
                # 在界面两侧加密
                for idx in range(len(z_nodes) - 1):
                    z_mid = 0.5 * (z_nodes[idx] + z_nodes[idx + 1])
                    if abs(z_mid - z_int) < (z_nodes[idx + 1] - z_nodes[idx]):
                        if z_mid not in z_refined:
                            z_refined.append(z_mid)
            z_nodes = np.sort(np.array(z_refined))

        return z_nodes

    def get_interface_positions(self):
        """返回所有界面位置"""
        return self.interfaces.copy()

    def layer_index(self, z):
        """返回 z 所在的层索引"""
        for i in range(self.n_layers):
            if self.interfaces[i] <= z <= self.interfaces[i + 1]:
                return i
        return -1

    def get_material_at(self, z):
        """返回 z 位置的材料"""
        idx = self.layer_index(z)
        if 0 <= idx < self.n_layers:
            return self.layers[idx]
        return None
