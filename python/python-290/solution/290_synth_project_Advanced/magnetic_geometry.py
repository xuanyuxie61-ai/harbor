"""
magnetic_geometry.py - 磁平衡几何与自适应网格生成模块

本模块构建阿尔芬波模拟所需的磁平衡几何，包括:
  - 磁通量面几何 (椭圆形截面)
  - 度量张量计算 (非正交曲线坐标)
  - 自适应网格生成 (基于 CVT 度量张量方法)
  - 等离子体边界几何提取
  - 网格数据 I/O

核心算法融合了以下种子项目:
  - cvt_metric (258): 基于度量张量的 Centroidal Voronoi 镶嵌
  - image_boundary (573): 边界轮廓提取与 POLY 格式输出
  - triangulation_display (1336): 三角网格连通性与邻接关系
  - gmsh_io (474): Gmsh 网格格式读写

物理背景:
  在非圆截面托卡马克中，磁通量面具有椭圆或 D 形截面。
  描述磁场几何需要度量张量:
    g_{ij} = (∂r/∂ξⁱ) · (∂r/∂ξʲ)
  其中 ξⁱ = (ψ, θ, φ) 为磁坐标。

  阿尔芬波方程在非正交坐标中:
    ∂ψ/∂t = (1/√g) ∂_i (√g g^{ij} ∂_j ψ)
  其中 g = det(g_{ij}) 为度量行列式。

作者: DA 博士级合成项目 PROJECT_290
"""

import numpy as np
import os
import json


class MagneticGeometry:
    """
    封装托卡马克磁平衡几何信息。

    提供从简单圆形截面到椭圆截面的几何描述，
    并计算度量张量、Jacobian 等几何量。
    """

    def __init__(self, R0, a_minor, elongation=1.0, triangularity=0.0,
                 n_flux_surfaces=32, n_theta_geo=64):
        """
        参数:
          R0: float, 大半径 [m]
          a_minor: float, 小半径 [m]
          elongation: float,  elongation κ (椭圆拉长比)
          triangularity: float, 三角变形 δ
          n_flux_surfaces: int, 通量面数量
          n_theta_geo: int, 极向角度分辨率
        """
        self.R0 = R0
        self.a_minor = a_minor
        self.kappa = elongation        # 拉长比 κ = b/a
        self.delta = triangularity     # 三角变形 δ
        self.n_flux = n_flux_surfaces
        self.n_theta = n_theta_geo

        # 通量面归一化半径数组
        self.rho_array = np.linspace(0.0, 1.0, self.n_flux)

        # 极向角度数组
        self.theta_array = np.linspace(0.0, 2.0 * np.pi, self.n_theta, endpoint=False)

        # 预计算几何量
        self._compute_geometry()

    def _compute_geometry(self):
        """
        预计算所有通量面上的几何量。

        Miller 参数化:
          R(r, θ) = R₀ + r cos(θ + δ sin θ)
          Z(r, θ) = κ r sin θ

        对于简单椭圆 (δ=0):
          R(r, θ) = R₀ + r cos θ
          Z(r, θ) = κ r sin θ
        """
        n_rho = self.n_flux
        n_th = self.n_theta

        # 2D 网格
        rho_2d, theta_2d = np.meshgrid(self.rho_array, self.theta_array, indexing='ij')
        r_2d = rho_2d * self.a_minor  # 物理半径

        # 改进的 Miller 参数化
        if abs(self.delta) > 1e-10:
            theta_eff = theta_2d + np.arcsin(self.delta) * np.sin(theta_2d)
        else:
            theta_eff = theta_2d

        # 大半径 R(ρ, θ) 和垂直坐标 Z(ρ, θ)
        self.R_2d = self.R0 + r_2d * np.cos(theta_eff)
        self.Z_2d = self.kappa * r_2d * np.sin(theta_eff)

        # 度量张量分量
        # g_ρρ = (∂R/∂ρ)² + (∂Z/∂ρ)²
        # g_θθ = (∂R/∂θ)² + (∂Z/∂θ)²
        # g_ρθ = (∂R/∂ρ)(∂R/∂θ) + (∂Z/∂ρ)(∂Z/∂θ)
        dR_drho = self.a_minor * np.cos(theta_eff)
        dZ_drho = self.kappa * self.a_minor * np.sin(theta_eff)

        if abs(self.delta) > 1e-10:
            dtheta_eff_dtheta = 1.0 + np.arcsin(self.delta) * np.cos(theta_2d)
        else:
            dtheta_eff_dtheta = 1.0

        dR_dtheta = -r_2d * np.sin(theta_eff) * dtheta_eff_dtheta
        dZ_dtheta = self.kappa * r_2d * np.cos(theta_2d)

        self.g_rho_rho = dR_drho ** 2 + dZ_drho ** 2
        self.g_theta_theta = dR_dtheta ** 2 + dZ_dtheta ** 2
        self.g_rho_theta = dR_drho * dR_dtheta + dZ_drho * dZ_dtheta

        # Jacobian: J = √(g_ρρ g_θθ - g_ρθ²)
        det_g = self.g_rho_rho * self.g_theta_theta - self.g_rho_theta ** 2
        det_g = np.maximum(det_g, 1e-30)  # 确保正定
        self.jacobian = np.sqrt(det_g)

        # 逆度量张量
        self.g_inv_rho_rho = self.g_theta_theta / det_g
        self.g_inv_theta_theta = self.g_rho_rho / det_g
        self.g_inv_rho_theta = -self.g_rho_theta / det_g

    def get_metric_at_point(self, rho, theta):
        """
        在指定 (ρ, θ) 点插值获取度量张量。

        参数:
          rho: float, 归一化半径
          theta: float, 极向角 [rad]

        返回:
          metric: dict, 包含 g_ρρ, g_θθ, g_ρθ, J 等
        """
        rho = np.clip(rho, 0.0, 1.0)
        theta = theta % (2.0 * np.pi)

        # 最近邻索引 (简单实现)
        i_rho = int(round(rho * (self.n_flux - 1)))
        i_rho = min(i_rho, self.n_flux - 1)
        i_theta = int(round(theta / (2.0 * np.pi) * self.n_theta)) % self.n_theta

        return {
            'g_rho_rho': self.g_rho_rho[i_rho, i_theta],
            'g_theta_theta': self.g_theta_theta[i_rho, i_theta],
            'g_rho_theta': self.g_rho_theta[i_rho, i_theta],
            'jacobian': self.jacobian[i_rho, i_theta],
            'R': self.R_2d[i_rho, i_theta],
            'Z': self.Z_2d[i_rho, i_theta],
        }

    def get_boundary_poly(self):
        """
        提取等离子体边界 (ρ=1) 的多边形表示。

        改编自 image_boundary (573) 的边界轮廓提取思想。
        将连续的边界离散为多边形节点和边列表，
        可用于后续网格生成或边界条件施加。

        返回:
          nodes: ndarray, shape (N, 2), 边界节点 (R, Z)
          edges: ndarray, shape (N, 2), 边连接 (i, i+1)
        """
        i_boundary = self.n_flux - 1
        R_b = self.R_2d[i_boundary, :]
        Z_b = self.Z_2d[i_boundary, :]
        n_nodes = len(R_b)

        nodes = np.column_stack([R_b, Z_b])
        edges = np.column_stack([
            np.arange(n_nodes),
            np.roll(np.arange(n_nodes), -1)
        ])

        return nodes, edges

    def write_poly_file(self, filename):
        """
        将等离子体边界写入 POLY 格式文件。

        POLY 格式 (Triangle/Shewchuk):
          第一行: <node_num> <dim> <n_attr> <n_marker>
          后续行: <node_idx> <x> <y>
          边段: <edge_num> <marker>
          后续行: <edge_idx> <n1> <n2>

        改编自 image_boundary 的 boundary_to_poly。
        """
        nodes, edges = self.get_boundary_poly()
        n_nodes = len(nodes)
        n_edges = len(edges)

        with open(filename, 'w') as f:
            # 节点段
            f.write(f"{n_nodes} 2 0 1\n")
            for i in range(n_nodes):
                f.write(f"{i} {nodes[i, 0]:.8e} {nodes[i, 1]:.8e} 1\n")
            # 边段
            f.write(f"{n_edges} 1\n")
            for i in range(n_edges):
                f.write(f"{i} {edges[i, 0]} {edges[i, 1]} 1\n")
            # 孔洞段
            f.write("0\n")

    def write_gmsh_mesh(self, filename):
        """
        将磁通量面网格写入 Gmsh MSH 格式。

        改编自 gmsh_io (474) 项目的 Gmsh 格式读写。
        使用 MSH 2.2 格式。

        参数:
          filename: str, 输出文件路径
        """
        # 构建节点: 每个通量面 × 极向角度的 (R, Z, φ=0) 点
        nodes_list = []
        node_id = 1
        node_map = {}

        for i_rho in range(self.n_flux):
            for i_theta in range(self.n_theta):
                R = self.R_2d[i_rho, i_theta]
                Z = self.Z_2d[i_rho, i_theta]
                nodes_list.append((node_id, R, Z, 0.0))
                node_map[(i_rho, i_theta)] = node_id
                node_id += 1

        n_nodes = len(nodes_list)

        # 构建四边形单元 (2节点线单元用于通量面)
        elements_list = []
        elem_id = 1

        # 类型 1: 线单元 (Gmsh type 1) - 极向连接
        for i_rho in range(self.n_flux):
            for i_theta in range(self.n_theta):
                n1 = node_map[(i_rho, i_theta)]
                n2 = node_map[(i_rho, (i_theta + 1) % self.n_theta)]
                elements_list.append((elem_id, 1, [n1, n2]))
                elem_id += 1

        # 类型 1: 线单元 - 径向连接
        for i_rho in range(self.n_flux - 1):
            for i_theta in range(self.n_theta):
                n1 = node_map[(i_rho, i_theta)]
                n2 = node_map[(i_rho + 1, i_theta)]
                elements_list.append((elem_id, 1, [n1, n2]))
                elem_id += 1

        n_elements = len(elements_list)

        with open(filename, 'w') as f:
            f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

            f.write(f"$Nodes\n{n_nodes}\n")
            for nid, x, y, z in nodes_list:
                f.write(f"{nid} {x:.8e} {y:.8e} {z:.8e}\n")
            f.write("$EndNodes\n")

            f.write(f"$Elements\n{n_elements}\n")
            for eid, etype, enodes in elements_list:
                node_str = " ".join(str(n) for n in enodes)
                f.write(f"{eid} {etype} 2 0 0 {node_str}\n")
            f.write("$EndElements\n")


class CVTAdaptiveGrid:
    """
    基于度量张量的 Centroidal Voronoi 镶嵌自适应网格。

    改编自 cvt_metric (258) 项目。
    在等离子体中，阿尔芬连续谱在某些区域变化剧烈
    (如共振面附近)，需要自适应加密网格。

    CVT 算法 (Lloyd 迭代):
      1. 初始化 n 个生成点
      2. 对于每个采样点，使用度量距离找最近生成点
         d²_A(p, g) = (p-g)^T A((p+g)/2) (p-g)
      3. 将生成点更新为所属 Voronoi 细胞的质心
      4. 重复至收敛

    度量张量 A(p) 由阿尔芬连续谱梯度决定:
      A(p) = I + α |∇ω_A(p)| / max|∇ω_A|
    使得连续谱变化剧烈的区域网格更密。
    """

    def __init__(self, n_generators, n_samples, geometry, alpha_refine=5.0):
        """
        参数:
          n_generators: int, 生成点数量
          n_samples: int, 采样点数量
          geometry: MagneticGeometry, 磁平衡几何
          alpha_refine: float, 加密因子
        """
        self.n_gen = n_generators
        self.n_samples = n_samples
        self.geometry = geometry
        self.alpha = alpha_refine
        self.generators = None
        self.n_iterations = 16

    def _metric_tensor(self, p_mid):
        """
        计算中点处的度量张量。

        A(p) = I + α |∇ω_A| / max|∇ω_A|

        简化实现: 使用椭圆度作为度量权重。

        参数:
          p_mid: ndarray, shape (2,), 中点坐标 (ρ, θ)

        返回:
          A: ndarray, shape (2, 2), 度量张量
        """
        rho = np.clip(p_mid[0], 0.01, 0.99)
        metric_info = self.geometry.get_metric_at_point(rho, p_mid[1])

        # 使用度量张量的迹作为权重
        trace_g = metric_info['g_rho_rho'] + metric_info['g_theta_theta']
        j = metric_info['jacobian']

        # 加密因子: Jacobian 小的区域需要更密的网格
        weight = 1.0 + self.alpha / (j + 1e-10)

        A = weight * np.eye(2)
        return A

    def _metric_distance_sq(self, p, g):
        """
        度量距离的平方:
          d²_A(p, g) = (p-g)^T A((p+g)/2) (p-g)

        改编自 cvt_metric。
        """
        diff = p - g
        mid = 0.5 * (p + g)
        A = self._metric_tensor(mid)
        return diff @ A @ diff

    def generate(self, seed=42):
        """
        运行 CVT Lloyd 迭代生成自适应网格。

        返回:
          generators: ndarray, shape (n_gen, 2), 生成点坐标
        """
        rng = np.random.RandomState(seed)

        # 初始化: 在 (ρ, θ) 域中随机放置生成点
        self.generators = np.zeros((self.n_gen, 2))
        self.generators[:, 0] = rng.uniform(0.05, 0.95, self.n_gen)
        self.generators[:, 1] = rng.uniform(0.0, 2.0 * np.pi, self.n_gen)

        for iteration in range(self.n_iterations):
            # 采样点
            samples = np.zeros((self.n_samples, 2))
            samples[:, 0] = rng.uniform(0.0, 1.0, self.n_samples)
            samples[:, 1] = rng.uniform(0.0, 2.0 * np.pi, self.n_samples)

            # 分配: 对每个采样点找最近生成点
            assignment = np.zeros(self.n_samples, dtype=int)
            for s in range(self.n_samples):
                min_dist = float('inf')
                for g in range(self.n_gen):
                    d2 = self._metric_distance_sq(samples[s], self.generators[g])
                    if d2 < min_dist:
                        min_dist = d2
                        assignment[s] = g

            # 更新: 计算每个 Voronoi 细胞的质心
            new_gen = np.zeros_like(self.generators)
            count = np.zeros(self.n_gen, dtype=int)
            for s in range(self.n_samples):
                g = assignment[s]
                new_gen[g] += samples[s]
                count[g] += 1

            # 仅更新有采样点的生成点
            for g in range(self.n_gen):
                if count[g] > 0:
                    self.generators[g] = new_gen[g] / count[g]

            # 边界约束
            self.generators[:, 0] = np.clip(self.generators[:, 0], 0.01, 0.99)
            self.generators[:, 1] = self.generators[:, 1] % (2.0 * np.pi)

        return self.generators


def compute_triangulation_neighbors(n_rho, n_theta):
    """
    计算结构化四边形网格的三角化邻接关系。

    改编自 triangulation_display (1336) 的网格连通性处理。
    将 (n_rho × n_theta) 的结构化网格三角化为
    2 × (n_rho-1) × n_theta 个三角形，并计算邻接关系。

    参数:
      n_rho: int, 径向网格数
      n_theta: int, 极向网格数

    返回:
      elements: ndarray, shape (n_tri, 3), 三角形节点索引
      neighbors: ndarray, shape (n_tri, 3), 邻接三角形索引 (-1 表示边界)
    """
    elements = []
    for i in range(n_rho - 1):
        for j in range(n_theta):
            j_next = (j + 1) % n_theta
            n0 = i * n_theta + j
            n1 = i * n_theta + j_next
            n2 = (i + 1) * n_theta + j
            n3 = (i + 1) * n_theta + j_next

            # 两个三角形构成一个四边形
            elements.append([n0, n1, n2])
            elements.append([n1, n3, n2])

    elements = np.array(elements, dtype=int)
    n_tri = len(elements)

    # 计算邻接关系 (简化版)
    neighbors = -np.ones((n_tri, 3), dtype=int)
    # 基于共享边的邻接检测
    for t1 in range(n_tri):
        for t2 in range(t1 + 1, n_tri):
            shared = len(set(elements[t1]) & set(elements[t2]))
            if shared == 2:
                # 找共享边在两个三角形中的位置
                for e1 in range(3):
                    edge1 = set([elements[t1, e1], elements[t1, (e1 + 1) % 3]])
                    if len(edge1 & set(elements[t2])) == 2:
                        for e2 in range(3):
                            edge2 = set([elements[t2, e2], elements[t2, (e2 + 1) % 3]])
                            if edge1 == edge2:
                                neighbors[t1, e1] = t2
                                neighbors[t2, e2] = t1

    return elements, neighbors
