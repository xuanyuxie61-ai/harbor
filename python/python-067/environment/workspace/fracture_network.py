# -*- coding: utf-8 -*-
"""
fracture_network.py
裂隙网络生成与拓扑分析模块

基于种子项目 227_cross_chaos（迭代函数系统 IFS）和 673_lights_out_game（网格状态切换）
的算法融合，用于生成具有自相似分形特征的二维裂隙网络，并分析其连通性。

核心模型：
    1. 分形裂隙网络生成（IFS 仿射变换）
    2. 裂隙开度场矩阵（网格状态模型）
    3. 裂隙连通性分析（图论 + 渗流阈值）

物理公式：
    裂隙水力传导系数（Cubic Law）:
        T = (ρ g b³) / (12 μ)
    
    等效渗透率（Snow, 1969）:
        k_eq = (1/12) * Σ(b_i³ * cos²θ_i) / A
    
    渗流阈值判定（Bond Percolation）:
        p_c ≈ 0.5 （二维方格网络）
"""

import numpy as np
from typing import List, Tuple, Optional
from random_generator import MiddleSquareGenerator


class FractureNetwork:
    """
    二维裂隙网络模型

    使用迭代函数系统（IFS）生成分形裂隙网络，并通过网格状态切换
    模型模拟裂隙的开闭状态（力学加载或化学堵塞）。
    """

    def __init__(self, domain_size: Tuple[float, float] = (100.0, 100.0),
                 nx: int = 50, ny: int = 50, seed: int = 1234):
        """
        初始化裂隙网络

        Parameters
        ----------
        domain_size : tuple
            模拟区域尺寸 (Lx, Ly)，单位 m
        nx, ny : int
            网格分辨率
        seed : int
            随机数种子
        """
        if nx <= 0 or ny <= 0:
            raise ValueError("nx 和 ny 必须为正整数")
        self.Lx, self.Ly = domain_size
        self.nx = nx
        self.ny = ny
        self.dx = self.Lx / nx
        self.dy = self.Ly / ny
        self.rng = MiddleSquareGenerator(seed=seed, d=4)

        # 裂隙开度场 [m]
        self.aperture = np.zeros((ny, nx))
        # 裂隙连通性矩阵 (布尔)
        self.connectivity = np.zeros((ny, nx), dtype=bool)
        # 裂隙方向角 [rad]
        self.orientation = np.zeros((ny, nx))
        # 水力传导系数 [m²/s]
        self.transmissivity = np.zeros((ny, nx))

        # 物理常数
        self.rho = 1000.0      # 水密度 [kg/m³]
        self.g = 9.81          # 重力加速度 [m/s²]
        self.mu = 1.0e-3       # 动力粘度 [Pa·s]

    def generate_ifs_fractures(self, n_iterations: int = 5000,
                                ifs_type: str = "cross") -> np.ndarray:
        """
        使用 IFS 生成分形裂隙网络点集

        基于 cross_chaos 的迭代函数系统：
            x_{k+1} = A * x_k + b_j
        
        其中 A 是压缩仿射变换矩阵，b_j 是平移向量。

        Parameters
        ----------
        n_iterations : int
            IFS 迭代次数
        ifs_type : str
            IFS 类型 ("cross" 或 "sierpinski")

        Returns
        -------
        np.ndarray
            裂隙节点坐标 (2, n_iterations)
        """
        if n_iterations <= 0:
            raise ValueError("n_iterations 必须为正")

        if ifs_type == "cross":
            # Cross IFS: 五映射系统
            A = np.array([[1.0/3.0, 0.0],
                          [0.0, 1.0/3.0]])
            b = np.array([
                [1.0/3.0, 0.0, 1.0/3.0, 2.0/3.0, 1.0/3.0],
                [0.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 2.0/3.0]
            ])
        elif ifs_type == "sierpinski":
            A = np.array([[0.5, 0.0],
                          [0.0, 0.5]])
            b = np.array([
                [0.0, 0.5, 0.25],
                [0.0, 0.0, 0.5]
            ])
        else:
            raise ValueError(f"不支持的 IFS 类型: {ifs_type}")

        n_maps = b.shape[1]
        x = np.zeros((2, n_iterations))
        x[:, 0] = [self.rng.random(), self.rng.random()]

        for i in range(1, n_iterations):
            j = int(self.rng.random() * n_maps) % n_maps
            x[:, i] = A @ x[:, i-1] + b[:, j]

        # 缩放到物理域
        x[0, :] *= self.Lx
        x[1, :] *= self.Ly
        return x

    def rasterize_fractures(self, fracture_points: np.ndarray,
                            base_aperture: float = 1.0e-4,
                            aperture_std: float = 0.3) -> np.ndarray:
        """
        将裂隙点集栅格化为网格开度场

        基于 image_double 的上采样思想，将裂隙节点映射到高分辨率网格，
        同时通过邻域影响扩大裂隙覆盖范围。

        Parameters
        ----------
        fracture_points : np.ndarray
            裂隙节点坐标 (2, N)
        base_aperture : float
            基准裂隙开度 [m]
        aperture_std : float
            开度对数正态分布标准差

        Returns
        -------
        np.ndarray
            裂隙开度场 (ny, nx)
        """
        if base_aperture <= 0:
            raise ValueError("base_aperture 必须为正")

        aperture = np.zeros((self.ny, self.nx))

        for i in range(fracture_points.shape[1]):
            ix = min(int(fracture_points[0, i] / self.dx), self.nx - 1)
            iy = min(int(fracture_points[1, i] / self.dy), self.ny - 1)
            # 对数正态分布开度
            log_b = np.log(base_aperture) + aperture_std * self.rng.randn()
            b_val = np.exp(log_b)
            b_val = max(b_val, 1.0e-6)  # 最小开度约束
            
            # 设置当前网格及邻域（3x3 窗口）以模拟裂隙段宽度
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni, nj = iy + di, ix + dj
                    if 0 <= ni < self.ny and 0 <= nj < self.nx:
                        # 距离衰减
                        dist = np.sqrt(di**2 + dj**2)
                        factor = max(0.3, 1.0 - 0.35 * dist)
                        aperture[ni, nj] = max(aperture[ni, nj], b_val * factor)

        self.aperture = aperture
        return aperture

    def compute_transmissivity(self) -> np.ndarray:
        """
        计算裂隙水力传导系数（Cubic Law）

        公式：
            T = (ρ g b³) / (12 μ)

        其中 b 为裂隙开度，ρ 为水密度，g 为重力加速度，μ 为动力粘度。

        Returns
        -------
        np.ndarray
            水力传导系数场 (ny, nx)
        """
        # TODO: 实现 Cubic Law 公式计算水力传导系数
        pass

    def update_connectivity(self, threshold: float = 1.0e-10) -> np.ndarray:
        """
        基于 lights_out_game 的网格邻域状态切换模型更新连通性

        裂隙连通性受应力场影响，采用邻域耦合模型：
            - 当某个网格单元裂隙开度超过阈值时，其四个邻域单元
              的连通状态可能被切换（力学耦合效应）

        Parameters
        ----------
        threshold : float
            最小传导系数阈值 [m²/s]

        Returns
        -------
        np.ndarray
            连通性矩阵 (ny, nx)
        """
        T = self.transmissivity
        conn = T > threshold

        # 邻域耦合：高传导单元影响邻域连通性
        new_conn = conn.copy()
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                if conn[i, j]:
                    # 影响四个邻域（类似 Lights Out 的邻域切换）
                    neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                    for ni, nj in neighbors:
                        if T[ni, nj] > threshold * 0.1:
                            new_conn[ni, nj] = True

        self.connectivity = new_conn
        return new_conn

    def check_percolation(self) -> Tuple[bool, List[Tuple[int, int]]]:
        """
        检查裂隙网络是否存在从底部到顶部的渗流路径

        使用广度优先搜索（BFS）进行渗流判定。

        Returns
        -------
        tuple
            (是否渗流, 渗流路径节点列表)
        """
        from collections import deque

        visited = np.zeros((self.ny, self.nx), dtype=bool)
        queue = deque()
        path = {}

        # 从底部边界开始搜索
        for j in range(self.nx):
            if self.connectivity[0, j]:
                queue.append((0, j))
                visited[0, j] = True
                path[(0, j)] = None

        # BFS
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        end_node = None

        while queue:
            i, j = queue.popleft()
            if i == self.ny - 1:
                end_node = (i, j)
                break
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.ny and 0 <= nj < self.nx:
                    if not visited[ni, nj] and self.connectivity[ni, nj]:
                        visited[ni, nj] = True
                        queue.append((ni, nj))
                        path[(ni, nj)] = (i, j)

        # 重构路径
        percolation_path = []
        if end_node is not None:
            node = end_node
            while node is not None:
                percolation_path.append(node)
                node = path[node]
            percolation_path.reverse()

        return end_node is not None, percolation_path

    def equivalent_permeability(self) -> float:
        """
        计算等效渗透率（Snow, 1969）

        公式：
            k_eq = (1/12) * Σ(b_i³ * cos²θ_i) / A

        Returns
        -------
        float
            等效渗透率 [m²]
        """
        if np.all(self.aperture == 0):
            return 0.0

        mask = self.aperture > 0
        b3_sum = np.sum(self.aperture[mask] ** 3)
        area = self.Lx * self.Ly
        k_eq = b3_sum / (12.0 * area)
        return k_eq

    def tortuosity(self, path: List[Tuple[int, int]]) -> float:
        """
        计算裂隙网络迂曲度

        公式：
            τ = L_actual / L_euclidean

        Parameters
        ----------
        path : list
            渗流路径节点列表

        Returns
        -------
        float
            迂曲度
        """
        if len(path) < 2:
            return 1.0

        actual_length = 0.0
        for k in range(len(path) - 1):
            i1, j1 = path[k]
            i2, j2 = path[k + 1]
            dx = (j2 - j1) * self.dx
            dy = (i2 - i1) * self.dy
            actual_length += np.sqrt(dx ** 2 + dy ** 2)

        # 欧氏距离（底部中心到顶部中心）
        euclidean_length = np.sqrt(self.Lx ** 2 + self.Ly ** 2)

        tau = actual_length / euclidean_length
        return max(tau, 1.0)

    def generate_full_network(self, n_fracture_points: int = 5000,
                              base_aperture: float = 1.0e-4) -> dict:
        """
        生成完整的裂隙网络（一键生成）

        Returns
        -------
        dict
            包含所有裂隙网络属性的字典
        """
        points = self.generate_ifs_fractures(n_fracture_points)
        self.rasterize_fractures(points, base_aperture)
        self.compute_transmissivity()
        self.update_connectivity()
        percolates, path = self.check_percolation()
        k_eq = self.equivalent_permeability()
        tau = self.tortuosity(path) if percolates else 1.0

        return {
            "aperture": self.aperture,
            "transmissivity": self.transmissivity,
            "connectivity": self.connectivity,
            "percolates": percolates,
            "path": path,
            "equivalent_permeability": k_eq,
            "tortuosity": tau,
            "porosity": np.mean(self.connectivity),
            "domain_size": (self.Lx, self.Ly),
            "resolution": (self.nx, self.ny)
        }
