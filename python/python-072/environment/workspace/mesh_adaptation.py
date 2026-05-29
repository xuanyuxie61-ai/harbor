"""
mesh_adaptation.py
==================
自适应网格细化模块

融合种子项目：
- 156_change_dynamic: 动态规划思想用于最优网格节点选择
- 1358_trinity: 三角形网格结构与拓扑操作

核心内容：
1. 基于相场梯度信息的自适应网格标记
2. 动态规划优化网格节点分布
3. 界面附近局部网格加密策略
4. 网格质量评估指标

自适应网格细化（AMR）准则：
    对于单元 K，若误差估计 η_K > θ * max(η)，则标记细化。

    基于相场的误差指示器：
        η_K = ||∇φ||_K * h_K

    其中 h_K 为单元尺寸，||∇φ||_K 为单元上 φ 的梯度范数。

动态规划网格优化：
    给定总网格数 N，将 N 个网格点分配到不同区域，使得
    全局误差最小：
        min Σ_i e_i(n_i)
        s.t. Σ_i n_i = N
    
    其中 e_i(n_i) 为区域 i 使用 n_i 个网格点的误差。
    该问题可用动态规划求解。
"""

import numpy as np


class MeshAdaptation:
    """
    基于相场的自适应网格管理器。
    """

    def __init__(self, nx, ny, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0):
        """
        初始化网格参数。

        Parameters
        ----------
        nx, ny : int
            基础网格点数。
        x_min, x_max : float
            x 方向范围。
        y_min, y_max : float
            y 方向范围。
        """
        self.nx = nx
        self.ny = ny
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.dx_base = (x_max - x_min) / (nx - 1)
        self.dy_base = (y_max - y_min) / (ny - 1)

    def compute_error_indicator(self, phi):
        """
        基于相场梯度的单元误差指示器。

        η_ij = |∇φ|_ij * h_ij

        Parameters
        ----------
        phi : ndarray, shape (nx, ny)
            序参量场。

        Returns
        -------
        ndarray
            每个网格点的误差指示器。
        """
        nx, ny = phi.shape
        dx = (self.x_max - self.x_min) / (nx - 1)
        dy = (self.y_max - self.y_min) / (ny - 1)

        grad_x = np.zeros_like(phi)
        grad_y = np.zeros_like(phi)

        grad_x[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * dx)
        grad_y[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * dy)

        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        h_local = np.sqrt(dx ** 2 + dy ** 2)

        return grad_mag * h_local

    def mark_refinement(self, phi, threshold_ratio=0.5):
        """
        标记需要细化的网格区域。

        标记准则：η > threshold_ratio * max(η)

        Parameters
        ----------
        phi : ndarray
            序参量场。
        threshold_ratio : float
            阈值比例。

        Returns
        -------
        ndarray
            布尔标记数组，True 表示需要细化。
        """
        eta = self.compute_error_indicator(phi)
        max_eta = np.max(eta)
        if max_eta < 1e-14:
            return np.zeros_like(phi, dtype=bool)

        threshold = threshold_ratio * max_eta
        return eta > threshold

    def dynamic_programming_mesh_distribution(self, n_total, error_funcs, regions):
        """
        使用动态规划优化网格点在多个区域间的分布。

        问题：将 n_total 个网格点分配到 regions 个区域，
        最小化总误差：
            min Σ_i error_funcs[i](n_i)
            s.t. Σ_i n_i = n_total, n_i ≥ n_min

        基于种子项目 156_change_dynamic 的动态规划思想。

        Parameters
        ----------
        n_total : int
            总网格点数。
        error_funcs : list of callable
            每个区域的误差函数 error_func(n) 返回使用 n 个点的误差。
        regions : int
            区域数。

        Returns
        -------
        list
            每个区域的最优网格点数。
        """
        n_min = 2  # 每个区域最少点数

        # dp[i][j] = 前 i 个区域使用 j 个点的最小误差
        dp = np.full((regions + 1, n_total + 1), np.inf)
        dp[0, 0] = 0.0

        # 记录决策
        decision = np.zeros((regions + 1, n_total + 1), dtype=int)

        for i in range(1, regions + 1):
            for j in range(n_total + 1):
                for k in range(n_min, j + 1):
                    err = error_funcs[i - 1](k)
                    if dp[i - 1, j - k] + err < dp[i, j]:
                        dp[i, j] = dp[i - 1, j - k] + err
                        decision[i, j] = k

        # 回溯求解
        distribution = []
        remaining = n_total
        for i in range(regions, 0, -1):
            k = decision[i, remaining]
            distribution.append(k)
            remaining -= k

        distribution.reverse()
        return distribution

    def interface_focused_mesh(self, phi, refinement_level=2):
        """
        生成以界面为中心的非均匀网格坐标。

        在界面附近（|φ| < 0.8）加密网格，远离界面保持粗网格。

        Parameters
        ----------
        phi : ndarray
            序参量场。
        refinement_level : int
            加密级别。

        Returns
        -------
        tuple
            (x_coords, y_coords) 非均匀坐标数组。
        """
        nx, ny = phi.shape

        # 标记界面区域
        interface_mask = np.abs(phi) < 0.8

        # 生成基础均匀坐标
        x_uniform = np.linspace(self.x_min, self.x_max, nx)
        y_uniform = np.linspace(self.y_min, self.y_max, ny)

        # 计算界面位置（phi=0 的大致 x 和 y 范围）
        x_interface_indices = np.where(np.any(interface_mask, axis=1))[0]
        y_interface_indices = np.where(np.any(interface_mask, axis=0))[0]

        if len(x_interface_indices) == 0:
            return x_uniform, y_uniform

        x_min_int = x_uniform[x_interface_indices[0]]
        x_max_int = x_uniform[x_interface_indices[-1]]
        y_min_int = y_uniform[y_interface_indices[0]]
        y_max_int = y_uniform[y_interface_indices[-1]]

        # 在界面附近加密
        x_coords = []
        for x in x_uniform:
            if x_min_int <= x <= x_max_int:
                # 在界面区域插入额外点
                x_coords.append(x)
                for r in range(1, refinement_level):
                    dx_fine = self.dx_base / (2 ** r)
                    x_coords.append(x + dx_fine)
            else:
                x_coords.append(x)

        y_coords = []
        for y in y_uniform:
            if y_min_int <= y <= y_max_int:
                y_coords.append(y)
                for r in range(1, refinement_level):
                    dy_fine = self.dy_base / (2 ** r)
                    y_coords.append(y + dy_fine)
            else:
                y_coords.append(y)

        return np.array(sorted(set(np.round(x_coords, 10)))), \
               np.array(sorted(set(np.round(y_coords, 10))))

    def compute_mesh_quality(self, phi):
        """
        计算网格质量指标。

        指标 1：界面分辨率
            Q_res = min(h_interface) / ε

        指标 2：梯度适配度
            Q_grad = mean(|∇φ| * h) / max(|∇φ| * h)

        Parameters
        ----------
        phi : ndarray
            序参量场。

        Returns
        -------
        dict
            质量指标字典。
        """
        nx, ny = phi.shape
        dx = (self.x_max - self.x_min) / (nx - 1)
        dy = (self.y_max - self.y_min) / (ny - 1)

        # 界面区域
        interface_mask = np.abs(phi) < 0.8

        # 界面分辨率
        h_interface = np.sqrt(dx ** 2 + dy ** 2)
        if np.any(interface_mask):
            # 若支持变网格，这里可用局部 h
            q_res = h_interface / 0.01  # epsilon=0.01 时
        else:
            q_res = np.inf

        # 梯度适配度
        eta = self.compute_error_indicator(phi)
        max_eta = np.max(eta)
        if max_eta > 1e-14:
            q_grad = np.mean(eta) / max_eta
        else:
            q_grad = 1.0

        return {
            'interface_resolution': q_res,
            'gradient_adaptivity': q_grad,
            'max_error_indicator': max_eta
        }


class TriangleGridTopology:
    """
    三角形网格拓扑操作。
    基于种子项目 1358_trinity 的三角形网格思想。

    将四边形网格对角线划分得到三角形网格，
    用于界面附近的局部有限元计算。
    """

    @staticmethod
    def quadrilateral_to_triangles(nx, ny):
        """
        将 nx×ny 的笛卡尔网格划分为三角形单元。

        每个四边形单元 (i,j)-(i+1,j)-(i+1,j+1)-(i,j+1) 分为两个三角形：
            T1: (i,j), (i+1,j), (i,j+1)
            T2: (i+1,j), (i+1,j+1), (i,j+1)

        Parameters
        ----------
        nx, ny : int
            网格点数。

        Returns
        -------
        tuple
            (vertices, triangles) 顶点和三角形连接表。
        """
        # 顶点坐标
        vertices = []
        for i in range(nx):
            for j in range(ny):
                vertices.append((i, j))
        vertices = np.array(vertices)

        # 三角形连接
        triangles = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                v00 = i * ny + j
                v10 = (i + 1) * ny + j
                v01 = i * ny + (j + 1)
                v11 = (i + 1) * ny + (j + 1)

                triangles.append([v00, v10, v01])
                triangles.append([v10, v11, v01])

        return vertices, np.array(triangles)

    @staticmethod
    def triangle_quality(p1, p2, p3):
        """
        计算三角形质量指标（内切圆半径 / 外接圆半径）。

        质量 = 2r / R = (b+c-a)(c+a-b)(a+b-c) / (abc)

        等边三角形质量 = 1.0，退化三角形质量 → 0。

        Parameters
        ----------
        p1, p2, p3 : ndarray
            三角形三个顶点。

        Returns
        -------
        float
            三角形质量 [0, 1]。
        """
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)

        if a * b * c < 1e-14:
            return 0.0

        quality = (b + c - a) * (c + a - b) * (a + b - c) / (a * b * c)
        return quality
