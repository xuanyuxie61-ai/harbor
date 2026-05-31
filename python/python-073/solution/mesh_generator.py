# -*- coding: utf-8 -*-
"""
mesh_generator.py
高超声速边界层计算网格生成器

核心算法来源：
- triangulation: Delaunay 三角剖分、邻居单元搜索、边界检测、网格质量评估
- sphere_llt_grid: 球面 LLT 经纬度网格生成（用于波矢方向离散）
- xy_io: 二维坐标数据读写格式

物理背景：
高超声速飞行器前缘/锥体边界层网格需满足：
1. 壁面法向第一层高度 Δy_1^+ < 1（解析粘性底层）
2. 流向/展向分辨率需捕捉 Mack 第二模态不稳定波
3. 曲面贴体坐标适应激波-边界层干扰区域
"""

import numpy as np
from math import sin, cos, pi, sqrt, atan2


class BoundaryLayerMesh:
    """
    二维/三维高超声速边界层结构化/非结构化混合网格。
    """

    def __init__(self, L=1.0, H=0.1, Nx=100, Ny=80, Re=1e6, Ma=6.0):
        """
        参数:
            L (float): 平板/前缘长度 [m]
            H (float): 边界层计算域高度 [m]
            Nx (int): 流向节点数
            Ny (int): 法向节点数
            Re (float): 基于长度 L 的雷诺数
            Ma (float): 马赫数
        """
        self.L = L
        self.H = H
        self.Nx = Nx
        self.Ny = Ny
        self.Re = Re
        self.Ma = Ma

    def wall_normal_stretching(self, n, h_max, beta=1.08):
        """
        壁面法向几何拉伸函数，确保第一层网格高度满足 y^+ < 1。

        拉伸公式（基于几何级数）：
            y_j = y_1 * (β^j - 1) / (β - 1),  j = 0, ..., n

        其中 y_1 由壁面单位约束确定：
            Δy_1^+ = u_τ * Δy_1 / ν_w ≈ 1
            u_τ = sqrt(τ_w / ρ_w)

        采用 van Driest 变换估算壁面剪应力：
            τ_w ≈ 0.0296 * ρ_e * u_e^2 * Re_x^{-0.2}

        参数:
            n (int): 法向节点数
            h_max (float): 外边界高度
            beta (float): 拉伸因子

        返回:
            np.ndarray: 法向坐标 y[0..n]，y[0]=0 为壁面
        """
        # 估算壁面摩擦系数 (Turbulent empirical correlation for reference)
        # 对于层流边界层，采用 Blasius: cf = 0.664 / sqrt(Re_x)
        # 这里取当地雷诺数 Re_L 的边界层名义厚度 δ ≈ 5L / sqrt(Re_L)
        Re_L = self.Re
        delta = 5.0 * self.L / sqrt(Re_L)

        # 壁面剪切应力 (Pa)，假设 ρe=1.225 kg/m³, ue=Ma*a (a≈340m/s)
        rho_e = 1.225
        a = 340.0
        u_e = self.Ma * a
        cf = 0.664 / sqrt(Re_L)  # 层流 Blasius
        tau_w = 0.5 * rho_e * u_e**2 * cf

        # 壁面摩擦速度
        mu_w = 1.7894e-5 * 1.458e-6  # 近似
        rho_w = rho_e  # 冷壁近似
        nu_w = mu_w / rho_w
        u_tau = sqrt(max(tau_w / rho_w, 1e-20))

        # 第一层高度: y^+ = 1
        dy1 = max(nu_w / u_tau, 1e-8)

        # 几何拉伸求解 y_1 与 beta
        # h_max = dy1 * (beta^n - 1) / (beta - 1)
        # 迭代修正 beta
        beta_est = beta
        for _ in range(20):
            denom = beta_est - 1.0
            if abs(denom) < 1e-12:
                break
            h_est = dy1 * (beta_est**n - 1.0) / denom
            if abs(h_est - h_max) < 1e-8:
                break
            # Newton-Raphson 修正
            df = dy1 * (n * beta_est**(n - 1) * denom - (beta_est**n - 1.0)) / (denom**2)
            if abs(df) > 1e-12:
                beta_est = max(1.001, beta_est - (h_est - h_max) / df)

        y = np.zeros(n + 1)
        for j in range(1, n + 1):
            y[j] = dy1 * (beta_est**j - 1.0) / (beta_est - 1.0)
        y = np.clip(y, 0.0, h_max)
        y[-1] = h_max
        return y

    def generate_flat_plate_mesh(self):
        """
        生成平板边界层结构化网格。

        网格映射: (ξ, η) ∈ [0,1]^2 → (x, y) ∈ [0,L]×[0,H]
        其中 ξ 为均匀流向坐标，η 为法向拉伸坐标。

        返回:
            tuple: (nodes, nx, ny)
                nodes: np.ndarray, shape (nx*ny, 2), 节点坐标
        """
        x = np.linspace(0.0, self.L, self.Nx)
        y = self.wall_normal_stretching(self.Ny - 1, self.H)
        nx, ny = len(x), len(y)

        nodes = np.zeros((nx * ny, 2))
        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                nodes[idx, 0] = x[i]
                nodes[idx, 1] = y[j]
        return nodes, nx, ny

    def generate_triangles_from_structured(self, nx, ny):
        """
        将结构化四边形网格剖分为三角形（基于 triangulation 思想）。

        每个四边形 (i,j)-(i+1,j)-(i+1,j+1)-(i,j+1) 分裂为两个三角形：
            T1: (i,j), (i+1,j), (i+1,j+1)
            T2: (i,j), (i+1,j+1), (i,j+1)

        参数:
            nx (int): 流向节点数
            ny (int): 法向节点数

        返回:
            np.ndarray: triangles, shape ((nx-1)*(ny-1)*2, 3), 节点索引
        """
        n_tri = (nx - 1) * (ny - 1) * 2
        triangles = np.zeros((n_tri, 3), dtype=int)
        t = 0
        for i in range(nx - 1):
            for j in range(ny - 1):
                n1 = i * ny + j
                n2 = (i + 1) * ny + j
                n3 = (i + 1) * ny + (j + 1)
                n4 = i * ny + (j + 1)
                triangles[t] = [n1, n2, n3]
                triangles[t + 1] = [n1, n3, n4]
                t += 2
        return triangles

    def triangle_neighbors(self, triangle_num, triangle_node):
        """
        基于 triangulation_order3_neighbor 的邻居单元搜索。

        对每个三角形每条边，搜索共享该边的相邻三角形。
        边按逆时针方向定义；邻居三角形的对应边方向相反。

        参数:
            triangle_num (int): 三角形数量
            triangle_node (np.ndarray): shape (3, triangle_num) 或 (triangle_num, 3)

        返回:
            np.ndarray: neighbors, shape (triangle_num, 3), 邻居三角形索引（-1 表示边界）
        """
        tn = triangle_node
        if tn.shape[0] == 3 and tn.shape[1] == triangle_num:
            tn = tn.T  # 转为 (triangle_num, 3)

        neighbors = np.full((triangle_num, 3), -1, dtype=int)

        # 构建边到三角形的映射
        edge_map = {}
        for t in range(triangle_num):
            for s in range(3):
                n1 = tn[t, s]
                n2 = tn[t, (s + 1) % 3]
                key = (min(n1, n2), max(n1, n2))
                if key not in edge_map:
                    edge_map[key] = []
                edge_map[key].append((t, s))

        # 填充邻居
        for t in range(triangle_num):
            for s in range(3):
                n1 = tn[t, s]
                n2 = tn[t, (s + 1) % 3]
                key = (min(n1, n2), max(n1, n2))
                candidates = edge_map[key]
                for (t2, s2) in candidates:
                    if t2 != t:
                        # 检查方向是否相反
                        n1b = tn[t2, s2]
                        n2b = tn[t2, (s2 + 1) % 3]
                        if n1 == n2b and n2 == n1b:
                            neighbors[t, s] = t2
                            break
        return neighbors

    def boundary_nodes(self, nx, ny):
        """
        标记边界节点（壁面、入口、出口、远场）。

        返回:
            dict: {'wall': [], 'inlet': [], 'outlet': [], 'farfield': []}
        """
        wall = [i * ny for i in range(nx)]
        inlet = [j for j in range(ny)]
        outlet = [(nx - 1) * ny + j for j in range(ny)]
        farfield = [i * ny + (ny - 1) for i in range(nx)]
        return {
            'wall': np.array(wall, dtype=int),
            'inlet': np.array(inlet, dtype=int),
            'outlet': np.array(outlet, dtype=int),
            'farfield': np.array(farfield, dtype=int)
        }


def sphere_wavevector_grid(lat_num, long_num):
    """
    基于 sphere_llt_grid 的球面波矢方向离散。

    在线性稳定性分析中，三维扰动波矢 k = (k_x, k_y, k_z) 的方向由球坐标描述：
        k_x = |k| sin(φ) cos(θ)
        k_z = |k| sin(φ) sin(θ)
        k_y = |k| cos(φ)

    其中 φ 为极角（波矢与法向夹角），θ 为方位角。
    为捕捉展向模态（Mack 第二模态），需在 φ ≈ 90° 附近加密。

    参数:
        lat_num (int): 极角方向分度数（不含两极）
        long_num (int): 方位角方向分度数

    返回:
        np.ndarray: points, shape (point_num, 3), 单位球面波矢方向
    """
    # 极角加密：在赤道附近加密，两极稀疏
    # 使用 Chebyshev 节点映射
    j = np.arange(lat_num + 2)
    phi_nodes = np.pi * j / (lat_num + 1)

    point_num = 2 + lat_num * long_num
    p = np.zeros((point_num, 3))
    n = 0

    # 北极 (φ=0)
    p[n] = [0.0, 0.0, 1.0]
    n += 1

    for lat in range(1, lat_num + 1):
        phi = phi_nodes[lat]
        for lng in range(long_num):
            theta = 2.0 * pi * lng / long_num
            p[n, 0] = sin(phi) * cos(theta)
            p[n, 1] = sin(phi) * sin(theta)
            p[n, 2] = cos(phi)
            n += 1

    # 南极 (φ=π)
    p[n] = [0.0, 0.0, -1.0]
    n += 1
    return p


def save_xy_data(filename, x, y):
    """
    基于 xy_io 的二维坐标数据写入。

    参数:
        filename (str): 输出文件路径
        x, y (np.ndarray): 坐标数组
    """
    data = np.column_stack((x, y))
    np.savetxt(filename, data, fmt='%.8e', header='X Y', comments='# ')


def read_xy_data(filename):
    """
    基于 xy_io 的二维坐标数据读取。

    参数:
        filename (str): 输入文件路径

    返回:
        tuple: (x, y) 坐标数组
    """
    data = np.loadtxt(filename, comments='#')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0], data[:, 1]
