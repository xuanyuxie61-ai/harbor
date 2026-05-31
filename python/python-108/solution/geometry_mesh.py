# -*- coding: utf-8 -*-
"""
geometry_mesh.py
微腔几何建模与非结构化网格生成模块

核心公式与物理背景
------------------
1. 微环谐振腔参数化边界
   微环腔体由中心圆环与耦合波导组成。外边界参数方程：
   x(t) = (R + r·cos(θ))·cos(φ),  y(t) = (R + r·cos(θ))·sin(φ)
   其中 R 为主半径，r 为截面半径。

2. 边界弧长微元
   ds = √( (dx/dt)² + (dy/dt)² ) dt
   梯形法则离散：s ≈ h·[ 0.5·f₀ + Σfᵢ + 0.5·fₙ ]

3. CVV (Cell-based Vector of Vectors) 数据结构
   用于存储变长行数据（如每个单元节点数不同）：
   offset[i] = Σ_{k=0}^{i-1} nrow[k]
   元素访问：A[i][j] ↔ flat[ offset[i] + j ]

融合来源
--------
- 1322_triangle_to_xml : 三角网格索引转换与拓扑管理
- 376_fem_io           : FEM节点/单元数据读写规范
- 147_cell             : CVV变长向量存储结构
- 016_arclength        : 参数化曲线弧长积分
"""

import numpy as np
from typing import List, Tuple, Optional


class CVV:
    """
    Cell-based Vector of Vectors (CVV) 数据结构。
    将变长的二维不规则数组压缩为一维 flat 数组，通过偏移表实现 O(1) 索引。
    """

    def __init__(self, row_lengths: List[int], dtype=float):
        """
        参数
        ----
        row_lengths : List[int]
            每一行的列数（长度可以不同）
        dtype : type
            元素数值类型
        """
        if not row_lengths:
            raise ValueError("row_lengths 不能为空")
        if any(n < 0 for n in row_lengths):
            raise ValueError("行长度必须非负")
        self._nr = np.array(row_lengths, dtype=int)
        self._roff = np.zeros(len(row_lengths) + 1, dtype=int)
        for i in range(len(row_lengths)):
            self._roff[i + 1] = self._roff[i] + self._nr[i]
        self._flat = np.zeros(self._roff[-1], dtype=dtype)

    # ---------- 核心索引操作 ----------
    def iget(self, i: int, j: int):
        """获取 A[i][j]"""
        if not (0 <= i < len(self._nr)):
            raise IndexError(f"行索引 {i} 越界，有效范围 [0, {len(self._nr)})")
        if not (0 <= j < self._nr[i]):
            raise IndexError(f"列索引 {j} 越界，第 {i} 行长度为 {self._nr[i]}")
        return self._flat[self._roff[i] + j]

    def iset(self, i: int, j: int, value):
        """设置 A[i][j] = value"""
        if not (0 <= i < len(self._nr)):
            raise IndexError(f"行索引 {i} 越界")
        if not (0 <= j < self._nr[i]):
            raise IndexError(f"列索引 {j} 越界，第 {i} 行长度为 {self._nr[i]}")
        self._flat[self._roff[i] + j] = value

    def iinc(self, i: int, j: int, delta):
        """增量更新 A[i][j] += delta"""
        self.iset(i, j, self.iget(i, j) + delta)

    def nget(self, i: int) -> int:
        """获取第 i 行的长度"""
        return self._nr[i]

    def nset(self, i: int, value: int):
        """设置第 i 行长度（要求扁平数组有足够空间，这里只做校验）"""
        if value != self._nr[i]:
            raise NotImplementedError("CVV 不支持运行时动态改变行长度")

    def size(self) -> Tuple[int, int]:
        """返回 (行数, 元素总数)"""
        return len(self._nr), self._roff[-1]

    def get_row(self, i: int) -> np.ndarray:
        """提取第 i 行作为一维数组副本"""
        return self._flat[self._roff[i]:self._roff[i] + self._nr[i]].copy()

    def set_row(self, i: int, arr: np.ndarray):
        """用一维数组 arr 填充第 i 行（长度必须匹配）"""
        if len(arr) != self._nr[i]:
            raise ValueError(f"输入长度 {len(arr)} 与行长度 {self._nr[i]} 不匹配")
        self._flat[self._roff[i]:self._roff[i] + self._nr[i]] = arr

    @property
    def flat(self) -> np.ndarray:
        return self._flat

    @property
    def offsets(self) -> np.ndarray:
        return self._roff


class MicrocavityGeometry:
    """
    微环谐振腔几何建模器。
    支持生成截面网格、标记边界、计算弧长，并提供 FEM 标准数据接口。
    """

    def __init__(self,
                 R_major: float = 10.0e-6,      # 主半径 [m]
                 r_minor: float = 1.5e-6,       # 截面半径 [m]
                 waveguide_gap: float = 0.3e-6,  # 耦合波导间距 [m]
                 waveguide_width: float = 0.5e-6,  # 波导宽度 [m]
                 n_ring: float = 3.47,          # 环芯折射率 (Si)
                 n_clad: float = 1.44,          # 包层折射率 (SiO2)
                 n_env: float = 1.00):          # 环境折射率
        self.R_major = R_major
        self.r_minor = r_minor
        self.waveguide_gap = waveguide_gap
        self.waveguide_width = waveguide_width
        self.n_ring = n_ring
        self.n_clad = n_clad
        self.n_env = n_env

    def ring_boundary_parametric(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        微环截面外边界参数方程 (t ∈ [0, 2π])。
        这里对截面圆做参数化：
            x(θ) = R_major + r_minor·cos(θ)
            y(θ) = r_minor·sin(θ)
        返回 (x, y) 数组。
        """
        t = np.asarray(t)
        x = self.R_major + self.r_minor * np.cos(t)
        y = self.r_minor * np.sin(t)
        return x, y

    def arc_length_trapezoidal(self, t1: float = 0.0, t2: float = 2 * np.pi, n: int = 1000) -> float:
        """
        用复合梯形法则计算参数曲线弧长：
            s = ∫_{t1}^{t2} √( (dx/dt)² + (dy/dt)² ) dt

        参数
        ----
        t1, t2 : float
            参数区间
        n : int
            均匀分段数，须 ≥ 1

        返回
        ----
        s : float
            弧长估计值
        """
        if n < 1:
            raise ValueError("n 必须 ≥ 1")
        h = (t2 - t1) / n
        # 导数解析式
        def dxdt(t):
            return -self.r_minor * np.sin(t)
        def dydt(t):
            return self.r_minor * np.cos(t)

        t_vals = np.linspace(t1, t2, n + 1)
        f_vals = np.sqrt(dxdt(t_vals) ** 2 + dydt(t_vals) ** 2)
        s = h * (0.5 * f_vals[0] + np.sum(f_vals[1:-1]) + 0.5 * f_vals[-1])
        return s

    def generate_cross_section_mesh(self,
                                    nr: int = 80,
                                    n_theta: int = 120,
                                    return_boundary_markers: bool = True):
        """
        在微环截面上生成结构化伪三角网格（极坐标映射到笛卡尔坐标）。
        返回节点坐标 (nodes)、单元连通性 (elements)、边界标记 (markers)。

        公式
        ----
        节点在极坐标 (ρ, θ) 下生成，映射到笛卡尔：
            x = R_major + ρ·cos(θ)
            y = ρ·sin(θ)
        其中 ρ ∈ [0, r_minor]，θ ∈ [0, 2π)。
        """
        if nr < 2 or n_theta < 3:
            raise ValueError("网格分辨率不足：nr≥2, n_theta≥3")

        rho = np.linspace(0.0, self.r_minor, nr)
        theta = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)

        # 生成节点：按 (rho, theta) 顺序铺展
        nodes = []
        for ri in rho:
            for th in theta:
                x = self.R_major + ri * np.cos(th)
                y = ri * np.sin(th)
                nodes.append([x, y])
        nodes = np.array(nodes, dtype=float)
        n_nodes = len(nodes)

        # 生成三角单元：每个四边形剖分为2个三角形
        elements = []
        for i in range(nr - 1):
            for j in range(n_theta):
                j_next = (j + 1) % n_theta
                n0 = i * n_theta + j
                n1 = i * n_theta + j_next
                n2 = (i + 1) * n_theta + j_next
                n3 = (i + 1) * n_theta + j
                elements.append([n0, n1, n2])
                elements.append([n0, n2, n3])
        elements = np.array(elements, dtype=int)

        # 边界标记：0=内部, 1=外边界, 2=内边界(中心)
        markers = np.zeros(n_nodes, dtype=int)
        for idx in range(n_nodes):
            rho_idx = idx // n_theta
            if rho_idx == nr - 1:
                markers[idx] = 1   # 外边界
            elif rho_idx == 0:
                markers[idx] = 2   # 中心点

        if return_boundary_markers:
            return nodes, elements, markers
        return nodes, elements

    def generate_waveguide_nodes(self, n_points: int = 50) -> np.ndarray:
        """
        生成直波导的节点坐标（位于微环外侧，用于计算耦合）。
        波导中心线位于 x = R_major + r_minor + gap + w/2，y ∈ [-L/2, L/2]。
        """
        center_x = self.R_major + self.r_minor + self.waveguide_gap + 0.5 * self.waveguide_width
        L = 4.0 * self.r_minor
        y = np.linspace(-L / 2, L / 2, n_points)
        x = np.full_like(y, center_x)
        return np.column_stack((x, y))

    def compute_mesh_quality(self, nodes: np.ndarray, elements: np.ndarray) -> dict:
        """
        计算网格质量指标：
            - 最小角、最大角
            - 面积比
        """
        angles_min = []
        angles_max = []
        areas = []
        for tri in elements:
            p0, p1, p2 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
            # 边向量
            v0 = p1 - p0
            v1 = p2 - p0
            v2 = p2 - p1
            # 边长
            a = np.linalg.norm(v1)
            b = np.linalg.norm(v2)
            c = np.linalg.norm(v0)
            # 余弦定理求角
            def angle_from_sides(aa, bb, cc):
                return np.arccos(np.clip((aa * aa + bb * bb - cc * cc) / (2 * aa * bb + 1e-30), -1.0, 1.0))
            A0 = angle_from_sides(a, c, b)
            A1 = angle_from_sides(b, c, a)
            A2 = np.pi - A0 - A1
            angles = np.array([A0, A1, A2]) * 180.0 / np.pi
            angles_min.append(np.min(angles))
            angles_max.append(np.max(angles))
            # 面积（叉积一半）
            area = 0.5 * abs(np.cross(v0, v1))
            areas.append(area)

        areas = np.array(areas)
        return {
            "min_angle_deg": np.min(angles_min),
            "max_angle_deg": np.max(angles_max),
            "mean_area": np.mean(areas),
            "min_area": np.min(areas),
            "max_area": np.max(areas),
            "total_area": np.sum(areas),
        }

    def fem_write_nodes(self, nodes: np.ndarray, filename: str):
        """将节点坐标写入标准 FEM 文本文件"""
        with open(filename, "w") as f:
            f.write(f"# FEM node coordinates\n")
            f.write(f"{nodes.shape[0]} {nodes.shape[1]}\n")
            for row in nodes:
                f.write(" ".join(f"{v:.16e}" for v in row) + "\n")

    def fem_write_elements(self, elements: np.ndarray, filename: str):
        """将单元连通性写入标准 FEM 文本文件"""
        with open(filename, "w") as f:
            f.write(f"# FEM element connectivity (0-based)\n")
            f.write(f"{elements.shape[0]} {elements.shape[1]}\n")
            for row in elements:
                f.write(" ".join(str(v) for v in row) + "\n")

    def fem_read_nodes(self, filename: str) -> np.ndarray:
        """从标准 FEM 文本文件读取节点坐标"""
        with open(filename, "r") as f:
            lines = f.readlines()
        # 跳过注释行
        data_lines = [l for l in lines if not l.strip().startswith("#")]
        header = data_lines[0].strip().split()
        n_rows, n_cols = int(header[0]), int(header[1])
        data = []
        for line in data_lines[1:]:
            parts = line.strip().split()
            if len(parts) >= n_cols:
                data.append([float(v) for v in parts[:n_cols]])
        arr = np.array(data, dtype=float)
        if arr.shape[0] != n_rows or arr.shape[1] != n_cols:
            raise ValueError(f"文件声明维度 ({n_rows},{n_cols}) 与实际 ({arr.shape}) 不符")
        return arr

    def fem_read_elements(self, filename: str) -> np.ndarray:
        """从标准 FEM 文本文件读取单元连通性"""
        with open(filename, "r") as f:
            lines = f.readlines()
        data_lines = [l for l in lines if not l.strip().startswith("#")]
        header = data_lines[0].strip().split()
        n_rows, n_cols = int(header[0]), int(header[1])
        data = []
        for line in data_lines[1:]:
            parts = line.strip().split()
            if len(parts) >= n_cols:
                data.append([int(v) for v in parts[:n_cols]])
        arr = np.array(data, dtype=int)
        if arr.shape[0] != n_rows or arr.shape[1] != n_cols:
            raise ValueError(f"文件声明维度 ({n_rows},{n_cols}) 与实际 ({arr.shape}) 不符")
        return arr
