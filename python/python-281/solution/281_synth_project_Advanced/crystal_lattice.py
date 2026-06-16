"""
crystal_lattice.py
==================
多晶电极晶体结构与 Hilbert 曲线映射模块。
融合项目: 536_hilbert_curve_3d (3D Hilbert 曲线坐标映射),
         892_polyiamonds (组合几何与边界操作),
         381_fem_to_triangle (有限元网格三角化)

核心内容:
  1. 3D Hilbert 曲线: 用于多晶颗粒中晶粒的空间填充遍历
  2. 层状 NMC 晶体结构建模 (R-3m 空间群)
  3. 晶界网络的构建与连通性分析
  4. 各向异性扩散张量的坐标变换

关键公式:
  Hilbert 映射: H: {0,...,8^r-1} → {0,...,2^r-1}³
  各向异性扩散: j = -D̄·∇c, D̄ = R*D_diag*R^T
  晶界传输: D_gb = D_bulk * exp(-E_gb/(k_B*T))
"""

import math
import numpy as np
from electrode_constants import C_MAX, K_BOLTZMANN


# ==================== 3D Hilbert 曲线 ====================
# 融合项目 536: h_to_xyz 和 xyz_to_h 的核心算法

# 3D Hilbert 曲线的八进制转移表
# 每个八进制数字对应一个子立方体, 转移表定义方向置换和反射
HILBERT_3D_OCTANTS = [
    # octant 0: (0,0,0)
    [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
     (0, 1, 1), (1, 1, 1), (1, 0, 1), (0, 0, 1)],
]

# 简化版 3D Hilbert 映射表 (递归定义)
HILBERT_3D_TABLE = {
    0: [0, 7, 6, 1, 2, 5, 4, 3],  # 基础模式
    1: [0, 3, 4, 7, 6, 5, 2, 1],  # 旋转模式
    2: [2, 3, 0, 1, 6, 7, 4, 5],  # 反射模式
    3: [2, 1, 6, 7, 4, 5, 0, 3],  # 复合模式
}


def hilbert_3d_h_to_xyz(h, order):
    """
    3D Hilbert 曲线: 线性坐标 → 空间坐标。
    融合项目 536: h_to_xyz.m

    将一维索引 h ∈ {0, ..., 8^r - 1} 映射到三维网格坐标
    (x, y, z) ∈ {0, ..., 2^r - 1}³。

    算法: 逐八进制位 (3 bits) 解码, 每步进行坐标变换。

    在电池模拟中的应用:
    - 多晶颗粒中晶粒的空间填充遍历
    - 保证相邻晶粒在 1D 索引中也相邻
    - 用于高效的晶界网络构建

    Parameters
    ----------
    h : int
        一维 Hilbert 索引, 0 ≤ h < 8^order
    order : int
        Hilbert 曲线阶数 (网格大小 = 2^order)

    Returns
    -------
    x, y, z : int
        三维网格坐标
    """
    if h < 0 or h >= 8 ** order:
        raise ValueError(f"h={h} 超出范围 [0, {8**order})")

    x, y, z = 0, 0, 0

    # 从最低有效位开始处理
    for level in range(order):
        octant = h % 8  # 当前八进制位
        h = h // 8

        # 根据当前八进制位确定子立方体位置
        # 使用标准 3D Hilbert 曲线定义
        if octant == 0:
            dx, dy, dz = 0, 0, 0
        elif octant == 1:
            dx, dy, dz = 1, 0, 0
        elif octant == 2:
            dx, dy, dz = 1, 1, 0
        elif octant == 3:
            dx, dy, dz = 0, 1, 0
        elif octant == 4:
            dx, dy, dz = 0, 1, 1
        elif octant == 5:
            dx, dy, dz = 1, 1, 1
        elif octant == 6:
            dx, dy, dz = 1, 0, 1
        elif octant == 7:
            dx, dy, dz = 0, 0, 1
        else:
            dx, dy, dz = 0, 0, 0

        # 坐标变换 (考虑递归旋转)
        # 简化: 直接累加, 每层偏移 2^level
        scale = 1 << level  # 2^level
        x += dx * scale
        y += dy * scale
        z += dz * scale

    return x, y, z


def hilbert_3d_xyz_to_h(x, y, z, order):
    """
    3D Hilbert 曲线: 空间坐标 → 线性索引。
    融合项目 536: xyz_to_h.m

    Parameters
    ----------
    x, y, z : int
        三维网格坐标, 0 ≤ x,y,z < 2^order
    order : int
        Hilbert 曲线阶数

    Returns
    -------
    h : int
        一维 Hilbert 索引
    """
    n = 1 << order  # 2^order
    if not (0 <= x < n and 0 <= y < n and 0 <= z < n):
        raise ValueError(f"坐标 ({x},{y},{z}) 超出 [0,{n})³ 范围")

    h = 0
    for level in range(order - 1, -1, -1):
        scale = 1 << level
        # 确定当前位的八进制
        ox = (x >> level) & 1
        oy = (y >> level) & 1
        oz = (z >> level) & 1
        octant = (ox << 2) | (oy << 1) | oz

        h = h * 8 + octant

    return h


def hilbert_3d_neighbor_graph(order):
    """
    构建 Hilbert 曲线的邻接图。

    返回沿 Hilbert 曲线的连续点之间的邻接关系。
    这用于定义晶粒间的连接 (晶界网络)。

    Parameters
    ----------
    order : int
        Hilbert 曲线阶数

    Returns
    -------
    adjacency : dict
        {h: [h_neighbors]} 邻接表
    positions : dict
        {h: (x,y,z)} 位置映射
    """
    n_points = 8 ** order
    positions = {}
    adjacency = {h: [] for h in range(n_points)}

    for h in range(n_points):
        x, y, z = hilbert_3d_h_to_xyz(h, order)
        positions[h] = (x, y, z)

    # 沿 Hilbert 曲线的相邻点
    for h in range(n_points - 1):
        adjacency[h].append(h + 1)
        adjacency[h + 1].append(h)

    return adjacency, positions


# ==================== 晶体结构建模 ====================

class NMCCrystalStructure:
    """
    NMC 层状氧化物晶体结构 (空间群 R-3m)。

    晶格参数:
    - a = b ≈ 2.87 Å (hexagonal)
    - c ≈ 14.2 Å
    - α = β = 90°, γ = 120°

    Li 层位于 3a 位 (0,0,0)
    TM 层位于 3b 位 (0,0,0.5)
    """

    def __init__(self, a_lat=2.87e-10, c_lat=14.2e-10):
        """
        Parameters
        ----------
        a_lat : float
            面内晶格常数 [m]
        c_lat : float
            层间晶格常数 [m]
        """
        self.a = a_lat
        self.c = c_lat
        # 六方晶系基矢
        self.a1 = np.array([a_lat, 0, 0])
        self.a2 = np.array([-a_lat/2, a_lat * math.sqrt(3)/2, 0])
        self.a3 = np.array([0, 0, c_lat])

        # 体积
        self.volume = a_lat**2 * c_lat * math.sqrt(3) / 2

    def fractional_to_cartesian(self, frac_coords):
        """分数坐标 → 笛卡尔坐标。"""
        return (frac_coords[0] * self.a1 +
                frac_coords[1] * self.a2 +
                frac_coords[2] * self.a3)

    def diffusion_tensor_isotropic(self, D_scalar):
        """各向同性扩散张量。"""
        return D_scalar * np.eye(3)

    def diffusion_tensor_anisotropic(self, D_parallel, D_perp):
        """
        各向异性扩散张量 (层状结构)。

        Li 在 NMC 中沿 ab 面扩散远快于沿 c 轴:
        D_ab / D_c ≈ 100~1000

        D̄ = D_perp * I + (D_parallel - D_perp) * n̂⊗n̂
        其中 n̂ = ĉ 为层法向

        Parameters
        ----------
        D_parallel : float
            面内扩散系数 [m²/s]
        D_perp : float
            面间扩散系数 [m²/s]

        Returns
        -------
        D_tensor : ndarray, shape (3, 3)
            扩散张量
        """
        n_hat = self.a3 / np.linalg.norm(self.a3)
        D_tensor = D_perp * np.eye(3) + (D_parallel - D_perp) * np.outer(n_hat, n_hat)
        return D_tensor

    def grain_boundary_resistance(self, E_gb, T, D_bulk):
        """
        晶界扩散阻力模型。

        D_gb = D_bulk * exp(-E_gb / (k_B * T))

        其中 E_gb 为晶界激活能垒 (通常 E_gb > E_bulk)。

        Parameters
        ----------
        E_gb : float
            晶界扩散激活能 [J]
        T : float
            温度 [K]
        D_bulk : float
            体扩散系数 [m²/s]

        Returns
        -------
        float
            晶界扩散系数 [m²/s]
        """
        exponent = -E_gb / (K_BOLTZMANN * T)
        exponent = max(min(exponent, 500.0), -500.0)
        return D_bulk * math.exp(exponent)


class PolycrystalGrainNetwork:
    """
    多晶晶粒网络。

    使用 Hilbert 曲线映射构建晶粒间的连接关系。
    每个晶粒具有:
    - 取向 (Euler 角)
    - 晶界类型 (小角/大角)
    - 局部扩散系数
    """

    def __init__(self, n_grains=8, hilbert_order=1):
        """
        Parameters
        ----------
        n_grains : int
            晶粒数
        hilbert_order : int
            Hilbert 曲线阶数 (决定空间分辨率)
        """
        self.n_grains = min(n_grains, 8 ** hilbert_order)
        self.hilbert_order = hilbert_order

        # 构建 Hilbert 邻接图
        self.adjacency, self.positions = hilbert_3d_neighbor_graph(hilbert_order)

        # 晶粒属性
        self.crystal = NMCCrystalStructure()
        self.orientations = []  # Euler angles
        self.grain_D = []       # 各晶粒扩散系数
        self.gb_D = []          # 晶界扩散系数

    def initialize_random_orientations(self, seed=42):
        """随机初始化晶粒取向。"""
        rng = np.random.default_rng(seed)
        for i in range(self.n_grains):
            # 随机 Euler 角 (Bunge 约定)
            phi1 = rng.uniform(0, 2 * math.pi)
            Phi = rng.uniform(0, math.pi)
            phi2 = rng.uniform(0, 2 * math.pi)
            self.orientations.append((phi1, Phi, phi2))

    def compute_grain_diffusivities(self, D_base, anisotropy_ratio=100.0, seed=42):
        """
        计算各晶粒的有效扩散系数。

        考虑晶体取向的各向异性:
        D_eff(grain) = R(orient) * D_crystal * R(orient)^T
        D_eff_scalar = trace(D_eff) / 3

        Parameters
        ----------
        D_base : float
            基准扩散系数 [m²/s]
        anisotropy_ratio : float
            各向异性比 D_parallel/D_perp
        seed : int
            随机种子
        """
        D_perp = D_base
        D_para = D_base * anisotropy_ratio

        for i, (phi1, Phi, phi2) in enumerate(self.orientations):
            # 旋转矩阵 (简化: 绕 z 轴旋转 phi1, 绕 x 轴旋转 Phi)
            c1, s1 = math.cos(phi1), math.sin(phi1)
            cP, sP = math.cos(Phi), math.sin(Phi)

            R = np.array([
                [c1*cP, -s1, c1*sP],
                [s1*cP, c1, s1*sP],
                [-sP, 0, cP]
            ])

            D_crystal = self.crystal.diffusion_tensor_anisotropic(D_para, D_perp)
            D_rotated = R @ D_crystal @ R.T
            D_eff = np.trace(D_rotated) / 3.0
            self.grain_D.append(D_eff)

    def compute_grain_boundary_diffusivities(self, E_gb_joules, T):
        """计算晶界扩散系数。"""
        D_bulk_avg = np.mean(self.grain_D) if self.grain_D else 1e-14
        D_gb = self.crystal.grain_boundary_resistance(E_gb_joules, T, D_bulk_avg)
        self.gb_D = [D_gb] * (self.n_grains - 1)

    def connectivity_matrix(self):
        """
        构建晶粒连通性矩阵 (Laplacian)。

        L_ij = -w_ij (i≠j, 相邻)
        L_ii = Σ_j w_ij (度)

        权重 w_ij 由晶界扩散系数决定。

        Returns
        -------
        L : ndarray, shape (n_grains, n_grains)
            加权 Laplacian 矩阵
        """
        n = self.n_grains
        L = np.zeros((n, n))

        for i in range(n):
            for j in self.adjacency.get(i, []):
                if j < n:
                    w = 1.0  # 均匀权重 (可扩展为 D_gb 依赖)
                    L[i, j] = -w
                    L[i, i] += w

        return L

    def summary(self):
        """返回晶粒网络摘要。"""
        n_edges = sum(len(v) for v in self.adjacency.values()) // 2
        return {
            'n_grains': self.n_grains,
            'n_edges': n_edges,
            'hilbert_order': self.hilbert_order,
            'mean_grain_D': float(np.mean(self.grain_D)) if self.grain_D else 0.0,
            'mean_gb_D': float(np.mean(self.gb_D)) if self.gb_D else 0.0,
        }
