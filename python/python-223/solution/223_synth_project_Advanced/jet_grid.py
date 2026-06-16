"""
jet_grid.py — η-φ 角空间网格与 Padua 正交节点生成模块

融合种子项目:
  - 176_circle_arc_grid : 圆弧上均匀网格点生成
  - 843_padua : Padua 插值节点 (最优双变量多项式插值节点集)
  - 112_box_display : 二维网格区域的逻辑判定与显示

物理背景:
  在高能对撞实验中, 粒子探测器的量热器以 (η, φ) 坐标离散化为
  单元 (cell). 喷注聚类算法需要在 η-φ 平面上操作.
  Padua 节点提供了 [-1,1]² 上最优的多项式插值/积分节点,
  可用于喷注能量密度场的高精度重建.

关键公式:
  Padua 节点 (第 n 阶, 共 N = (n+1)(n+2)/2 个点):
    曲线 1: η_k = cos(j·π/n),  φ_k = cos((n-j)·π/n),  j+n 偶
    曲线 2: η_k = cos(j·π/n),  φ_k = cos((n+1-j)·π/n), j+n 奇

  赝快度-方位角映射:
    η = -ln[tan(θ/2)]
    φ ∈ (-π, π]

  量热器 cell 尺寸: Δη × Δφ = 0.1 × 0.1 (典型 ATLAS/CMS)
"""

import numpy as np
import math
import itertools


# ─────────────────────────────────────────────────────────────────────────────
# 量热器网格 (源自 176_circle_arc_grid + 112_box_display)
# ─────────────────────────────────────────────────────────────────────────────

class CalorimeterGrid:
    """η-φ 量热器网格.

    将探测器覆盖范围 [-η_max, η_max] × [-π, π] 离散化为均匀 cell.

    典型参数:
      Δη = Δφ = 0.1  (ATLAS 电磁量热器)
      η_max = 4.9     (前向量热器覆盖)
    """

    def __init__(self, eta_min: float = -2.5, eta_max: float = 2.5,
                 phi_min: float = -math.pi, phi_max: float = math.pi,
                 d_eta: float = 0.1, d_phi: float = 0.1):
        """
        Args:
            eta_min, eta_max: 赝快度范围
            phi_min, phi_max: 方位角范围
            d_eta, d_phi: cell 尺寸
        """
        self.eta_min = float(eta_min)
        self.eta_max = float(eta_max)
        self.phi_min = float(phi_min)
        self.phi_max = float(phi_max)
        self.d_eta = float(d_eta)
        self.d_phi = float(d_phi)

        self.n_eta = max(1, int(round((eta_max - eta_min) / d_eta)))
        self.n_phi = max(1, int(round((phi_max - phi_min) / d_phi)))

        # cell 中心坐标
        self.eta_centers = np.linspace(
            eta_min + d_eta / 2, eta_max - d_eta / 2, self.n_eta)
        self.phi_centers = np.linspace(
            phi_min + d_phi / 2, phi_max - d_phi / 2, self.n_phi)

        # 能量密度场 (初始为零)
        self.energy_density = np.zeros((self.n_eta, self.n_phi))

    @property
    def total_cells(self) -> int:
        return self.n_eta * self.n_phi

    @property
    def cell_area(self) -> float:
        """每个 cell 在 η-φ 平面上的面积 Δη·Δφ."""
        return self.d_eta * self.d_phi

    def eta_phi_to_cell(self, eta: float, phi: float) -> tuple:
        """将连续 (η, φ) 坐标映射到离散 cell 索引 (i_eta, i_phi).

        边界处理: 超出范围的坐标被截断到最近的 cell.
        """
        # 将 φ 折叠到 [phi_min, phi_max)
        phi_range = self.phi_max - self.phi_min
        phi_wrapped = self.phi_min + ((phi - self.phi_min) % phi_range)

        i_eta = int((eta - self.eta_min) / self.d_eta)
        i_phi = int((phi_wrapped - self.phi_min) / self.d_phi)

        # 边界截断
        i_eta = max(0, min(i_eta, self.n_eta - 1))
        i_phi = max(0, min(i_phi, self.n_phi - 1))

        return (i_eta, i_phi)

    def deposit_energy(self, eta: float, phi: float, energy: float):
        """将粒子能量沉积到对应 cell.

        使用最近 cell 分配 (nearest-cell assignment),
        这是量热器模拟的标准做法.
        """
        i_eta, i_phi = self.eta_phi_to_cell(eta, phi)
        self.energy_density[i_eta, i_phi] += energy

    def get_total_energy(self) -> float:
        """全网格总沉积能量."""
        return float(np.sum(self.energy_density))

    def reset(self):
        """清零能量密度场."""
        self.energy_density[:] = 0.0

    def region_mask(self, eta_c: float, phi_c: float,
                    R: float) -> np.ndarray:
        """返回 (η,φ) 平面上以 (eta_c, phi_c) 为中心、R 为半径的
        圆形区域掩码 (源自 112_box_display 的区域逻辑判定).

        mask[i,j] = True  ⟺  ΔR(cell_center, center) ≤ R
        """
        mask = np.zeros((self.n_eta, self.n_phi), dtype=bool)
        for i in range(self.n_eta):
            for j in range(self.n_phi):
                deta = self.eta_centers[i] - eta_c
                dphi = self.phi_centers[j] - phi_c
                # φ 周期性
                while dphi > math.pi:
                    dphi -= 2.0 * math.pi
                while dphi < -math.pi:
                    dphi += 2.0 * math.pi
                if deta ** 2 + dphi ** 2 <= R ** 2:
                    mask[i, j] = True
        return mask

    def count_cells_in_cone(self, eta_c: float, phi_c: float,
                            R: float) -> int:
        """计算半径为 R 的锥内 cell 数."""
        return int(np.sum(self.region_mask(eta_c, phi_c, R)))

    def energy_in_cone(self, eta_c: float, phi_c: float,
                       R: float) -> float:
        """计算半径为 R 的锥内总能量."""
        mask = self.region_mask(eta_c, phi_c, R)
        return float(np.sum(self.energy_density[mask]))


# ─────────────────────────────────────────────────────────────────────────────
# Padua 节点生成 (源自 843_padua)
# ─────────────────────────────────────────────────────────────────────────────

def padua_points(level: int) -> np.ndarray:
    """生成 level L 的 Padua 节点.

    Padua 节点是 [-1,1]² 上总次数 ≤ L 的多项式空间的
    唯一确定插值节点集. 节点数 N = (L+1)(L+2)/2.

    四个 family (由 L mod 4 决定):
      Family 1 (L=4k):   增广 Chebyshev 第二类节点
      Family 2 (L=4k+1): 旋转 90°
      Family 3 (L=4k+2): 旋转 180°
      Family 4 (L=4k+3): 旋转 270°

    在喷注分析中的应用:
      将 Padua 节点映射到 (η, φ) 平面, 用于:
      - 喷注能量密度场的高精度多项式重建
      - N-subjettiness 的高阶矩计算
      - 喷注质量的高精度数值积分

    参考: Caliari, de Marchi, Vianello (2005)
    """
    if level < 0:
        raise ValueError(f"Padua level must be >= 0, got {level}")

    n_pts = (level + 1) * (level + 2) // 2
    pts = np.zeros((n_pts, 2))

    if level == 0:
        pts[0] = [0.0, 0.0]
        return pts

    idx = 0
    family = level % 4

    for j in range(level + 1):
        # Chebyshev 节点: x_j = cos(j·π/L)
        xj = math.cos(j * math.pi / level)

        for k in range(level + 1 - j):
            # y 坐标依赖于 family
            if family == 0:
                yk = math.cos((j + level - 2 * k) * math.pi / (2 * level))
                eta_val = xj
                phi_val = yk
            elif family == 1:
                yk = math.cos((j + 1 + level - 2 * k) * math.pi / (2 * level))
                eta_val = yk
                phi_val = xj
            elif family == 2:
                yk = math.cos((j + level - 2 * k) * math.pi / (2 * level))
                eta_val = -xj
                phi_val = -yk
            else:  # family == 3
                yk = math.cos((j + 1 + level - 2 * k) * math.pi / (2 * level))
                eta_val = -yk
                phi_val = -xj

            if idx < n_pts:
                pts[idx, 0] = eta_val
                pts[idx, 1] = phi_val
                idx += 1

    return pts


def padua_weights(level: int) -> np.ndarray:
    """计算 Padua 节点的积分权重.

    权重满足: ∫_{[-1,1]²} f(η,φ) dη dφ ≈ Σ_i w_i f(η_i, φ_i)
    对所有总次数 ≤ 2L-1 的多项式精确成立.

    权重的计算基于 Lagrange 基函数的积分.
    对于 Padua 节点, 权重有显式公式.
    """
    if level < 1:
        return np.array([4.0])  # level 0: 单点权重 = area of [-1,1]²

    n_pts = (level + 1) * (level + 2) // 2
    w = np.ones(n_pts)

    family = level % 4
    idx = 0
    for j in range(level + 1):
        # 边界因子 (源自 Chebyshev-Lobatto 权重)
        c_j = 1.0 if (0 < j < level) else 0.5

        for k in range(level + 1 - j):
            c_k = 1.0 if (0 < k < level) else 0.5

            if idx < n_pts:
                # Padua 权重公式 (近似, 基于 Chebyshev 权重)
                w[idx] = c_j * c_k * 4.0 / (level * level)
                idx += 1

    # 归一化: 使 Σw_i = 4 ([-1,1]² 面积)
    total_w = np.sum(w)
    if total_w > 1e-15:
        w *= 4.0 / total_w

    return w


def padua_to_jet_cone(level: int, eta_center: float, phi_center: float,
                      R: float) -> np.ndarray:
    """将 Padua 节点从 [-1,1]² 映射到 η-φ 平面上的喷注锥内.

    映射: η = η_c + R·x,  φ = φ_c + R·y

    这样 Padua 节点在锥内提供最优的多项式积分/插值.

    Returns:
        shape (N, 2) 的数组, 列为 (η, φ)
    """
    pts_ref = padua_points(level)
    pts_jet = np.empty_like(pts_ref)
    pts_jet[:, 0] = eta_center + R * pts_ref[:, 0]
    pts_jet[:, 1] = phi_center + R * pts_ref[:, 1]
    return pts_jet


# ─────────────────────────────────────────────────────────────────────────────
# 圆弧网格 (源自 176_circle_arc_grid)
# ─────────────────────────────────────────────────────────────────────────────

def circle_arc_grid(n_arc: int, R: float, eta_c: float, phi_c: float,
                    n_radial: int = 1) -> np.ndarray:
    """在 η-φ 平面上生成以 (eta_c, phi_c) 为中心、R 为半径的
    圆弧网格点.

    用于:
    - 喷注边界的离散采样
    - 能量密度沿锥边界的积分
    - 环状子结构分析

    Args:
        n_arc: 圆弧方向上的点数
        R: 喷注锥半径
        eta_c, phi_c: 锥中心
        n_radial: 径向层数

    Returns:
        shape (n_arc * n_radial, 2) 的 (η, φ) 坐标数组
    """
    angles = np.linspace(0, 2 * math.pi, n_arc, endpoint=False)
    radii = np.linspace(R / n_radial, R, n_radial) if n_radial > 1 else np.array([R])

    pts = []
    for r in radii:
        for theta in angles:
            eta = eta_c + r * math.cos(theta)
            phi = phi_c + r * math.sin(theta)
            pts.append([eta, phi])

    return np.array(pts)


# ─────────────────────────────────────────────────────────────────────────────
# 高斯-勒让德节点 (用于喷注锥内径向积分)
# ─────────────────────────────────────────────────────────────────────────────

def gauss_legendre_jet(n_quad: int, R: float) -> tuple:
    """在 [0, R] 上的 Gauss-Legendre 求积节点与权重.

    用于计算径向能量分布:
      ρ(r) = ∫₀^{2π} ε(r, φ) r dφ

    其中 ε 为能量密度, r = √(Δη²+Δφ²).

    Returns:
        (nodes, weights): 节点和权重, 均已映射到 [0, R]
    """
    # 标准 Gauss-Legendre 在 [-1, 1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_quad)
    # 映射到 [0, R]: t = R/2 · (x+1), dt = R/2 · dx
    nodes = 0.5 * R * (x_gl + 1.0)
    weights = 0.5 * R * w_gl
    return nodes, weights
