# -*- coding: utf-8 -*-
"""
mesh_topology.py
=================
等离子体相空间网格拓扑与符号距离函数

融合种子项目:
  379_fem_to_medit  — FEM 网格格式转换 (节点/单元/边界掩码)
  305_dist_plot     — DistMesh 符号距离函数 (dcircle, dpoly, dsegment)
  886_polygon_integrals — 多边形精确矩积分 (Steger 1996)
  1422_xyl_display  — XY/XYL 点线拓扑数据读取

物理背景
--------
Vlasov 方程定义在相空间 (x, v) 上。在速度空间中，
等离子体分布函数被限制在有界区域内，区域边界由符号距离函数描述。

多边形矩积分用于计算速度空间分布函数区域的各阶速度矩:
  M_{pq} = ∫∫_Ω v^p u^q f(v,u) dv du

符号距离函数 φ(x):
  φ < 0: 区域内部
  φ = 0: 区域边界
  φ > 0: 区域外部
"""

import numpy as np
from typing import Tuple, List, Optional


# ===== 符号距离函数 ========================================================

def dcircle(p: np.ndarray, xc: float, yc: float, r: float) -> np.ndarray:
    """
    点到圆的符号距离

    .. math:: d(p) = \\sqrt{(p_x - x_c)^2 + (p_y - y_c)^2} - r

    d < 0: 圆内; d = 0: 圆上; d > 0: 圆外

    用于描述各向同性等离子体在速度空间中的 Maxwellian 分布等值面。
    """
    p = np.atleast_2d(p)
    return np.sqrt((p[:, 0] - xc) ** 2 + (p[:, 1] - yc) ** 2) - r


def dsegment(p: np.ndarray, pv: np.ndarray) -> np.ndarray:
    """
    点到线段集的无符号距离

    Parameters
    ----------
    p : np.ndarray, shape (np, 2)
        待计算距离的点集.
    pv : np.ndarray, shape (nvs, 2)
        线段端点 (连续端点定义线段).

    Returns
    -------
    np.ndarray, shape (np, nvs-1)
        每个点到每条线段的距离.
    """
    p = np.atleast_2d(p)
    pv = np.atleast_2d(pv)
    nvs = pv.shape[0]
    np_pts = p.shape[0]
    ds = np.zeros((np_pts, nvs - 1), dtype=np.float64)

    for iv in range(nvs - 1):
        v = pv[iv + 1] - pv[iv]
        for ip in range(np_pts):
            w = p[ip] - pv[iv]
            c1 = np.dot(v, w)
            c2 = np.dot(v, v)
            if c2 < 1.0e-300:
                ds[ip, iv] = np.sqrt(np.sum((p[ip] - pv[iv]) ** 2))
            elif c1 <= 0.0:
                ds[ip, iv] = np.sqrt(np.sum((p[ip] - pv[iv]) ** 2))
            elif c2 <= c1:
                ds[ip, iv] = np.sqrt(np.sum((p[ip] - pv[iv + 1]) ** 2))
            else:
                proj = pv[iv] + (c1 / c2) * v
                ds[ip, iv] = np.sqrt(np.sum((p[ip] - proj) ** 2))

    return ds


def dpoly(p: np.ndarray, pv: np.ndarray) -> np.ndarray:
    """
    点到多边形的符号距离

    内部为负，外部为正，边界为零。
    用于描述非圆形速度空间约束区域 (如束流等离子体)。
    """
    p = np.atleast_2d(p)
    pv = np.atleast_2d(pv)

    ds = dsegment(p, pv)
    d = np.min(ds, axis=1)

    # 使用 winding number 判断内外
    inside = _point_in_polygon(p, pv)
    d = np.where(inside, -np.abs(d), np.abs(d))

    return d


def _point_in_polygon(p: np.ndarray, pv: np.ndarray) -> np.ndarray:
    """射线法判断点是否在多边形内部."""
    p = np.atleast_2d(p)
    pv = np.atleast_2d(pv)
    n_vs = pv.shape[0] - 1  # 首尾重合
    np_pts = p.shape[0]
    inside = np.zeros(np_pts, dtype=bool)

    j = n_vs - 1
    for i in range(n_vs):
        xi, yi = pv[i, 0], pv[i, 1]
        xj, yj = pv[j, 0], pv[j, 1]
        for ip in range(np_pts):
            px, py = p[ip, 0], p[ip, 1]
            if ((yi > py) != (yj > py)) and \
               (px < (xj - xi) * (py - yi) / (yj - yi + 1e-300) + xi):
                inside[ip] = ~inside[ip]
        j = i

    return inside


# ===== 多边形矩积分 (Steger 1996) =========================================

def moment_polygon(n: int, x: np.ndarray, y: np.ndarray,
                   p: int, q: int) -> float:
    """
    多边形非归一化矩 Nu(p,q) = ∫∫_P x^p y^q dx dy

    Steger (1996) 算法: 将面积分转化为边界线积分

    .. math::
        \\nu_{pq} = \\frac{1}{(p+q+2)(p+q+1)\\binom{p+q}{p}}
                    \\sum_{i=0}^{n-1} (x_{j} y_{i} - x_{i} y_{j})
                    \\sum_{k=0}^{p}\\sum_{l=0}^{q}
                    \\binom{k+l}{l}\\binom{p+q-k-l}{q-l}
                    x_i^k x_j^{p-k} y_i^l y_j^{q-l}

    用于计算等离子体速度空间区域的速度矩和能量矩。
    """
    from grid_integer_lib import i4_choose

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    nu_pq = 0.0
    xj = x[n - 1]
    yj = y[n - 1]

    for i in range(n):
        xi = x[i]
        yi = y[i]
        s_pq = 0.0

        for k in range(p + 1):
            for l in range(q + 1):
                coeff = (i4_choose(k + l, l)
                         * i4_choose(p + q - k - l, q - l))
                s_pq += coeff * (xi ** k) * (xj ** (p - k)) * \
                        (yi ** l) * (yj ** (q - l))

        nu_pq += (xj * yi - xi * yj) * s_pq
        xj = xi
        yj = yi

    denom = (p + q + 2) * (p + q + 1) * i4_choose(p + q, p)
    nu_pq /= denom

    return nu_pq


def moment_normalized(n: int, x: np.ndarray, y: np.ndarray,
                      p: int, q: int) -> float:
    """
    归一化矩 Alpha(p,q) = Nu(p,q) / Nu(0,0) = Nu(p,q) / Area

    Alpha(1,0) = <x>, Alpha(0,1) = <y> 为质心坐标。
    """
    area = moment_polygon(n, x, y, 0, 0)
    if abs(area) < 1.0e-300:
        return 0.0
    return moment_polygon(n, x, y, p, q) / area


def moment_central(n: int, x: np.ndarray, y: np.ndarray,
                   p: int, q: int) -> float:
    """
    中心矩 Mu(p,q) = ∫∫ (x - <x>)^p (y - <y>)^q dx dy / Area

    Mu(2,0) = <(x-<x>)²> 为 x 方向方差,
    Mu(0,2) = <(y-<y>)²> 为 y 方向方差.
    """
    from grid_integer_lib import i4_choose

    alpha_10 = moment_normalized(n, x, y, 1, 0)
    alpha_01 = moment_normalized(n, x, y, 0, 1)

    mu_pq = 0.0
    for i in range(p + 1):
        for j in range(q + 1):
            alpha_ij = moment_normalized(n, x, y, i, j)
            sign = (-1) ** (p + q - i - j)
            mu_pq += (sign * i4_choose(p, i) * i4_choose(q, j)
                      * alpha_10 ** (p - i) * alpha_01 ** (q - j) * alpha_ij)

    return mu_pq


# ===== 椭圆距离函数 ========================================================

def dellipse(p: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    点到椭圆的近似符号距离

    椭圆方程: (x/a)² + (y/b)² = 1

    使用伸缩变换后的近似距离:
      d ≈ (√((x/a)² + (y/b)²) - 1) · min(a,b)

    在等离子体物理中，椭圆等值面对应于各向异性温度分布:
      f(v⊥, v∥) ∝ exp(-v⊥²/(2v_th⊥²) - v∥²/(2v_th∥²))
    """
    p = np.atleast_2d(p)
    if a < 1.0e-300 or b < 1.0e-300:
        raise ValueError("dellipse: semi-axes must be positive")

    # 近似符号距离 (精确到 O(ε²) 对于小偏心率)
    x_scaled = p[:, 0] / a
    y_scaled = p[:, 1] / b
    r_scaled = np.sqrt(x_scaled ** 2 + y_scaled ** 2)

    # 梯度修正因子
    grad_factor = np.sqrt((x_scaled / a) ** 2 + (y_scaled / b) ** 2)
    grad_factor = np.maximum(grad_factor, 1.0e-300)

    d = (r_scaled - 1.0) / grad_factor
    return d


# ===== 等离子体相空间网格 ==================================================

class PlasmaPhaseSpaceMesh:
    """
    1D-1V 等离子体相空间网格

    网格结构:
      x ∈ [x_min, x_max]  空间维度 (周期性)
      v ∈ [v_min, v_max]  速度维度 (截断)

    边界由符号距离函数定义:
      - 空间: 周期性
      - 速度: 硬壁反射或吸收边界
    """

    def __init__(self, x_min: float, x_max: float, nx: int,
                 v_min: float, v_max: float, nv: int):
        self.x_min = x_min
        self.x_max = x_max
        self.nx = nx
        self.v_min = v_min
        self.v_max = v_max
        self.nv = nv

        self.dx = (x_max - x_min) / max(nx, 1)
        self.dv = (v_max - v_min) / max(nv, 1)

        # 单元中心网格
        self.x_grid = np.linspace(x_min + 0.5 * self.dx,
                                   x_max - 0.5 * self.dx, nx)
        self.v_grid = np.linspace(v_min + 0.5 * self.dv,
                                   v_max - 0.5 * self.dv, nv)

        # 单元界面网格
        self.x_intf = np.linspace(x_min, x_max, nx + 1)
        self.v_intf = np.linspace(v_min, v_max, nv + 1)

        # 构建边界掩码和符号距离函数
        self.boundary_mask = np.zeros((nx, nv), dtype=bool)
        self.sdf = np.zeros((nx, nv), dtype=np.float64)
        self._setup_boundary()

    def _setup_boundary(self):
        """设置速度空间边界区域."""
        for i in range(self.nx):
            for j in range(self.nv):
                v = self.v_grid[j]
                v_boundary = 0.95 * max(abs(self.v_min), abs(self.v_max))
                self.boundary_mask[i, j] = abs(v) > v_boundary
                self.sdf[i, j] = abs(v) - v_boundary

    def area(self) -> float:
        """计算有效相空间面积 (排除边界区域)."""
        return float(np.sum(~self.boundary_mask) * self.dx * self.dv)

    def cell_area(self) -> float:
        """单个网格单元面积."""
        return self.dx * self.dv

    def total_cells(self) -> int:
        """总网格单元数."""
        return self.nx * self.nv

    def interior_cells(self) -> int:
        """内部 (非边界) 单元数."""
        return int(np.sum(~self.boundary_mask))

    def is_interior(self, i_x: int, i_v: int) -> bool:
        """判断网格单元 (i_x, i_v) 是否在内部."""
        if i_x < 0 or i_x >= self.nx or i_v < 0 or i_v >= self.nv:
            return False
        return not self.boundary_mask[i_x, i_v]

    def v_index_nearest(self, v: float) -> int:
        """找到最接近速度 v 的网格索引."""
        idx = int(round((v - self.v_min) / self.dv - 0.5))
        return max(0, min(self.nv - 1, idx))


# ===== 相空间区域划分 ======================================================

class PhaseSpacePartition:
    """
    相空间区域划分 — 用于波-粒子共振分析

    将速度空间划分为:
      - 共振区: |v - v_phase| < Δv_res
      - 捕获区: 粒子被波势阱捕获
      - 通行区: 自由通行粒子
    """

    def __init__(self, v_grid: np.ndarray, v_phase: float,
                 trapping_width: float):
        self.v_grid = v_grid
        self.v_phase = v_phase
        self.trapping_width = trapping_width

        # 共振区标志
        self.resonance_mask = (
            np.abs(v_grid - v_phase) < 3.0 * trapping_width)

        # 捕获区: v 满足 (v - v_phase)² / 2 < Φ_wave
        v_shift = v_grid - v_phase
        kinetic = 0.5 * v_shift ** 2
        potential = 0.5 * trapping_width ** 2
        self.trapping_mask = kinetic < potential

    def resonance_sdf(self) -> np.ndarray:
        """共振区的符号距离函数."""
        return np.abs(self.v_grid - self.v_phase) - 3.0 * self.trapping_width

    def trapping_sdf(self) -> np.ndarray:
        """捕获区的符号距离函数."""
        v_shift = self.v_grid - self.v_phase
        kinetic = 0.5 * v_shift ** 2
        potential = 0.5 * self.trapping_width ** 2
        return kinetic - potential

    def resonance_width(self) -> float:
        """共振区宽度."""
        return float(6.0 * self.trapping_width)

    def trapping_fraction(self, f: np.ndarray, dv: float) -> float:
        """
        计算被捕获粒子占总粒子的比例

        F_trap = ∫_{trap} f dv / ∫ f dv
        """
        f = np.asarray(f, dtype=np.float64)
        total = np.sum(f) * dv
        if total < 1.0e-300:
            return 0.0
        trapped = np.sum(f[self.trapping_mask]) * dv
        return trapped / total


# ===== 网格格式转换 ========================================================

class MeshDataConverter:
    """
    网格数据格式转换器

    在不同网格表示之间转换:
      - 结构化 (i,j) → 坐标 (x,v)
      - 边界掩码 → 边界线段列表
      - 节点-单元表示 → 接口数据

    源自 FEM ↔ MEDIT 格式转换思想。
    """

    @staticmethod
    def structured_to_nodes_elements(
        x: np.ndarray, v: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        将结构化网格转换为节点-单元表示

        Returns
        -------
        nodes : np.ndarray, shape (nx*nv, 2)
            节点坐标.
        elements : np.ndarray, shape (ne, 4)
            四边形单元节点索引.
        boundary_mask : np.ndarray, shape (nx*nv,)
            边界节点标志.
        """
        nx = len(x)
        nv = len(v)
        xx, vv = np.meshgrid(x, v, indexing='ij')
        nodes = np.column_stack([xx.ravel(), vv.ravel()])

        # 构建四边形单元
        elements = []
        for i in range(nx - 1):
            for j in range(nv - 1):
                n0 = i * nv + j
                n1 = (i + 1) * nv + j
                n2 = (i + 1) * nv + (j + 1)
                n3 = i * nv + (j + 1)
                elements.append([n0, n1, n2, n3])
        elements = np.array(elements, dtype=int)

        # 边界节点
        boundary_mask = np.zeros(nx * nv, dtype=bool)
        for i in range(nx):
            for j in range(nv):
                idx = i * nv + j
                if i == 0 or i == nx - 1 or j == 0 or j == nv - 1:
                    boundary_mask[idx] = True

        return nodes, elements, boundary_mask

    @staticmethod
    def boundary_mask_to_segments(
        mask: np.ndarray, nx: int, nv: int
    ) -> np.ndarray:
        """
        将 2D 边界掩码转换为边界线段列表

        Returns
        -------
        segments : np.ndarray, shape (n_seg, 4)
            每条线段 [x1, v1, x2, v2].
        """
        segments = []
        for i in range(nx):
            for j in range(nv):
                if not mask[i, j]:
                    continue
                # 检查4个邻面是否暴露
                if i > 0 and not mask[i - 1, j]:
                    segments.append([i, j, i, j + 1])
                if i < nx - 1 and not mask[i + 1, j]:
                    segments.append([i + 1, j, i + 1, j + 1])
                if j > 0 and not mask[i, j - 1]:
                    segments.append([i, j, i + 1, j])
                if j < nv - 1 and not mask[i, j + 1]:
                    segments.append([i, j + 1, i + 1, j + 1])
        return np.array(segments, dtype=np.float64) if segments else \
            np.zeros((0, 4), dtype=np.float64)
