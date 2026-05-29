"""
eddy_geometry.py
涡旋几何检测、边界识别与非结构化网格分析模块

科学背景:
海洋中尺度涡旋的检测与几何特征量化:
  - 涡旋边界通过涡度/速度的阈值或 OW (Okubo-Weiss) 参数识别
  - 多边形矩计算量化涡旋的惯性、形状和强度
  - 非结构化网格用于复杂海岸边界下的涡旋模拟

Okubo-Weiss 参数:
  W = S_n^2 + S_s^2 - \omega^2
  其中:
    S_n = ∂u/∂x - ∂v/∂y    (法向应变)
    S_s = ∂v/∂x + ∂u/∂y    (切向应变)
    \omega = ∂v/∂x - ∂u/∂y  (涡度)

  W < 0: 涡旋核心区 (旋转主导)
  W > 0: 应变区 (变形主导)

涡旋动能:
  KE = (1/2) ∫_A (u^2 + v^2) dA

本模块实现:
  - OW 参数场计算与涡旋核心区检测
  - 多边形精确矩计算 (面积、质心、惯性矩)
  - 距离函数与简化非结构化网格生成
  - 涡旋追踪与属性统计

融合来源:
- 325_edge: 边缘/锋面检测思想 (阈值函数)
- 886_polygon_integrals: 多边形矩精确计算
- 308_distmesh: 距离函数与网格生成
"""

import numpy as np
import math
from numerics_core import safe_divide
from typing import Tuple, List, Dict, Optional, Callable


# ============================================================
# 1. Okubo-Weiss 参数与涡旋检测
# ============================================================

def compute_okubo_weiss(u: np.ndarray, v: np.ndarray,
                        dx: float, dy: float) -> np.ndarray:
    """
    计算 Okubo-Weiss 参数场.

    W = S_n^2 + S_s^2 - \omega^2
      = (u_x - v_y)^2 + (v_x + u_y)^2 - (v_x - u_y)^2
      = u_x^2 + v_y^2 + 2*u_y*v_x - 2*u_x*v_y

    中心差分计算梯度:
      ∂u/∂x ≈ (u[i,j+1] - u[i,j-1]) / (2*dx)
    """
    Ny, Nx = u.shape
    if u.shape != v.shape:
        raise ValueError("u and v must have same shape")

    # 中心差分 (内部)
    ux = np.zeros_like(u)
    uy = np.zeros_like(u)
    vx = np.zeros_like(v)
    vy = np.zeros_like(v)

    ux[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dx)
    uy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2.0 * dy)
    vx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2.0 * dx)
    vy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2.0 * dy)

    # 周期边界
    ux[:, 0] = (u[:, 1] - u[:, -1]) / (2.0 * dx)
    ux[:, -1] = (u[:, 0] - u[:, -2]) / (2.0 * dx)
    uy[0, :] = (u[1, :] - u[-1, :]) / (2.0 * dy)
    uy[-1, :] = (u[0, :] - u[-2, :]) / (2.0 * dy)
    vx[:, 0] = (v[:, 1] - v[:, -1]) / (2.0 * dx)
    vx[:, -1] = (v[:, 0] - v[:, -2]) / (2.0 * dx)
    vy[0, :] = (v[1, :] - v[-1, :]) / (2.0 * dy)
    vy[-1, :] = (v[0, :] - v[-2, :]) / (2.0 * dy)

    Sn = ux - vy
    Ss = vx + uy
    omega = vx - uy

    W = Sn ** 2 + Ss ** 2 - omega ** 2
    return W


def detect_vortex_cores(W: np.ndarray, threshold_factor: float = 0.2) -> Tuple[np.ndarray, float]:
    """
    检测涡旋核心区.

    阈值: W < W_std * threshold_factor, 其中 W_std 为 W 的标准差.
    """
    W_mean = np.mean(W)
    W_std = np.std(W)
    threshold = W_mean - threshold_factor * W_std
    cores = W < threshold
    return cores, threshold


# ============================================================
# 2. 多边形矩计算 (from 886_polygon_integrals)
# ============================================================

def polygon_moments(vertices: np.ndarray, max_order: int = 2) -> Dict[Tuple[int, int], float]:
    """
    精确计算多边形上的原点矩:
      M_{p,q} = \iint_P x^p y^q dx dy

    算法: Steger 方法, 沿多边形边解析积分.
      M_{p,q} = sum_{edge} \int_0^1 [(x_i + t*dx)^p (y_i + t*dy)^q] * (x_i*dy - y_i*dx) dt
    其中 dx = x_{i+1} - x_i, dy = y_{i+1} - y_i.

    Parameters
    ----------
    vertices : np.ndarray, shape (n,2)
        多边形顶点 (逆时针)
    max_order : int
        最高矩阶数

    Returns
    -------
    dict
        {(p,q): M_{p,q}}
    """
    n = vertices.shape[0]
    if n < 3:
        return {(0, 0): 0.0}

    moments = {}
    for p in range(max_order + 1):
        for q in range(max_order + 1 - p):
            moments[(p, q)] = 0.0

    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        dx = x1 - x0
        dy = y1 - y0
        cross = x0 * dy - y0 * dx

        # 积分 \int_0^1 (x0 + t*dx)^p (y0 + t*dy)^q dt
        # 二项式展开
        for p in range(max_order + 1):
            for q in range(max_order + 1 - p):
                val = 0.0
                for a in range(p + 1):
                    for b in range(q + 1):
                        coeff = (math.comb(p, a) * math.comb(q, b) *
                                 (x0 ** (p - a)) * (dx ** a) *
                                 (y0 ** (q - b)) * (dy ** b))
                        power = a + b
                        if power >= 0:
                            coeff /= (power + 1)
                        val += coeff
                moments[(p, q)] += val * cross

    # 方向修正 (逆时针为正)
    if moments[(0, 0)] < 0:
        for key in moments:
            moments[key] *= -1

    return moments


def polygon_central_moments(vertices: np.ndarray, max_order: int = 2) -> Dict[Tuple[int, int], float]:
    """
    中心矩: 以质心为原点.

    质心: (x_c, y_c) = (M_{1,0}/M_{0,0}, M_{0,1}/M_{0,0})
    中心矩通过平移公式:
      \mu_{p,q} = sum_{i=0}^p sum_{j=0}^q C(p,i) C(q,j) (-x_c)^{p-i} (-y_c)^{q-j} M_{i,j}
    """
    M = polygon_moments(vertices, max_order)
    area = M[(0, 0)]
    if area < 1e-15:
        return M

    xc = M[(1, 0)] / area
    yc = M[(0, 1)] / area

    mu = {}
    for p in range(max_order + 1):
        for q in range(max_order + 1 - p):
            val = 0.0
            for i in range(p + 1):
                for j in range(q + 1):
                    val += (math.comb(p, i) * math.comb(q, j) *
                            ((-xc) ** (p - i)) * ((-yc) ** (q - j)) *
                            M.get((i, j), 0.0))
            mu[(p, q)] = val

    return mu


def polygon_inertia_tensor(vertices: np.ndarray) -> np.ndarray:
    """
    惯性张量 (相对于质心):
      I_xx = \mu_{0,2}
      I_yy = \mu_{2,0}
      I_xy = \mu_{1,1}
    """
    mu = polygon_central_moments(vertices, max_order=2)
    return np.array([[mu[(0, 2)], -mu[(1, 1)]],
                     [-mu[(1, 1)], mu[(2, 0)]]])


def polygon_eccentricity(vertices: np.ndarray) -> float:
    """
    多边形离心率 (形状度量):
      e = sqrt(1 - (lambda_min / lambda_max))
    lambda 为惯性张量的特征值.
    """
    I = polygon_inertia_tensor(vertices)
    eigvals = np.linalg.eigvalsh(I)
    if eigvals[1] < 1e-15:
        return 0.0
    return np.sqrt(1.0 - eigvals[0] / eigvals[1])


# ============================================================
# 3. 距离函数与简化网格 (from 308_distmesh)
# ============================================================

def signed_distance_circle(p: np.ndarray, xc: float, yc: float, r: float) -> np.ndarray:
    """圆的带符号距离函数."""
    return np.sqrt((p[:, 0] - xc) ** 2 + (p[:, 1] - yc) ** 2) - r


def signed_distance_rectangle(p: np.ndarray, x1: float, y1: float,
                              x2: float, y2: float) -> np.ndarray:
    """矩形的带符号距离函数."""
    dx = np.maximum(np.maximum(x1 - p[:, 0], p[:, 0] - x2), 0.0)
    dy = np.maximum(np.maximum(y1 - p[:, 1], p[:, 1] - y2), 0.0)
    return np.sqrt(dx ** 2 + dy ** 2)


def signed_distance_union(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """并集距离函数."""
    return np.minimum(d1, d2)


def signed_distance_difference(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """差集距离函数."""
    return np.maximum(d1, -d2)


def simple_mesh_2d(fd: Callable, fh: Callable, bbox: Tuple[float, float, float, float],
                   h0: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    简化版 2D 非结构化网格生成器 (基于 distmesh 思想).

    算法:
      1. 在边界框内均匀撒点
      2. 保留 fd(p) < 0 的内部点
      3. 对内部点做 Delaunay 三角剖分
      4. 移除边界外的三角形

    Parameters
    ----------
    fd : callable
        带符号距离函数 fd(p) < 0 表示内部
    fh : callable
        网格尺寸密度函数
    bbox : (xmin, ymin, xmax, ymax)
        边界框
    h0 : float
        基准网格尺寸

    Returns
    -------
    p : np.ndarray, shape (n,2)
        节点坐标
    t : np.ndarray, shape (m,3)
        三角形连接 (节点索引)
    """
    from scipy.spatial import Delaunay

    xmin, ymin, xmax, ymax = bbox
    # 初始均匀网格
    nx = int(np.ceil((xmax - xmin) / h0))
    ny = int(np.ceil((ymax - ymin) / h0))
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    p_init = np.column_stack([X.flatten(), Y.flatten()])

    # 保留内部点
    d = fd(p_init)
    p = p_init[d < 0]

    if len(p) < 3:
        return p, np.zeros((0, 3), dtype=int)

    # Delaunay 三角剖分
    tri = Delaunay(p)
    t = tri.simplices

    # 移除质心在边界外的三角形
    pc = (p[t[:, 0]] + p[t[:, 1]] + p[t[:, 2]]) / 3.0
    t = t[fd(pc) < 0]

    return p, t


# ============================================================
# 4. 涡旋对象与追踪器
# ============================================================

class Eddy:
    """单个涡旋的几何与物理属性."""

    def __init__(self, label: int, mask: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray,
                 q: np.ndarray, u: np.ndarray, v: np.ndarray):
        self.label = label
        self.mask = mask
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.dx = x_grid[1] - x_grid[0]
        self.dy = y_grid[1] - y_grid[0]

        # 计算几何属性
        self._compute_geometry()
        # 计算物理属性
        self._compute_physics(q, u, v)

    def _compute_geometry(self):
        """从掩码提取多边形边界并计算矩."""
        # 如果没有 skimage, 使用简化方法
        try:
            from skimage.measure import find_contours
            contours = find_contours(self.mask, 0.5)
        except Exception:
            contours = self._simple_contour()

        if contours:
            # 取最长轮廓
            longest = max(contours, key=len)
            # 映射到物理坐标
            vertices = np.zeros_like(longest)
            vertices[:, 0] = self.x_grid[np.clip(longest[:, 1].astype(int), 0, len(self.x_grid) - 1)]
            vertices[:, 1] = self.y_grid[np.clip(longest[:, 0].astype(int), 0, len(self.y_grid) - 1)]
            self.vertices = vertices

            M = polygon_moments(vertices, max_order=2)
            self.area = M[(0, 0)]
            if self.area > 1e-15:
                self.centroid_x = M[(1, 0)] / self.area
                self.centroid_y = M[(0, 1)] / self.area
            else:
                self.centroid_x = np.mean(self.x_grid[self.mask.any(axis=0)])
                self.centroid_y = np.mean(self.y_grid[self.mask.any(axis=1)])
            self.eccentricity = polygon_eccentricity(vertices)
        else:
            self.vertices = np.zeros((0, 2))
            self.area = 0.0
            self.centroid_x = 0.0
            self.centroid_y = 0.0
            self.eccentricity = 0.0

    def _simple_contour(self) -> List[np.ndarray]:
        """简化轮廓提取 (4-连通边界)."""
        Ny, Nx = self.mask.shape
        boundary = np.zeros_like(self.mask, dtype=bool)
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                if self.mask[j, i]:
                    if not (self.mask[j - 1, i] and self.mask[j + 1, i] and
                            self.mask[j, i - 1] and self.mask[j, i + 1]):
                        boundary[j, i] = True
        # 提取边界点并排序 (简化)
        points = np.column_stack(np.where(boundary))
        if len(points) < 3:
            return []
        # 简单返回
        return [points[:, [1, 0]].astype(float)]

    def _compute_physics(self, q: np.ndarray, u: np.ndarray, v: np.ndarray):
        """计算涡旋的物理属性."""
        if np.any(self.mask):
            self.mean_vorticity = float(np.mean(q[self.mask]))
            self.max_vorticity = float(np.max(np.abs(q[self.mask])))
            self.kinetic_energy = float(0.5 * np.sum((u[self.mask] ** 2 + v[self.mask] ** 2)) *
                                        self.dx * self.dy)
            self.circulation = float(np.sum(q[self.mask]) * self.dx * self.dy)
        else:
            self.mean_vorticity = 0.0
            self.max_vorticity = 0.0
            self.kinetic_energy = 0.0
            self.circulation = 0.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "centroid": (float(self.centroid_x), float(self.centroid_y)),
            "area": float(self.area),
            "eccentricity": float(self.eccentricity),
            "mean_vorticity": float(self.mean_vorticity),
            "max_vorticity": float(self.max_vorticity),
            "kinetic_energy": float(self.kinetic_energy),
            "circulation": float(self.circulation)
        }


def extract_eddies(q: np.ndarray, u: np.ndarray, v: np.ndarray,
                   x_grid: np.ndarray, y_grid: np.ndarray,
                   threshold_factor: float = 0.2) -> List[Eddy]:
    """
    从速度场提取涡旋对象列表.

    使用 Okubo-Weiss 参数检测涡旋核心区, 然后通过连通分量标记.
    """
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    W = compute_okubo_weiss(u, v, dx, dy)
    cores, _ = detect_vortex_cores(W, threshold_factor)

    # 连通分量标记 (简化版)
    Ny, Nx = cores.shape
    labels = np.zeros_like(cores, dtype=int)
    current_label = 0

    for j in range(Ny):
        for i in range(Nx):
            if cores[j, i] and labels[j, i] == 0:
                current_label += 1
                # Flood fill (4-连通)
                stack = [(j, i)]
                while stack:
                    cj, ci = stack.pop()
                    if 0 <= cj < Ny and 0 <= ci < Nx and cores[cj, ci] and labels[cj, ci] == 0:
                        labels[cj, ci] = current_label
                        stack.extend([(cj - 1, ci), (cj + 1, ci), (cj, ci - 1), (cj, ci + 1)])

    eddies = []
    for lbl in range(1, current_label + 1):
        mask = labels == lbl
        # 过滤过小的区域
        if np.sum(mask) < 4:
            continue
        eddy = Eddy(lbl, mask, x_grid, y_grid, q, u, v)
        if eddy.area > 1e-10:
            eddies.append(eddy)

    return eddies


if __name__ == "__main__":
    # 测试多边形矩
    verts = np.array([[0, 0], [2, 0], [2, 1], [0, 1]])
    M = polygon_moments(verts, max_order=2)
    print("Rectangle moments:", M)
    print("Area:", M[(0, 0)])

    # 测试 OW
    x = np.linspace(0, 2*np.pi, 64)
    y = np.linspace(0, 2*np.pi, 64)
    X, Y = np.meshgrid(x, y)
    u = -np.sin(X) * np.cos(Y)
    v = np.cos(X) * np.sin(Y)
    W = compute_okubo_weiss(u, v, x[1]-x[0], y[1]-y[0])
    print("OW range:", np.min(W), np.max(W))
