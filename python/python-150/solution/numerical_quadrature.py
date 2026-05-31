"""
numerical_quadrature.py
=======================
数值积分模块：分子表面与键积分

融合种子项目:
  - 1304_triangle_felippa_rule : 三角形区域上的对称 Gauss 求积规则
  - 685_line_nco_rule          : Newton-Cotes Open 一维求积

科学背景:
  分子性质的量子力学计算常涉及空间积分，例如:
    - 交换-相关能 E_xc = ∫ ε_xc[ρ(r)] ρ(r) dr
    - 重叠积分 S_μν = ∫ φ_μ(r) φ_ν(r) dr
    - 键能路径积分 ∫ V(r(s)) ds

  本模块提供三角形面元上的高精度求积（用于分子表面积分）
  以及线段上的 Newton-Cotes Open 求积（用于沿化学键的线积分）。
"""

import numpy as np
from typing import Tuple


# ===================================================================
# 1. 三角形求积 (Felippa 对称规则, 源自 triangle_felippa_rule)
# ===================================================================

def triangle_unit_monomial_integral(m: int, n: int) -> float:
    """
    单位三角形 (顶点 (0,0), (1,0), (0,1)) 上的单项式积分:
        I = ∫∫ x^m y^n dx dy = m! n! / (m + n + 2)!
    """
    from math import factorial
    return factorial(m) * factorial(n) / factorial(m + n + 2)


def triangle_unit_o01() -> Tuple[np.ndarray, np.ndarray]:
    """1 点规则, 精度 1: 重心 (1/3, 1/3), 权重 1/2。"""
    xy = np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64)
    w = np.array([0.5], dtype=np.float64)
    return w, xy


def triangle_unit_o03() -> Tuple[np.ndarray, np.ndarray]:
    """3 点规则, 精度 2。"""
    xy = np.array([
        [0.666666666666667, 0.166666666666667],
        [0.166666666666667, 0.666666666666667],
        [0.166666666666667, 0.166666666666667]
    ], dtype=np.float64)
    w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=np.float64)
    return w, xy


def triangle_unit_o07() -> Tuple[np.ndarray, np.ndarray]:
    """7 点规则, 精度 5。"""
    a = 0.059715871789770
    b = 0.797426985353087
    c = 0.101286507323456
    d = 0.25

    xy = np.array([
        [a, a],
        [1.0 - 2.0 * a, a],
        [a, 1.0 - 2.0 * a],
        [b, c],
        [c, b],
        [c, c],
        [d, d]
    ], dtype=np.float64)

    w1 = 0.1125
    w2 = (155.0 - np.sqrt(15.0)) / 1200.0
    w3 = (155.0 + np.sqrt(15.0)) / 1200.0
    w4 = 0.225
    w = np.array([w2, w2, w2, w3, w3, w3, w4], dtype=np.float64) * 0.5
    return w, xy


def triangle_unit_volume() -> float:
    """单位三角形面积 = 1/2。"""
    return 0.5


def transform_to_triangle(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """
    将单位三角形上的参考点映射到实际三角形顶点定义的区域。
    r = v0 + (v1 - v0) * xi + (v2 - v0) * eta
    """
    v0, v1, v2 = vertices[0], vertices[1], vertices[2]
    J = np.array([v1 - v0, v2 - v0]).T
    return (v0.reshape(1, -1) + points @ J.T)


def integrate_triangle(f, vertices: np.ndarray, rule: str = "o07") -> float:
    """
    在由 vertices (3×2 或 3×3) 定义的三角形上求 ∫ f(r) dA。
    """
    if rule == "o01":
        w, xy = triangle_unit_o01()
    elif rule == "o03":
        w, xy = triangle_unit_o03()
    elif rule == "o07":
        w, xy = triangle_unit_o07()
    else:
        w, xy = triangle_unit_o07()

    pts = transform_to_triangle(xy, vertices)
    vals = np.array([f(p) for p in pts], dtype=np.float64)
    # 雅可比行列式 = 2 * 面积(单位三角形)
    v0, v1, v2 = vertices[:3]
    jac = np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    return float(np.sum(w * vals) * jac)


# ===================================================================
# 2. Newton-Cotes Open 一维求积 (源自 line_nco_rule)
# ===================================================================

def line_nco_rule(n: int, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Newton-Cotes Open 规则：在 (a, b) 内取 n 个等距内点，
    构造 Lagrange 基多项式并解析积分得到权重。

    数学推导:
      节点 x_i = a + i * h, 其中 h = (b-a)/(n+1), i = 1..n
      权重 w_j = ∫_a^b ℓ_j(x) dx,
      ℓ_j(x) = Π_{m≠j} (x - x_m) / (x_j - x_m)

    这里通过直接求解 Vandermonde 线性系统得到权重:
      V_{ij} = x_j^{i},  i=0..n-1
      解 V^T w = ∫_a^b x^i dx = (b^{i+1} - a^{i+1})/(i+1)
    """
    if n <= 0:
        return np.array([]), np.array([])
    h = (b - a) / (n + 1)
    x = np.array([a + i * h for i in range(1, n + 1)], dtype=np.float64)
    # 右端项
    rhs = np.zeros(n, dtype=np.float64)
    for i in range(n):
        power = i
        rhs[i] = (b ** (power + 1) - a ** (power + 1)) / (power + 1)
    # Vandermonde
    V = np.vander(x, N=n, increasing=True)
    w = np.linalg.solve(V.T, rhs)
    return x, w


def integrate_line(f, a: float, b: float, n: int = 5) -> float:
    """
    用 Newton-Cotes Open 规则计算 ∫_a^b f(x) dx。
    """
    x, w = line_nco_rule(n, a, b)
    if len(x) == 0:
        return 0.0
    vals = np.array([f(xi) for xi in x], dtype=np.float64)
    return float(np.dot(w, vals))


# ===================================================================
# 3. 分子应用：表面积分与键积分
# ===================================================================

def gaussian_basis_2d(r: np.ndarray, center: np.ndarray, alpha: float) -> float:
    """
    2D 高斯基函数: φ(r) = exp(-α |r - center|²)。
    """
    d = r - center
    return np.exp(-alpha * np.dot(d, d))


def compute_molecular_surface_integral(atoms: np.ndarray,
                                       alpha: float = 1.0,
                                       rule: str = "o07") -> float:
    """
    构造分子凸包表面的三角形网格（简化版：每三个相邻原子构成近似面元），
    并在表面三角形上积分高斯包络。
    用于估算分子的表面积相关描述符。
    """
    n = atoms.shape[0]
    if n < 3:
        return 0.0
    total = 0.0
    # 简化：取前 min(n, 12) 个原子，构造 Delaunay-like 三角形
    count = min(n, 12)
    for i in range(count):
        for j in range(i + 1, count):
            for k in range(j + 1, count):
                verts = atoms[[i, j, k]]
                # 跳过退化的三角形
                area = 0.5 * np.linalg.norm(np.cross(verts[1] - verts[0], verts[2] - verts[0]))
                if area < 1e-6:
                    continue
                center = np.mean(verts, axis=0)
                val = integrate_triangle(lambda r: gaussian_basis_2d(r, center, alpha), verts, rule)
                total += val
    return float(total)


def compute_bond_path_integral(atoms: np.ndarray, bond: Tuple[int, int],
                               potential_func, n_points: int = 5) -> float:
    """
    沿化学键路径积分势能函数 V(s)。
    s ∈ [0, 1] 为键长参数化，r(s) = r_a + s (r_b - r_a)。
    """
    a, b = bond
    r_a = atoms[a]
    r_b = atoms[b]
    def path_func(s):
        r = r_a + s * (r_b - r_a)
        return potential_func(r)
    # 变量替换 ds, 积分区间 [0, 1]
    return integrate_line(path_func, 0.0, 1.0, n_points)
