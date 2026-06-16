"""
pdf_interp.py — PDF 分段线性插值与 Delaunay 搜索 (映射自 pwl_interp_2d_scattered)
=============================================================
本模块实现 PDF 在非均匀 x 网格上的分段线性插值:
  (1) Delaunay 三角剖分搜索 (1D 版本 = 区间搜索);
  (2) 分段线性 (PWL) 插值;
  (3) 2D (x, Q²) 双线性插值;
  (4) 导数计算 (分段线性导数为分段常数).

映射自 pwl_interp_2d_scattered:
  原始代码: 在 2D 散乱点集上进行 Delaunay 三角剖分 + PWL 插值
  本模块:   在 1D x 网格 (或 2D (x, Q²) 网格) 上进行类似操作

核心公式 (1D PWL 插值):
    f(x) = α f(x_L) + β f(x_R)
    α = (x_R − x) / (x_R − x_L)
    β = (x − x_L) / (x_R − x_L)
    α + β = 1

核心公式 (2D 双线性插值):
    f(x, q) = Σ_{i,j} w_{ij} f(x_i, q_j)
    w_{ij} = α_i × β_j

核心公式 (Delaunay 搜索, 1D 版本):
    在排序的 x 网格上, 使用二分查找定位 x 所在区间:
    找到 k 使 x_k ≤ x < x_{k+1}
    时间复杂度: O(log N)

核心公式 (插值误差界):
    |f(x) − f_h(x)| ≤ h²/8 × max |f''(x)|
    其中 h 为最大网格间距.
"""
from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional

from phys_consts import EPS_NUMERICAL
from dglap_evolution import DGLAPEvolutionResult


# ============================================================
# 1. 1D Delaunay 区间搜索
# ============================================================
def delaunay_search_1d(x_grid: List[float], x_query: float
                       ) -> Tuple[int, float, float]:
    """
    在排序的 x_grid 上搜索 x_query 所在区间 (映射自 triangulation_search_delaunay):

    1D Delaunay "三角剖分" = 区间分割.
    找到索引 k 使 x_grid[k] ≤ x_query < x_grid[k+1].

    使用二分搜索 (O(log N)).

    参数:
        x_grid:   排序的网格点
        x_query:  查询点
    返回:
        (k, alpha, beta): 区间索引, 左权重, 右权重
        alpha = (x_R − x) / (x_R − x_L)
        beta  = (x − x_L) / (x_R − x_L)
    """
    n = len(x_grid)
    if n < 2:
        return 0, 1.0, 0.0

    # 边界处理
    if x_query <= x_grid[0]:
        return 0, 1.0, 0.0
    if x_query >= x_grid[-1]:
        return n - 2, 0.0, 1.0

    # 二分搜索
    lo, hi = 0, n - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if x_grid[mid] <= x_query:
            lo = mid
        else:
            hi = mid

    k = lo
    x_L = x_grid[k]
    x_R = x_grid[k + 1]
    dx = x_R - x_L
    if dx < EPS_NUMERICAL:
        return k, 0.5, 0.5
    alpha = (x_R - x_query) / dx
    beta = (x_query - x_L) / dx
    return k, max(0.0, alpha), max(0.0, beta)


# ============================================================
# 2. 1D PWL 插值
# ============================================================
def pwl_interp_1d(x_grid: List[float], f_values: List[float],
                  x_query: float) -> float:
    """
    1D 分段线性插值 (映射自 pwl_interp_2d_scattered_value):

    f(x) = α f(x_L) + β f(x_R)

    参数:
        x_grid:   排序的网格点
        f_values: 网格点上的函数值
        x_query:  查询点
    返回:
        插值值
    """
    k, alpha, beta = delaunay_search_1d(x_grid, x_query)
    n = len(f_values)
    k_R = min(k + 1, n - 1)
    return alpha * f_values[k] + beta * f_values[k_R]


def pwl_interp_1d_batch(x_grid: List[float], f_values: List[float],
                        x_queries: List[float]) -> List[float]:
    """批量 1D PWL 插值"""
    return [pwl_interp_1d(x_grid, f_values, x) for x in x_queries]


# ============================================================
# 3. 2D 双线性插值 (x, Q²)
# ============================================================
def pwl_interp_2d(
    x_grid: List[float], q2_grid: List[float],
    f_matrix: List[List[float]],
    x_query: float, q2_query: float
) -> float:
    """
    2D 双线性插值 (映射自 pwl_interp_2d_scattered_value):

    f(x, Q²) = Σ_{i,j} α_i β_j f(x_i, Q²_j)

    参数:
        x_grid:   x 网格
        q2_grid:  Q² 网格
        f_matrix: f[iq][ix] = f(x_ix, Q²_iq)
        x_query:  查询 x
        q2_query: 查询 Q²
    返回:
        插值值
    """
    # x 方向搜索
    kx, ax, bx = delaunay_search_1d(x_grid, x_query)
    # Q² 方向搜索
    kq, aq, bq = delaunay_search_1d(q2_grid, q2_query)

    nq = len(f_matrix)
    nx = len(x_grid) if x_grid else 0
    kx_R = min(kx + 1, nx - 1)
    kq_R = min(kq + 1, nq - 1)

    # 四角插值
    f_LL = f_matrix[kq][kx] if kq < nq and kx < nx else 0.0
    f_LR = f_matrix[kq][kx_R] if kq < nq and kx_R < nx else 0.0
    f_UL = f_matrix[kq_R][kx] if kq_R < nq and kx < nx else 0.0
    f_UU = f_matrix[kq_R][kx_R] if kq_R < nq and kx_R < nx else 0.0

    return (aq * ax * f_LL + aq * bx * f_LR
            + bq * ax * f_UL + bq * bx * f_UU)


# ============================================================
# 4. PDF 插值包装器
# ============================================================
def interpolate_pdf_at_point(
    evo_result: DGLAPEvolutionResult, x: float, q2: float,
    flavor: str = 'g'
) -> float:
    """
    在任意 (x, Q²) 点插值 PDF.

    参数:
        evo_result: DGLAP 演化结果
        x:          x 值
        q2:         Q² 值
        flavor:     'uv', 'dv', 'g', 's'
    返回:
        x f(x, Q²) 的插值
    """
    flavor_idx = {'uv': 0, 'dv': 1, 'g': 2, 's': 3}.get(flavor, 2)
    nq = len(evo_result.q2_grid)
    nx = len(evo_result.x_grid)

    # 构建 flavor 在 (x, Q²) 网格上的矩阵
    f_matrix = []
    for iq in range(nq):
        if iq in evo_result.pdf_history:
            f_vals = evo_result.pdf_history[iq][flavor_idx]
            f_matrix.append(list(f_vals))
        else:
            f_matrix.append([0.0] * nx)

    return pwl_interp_2d(evo_result.x_grid, evo_result.q2_grid,
                         f_matrix, x, q2)


def compute_xf_derivative(
    evo_result: DGLAPEvolutionResult, x: float, q2: float,
    flavor: str = 'g', direction: str = 'x'
) -> float:
    """
    PDF 的导数 (通过有限差分):
        ∂(xf)/∂x ≈ [xf(x+h) − xf(x−h)] / (2h)
        ∂(xf)/∂ln(Q²) ≈ [xf(Q²+h) − xf(Q²−h)] / (2h)
    """
    if direction == 'x':
        h = max(x * 0.01, 1e-6)
        fp = interpolate_pdf_at_point(evo_result, x + h, q2, flavor)
        fm = interpolate_pdf_at_point(evo_result, x - h, q2, flavor)
        return (fp - fm) / (2.0 * h)
    elif direction == 'logq2':
        h = max(math.log(q2) * 0.01, 0.01)
        fp = interpolate_pdf_at_point(evo_result, x, q2 * math.exp(h), flavor)
        fm = interpolate_pdf_at_point(evo_result, x, q2 * math.exp(-h), flavor)
        return (fp - fm) / (2.0 * h)
    else:
        return 0.0


# ============================================================
# 5. 插值误差估计
# ============================================================
def interpolation_error_estimate(
    x_grid: List[float], f_values: List[float],
    x_query: float, f_second_deriv_max: float = 1.0
) -> float:
    """
    PWL 插值误差界:
        |f(x) − f_h(x)| ≤ h²/8 × max|f''|

    其中 h = x_R − x_L 为当前区间宽度.
    """
    k, _, _ = delaunay_search_1d(x_grid, x_query)
    n = len(x_grid)
    k_R = min(k + 1, n - 1)
    h = x_grid[k_R] - x_grid[k]
    return h * h / 8.0 * abs(f_second_deriv_max)
