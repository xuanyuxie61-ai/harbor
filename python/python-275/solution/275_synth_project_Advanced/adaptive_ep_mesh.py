"""
adaptive_ep_mesh.py
===================
自适应网格细化以精确分辨例外点邻域.
融合种子项目:
  - 1037_nkoch1_LPFC_adaptation: LPFC 模型的自适应网格
  - 254_cvt_circle_uniform: CVT 自适应采样

物理动机:
  EP 附近物理量剧烈变化 (本征值平方根分支、本征矢 coalescence).
  均匀网格无法高效分辨 EP 邻域结构.
  自适应策略: 根据 |Δ(k)| 或条件数梯度加密网格.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, List, Dict, Callable
import nonhermitian_hamiltonian as nh


# ---------------------------------------------------------------------------
# 自适应 1D 网格 (源自 LPFC adaptation)
# ---------------------------------------------------------------------------
def adaptive_refinement_1d(indicator_func: Callable,
                           x_min: float, x_max: float,
                           tol: float = 1e-4,
                           max_levels: int = 8,
                           min_spacing: float = 1e-8) -> np.ndarray:
    """基于指示函数的自适应 1D 网格细化.

    算法:
      1. 初始均匀网格.
      2. 在每个区间, 计算指示函数的变化 (或二阶差分).
      3. 若变化 > tol, 将该区间二等分.
      4. 重复直到所有区间变化 < tol 或达到最大层数.

    indicator_func : x -> float, 如 |Δ(x)| 或条件数.

    Returns
    -------
    x_refined : 自适应加密后的网格点.
    """
    x = np.linspace(x_min, x_max, 17)
    for level in range(max_levels):
        if len(x) > 2000:
            break
        new_x = [x[0]]
        refined = False
        for i in range(len(x) - 1):
            x_left, x_right = x[i], x[i + 1]
            dx = x_right - x_left
            if dx < min_spacing:
                new_x.append(x_right)
                continue
            f_left = indicator_func(x_left)
            f_right = indicator_func(x_right)
            f_mid = indicator_func(0.5 * (x_left + x_right))
            variation = abs(f_left - 2 * f_mid + f_right)
            if variation > tol:
                new_x.append(0.5 * (x_left + x_right))
                refined = True
            new_x.append(x_right)
        x = np.array(sorted(set(new_x)))
        if not refined:
            break
    return x


def adaptive_refinement_2d(indicator_func: Callable,
                           bounds: List[Tuple[float, float]],
                           tol: float = 1e-3,
                           max_levels: int = 5,
                           initial_n: int = 10) -> np.ndarray:
    """二维自适应网格 (quadtree 风格).

    Returns
    -------
    points : (N, 2) 自适应点集.
    """
    points = []
    cells = [(bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1], 0)]
    while cells:
        x0, x1, y0, y1, level = cells.pop(0)
        if level >= max_levels:
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            points.append([cx, cy])
            continue
        f_corners = [indicator_func(x0, y0), indicator_func(x1, y0),
                     indicator_func(x0, y1), indicator_func(x1, y1)]
        variation = max(f_corners) - min(f_corners)
        if variation > tol:
            cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            cells.append((x0, cx, y0, cy, level + 1))
            cells.append((cx, x1, y0, cy, level + 1))
            cells.append((x0, cx, cy, y1, level + 1))
            cells.append((cx, x1, cy, y1, level + 1))
        else:
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            points.append([cx, cy])
    return np.array(points) if points else np.zeros((0, 2))


# ---------------------------------------------------------------------------
# 基于 EP 指示函数的自适应
# ---------------------------------------------------------------------------
def ep_indicator_1d(k: float, t1: float, t2: float, gamma: float) -> float:
    """EP 指示函数: |Δ(k)| (小值表示接近 EP)."""
    H = nh.ssh_hamiltonian_1d(t1, t2, gamma, k)
    return float(abs(nh.discriminant_ep_indicator(H)))


def build_ep_adaptive_k_grid(t1: float, t2: float, gamma: float,
                             tol: float = 1e-4,
                             max_levels: int = 8) -> np.ndarray:
    """在 k 空间构造自适应网格, 在 EP 附近加密.

    Returns
    -------
    k_grid : 自适应 k 网格.
    """
    def indicator(k):
        return ep_indicator_1d(k, t1, t2, gamma)
    return adaptive_refinement_1d(indicator, 0, 2 * np.pi,
                                  tol=tol, max_levels=max_levels)


def ep_localization_error(k_grid: np.ndarray,
                          t1: float, t2: float,
                          gamma: float) -> Dict:
    """评估自适应网格对 EP 的分辨精度.

    计算:
      - min_delta: 网格上最小的 |Δ(k)|.
      - k_at_min: 达到最小的 k 值.
      - spacing_at_min: 该处的局部网格间距.

    Returns
    -------
    info : dict.
    """
    deltas = np.zeros(len(k_grid))
    for i, k in enumerate(k_grid):
        deltas[i] = ep_indicator_1d(k, t1, t2, gamma)
    idx_min = np.argmin(deltas)
    k_min = k_grid[idx_min]
    if idx_min > 0 and idx_min < len(k_grid) - 1:
        spacing = 0.5 * (k_grid[idx_min + 1] - k_grid[idx_min - 1])
    elif idx_min == 0:
        spacing = k_grid[1] - k_grid[0] if len(k_grid) > 1 else 0.0
    else:
        spacing = k_grid[-1] - k_grid[-2] if len(k_grid) > 1 else 0.0
    return {
        "min_delta": float(deltas[idx_min]),
        "k_at_min": float(k_min),
        "spacing_at_min": float(spacing),
        "n_grid_points": len(k_grid),
    }


# ---------------------------------------------------------------------------
# 基于条件数的自适应 (用于追踪本征矢 coalescence)
# ---------------------------------------------------------------------------
def condition_number_indicator(k: float, t1: float, t2: float,
                               gamma: float) -> float:
    """哈密顿量条件数 κ(H(k)).

    在 EP 处, κ → ∞ (矩阵亏损).
    """
    H = nh.ssh_hamiltonian_1d(t1, t2, gamma, k)
    return float(np.linalg.cond(H))


def adaptive_condition_mesh(t1: float, t2: float, gamma: float,
                            tol: float = 0.5,
                            max_levels: int = 6) -> np.ndarray:
    """基于条件数梯度的自适应 k 网格."""
    def indicator(k):
        try:
            return np.log10(max(condition_number_indicator(k, t1, t2, gamma), 1.0))
        except Exception:
            return 0.0
    return adaptive_refinement_1d(indicator, 0, 2 * np.pi,
                                  tol=tol, max_levels=max_levels)
