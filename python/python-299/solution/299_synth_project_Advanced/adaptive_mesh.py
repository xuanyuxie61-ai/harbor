"""
adaptive_mesh.py — 自适应网格加密 (有理背包算法)
================================================

种子项目映射: 627_knapsack_rational — 有理背包问题.

原项目:
  给定 N 个物品, 每个有 profit P(i) 和 weight W(i),
  在总重量 ≤ MASS_LIMIT 的约束下最大化总利润.
  有理版本允许 0 ≤ x(i) ≤ 1 的分数选择.
  贪心算法: 按 P/W 降序排列, 贪心选择直到装满.

本项目映射:
  物品 → 速度空间网格单元
  profit → 该单元的误差减小量 (基于局部截断误差估计)
  weight → 该单元的计算代价 (∝ 单元尺寸)
  mass_limit → 总计算预算 (最大网格点数)
  x(i) → 加密分数 (0=不加密, 1=完全加密, 中间=部分加密)

物理动机:
  Fokker-Planck 方程的解在某些速度区域变化剧烈
  (如热尾部, 束流区), 需要加密网格来解析.
  自适应加密可以在固定计算预算下最大化整体精度.
"""

import numpy as np


# ===========================================================================
#  §1  有理背包求解器  (源自 knapsack_rational)
# ===========================================================================
def knapsack_rational(n, mass_limit, profits, weights):
    """有理背包问题的贪心求解.

    映射自 knapsack_rational.m:
      for i = 1:n
        if mass_limit ≤ mass → x(i) = 0
        elif mass + w(i) ≤ mass_limit → x(i) = 1, mass += w(i)
        else → x(i) = (mass_limit - mass) / w(i), mass = mass_limit

    Parameters
    ----------
    n : int  物品数量
    mass_limit : float  重量上限
    profits : ndarray(n)  利润 (必须非负)
    weights : ndarray(n)  重量 (必须非负)

    预排序: 物品已按 profit_density = P/W 降序排列.

    Returns
    -------
    x : ndarray(n)  选择向量 (0 ≤ x ≤ 1)
    mass : float  总重量
    profit : float  总利润
    """
    x = np.zeros(n)
    mass = 0.0
    profit = 0.0

    for i in range(n):
        if mass_limit <= mass:
            x[i] = 0.0
        elif mass + weights[i] <= mass_limit:
            x[i] = 1.0
            mass += weights[i]
            profit += profits[i]
        else:
            x[i] = (mass_limit - mass) / max(weights[i], 1e-30)
            mass = mass_limit
            profit += profits[i] * x[i]

    return x, mass, profit


# ===========================================================================
#  §2  误差指示子
# ===========================================================================
def compute_error_indicator(x, f, dx=None):
    """基于局部截断误差的网格加密指示子.

    使用二阶导数作为误差指示:
      η_i = |f''(x_i)| · Δx_i²

    物理含义: 分布函数曲率大的区域需要更密的网格.

    对于 Fokker-Planck 方程, 曲率大的区域通常是:
    - 分布函数的峰值附近
    - 热尾部的过渡区
    - 束流/双峰结构的中间区域

    Parameters
    ----------
    x : ndarray  速度网格
    f : ndarray  分布函数
    dx : ndarray or None  局部网格间距

    Returns
    -------
    eta : ndarray  误差指示子
    """
    N = len(x)
    if dx is None:
        dx_arr = np.diff(x)
        dx = np.zeros(N)
        dx[1:-1] = 0.5 * (dx_arr[:-1] + dx_arr[1:])
        dx[0] = dx_arr[0]
        dx[-1] = dx_arr[-1]

    # 二阶导数 (中心差分)
    d2f = np.zeros(N)
    for i in range(1, N - 1):
        dx_left = x[i] - x[i-1]
        dx_right = x[i+1] - x[i]
        dx_avg = 0.5 * (dx_left + dx_right)
        d2f[i] = 2.0 * (
            f[i+1] / (dx_right * (dx_left + dx_right))
            - f[i] / (dx_left * dx_right)
            + f[i-1] / (dx_left * (dx_left + dx_right))
        )

    # 边界外推
    d2f[0] = d2f[1]
    d2f[-1] = d2f[-2]

    # 误差指示子: η = |f''| · Δx²
    eta = np.abs(d2f) * dx**2

    return eta


def compute_gradient_indicator(x, f):
    """基于梯度的加密指示子: η = |f'(x)| · Δx.

    适用于捕捉分布函数的间断或急剧变化.
    """
    N = len(x)
    dx_arr = np.diff(x)
    df = np.zeros(N)
    for i in range(N - 1):
        df[i] = (f[i+1] - f[i]) / max(dx_arr[i], 1e-30)
    df[-1] = df[-2]

    dx = np.zeros(N)
    dx[:-1] = dx_arr
    dx[-1] = dx_arr[-1]

    return np.abs(df) * dx


# ===========================================================================
#  §3  自适应网格优化
# ===========================================================================
def optimize_mesh_allocation(x, f, budget_factor=2.0):
    """使用有理背包算法优化网格加密.

    对每个单元, 定义:
      profit_i = η_i (误差减小量)
      weight_i = Δx_i / min(Δx) (计算代价, 以最小单元为基准)
      mass_limit = N_current * budget_factor (总预算)

    返回每个单元的加密分数 x_i ∈ [0, 1].
    x_i = 1 → 该单元需要完全加密 (一分为二)
    x_i = 0 → 该单元不需要加密

    Parameters
    ----------
    x : ndarray  当前速度网格
    f : ndarray  当前分布函数
    budget_factor : float  预算放大因子 (> 1 表示允许加密)

    Returns
    -------
    refinement : ndarray  加密分数
    new_grid : ndarray  新的速度网格 (已加密)
    info : dict  诊断信息
    """
    N = len(x)
    dx_arr = np.diff(x)

    # 误差指示子
    eta = compute_error_indicator(x, f)

    # 每个单元的 "利润" = 误差 (加密后误差减半)
    profits = eta[:-1] * dx_arr  # 加权误差

    # 每个单元的 "重量" = 计算代价
    min_dx = np.min(dx_arr)
    weights = dx_arr / max(min_dx, 1e-30)

    # 按 profit density 降序排列
    density = profits / np.maximum(weights, 1e-30)
    order = np.argsort(-density)

    # 背包约束
    mass_limit = N * budget_factor

    # 求解有理背包
    x_knap, total_mass, total_profit = knapsack_rational(
        N - 1, mass_limit, profits[order], weights[order]
    )

    # 还原顺序
    refinement = np.zeros(N - 1)
    for i, idx in enumerate(order):
        refinement[idx] = x_knap[i]

    # 生成新网格
    new_points = []
    for i in range(N - 1):
        new_points.append(x[i])
        if refinement[i] > 0.5:
            # 在该单元中点添加新点
            mid = 0.5 * (x[i] + x[i+1])
            new_points.append(mid)
    new_points.append(x[-1])
    new_grid = np.array(sorted(new_points))

    info = {
        "n_original": N,
        "n_new": len(new_grid),
        "n_refined": int(np.sum(refinement > 0.5)),
        "total_profit": total_profit,
        "total_mass": total_mass,
        "budget_limit": mass_limit,
        "max_eta": np.max(eta),
        "mean_eta": np.mean(eta),
    }

    return refinement, new_grid, info


# ===========================================================================
#  §4  网格质量评估
# ===========================================================================
def mesh_quality_metrics(x):
    """评估网格质量.

    Returns
    -------
    metrics : dict
    """
    dx = np.diff(x)
    N = len(x)
    return {
        "n_points": N,
        "min_dx": np.min(dx),
        "max_dx": np.max(dx),
        "mean_dx": np.mean(dx),
        "std_dx": np.std(dx),
        "stretch_ratio": np.max(dx) / max(np.min(dx), 1e-30),
        "uniformity": 1.0 - np.std(dx) / max(np.mean(dx), 1e-30),
    }
