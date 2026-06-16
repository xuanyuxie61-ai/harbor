"""
global_search.py
================
全局搜索方法 —— 排列枚举、蒙特卡罗采样与多起始点策略

融合种子项目:
  - 1363_tsp_brute: 全排列枚举, 路径代价评估
  - 234_cube_integrals: 单位立方体采样, 单调积分

核心公式:
  1. 排列枚举: 对 n 个变量的所有 n! 排列评估目标
  2. 拉丁超立方采样: LHS 保证每个维度的均匀覆盖
  3. 多起始点策略: 从 N 个随机起点运行局部优化器, 取最优
  4. 模拟退火: 以概率 exp(-Δf/T) 接受劣解, T 按冷却表降低
  5. 路径代价: C(p) = Σ d(p_i, p_{i+1}) (TSP 型)
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict, List
import math
from itertools import permutations


# ---------------------------------------------------------------------------
# 1. 全排列枚举 (源自 1363_tsp_brute)
# ---------------------------------------------------------------------------

def generate_permutations(n: int):
    """生成 1..n 的所有排列 (源自 1363_tsp_brute).
    源自 perm1_next3 的 Python 等价.
    返回排列迭代器.
    """
    return permutations(range(n))


def path_cost(distance_matrix: np.ndarray, perm: tuple) -> float:
    """计算路径总代价 (源自 1363_tsp_brute/path_cost).
    C(p) = Σ_{i=0}^{n-2} d(p_i, p_{i+1}) + d(p_{n-1}, p_0)

    对 TSP: d 为城市间距离矩阵.
    在优化中: d 可解释为设计点间的"代价距离".
    """
    n = len(perm)
    total = 0.0
    for i in range(n - 1):
        total += distance_matrix[perm[i], perm[i + 1]]
    total += distance_matrix[perm[n - 1], perm[0]]
    return total


def brute_force_tsp(distance_matrix: np.ndarray) -> Tuple[float, tuple]:
    """暴力法求解 TSP (源自 1363_tsp_brute/tsp_brute).
    枚举所有排列, 找最小路径代价.
    复杂度: O(n!), 仅适用于 n ≤ 10.

    返回 (min_cost, best_perm).
    """
    n = distance_matrix.shape[0]
    if n > 10:
        print(f"警告: n={n} > 10, 暴力法可能极慢!")

    total_min = float('inf')
    total_max = -float('inf')
    total_ave = 0.0
    paths = 0
    p_min = None

    for p in generate_permutations(n):
        total = path_cost(distance_matrix, p)
        total_ave += total
        paths += 1
        if total < total_min:
            total_min = total
            p_min = p
        if total > total_max:
            total_max = total

    if paths > 0:
        total_ave /= paths

    return total_min, p_min


def permutation_search_optimizer(f: Callable, dim: int,
                                 n_points: int = 8,
                                 bounds: Tuple[float, float] = (-2, 2)
                                 ) -> Dict:
    """排列搜索优化: 将搜索空间离散化, 枚举所有排列.

    1. 在每个维度生成 n_points 个候选值
    2. 对 n_points 个位置的所有排列, 评估目标函数
    3. 返回最优排列对应的解

    复杂度: O(n_points^dim) 的缩减版 — 只枚举排列.
    适用于 dim 小 (≤ 8) 且 n_points 小的情形.
    """
    # 生成候选点网格
    candidates = np.linspace(bounds[0], bounds[1], n_points)

    # 对每个排列评估
    best_val = float('inf')
    best_x = None

    perms = list(generate_permutations(min(n_points, 8)))
    for perm in perms:
        x = np.array([candidates[p] for p in perm[:dim]])
        val = f(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()

    return {"x": best_x, "f_val": best_val,
            "n_evaluations": len(perms), "method": "permutation_search"}


# ---------------------------------------------------------------------------
# 2. 拉丁超立方采样 (LHS)
# ---------------------------------------------------------------------------

def latin_hypercube_sample(n_samples: int, dim: int,
                           bounds: Optional[np.ndarray] = None,
                           rng: Optional[np.random.Generator] = None
                           ) -> np.ndarray:
    """拉丁超立方采样 (LHS).
    保证每个维度的每个区间 [i/n, (i+1)/n] 至少有一个样本.

    算法:
      1. 对每个维度, 将 [0,1] 分为 n_samples 个等概率区间
      2. 每个区间内随机取一个点
      3. 对每个维度独立随机排列

    相比纯随机采样, LHS 在相同样本数下有更低的不确定性.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if bounds is None:
        bounds = np.array([[0.0, 1.0]] * dim)

    result = np.zeros((n_samples, dim))
    for d in range(dim):
        # 在每个区间内随机采样
        cuts = np.linspace(0, 1, n_samples + 1)
        samples = rng.uniform(cuts[:-1], cuts[1:])
        # 随机排列
        rng.shuffle(samples)
        # 映射到实际范围
        result[:, d] = bounds[d, 0] + samples * (bounds[d, 1] - bounds[d, 0])

    return result


# ---------------------------------------------------------------------------
# 3. 多起始点局部优化
# ---------------------------------------------------------------------------

def multi_start_optimization(f: Callable, grad_f: Callable,
                             dim: int,
                             n_starts: int = 20,
                             bounds: Tuple[float, float] = (-3, 3),
                             local_optimizer=None,
                             tol: float = 1e-8,
                             rng: Optional[np.random.Generator] = None
                             ) -> Dict:
    """多起始点优化策略.

    1. 生成 n_starts 个随机起始点 (使用 LHS)
    2. 从每个起始点运行局部优化器
    3. 返回全局最优解

    对多模态函数, 这是找到全局最优的简单有效方法.
    复杂度: O(n_starts × local_cost).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if local_optimizer is None:
        from quasi_newton import bfgs
        local_optimizer = bfgs

    bounds_arr = np.array([[bounds[0], bounds[1]]] * dim)
    starts = latin_hypercube_sample(n_starts, dim, bounds_arr, rng)

    results = []
    for i in range(n_starts):
        x0 = starts[i]
        res = local_optimizer(f, grad_f, x0, tol=tol, max_iter=500)
        results.append({
            "start_idx": i,
            "x0": x0.copy(),
            "x": res["x"].copy(),
            "f_val": res["f_val"],
            "grad_norm": res["grad_norm"],
            "converged": res["converged"],
            "iterations": res["iterations"]
        })

    # 找最优
    best_idx = min(range(len(results)), key=lambda i: results[i]["f_val"])
    best = results[best_idx]

    return {
        "best_x": best["x"],
        "best_f_val": best["f_val"],
        "best_start_idx": best_idx,
        "all_results": results,
        "n_starts": n_starts,
        "success_rate": sum(1 for r in results if r["converged"]) / n_starts
    }


# ---------------------------------------------------------------------------
# 4. 模拟退火
# ---------------------------------------------------------------------------

def simulated_annealing(f: Callable, x0: np.ndarray,
                        T_init: float = 100.0,
                        T_min: float = 1e-6,
                        cooling_rate: float = 0.95,
                        n_iter_per_temp: int = 50,
                        step_size: float = 0.5,
                        rng: Optional[np.random.Generator] = None
                        ) -> Dict:
    """模拟退火 (Kirkpatrick et al., 1983).

    算法:
      1. 初始化 x = x0, T = T_init
      2. 对每个温度:
         a. 生成候选: x' = x + N(0, step_size²)
         b. 若 f(x') < f(x): 接受
         c. 否则以概率 exp(-(f(x')-f(x))/T) 接受
      3. T ← cooling_rate · T

    收敛定理: 若冷却足够慢, 以概率 1 收敛到全局最优.
    实际中冷却率通常取 0.85-0.99.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    x = x0.copy().astype(float)
    fx = f(x)
    x_best = x.copy()
    fx_best = fx

    T = T_init
    history = []

    while T > T_min:
        for _ in range(n_iter_per_temp):
            # 生成候选
            x_new = x + rng.standard_normal(len(x)) * step_size * T
            fx_new = f(x_new)
            df = fx_new - fx

            # Metropolis 准则
            if df < 0:
                x, fx = x_new, fx_new
            else:
                prob = np.exp(-df / max(T, 1e-300))
                if rng.uniform() < prob:
                    x, fx = x_new, fx_new

            if fx < fx_best:
                x_best = x.copy()
                fx_best = fx

        history.append({"T": T, "f_current": fx, "f_best": fx_best})
        T *= cooling_rate

    return {
        "x": x_best,
        "f_val": fx_best,
        "history": history,
        "n_iterations": len(history) * n_iter_per_temp
    }


# ---------------------------------------------------------------------------
# 5. 随机搜索 + 局部 refinement
# ---------------------------------------------------------------------------

def random_search_with_refinement(f: Callable, grad_f: Callable,
                                  dim: int,
                                  n_random: int = 100,
                                  n_refine: int = 5,
                                  bounds: Tuple[float, float] = (-3, 3),
                                  rng: Optional[np.random.Generator] = None
                                  ) -> Dict:
    """随机搜索 + 局部精化.

    1. 随机采样 n_random 个点
    2. 选最好的 n_refine 个点
    3. 从这些点运行 BFGS 精化

    结合了全局探索与局部开发的优点.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    from quasi_newton import bfgs

    # 随机采样
    samples = rng.uniform(bounds[0], bounds[1], (n_random, dim))
    values = np.array([f(s) for s in samples])

    # 选最好的 n_refine 个
    top_idx = np.argsort(values)[:n_refine]

    results = []
    for idx in top_idx:
        x0 = samples[idx]
        res = bfgs(f, grad_f, x0, tol=1e-10, max_iter=500)
        results.append({
            "start_idx": idx,
            "f_initial": values[idx],
            "x": res["x"].copy(),
            "f_val": res["f_val"],
            "converged": res["converged"]
        })

    best_idx = min(range(len(results)), key=lambda i: results[i]["f_val"])
    return {
        "best_x": results[best_idx]["x"],
        "best_f_val": results[best_idx]["f_val"],
        "all_results": results
    }


# ---------------------------------------------------------------------------
# 6. 网格搜索 (粗到细)
# ---------------------------------------------------------------------------

def grid_search_coarse_to_fine(f: Callable, dim: int,
                               bounds: Tuple[float, float] = (-2, 2),
                               n_coarse: int = 10,
                               n_levels: int = 3,
                               refinement: float = 0.5) -> Dict:
    """粗到细网格搜索.

    1. 在最粗网格上评估所有点
    2. 在最优附近加密网格
    3. 重复 n_levels 次

    每级网格点间距: Δx_k = (b-a) · refinement^k / n_coarse.
    总评估: O(n_coarse^dim · n_levels).
    """
    current_bounds = np.array([[bounds[0], bounds[1]]] * dim)
    best_x = None
    best_val = float('inf')

    for level in range(n_levels):
        # 构造网格
        axes = [np.linspace(current_bounds[d, 0], current_bounds[d, 1], n_coarse)
                for d in range(dim)]
        grid = np.array(np.meshgrid(*axes)).T.reshape(-1, dim)

        # 评估
        for x in grid:
            val = f(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()

        # 缩小搜索范围
        delta = (current_bounds[:, 1] - current_bounds[:, 0]) * refinement
        current_bounds[:, 0] = np.maximum(best_x - delta, bounds[0])
        current_bounds[:, 1] = np.minimum(best_x + delta, bounds[1])

    return {
        "x": best_x,
        "f_val": best_val,
        "n_levels": n_levels,
        "bounds_final": current_bounds.tolist()
    }
