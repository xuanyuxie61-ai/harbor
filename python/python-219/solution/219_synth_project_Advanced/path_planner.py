"""
path_planner.py
===============

TSP 路径规划与多目标排序模块。

融合种子项目:
  - 1366_tsp_moler : Cleve Moler 的 TSP 求解器 (模拟退火 / 2-opt)

在最优控制中, 该模块用于:
  1. 多目标访问顺序优化 (如多颗卫星的变轨顺序)
  2. 多阶段任务的阶段排序
  3. 为分支定界提供下界估计

数学公式:
---------
1. TSP 目标:
     min_{permutation p} sum_{i=1}^{n} D[p_i, p_{i+1}]
     其中 p_{n+1} = p_1 (回路)

2. 2-opt 邻域:
     交换路径中两条边 (i, i+1) 和 (j, j+1),
     新路径: ... -> p_i -> p_j -> p_{j-1} -> ... -> p_{i+1} -> p_{j+1} -> ...
     代价变化: Delta = D[i,j] + D[i+1,j+1] - D[i,i+1] - D[j,j+1]

3. 模拟退火接受准则:
     若 Delta < 0, 接受;
     否则以概率 exp(-Delta / T) 接受, T 为温度.

4. 冷却计划:
     T_{k+1} = alpha * T_k,  alpha in (0, 1)
"""

from __future__ import annotations

import math
from typing import List, Tuple


# ===========================================================================
# 1. 路径长度计算 (来自 1366_tsp_moler/path_length)
# ===========================================================================
def path_length(path: List[int], D: List[List[float]]) -> float:
    """计算路径总长度 (含回路).

    Parameters
    ----------
    path : List[int]
        城市访问顺序 (0-based 索引).
    D : List[List[float]]
        n x n 距离矩阵.

    Returns
    -------
    float
        路径总长度.
    """
    n = len(path)
    total = 0.0
    for i in range(n):
        j = (i + 1) % n
        total += D[path[i]][path[j]]
    return total


# ===========================================================================
# 2. 距离矩阵构造 (来自 1366_tsp_moler/distances)
# ===========================================================================
def euclidean_distance_matrix(
    coords: List[Tuple[float, float]]
) -> List[List[float]]:
    """构造 Euclidean 距离矩阵.

    Parameters
    ----------
    coords : List[(float, float)]
        n 个城市的 (x, y) 坐标.

    Returns
    -------
    List[List[float]]
        n x n 距离矩阵.
    """
    n = len(coords)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt(
                (coords[i][0] - coords[j][0]) ** 2
                + (coords[i][1] - coords[j][1]) ** 2
            )
            D[i][j] = d
            D[j][i] = d
    return D


# ===========================================================================
# 3. 2-opt 局部搜索 (来自 1366_tsp_moler/traveler)
# ===========================================================================
def two_opt_improve(
    path: List[int], D: List[List[float]], max_iterations: int = 1000
) -> Tuple[List[int], float]:
    """2-opt 局部搜索改进.

    算法:
        重复直到无法改进或达到最大迭代:
            对每对 (i, j) with i < j:
                计算 2-opt 交换的代价变化 Delta
                若 Delta < 0, 执行交换并更新路径

    Parameters
    ----------
    path : List[int]
        初始路径.
    D : List[List[float]]
        距离矩阵.
    max_iterations : int
        最大迭代次数.

    Returns
    -------
    (improved_path, improved_length) : (List[int], float)
    """
    n = len(path)
    current_path = list(path)
    current_length = path_length(current_path, D)

    for _ in range(max_iterations):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue  # 跳过回路边

                # 2-opt 交换: 反转 path[i+1 : j+1]
                # 代价变化: Delta = D[i, j] + D[i+1, j+1] - D[i, i+1] - D[j, j+1]
                # 注意索引循环
                a, b = current_path[i], current_path[(i + 1) % n]
                c, d = current_path[j], current_path[(j + 1) % n]
                delta = D[a][c] + D[b][d] - D[a][b] - D[c][d]

                if delta < -1e-10:
                    # 执行反转
                    current_path[i + 1 : j + 1] = reversed(
                        current_path[i + 1 : j + 1]
                    )
                    current_length += delta
                    improved = True

        if not improved:
            break

    return current_path, current_length


# ===========================================================================
# 4. 模拟退火 TSP (来自 1366_tsp_moler/traveler 思想)
# ===========================================================================
def simulated_annealing_tsp(
    D: List[List[float]],
    initial_temp: float = 100.0,
    cooling_rate: float = 0.95,
    max_iterations: int = 1000,
    seed: int = 12345,
) -> Tuple[List[int], float]:
    """模拟退火 TSP 求解.

    算法:
        1. 随机初始路径
        2. 对每次迭代:
            a. 随机选择 2-opt 交换
            b. 计算 Delta
            c. 若 Delta < 0 或 exp(-Delta / T) > U, 接受
            d. T <- alpha * T
        3. 返回最优路径

    Parameters
    ----------
    D : List[List[float]]
        n x n 距离矩阵.
    initial_temp : float
        初始温度.
    cooling_rate : float
        冷却率 alpha.
    max_iterations : int
        最大迭代次数.
    seed : int
        伪随机种子.

    Returns
    -------
    (best_path, best_length) : (List[int], float)
    """
    n = len(D)
    if n <= 1:
        return list(range(n)), 0.0

    # 确定性伪随机初始路径
    state = seed
    path = list(range(n))
    for i in range(n - 1, 0, -1):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        j = state % (i + 1)
        path[i], path[j] = path[j], path[i]

    current_length = path_length(path, D)
    best_path = list(path)
    best_length = current_length

    T = initial_temp
    state = seed + 1

    for _ in range(max_iterations):
        # 随机 2-opt 交换
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        i = state % (n - 1)
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        j = i + 2 + state % (n - i - 1)
        if j >= n:
            j = n - 1
        if i == 0 and j == n - 1:
            continue

        a, b = path[i], path[(i + 1) % n]
        c, d = path[j], path[(j + 1) % n]
        delta = D[a][c] + D[b][d] - D[a][b] - D[c][d]

        # Metropolis 准则
        if delta < 0:
            accept = True
        else:
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            u = state / 0x7FFFFFFF
            accept = u < math.exp(-delta / T)

        if accept:
            path[i + 1 : j + 1] = reversed(path[i + 1 : j + 1])
            current_length += delta
            if current_length < best_length:
                best_path = list(path)
                best_length = current_length

        T *= cooling_rate

    return best_path, best_length


# ===========================================================================
# 5. 组合求解器
# ===========================================================================
def solve_tsp(
    coords: List[Tuple[float, float]],
    method: str = "hybrid",
) -> Tuple[List[int], float]:
    """TSP 求解器 (统一接口).

    Parameters
    ----------
    coords : List[(float, float)]
        n 个城市的坐标.
    method : str
        "2opt", "sa", "hybrid".

    Returns
    -------
    (best_path, best_length) : (List[int], float)
    """
    if method not in ("2opt", "sa", "hybrid"):
        raise ValueError(f"未知方法: {method}")

    D = euclidean_distance_matrix(coords)
    n = len(coords)
    if n <= 1:
        return list(range(n)), 0.0

    # 初始路径 (顺序)
    initial_path = list(range(n))

    if method == "2opt":
        return two_opt_improve(initial_path, D)
    elif method == "sa":
        return simulated_annealing_tsp(D)
    else:  # hybrid
        # 先 SA, 再 2-opt 精化
        sa_path, sa_len = simulated_annealing_tsp(D)
        return two_opt_improve(sa_path, D)


# ===========================================================================
# 自检
# ===========================================================================
def self_check() -> None:
    """路径规划模块自检."""
    print("[TSP] 4 个城市 (正方形):")
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    path, length = solve_tsp(coords, method="hybrid")
    print(f"  路径: {path}")
    print(f"  长度: {length:.4f}")

    print("[TSP] 6 个城市 (随机):")
    coords2 = [
        (0.0, 0.0),
        (1.0, 0.5),
        (2.0, 0.0),
        (2.0, 1.5),
        (1.0, 2.0),
        (0.0, 1.5),
    ]
    path2, length2 = solve_tsp(coords2, method="hybrid")
    print(f"  路径: {path2}")
    print(f"  长度: {length2:.4f}")


if __name__ == "__main__":
    self_check()
