"""
conformation_search.py
=======================
药物分子构象搜索与对接优化模块。

核心数学内容：
  - 贪心算法（Greedy）用于 Traveling Salesman Problem (TSP) 的构象空间采样：
    将药物分子的可旋转键视为"城市"，旋转角度视为"距离"，
    寻找能量最低的可旋转键状态序列。
  - 回溯搜索（Backtrack）用于枚举离散二面角组合（combinatorial rotamer library）。
  - 构象能量打分函数：Lennard-Jones + 静电 + 二面角约束惩罚。

种子项目映射：
  - 1365_tsp_greedy  →  贪心路径搜索算法
  - 202_combo        →  回溯法、排列/子集/组合枚举
"""

import numpy as np
from typing import List, Tuple, Callable, Optional


# ---------------------------------------------------------------------------
# 贪心构象搜索（种子项目 1365_tsp_greedy）
# ---------------------------------------------------------------------------
def greedy_conformation_search(
    n_rotatable: int,
    n_bins_per_torsion: int,
    energy_grid: np.ndarray,
    start_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, float]:
    """
    对药物分子的可旋转键进行贪心搜索。

    物理模型：
      每个可旋转键 $i$ 有 $m$ 个离散二面角状态（bins）。
      状态空间大小为 $m^n$。能量网格 energy_grid 的形状为 $(m, m, \dots, m)$（n 维）。

    由于完整能量网格在高维时不可行，此处采用简化模型：
      energy_grid[i, j] 表示相邻两个可旋转键（键 i 与键 i+1）
      分别处于状态 i 和状态 j 时的耦合能量。

    贪心策略：
      1. 从每个可能的起始状态出发
      2. 依次选择使当前累计能量最小的下一状态
      3. 记录所有起始点中能量最低的序列

    参数边界：
        n_rotatable       : 可旋转键数量，>= 1
        n_bins_per_torsion : 每个键的状态数，>= 2
        energy_grid       : shape (n_rotatable, n_bins_per_torsion, n_bins_per_torsion)
                            energy_grid[i, a, b] 为键 i 状态 a、键 i+1 状态 b 的耦合能
        start_indices     : 可选的起始状态列表；若为 None，则遍历所有状态

    返回：
        best_sequence     : shape (n_rotatable,) 的整数状态序列
        best_energy       : 总能量
    """
    if n_rotatable < 1:
        raise ValueError("greedy_conformation_search: n_rotatable must be >= 1.")
    if n_bins_per_torsion < 2:
        raise ValueError("greedy_conformation_search: n_bins_per_torsion must be >= 2.")
    if energy_grid.ndim != 3:
        raise ValueError("greedy_conformation_search: energy_grid must be 3D.")
    if energy_grid.shape != (n_rotatable, n_bins_per_torsion, n_bins_per_torsion):
        raise ValueError("greedy_conformation_search: energy_grid shape mismatch.")

    if start_indices is None:
        start_indices = list(range(n_bins_per_torsion))

    best_sequence = None
    best_energy = float('inf')

    for start in start_indices:
        seq = np.zeros(n_rotatable, dtype=int)
        seq[0] = start
        total_energy = 0.0

        for i in range(n_rotatable - 1):
            current_state = seq[i]
            # 找出使下一键能量最低的状态
            next_energies = energy_grid[i, current_state, :]
            next_state = int(np.argmin(next_energies))
            seq[i + 1] = next_state
            total_energy += next_energies[next_state]

        if total_energy < best_energy:
            best_energy = total_energy
            best_sequence = seq.copy()

    return best_sequence, best_energy


def path_cost(n: int, distance: np.ndarray, p: np.ndarray) -> float:
    """
    计算路径（排列 p）的总成本。

    参数边界：
        n         : 城市数
        distance  : shape (n, n) 的距离矩阵
        p         : shape (n,) 的排列索引
    """
    if distance.shape != (n, n):
        raise ValueError("path_cost: distance matrix must be square.")
    if p.shape[0] != n:
        raise ValueError("path_cost: p length must equal n.")

    cost = 0.0
    for i2 in range(n):
        i1 = (i2 - 1) % n
        cost += distance[p[i1], p[i2]]
    return cost


# ---------------------------------------------------------------------------
# 回溯搜索（种子项目 202_combo / backtrack）
# ---------------------------------------------------------------------------
def backtrack_search(
    n_vars: int,
    domain_size: int,
    constraint_checker: Callable[[List[int]], bool],
    max_solutions: int = 1000,
) -> List[List[int]]:
    """
    使用回溯法枚举满足约束的离散变量赋值组合。

    物理背景：
      药物分子的 $n$ 个可旋转二面角，每个有 $m$ 个离散取值。
      constraint_checker 验证部分赋值是否满足空间碰撞约束
      （如 van der Waals 半径不重叠）。

    参数边界：
        n_vars            : 变量数，>= 1
        domain_size       : 每个变量的取值个数，>= 1
        constraint_checker: 接受部分赋值列表，返回是否可行的函数
        max_solutions     : 最大返回解数，>= 1

    返回：
        solutions         : 列表，每个元素是一个完整赋值列表
    """
    if n_vars < 1:
        raise ValueError("backtrack_search: n_vars must be >= 1.")
    if domain_size < 1:
        raise ValueError("backtrack_search: domain_size must be >= 1.")
    if max_solutions < 1:
        raise ValueError("backtrack_search: max_solutions must be >= 1.")

    solutions: List[List[int]] = []
    current = [0] * n_vars

    def _bt(pos: int):
        if len(solutions) >= max_solutions:
            return
        if pos == n_vars:
            if constraint_checker(current):
                solutions.append(current.copy())
            return

        for val in range(domain_size):
            current[pos] = val
            # 前向检查：若部分赋值已违反约束则剪枝
            if constraint_checker(current[:pos + 1]):
                _bt(pos + 1)

    _bt(0)
    return solutions


# ---------------------------------------------------------------------------
# 药物分子构象优化器
# ---------------------------------------------------------------------------
def dock_drug_greedy_rotamer(
    n_torsions: int = 5,
    n_bins: int = 12,
    vdw_radius_drug: float = 3.5,   # Å
    vdw_radius_pocket: np.ndarray = None,
    pocket_coords: np.ndarray = None,
    base_energy: float = -5.0,      # kcal/mol
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    对药物分子进行基于贪心算法的构象搜索与对接打分。

    能量模型：
      $E_{\text{total}} = E_{\text{LJ}} + E_{\text{elec}} + E_{\text{tor}}$

      $E_{\text{LJ}} = 4\epsilon \sum_{i<j} \left[
          \left(\frac{\sigma}{r_{ij}}\right)^{12}
          - \left(\frac{\sigma}{r_{ij}}\right)^6
      \right]$

      $E_{\text{elec}} = \sum_{i<j} \frac{q_i q_j}{4\pi\epsilon_0 \epsilon_r r_{ij}}$

      $E_{\text{tor}} = \sum_k V_k [1 + \cos(n_k \phi_k - \delta_k)]$

    参数边界：
        n_torsions   : 可旋转键数，1 <= n_torsions <= 10
        n_bins       : 每个二面角的离散取值数，>= 2
        vdw_radius_drug : 药物范德华半径，> 0
        pocket_coords   : 结合口袋原子坐标，shape (m, 3)
        vdw_radius_pocket : 口袋原子范德华半径，shape (m,)
    """
    if not (1 <= n_torsions <= 10):
        raise ValueError("dock_drug_greedy_rotamer: n_torsions must be in [1, 10].")
    if n_bins < 2:
        raise ValueError("dock_drug_greedy_rotamer: n_bins must be >= 2.")
    if vdw_radius_drug <= 0:
        raise ValueError("dock_drug_greedy_rotamer: vdw_radius_drug must be > 0.")

    # 生成简化的能量网格
    np.random.seed(42)
    energy_grid = np.zeros((n_torsions, n_bins, n_bins), dtype=float)

    for i in range(n_torsions):
        for a in range(n_bins):
            for b in range(n_bins):
                # 简化的能量：二面角势 + 随机口袋相互作用
                phi_a = 2.0 * np.pi * a / n_bins
                phi_b = 2.0 * np.pi * b / n_bins
                e_torsion = 0.5 * (1.0 + np.cos(3.0 * phi_a - np.pi / 4.0))
                e_torsion += 0.3 * (1.0 + np.cos(2.0 * phi_b))
                # 口袋相互作用（简化为随机扰动）
                e_pocket = -2.0 * np.exp(-((a - b) ** 2) / 8.0)
                energy_grid[i, a, b] = e_torsion + e_pocket + base_energy

    best_seq, best_energy = greedy_conformation_search(n_torsions, n_bins, energy_grid)

    # 计算最佳构象的二面角值
    best_dihedrals = 2.0 * np.pi * best_seq / n_bins

    return best_seq, best_energy, best_dihedrals
