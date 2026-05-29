"""
load_optimizer.py
博士级大变形非线性有限元分析 — 荷载步组合优化模块

融合原项目:
  - 1179_subset_sum_backtrack: 子集和回溯算法
  - 743_mcnuggets_diophantine: 非负整数线性组合 (丢番图方程)

核心数学:
  在非线性有限元分析中，荷载通常需要分多个步长施加。
  本模块解决两个优化问题:

  1. 子集和回溯 — 从候选荷载步长集合中选择子集，使其和等于目标总荷载:
     给定候选集合 V = {v1, v2, ..., vn} 和目标值 S，
     寻找子集 U ⊆ V 使得 Σ_{u∈U} u = S

     回溯算法:
       - 将 V 按升序排列
       - 深度优先搜索，逐步尝试包含/不包含每个元素
       - 剪枝: 若当前和 + 剩余最大和 < S，或当前和 > S，则回溯

  2. 丢番图整数组合 — 计算达到目标荷载的不同步长组合数:
     给定步长集合 A = {a1, a2, ..., am} 和目标荷载 B，
     寻找所有非负整数解 x = (x1, ..., xm) 使得:
       a1*x1 + a2*x2 + ... + am*xm = B

     这在多尺度加载中代表: 每种荷载模式 a_i 施加 x_i 次

  3. 自适应荷载步长控制:
     基于 Newton-Raphson 迭代收敛性动态调整步长:
       - 若收敛快（迭代次数 < 3），步长增大 1.5 倍
       - 若收敛慢（迭代次数 > 8），步长减半
       - 最大/最小步长约束
"""

import numpy as np


class LoadOptimizerError(Exception):
    pass


def subset_sum_backtrack(values, target):
    """
    子集和回溯算法

    源自原项目 1179_subset_sum_backtrack (subset_sum_backtrack)

    数学:
      寻找所有子集 U ⊆ V 使得 sum(U) = target

    输入:
        values: list of float，候选值（将排序）
        target: float，目标和
    输出:
        solutions: list of list，每个子列表是一个解的索引集合
    """
    values = sorted([float(v) for v in values if v > 0])
    n = len(values)
    solutions = []

    def backtrack(start_idx, current_sum, current_indices):
        # 精确匹配
        if abs(current_sum - target) < 1e-9:
            solutions.append(current_indices.copy())
            return
        # 超出目标或无法达到
        if current_sum > target:
            return
        remaining_max = sum(values[start_idx:])
        if current_sum + remaining_max < target - 1e-9:
            return

        for i in range(start_idx, n):
            current_indices.append(i)
            backtrack(i + 1, current_sum + values[i], current_indices)
            current_indices.pop()

    backtrack(0, 0.0, [])
    return solutions


def diophantine_nonnegative_solutions(coeffs, target):
    """
    非负整数线性丢番图方程求解

    源自原项目 743_mcnuggets_diophantine (mcnuggets_diophantine)

    数学:
      a1*x1 + a2*x2 + ... + an*xn = B
      xi >= 0, 整数

    输入:
        coeffs: list of int，正整数系数
        target: int，目标值（非负整数）
    输出:
        solutions: list of tuple，每个 tuple 是一个解
    """
    coeffs = [int(c) for c in coeffs if c > 0]
    target = int(target)
    if target < 0:
        return []
    n = len(coeffs)
    solutions = []

    # 递归搜索
    def search(idx, remaining, current):
        if idx == n - 1:
            if remaining % coeffs[idx] == 0:
                x = remaining // coeffs[idx]
                solutions.append(tuple(current + [x]))
            return
        max_x = remaining // coeffs[idx]
        for x in range(max_x + 1):
            current.append(x)
            search(idx + 1, remaining - x * coeffs[idx], current)
            current.pop()

    search(0, target, [])
    return solutions


def adaptive_load_stepping(initial_load, target_load, min_step, max_step,
                           convergence_history=None):
    """
    自适应荷载步长控制

    数学:
      设前一步 Newton 迭代次数为 n_iter:
        if n_iter <= 3:   step_new = min(1.5 * step, max_step)
        elif n_iter >= 8: step_new = max(0.5 * step, min_step)
        else:             step_new = step

    输入:
        initial_load: 当前已施加荷载
        target_load: 目标总荷载
        min_step, max_step: 最小/最大步长
        convergence_history: list of int，每步的迭代次数
    输出:
        step_sizes: list of float，建议的步长序列
    """
    remaining = target_load - initial_load
    if remaining <= 1e-12:
        return []

    step_sizes = []
    current_step = max_step
    current_load = initial_load

    iter_idx = 0
    while current_load < target_load - 1e-9:
        if convergence_history is not None and iter_idx < len(convergence_history):
            n_iter = convergence_history[iter_idx]
            if n_iter <= 3:
                current_step = min(current_step * 1.5, max_step)
            elif n_iter >= 8:
                current_step = max(current_step * 0.5, min_step)

        actual_step = min(current_step, target_load - current_load)
        if actual_step < min_step:
            actual_step = target_load - current_load

        step_sizes.append(float(actual_step))
        current_load += actual_step
        iter_idx += 1

        # 安全退出
        if len(step_sizes) > 10000:
            break

    return step_sizes


def optimize_load_increments(candidate_steps, target_total, strategy='subset_sum'):
    """
    荷载增量组合优化主函数

    输入:
        candidate_steps: list of float，候选荷载步长
        target_total: float，目标总荷载
        strategy: 'subset_sum' 或 'diophantine'
    输出:
        best_sequence: list of float，优化的步长序列
        n_solutions: int，可行解的数量
    """
    if strategy == 'subset_sum':
        sols = subset_sum_backtrack(candidate_steps, target_total)
        if not sols:
            # 无精确解，返回贪心近似
            seq = []
            remaining = target_total
            for v in sorted(candidate_steps, reverse=True):
                while remaining >= v - 1e-9:
                    seq.append(v)
                    remaining -= v
            if remaining > 1e-9:
                seq.append(remaining)
            return sorted(seq), 0

        # 选择步数最少的解
        best = min(sols, key=len)
        best_sequence = [candidate_steps[i] for i in best]
        return best_sequence, len(sols)

    elif strategy == 'diophantine':
        # 将问题离散化为整数
        scale = 1000
        int_coeffs = [int(round(c * scale)) for c in candidate_steps]
        int_target = int(round(target_total * scale))
        sols = diophantine_nonnegative_solutions(int_coeffs, int_target)
        if not sols:
            return optimize_load_increments(candidate_steps, target_total, 'subset_sum')

        best = min(sols, key=sum)
        best_sequence = []
        for i, count in enumerate(best):
            best_sequence.extend([candidate_steps[i]] * count)
        return best_sequence, len(sols)

    else:
        raise LoadOptimizerError(f"Unknown strategy: {strategy}")
