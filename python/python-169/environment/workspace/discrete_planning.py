"""
离散规划与整数资源分配模块
==========================
基于种子项目:
  - 155_change_diophantine : 非负整数Diophantine方程的所有解枚举
  - 1020_reid_tiling       : 精确覆盖问题的RREF求解

核心数学模型:
  1. 线性Diophantine方程:
     a_1 x_1 + a_2 x_2 + ... + a_n x_n = b,   x_i ∈ ℕ₀
     可行性条件: gcd(a_1,...,a_n) | b。
     回溯枚举：对每个变量从 ⌊r/a_j⌋ 递减赋值，递归求解余量 r。

  2. 精确覆盖与RREF（Reduced Row Echelon Form）:
     对二元线性系统 A x = b (A_{ij}∈{0,1}, x_i∈{0,1})，
     通过高斯消元化为RREF，识别主元变量和自由变量，
     枚举自由变量的所有2^k组合，检验解是否全为0/1。

  3. 在机械臂轨迹规划中的应用:
     - Diophantine: 将总控制周期 T 分配到 n 个关节的离散速度档位数
     - 精确覆盖: 将工作空间离散化为网格单元，选择最少数量的
       机械臂姿态使得每个单元都被覆盖恰好一次（传感器/工作头布置）
"""

import numpy as np
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Diophantine方程求解器（155_change_diophantine）
# ---------------------------------------------------------------------------

def _i4vec_gcd(a: np.ndarray) -> int:
    """计算整数数组的最大公约数。"""
    a = np.asarray(a, dtype=int)
    g = abs(a[0])
    for val in a[1:]:
        val = abs(int(val))
        while val:
            g, val = val, g % val
        if g == 1:
            break
    return g


def diophantine_nd_solutions(a: np.ndarray, b: int,
                              max_solutions: int = 10000) -> np.ndarray:
    r"""
    求解 a_1 x_1 + ... + a_n x_n = b 的所有非负整数解。
    使用回溯剪枝算法。

    输入:
      a: (n,) 正整数系数
      b: 非负整数右端项
    返回:
      solutions: (k, n) 整数数组，所有解的列表
    """
    a = np.asarray(a, dtype=int)
    b = int(b)
    n = a.size
    if n == 0:
        return np.array([])
    if b < 0:
        return np.array([])
    # GCD可行性检验
    g = _i4vec_gcd(a)
    if b % g != 0:
        return np.array([])
    # 简化
    a = a // g
    b = b // g
    # 按系数降序排列（利于剪枝）
    sort_idx = np.argsort(-a)
    a_sorted = a[sort_idx]
    solutions = []

    def backtrack(idx: int, remaining: int, current: List[int]):
        if remaining < 0:
            return
        if idx == n - 1:
            if remaining % a_sorted[idx] == 0:
                x_last = remaining // a_sorted[idx]
                sol = current + [x_last]
                # 恢复原始顺序
                full_sol = [0] * n
                for s_i, val in zip(sort_idx, sol):
                    full_sol[s_i] = val
                solutions.append(full_sol)
            return
        max_val = remaining // a_sorted[idx]
        for val in range(max_val, -1, -1):
            if len(solutions) >= max_solutions:
                return
            backtrack(idx + 1, remaining - val * a_sorted[idx], current + [val])

    backtrack(0, b, [])
    if not solutions:
        return np.array([])
    return np.array(solutions, dtype=int)


def allocate_control_cycles(total_cycles: int, joint_weights: np.ndarray,
                            max_solutions: int = 100) -> np.ndarray:
    r"""
    将总控制周期分配到各关节的离散速度档位。
    模型:
       w_1 x_1 + ... + w_n x_n = total_cycles
    其中 x_i 是第i个关节获得的额外控制周期数（用于提高该轴精度）。
    返回若干可行分配方案。
    """
    joint_weights = np.asarray(joint_weights, dtype=int)
    # 确保权重为正
    joint_weights = np.maximum(joint_weights, 1)
    sols = diophantine_nd_solutions(joint_weights, total_cycles, max_solutions)
    return sols


# ---------------------------------------------------------------------------
# 精确覆盖 / RREF（1020_reid_tiling）
# ---------------------------------------------------------------------------

def rref_compute(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    r"""
    计算矩阵的简化行阶梯形（RREF）。
    使用部分主元高斯消元。
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    A = A.copy()
    row = 0
    pivot_cols = []
    for col in range(n):
        # 找主元
        pivot_val = 0.0
        pivot_row = -1
        for r in range(row, m):
            if abs(A[r, col]) > abs(pivot_val):
                pivot_val = A[r, col]
                pivot_row = r
        if abs(pivot_val) < tol:
            continue
        # 交换行
        A[[row, pivot_row]] = A[[pivot_row, row]]
        # 归一化主元行
        A[row] = A[row] / A[row, col]
        # 消去其他行
        for r in range(m):
            if r != row and abs(A[r, col]) > tol:
                A[r] = A[r] - A[r, col] * A[row]
        pivot_cols.append(col)
        row += 1
        if row >= m:
            break
    return A, pivot_cols


def exact_cover_binary(A: np.ndarray, b: np.ndarray,
                       max_solutions: int = 1000) -> List[np.ndarray]:
    r"""
    求解二元精确覆盖问题 A x = b，其中 A∈{0,1}^{m×n}，期望 x∈{0,1}^n。
    使用RREF方法：化简后枚举自由变量组合。

    算法:
      1. 构造增广矩阵 [A | b]
      2. 计算RREF
      3. 识别主元列和自由列
      4. 枚举自由列的所有2^k组合
      5. 反解主元列，检验是否全为0/1
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    m, n = A.shape
    if b.size != m:
        raise ValueError("b维度不匹配")
    # 增广矩阵
    Ab = np.hstack([A, b.reshape(-1, 1)])
    R, pivot_cols = rref_compute(Ab)
    free_cols = [c for c in range(n) if c not in pivot_cols]
    k = len(free_cols)
    solutions = []
    for mask in range(1 << k):
        if len(solutions) >= max_solutions:
            break
        x = np.zeros(n, dtype=float)
        for i, fc in enumerate(free_cols):
            x[fc] = 1.0 if (mask & (1 << i)) else 0.0
        # 回代主元变量
        valid = True
        for r in range(m):
            # 找到该行主元列
            pc = -1
            for c in range(n):
                if abs(R[r, c] - 1.0) < 1e-8:
                    pc = c
                    break
            if pc < 0:
                # 检查一致性: 0 = b?
                if abs(R[r, -1]) > 1e-8:
                    valid = False
                    break
                continue
            x[pc] = R[r, -1]
            for fc in free_cols:
                x[pc] -= R[r, fc] * x[fc]
            # 检验0/1和数值稳定性
            if abs(x[pc] - round(x[pc])) > 1e-6:
                valid = False
                break
            x[pc] = round(x[pc])
            if x[pc] < -1e-6 or x[pc] > 1.0 + 1e-6:
                valid = False
                break
        if valid:
            solutions.append(x.astype(int))
    return solutions


def workspace_coverage_exact_cover(n_poses: int, n_cells: int,
                                    coverage_matrix: np.ndarray) -> List[np.ndarray]:
    r"""
    工作空间覆盖的精确覆盖问题：
      有 n_cells 个离散工作空间单元，n_poses 个候选机械臂姿态。
      coverage_matrix[i,j] = 1 表示姿态 j 覆盖单元 i。
      寻找最少数量的姿态使得每个单元被恰好覆盖一次。

    这里简化为：给定覆盖矩阵，找出所有满足 A x = 1 的 0/1 解。
    """
    A = np.asarray(coverage_matrix, dtype=float)
    m, n = A.shape
    b = np.ones(m, dtype=float)
    return exact_cover_binary(A, b)
