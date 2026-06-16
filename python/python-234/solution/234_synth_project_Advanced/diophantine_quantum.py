"""
diophantine_quantum.py
----------------------
量子数守恒的 Diophantine 方程枚举: 在 B 介子衰变道分类中
自动找出所有满足电荷、重子数、奇异数、粲数守恒的末态组合。
映射自种子项目 743_mcnuggets_diophantine (非负整数 Diophantine 方程求解)。

物理动机:
  在 B 介子衰变中, 初态 (B 介子) 有确定的量子数:
      Q (电荷), B (重子数), S (奇异数), C (粲数), B' (底数)
  末态粒子的量子数之和必须等于初态:
      Σ_i Q_i = Q_B,   Σ_i B_i = B_B,   Σ_i S_i = S_B,  ...
  若允许末态粒子为给定的 n 种粒子 (如 π, K, D, J/psi 等), 每个可出现
  多次, 则找所有非负整数解 (n_1, ..., n_k) 满足:
      q_1 n_1 + q_2 n_2 + ... + q_k n_k = Q_B
      b_1 n_1 + b_2 n_2 + ... + b_k n_k = B_B
      ...
  这是一个多维 Diophantine 方程组。对 k 个粒子种类, d 个守恒量,
  方程组为 d × k 矩阵 A 乘以 n-向量 x = b (d-向量)。

  本模块:
    1) 实现多维非负整数 Diophantine 方程求解 (仿 743);
    2) 针对 B 衰变的量子数守恒, 枚举允许的末态;
    3) 进一步用质量约束 |Σ E_i - m_B| < threshold 筛选。

数学:
  对 1D 情况 a·x = b, 解的数目与 Frobenius number 相关:
      g(a_1, ..., a_k) = max { b : a·x = b 无非负整数解 }
  对 a = [6, 9, 20] (McNuggets 问题), g = 43。
  对 B 衰变, a 为粒子电荷/重子数, 通常 g 很小, 因此解空间有限。
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple

from b_physics_constants import QUANTUM_NUMBERS


# ========================================================================== #
#             多维非负整数 Diophantine 求解 (仿 743)                       #
# ========================================================================== #
def diophantine_1d_nonnegative(a: List[int], b: int) -> List[List[int]]:
    """
    求解 1D Diophantine 方程 a_1 x_1 + ... + a_k x_k = b, x_i >= 0 整数。
    算法 (仿 743.diophantine_nd_nonnegative):
        回溯法: 从第一个系数开始, 逐个尝试 x_i = 0, 1, 2, ..., floor(r/a_i),
        其中 r 为剩余 RHS。到达最后一个系数时检查余数是否为 0。
    """
    if any(ai <= 0 for ai in a):
        raise ValueError("所有系数必须为正整数")
    if b < 0:
        return []
    k = len(a)
    solutions: List[List[int]] = []
    y = [0] * k

    def backtrack(j: int, r: int):
        if j == k - 1:
            # 最后一个: 检查 r 是否能被 a[j] 整除
            if r % a[j] == 0:
                y[j] = r // a[j]
                solutions.append(y.copy())
            return
        # 尝试 x_j = 0, 1, ..., floor(r / a_j)
        max_val = r // a[j]
        for val in range(max_val + 1):
            y[j] = val
            backtrack(j + 1, r - val * a[j])

    backtrack(0, b)
    return solutions


def diophantine_nd_nonnegative(
        A: List[List[int]],
        b: List[int],
        max_per_var: int = 10,
) -> List[List[int]]:
    """
    求解多维 Diophantine 方程组 A x = b, x >= 0 整数。
    A: d × k 矩阵 (d 个守恒方程, k 个粒子种类)
    b: 长度 d 的 RHS 向量
    算法: 回溯 + 约束传播 (在每个维度上检查余数是否可达)。

    由于多维 Diophantine 为 NP-hard, 对 k 较大时可能指数爆炸;
    物理上 k <= 20, d <= 5, max_per_var 限制单变量最大取值, 保证可计算。
    """
    if not A or not A[0]:
        return []
    d = len(A)
    k = len(A[0])
    if len(b) != d:
        raise ValueError(f"RHS 长度 {len(b)} 不等于方程数 {d}")
    solutions: List[List[int]] = []
    x = [0] * k

    def backtrack(j: int, residual: List[int]):
        if j == k - 1:
            # 检查最后一个变量能否使所有残差为 0
            valid = True
            xj_candidate = None
            for r in range(d):
                if A[r][j] == 0:
                    if residual[r] != 0:
                        valid = False
                        break
                else:
                    if residual[r] % A[r][j] != 0:
                        valid = False
                        break
                    cand = residual[r] // A[r][j]
                    if xj_candidate is None:
                        xj_candidate = cand
                    elif xj_candidate != cand:
                        valid = False
                        break
            if valid and xj_candidate is not None and xj_candidate >= 0:
                x[j] = xj_candidate
                solutions.append(x.copy())
            return
        # 约束传播: 计算 x_j 的可行上界
        upper = max_per_var
        for r in range(d):
            if A[r][j] > 0:
                upper = min(upper, residual[r] // A[r][j])
        for val in range(upper + 1):
            x[j] = val
            new_res = [residual[r] - val * A[r][j] for r in range(d)]
            # 可行性剪枝: 剩余变量能否达到 residual
            if all(nr >= 0 for nr in new_res):
                backtrack(j + 1, new_res)

    backtrack(0, list(b))
    return solutions


# ========================================================================== #
#                   B 衰变末态枚举                                          #
# ========================================================================== #
# 候选末态粒子列表 (轻介子 + D 介子):
B_DECAY_CANDIDATES = [
    "pip", "pim", "pi0", "Kp", "Km", "KS",
]


def enumerate_charge_conserving_final_states(
        initial_Q: int,
        initial_S: int,
        candidates: List[str] = None,
) -> List[Dict[str, int]]:
    """
    枚举所有满足电荷和奇异数守恒的末态组合。
    initial_Q: 初态电荷 (B0: 0, B+: +1, Bs: 0)
    initial_S: 初态奇异数 (B0, Bp: 0, Bs: -1)
    candidates: 候选粒子列表 (默认 B_DECAY_CANDIDATES)

    返回: 列表, 每项为 {particle: count} 的字典。
    """
    if candidates is None:
        candidates = B_DECAY_CANDIDATES
    # 构建系数矩阵 A: 行 = [Q, S], 列 = 候选粒子
    A = []
    for qn in ["Q", "S"]:
        row = []
        for p in candidates:
            qn_tuple = QUANTUM_NUMBERS.get(p)
            if qn_tuple is None:
                raise ValueError(f"未知粒子 {p}")
            idx = {"Q": 0, "B": 1, "S": 2, "C": 3, "Bprime": 4}[qn]
            row.append(qn_tuple[idx])
        A.append(row)
    b = [initial_Q, initial_S]
    # 总多重数约束: 末态粒子数 2~6 (物理合理范围)
    # 添加一行: Σ n_i = N, N = 2..6
    solutions = []
    for N in range(2, 7):
        A_ext = [row[:] for row in A] + [[1] * len(candidates)]
        b_ext = b + [N]
        sols = diophantine_nd_nonnegative(A_ext, b_ext, max_per_var=N)
        for sol in sols:
            d = {}
            for k, p in enumerate(candidates):
                if sol[k] > 0:
                    d[p] = sol[k]
            solutions.append(d)
    return solutions


def filter_by_mass(
        final_states: List[Dict[str, int]],
        m_parent: float,
        threshold: float = 200.0,   # MeV: 允许动能上限
) -> List[Dict[str, int]]:
    """
    进一步用质量约束筛选:
        Σ n_i m_i <= m_parent (能量守恒)
        Σ n_i m_i >= m_parent - threshold (允许部分动能)
    质量表从 b_physics_constants 获取。
    """
    from b_physics_constants import (
        M_PI, M_K, M_KS, M_D0, M_DP, M_JPSI,
    )
    # D^- 与 D^+ 质量相同 (CPT)
    M_DM = M_DP
    mass_map = {
        "pip": M_PI, "pim": M_PI, "pi0": M_PI,
        "Kp": M_K, "Km": M_K, "KS": M_KS,
        "Dp": M_DP, "Dm": M_DM, "D0": M_D0, "D0bar": M_D0,
        "Jpsi": M_JPSI,
    }
    result = []
    for state in final_states:
        total_mass = 0.0
        for p, n in state.items():
            if p not in mass_map:
                total_mass = float("inf")
                break
            total_mass += n * mass_map[p]
        if total_mass <= m_parent and total_mass >= m_parent - threshold:
            result.append(state)
    return result


# ========================================================================== #
#                    McNuggets 问题类比 (仿 743)                            #
# ========================================================================== #
def frobenius_number_estimate(a: List[int], max_b: int = 200) -> int:
    """
    估计 Frobenius 数 g(a_1, ..., a_k): 最大的 b 使得 a·x = b 无非负整数解。
    算法: 从 max_b 向下搜索, 找到第一个无解的 b。
    对 a = [6, 9, 20] (McNuggets), g = 43。
    """
    for b in range(max_b, -1, -1):
        sols = diophantine_1d_nonnegative(a, b)
        if len(sols) == 0:
            return b
    return -1


def count_decay_modes_by_invariant_mass(
        a: List[int],
        max_mass_bin: int,
) -> List[int]:
    """
    统计每个不变质量 bin 中有多少种衰变模式 (解的个数)。
    类比 McNuggets 问题中 "用 6, 9, 20 凑 b 个 McNuggets 的方案数"。
    返回长度 max_mass_bin + 1 的列表, 第 b 项为方案数。
    """
    counts = []
    for b in range(max_mass_bin + 1):
        sols = diophantine_1d_nonnegative(a, b)
        counts.append(len(sols))
    return counts
