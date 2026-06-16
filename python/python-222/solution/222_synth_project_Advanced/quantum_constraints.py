# -*- coding: utf-8 -*-
"""
quantum_constraints.py
======================

量子数守恒约束的整数矩阵行简化阶梯形 (RREF) 求解。

融合种子项目:
    - 569_i4mat_rref2: 整数矩阵行简化阶梯形

物理背景:
    强子化过程必须严格守恒以下量子数:
        - 电荷 Q
        - 重子数 B
        - 奇异数 S
        - 粲数 C, 底数 B', 顶数 T
        - 同位旋第三分量 I_3

    对每个味道的夸克, 量子数构成整数向量:
        q = (Q, B, S, C, B', T, I_3)

    强子化约束:
        sum_i q_i (initial) = sum_f q_f (final)

    该整数线性系统的可解性可通过 RREF 判定:
        [A | b] -> RREF -> 判定 rank(A) vs rank([A|b])
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict


# ======================================================================
# 夸克量子数表 (标准模型)
# ======================================================================
# (flavor, Q, B, S, C, Bprime, T, I3)
QUARK_QUANTUM_NUMBERS: Dict[str, Tuple[int, int, int, int, int, int, int]] = {
    'u':     (+2/3, +1/3,  0,  0,  0,  0, +1/2),
    'd':     (-1/3, +1/3,  0,  0,  0,  0, -1/2),
    's':     (-1/3, +1/3, -1,  0,  0,  0,  0),
    'c':     (+2/3, +1/3,  0, +1,  0,  0,  0),
    'b':     (-1/3, +1/3,  0,  0, -1,  0,  0),
    't':     (+2/3, +1/3,  0,  0,  0, +1,  0),
    'ubar':  (-2/3, -1/3,  0,  0,  0,  0, -1/2),
    'dbar':  (+1/3, -1/3,  0,  0,  0,  0, +1/2),
    'sbar':  (+1/3, -1/3, +1,  0,  0,  0,  0),
    'cbar':  (-2/3, -1/3,  0, -1,  0,  0,  0),
    'bbar':  (+1/3, -1/3,  0,  0, +1,  0,  0),
    'tbar':  (-2/3, -1/3,  0,  0,  0, -1,  0),
    'g':     (0, 0, 0, 0, 0, 0, 0),
}

# 常见强子量子数
HADRON_QUANTUM_NUMBERS: Dict[str, Tuple[int, int, int, int, int, int, int]] = {
    'pi+':    (+1, 0, 0, 0, 0, 0, +1),
    'pi-':    (-1, 0, 0, 0, 0, 0, -1),
    'pi0':    (0, 0, 0, 0, 0, 0, 0),
    'K+':     (+1, 0, +1, 0, 0, 0, +1/2),
    'K-':     (-1, 0, -1, 0, 0, 0, -1/2),
    'K0':     (0, 0, +1, 0, 0, 0, -1/2),
    'p':      (+1, +1, 0, 0, 0, 0, +1/2),
    'n':      (0, +1, 0, 0, 0, 0, -1/2),
    'Lambda': (0, +1, -1, 0, 0, 0, 0),
}


# ======================================================================
# 整数矩阵 RREF (来自 569_i4mat_rref2)
# ======================================================================
def _gcd(a: int, b: int) -> int:
    """最大公约数 (Euclidean)"""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // _gcd(a, b) if a and b else 0


def i4mat_rref2(m: int, n1: int, n2: int, a: List[List[int]]) -> Tuple[List[List[int]], int]:
    """
    整数矩阵行简化阶梯形 (来自 569_i4mat_rref2.m)。

    通过纯整数运算避免浮点误差, 适合量子数守恒判定。

    Parameters
    ----------
    m : int
        行数
    n1 : int
        起始列 (通常为 0)
    n2 : int
        终止列 (包含)
    a : list of list of int
        输入矩阵

    Returns
    -------
    (a_rref, rank) : 行简化阶梯形矩阵与秩
    """
    # 深拷贝
    a = [row[:] for row in a]

    i_col = n1
    pivot_row = 0

    while pivot_row < m and i_col <= n2:
        # 在当前列找非零元
        pivot = -1
        for i in range(pivot_row, m):
            if a[i][i_col] != 0:
                pivot = i
                break

        if pivot < 0:
            i_col += 1
            continue

        # 交换到当前行
        if pivot != pivot_row:
            a[pivot_row], a[pivot] = a[pivot], a[pivot_row]

        # 使主元为正
        if a[pivot_row][i_col] < 0:
            for j in range(n2 + 1):
                a[pivot_row][j] = -a[pivot_row][j]

        # 消除其他行
        for i in range(m):
            if i == pivot_row:
                continue
            if a[i][i_col] == 0:
                continue
            # 用整数运算消除: 行 i = a[i,i_col] * row_pivot - a[pivot,i_col] * row_i
            factor_i = a[i][i_col]
            factor_p = a[pivot_row][i_col]
            for j in range(n2 + 1):
                a[i][j] = factor_p * a[i][j] - factor_i * a[pivot_row][j]
            # 约去公因子
            g = 0
            for j in range(n2 + 1):
                g = _gcd(g, abs(a[i][j]))
            if g > 1:
                for j in range(n2 + 1):
                    a[i][j] //= g

        pivot_row += 1
        i_col += 1

    rank = pivot_row
    return a, rank


def check_quantum_conservation(initial_partons: List[str],
                                final_hadrons: List[str]) -> Tuple[bool, List[int]]:
    """
    检查夸克-强子转化过程的量子数守恒。

    构造增广矩阵:
        [H | -Q]
    其中 H 的列为强子量子数, Q 为夸克总量子数向量。
    RREF 后, 检查最后一列是否全零 (相容性)。

    Returns
    -------
    (conserved, residuals) : 是否守恒, 残差向量
    """
    # 计算初始夸克量子数总和
    q_total = [0.0] * 7
    for p in initial_partons:
        if p in QUARK_QUANTUM_NUMBERS:
            qn = QUARK_QUANTUM_NUMBERS[p]
            for i in range(7):
                q_total[i] += qn[i]

    # 计算末态强子量子数总和
    h_total = [0.0] * 7
    for h in final_hadrons:
        if h in HADRON_QUANTUM_NUMBERS:
            qn = HADRON_QUANTUM_NUMBERS[h]
            for i in range(7):
                h_total[i] += qn[i]

    # 残差
    residuals = [q_total[i] - h_total[i] for i in range(7)]

    # 判定守恒 (容差处理半整数)
    conserved = all(abs(r) < 0.01 for r in residuals)
    return conserved, residuals


def build_quantum_constraint_matrix(partons: List[str], n_hadron_slots: int) -> List[List[int]]:
    """
    构造量子数约束矩阵:
        行: 7 个量子数
        列: n_hadron_slots 个强子候选 + 1 列 (初始量子数向量)

    返回增广矩阵, 用于 RREF 求解。
    """
    m = 7
    n = n_hadron_slots + 1

    # 缩放为整数 (乘以 2 处理半整数)
    A = [[0] * n for _ in range(m)]

    # 初始部分子量子数 (最后一列, 负号表示约束)
    q_total = [0.0] * 7
    for p in partons:
        if p in QUARK_QUANTUM_NUMBERS:
            qn = QUARK_QUANTUM_NUMBERS[p]
            for i in range(7):
                q_total[i] += qn[i]

    for i in range(7):
        A[i][n-1] = -int(round(2 * q_total[i]))

    return A


def rref_integer_safety_check(A: List[List[int]]) -> Tuple[int, bool]:
    """
    对增广矩阵执行 RREF, 返回 (rank, consistent)。
    consistent = rank(A[:,:-1]) == rank(A)
    """
    m = len(A)
    n = len(A[0])

    # 系数矩阵秩
    A_sub = [row[:-1] for row in A]
    _, rank_sub = i4mat_rref2(m, 0, n - 2, [row[:] for row in A_sub])

    # 增广矩阵秩
    _, rank_aug = i4mat_rref2(m, 0, n - 1, [row[:] for row in A])

    return rank_sub, rank_sub == rank_aug


# ======================================================================
# 自测
# ======================================================================
if __name__ == "__main__":
    print("=== Quantum Number Conservation Check ===")

    # e+e- -> u ubar -> pi+ pi-
    initial = ['u', 'ubar']
    final = ['pi+', 'pi-']
    ok, res = check_quantum_conservation(initial, final)
    print(f"  u ubar -> pi+ pi-: conserved={ok}, residuals={[f'{r:.3f}' for r in res]}")

    # u d d -> p pi- (baryon number)
    initial2 = ['u', 'd', 'd']
    final2 = ['p', 'pi-']
    ok2, res2 = check_quantum_conservation(initial2, final2)
    print(f"  udd -> p pi-: conserved={ok2}, residuals={[f'{r:.3f}' for r in res2]}")

    # RREF test
    print("\n=== Integer RREF Test ===")
    A = [[1, 2, 3, 4],
         [2, 4, 6, 8],
         [3, 6, 9, 12]]
    Ar, rank = i4mat_rref2(3, 0, 3, A)
    print(f"  Input: 3x4 matrix")
    print(f"  Rank: {rank}")
    for row in Ar:
        print(f"    {row}")
