# -*- coding: utf-8 -*-
"""
optimizer.py
============
组合优化与约束满足模块。

融合原始项目:
  - 739_matrix_chain_brute: 矩阵链乘法最优括号化（组合优化）
  - 1057_satisfy_brute: 暴力搜索满足约束（布尔可满足性思想）

核心数学公式
------------
1. 矩阵链乘法最优括号化（融入 matrix_chain_brute）:
   给定矩阵维度序列 d_0, d_1, ..., d_n，
   求括号化方案使得标量乘法次数最少。
   乘法次数: cost = Σ_{k=i}^{j-1} d_{i-1} d_k d_j
   动态规划递推:
   m[i,j] = min_{i≤k<j} { m[i,k] + m[k+1,j] + d_{i-1} d_k d_j }
   本模块提供暴力搜索与动态规划两种实现。

2. 布尔可满足性暴力搜索（融入 satisfy_brute）:
   对 n 个布尔变量，枚举全部 2^n 种赋值，
   检查是否满足给定逻辑公式 φ(x_1, ..., x_n)。
   时间复杂度 O(2^n)。

3. 算子应用顺序优化（科学计算场景）:
   对一系列线性算子 L_1, L_2, ..., L_k 的复合应用，
   寻找最优计算顺序以最小化浮点运算量（FLOPs）。
   类比矩阵链：每个算子的输入/输出维度对应矩阵规模。

4. 约束满足问题（CSP）在水平集中的应用:
   多相流接触角条件:
   θ_1 + θ_2 + θ_3 = 2π
   σ_{12} cos θ_3 = σ_{13} cos θ_2 + σ_{23} cos θ_1
   （Young 方程）
   对给定表面张力 σ_{ij}，搜索满足 Young 方程的接触角组合。
"""

import numpy as np
import itertools


class MatrixChainOptimizer:
    """
    矩阵链乘法最优括号化（源自 739_matrix_chain_brute）。
    """

    @staticmethod
    def matrix_chain_dp(dims):
        """
        动态规划求解最优括号化。
        参数:
            dims : list of int, 矩阵维度 [d0, d1, ..., dn]
        返回:
            min_cost : int, 最小乘法次数
            split    : ndarray, 分割点矩阵
        """
        n = len(dims) - 1
        m = np.zeros((n, n), dtype=np.int64)
        s = np.zeros((n, n), dtype=np.int32)

        for chain_len in range(2, n + 1):
            for i in range(n - chain_len + 1):
                j = i + chain_len - 1
                m[i, j] = np.iinfo(np.int64).max
                for k in range(i, j):
                    cost = m[i, k] + m[k + 1, j] + dims[i] * dims[k + 1] * dims[j + 1]
                    if cost < m[i, j]:
                        m[i, j] = cost
                        s[i, j] = k

        return m[0, n - 1], s

    @staticmethod
    def matrix_chain_brute(dims):
        """
        暴力搜索所有括号化方案（适用于 n ≤ 8 的小规模问题）。
        使用 Catalan 数枚举所有二叉树结构。
        """
        n = len(dims) - 1
        if n <= 1:
            return 0, []

        def brute_cost(i, j):
            if i == j:
                return 0, []
            min_cost = np.iinfo(np.int64).max
            best_k = i
            for k in range(i, j):
                left_cost, left_split = brute_cost(i, k)
                right_cost, right_split = brute_cost(k + 1, j)
                cost = left_cost + right_cost + dims[i] * dims[k + 1] * dims[j + 1]
                if cost < min_cost:
                    min_cost = cost
                    best_k = k
            return min_cost, [best_k]

        cost, _ = brute_cost(0, n - 1)
        return cost

    @staticmethod
    def get_optimal_order(s, i, j):
        """
        从分割矩阵 s 重构最优乘法顺序。
        """
        if i == j:
            return f"M{i+1}"
        k = s[i, j]
        left = MatrixChainOptimizer.get_optimal_order(s, i, k)
        right = MatrixChainOptimizer.get_optimal_order(s, k + 1, j)
        return f"({left} × {right})"


class ConstraintSatisfier:
    """
    约束满足暴力搜索（源自 1057_satisfy_brute）。
    """

    @staticmethod
    def brute_force_satisfy(n_vars, formula_func, max_solutions=1000):
        """
        对 n 个布尔变量，枚举所有 2^n 种赋值，找出使 formula_func 为 True 的解。

        参数:
            n_vars       : int, 变量个数
            formula_func : callable, f(bvec) -> bool, bvec 为长度为 n_vars 的 bool 列表
            max_solutions: int, 最大返回解数
        返回:
            solutions : list of list of bool
            count     : int, 总解数
        """
        solutions = []
        total = 2 ** n_vars
        count = 0
        for idx in range(total):
            bvec = [(idx >> j) & 1 for j in range(n_vars)]
            if formula_func(bvec):
                count += 1
                if len(solutions) < max_solutions:
                    solutions.append(bvec)
        return solutions, count

    @staticmethod
    def young_equation_solver(sigma12, sigma13, sigma23, tol=1e-4, n_grid=200):
        """
        对三相接角问题，搜索满足 Young 方程的接触角组合。
        Young 方程:
            σ_{12} = σ_{13} cos θ_3 + σ_{23} cos θ_2
            σ_{13} = σ_{12} cos θ_3 + σ_{23} cos θ_1
            σ_{23} = σ_{12} cos θ_2 + σ_{13} cos θ_1
        在二维情况下简化为:
            σ_{12} cos θ_3 = σ_{13} cos θ_2 + σ_{23} cos θ_1
            θ_1 + θ_2 + θ_3 = 2π

        参数:
            sigma12, sigma13, sigma23 : float, 表面张力
            tol  : 容差
            n_grid : 搜索网格数
        返回:
            solutions : list of (θ1, θ2, θ3) in radians
        """
        solutions = []
        for i in range(n_grid):
            theta1 = np.pi * i / n_grid
            for j in range(n_grid):
                theta2 = np.pi * j / n_grid
                theta3 = 2.0 * np.pi - theta1 - theta2
                if theta3 < 0 or theta3 > 2.0 * np.pi:
                    continue
                lhs = sigma12 * np.cos(theta3)
                rhs = sigma13 * np.cos(theta2) + sigma23 * np.cos(theta1)
                if abs(lhs - rhs) < tol:
                    solutions.append((theta1, theta2, theta3))
        return solutions


class OperatorSequenceOptimizer:
    """
    科学计算算子应用顺序优化器。
    将预处理算子、差分算子、滤波算子的复合应用类比为矩阵链乘法，
    寻找最小计算量的执行顺序。
    """

    @staticmethod
    def estimate_flops(dims, order):
        """
        估计给定执行顺序的总 FLOPs。
        dims : list of int, 每个算子的输入/输出维度
        order: list of int, 执行顺序索引
        """
        total = 0
        # 简化模型：每步运算量为 dim[i] * dim[i+1]
        for i in range(len(order) - 1):
            total += dims[order[i]] * dims[order[i + 1]]
        return total

    @staticmethod
    def optimize_preconditioner_chain(n_ops, dim_in, dim_mid, dim_out):
        """
        对预处理链进行最优排序。
        例如：平滑器 → 粗网格校正 → 插值 → 限制
        """
        dims = [dim_in, dim_mid, dim_mid, dim_out, dim_out]
        cost, s = MatrixChainOptimizer.matrix_chain_dp(dims)
        return cost, s
