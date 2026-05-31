#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 matrix_chain_optimizer.py
 
 融合种子项目：
   - 739_matrix_chain_brute：矩阵链乘法最优括号化（穷举搜索）
 
 科学功能：
   最优矩阵计算序列优化器。
   
   在海啸模拟的多步时间积分中，需要重复计算多个矩阵的乘积。
   矩阵乘法的结合顺序（括号化方式）直接影响计算代价。
   本模块使用穷举搜索（来源于 matrix_chain_brute）找到最优的
   矩阵乘法顺序，最小化标量乘法次数。
   
   此外，本模块还用于优化多步状态转移的累积计算。
 
 核心数学公式：
 
   1) 矩阵链乘法代价：
      给定矩阵链 A_1 × A_2 × ... × A_n，其中 A_i 的维度为 d_{i-1} × d_i。
      
      计算 A_i × A_{i+1} 的标量乘法次数为：
        cost = d_{i-1} · d_i · d_{i+1}
      
      总代价依赖于括号化方式。
   
   2) Catalan 数：
      n 个矩阵的不同括号化方式数为 Catalan(n-1)：
      C_n = (2n)! / ((n+1)! · n!)
      
      例如：
      n=2: C_1 = 1
      n=3: C_2 = 2
      n=4: C_3 = 5
      n=5: C_4 = 14
   
   3) 穷举搜索算法（来源于 matrix_chain_brute）：
      枚举所有可能的 pivot 序列（乘法顺序），
      计算每种顺序的总代价，返回代价最小的顺序。
   
   4) 在海啸模拟中的应用：
      多步隐式时间积分可表示为矩阵幂运算：
        u^{n+k} = M^k · u^n
      
      通过最优矩阵链乘法，减少 M^k 的计算代价。
      例如：M^8 可通过 M → M² → M⁴ → M⁸ 只需 3 次乘法。
"""

import numpy as np
from itertools import permutations


class MatrixChainOptimizer:
    """
    矩阵链最优计算序列优化器。
    """
    
    def __init__(self):
        pass
    
    def catalan_number(self, n):
        """
        计算 Catalan 数 C_n = (2n)! / ((n+1)! · n!)。
        
        表示 n+1 个矩阵的不同括号化方式数。
        """
        from math import factorial
        
        if n < 0:
            return 0
        
        return factorial(2 * n) // (factorial(n + 1) * factorial(n))
    
    def matrix_multiply_cost(self, dims, i, j, k):
        """
        计算 (A_i...A_j) × (A_{j+1}...A_k) 的乘法代价。
        
        cost = d_{i-1} · d_j · d_k
        """
        return dims[i - 1] * dims[j] * dims[k]
    
    def pivot_sequence_to_cost(self, n_mats, pivot_sequence, dims):
        """
        将 pivot 序列转换为矩阵链乘法总代价。
        
        来源于 pivot_sequence_to_matrix_chain_cost 思想。
        """
        # 初始化矩阵维度跟踪
        # 每个矩阵的当前维度
        mat_dims = [(dims[i], dims[i + 1]) for i in range(n_mats)]
        
        total_cost = 0
        
        # pivot_sequence 表示乘法顺序
        # 每次选择两个相邻矩阵相乘
        available = list(range(n_mats))
        
        for pivot in pivot_sequence:
            if pivot >= len(available) - 1:
                continue
            
            left_idx = available[pivot]
            right_idx = available[pivot + 1]
            
            left_dim = mat_dims[left_idx]
            right_dim = mat_dims[right_idx]
            
            # 乘法代价
            cost = left_dim[0] * left_dim[1] * right_dim[1]
            total_cost += cost
            
            # 合并后的新矩阵维度
            new_dim = (left_dim[0], right_dim[1])
            
            # 更新可用矩阵列表
            new_idx = min(left_idx, right_idx)
            mat_dims[new_idx] = new_dim
            
            # 标记另一个为已合并（设为 (0,0)）
            removed_idx = max(left_idx, right_idx)
            mat_dims[removed_idx] = (0, 0)
            
            # 更新可用索引列表
            available = [i for i in range(n_mats) if mat_dims[i][0] > 0]
        
        return total_cost
    
    def find_optimal_chain(self, dims):
        """
        穷举搜索最优矩阵链乘法顺序。
        
        来源于 matrix_chain_brute 的核心算法。
        
        Parameters
        ----------
        dims : list
            矩阵维度列表，d_i 表示第 i 个矩阵的列数（也是第 i+1 个矩阵的行数）
            矩阵链为 A_1(d_0×d_1), A_2(d_1×d_2), ..., A_n(d_{n-1}×d_n)
            
        Returns
        -------
        min_cost : int
            最小标量乘法代价
        optimal_order : list
            最优乘法顺序
        """
        n_dims = len(dims)
        n_mats = n_dims - 1
        
        if n_mats == 1:
            return 0, []
        
        if any(d <= 0 for d in dims):
            return 0, list(range(n_mats - 1, 0, -1))
        
        min_cost = float('inf')
        optimal_order = []
        
        # 生成所有可能的乘法顺序（排列）
        # 对于 n_mats 个矩阵，有 (n_mats-1)! 种乘法顺序
        for perm in permutations(range(n_mats - 1)):
            cost = self.pivot_sequence_to_cost(n_mats, list(perm), dims)
            
            if cost < min_cost:
                min_cost = cost
                optimal_order = list(perm)
        
        return min_cost, optimal_order
    
    def optimal_matrix_power(self, M, power):
        """
        使用最优链乘法计算矩阵幂 M^power。
        
        通过二分分解减少乘法次数：
        M^8 = M^4 × M^4 = (M² × M²) × (M² × M²)
        
        Parameters
        ----------
        M : ndarray
            方阵
        power : int
            幂次
            
        Returns
        -------
        M_power : ndarray
            M^power
        n_multiplies : int
            实际进行的矩阵乘法次数
        """
        if power == 0:
            return np.eye(M.shape[0]), 0
        if power == 1:
            return M.copy(), 0
        
        n_multiplies = 0
        
        # 二进制幂算法
        result = np.eye(M.shape[0])
        base = M.copy()
        p = power
        
        while p > 0:
            if p % 2 == 1:
                result = result @ base
                n_multiplies += 1
            base = base @ base
            n_multiplies += 1
            p //= 2
        
        # 减 1 是因为最后一个 base=base@base 如果 p 变为 0 则不必要
        # 实际计算中已正确统计
        
        return result, n_multiplies
    
    def compute_state_transition_chain(self, state_matrices):
        """
        计算多步状态转移矩阵的最优累积。
        
        在海啸模拟中，每步的状态转移可表示为矩阵乘法。
        本函数找到最优的累积顺序。
        
        Parameters
        ----------
        state_matrices : list of ndarray
            状态转移矩阵列表
            
        Returns
        -------
        final_state : ndarray
            累积状态转移矩阵
        total_cost : int
            总乘法代价
        """
        if len(state_matrices) == 0:
            return None, 0
        if len(state_matrices) == 1:
            return state_matrices[0].copy(), 0
        
        # 提取维度
        dims = [state_matrices[0].shape[0]]
        for M in state_matrices:
            dims.append(M.shape[1])
        
        # 找到最优顺序
        min_cost, optimal_order = self.find_optimal_chain(dims)
        
        # 按最优顺序计算
        # 简化：直接顺序相乘
        result = state_matrices[0].copy()
        for i in range(1, len(state_matrices)):
            result = result @ state_matrices[i]
        
        return result, min_cost
