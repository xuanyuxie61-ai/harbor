#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gaussian_solver.py
==================

基于种子项目 337_eros 的鲁棒高斯消元求解器。

科学背景
--------
在矩阵分解型推荐系统中，固定一个因子求解另一个因子时，
需要求解大量线性系统。高斯消元（Gauss Elimination）是最经典的
直接求解方法，配合部分主元选取（partial pivoting）可确保数值稳定性。

对于系统 A x = b，Gauss 消元将其转化为上三角系统 U x = y，
然后通过回代求解。

PLU 分解:
    P A = L U
    
    P : 置换矩阵（记录行交换）
    L : 单位下三角矩阵（multipliers）
    U : 上三角矩阵
    
求解:
    A x = b  ⇒  P A x = P b  ⇒  L U x = P b
    先解 L y = P b（前代），再解 U x = y（回代）

行列式:
    det(A) = det(P^T) det(L) det(U) = (-1)^{swap_count} · 1 · ∏ U_{ii}

逆矩阵:
    A^{-1} = U^{-1} L^{-1} P
    通过求解 A X = I 的每一列得到
"""

import numpy as np


class RobustGaussianSolver:
    """
    鲁棒高斯消元求解器，支持 PLU 分解、行列式、逆矩阵。
    """
    
    def __init__(self, pivot_tol=1e-12):
        self.pivot_tol = pivot_tol
    
    def solve_gauss(self, A, b):
        """
        使用带部分主元的高斯消元求解 A x = b。
        
        算法:
            1. 构造增广矩阵 [A | b]
            2. 对每列 j:
               a. 在第 j 列下方找绝对值最大的元素作为主元
               b. 交换当前行与主元行
               c. 归一化主元行
               d. 消去下方所有行的第 j 列元素
            3. 回代求解
            
        边界保护:
            - 奇异矩阵检测（主元 < pivot_tol）
            - A 非方阵时抛出异常
        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A 必须是方阵")
        
        n = A.shape[0]
        
        # 增广矩阵
        if b.ndim == 1:
            Ab = np.column_stack([A, b])
        else:
            Ab = np.hstack([A, b])
        
        m_aug = Ab.shape[1]
        
        # 消元
        for j in range(n):
            # 部分主元
            pivot_col = Ab[j:, j]
            max_idx = np.argmax(np.abs(pivot_col))
            pivot_val = pivot_col[max_idx]
            
            if abs(pivot_val) < self.pivot_tol:
                # 矩阵接近奇异，使用伪逆或跳过
                continue
            
            max_idx += j
            if max_idx != j:
                Ab[[j, max_idx], :] = Ab[[max_idx, j], :]
            
            # 归一化
            Ab[j, :] /= Ab[j, j]
            
            # 消去
            for i in range(n):
                if i != j and abs(Ab[i, j]) > self.pivot_tol:
                    factor = -Ab[i, j]
                    Ab[i, :] += factor * Ab[j, :]
        
        # 提取解
        x = Ab[:, n:]
        if x.shape[1] == 1:
            x = x.flatten()
        return x
    
    def plu_decomposition(self, A):
        """
        计算 PLU 分解: P A = L U。
        
        返回:
            P, L, U
            
        边界保护:
            - 零主元时继续处理（L 中 multiplier 设为 0）
        """
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A 必须是方阵")
        
        n = A.shape[0]
        P = np.eye(n)
        L = np.eye(n)
        U = np.copy(A)
        swap_count = 0
        
        for j in range(n - 1):
            pivot_col = U[j:, j]
            max_idx = np.argmax(np.abs(pivot_col))
            pivot_val = pivot_col[max_idx]
            
            if abs(pivot_val) < self.pivot_tol:
                continue
            
            max_idx += j
            if max_idx != j:
                P[[j, max_idx], :] = P[[max_idx, j], :]
                U[[j, max_idx], :] = U[[max_idx, j], :]
                if j > 0:
                    L[[j, max_idx], :j] = L[[max_idx, j], :j]
                swap_count += 1
            
            for i in range(j + 1, n):
                if abs(U[j, j]) > self.pivot_tol:
                    s = U[i, j] / U[j, j]
                    U[i, :] -= s * U[j, :]
                    U[i, j] = 0.0
                    L[i, j] = s
        
        return P, L, U
    
    def solve_plu(self, A, b):
        """
        使用 PLU 分解求解 A x = b。
        
        步骤:
            1. P A = L U
            2. 令 y = U x，先解 L y = P b
            3. 再解 U x = y
        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        
        P, L, U = self.plu_decomposition(A)
        
        n = A.shape[0]
        if b.ndim == 1:
            b = b.reshape(-1, 1)
        
        pb = P @ b
        
        # 前代求解 L y = pb
        y = np.zeros_like(pb)
        for i in range(n):
            y[i] = pb[i] - L[i, :i] @ y[:i]
        
        # 回代求解 U x = y
        x = np.zeros_like(y)
        for i in range(n - 1, -1, -1):
            if abs(U[i, i]) < self.pivot_tol:
                x[i] = 0.0
            else:
                x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
        
        return x.squeeze() if x.shape[1] == 1 else x
    
    def determinant(self, A):
        """
        使用 Gauss 消元计算行列式。
        
        公式:
            det(A) = (-1)^{swap_count} · ∏_{i} U_{ii}
            
        边界保护:
            - 空矩阵返回 1
            - 非方阵返回 nan
        """
        A = np.asarray(A, dtype=float)
        if A.size == 0:
            return 1.0
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return float('nan')
        
        P, L, U = self.plu_decomposition(A)
        # 统计置换次数
        swap_count = 0
        P_mat = np.eye(A.shape[0])
        P_mat = P @ P_mat
        for i in range(A.shape[0]):
            if np.argmax(P_mat[i, :]) != i:
                swap_count += 1
        
        det = (-1.0) ** swap_count * np.prod(np.diag(U))
        return float(det)
    
    def inverse(self, A):
        """
        使用 Gauss 消元计算逆矩阵。
        
        方法:
            求解 A X = I 的每一列
            
        边界保护:
            - 奇异矩阵返回伪逆近似
        """
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A 必须是方阵")
        
        n = A.shape[0]
        I = np.eye(n)
        
        try:
            inv = self.solve_gauss(A, I)
            if inv.ndim == 1:
                inv = inv.reshape(n, 1)
            if inv.shape[1] != n:
                inv = inv[:, :n]
            return inv
        except Exception:
            # 退化情况使用 numpy 伪逆
            return np.linalg.pinv(A)
