#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class RobustGaussianSolver:
    
    def __init__(self, pivot_tol=1e-12):
        self.pivot_tol = pivot_tol
    
    def solve_gauss(self, A, b):
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A 必须是方阵")
        
        n = A.shape[0]
        

        if b.ndim == 1:
            Ab = np.column_stack([A, b])
        else:
            Ab = np.hstack([A, b])
        
        m_aug = Ab.shape[1]
        

        for j in range(n):

            pivot_col = Ab[j:, j]
            max_idx = np.argmax(np.abs(pivot_col))
            pivot_val = pivot_col[max_idx]
            
            if abs(pivot_val) < self.pivot_tol:

                continue
            
            max_idx += j
            if max_idx != j:
                Ab[[j, max_idx], :] = Ab[[max_idx, j], :]
            

            Ab[j, :] /= Ab[j, j]
            

            for i in range(n):
                if i != j and abs(Ab[i, j]) > self.pivot_tol:
                    factor = -Ab[i, j]
                    Ab[i, :] += factor * Ab[j, :]
        

        x = Ab[:, n:]
        if x.shape[1] == 1:
            x = x.flatten()
        return x
    
    def plu_decomposition(self, A):
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
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        
        P, L, U = self.plu_decomposition(A)
        
        n = A.shape[0]
        if b.ndim == 1:
            b = b.reshape(-1, 1)
        
        pb = P @ b
        

        y = np.zeros_like(pb)
        for i in range(n):
            y[i] = pb[i] - L[i, :i] @ y[:i]
        

        x = np.zeros_like(y)
        for i in range(n - 1, -1, -1):
            if abs(U[i, i]) < self.pivot_tol:
                x[i] = 0.0
            else:
                x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
        
        return x.squeeze() if x.shape[1] == 1 else x
    
    def determinant(self, A):
        A = np.asarray(A, dtype=float)
        if A.size == 0:
            return 1.0
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return float('nan')
        
        P, L, U = self.plu_decomposition(A)

        swap_count = 0
        P_mat = np.eye(A.shape[0])
        P_mat = P @ P_mat
        for i in range(A.shape[0]):
            if np.argmax(P_mat[i, :]) != i:
                swap_count += 1
        
        det = (-1.0) ** swap_count * np.prod(np.diag(U))
        return float(det)
    
    def inverse(self, A):
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

            return np.linalg.pinv(A)
