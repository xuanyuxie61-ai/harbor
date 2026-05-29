"""
sparse_solver.py
稀疏矩阵求解模块

融合原项目:
- 975_r8ccs: 压缩列存储（CCS）稀疏矩阵格式及矩阵-向量乘法

功能:
1. 将稠密矩阵转换为 CCS 稀疏格式
2. 稀疏矩阵-向量乘法
3. 共轭梯度法（CG）求解大规模对称正定线性系统
4. 在分子动力学中用于大规模邻居矩阵和约束求解
"""

import numpy as np
from typing import Tuple, Optional


class SparseCCS:
    """
    压缩列存储（Compressed Column Storage, CCS）稀疏矩阵。
    
    融合原项目 975_r8ccs:
        存储格式:
            - values: 非零元素值数组 (nz_num,)
            - row_indices: 非零元素的行索引 (nz_num,)
            - col_pointers: 每列的起始索引 (n+1,)
        
        第 j 列的非零元素存储在:
            values[col_pointers[j] : col_pointers[j+1]]
            row_indices[col_pointers[j] : col_pointers[j+1]]
    
    物理应用:
        在聚合物 MD 中，大规模 Hessian 矩阵和约束矩阵通常是稀疏的。
        CCS 格式可显著降低内存占用和计算复杂度。
    """
    
    def __init__(self, m: int, n: int):
        """
        参数:
            m: 行数
            n: 列数
        """
        self.m = m
        self.n = n
        self.values = np.array([], dtype=float)
        self.row_indices = np.array([], dtype=int)
        self.col_pointers = np.zeros(n + 1, dtype=int)
        self.nz_num = 0
    
    @classmethod
    def from_dense(cls, A: np.ndarray, threshold: float = 1e-12) -> "SparseCCS":
        """
        从稠密矩阵构建 CCS 稀疏矩阵。
        
        参数:
            A: (m, n) 稠密矩阵
            threshold: 低于此值的元素视为零
        
        返回:
            SparseCCS 对象
        """
        A = np.asarray(A)
        m, n = A.shape
        sparse = cls(m, n)
        
        values = []
        row_indices = []
        col_pointers = [0]
        
        for j in range(n):
            for i in range(m):
                if abs(A[i, j]) > threshold:
                    values.append(A[i, j])
                    row_indices.append(i)
            col_pointers.append(len(values))
        
        sparse.values = np.array(values, dtype=float)
        sparse.row_indices = np.array(row_indices, dtype=int)
        sparse.col_pointers = np.array(col_pointers, dtype=int)
        sparse.nz_num = len(values)
        
        return sparse
    
    def to_dense(self) -> np.ndarray:
        """
        转换为稠密矩阵（主要用于调试）。
        
        返回:
            (m, n) 稠密矩阵
        """
        A = np.zeros((self.m, self.n), dtype=float)
        for j in range(self.n):
            for idx in range(self.col_pointers[j], self.col_pointers[j + 1]):
                i = self.row_indices[idx]
                A[i, j] = self.values[idx]
        return A
    
    def matvec(self, x: np.ndarray) -> np.ndarray:
        """
        稀疏矩阵-向量乘法 y = A @ x。
        
        融合原项目 975_r8ccs 的 r8ccs_mv 算法:
            y = 0
            for j in 0..n-1:
                for k in col_pointers[j] .. col_pointers[j+1]-1:
                    i = row_indices[k]
                    y[i] += values[k] * x[j]
        
        参数:
            x: (n,) 向量
        
        返回:
            (m,) 结果向量
        """
        if len(x) != self.n:
            raise ValueError(f"向量维度不匹配: {len(x)} != {self.n}")
        
        y = np.zeros(self.m, dtype=float)
        
        for j in range(self.n):
            xj = x[j]
            for idx in range(self.col_pointers[j], self.col_pointers[j + 1]):
                i = self.row_indices[idx]
                y[i] += self.values[idx] * xj
        
        return y
    
    def transpose_matvec(self, x: np.ndarray) -> np.ndarray:
        """
        计算 A^T @ x。
        
        参数:
            x: (m,) 向量
        
        返回:
            (n,) 结果向量
        """
        if len(x) != self.m:
            raise ValueError(f"向量维度不匹配: {len(x)} != {self.m}")
        
        y = np.zeros(self.n, dtype=float)
        
        for j in range(self.n):
            s = 0.0
            for idx in range(self.col_pointers[j], self.col_pointers[j + 1]):
                i = self.row_indices[idx]
                s += self.values[idx] * x[i]
            y[j] = s
        
        return y
    
    def get_element(self, i: int, j: int) -> float:
        """
        获取矩阵元素 A[i,j]。
        
        参数:
            i, j: 行列索引
        
        返回:
            元素值（若不存在则返回 0）
        """
        if i < 0 or i >= self.m or j < 0 or j >= self.n:
            raise IndexError("索引越界")
        
        for idx in range(self.col_pointers[j], self.col_pointers[j + 1]):
            if self.row_indices[idx] == i:
                return self.values[idx]
        return 0.0
    
    def set_element(self, i: int, j: int, value: float):
        """
        设置矩阵元素 A[i,j]。
        
        注意: 若元素不存在则添加，存在则更新。
        
        参数:
            i, j: 行列索引
            value: 新值
        """
        if i < 0 or i >= self.m or j < 0 or j >= self.n:
            raise IndexError("索引越界")
        
        # 查找是否已存在
        for idx in range(self.col_pointers[j], self.col_pointers[j + 1]):
            if self.row_indices[idx] == i:
                self.values[idx] = value
                return
        
        # 不存在: 需要插入（简化实现: 重建）
        dense = self.to_dense()
        dense[i, j] = value
        new_sparse = SparseCCS.from_dense(dense)
        self.values = new_sparse.values
        self.row_indices = new_sparse.row_indices
        self.col_pointers = new_sparse.col_pointers
        self.nz_num = new_sparse.nz_num
    
    def sparsity_ratio(self) -> float:
        """
        计算稀疏度（非零元素比例）。
        
        返回:
            非零元素占总元素的比例
        """
        total = self.m * self.n
        if total == 0:
            return 0.0
        return self.nz_num / total


def conjugate_gradient(
    A: SparseCCS,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, int, float]:
    """
    共轭梯度法求解 Ax = b（A 必须对称正定）。
    
    算法:
        r_0 = b - A x_0
        p_0 = r_0
        for k = 0, 1, 2, ...:
            α_k = (r_k^T r_k) / (p_k^T A p_k)
            x_{k+1} = x_k + α_k p_k
            r_{k+1} = r_k - α_k A p_k
            if ||r_{k+1}|| < tol: break
            β_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
            p_{k+1} = r_{k+1} + β_k p_k
    
    参数:
        A: 稀疏矩阵（对称正定）
        b: 右端向量
        x0: 初始猜测
        tol: 收敛容差
        max_iter: 最大迭代次数
    
    返回:
        (x, iterations, residual_norm)
    """
    n = A.n
    b = np.asarray(b, dtype=float)
    
    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.asarray(x0, dtype=float).copy()
    
    r = b - A.matvec(x)
    p = r.copy()
    rs_old = np.dot(r, r)
    
    for k in range(max_iter):
        Ap = A.matvec(p)
        pAp = np.dot(p, Ap)
        
        if abs(pAp) < 1e-15:
            break
        
        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap
        
        rs_new = np.dot(r, r)
        residual = np.sqrt(rs_new)
        
        if residual < tol:
            return x, k + 1, residual
        
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    
    return x, max_iter, np.sqrt(rs_old)


def build_neighbor_sparse_matrix(
    positions: np.ndarray,
    box: np.ndarray,
    cutoff: float,
) -> SparseCCS:
    """
    从 MD 构象构建邻居稀疏矩阵。
    
    矩阵元素:
        A[i,j] = 1  if ||r_i - r_j|| < cutoff
        A[i,j] = 0  otherwise
    
    参数:
        positions: (N, 3) 位置数组
        box: (3,) 盒子尺寸
        cutoff: 截断半径
    
    返回:
        SparseCCS 邻居矩阵
    """
    N = positions.shape[0]
    dense = np.zeros((N, N), dtype=float)
    
    for i in range(N):
        for j in range(i + 1, N):
            dr = positions[i] - positions[j]
            dr = dr - box * np.rint(dr / box)
            r = np.linalg.norm(dr)
            if r < cutoff:
                dense[i, j] = 1.0
                dense[j, i] = 1.0
    
    return SparseCCS.from_dense(dense)
