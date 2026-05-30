
import numpy as np
from typing import Tuple, Optional


class SparseCCS:
    
    def __init__(self, m: int, n: int):
        self.m = m
        self.n = n
        self.values = np.array([], dtype=float)
        self.row_indices = np.array([], dtype=int)
        self.col_pointers = np.zeros(n + 1, dtype=int)
        self.nz_num = 0
    
    @classmethod
    def from_dense(cls, A: np.ndarray, threshold: float = 1e-12) -> "SparseCCS":
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
        A = np.zeros((self.m, self.n), dtype=float)
        for j in range(self.n):
            for idx in range(self.col_pointers[j], self.col_pointers[j + 1]):
                i = self.row_indices[idx]
                A[i, j] = self.values[idx]
        return A
    
    def matvec(self, x: np.ndarray) -> np.ndarray:
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
        if i < 0 or i >= self.m or j < 0 or j >= self.n:
            raise IndexError("索引越界")
        
        for idx in range(self.col_pointers[j], self.col_pointers[j + 1]):
            if self.row_indices[idx] == i:
                return self.values[idx]
        return 0.0
    
    def set_element(self, i: int, j: int, value: float):
        if i < 0 or i >= self.m or j < 0 or j >= self.n:
            raise IndexError("索引越界")
        

        for idx in range(self.col_pointers[j], self.col_pointers[j + 1]):
            if self.row_indices[idx] == i:
                self.values[idx] = value
                return
        

        dense = self.to_dense()
        dense[i, j] = value
        new_sparse = SparseCCS.from_dense(dense)
        self.values = new_sparse.values
        self.row_indices = new_sparse.row_indices
        self.col_pointers = new_sparse.col_pointers
        self.nz_num = new_sparse.nz_num
    
    def sparsity_ratio(self) -> float:
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
