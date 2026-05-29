"""
Sparse matrix formats and operations for efficient FEM stiffness assembly.

Adapted from:
  - ge_to_ccs (general to compressed-column sparse conversion)
  - r8ci (circulant matrix-vector multiply for periodic domains)
"""

import numpy as np
from scipy import sparse as sp


class SparseCCS:
    """
    Compressed-Column Storage (CCS) sparse matrix wrapper.
    """
    def __init__(self, data, rowind, colptr, shape):
        self.data = np.asarray(data, dtype=float)
        self.rowind = np.asarray(rowind, dtype=int)
        self.colptr = np.asarray(colptr, dtype=int)
        self.shape = shape
        self.nz_num = len(data)

    def to_dense(self):
        """Reconstruct dense matrix."""
        m, n = self.shape
        A = np.zeros((m, n), dtype=float)
        for j in range(n):
            for k in range(self.colptr[j], self.colptr[j + 1]):
                i = self.rowind[k]
                A[i, j] = self.data[k]
        return A

    def to_scipy_csr(self):
        """Export to scipy CSR for fast solve."""
        dense = self.to_dense()
        return sp.csr_matrix(dense)


def ge_to_ccs(Age):
    """
    Convert a dense general (GE) matrix to CCS format.
    Adapted from ge_to_ccs.m.
    """
    Age = np.asarray(Age, dtype=float)
    m, n = Age.shape
    rowind = []
    data = []
    colptr = np.zeros(n + 1, dtype=int)
    k = 0
    for j in range(n):
        colptr[j] = k
        for i in range(m):
            if abs(Age[i, j]) > 0.0:
                rowind.append(i)
                data.append(Age[i, j])
                k += 1
    colptr[n] = k
    nz_num = k
    return SparseCCS(data, rowind, colptr, (m, n))


def circulant_matrix_vector(n, first_row, x):
    """
    Multiply an N x N circulant matrix (defined by its first row) with vector x.
    Adapted from r8ci_mv.m.
    """
    first_row = np.asarray(first_row, dtype=float)
    x = np.asarray(x, dtype=float)
    b = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(i):
            b[i] += first_row[n + j - i] * x[j]
        for j in range(i, n):
            b[i] += first_row[j - i] * x[j]
    return b


def build_periodic_fem_stiffness(n, diffusivity, dx):
    """
    Build a 1-D FEM stiffness matrix with periodic boundary conditions
    using a circulant structure for the diffusion operator.
    """
    # Standard second-difference stencil
    diag = 2.0 * diffusivity / (dx * dx)
    off = -diffusivity / (dx * dx)
    first_row = np.zeros(n, dtype=float)
    first_row[0] = diag
    first_row[1] = off
    first_row[-1] = off
    # Build dense for conversion
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        A[i, i] = diag
        A[i, (i + 1) % n] = off
        A[i, (i - 1) % n] = off
    return A


def solve_sparse_system(A_dense, b):
    """
    Solve A x = b using scipy sparse LU for numerical stability.
    """
    A_csr = sp.csr_matrix(A_dense)
    return sp.linalg.spsolve(A_csr, b)


def build_mass_matrix_csc(n_nodes, dx):
    """
    Consistent mass matrix for linear FEM hat functions.

    对于标准线性 hat 函数，一致质量矩阵(consistent mass matrix)的元素为:
        M_{ii} = 2*dx/3          (内部节点)
        M_{i,i±1} = dx/6         (相邻节点)
        M_{00} = M_{nn} = dx/3   (边界节点)

    要求以 SparseCCS 格式返回质量矩阵。
    """
    # TODO Hole 2: 构建 FEM 一致质量矩阵并以 SparseCCS 格式返回
    # 提示: 先构造密集矩阵 M_dense，再使用 ge_to_ccs(M_dense) 转换为 SparseCCS
    raise NotImplementedError("Hole 2: 请实现 FEM 一致质量矩阵的 SparseCCS 表示")
