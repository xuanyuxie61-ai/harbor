
import numpy as np
from utils import check_finite


class CCSMatrix:

    def __init__(self, m, n, colptr, rowind, data):
        self.m = m
        self.n = n
        self.colptr = np.asarray(colptr, dtype=int)
        self.rowind = np.asarray(rowind, dtype=int)
        self.data = np.asarray(data, dtype=float)
        self.nnz = len(self.data)
        if self.colptr[0] != 0:
            raise ValueError("CCSMatrix: colptr[0] must be 0")
        if self.colptr[-1] != self.nnz:
            raise ValueError("CCSMatrix: colptr[-1] must equal nnz")

    def multiply_vector(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self.n:
            raise ValueError("CCSMatrix.multiply_vector: dimension mismatch")
        y = np.zeros(self.m, dtype=float)
        for j in range(self.n):
            for k in range(self.colptr[j], self.colptr[j + 1]):
                i = self.rowind[k]
                y[i] += self.data[k] * x[j]
        return y

    def transpose_multiply_vector(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self.m:
            raise ValueError("CCSMatrix.transpose_multiply_vector: dimension mismatch")
        y = np.zeros(self.n, dtype=float)
        for j in range(self.n):
            s = 0.0
            for k in range(self.colptr[j], self.colptr[j + 1]):
                i = self.rowind[k]
                s += self.data[k] * x[i]
            y[j] = s
        return y

    def to_dense(self):
        A = np.zeros((self.m, self.n), dtype=float)
        for j in range(self.n):
            for k in range(self.colptr[j], self.colptr[j + 1]):
                i = self.rowind[k]
                A[i, j] = self.data[k]
        return A

    @staticmethod
    def from_dense(A, tol=1e-14):
        A = np.asarray(A, dtype=float)
        m, n = A.shape
        colptr = [0]
        rowind = []
        data = []
        for j in range(n):
            col_nnz = 0
            for i in range(m):
                if abs(A[i, j]) > tol:
                    rowind.append(i)
                    data.append(A[i, j])
                    col_nnz += 1
            colptr.append(colptr[-1] + col_nnz)
        return CCSMatrix(m, n, colptr, rowind, data)


def matrix_chain_optimal_order(dims):
    n = len(dims) - 1
    if n <= 0:
        raise ValueError("matrix_chain_optimal_order: need at least one matrix")

    INF = float('inf')
    C = [[0] * n for _ in range(n)]
    split = [[-1] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            C[i][j] = INF
            for k in range(i, j):
                cost = C[i][k] + C[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                if cost < C[i][j]:
                    C[i][j] = cost
                    split[i][j] = k
    return C[0][n - 1], split


def build_parenthesization(split, i, j):
    if i == j:
        return f"A{i}"
    k = split[i][j]
    left = build_parenthesization(split, i, k)
    right = build_parenthesization(split, k + 1, j)
    return f"({left} × {right})"


class SparseLinearOperator:

    def __init__(self, A_ccs):
        self.A = A_ccs
        self.shape = (A_ccs.m, A_ccs.n)

    def matvec(self, x):
        return self.A.multiply_vector(x)

    def rmatvec(self, x):
        return self.A.transpose_multiply_vector(x)


def sparse_matrix_multiply_chain(matrices, order=None):
    if len(matrices) == 0:
        raise ValueError("sparse_matrix_multiply_chain: empty matrix list")
    if len(matrices) == 1:
        return matrices[0]


    all_dense = all(isinstance(M, np.ndarray) for M in matrices)
    if all_dense and order is None and len(matrices) > 2:
        dims = [M.shape[0] for M in matrices] + [matrices[-1].shape[1]]
        _, split = matrix_chain_optimal_order(dims)

        pass

    result = matrices[0]
    for idx in range(1, len(matrices)):
        B = matrices[idx]
        if isinstance(result, CCSMatrix) and isinstance(B, CCSMatrix):

            result = result.to_dense() @ B.to_dense()
        elif isinstance(result, CCSMatrix):
            result = result.to_dense() @ B
        elif isinstance(B, CCSMatrix):
            result = result @ B.to_dense()
        else:
            result = result @ B
    return result
