
import numpy as np


class CRSMatrix:

    def __init__(self, m, n, nnz=0):
        self.m = m
        self.n = n
        self.nnz = nnz
        self.val = np.zeros(nnz, dtype=float)
        self.col = np.zeros(nnz, dtype=int)
        self.rowptr = np.zeros(m + 1, dtype=int)

    def matvec(self, x):
        if x.shape[0] != self.n:
            raise ValueError(f"matvec: 向量维度 {x.shape[0]} 不匹配矩阵列数 {self.n}")
        y = np.zeros(self.m, dtype=float)
        for i in range(self.m):
            s = 0.0
            for idx in range(self.rowptr[i], self.rowptr[i + 1]):
                s += self.val[idx] * x[self.col[idx]]
            y[i] = s
        return y

    def matvec_transpose(self, x):
        if x.shape[0] != self.m:
            raise ValueError(f"matvec_transpose: 向量维度 {x.shape[0]} 不匹配矩阵行数 {self.m}")
        y = np.zeros(self.n, dtype=float)
        for i in range(self.m):
            xi = x[i]
            for idx in range(self.rowptr[i], self.rowptr[i + 1]):
                y[self.col[idx]] += self.val[idx] * xi
        return y

    def to_dense(self):
        A = np.zeros((self.m, self.n), dtype=float)
        for i in range(self.m):
            for idx in range(self.rowptr[i], self.rowptr[i + 1]):
                A[i, self.col[idx]] = self.val[idx]
        return A

    @staticmethod
    def from_dense(dense):
        m, n = dense.shape
        rows, cols = np.nonzero(np.abs(dense) > 1.0e-15)
        nnz = len(rows)
        val = dense[rows, cols].astype(float)
        col = cols.astype(int)
        rowptr = np.zeros(m + 1, dtype=int)
        for i in range(m):
            rowptr[i + 1] = rowptr[i] + np.count_nonzero(rows == i)



        rowptr = np.zeros(m + 1, dtype=int)
        for r in rows:
            rowptr[r + 1] += 1
        rowptr = np.cumsum(rowptr)
        obj = CRSMatrix(m, n, nnz)
        obj.val = val
        obj.col = col
        obj.rowptr = rowptr
        return obj

    def copy(self):
        obj = CRSMatrix(self.m, self.n, self.nnz)
        obj.val = self.val.copy()
        obj.col = self.col.copy()
        obj.rowptr = self.rowptr.copy()
        return obj


def build_sparse_dif2(n):
    nnz = 3 * n - 2
    val = np.zeros(nnz, dtype=float)
    col = np.zeros(nnz, dtype=int)
    rowptr = np.zeros(n + 1, dtype=int)
    idx = 0
    for i in range(n):
        rowptr[i] = idx
        if i > 0:
            val[idx] = -1.0
            col[idx] = i - 1
            idx += 1
        val[idx] = 2.0
        col[idx] = i
        idx += 1
        if i < n - 1:
            val[idx] = -1.0
            col[idx] = i + 1
            idx += 1
    rowptr[n] = idx
    A = CRSMatrix(n, n, idx)
    A.val = val
    A.col = col
    A.rowptr = rowptr
    return A


def sparse_solve_cg(A, b, x0=None, tol=1.0e-10, max_iter=1000):
    n = A.n
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = x0.copy()

    r = b - A.matvec(x)
    p = r.copy()
    rsold = np.dot(r, r)

    if np.sqrt(rsold) < tol:
        return x

    for _ in range(max_iter):
        Ap = A.matvec(p)
        alpha = rsold / (np.dot(p, Ap) + 1.0e-30)
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.dot(r, r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


def sparse_solve_jacobi(A, b, x0=None, tol=1.0e-10, max_iter=2000):
    n = A.n
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = x0.copy()


    D = np.zeros(n, dtype=float)
    for i in range(n):
        for idx in range(A.rowptr[i], A.rowptr[i + 1]):
            if A.col[idx] == i:
                D[i] = A.val[idx]
                break
        if abs(D[i]) < 1.0e-15:
            raise ValueError("Jacobi: 对角线元素为零，无法迭代。")

    for _ in range(max_iter):
        x_new = np.zeros(n, dtype=float)
        for i in range(n):
            s = 0.0
            for idx in range(A.rowptr[i], A.rowptr[i + 1]):
                j = A.col[idx]
                if j != i:
                    s += A.val[idx] * x[j]
            x_new[i] = (b[i] - s) / D[i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x
