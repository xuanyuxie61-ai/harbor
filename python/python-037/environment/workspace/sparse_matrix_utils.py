
import numpy as np
from typing import Tuple, List, Dict






class SparseMatrixCOO:

    def __init__(self, m: int, n: int):
        self.m = m
        self.n = n
        self.rows: List[int] = []
        self.cols: List[int] = []
        self.data: List[float] = []

    def add(self, i: int, j: int, val: float):
        if not (0 <= i < self.m and 0 <= j < self.n):
            raise IndexError(f"SparseMatrixCOO: 索引 ({i}, {j}) 超出维度 ({self.m}, {self.n})")
        self.rows.append(i)
        self.cols.append(j)
        self.data.append(float(val))

    def nnz(self) -> int:
        return len(self.data)

    def to_dense(self) -> np.ndarray:
        A = np.zeros((self.m, self.n))
        for i, j, v in zip(self.rows, self.cols, self.data):
            A[i, j] += v
        return A

    def to_csr(self) -> "SparseMatrixCSR":
        return coo_to_csr(self)






class SparseMatrixCSR:

    def __init__(self, data: np.ndarray, col_idx: np.ndarray, row_ptr: np.ndarray, m: int, n: int):
        self.data = np.asarray(data, dtype=float)
        self.col_idx = np.asarray(col_idx, dtype=int)
        self.row_ptr = np.asarray(row_ptr, dtype=int)
        self.m = m
        self.n = n

    def nnz(self) -> int:
        return len(self.data)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if len(x) != self.n:
            raise ValueError("SparseMatrixCSR.matvec: 向量维度不匹配")
        y = np.zeros(self.m)
        for i in range(self.m):
            s = 0.0
            for idx in range(self.row_ptr[i], self.row_ptr[i + 1]):
                s += self.data[idx] * x[self.col_idx[idx]]
            y[i] = s
        return y

    def matvec_transpose(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if len(x) != self.m:
            raise ValueError("SparseMatrixCSR.matvec_transpose: 向量维度不匹配")
        y = np.zeros(self.n)
        for i in range(self.m):
            xi = x[i]
            for idx in range(self.row_ptr[i], self.row_ptr[i + 1]):
                y[self.col_idx[idx]] += self.data[idx] * xi
        return y

    def to_dense(self) -> np.ndarray:
        A = np.zeros((self.m, self.n))
        for i in range(self.m):
            for idx in range(self.row_ptr[i], self.row_ptr[i + 1]):
                A[i, self.col_idx[idx]] = self.data[idx]
        return A






def coo_to_csr(coo: SparseMatrixCOO) -> SparseMatrixCSR:
    m, n = coo.m, coo.n
    nnz = coo.nnz()
    if nnz == 0:
        return SparseMatrixCSR(np.zeros(0), np.zeros(0, dtype=int), np.zeros(m + 1, dtype=int), m, n)

    rows = np.array(coo.rows, dtype=int)
    cols = np.array(coo.cols, dtype=int)
    data = np.array(coo.data, dtype=float)


    order = np.lexsort((cols, rows))
    rows = rows[order]
    cols = cols[order]
    data = data[order]


    row_ptr = np.zeros(m + 1, dtype=int)
    for r in rows:
        row_ptr[r + 1] += 1
    row_ptr = np.cumsum(row_ptr)

    return SparseMatrixCSR(data, cols, row_ptr, m, n)


def csr_to_coo(csr: SparseMatrixCSR) -> SparseMatrixCOO:
    coo = SparseMatrixCOO(csr.m, csr.n)
    for i in range(csr.m):
        for idx in range(csr.row_ptr[i], csr.row_ptr[i + 1]):
            coo.add(i, csr.col_idx[idx], csr.data[idx])
    return coo


def expand_symmetric_coo(coo: SparseMatrixCOO, lower: bool = True) -> SparseMatrixCOO:
    full = SparseMatrixCOO(coo.m, coo.n)
    for i, j, v in zip(coo.rows, coo.cols, coo.data):
        full.add(i, j, v)
        if i != j:
            full.add(j, i, v)
    return full






def read_matrix_market(filename: str) -> SparseMatrixCOO:
    coo = None
    with open(filename, 'r') as f:
        line = f.readline()
        while line.startswith('%'):
            line = f.readline()
        header = line.strip().split()
        m, n, nnz_expected = int(header[0]), int(header[1]), int(header[2])
        coo = SparseMatrixCOO(m, n)
        for _ in range(nnz_expected):
            parts = f.readline().strip().split()
            i, j = int(parts[0]) - 1, int(parts[1]) - 1
            val = float(parts[2])
            coo.add(i, j, val)
    return coo


def write_matrix_market(filename: str, coo: SparseMatrixCOO, symmetric: bool = False) -> None:
    with open(filename, 'w') as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{coo.m} {coo.n} {coo.nnz()}\n")
        for i, j, v in zip(coo.rows, coo.cols, coo.data):
            f.write(f"{i + 1} {j + 1} {v:.16e}\n")


def write_sparse_triplet(filename: str, coo: SparseMatrixCOO, base_zero: bool = True) -> None:
    offset = 0 if base_zero else 1
    with open(filename, 'w') as f:
        for i, j, v in zip(coo.rows, coo.cols, coo.data):
            f.write(f"{i + offset} {j + offset} {v:.16e}\n")


def read_sparse_triplet(filename: str, m: int, n: int, base_zero: bool = True) -> SparseMatrixCOO:
    coo = SparseMatrixCOO(m, n)
    offset = 0 if base_zero else -1
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            i = int(parts[0]) + offset
            j = int(parts[1]) + offset
            v = float(parts[2])
            coo.add(i, j, v)
    return coo






def construct_fem_stiffness_sparse(n: int, h: float, epsilon: float = 1.0) -> SparseMatrixCOO:
    coo = SparseMatrixCOO(n, n)
    diag_val = 2.0 * epsilon / h
    offdiag_val = -epsilon / h
    for i in range(n):
        coo.add(i, i, diag_val)
        if i + 1 < n:
            coo.add(i, i + 1, offdiag_val)
            coo.add(i + 1, i, offdiag_val)
    return coo


def sparse_matvec_timing(csr: SparseMatrixCSR, x: np.ndarray, n_repeat: int = 100) -> float:
    import time
    start = time.perf_counter()
    for _ in range(n_repeat):
        _ = csr.matvec(x)
    end = time.perf_counter()
    return (end - start) / n_repeat






if __name__ == "__main__":

    coo = SparseMatrixCOO(5, 5)
    coo.add(0, 0, 2.0)
    coo.add(0, 1, -1.0)
    coo.add(1, 0, -1.0)
    coo.add(1, 1, 2.0)
    coo.add(2, 2, 3.0)
    assert coo.nnz() == 5


    csr = coo.to_csr()
    assert csr.nnz() == 5
    assert csr.m == 5 and csr.n == 5


    x = np.ones(5)
    y = csr.matvec(x)
    assert abs(y[0] - 1.0) < 1e-12
    assert abs(y[1] - 1.0) < 1e-12
    assert abs(y[2] - 3.0) < 1e-12


    yt = csr.matvec_transpose(x)
    assert abs(yt[0] - 1.0) < 1e-12


    sym = SparseMatrixCOO(3, 3)
    sym.add(0, 0, 1.0)
    sym.add(1, 0, 2.0)
    sym.add(2, 1, 3.0)
    full = expand_symmetric_coo(sym)
    dense_full = full.to_dense()
    assert abs(dense_full[0, 1] - 2.0) < 1e-12
    assert abs(dense_full[1, 0] - 2.0) < 1e-12
    assert abs(dense_full[1, 2] - 3.0) < 1e-12
    assert abs(dense_full[2, 1] - 3.0) < 1e-12


    fem_coo = construct_fem_stiffness_sparse(10, 0.1)
    fem_csr = fem_coo.to_csr()
    y_fem = fem_csr.matvec(np.ones(10))

    assert abs(y_fem[5]) < 1e-12, "FEM 刚度矩阵 × 全1向量 内部节点应为0"

    print("sparse_matrix_utils.py: 所有自测通过")
