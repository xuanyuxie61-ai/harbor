
import numpy as np


class SparseClimateMatrix:

    def __init__(self, n, nz):
        if n < 1:
            raise ValueError("矩阵阶数 n 必须 >= 1")
        if nz < n + 1:
            raise ValueError("nz 必须 >= n + 1")
        self.n = n
        self.nz = nz
        self.ija = np.zeros(nz, dtype=np.int64)
        self.a = np.zeros(nz, dtype=np.float64)


        self.ija[0] = n + 1
        self.ija[n] = nz

    def set_diagonal(self, diag_vals):
        diag_vals = np.asarray(diag_vals, dtype=np.float64)
        if diag_vals.shape[0] != self.n:
            raise ValueError("对角元长度必须等于 n")
        self.a[0:self.n] = diag_vals

    def add_off_diagonal(self, row, col, val):
        if row < 0 or row >= self.n or col < 0 or col >= self.n:
            raise IndexError("行列索引越界")
        if row == col:
            self.a[row] += val
            return


        start = int(self.ija[row])
        end = int(self.ija[row + 1]) if row + 1 < self.n else self.nz

        for k in range(start, end):
            if self.ija[k] == col:
                self.a[k] += val
                return

        raise NotImplementedError(
            "动态插入需维护 R8RI 结构，请在构造时预分配所有非零元"
        )

    def mv(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.n:
            raise ValueError(f"向量维度 {x.shape[0]} 不等于矩阵阶数 {self.n}")
        b = self.a[0:self.n] * x
        for i in range(self.n):
            k_start = int(self.ija[i])
            k_end = int(self.ija[i + 1]) if i + 1 < self.n else self.nz
            for k in range(k_start, k_end):
                j = int(self.ija[k])
                b[i] += self.a[k] * x[j]
        return b

    def to_dense(self):
        dense = np.zeros((self.n, self.n), dtype=np.float64)
        for k in range(self.n):
            dense[k, k] = self.a[k]
        for i in range(self.n):
            k_start = int(self.ija[i])
            k_end = int(self.ija[i + 1]) if i + 1 < self.n else self.nz
            for k in range(k_start, k_end):
                j = int(self.ija[k])
                dense[i, j] = self.a[k]
        return dense


def build_climate_laplacian_sparse(nx, ny):
    n = nx * ny

    nz = n + 1 + 4 * n
    mat = SparseClimateMatrix(n, nz)
    diag = np.zeros(n, dtype=np.float64)
    off_idx = n + 1
    row_starts = np.zeros(n + 1, dtype=np.int64)
    row_starts[0] = n + 1


    counts = np.zeros(n, dtype=np.int64)
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            c = 0
            if i > 0:
                c += 1
            if i + 1 < nx:
                c += 1
            if j > 0:
                c += 1
            if j + 1 < ny:
                c += 1
            counts[idx] = c

    for idx in range(1, n):
        row_starts[idx] = row_starts[idx - 1] + counts[idx - 1]
    row_starts[n] = nz

    mat.ija[0:n] = row_starts[0:n]
    mat.ija[n] = nz

    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            diag[idx] = 4.0
            k = row_starts[idx]
            if i > 0:
                mat.ija[k] = idx - 1
                mat.a[k] = -1.0
                k += 1
            if i + 1 < nx:
                mat.ija[k] = idx + 1
                mat.a[k] = -1.0
                k += 1
            if j > 0:
                mat.ija[k] = idx - nx
                mat.a[k] = -1.0
                k += 1
            if j + 1 < ny:
                mat.ija[k] = idx + nx
                mat.a[k] = -1.0
                k += 1

    mat.a[0:n] = diag
    return mat


def test_sparse():
    mat = build_climate_laplacian_sparse(3, 3)
    x = np.ones(9)
    b = mat.mv(x)
    assert b.shape == (9,), "维度错误"
    print("sparse_climate_matrix 自测试通过")


if __name__ == "__main__":
    test_sparse()
