"""
sparse_climate_matrix.py

稀疏气候矩阵运算模块（基于 992_r8ri 核心算法改造）

原项目 r8ri 提供了 Row-Indexed 稀疏矩阵存储格式及其运算。
在本气候归因框架中，我们将其用于存储和操作大规模气候网格的
协方差矩阵与空间离散算子。

核心公式：
- 稀疏矩阵-向量乘法：
    b_i = a_{ii} x_i + sum_{k=ija(i)}^{ija(i+1)-1} a_k x_{ija(k)}
- R8RI 存储格式：
  * A[0:N] 存储对角元
  * IJA[0:N+1] 存储每行第一个非对角元的索引
  * A[N+1:NZ] 存储非对角元值
  * IJA[N+1:NZ] 存储非对角元的列号
"""

import numpy as np


class SparseClimateMatrix:
    """基于 R8RI 格式的大规模气候稀疏矩阵。"""

    def __init__(self, n, nz):
        """
        初始化稀疏矩阵。

        Parameters
        ----------
        n : int
            矩阵阶数（气候网格点数）。
        nz : int
            R8RI 存储所需总长度 = N + 1 + 非零非对角元个数。
        """
        if n < 1:
            raise ValueError("矩阵阶数 n 必须 >= 1")
        if nz < n + 1:
            raise ValueError("nz 必须 >= n + 1")
        self.n = n
        self.nz = nz
        self.ija = np.zeros(nz, dtype=np.int64)
        self.a = np.zeros(nz, dtype=np.float64)
        # R8RI 格式约定：ija[0] = n + 2（Python 中对应 1-based 的 N+2）
        # 但在 0-based 索引下，我们令 ija[0] = n + 1 表示对角元后第一个非对角位置
        self.ija[0] = n + 1
        self.ija[n] = nz

    def set_diagonal(self, diag_vals):
        """设置对角元。"""
        diag_vals = np.asarray(diag_vals, dtype=np.float64)
        if diag_vals.shape[0] != self.n:
            raise ValueError("对角元长度必须等于 n")
        self.a[0:self.n] = diag_vals

    def add_off_diagonal(self, row, col, val):
        """
        添加非对角元（按行顺序插入）。
        此方法假设调用者已正确维护 ija 的索引结构。
        """
        if row < 0 or row >= self.n or col < 0 or col >= self.n:
            raise IndexError("行列索引越界")
        if row == col:
            self.a[row] += val
            return
        # 简化版：直接追加到尾部（适用于初始化阶段）
        # 实际 R8RI 要求按行有序，此处做简化但保持格式正确
        start = int(self.ija[row])
        end = int(self.ija[row + 1]) if row + 1 < self.n else self.nz
        # 查找该位置是否已有值
        for k in range(start, end):
            if self.ija[k] == col:
                self.a[k] += val
                return
        # 如果未找到，需要扩展（简化处理：此处仅适用于初始化）
        raise NotImplementedError(
            "动态插入需维护 R8RI 结构，请在构造时预分配所有非零元"
        )

    def mv(self, x):
        """
        稀疏矩阵-向量乘法 b = A * x。

        基于 r8ri_mv 核心算法：
            b_i = a_{ii} * x_i + sum_{k=ija(i)}^{ija(i+1)-1} a_k * x_{ija(k)}
        """
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
        """
        将 R8RI 稀疏矩阵转为稠密矩阵（基于 r8ri_to_r8ge）。
        仅用于小规模验证，大规模气候网格避免使用。
        """
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
    """
    构建二维气候网格的离散 Laplacian 稀疏矩阵。

    对于格点 (i,j)，其 Laplacian 为：
        L u_{i,j} = 4 u_{i,j} - u_{i+1,j} - u_{i-1,j} - u_{i,j+1} - u_{i,j-1}

    用于空间平滑和扩散过程建模。
    """
    n = nx * ny
    # 每个内部点最多有 4 个非对角邻居 + 1 个对角
    nz = n + 1 + 4 * n
    mat = SparseClimateMatrix(n, nz)
    diag = np.zeros(n, dtype=np.float64)
    off_idx = n + 1
    row_starts = np.zeros(n + 1, dtype=np.int64)
    row_starts[0] = n + 1

    # 先计算每行的非零元数量
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
    """自测试函数。"""
    mat = build_climate_laplacian_sparse(3, 3)
    x = np.ones(9)
    b = mat.mv(x)
    assert b.shape == (9,), "维度错误"
    print("sparse_climate_matrix 自测试通过")


if __name__ == "__main__":
    test_sparse()
