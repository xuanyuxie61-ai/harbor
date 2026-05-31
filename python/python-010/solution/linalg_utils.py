"""
linalg_utils.py
===============
线性代数工具模块

融入 r8ge（稠密矩阵 LU 分解）与 r8crs（稀疏矩阵压缩行存储及矩阵-向量乘法）的核心算法，
为 Poisson 方程求解与引力势计算提供底层线性代数支撑。

核心公式
--------
稠密矩阵 PLU 分解:
    A = P L U
    其中 P 为置换矩阵，L 为单位下三角，U 为上三角。

稀疏矩阵 CRS 格式:
    对于 m×n 矩阵，非零元存储为:
        val[k] : 第 k 个非零元的值
        col[k] : 第 k 个非零元的列索引
        row[i] : 第 i 行第一个非零元在 val 中的索引

    矩阵-向量乘法:
        y_i = Σ_{k=row[i]}^{row[i+1]-1} val[k] · x_{col[k]}
"""

import numpy as np
from typing import Tuple


class DenseLU:
    """
    稠密矩阵的 PLU 分解（融入 r8ge_fa 的 LINPACK 风格算法）。
    """

    def __init__(self, A: np.ndarray):
        """
        Parameters
        ----------
        A : np.ndarray, shape (n, n)
            待分解的方阵
        """
        self.n = A.shape[0]
        if A.shape[0] != A.shape[1]:
            raise ValueError("PLU 分解仅适用于方阵")
        self.A_lu, self.pivot, self.info = self._plu_factorize(A.copy())

    def _plu_factorize(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        执行 PLU 分解。

        Returns
        -------
        A_lu : np.ndarray
            上三角 U 与乘子 L 的紧凑存储
        pivot : np.ndarray
            主元行索引
        info : int
            0 表示成功，非零表示在第 info 步出现零主元
        """
        n = self.n
        A_lu = A.astype(float)
        pivot = np.zeros(n, dtype=int)
        info = 0

        for k in range(n - 1):
            # 选主元
            l = k
            for i in range(k + 1, n):
                if abs(A_lu[l, k]) < abs(A_lu[i, k]):
                    l = i
            pivot[k] = l

            if abs(A_lu[l, k]) < 1e-15:
                info = k + 1
                continue

            # 交换行
            if l != k:
                A_lu[[l, k], k:] = A_lu[[k, l], k:]

            # 计算乘子（LINPACK 风格：存储负的乘子）
            A_lu[k + 1 :, k] = -A_lu[k + 1 :, k] / A_lu[k, k]

            # 行消去
            for j in range(k + 1, n):
                A_lu[k + 1 :, j] += A_lu[k + 1 :, k] * A_lu[k, j]

        pivot[n - 1] = n - 1
        if abs(A_lu[n - 1, n - 1]) < 1e-15:
            info = n

        return A_lu, pivot, info

    def solve(self, b: np.ndarray) -> np.ndarray:
        """
        利用 PLU 分解求解 Ax = b。

        算法分为前向代入（解 Ly = Pb）与回代（解 Ux = y）。
        注意 A_lu 的严格下三角存储的是 -L_{i,j}（LINPACK 风格）。
        """
        if self.info != 0:
            raise RuntimeError(f"PLU 分解失败，info = {self.info}")
        n = self.n
        x = b.astype(float).copy()

        # 应用置换: x = P b
        for k in range(n - 1):
            l = self.pivot[k]
            if l != k:
                x[k], x[l] = x[l], x[k]

        # 前向代入: L y = P b
        # 由于 A_lu[i,j] (i>j) = -L_{i,j}, 因此:
        # y_i = (Pb)_i - Σ_{j<i} L_{i,j} y_j = (Pb)_i + Σ_{j<i} A_lu[i,j] y_j
        for k in range(n - 1):
            x[k + 1 :] += x[k] * self.A_lu[k + 1 :, k]

        # 回代: U x = y
        # U 存储在 A_lu 的对角线及以上
        for k in range(n - 1, -1, -1):
            if abs(self.A_lu[k, k]) < 1e-15:
                raise RuntimeError("零主元，无法回代")
            # x_k = y_k - Σ_{j>k} U_{k,j} x_j
            if k < n - 1:
                x[k] -= np.dot(self.A_lu[k, k + 1 :], x[k + 1 :])
            x[k] /= self.A_lu[k, k]

        return x


class SparseCRS:
    """
    稀疏矩阵的压缩行存储（CRS）格式（融入 r8crs 核心算法）。
    """

    def __init__(
        self, m: int, n: int, row_ptr: np.ndarray, col_idx: np.ndarray, val: np.ndarray
    ):
        """
        Parameters
        ----------
        m, n : int
            矩阵维度
        row_ptr : np.ndarray, shape (m+1,)
            行指针
        col_idx : np.ndarray
            列索引
        val : np.ndarray
            非零元值
        """
        self.m = m
        self.n = n
        self.row_ptr = row_ptr.astype(int)
        self.col_idx = col_idx.astype(int)
        self.val = val.astype(float)
        self.nz = len(val)
        if self.row_ptr[0] != 0:
            raise ValueError("row_ptr[0] 必须为 0")
        if self.row_ptr[-1] != self.nz:
            raise ValueError("row_ptr[-1] 必须等于非零元个数")

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """
        稀疏矩阵-向量乘法 y = A @ x。

        公式:
            y_i = Σ_{k=row_ptr[i]}^{row_ptr[i+1]-1} val[k] * x[col_idx[k]]
        """
        if len(x) != self.n:
            raise ValueError(f"向量维度不匹配: {len(x)} != {self.n}")
        y = np.zeros(self.m, dtype=float)
        for i in range(self.m):
            for k in range(self.row_ptr[i], self.row_ptr[i + 1]):
                j = self.col_idx[k]
                y[i] += self.val[k] * x[j]
        return y

    @classmethod
    def from_dense(cls, A: np.ndarray, tol: float = 1e-15) -> "SparseCRS":
        """
        从稠密矩阵构造 CRS 格式，忽略绝对值小于 tol 的元素。
        """
        m, n = A.shape
        row_ptr = [0]
        col_idx = []
        val = []
        for i in range(m):
            for j in range(n):
                if abs(A[i, j]) > tol:
                    col_idx.append(j)
                    val.append(A[i, j])
            row_ptr.append(len(val))
        return cls(m, n, np.array(row_ptr), np.array(col_idx), np.array(val))

    def to_dense(self) -> np.ndarray:
        """
        转回稠密矩阵（主要用于验证与测试）。
        """
        A = np.zeros((self.m, self.n), dtype=float)
        for i in range(self.m):
            for k in range(self.row_ptr[i], self.row_ptr[i + 1]):
                j = self.col_idx[k]
                A[i, j] = self.val[k]
        return A


def build_laplacian_1d(n: int, dx: float) -> SparseCRS:
    """
    构造一维离散 Laplace 算子的 CRS 稀疏矩阵（Dirichlet 边界）。

    离散格式（中心差分）:
        ∇²φ ≈ (φ_{i-1} - 2φ_i + φ_{i+1}) / dx²

    矩阵形式:
        L_{i,i} = -2/dx²
        L_{i,i±1} = 1/dx²
    """
    if n <= 2:
        raise ValueError("n 必须大于 2")
    row_ptr = np.zeros(n + 1, dtype=int)
    col_idx = []
    val = []
    inv_dx2 = 1.0 / (dx * dx)
    for i in range(n):
        row_ptr[i] = len(val)
        # 对角元
        col_idx.append(i)
        val.append(-2.0 * inv_dx2)
        # 次对角
        if i > 0:
            col_idx.append(i - 1)
            val.append(inv_dx2)
        if i < n - 1:
            col_idx.append(i + 1)
            val.append(inv_dx2)
    row_ptr[n] = len(val)
    return SparseCRS(n, n, row_ptr, np.array(col_idx), np.array(val))


def build_laplacian_3d_poisson(N: int, L: float) -> Tuple[SparseCRS, float]:
    """
    构造三维笛卡尔网格上的离散 Poisson 算子 CRS 稀疏矩阵。

    采用七点 stencil:
        ∇²φ ≈ (φ_{i-1,j,k} + φ_{i+1,j,k} + φ_{i,j-1,k} + φ_{i,j+1,k}
               + φ_{i,j,k-1} + φ_{i,j,k+1} - 6φ_{i,j,k}) / dx²

    Parameters
    ----------
    N : int
        每维网格数
    L : float
        盒子边长

    Returns
    -------
    L_op : SparseCRS
        N³ × N³ 稀疏矩阵
    dx : float
        网格间距
    """
    if N <= 2:
        raise ValueError("N 必须大于 2")
    dx = L / N
    n_total = N * N * N
    row_ptr = np.zeros(n_total + 1, dtype=int)
    col_idx = []
    val = []
    inv_dx2 = 1.0 / (dx * dx)

    def idx(i, j, k):
        return i * N * N + j * N + k

    for i in range(N):
        for j in range(N):
            for k in range(N):
                row_ptr[idx(i, j, k)] = len(val)
                center = -6.0 * inv_dx2
                # 自身
                col_idx.append(idx(i, j, k))
                val.append(center)
                # 六个邻居
                neighbors = []
                if i > 0:
                    neighbors.append(idx(i - 1, j, k))
                if i < N - 1:
                    neighbors.append(idx(i + 1, j, k))
                if j > 0:
                    neighbors.append(idx(i, j - 1, k))
                if j < N - 1:
                    neighbors.append(idx(i, j + 1, k))
                if k > 0:
                    neighbors.append(idx(i, j, k - 1))
                if k < N - 1:
                    neighbors.append(idx(i, j, k + 1))
                for nb in neighbors:
                    col_idx.append(nb)
                    val.append(inv_dx2)

    row_ptr[n_total] = len(val)
    return SparseCRS(n_total, n_total, row_ptr, np.array(col_idx), np.array(val)), dx


def solve_tridiagonal(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> np.ndarray:
    """
    Thomas 算法求解三对角方程组:
        a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i

    时间复杂度 O(n)，空间复杂度 O(n)。
    """
    n = len(b)
    if len(a) != n or len(c) != n or len(d) != n:
        raise ValueError("三对角向量维度不一致")
    cp = c.copy()
    bp = b.copy()
    dp = d.copy()
    # 前向消去
    for i in range(1, n):
        if abs(bp[i - 1]) < 1e-15:
            raise RuntimeError("Thomas 算法：零主元")
        w = a[i] / bp[i - 1]
        bp[i] -= w * cp[i - 1]
        dp[i] -= w * dp[i - 1]
    # 回代
    x = np.zeros(n, dtype=float)
    if abs(bp[n - 1]) < 1e-15:
        raise RuntimeError("Thomas 算法：零主元")
    x[n - 1] = dp[n - 1] / bp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (dp[i] - cp[i] * x[i + 1]) / bp[i]
    return x


if __name__ == "__main__":
    # 自检
    A = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]])
    lu = DenseLU(A)
    b = np.array([1.0, 2.0, 3.0])
    x = lu.solve(b)
    print("DenseLU 解:", x)
    print("残差:", A @ x - b)

    crs = SparseCRS.from_dense(A)
    y = crs.matvec(b)
    print("SparseCRS matvec:", y)
    print("与稠密乘法差:", A @ b - y)
