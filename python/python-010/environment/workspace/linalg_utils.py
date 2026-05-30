
import numpy as np
from typing import Tuple


class DenseLU:

    def __init__(self, A: np.ndarray):
        self.n = A.shape[0]
        if A.shape[0] != A.shape[1]:
            raise ValueError("PLU 分解仅适用于方阵")
        self.A_lu, self.pivot, self.info = self._plu_factorize(A.copy())

    def _plu_factorize(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        n = self.n
        A_lu = A.astype(float)
        pivot = np.zeros(n, dtype=int)
        info = 0

        for k in range(n - 1):

            l = k
            for i in range(k + 1, n):
                if abs(A_lu[l, k]) < abs(A_lu[i, k]):
                    l = i
            pivot[k] = l

            if abs(A_lu[l, k]) < 1e-15:
                info = k + 1
                continue


            if l != k:
                A_lu[[l, k], k:] = A_lu[[k, l], k:]


            A_lu[k + 1 :, k] = -A_lu[k + 1 :, k] / A_lu[k, k]


            for j in range(k + 1, n):
                A_lu[k + 1 :, j] += A_lu[k + 1 :, k] * A_lu[k, j]

        pivot[n - 1] = n - 1
        if abs(A_lu[n - 1, n - 1]) < 1e-15:
            info = n

        return A_lu, pivot, info

    def solve(self, b: np.ndarray) -> np.ndarray:
        if self.info != 0:
            raise RuntimeError(f"PLU 分解失败，info = {self.info}")
        n = self.n
        x = b.astype(float).copy()


        for k in range(n - 1):
            l = self.pivot[k]
            if l != k:
                x[k], x[l] = x[l], x[k]




        for k in range(n - 1):
            x[k + 1 :] += x[k] * self.A_lu[k + 1 :, k]



        for k in range(n - 1, -1, -1):
            if abs(self.A_lu[k, k]) < 1e-15:
                raise RuntimeError("零主元，无法回代")

            if k < n - 1:
                x[k] -= np.dot(self.A_lu[k, k + 1 :], x[k + 1 :])
            x[k] /= self.A_lu[k, k]

        return x


class SparseCRS:

    def __init__(
        self, m: int, n: int, row_ptr: np.ndarray, col_idx: np.ndarray, val: np.ndarray
    ):
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
        A = np.zeros((self.m, self.n), dtype=float)
        for i in range(self.m):
            for k in range(self.row_ptr[i], self.row_ptr[i + 1]):
                j = self.col_idx[k]
                A[i, j] = self.val[k]
        return A


def build_laplacian_1d(n: int, dx: float) -> SparseCRS:
    if n <= 2:
        raise ValueError("n 必须大于 2")
    row_ptr = np.zeros(n + 1, dtype=int)
    col_idx = []
    val = []
    inv_dx2 = 1.0 / (dx * dx)
    for i in range(n):
        row_ptr[i] = len(val)

        col_idx.append(i)
        val.append(-2.0 * inv_dx2)

        if i > 0:
            col_idx.append(i - 1)
            val.append(inv_dx2)
        if i < n - 1:
            col_idx.append(i + 1)
            val.append(inv_dx2)
    row_ptr[n] = len(val)
    return SparseCRS(n, n, row_ptr, np.array(col_idx), np.array(val))


def build_laplacian_3d_poisson(N: int, L: float) -> Tuple[SparseCRS, float]:
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

                col_idx.append(idx(i, j, k))
                val.append(center)

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
    n = len(b)
    if len(a) != n or len(c) != n or len(d) != n:
        raise ValueError("三对角向量维度不一致")
    cp = c.copy()
    bp = b.copy()
    dp = d.copy()

    for i in range(1, n):
        if abs(bp[i - 1]) < 1e-15:
            raise RuntimeError("Thomas 算法：零主元")
        w = a[i] / bp[i - 1]
        bp[i] -= w * cp[i - 1]
        dp[i] -= w * dp[i - 1]

    x = np.zeros(n, dtype=float)
    if abs(bp[n - 1]) < 1e-15:
        raise RuntimeError("Thomas 算法：零主元")
    x[n - 1] = dp[n - 1] / bp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (dp[i] - cp[i] * x[i + 1]) / bp[i]
    return x


if __name__ == "__main__":

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
