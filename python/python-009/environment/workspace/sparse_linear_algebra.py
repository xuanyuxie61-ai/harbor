
import numpy as np
from typing import Tuple, Optional


class CRSMatrix:

    def __init__(self, m: int, n: int, nz: int,
                 row_ptr: np.ndarray,
                 col: np.ndarray,
                 val: np.ndarray):
        if m <= 0 or n <= 0:
            raise ValueError(f"矩阵维度必须为正，得到 m={m}, n={n}")
        if nz < 0:
            raise ValueError(f"非零元个数不能为负，得到 nz={nz}")
        if row_ptr.shape[0] != m + 1:
            raise ValueError(f"row_ptr 长度应为 {m+1}，得到 {row_ptr.shape[0]}")
        if col.shape[0] != nz or val.shape[0] != nz:
            raise ValueError(f"col 和 val 长度应为 {nz}")
        if row_ptr[0] != 0 or row_ptr[m] != nz:
            raise ValueError(f"row_ptr 边界错误: row_ptr[0]={row_ptr[0]}, row_ptr[m]={row_ptr[m]}")
        if np.any(col < 0) or np.any(col >= n):
            raise ValueError(f"列索引越界，有效范围 [0, {n-1}]")
        if not np.all(np.diff(row_ptr) >= 0):
            raise ValueError("row_ptr 必须非递减")

        self.m = m
        self.n = n
        self.nz = nz
        self.row_ptr = row_ptr.astype(np.int64)
        self.col = col.astype(np.int64)
        self.val = val.astype(np.float64)

    def multiply(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).reshape(-1)
        if x.shape[0] != self.n:
            raise ValueError(f"向量维度不匹配: x 长度 {x.shape[0]}，矩阵列数 {self.n}")

        y = np.zeros(self.m, dtype=np.float64)
        for i in range(self.m):
            for k in range(self.row_ptr[i], self.row_ptr[i + 1]):
                y[i] += self.val[k] * x[self.col[k]]
        return y

    def transpose_multiply(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).reshape(-1)
        if x.shape[0] != self.m:
            raise ValueError(f"向量维度不匹配: x 长度 {x.shape[0]}，矩阵行数 {self.m}")

        y = np.zeros(self.n, dtype=np.float64)
        for i in range(self.m):
            xi = x[i]
            for k in range(self.row_ptr[i], self.row_ptr[i + 1]):
                y[self.col[k]] += self.val[k] * xi
        return y

    def to_dense(self) -> np.ndarray:
        A = np.zeros((self.m, self.n), dtype=np.float64)
        for i in range(self.m):
            for k in range(self.row_ptr[i], self.row_ptr[i + 1]):
                A[i, self.col[k]] = self.val[k]
        return A

    @staticmethod
    def from_dense(A_dense: np.ndarray, drop_tolerance: float = 0.0) -> "CRSMatrix":
        A_dense = np.asarray(A_dense, dtype=np.float64)
        m, n = A_dense.shape
        row_ptr = [0]
        col_list = []
        val_list = []

        for i in range(m):
            count = 0
            for j in range(n):
                if abs(A_dense[i, j]) > drop_tolerance:
                    col_list.append(j)
                    val_list.append(A_dense[i, j])
                    count += 1
            row_ptr.append(row_ptr[-1] + count)

        return CRSMatrix(
            m, n, len(val_list),
            np.array(row_ptr, dtype=np.int64),
            np.array(col_list, dtype=np.int64),
            np.array(val_list, dtype=np.float64)
        )


def crs_gmres(crs_A: CRSMatrix, b: np.ndarray, x0: Optional[np.ndarray] = None,
              tol: float = 1e-10, max_iter: int = 1000, restart: int = 50) -> Tuple[np.ndarray, int, float]:
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = crs_A.n
    if b.shape[0] != crs_A.m:
        raise ValueError("右端项维度与矩阵不匹配")

    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = np.asarray(x0, dtype=np.float64).reshape(-1).copy()

    bnrm = np.linalg.norm(b)
    if bnrm < 1e-30:
        return np.zeros(n, dtype=np.float64), 0, 0.0

    total_iters = 0

    for _outer in range(max_iter // restart + 1):
        if total_iters >= max_iter:
            break

        r = b - crs_A.multiply(x)
        beta = np.linalg.norm(r)
        if beta / bnrm <= tol:
            break

        V = np.zeros((n, restart + 1), dtype=np.float64)
        H = np.zeros((restart + 1, restart), dtype=np.float64)
        cs = np.zeros(restart, dtype=np.float64)
        sn = np.zeros(restart, dtype=np.float64)
        e1 = np.zeros(restart + 1, dtype=np.float64)
        e1[0] = beta

        V[:, 0] = r / beta

        for j in range(restart):
            if total_iters >= max_iter:
                break
            total_iters += 1

            w = crs_A.multiply(V[:, j])
            for i in range(j + 1):
                H[i, j] = np.dot(w, V[:, i])
                w -= H[i, j] * V[:, i]
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] < 1e-30:
                V[:, j + 1] = 0.0
            else:
                V[:, j + 1] = w / H[j + 1, j]

            for i in range(j):
                temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = temp

            denom = np.sqrt(H[j, j]**2 + H[j + 1, j]**2)
            if denom < 1e-30:
                cs[j] = 1.0
                sn[j] = 0.0
            else:
                cs[j] = H[j, j] / denom
                sn[j] = H[j + 1, j] / denom

            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0
            temp = cs[j] * e1[j]
            e1[j + 1] = -sn[j] * e1[j]
            e1[j] = temp

            error = abs(e1[j + 1]) / bnrm
            if error <= tol:
                y = np.linalg.lstsq(H[:j + 1, :j + 1], e1[:j + 1], rcond=None)[0]
                x += V[:, :j + 1] @ y
                return x, total_iters, error

        y = np.linalg.lstsq(H[:restart, :restart], e1[:restart], rcond=None)[0]
        x += V[:, :restart] @ y

    r = b - crs_A.multiply(x)
    final_res = np.linalg.norm(r) / bnrm
    return x, total_iters, final_res


def crs_ilu_preconditioner(crs_A: CRSMatrix, fill_level: int = 0) -> CRSMatrix:
    if crs_A.m != crs_A.n:
        raise ValueError("ILU仅适用于方阵")
    n = crs_A.m

    A_dense = crs_A.to_dense()
    L = np.eye(n, dtype=np.float64)
    U = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i, n):
            s = A_dense[i, j]
            for k in range(i):
                s -= L[i, k] * U[k, j]
            U[i, j] = s

        for j in range(i + 1, n):
            s = A_dense[j, i]
            for k in range(i):
                s -= L[j, k] * U[k, i]
            if abs(U[i, i]) > 1e-30:
                L[j, i] = s / U[i, i]
            else:
                L[j, i] = 0.0

    M = L @ U
    return CRSMatrix.from_dense(M)
