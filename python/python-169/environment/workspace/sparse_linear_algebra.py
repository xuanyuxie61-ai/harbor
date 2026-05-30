
import numpy as np
from typing import Tuple, Optional, Callable






def dense_to_crs(A: np.ndarray):
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    val = []
    col_idx = []
    row_ptr = [0]
    for i in range(m):
        for j in range(n):
            if abs(A[i, j]) > 1e-14:
                val.append(A[i, j])
                col_idx.append(j)
        row_ptr.append(len(val))
    return np.array(val, dtype=float), np.array(col_idx, dtype=int), np.array(row_ptr, dtype=int)


def crs_matvec(val, col_idx, row_ptr, x):
    x = np.asarray(x, dtype=float)
    m = row_ptr.size - 1
    y = np.zeros(m, dtype=float)
    for i in range(m):
        for k in range(row_ptr[i], row_ptr[i + 1]):
            y[i] += val[k] * x[col_idx[k]]
    return y






def ilu0_crs(val, col_idx, row_ptr):
    m = row_ptr.size - 1

    A_dense = np.zeros((m, m), dtype=float)
    for i in range(m):
        for k in range(row_ptr[i], row_ptr[i + 1]):
            j = col_idx[k]
            A_dense[i, j] = val[k]
    L = np.eye(m, dtype=float)
    U = A_dense.copy()
    for i in range(m):
        if abs(U[i, i]) < 1e-14:
            U[i, i] = 1e-14
        for j in range(i + 1, m):
            if abs(U[j, i]) > 1e-14:
                factor = U[j, i] / U[i, i]
                L[j, i] = factor
                U[j, i:] = U[j, i:] - factor * U[i, i:]

    L_val, L_col, L_row = dense_to_crs(L)
    U_val, U_col, U_row = dense_to_crs(U)
    return L_val, L_col, L_row, U_val, U_col, U_row


def ilu_solve(L_val, L_col, L_row, U_val, U_col, U_row, b):
    m = L_row.size - 1
    b = np.asarray(b, dtype=float)

    y = np.zeros(m, dtype=float)
    for i in range(m):
        s = b[i]
        for k in range(L_row[i], L_row[i + 1]):
            j = L_col[k]
            if j < i:
                s -= L_val[k] * y[j]
        diag = 1.0
        for k in range(L_row[i], L_row[i + 1]):
            if L_col[k] == i:
                diag = L_val[k]
                break
        if abs(diag) < 1e-14:
            diag = 1e-14
        y[i] = s / diag

    x = np.zeros(m, dtype=float)
    for i in range(m - 1, -1, -1):
        s = y[i]
        diag = 0.0
        for k in range(U_row[i], U_row[i + 1]):
            j = U_col[k]
            if j > i:
                s -= U_val[k] * x[j]
            elif j == i:
                diag = U_val[k]
        if abs(diag) < 1e-14:
            diag = 1e-14
        x[i] = s / diag
    return x






def gmres_solve(A, b, x0=None, restart: int = 20, max_iter: int = 200,
                tol: float = 1e-8, preconditioner: Optional[Callable] = None,
                use_crs: bool = False):
    b = np.asarray(b, dtype=float).reshape(-1)
    n = b.size
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).reshape(-1).copy()

    if use_crs and isinstance(A, np.ndarray):
        val, col_idx, row_ptr = dense_to_crs(A)
        def A_op(v):
            return crs_matvec(val, col_idx, row_ptr, v)
    elif isinstance(A, np.ndarray):
        def A_op(v):
            return A @ v
    else:
        A_op = A

    if preconditioner is None:
        def M_inv(r):
            return r
    else:
        M_inv = preconditioner

    residual_norms = []
    beta = np.linalg.norm(b - A_op(x))
    residual_norms.append(beta)
    if beta < tol:
        return x, 0, residual_norms

    for outer in range(max_iter // restart):
        r = b - A_op(x)
        z = M_inv(r)
        beta = np.linalg.norm(z)
        if beta < tol:
            break
        V = np.zeros((n, restart + 1), dtype=float)
        H = np.zeros((restart + 1, restart), dtype=float)
        cs = np.zeros(restart, dtype=float)
        sn = np.zeros(restart, dtype=float)
        e1 = np.zeros(restart + 1, dtype=float)
        e1[0] = beta
        V[:, 0] = z / beta

        converged = False
        for j in range(restart):
            w = A_op(V[:, j])
            w = M_inv(w)

            for i in range(j + 1):
                H[i, j] = np.dot(w, V[:, i])
                w = w - H[i, j] * V[:, i]
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] < 1e-14:

                H[j + 1, j] = 1e-14
            V[:, j + 1] = w / H[j + 1, j]


            for i in range(j):
                temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = temp

            if abs(H[j + 1, j]) < 1e-14 and abs(H[j, j]) < 1e-14:
                cs[j] = 1.0
                sn[j] = 0.0
            else:
                denom = np.sqrt(H[j, j]**2 + H[j + 1, j]**2)
                cs[j] = H[j, j] / denom
                sn[j] = H[j + 1, j] / denom

            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0
            e1[j + 1] = -sn[j] * e1[j]
            e1[j] = cs[j] * e1[j]

            res = abs(e1[j + 1])
            residual_norms.append(res)
            if res < tol:

                y = np.linalg.solve(H[:j + 1, :j + 1], e1[:j + 1])
                x = x + V[:, :j + 1] @ y
                converged = True
                break

        if not converged:

            y = np.linalg.solve(H[:restart, :restart], e1[:restart])
            x = x + V[:, :restart] @ y

        if converged:
            break

    return x, len(residual_norms), residual_norms






def cgne_solve_wrapper(A, b, x0=None, max_iter: int = 500, tol: float = 1e-10):
    b = np.asarray(b, dtype=float).reshape(-1)
    if isinstance(A, np.ndarray):
        m, n = A.shape
        if x0 is None:
            x = np.zeros(n, dtype=float)
        else:
            x = np.asarray(x0, dtype=float).reshape(-1).copy()
        r = b - A @ x
        z = A.T @ r
    else:


        if hasattr(A, 'T_matvec'):
            A_T = A.T_matvec
        else:
            raise ValueError("CGNE需要A^T操作，请提供.T_matvec属性或使用稠密矩阵")
        n = b.size
        if x0 is None:
            x = np.zeros(n, dtype=float)
        else:
            x = np.asarray(x0, dtype=float).reshape(-1).copy()
        r = b - A(x)
        z = A_T(r)










    raise NotImplementedError("Hole 3: 请实现CGNE核心迭代算法")






class SparseSolver:

    def __init__(self, A: np.ndarray):
        self.A = np.asarray(A, dtype=float)
        self.n = self.A.shape[0]

        if self.A.shape[0] == self.A.shape[1]:
            val, col_idx, row_ptr = dense_to_crs(self.A)
            self._L_val, self._L_col, self._L_row, self._U_val, self._U_col, self._U_row = ilu0_crs(val, col_idx, row_ptr)
            self.has_precond = True
        else:
            self.has_precond = False

    def preconditioner(self, r):
        if self.has_precond:
            return ilu_solve(self._L_val, self._L_col, self._L_row,
                             self._U_val, self._U_col, self._U_row, r)
        return r

    def solve_gmres(self, b, x0=None, restart=30, max_iter=300, tol=1e-8):
        if self.has_precond:
            precond = lambda r: self.preconditioner(r)
        else:
            precond = None
        return gmres_solve(self.A, b, x0=x0, restart=restart,
                           max_iter=max_iter, tol=tol,
                           preconditioner=precond)

    def solve_cgne(self, b, x0=None, max_iter=500, tol=1e-10):
        return cgne_solve_wrapper(self.A, b, x0=x0, max_iter=max_iter, tol=tol)
