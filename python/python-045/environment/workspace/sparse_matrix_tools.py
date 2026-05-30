#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def ge_to_st(Age):
    Age = np.asarray(Age)
    m, n = Age.shape
    nz_num = np.count_nonzero(Age)
    ist = np.zeros(nz_num, dtype=np.int32)
    jst = np.zeros(nz_num, dtype=np.int32)
    Ast = np.zeros(nz_num, dtype=Age.dtype)

    k = 0
    for j in range(n):
        for i in range(m):
            if Age[i, j] != 0.0:
                ist[k] = i
                jst[k] = j
                Ast[k] = Age[i, j]
                k += 1
    return nz_num, ist, jst, Ast


def st_to_dense(ist, jst, Ast, m, n):
    Age = np.zeros((m, n), dtype=Ast.dtype)
    for i, j, v in zip(ist, jst, Ast):
        Age[i, j] = v
    return Age


def sparse_matvec(ist, jst, Ast, x):
    x = np.asarray(x)
    y = np.zeros(len(x), dtype=np.result_type(Ast.dtype, x.dtype))
    for i, j, v in zip(ist, jst, Ast):
        y[i] += v * x[j]
    return y


class DenseLUSolver:

    def __init__(self, A):
        self.A = np.array(A, dtype=np.float64, copy=True)
        self.n = self.A.shape[0]
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("矩阵必须是方阵")
        self._lu = None
        self._ipvt = None
        self._info = None
        self._factorized = False

    def _daxpy(self, n, sa, x, incx, y, incy):
        if n <= 0 or sa == 0.0:
            return y
        if incx == 1 and incy == 1:
            y[:n] = y[:n] + sa * x[:n]
        else:
            ix, iy = 0, 0
            if incx < 0:
                ix = (-n + 1) * incx
            if incy < 0:
                iy = (-n + 1) * incy
            for _ in range(n):
                y[iy] = y[iy] + sa * x[ix]
                ix += incx
                iy += incy
        return y

    def _idamax(self, n, x, incx):
        if n <= 0:
            return 0
        if n == 1 or incx == 1:
            imax = 0
            xmax = abs(x[0])
            for i in range(1, n):
                if abs(x[i]) > xmax:
                    imax = i
                    xmax = abs(x[i])
            return imax
        else:
            ix = 0 if incx >= 0 else (-n + 1) * incx
            imax = 0
            xmax = abs(x[ix])
            ix += incx
            for i in range(1, n):
                if abs(x[ix]) > xmax:
                    imax = i
                    xmax = abs(x[ix])
                ix += incx
            return imax

    def dgefa(self):
        info = 0
        ipvt = np.zeros(self.n, dtype=np.int32)
        a = self._lu if self._lu is not None else self.A.copy()

        for k in range(self.n - 1):

            l = self._idamax(self.n - k, a[k:self.n, k], 1) + k
            ipvt[k] = l

            if abs(a[l, k]) < 1e-15:
                info = k + 1
                continue


            if l != k:
                a[[l, k], k:self.n] = a[[k, l], k:self.n]


            a[k + 1:self.n, k] = -a[k + 1:self.n, k] / a[k, k]


            for j in range(k + 1, self.n):
                t = a[k, j]
                if l != k:
                    a[l, j], a[k, j] = a[k, j], a[l, j]
                a[k + 1:self.n, j] = self._daxpy(
                    self.n - k - 1, t, a[k + 1:self.n, k], 1,
                    a[k + 1:self.n, j], 1)

        ipvt[self.n - 1] = self.n - 1
        if abs(a[self.n - 1, self.n - 1]) < 1e-15:
            info = self.n

        self._lu = a
        self._ipvt = ipvt
        self._info = info
        self._factorized = True
        return info

    def dgesl(self, b, job=0):
        if not self._factorized:
            self.dgefa()
        if self._info != 0:
            raise ValueError(f"矩阵奇异，info={self._info}")

        x = np.array(b, dtype=np.float64, copy=True)
        n = self.n
        a = self._lu
        ipvt = self._ipvt

        if job == 0:

            for k in range(n - 1):
                l = ipvt[k]
                t = x[l]
                if l != k:
                    x[l] = x[k]
                    x[k] = t
                x[k + 1:n] = x[k + 1:n] + t * a[k + 1:n, k]

            for k in range(n - 1, -1, -1):
                x[k] = x[k] / a[k, k]
                t = -x[k]
                x[:k] = x[:k] + t * a[:k, k]
        else:

            for k in range(n):
                x[k] = (x[k] - np.dot(a[:k, k], x[:k])) / a[k, k]

            for k in range(n - 2, -1, -1):
                x[k] = x[k] + np.dot(a[k + 1:n, k], x[k + 1:n])
                l = ipvt[k]
                if l != k:
                    x[l], x[k] = x[k], x[l]

        return x

    def solve(self, b):
        return self.dgesl(b, job=0)

    def determinant(self):
        if not self._factorized:
            self.dgefa()
        det = np.prod(np.diag(self._lu))

        swaps = sum(1 for k in range(self.n) if self._ipvt[k] != k)
        if swaps % 2 == 1:
            det = -det
        return det

    def condition_estimate(self):
        if not self._factorized:
            self.dgefa()
        diag_u = np.abs(np.diag(self._lu))
        if np.any(diag_u < 1e-15):
            return np.inf

        return np.max(diag_u) / np.min(diag_u)


class BandedMatrixSolver:

    def __init__(self, A, lower_bandwidth, upper_bandwidth):
        self.A = np.array(A, dtype=np.float64, copy=True)
        self.n = A.shape[0]
        self.kl = lower_bandwidth
        self.ku = upper_bandwidth

    def solve(self, b):
        try:
            import scipy.linalg as la

            ab = np.zeros((2 * self.kl + self.ku + 1, self.n), dtype=np.float64)
            for j in range(self.n):
                for i in range(max(0, j - self.ku), min(self.n, j + self.kl + 1)):
                    ab[self.kl + self.ku + i - j, j] = self.A[i, j]
            x = la.solve_banded((self.kl, self.ku), ab, b)
            return x
        except ImportError:
            return np.linalg.solve(self.A, b)


if __name__ == "__main__":

    A = np.array([[2.0, 1.0, -1.0],
                  [-3.0, -1.0, 2.0],
                  [-2.0, 1.0, 2.0]], dtype=np.float64)
    solver = DenseLUSolver(A)
    info = solver.dgefa()
    print(f"LU分解 info={info}")
    b = np.array([8.0, -11.0, -3.0])
    x = solver.solve(b)
    print(f"解 x = {x}")
    print(f"残差 ||Ax-b|| = {np.linalg.norm(A @ x - b):.2e}")

    nz, ist, jst, Ast = ge_to_st(A)
    print(f"非零元个数: {nz}")

    q = sparse_matvec(ist, jst, Ast, np.ones(3))
    print(f"稀疏矩阵-向量乘法: {q}")
