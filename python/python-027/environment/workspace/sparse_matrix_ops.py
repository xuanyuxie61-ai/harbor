# -*- coding: utf-8 -*-

import numpy as np


class R8RIMatrix:

    def __init__(self, n, nz=None):
        if n < 1:
            raise ValueError("矩阵阶数必须 >= 1")
        self.n = n
        if nz is None:
            nz = 3 * n - 1
        self.nz = nz
        self.ija = np.zeros(nz, dtype=int)
        self.a = np.zeros(nz)
        self._initialized = False

    def build_from_dense(self, dense_mat, threshold=1.0e-15):
        if dense_mat.shape != (self.n, self.n):
            raise ValueError("稠密矩阵维度不匹配")


        nnz_offdiag = 0
        for i in range(self.n):
            for j in range(self.n):
                if i != j and abs(dense_mat[i, j]) > threshold:
                    nnz_offdiag += 1

        self.nz = self.n + 2 + nnz_offdiag
        self.ija = np.zeros(self.nz, dtype=int)
        self.a = np.zeros(self.nz)


        for i in range(self.n):
            self.a[i] = dense_mat[i, i]


        k = self.n + 1
        for i in range(self.n):
            self.ija[i] = k + 1
            for j in range(self.n):
                if i != j and abs(dense_mat[i, j]) > threshold:
                    self.a[k] = dense_mat[i, j]
                    self.ija[k] = j
                    k += 1

        self.ija[self.n] = k + 1
        self._initialized = True

    @staticmethod
    def build_dif2(n):
        if n < 2:
            raise ValueError("n 必须 >= 2")

        nz = 3 * n - 1
        mat = R8RIMatrix(n, nz)


        mat.a[0:n] = 2.0


        k = n + 1
        for i in range(n):
            mat.ija[i] = k + 1
            if i == 0 or i == n - 1:
                k += 1
            else:
                k += 2

        mat.ija[n] = k + 1
        mat.a[n] = 0.0


        k = n
        for i in range(n):
            if i == 0:
                k += 1
                mat.ija[k] = 1
                mat.a[k] = -1.0
            elif i < n - 1:
                k += 1
                mat.ija[k] = i - 1
                mat.a[k] = -1.0
                k += 1
                mat.ija[k] = i + 1
                mat.a[k] = -1.0
            else:
                k += 1
                mat.ija[k] = n - 2
                mat.a[k] = -1.0

        mat._initialized = True
        return mat

    def to_dense(self):
        if not self._initialized:
            raise RuntimeError("矩阵未初始化")

        dense = np.zeros((self.n, self.n))


        for k in range(self.n):
            dense[k, k] = self.a[k]


        for i in range(self.n):
            start = self.ija[i] - 1
            end = self.ija[i+1] - 1
            for idx in range(start, end):
                if idx < self.nz:
                    j = self.ija[idx]
                    dense[i, j] = self.a[idx]

        return dense

    def matvec(self, x):
        if not self._initialized:
            raise RuntimeError("矩阵未初始化")
        x = np.asarray(x).ravel()
        if len(x) != self.n:
            raise ValueError(f"向量维度 {len(x)} 与矩阵阶数 {self.n} 不匹配")

        b = np.zeros(self.n)


        b[:self.n] = self.a[:self.n] * x[:self.n]


        for i in range(self.n):
            start = self.ija[i] - 1
            end = self.ija[i+1] - 1
            for idx in range(start, end):
                if idx < self.nz and idx >= 0:
                    j = self.ija[idx]
                    if 0 <= j < self.n:
                        b[i] += self.a[idx] * x[j]

        return b

    def matvec_transpose(self, x):
        if not self._initialized:
            raise RuntimeError("矩阵未初始化")
        x = np.asarray(x).ravel()
        if len(x) != self.n:
            raise ValueError("向量维度不匹配")

        b = np.zeros(self.n)


        b[:self.n] = self.a[:self.n] * x[:self.n]


        for i in range(self.n):
            start = self.ija[i] - 1
            end = self.ija[i+1] - 1
            for idx in range(start, end):
                if idx < self.nz and idx >= 0:
                    j = self.ija[idx]
                    if 0 <= j < self.n:
                        b[j] += self.a[idx] * x[i]

        return b

    def get_memory_usage(self):
        dense_bytes = self.n * self.n * 8
        sparse_bytes = self.nz * (8 + 4)
        if dense_bytes <= 0:
            return 0.0
        saving = 1.0 - sparse_bytes / dense_bytes
        return max(0.0, saving)


def demo_sparse_ops():
    n = 100
    mat = R8RIMatrix.build_dif2(n)


    x = np.ones(n)
    y_sparse = mat.matvec(x)


    dense = mat.to_dense()
    y_dense = dense.dot(x)

    error = np.linalg.norm(y_sparse - y_dense)
    saving = mat.get_memory_usage()

    print("稀疏矩阵运算演示:")
    print(f"  矩阵阶数 N       = {n}")
    print(f"  稠密存储元素数   = {n*n}")
    print(f"  稀疏存储元素数   = {mat.nz}")
    print(f"  内存节省比例     = {saving*100:.1f}%")
    print(f"  matvec误差       = {error:.3e}")

    return mat


if __name__ == "__main__":
    demo_sparse_ops()
