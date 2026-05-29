# -*- coding: utf-8 -*-
"""
sparse_matrix_ops.py
稀疏矩阵运算模块
基于种子项目 992_r8ri (行索引稀疏矩阵格式) 重构

本模块实现 R8RI (Row-Indexed Sparse Storage) 格式及其运算，
用于高效存储和运算鞘层模拟中产生的大型稀疏矩阵（如Poisson方程离散矩阵）。
"""

import numpy as np


class R8RIMatrix:
    """
    R8RI 行索引稀疏矩阵类
    
    存储格式（基于 Numerical Recipes）:
        - 前 N 个 A 元素存储对角线
        - 前 N 个 IJA 元素存储每行第一个非对角元素在 A 中的索引
        - IJA[N+1] 存储最后一个非对角元素的下一个位置
        - A[N+2:] 存储非对角元素的值
        - IJA[N+2:] 存储非对角元素的列号
    
    这种格式对于大型稀疏矩阵运算比稠密矩阵节省大量内存。
    """

    def __init__(self, n, nz=None):
        """
        初始化空矩阵
        
        Parameters:
            n:  矩阵阶数
            nz: 总存储空间（包含对角线），若不指定则分配 3*N-1（三对角）
        """
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
        """
        从稠密矩阵构造 R8RI 格式
        
        Parameters:
            dense_mat: 稠密矩阵
            threshold: 零元素阈值
        """
        if dense_mat.shape != (self.n, self.n):
            raise ValueError("稠密矩阵维度不匹配")

        # 统计非零非对角元素数量
        nnz_offdiag = 0
        for i in range(self.n):
            for j in range(self.n):
                if i != j and abs(dense_mat[i, j]) > threshold:
                    nnz_offdiag += 1

        self.nz = self.n + 2 + nnz_offdiag
        self.ija = np.zeros(self.nz, dtype=int)
        self.a = np.zeros(self.nz)

        # 存储对角线
        for i in range(self.n):
            self.a[i] = dense_mat[i, i]

        # 存储非对角元素
        k = self.n + 1  # 从 N+2 开始（MATLAB索引从1，Python从0，所以是 N+1）
        for i in range(self.n):
            self.ija[i] = k + 1  # 对应MATLAB中 IJA(i+1) = k+1
            for j in range(self.n):
                if i != j and abs(dense_mat[i, j]) > threshold:
                    self.a[k] = dense_mat[i, j]
                    self.ija[k] = j
                    k += 1

        self.ija[self.n] = k + 1
        self._initialized = True

    @staticmethod
    def build_dif2(n):
        """
        构造二阶差分矩阵的 R8RI 格式（基于 r8ri_dif2.m）
        
        矩阵形式:
            [ 2 -1  0 ...  0 ]
            [-1  2 -1 ...  0 ]
            [ ...           ]
            [ 0 ... -1  2 -1]
            [ 0 ...  0 -1  2]
        
        这是离散 Poisson 方程在一维下的标准矩阵。
        """
        if n < 2:
            raise ValueError("n 必须 >= 2")

        nz = 3 * n - 1
        mat = R8RIMatrix(n, nz)

        # 对角线
        mat.a[0:n] = 2.0

        # IJA 前 N 个元素
        k = n + 1  # 对应 MATLAB 的 n+2
        for i in range(n):
            mat.ija[i] = k + 1
            if i == 0 or i == n - 1:
                k += 1
            else:
                k += 2

        mat.ija[n] = k + 1
        mat.a[n] = 0.0

        # 非对角元素
        k = n
        for i in range(n):
            if i == 0:
                k += 1
                mat.ija[k] = 1  # 第2列 (j=1)
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
        """
        转换为稠密矩阵（基于 r8ri_to_r8ge.m）
        """
        if not self._initialized:
            raise RuntimeError("矩阵未初始化")

        dense = np.zeros((self.n, self.n))

        # 对角线
        for k in range(self.n):
            dense[k, k] = self.a[k]

        # 非对角元素
        for i in range(self.n):
            start = self.ija[i] - 1  # MATLAB->Python 索引转换
            end = self.ija[i+1] - 1
            for idx in range(start, end):
                if idx < self.nz:
                    j = self.ija[idx]
                    dense[i, j] = self.a[idx]

        return dense

    def matvec(self, x):
        """
        稀疏矩阵-向量乘法 y = A * x（基于 r8ri_mv.m）
        
        复杂度: O(N + nnz)，远优于稠密矩阵的 O(N^2)
        """
        if not self._initialized:
            raise RuntimeError("矩阵未初始化")
        x = np.asarray(x).ravel()
        if len(x) != self.n:
            raise ValueError(f"向量维度 {len(x)} 与矩阵阶数 {self.n} 不匹配")

        b = np.zeros(self.n)

        # 对角线贡献
        b[:self.n] = self.a[:self.n] * x[:self.n]

        # 非对角线贡献
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
        """
        转置矩阵-向量乘法 y = A^T * x（基于 r8ri_mtv.m）
        """
        if not self._initialized:
            raise RuntimeError("矩阵未初始化")
        x = np.asarray(x).ravel()
        if len(x) != self.n:
            raise ValueError("向量维度不匹配")

        b = np.zeros(self.n)

        # 对角线贡献
        b[:self.n] = self.a[:self.n] * x[:self.n]

        # 非对角线转置贡献
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
        """
        返回 R8RI 格式相比稠密格式的内存节省比例
        """
        dense_bytes = self.n * self.n * 8  # double = 8 bytes
        sparse_bytes = self.nz * (8 + 4)   # a: 8 bytes, ija: 4 bytes (int32)
        if dense_bytes <= 0:
            return 0.0
        saving = 1.0 - sparse_bytes / dense_bytes
        return max(0.0, saving)


def demo_sparse_ops():
    """演示稀疏矩阵运算"""
    n = 100
    mat = R8RIMatrix.build_dif2(n)

    # 测试矩阵-向量乘法
    x = np.ones(n)
    y_sparse = mat.matvec(x)

    # 与稠密结果对比
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
