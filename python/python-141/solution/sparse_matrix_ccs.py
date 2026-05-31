"""
稀疏矩阵压缩列存储(CCS)模块
========================================
基于种子项目 975_r8ccs 的核心算法改造。

在金融工程应用中，Heston随机波动率PDE经有限差分离散化后，
生成的大型稀疏线性系统采用CCS格式存储，以显著降低内存开销。

数学背景:
---------
对于 M×N 矩阵 A，CCS格式使用三个数组：
  - colptr: 长度为 N+1，colptr[j] 表示第 j 列第一个非零元在 a 中的位置
  - rowind: 长度为 nz_num，存储非零元的行索引（每列内升序）
  - a:      长度为 nz_num，存储非零元的数值

矩阵-向量乘法:
    (A·x)_i = Σ_j A_{ij} x_j = Σ_{k=colptr[j]}^{colptr[j+1]-1} a[k] · x[rowind[k]]

转置乘法:
    (A^T·x)_j = Σ_i A_{ij} x_i = Σ_{k=colptr[j]}^{colptr[j+1]-1} a[k] · x[rowind[k]]
"""

import numpy as np


class SparseMatrixCCS:
    """
    双精度稀疏矩阵的压缩列存储(CCS)实现。
    等效于 MATLAB sparse 格式与 Harwell-Boeing RUA 格式。
    """

    def __init__(self, m, n, nz_num, colptr, rowind, a):
        """
        参数:
        ------
        m, n     : int, 矩阵行列数
        nz_num   : int, 非零元个数
        colptr   : array(N+1), 列指针
        rowind   : array(nz_num), 行索引（每列内升序）
        a        : array(nz_num), 非零元数值
        """
        self.m = int(m)
        self.n = int(n)
        self.nz_num = int(nz_num)
        self.colptr = np.asarray(colptr, dtype=np.int64)
        self.rowind = np.asarray(rowind, dtype=np.int64)
        self.a = np.asarray(a, dtype=np.float64)
        self._validate()

    def _validate(self):
        if self.m <= 0 or self.n <= 0:
            raise ValueError("矩阵维度必须为正整数")
        if self.nz_num < 0:
            raise ValueError("非零元个数不能为负")
        if len(self.colptr) != self.n + 1:
            raise ValueError(f"colptr长度应为{n+1}, 实际为{len(self.colptr)}")
        if len(self.rowind) != self.nz_num:
            raise ValueError(f"rowind长度应为{nz_num}, 实际为{len(self.rowind)}")
        if len(self.a) != self.nz_num:
            raise ValueError(f"a长度应为{nz_num}, 实际为{len(self.a)}")
        if self.colptr[0] != 0:
            raise ValueError("colptr[0]必须为0（采用0基索引）")
        if self.colptr[self.n] != self.nz_num:
            raise ValueError("colptr[-1]必须等于nz_num")
        for j in range(self.n):
            if self.colptr[j] > self.colptr[j + 1]:
                raise ValueError(f"colptr在第{j}列出现递减")
            for k in range(self.colptr[j], self.colptr[j + 1] - 1):
                if self.rowind[k] >= self.rowind[k + 1]:
                    raise ValueError(f"第{j}列rowind未按升序排列")
                if self.rowind[k] < 0 or self.rowind[k] >= self.m:
                    raise ValueError(f"行索引越界: {self.rowind[k]}")

    def mv(self, x):
        """
        矩阵-向量乘法: b = A · x

        算法复杂度: O(nz_num)
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.n,):
            raise ValueError(f"x维度应为({self.n},), 实际为{x.shape}")
        b = np.zeros(self.m, dtype=np.float64)
        for j in range(self.n):
            xj = x[j]
            for k in range(self.colptr[j], self.colptr[j + 1]):
                i = self.rowind[k]
                b[i] += self.a[k] * xj
        return b

    def mtv(self, x):
        """
        转置矩阵-向量乘法: b = A^T · x

        算法复杂度: O(nz_num)
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.m,):
            raise ValueError(f"x维度应为({self.m},), 实际为{x.shape}")
        b = np.zeros(self.n, dtype=np.float64)
        for j in range(self.n):
            s = 0.0
            for k in range(self.colptr[j], self.colptr[j + 1]):
                i = self.rowind[k]
                s += self.a[k] * x[i]
            b[j] = s
        return b

    def get(self, i, j):
        """
        获取元素 A(i,j)。若不存在则返回0。
        使用二分搜索，复杂度 O(log(col非零元个数))。
        """
        if i < 0 or i >= self.m or j < 0 or j >= self.n:
            raise IndexError("索引越界")
        left = self.colptr[j]
        right = self.colptr[j + 1] - 1
        while left <= right:
            mid = (left + right) // 2
            if self.rowind[mid] == i:
                return self.a[mid]
            elif self.rowind[mid] < i:
                left = mid + 1
            else:
                right = mid - 1
        return 0.0

    def set(self, i, j, aij):
        """
        设置已有非零元 A(i,j) = aij。
        若该位置不在预分配的非零元结构中，则抛出错误。
        """
        if i < 0 or i >= self.m or j < 0 or j >= self.n:
            raise IndexError("索引越界")
        left = self.colptr[j]
        right = self.colptr[j + 1] - 1
        while left <= right:
            mid = (left + right) // 2
            if self.rowind[mid] == i:
                self.a[mid] = aij
                return
            elif self.rowind[mid] < i:
                left = mid + 1
            else:
                right = mid - 1
        raise ValueError(f"位置({i},{j})不在预分配的非零元结构中")

    def to_dense(self):
        """转为稠密矩阵（仅用于小规模调试）。"""
        A = np.zeros((self.m, self.n), dtype=np.float64)
        for j in range(self.n):
            for k in range(self.colptr[j], self.colptr[j + 1]):
                i = self.rowind[k]
                A[i, j] = self.a[k]
        return A

    @staticmethod
    def dif2(m, n):
        """
        构造二阶差分算子的CCS稀疏矩阵。
        对应一维热方程/Black-Scholes空间离散的拉普拉斯部分：
            L = tridiag(-1, 2, -1)
        该矩阵在PDE离散中用于空间二阶导数 ∂²/∂x² 的近似。
        """
        if m != n:
            raise ValueError("二阶差分矩阵必须为方阵")
        if n < 2:
            raise ValueError("维度至少为2")
        nz_num = 3 * n - 2
        colptr = np.zeros(n + 1, dtype=np.int64)
        colptr[0] = 0
        colptr[1] = 2
        for j in range(2, n):
            colptr[j] = colptr[j - 1] + 3
        colptr[n] = colptr[n - 1] + 2

        rowind = np.zeros(nz_num, dtype=np.int64)
        a = np.zeros(nz_num, dtype=np.float64)
        k = 0
        # 第0列
        rowind[k] = 0; a[k] = 2.0; k += 1
        rowind[k] = 1; a[k] = -1.0; k += 1
        # 中间列
        for j in range(1, n - 1):
            rowind[k] = j - 1; a[k] = -1.0; k += 1
            rowind[k] = j;     a[k] =  2.0; k += 1
            rowind[k] = j + 1; a[k] = -1.0; k += 1
        # 最后一列
        rowind[k] = n - 2; a[k] = -1.0; k += 1
        rowind[k] = n - 1; a[k] =  2.0; k += 1

        return SparseMatrixCCS(m, n, nz_num, colptr, rowind, a)

    @staticmethod
    def from_dense(A, tol=1e-15):
        """从稠密矩阵构造CCS稀疏矩阵。"""
        A = np.asarray(A, dtype=np.float64)
        m, n = A.shape
        colptr = np.zeros(n + 1, dtype=np.int64)
        rowind_list = []
        a_list = []
        nz = 0
        for j in range(n):
            colptr[j] = nz
            for i in range(m):
                if abs(A[i, j]) > tol:
                    rowind_list.append(i)
                    a_list.append(A[i, j])
                    nz += 1
        colptr[n] = nz
        return SparseMatrixCCS(m, n, nz, colptr,
                               np.array(rowind_list, dtype=np.int64),
                               np.array(a_list, dtype=np.float64))
