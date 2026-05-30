
import numpy as np


class SparseMatrixCCS:

    def __init__(self, m, n, nz_num, colptr, rowind, a):
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
        A = np.zeros((self.m, self.n), dtype=np.float64)
        for j in range(self.n):
            for k in range(self.colptr[j], self.colptr[j + 1]):
                i = self.rowind[k]
                A[i, j] = self.a[k]
        return A

    @staticmethod
    def dif2(m, n):
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

        rowind[k] = 0; a[k] = 2.0; k += 1
        rowind[k] = 1; a[k] = -1.0; k += 1

        for j in range(1, n - 1):
            rowind[k] = j - 1; a[k] = -1.0; k += 1
            rowind[k] = j;     a[k] =  2.0; k += 1
            rowind[k] = j + 1; a[k] = -1.0; k += 1

        rowind[k] = n - 2; a[k] = -1.0; k += 1
        rowind[k] = n - 1; a[k] =  2.0; k += 1

        return SparseMatrixCCS(m, n, nz_num, colptr, rowind, a)

    @staticmethod
    def from_dense(A, tol=1e-15):
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
