"""
sparse_matrix_ops.py
广义对角稀疏矩阵运算与SOR迭代求解

核心数学模型：
1. 广义对角存储（R8GD格式）：
   对于 n×n 矩阵 A，仅存储 ndiag 个对角线
   offset[k] 表示第 k 个存储对角线相对于主对角线的偏移
   
2. 矩阵向量乘法：
   b_i = Σ_{k=1}^{ndiag} A_{i,k} * x_{i+offset[k]}
   其中仅当 1 <= i+offset[k] <= n 时参与求和
   
3. SOR 迭代法（逐次超松弛）：
   对于方程组 A x = b，分裂 A = D + L + U
   (D + ωL) x^{(k+1)} = ωb - [ωU + (ω-1)D] x^{(k)}
   
   分量形式：
   x_i^{new} = (1-ω) x_i + (ω/a_ii) * [b_i - Σ_{j<i} a_{ij} x_j^{new} - Σ_{j>i} a_{ij} x_j]
   
   松弛参数 ω ∈ (0, 2)，最优值通常接近 1.5-1.9
"""

import numpy as np


class R8GDMatrix:
    """
    广义对角稀疏矩阵（Generalized Diagonal）
    融合自 r8gd_mv 项目
    """

    def __init__(self, n, ndiag, offsets, values):
        """
        Parameters
        ----------
        n : int
            矩阵阶数
        ndiag : int
            存储的对角线数量
        offsets : ndarray, shape (ndiag,)
            各对角线偏移量（0为主对角线，正为上对角线，负为下对角线）
        values : ndarray, shape (n, ndiag)
            各对角线存储的值
        """
        self.n = max(int(n), 1)
        self.ndiag = max(int(ndiag), 1)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.values = np.asarray(values, dtype=np.float64)

        if self.values.shape != (self.n, self.ndiag):
            raise ValueError(f"values shape {self.values.shape} inconsistent with ({self.n}, {self.ndiag})")
        if self.offsets.shape[0] != self.ndiag:
            raise ValueError("offsets length mismatch")

    def mv(self, x):
        """
        矩阵向量乘法 y = A * x
        
        y_i = Σ_{d=1}^{ndiag} A_{i,d} * x_{i+offset[d]}
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] != self.n:
            raise ValueError("vector dimension mismatch")

        y = np.zeros(self.n, dtype=np.float64)
        for i in range(self.n):
            for d in range(self.ndiag):
                j = i + self.offsets[d]
                if 0 <= j < self.n:
                    y[i] += self.values[i, d] * x[j]
        return y

    def mtv(self, x):
        """
        转置矩阵向量乘法 y = A^T * x
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] != self.n:
            raise ValueError("vector dimension mismatch")

        y = np.zeros(self.n, dtype=np.float64)
        for i in range(self.n):
            for d in range(self.ndiag):
                j = i + self.offsets[d]
                if 0 <= j < self.n:
                    y[j] += self.values[i, d] * x[i]
        return y

    def to_dense(self):
        """转换为稠密矩阵（仅用于小规模调试）"""
        A = np.zeros((self.n, self.n), dtype=np.float64)
        for i in range(self.n):
            for d in range(self.ndiag):
                j = i + self.offsets[d]
                if 0 <= j < self.n:
                    A[i, j] = self.values[i, d]
        return A


class SORSolver:
    """
    SOR 迭代求解器
    融合自 sor1 项目
    """

    def __init__(self, omega=1.5, max_iter=1000, tol=1e-10):
        """
        Parameters
        ----------
        omega : float
            松弛参数，0 < ω < 2
        max_iter : int
            最大迭代次数
        tol : float
            收敛容差
        """
        self.omega = max(min(float(omega), 1.999), 0.001)
        self.max_iter = max(int(max_iter), 1)
        self.tol = max(float(tol), 1e-15)

    def solve(self, A, b, x0=None):
        """
        使用 SOR 求解 A x = b
        
        Parameters
        ----------
        A : ndarray, shape (n, n)
            系数矩阵（要求对角元非零）
        b : ndarray, shape (n,)
        x0 : ndarray or None
            初始猜测
        
        Returns
        -------
        x : ndarray
        residual_norm : float
        iterations : int
        converged : bool
        """
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        n = A.shape[0]

        if x0 is None:
            x = np.zeros(n, dtype=np.float64)
        else:
            x = np.asarray(x0, dtype=np.float64).copy()

        # 检查对角元
        diag = np.diag(A)
        if np.any(np.abs(diag) < 1e-14):
            # 添加微小扰动保证可解性
            min_diag = np.min(np.abs(diag[diag != 0])) if np.any(diag != 0) else 1.0
            for i in range(n):
                if abs(diag[i]) < 1e-14:
                    A[i, i] = min_diag * 1e-6

        for it in range(self.max_iter):
            x_old = x.copy()
            for i in range(n):
                sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i + 1:], x[i + 1:])
                if abs(A[i, i]) > 1e-15:
                    x[i] = (1.0 - self.omega) * x[i] + self.omega * (b[i] - sigma) / A[i, i]
                else:
                    x[i] = x[i]  # 保持不变

            # 收敛判断
            diff = np.linalg.norm(x - x_old)
            if diff < self.tol:
                residual = A @ x - b
                return x, np.linalg.norm(residual), it + 1, True

        residual = A @ x - b
        return x, np.linalg.norm(residual), self.max_iter, False

    def solve_sparse(self, r8gd_A, b, x0=None):
        """
        对 R8GD 稀疏矩阵使用 SOR 求解
        
        利用稀疏结构避免零元运算
        """
        b = np.asarray(b, dtype=np.float64)
        n = r8gd_A.n

        if x0 is None:
            x = np.zeros(n, dtype=np.float64)
        else:
            x = np.asarray(x0, dtype=np.float64).copy()

        # 构建对角线索引
        diag_idx = -1
        for d in range(r8gd_A.ndiag):
            if r8gd_A.offsets[d] == 0:
                diag_idx = d
                break

        if diag_idx < 0:
            raise ValueError("R8GD matrix must contain main diagonal")

        for it in range(self.max_iter):
            x_old = x.copy()
            for i in range(n):
                sigma = 0.0
                for d in range(r8gd_A.ndiag):
                    j = i + r8gd_A.offsets[d]
                    if j != i and 0 <= j < n:
                        sigma += r8gd_A.values[i, d] * x[j]

                a_ii = r8gd_A.values[i, diag_idx]
                if abs(a_ii) > 1e-15:
                    x[i] = (1.0 - self.omega) * x[i] + self.omega * (b[i] - sigma) / a_ii

            diff = np.linalg.norm(x - x_old)
            if diff < self.tol:
                residual = r8gd_A.mv(x) - b
                return x, np.linalg.norm(residual), it + 1, True

        residual = r8gd_A.mv(x) - b
        return x, np.linalg.norm(residual), self.max_iter, False
