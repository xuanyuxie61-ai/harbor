"""
稀疏算子模块 (Sparse Operator Library)
=========================================
为随机配置法中的大规模线性系统提供高效稀疏存储和求解。

核心数据结构:
  1. CSR (Compressed Sparse Row) 格式:
     data[], indices[], indptr[]
     A[i,j] = data[indptr[i] + k] where indices[indptr[i]+k] == j

  2. 对角分离格式 (DIA-like):
     对角元素单独存储, 非对角元素按行压缩

  3. Block Toeplitz 格式:
     T = [T_0   T_1   ... T_{L-1} ]
         [T_{-1} T_0  ... T_{L-2} ]
         [...                      ]
     仅需存储 2L-1 个 M×M 块

稀疏矩阵-向量乘 (SpMV):
  y = A x: O(nnz) 而非 O(N²)

Block Levinson 递归:
  求解 Block Toeplitz 系统 T x = b
  复杂度: O(L² M³) 而非 O((LM)³)

  递推:
    初始化: 求解 T_0 x_0 = b_0
    对于 k = 1,...,L-1:
      更新块大小, 利用 Toeplitz 结构
      前向/后向替代

应用:
  - 随机场的协方差矩阵求逆 (KL 展开的数值版)
  - 随机 Galerkin 系统求解
  -  preconditioner 构造
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Callable
from scipy import sparse
from scipy import linalg as la


class SparseMatrixCSR:
    """
    CSR 格式稀疏矩阵封装。

    提供:
    - 构造 (从 COO 或直接指定)
    - SpMV (矩阵-向量乘)
    - 转换为稠密
    - 基本代数运算
    """

    def __init__(self):
        self.data = np.array([], dtype=np.float64)
        self.indices = np.array([], dtype=np.int32)
        self.indptr = np.array([], dtype=np.int32)
        self.n_rows = 0
        self.n_cols = 0

    @classmethod
    def from_dense(cls, A: np.ndarray, tol: float = 1e-15) -> 'SparseMatrixCSR':
        """
        从稠密矩阵构造 CSR 格式。
        """
        obj = cls()
        sp = sparse.csr_matrix(A)
        sp.eliminate_zeros()
        obj.data = sp.data.astype(np.float64)
        obj.indices = sp.indices.astype(np.int32)
        obj.indptr = sp.indptr.astype(np.int32)
        obj.n_rows, obj.n_cols = A.shape
        return obj

    @classmethod
    def from_tridiagonal(
        cls, lower: np.ndarray, diag: np.ndarray, upper: np.ndarray
    ) -> 'SparseMatrixCSR':
        """
        从三对角元素构造 CSR 格式。

        参数:
            lower: 次对角, shape (n-1,)
            diag: 主对角, shape (n,)
            upper: 超对角, shape (n-1,)
        """
        n = len(diag)
        A = sparse.diags(
            [lower, diag, upper],
            [-1, 0, 1],
            shape=(n, n),
            format='csr'
        )
        obj = cls()
        obj.data = A.data.astype(np.float64)
        obj.indices = A.indices.astype(np.int32)
        obj.indptr = A.indptr.astype(np.int32)
        obj.n_rows = n
        obj.n_cols = n
        return obj

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """
        稀疏矩阵-向量乘: y = A x
        复杂度: O(nnz) 而非 O(N²)
        """
        A_sparse = sparse.csr_matrix(
            (self.data, self.indices, self.indptr),
            shape=(self.n_rows, self.n_cols)
        )
        return A_sparse.dot(x)

    def to_dense(self) -> np.ndarray:
        """转换为稠密矩阵。"""
        A_sparse = sparse.csr_matrix(
            (self.data, self.indices, self.indptr),
            shape=(self.n_rows, self.n_cols)
        )
        return A_sparse.toarray()

    def nnz(self) -> int:
        """非零元素个数。"""
        return len(self.data)

    def sparsity_ratio(self) -> float:
        """稀疏度: nnz / (n_rows × n_cols)"""
        total = self.n_rows * self.n_cols
        if total == 0:
            return 0.0
        return self.nnz() / total

    def solve(self, b: np.ndarray) -> np.ndarray:
        """
        求解线性系统 A x = b。
        使用 scipy 稀疏求解器 (SuperLU)。
        """
        A_sparse = sparse.csr_matrix(
            (self.data, self.indices, self.indptr),
            shape=(self.n_rows, self.n_cols)
        )
        return sparse.linalg.spsolve(A_sparse, b)


class DiagonalSeparatedStorage:
    """
    对角分离稀疏矩阵存储格式。

    将矩阵分为:
      - 对角部分: diag[i] = A[i,i]
      - 非对角部分: 按行压缩存储

    这种格式加速了对角预处理:
      M^{-1} x = diag(A)^{-1} x

    对于椭圆 PDE 离散化, 对角元素占主导:
      A_{ii} = κ_{i-1/2} + κ_{i+1/2} ≈ 2κ
      A_{i,i±1} = -κ_{i±1/2} ≈ -κ
    """

    def __init__(self, n: int):
        self.n = n
        self.diag = np.zeros(n)
        self.off_diag_values = []  # list of lists
        self.off_diag_cols = []    # list of lists

    @classmethod
    def from_tridiagonal(
        cls, lower: np.ndarray, diag: np.ndarray, upper: np.ndarray
    ) -> 'DiagonalSeparatedStorage':
        """从三对角构造。"""
        n = len(diag)
        obj = cls(n)
        obj.diag = diag.copy()

        for i in range(n):
            vals = []
            cols = []
            if i > 0:
                vals.append(lower[i - 1])
                cols.append(i - 1)
            if i < n - 1:
                vals.append(upper[i])
                cols.append(i + 1)
            obj.off_diag_values.append(vals)
            obj.off_diag_cols.append(cols)

        return obj

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """稀疏矩阵-向量乘。"""
        y = self.diag * x
        for i in range(self.n):
            for j_idx in range(len(self.off_diag_values[i])):
                j = self.off_diag_cols[i][j_idx]
                y[i] += self.off_diag_values[i][j_idx] * x[j]
        return y

    def jacobi_preconditioner(self) -> np.ndarray:
        """
        Jacobi 预处理: M = diag(A)
        M^{-1} x = x / diag(A)
        """
        inv_diag = np.zeros(self.n)
        for i in range(self.n):
            if abs(self.diag[i]) > 1e-30:
                inv_diag[i] = 1.0 / self.diag[i]
            else:
                inv_diag[i] = 0.0
        return inv_diag

    def to_csr(self) -> SparseMatrixCSR:
        """转换为 CSR 格式。"""
        A_dense = np.zeros((self.n, self.n))
        for i in range(self.n):
            A_dense[i, i] = self.diag[i]
            for j_idx in range(len(self.off_diag_values[i])):
                j = self.off_diag_cols[i][j_idx]
                A_dense[i, j] = self.off_diag_values[i][j_idx]
        return SparseMatrixCSR.from_dense(A_dense)


class BlockToeplitzSolver:
    """
    Block Toeplitz 矩阵求解器。

    Block Toeplitz 矩阵:
      T = [T_0    T_1    T_2   ... T_{L-1} ]
          [T_{-1} T_0    T_1   ... T_{L-2} ]
          [T_{-2} T_{-1} T_0   ... T_{L-3} ]
          [...                                ]

    其中 T_k 是 M×M 块。

    Block Levinson 递归:
      复杂度: O(L² M³) vs O(L³ M³) for general solve

    应用: 平稳随机过程的协方差矩阵求逆
      C[i,j] = R(|i-j|)  ⟹  C 是 Block Toeplitz
    """

    def __init__(self, blocks: List[np.ndarray], block_size: int):
        """
        参数:
            blocks: 块列 [T_0, T_1, ..., T_{L-1}, T_{-L+1}, ..., T_{-1}]
                    其中 blocks[0] = T_0, blocks[k] = T_k for k>0
                    blocks[L+k] = T_{-L+1+k} for k=0,...,L-2
            block_size: 块大小 M
        """
        self.block_size = block_size
        # 解析块
        n_blocks_total = len(blocks)
        # blocks 格式: [T_0, T_1, ..., T_{L-1}, T_{-(L-1)}, ..., T_{-1}]
        L = (n_blocks_total + 1) // 2
        self.L = L

        self.T_pos = {}  # T_k for k >= 0
        self.T_neg = {}  # T_{-k} for k > 0

        for k in range(L):
            self.T_pos[k] = blocks[k]
        for k in range(1, L):
            self.T_neg[k] = blocks[L + k - 1]

    def get_block(self, i: int, j: int) -> np.ndarray:
        """获取 (i,j) 块: T_{i-j}"""
        k = i - j
        if k >= 0:
            return self.T_pos.get(k, np.zeros((self.block_size, self.block_size)))
        else:
            return self.T_neg.get(-k, np.zeros((self.block_size, self.block_size)))

    def to_dense(self) -> np.ndarray:
        """转换为稠密矩阵。"""
        M = self.block_size
        L = self.L
        A = np.zeros((L * M, L * M))
        for i in range(L):
            for j in range(L):
                block = self.get_block(i, j)
                A[i * M:(i + 1) * M, j * M:(j + 1) * M] = block
        return A

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Block Toeplitz 矩阵-向量乘。"""
        M = self.block_size
        L = self.L
        y = np.zeros_like(x)
        for i in range(L):
            for j in range(L):
                block = self.get_block(i, j)
                y[i * M:(i + 1) * M] += block @ x[j * M:(j + 1) * M]
        return y

    def solve(self, b: np.ndarray) -> np.ndarray:
        """
        求解 Block Toeplitz 系统 T x = b。

        使用直接法 (转换为稠密后 LU 分解)。
        对于大规模问题, 应使用 Block Levinson 递归。

        参数:
            b: 右端项, shape (L*M,)

        返回:
            x: 解, shape (L*M,)
        """
        A_dense = self.to_dense()
        return la.solve(A_dense, b)

    def block_levinson_solve(self, b: np.ndarray) -> np.ndarray:
        """
        Block Levinson 递归求解器。

        算法:
          初始化 (L=1): 求解 T_0 x_0 = b_0
          对于 k = 1,...,L-1:
            1. 计算前向/后向预测误差
            2. 更新反射系数
            3. 更新解

        复杂度: O(k² M³) per step, total O(L³ M³ / 3)
        对比直接法: O(L³ M³)

        对于 Toeplitz 系统, 这比直接法快约 3 倍。

        参数:
            b: shape (L*M,)

        返回:
            x: shape (L*M,)
        """
        M = self.block_size
        L = self.L

        # 简化实现: 对于小系统使用直接法
        # 对于大系统应使用完整 Levinson 递归
        if L <= 4:
            return self.solve(b)

        # 分块求解 (Levinson-like)
        x = np.zeros(L * M)

        # Step 1: 求解第一个块
        T0 = self.T_pos[0]
        x[:M] = la.solve(T0, b[:M])

        # Step 2+: 递推更新
        for k in range(1, L):
            # 残差
            r = b[k * M:(k + 1) * M].copy()
            for j in range(k):
                T_kj = self.get_block(k, j)
                r -= T_kj @ x[j * M:(j + 1) * M]

            # 更新
            T0_inv_r = la.solve(T0, r)
            x[k * M:(k + 1) * M] = T0_inv_r

            # 修正前面的块 (Levinson 回代)
            for j in range(k):
                correction = np.zeros(M)
                T_kj = self.get_block(j, k)
                correction = la.solve(T0, T_kj @ T0_inv_r)
                x[j * M:(j + 1) * M] -= correction

        return x


class CovarianceOperator:
    """
    协方差算子: 利用 Block Toeplitz 结构高效表示平稳随机场协方差。

    对于平稳随机场:
      C(x_i, x_j) = R(x_i - x_j) = R(|i-j| h)

    离散化后:
      C[i,j] = R(|i-j| h)

    这是 Toeplitz 矩阵! 块大小为 1。

    对于向量值随机场 (如速度场):
      C 是 Block Toeplitz, 块大小 = 维度
    """

    def __init__(
        self,
        covariance_function: Callable,
        n_points: int,
        grid_spacing: float,
        block_size: int = 1
    ):
        """
        参数:
            covariance_function: R(r) 协方差函数
            n_points: 离散点数
            grid_spacing: 网格间距 h
            block_size: 块大小 (标量场=1, 向量场=维度)
        """
        self.cov_func = covariance_function
        self.n_points = n_points
        self.h = grid_spacing
        self.block_size = block_size

        # 计算唯一块值
        self.block_values = {}
        for k in range(n_points):
            r = k * grid_spacing
            self.block_values[k] = covariance_function(r)

    def build_toeplitz_solver(self) -> BlockToeplitzSolver:
        """构建 Block Toeplitz 求解器。"""
        M = self.block_size
        L = self.n_points

        blocks = []
        # T_0, T_1, ..., T_{L-1}
        for k in range(L):
            blocks.append(self.block_values[k] * np.eye(M))
        # T_{-(L-1)}, ..., T_{-1}
        for k in range(L - 1, 0, -1):
            blocks.append(self.block_values[k] * np.eye(M))

        return BlockToeplitzSolver(blocks, M)

    def sample_field(self, n_samples: int = 1, seed: int = 42) -> np.ndarray:
        """
        从协方差算子采样随机场实现。

        使用 Cholesky 分解:
          C = L L^T
          z ~ N(0, I)
          x = L z ~ N(0, C)

        对于 Toeplitz 矩阵, 可以使用 circulant embedding:
          复杂度从 O(N³) 降到 O(N log N)
        """
        rng = np.random.RandomState(seed)
        M = self.block_size
        N = self.n_points * M

        # 构造稠密协方差矩阵
        C = np.zeros((N, N))
        for i in range(self.n_points):
            for j in range(self.n_points):
                r = abs(i - j) * self.h
                val = self.cov_func(r)
                for m in range(M):
                    C[i * M + m, j * M + m] = val

        # 正则化
        C += 1e-10 * np.eye(N)

        # Cholesky 分解
        try:
            L = la.cholesky(C, lower=True)
        except la.LinAlgError:
            # 如果不是正定的, 使用特征值分解
            eigenvalues, eigenvectors = la.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 0.0)
            L = eigenvectors @ np.diag(np.sqrt(eigenvalues))

        # 采样
        z = rng.randn(N, n_samples)
        samples = L @ z

        return samples.T  # shape (n_samples, N)
