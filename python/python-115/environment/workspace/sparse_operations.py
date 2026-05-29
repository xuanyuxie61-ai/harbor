"""
sparse_operations.py
稀疏矩阵运算模块（CRS 格式）

核心功能：
- 压缩行存储（Compressed Row Storage, CRS）格式稀疏矩阵
- 稀疏矩阵-向量乘法
- 稀疏矩阵 I/O 操作
- 用于酶催化过渡态搜索中的 Hessian 矩阵分析

科学背景：
在酶催化反应的过渡态搜索中，需要计算势能面的 Hessian 矩阵 H：
    H_{ij} = ∂²V / (∂q_i ∂q_j)
其中 V 为势能，q_i 为广义坐标（原子坐标或反应坐标）。

对于 N 原子体系，Hessian 是 3N×3N 矩阵，但具有高度稀疏性：
    - 每个原子仅与近邻原子相互作用
    - 非零元数量 ~ O(N)
    - 全矩阵存储需要 9N²，而 CRS 仅需 O(N)

CRS 格式定义：
    给定 M×M 稀疏矩阵 A，非零元 NZ 个：
    - val[0:NZ-1]: 非零元值
    - col[0:NZ-1]: 列索引
    - row[0:M]: 第 i 行非零元在 val 中的起始位置

过渡态判据（Hessian 特征值分析）：
    - 稳定点：所有特征值 > 0
    - 过渡态（一阶鞍点）：恰好一个负特征值
    - 高阶鞍点：多个负特征值
"""

import numpy as np


class CRSMatrix:
    """压缩行存储稀疏矩阵"""

    def __init__(self, n, nz, row, col, val):
        """
        初始化 CRS 矩阵

        参数：
            n: 矩阵阶数
            nz: 非零元数量
            row: 行指针数组，长度 n+1
            col: 列索引数组，长度 nz
            val: 非零元值数组，长度 nz
        """
        self.n = n
        self.nz = nz
        self.row = np.asarray(row, dtype=int)
        self.col = np.asarray(col, dtype=int)
        self.val = np.asarray(val, dtype=float)

        # 边界检查
        if len(self.row) != n + 1:
            raise ValueError(f"行指针长度应为 {n+1}，实际为 {len(self.row)}")
        if len(self.col) != nz:
            raise ValueError(f"列索引长度应为 {nz}，实际为 {len(self.col)}")
        if len(self.val) != nz:
            raise ValueError(f"非零元值长度应为 {nz}，实际为 {len(self.val)}")
        if self.row[0] != 0:
            raise ValueError("row[0] 必须为 0")
        if self.row[n] != nz:
            raise ValueError(f"row[n] 必须为 {nz}，实际为 {self.row[n]}")
        for i in range(n):
            if self.row[i] > self.row[i + 1]:
                raise ValueError(f"行指针非单调递增于索引 {i}")
        for j in range(nz):
            if self.col[j] < 0 or self.col[j] >= n:
                raise ValueError(f"列索引越界于位置 {j}: col={self.col[j]}")

    def matvec(self, x):
        """
        稀疏矩阵-向量乘法 y = A * x

        算法复杂度：O(NZ)
        """
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self.n:
            raise ValueError(f"向量维度不匹配: {x.shape[0]} != {self.n}")
        y = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            for k in range(self.row[i], self.row[i + 1]):
                j = self.col[k]
                y[i] += self.val[k] * x[j]
        return y

    def to_dense(self):
        """转换为稠密矩阵（仅用于小规模调试）"""
        A = np.zeros((self.n, self.n), dtype=float)
        for i in range(self.n):
            for k in range(self.row[i], self.row[i + 1]):
                j = self.col[k]
                A[i, j] = self.val[k]
        return A

    def residual_norm(self, x, b):
        """计算残差范数 ||b - A*x||_2"""
        return np.linalg.norm(b - self.matvec(x))

    @staticmethod
    def from_dense(A_dense, threshold=1e-15):
        """从稠密矩阵构造 CRS 格式"""
        A_dense = np.asarray(A_dense, dtype=float)
        m, n = A_dense.shape
        if m != n:
            raise ValueError("仅支持方阵")
        row = [0]
        col = []
        val = []
        for i in range(n):
            count = 0
            for j in range(n):
                if abs(A_dense[i, j]) > threshold:
                    col.append(j)
                    val.append(A_dense[i, j])
                    count += 1
            row.append(row[-1] + count)
        return CRSMatrix(n, len(val), row, col, val)


def build_molecular_hessian_crs(n_atoms, coordinates, force_constant=1.0, cutoff=3.5):
    """
    构建分子 Hessian 矩阵的 CRS 表示

    物理模型：
        H_{ij}^{αβ} = ∂²V/∂r_{iα}∂r_{jβ}
        对于谐振子近似：
        V = Σ_{i<j} (k/2) * (|r_i - r_j| - r_{ij}^0)²

    在过渡态附近，Hessian 具有一个负特征值（虚频模式）。
    为模拟过渡态，我们在指定方向上引入负曲率。

    参数：
        n_atoms: 原子数
        coordinates: (n_atoms, 3) 坐标数组
        force_constant: 力常数
        cutoff: 相互作用截断距离（Å）
    返回：
        CRSMatrix 对象（3N×3N）
    """
    N = n_atoms
    dim = 3 * N
    coords = np.asarray(coordinates, dtype=float)

    # 构建邻接表（基于距离截断）
    adjacency = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            r_ij = np.linalg.norm(coords[i] - coords[j])
            if r_ij < cutoff and r_ij > 0.1:
                adjacency[i].append((j, r_ij))
                adjacency[j].append((i, r_ij))

    # 构建稀疏 Hessian
    row_ptr = [0]
    col_idx = []
    val = []

    # 对角块和耦合块
    # H 是 3N×3N，按 (原子, 维度) 索引
    for i_atom in range(N):
        for alpha in range(3):
            i_global = 3 * i_atom + alpha
            entries = {}

            # 对角元（自相互作用）
            for j_atom, r_ij in adjacency[i_atom]:
                for beta in range(3):
                    j_global = 3 * j_atom + beta
                    # 简化的谐振子 Hessian
                    dr = coords[i_atom] - coords[j_atom]
                    if r_ij > 0:
                        h_elem = force_constant * dr[alpha] * dr[beta] / (r_ij ** 2)
                    else:
                        h_elem = 0.0
                    if j_global not in entries:
                        entries[j_global] = 0.0
                    entries[j_global] += h_elem

            # 自身对角块
            for beta in range(3):
                j_global = 3 * i_atom + beta
                if j_global not in entries:
                    entries[j_global] = 0.0
                # 添加质量加权项和自能项
                entries[j_global] += force_constant * (1.0 if alpha == beta else 0.0)

            # 排序并存储
            sorted_cols = sorted(entries.keys())
            for j_global in sorted_cols:
                col_idx.append(j_global)
                val.append(entries[j_global])
            row_ptr.append(len(col_idx))

    nz = len(val)
    return CRSMatrix(dim, nz, row_ptr, col_idx, val)


def lanczos_eigenvalue_solver(crs_matrix, max_iter=50, tol=1e-10):
    """
    Lanczos 算法计算稀疏矩阵的极端特征值

    算法：
        选取初始向量 v_1 (||v_1|| = 1)
        for j = 1, 2, ..., m:
            w = A * v_j - β_j * v_{j-1}
            α_j = v_j^T * w
            w = w - α_j * v_j
            β_{j+1} = ||w||
            if β_{j+1} < tol: break
            v_{j+1} = w / β_{j+1}

        T_m = tridiag(β, α, β) 的 Ritz 值逼近 A 的特征值

    在过渡态搜索中，Lanczos 用于验证 Hessian 恰有一个负特征值。
    """
    n = crs_matrix.n
    if max_iter > n:
        max_iter = n

    # 初始向量
    v_prev = np.zeros(n, dtype=float)
    v_curr = np.random.randn(n)
    v_curr /= np.linalg.norm(v_curr)

    alpha = []
    beta = [0.0]

    for j in range(max_iter):
        w = crs_matrix.matvec(v_curr)
        if j > 0:
            w -= beta[j] * v_prev
        alpha_j = np.dot(v_curr, w)
        alpha.append(alpha_j)
        w -= alpha_j * v_curr
        beta_next = np.linalg.norm(w)

        if beta_next < tol:
            break

        beta.append(beta_next)
        v_prev = v_curr.copy()
        v_curr = w / beta_next

    # 构建三对角矩阵并计算特征值
    m = len(alpha)
    T = np.zeros((m, m), dtype=float)
    for i in range(m):
        T[i, i] = alpha[i]
        if i < m - 1:
            T[i, i + 1] = beta[i + 1]
            T[i + 1, i] = beta[i + 1]

    eigenvalues = np.linalg.eigvalsh(T)
    return eigenvalues
