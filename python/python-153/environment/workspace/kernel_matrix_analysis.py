"""
kernel_matrix_analysis.py
基于项目 1004_r8vm (Vandermonde矩阵), 161_chebyshev_matrix (谱微分),
与 207_condition (条件数估计) 的量子核矩阵分析模块。

核心数学模型:
1. 量子核矩阵:
   K_{ij} = |<0^n| U^dagger(x_i) U(x_j) |0^n>|^2
   对称半正定，满足 Mercer's 定理。

2. Vandermonde 矩阵紧凑存储:
   V_{ij} = x_j^{i-1},  i=1..m, j=1..n
   行列式: det(V) = prod_{1<=j<i<=n} (x_i - x_j)
   用于量子振幅的多项式插值编码。

3. Chebyshev 谱微分矩阵:
   x_k = cos(k*pi/n), k=0..n  (Chebyshev-Gauss-Lobatto 节点)
   D_{ij} = (c_i/c_j) * 1/(x_i - x_j), i≠j
   D_{ii} = -sum_{j≠i} D_{ij}
   用于量子动力学方程的高效空间离散。

4. 条件数分析:
   kappa_1(A) = ||A||_1 * ||A^{-1}||_1
   Hager 算法: 迭代寻找使 ||A^{-1} x||_1 最大的单位向量 x
   LINPACK 算法: 基于 LU 分解的贪心构造极端右端向量
"""

import numpy as np
from typing import Tuple, Optional


def vandermonde_determinant(x: np.ndarray) -> float:
    """
    计算 Vandermonde 矩阵的行列式。
    det(V) = prod_{1<=j<i<=n} (x_i - x_j)
    时间复杂度 O(n^2)。
    """
    n = len(x)
    if n == 0:
        return 1.0
    det = 1.0
    for i in range(1, n):
        for j in range(i):
            det *= (x[i] - x[j])
    return det


def chebyshev_grid(n: int) -> np.ndarray:
    """
    生成 n 阶 Chebyshev-Gauss-Lobatto 节点。
    x_k = cos(k * pi / n), k = 0, 1, ..., n
    在 [-1, 1] 上非均匀分布，两端密集。
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.array([1.0])
    k = np.arange(n + 1)
    return np.cos(np.pi * k / n)


def chebyshev_differentiation_matrix(n: int) -> np.ndarray:
    """
    构造 (n+1) x (n+1) 的 Chebyshev 谱微分矩阵 D。
    对定义在 Chebyshev 网格上的函数值向量 v，离散导数为 w = D @ v。

    非对角元: D_{ij} = (c_i / c_j) * 1 / (x_i - x_j), i≠j
    对角元:   D_{ii} = -sum_{j≠i} D_{ij}  (保证 D @ 1 = 0)
    其中 c_0 = c_n = 2, c_k = 1 (k=1..n-1)
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    x = chebyshev_grid(n)
    c = np.ones(n + 1)
    c[0] = 2.0
    c[n] = 2.0
    c = c * ((-1.0) ** np.arange(n + 1))

    X = np.tile(x[:, np.newaxis], (1, n + 1))
    dX = X - X.T

    # 非对角元
    D = (c[:, np.newaxis] / c[np.newaxis, :]) / (dX + np.eye(n + 1))
    D = D - np.diag(np.diag(D))  # 移除对角元

    # 对角元: 行和取负
    D = D - np.diag(D.sum(axis=1))

    return D


def plu_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    带部分主元选取的 PLU 分解。
    PA = LU, 其中 P 为置换矩阵，L 为单位下三角，U 为上三角。

    算法:
    对 k = 1, ..., n-1:
        1. 在第 k 列下方找绝对值最大元素作为主元
        2. 记录 pivot 索引，若主元为 0 则矩阵奇异
        3. 行交换
        4. 计算乘子 L_{ik} = A_{ik} / A_{kk}
        5. 行消元
    """
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    L = np.eye(n)
    U = A.copy().astype(np.float64)
    P = np.eye(n)

    for k in range(n - 1):
        # 部分主元选取
        pivot_idx = k + np.argmax(np.abs(U[k:, k]))
        if abs(U[pivot_idx, k]) < 1e-15:
            continue  # 矩阵奇异或接近奇异

        # 行交换
        if pivot_idx != k:
            U[[k, pivot_idx], :] = U[[pivot_idx, k], :]
            P[[k, pivot_idx], :] = P[[pivot_idx, k], :]
            if k > 0:
                L[[k, pivot_idx], :k] = L[[pivot_idx, k], :k]

        # 消元
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return P, L, U


def hager_condition_number_estimate(A: np.ndarray, max_iter: int = 5) -> float:
    """
    Hager L1 条件数估计算法。
    估计 kappa_1(A) = ||A||_1 * ||A^{-1}||_1。

    算法步骤:
    1. 初始化 b = ones(n) / n
    2. 循环:
       a) 解 A x = b
       b) c = sum(|x|), 更新 b = sign(x)
       c) 解 A^T y = b
       d) 找到 |y| 最大分量的索引 i_max
       e) 若索引重复或 c 不再增长则停止
    3. cond = c * ||A||_1
    """
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    anorm = np.linalg.norm(A, ord=1)

    # 初始化
    b = np.ones(n) / n
    old_index = -1

    for _ in range(max_iter):
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.inf

        c = np.sum(np.abs(x))
        b = np.sign(x)
        # 处理 x_i = 0 的情况
        b[x == 0] = 1.0

        try:
            y = np.linalg.solve(A.T, b)
        except np.linalg.LinAlgError:
            return np.inf

        new_index = np.argmax(np.abs(y))
        if new_index == old_index or abs(np.abs(y[new_index]) - c) < 1e-10:
            break
        old_index = new_index
        b = np.zeros(n)
        b[new_index] = 1.0

    # 最终求解
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.inf

    c_final = np.sum(np.abs(x))
    cond = c_final * anorm
    return cond


def sample_condition_estimate(A: np.ndarray, n_samples: int = 20) -> float:
    """
    随机采样法估计 L1 条件数。
    生成随机单位向量 x，估计 ||A||_1 和 ||A^{-1}||_1。
    """
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    a_norm = 0.0
    ainv_norm = 0.0

    for _ in range(n_samples):
        # 单位球面上随机向量
        x = np.random.randn(n)
        x = x / (np.linalg.norm(x) + 1e-15)

        ax = A @ x
        a_norm = max(a_norm, np.linalg.norm(ax, ord=1))

        try:
            ainv_x = np.linalg.solve(A, x)
            ainv_norm = max(ainv_norm, np.linalg.norm(ainv_x, ord=1))
        except np.linalg.LinAlgError:
            return np.inf

    if ainv_norm < 1e-15:
        return np.inf
    return a_norm * ainv_norm


class QuantumKernelMatrix:
    """
    量子核矩阵的构造、分析与求解模块。
    """

    def __init__(self, kernel_func, data_points: np.ndarray):
        """
        参数:
            kernel_func: 核函数 k(x_i, x_j)
            data_points: 形状为 (n_samples, n_features) 的数据矩阵
        """
        self.data_points = np.array(data_points, dtype=np.float64)
        self.n_samples = self.data_points.shape[0]
        self.kernel_func = kernel_func
        self._K: Optional[np.ndarray] = None
        self._K_inv: Optional[np.ndarray] = None

    def compute_kernel_matrix(self) -> np.ndarray:
        """计算核矩阵 K_{ij} = k(x_i, x_j)。"""
        K = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            for j in range(i, self.n_samples):
                val = self.kernel_func(self.data_points[i], self.data_points[j])
                # 边界处理: 核值必须在 [0, 1] 内
                val = max(0.0, min(1.0, val))
                K[i, j] = val
                K[j, i] = val

        # 数值鲁棒性: 对称化
        K = (K + K.T) / 2.0

        # 确保半正定: 对负特征值截断
        eigvals = np.linalg.eigvalsh(K)
        if np.min(eigvals) < -1e-10:
            K += (-np.min(eigvals) + 1e-10) * np.eye(self.n_samples)

        self._K = K
        return K

    def condition_number(self) -> float:
        """计算核矩阵的条件数 (2-范数)。"""
        if self._K is None:
            self.compute_kernel_matrix()

        K = self._K
        eigvals = np.linalg.eigvalsh(K)
        pos_eigvals = eigvals[eigvals > 1e-15]
        if len(pos_eigvals) == 0:
            return np.inf
        return np.max(eigvals) / np.min(pos_eigvals)

    def hager_cond_estimate(self) -> float:
        """使用 Hager 算法估计 L1 条件数。"""
        if self._K is None:
            self.compute_kernel_matrix()
        return hager_condition_number_estimate(self._K)

    def solve_kernel_system(self, y: np.ndarray, reg: float = 1e-6) -> np.ndarray:
        """
        求解核岭回归系统: (K + reg*I) alpha = y
        使用 LU 分解求解。
        """
        if self._K is None:
            self.compute_kernel_matrix()

        K_reg = self._K + reg * np.eye(self.n_samples)
        try:
            alpha = np.linalg.solve(K_reg, y)
        except np.linalg.LinAlgError:
            # 若直接求解失败，使用伪逆
            alpha = np.linalg.lstsq(K_reg, y, rcond=1e-10)[0]

        self._K_inv = alpha
        return alpha

    def kernel_target_alignment(self, y: np.ndarray) -> float:
        """
        计算核目标对齐度 (Kernel Target Alignment, KTA)。
        KTA = <K, y y^T>_F / (||K||_F * ||y y^T||_F)
        衡量核矩阵与标签矩阵的相似度。
        """
        if self._K is None:
            self.compute_kernel_matrix()

        y = np.array(y, dtype=np.float64)
        Y = np.outer(y, y)

        k_norm = np.linalg.norm(self._K, "fro")
        y_norm = np.linalg.norm(Y, "fro")
        if k_norm < 1e-15 or y_norm < 1e-15:
            return 0.0

        inner = np.sum(self._K * Y)
        return inner / (k_norm * y_norm)


def quantum_kernel_with_vandermonde(
    x: np.ndarray,
    x_prime: np.ndarray,
    n_qubits: int = 4
) -> float:
    """
    使用 Vandermonde 编码的量子核函数。
    将数据点编码为 Vandermonde 矩阵的特征向量，计算重叠。
    """
    if len(x) != len(x_prime):
        raise ValueError("Input vectors must have same length")

    n = min(len(x), n_qubits)
    # 构造 Vandermonde 矩阵 (紧凑存储，仅使用定义向量)
    v_x = np.array([x[i] ** j for j in range(n) for i in range(n)], dtype=np.float64)
    v_xp = np.array([x_prime[i] ** j for j in range(n) for i in range(n)], dtype=np.float64)

    # 归一化并计算重叠
    norm_x = np.linalg.norm(v_x)
    norm_xp = np.linalg.norm(v_xp)
    if norm_x < 1e-15 or norm_xp < 1e-15:
        return 0.0

    overlap = np.dot(v_x, v_xp) / (norm_x * norm_xp)
    return overlap ** 2
