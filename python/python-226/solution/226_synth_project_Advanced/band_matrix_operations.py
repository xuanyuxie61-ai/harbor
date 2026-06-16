"""
带状矩阵存储与Jacobi迭代模块
============================
实现对称正定带状矩阵的紧凑存储与高效运算，
以及Jacobi迭代法求解中微子哈密顿量的本征值问题。

核心数据结构：
1. R8PBL格式（Real 8-byte Positive definite Band Lower）：
   对于N×N对称正定矩阵A，半带宽ML
   存储数组: a[ML+1, N]
   - a[0, j] = A[j, j] (对角元)
   - a[k, j] = A[j+k, j] (第k条下次对角线)

2. 矩阵向量乘法：
   b_i = Σ_j A_{ij} x_j = a[0,i]*x[i] + Σ_{k=1}^{ML} a[k,i-k]*x[i-k] + Σ_{k=1}^{ML} a[k,i]*x[i+k]

3. Jacobi迭代：
   x^{(k+1)}_i = (b_i - Σ_{j≠i} A_{ij} x^{(k)}_j) / A_{ii}
   收敛条件: ρ(D⁻¹(L+U)) < 1 (谱半径)

4. 中微子哈密顿量的本征值分解：
   H v = λ v
   使用Jacobi旋转法精确求解3×3厄米特矩阵

数据来源：
- 603_jacobi: Jacobi迭代的基础实现
- 987_r8pbl: 对称正定带状矩阵的紧凑存储格式
"""

import numpy as np
from typing import Tuple, Optional


class R8PBLMatrix:
    """
    对称正定带状矩阵（R8PBL格式）。

    紧凑存储: a[ML+1, N]
    - 行0: 对角元 A[j,j]
    - 行k: 第k条下次对角线 A[j+k, j]

    内存效率：从 O(N²) 降低到 O(N × ML)

    物理应用：
    - 中微子传播算子的带状结构
    - 有限差分/有限元离散化的刚度矩阵
    """

    def __init__(self, N: int, ML: int):
        """
        初始化R8PBL矩阵。

        参数：
            N: 矩阵维度
            ML: 半带宽（下次对角线数）
        """
        if N <= 0:
            raise ValueError(f"矩阵维度必须为正: N={N}")
        if ML < 0 or ML >= N:
            raise ValueError(f"半带宽无效: ML={ML}, N={N}")

        self.N = N
        self.ML = ML
        self.data = np.zeros((ML + 1, N), dtype=float)

    def set_diagonal(self, values: np.ndarray):
        """
        设置对角元。

        参数：
            values: 对角元数组 (N,)
        """
        if len(values) != self.N:
            raise ValueError(f"对角元长度不匹配: {len(values)} vs {self.N}")
        self.data[0, :] = values

    def set_subdiagonal(self, k: int, values: np.ndarray):
        """
        设置第k条下次对角线。

        参数：
            k: 对角线索引 (1 ≤ k ≤ ML)
            values: 对角线值 (N-k,)
        """
        if k < 1 or k > self.ML:
            raise ValueError(f"对角线索引越界: k={k}, ML={self.ML}")
        if len(values) != self.N - k:
            raise ValueError(f"对角线长度不匹配: {len(values)} vs {self.N - k}")
        self.data[k, :self.N - k] = values

    def get_element(self, i: int, j: int) -> float:
        """
        获取矩阵元素 A[i, j]。

        利用对称性: A[i, j] = A[j, i]

        参数：
            i, j: 行列索引

        返回：
            A[i, j]: 矩阵元素
        """
        if i < 0 or i >= self.N or j < 0 or j >= self.N:
            raise IndexError(f"索引越界: ({i}, {j})")

        # 确保 i >= j
        if i < j:
            i, j = j, i

        k = i - j
        if k > self.ML:
            return 0.0  # 在带状区域外

        return self.data[k, j]

    def set_element(self, i: int, j: int, value: float):
        """
        设置矩阵元素 A[i, j]。

        参数：
            i, j: 行列索引
            value: 值
        """
        if i < j:
            i, j = j, i  # 利用对称性

        k = i - j
        if k > self.ML:
            if abs(value) > 1e-15:
                raise ValueError(f"位置({i},{j})在带状区域外(k={k}>ML={self.ML})")
            return

        self.data[k, j] = value

    def matrix_vector_product(self, x: np.ndarray) -> np.ndarray:
        """
        计算矩阵向量乘积 b = A x。

        算法：
        b_i = a[0,i]*x[i] + Σ_{k=1}^{ML} (a[k,i]*x[i+k] + a[k,i-k]*x[i-k])

        利用对称性，只需遍历上三角部分。

        参数：
            x: 输入向量 (N,)

        返回：
            b: 输出向量 (N,)
        """
        if len(x) != self.N:
            raise ValueError(f"向量长度不匹配: {len(x)} vs {self.N}")

        b = np.zeros(self.N)

        # 对角元贡献
        b += self.data[0, :] * x

        # 各条下次对角线贡献
        for k in range(1, self.ML + 1):
            for j in range(self.N - k):
                # A[j+k, j] = a[k, j]
                val = self.data[k, j]
                b[j + k] += val * x[j]  # 下三角
                b[j] += val * x[j + k]  # 上三角（对称）

        return b

    def to_dense(self) -> np.ndarray:
        """
        转换为稠密矩阵。

        返回：
            A: N×N稠密矩阵
        """
        A = np.zeros((self.N, self.N))

        for j in range(self.N):
            # 对角元
            A[j, j] = self.data[0, j]

            # 下次对角线
            for k in range(1, self.ML + 1):
                if j + k < self.N:
                    A[j + k, j] = self.data[k, j]
                    A[j, j + k] = self.data[k, j]  # 对称

        return A

    @classmethod
    def from_dense(cls, A: np.ndarray, ML: int = None) -> 'R8PBLMatrix':
        """
        从稠密矩阵构造R8PBL矩阵。

        参数：
            A: N×N对称矩阵
            ML: 半带宽（如果为None则自动检测）

        返回：
            band_matrix: R8PBLMatrix对象
        """
        N = A.shape[0]
        if A.shape != (N, N):
            raise ValueError(f"矩阵必须是方阵: {A.shape}")

        # 自动检测半带宽
        if ML is None:
            ML = 0
            for i in range(N):
                for j in range(i):
                    if abs(A[i, j]) > 1e-15:
                        ML = max(ML, i - j)

        band = cls(N, ML)

        for j in range(N):
            band.data[0, j] = A[j, j]
            for k in range(1, ML + 1):
                if j + k < N:
                    band.data[k, j] = A[j + k, j]

        return band

    def is_positive_definite(self, tol: float = 1e-10) -> bool:
        """
        检查矩阵是否正定。

        方法：尝试Cholesky分解

        参数：
            tol: 容差

        返回：
            is_pd: 是否正定
        """
        try:
            A_dense = self.to_dense()
            np.linalg.cholesky(A_dense)
            return True
        except np.linalg.LinAlgError:
            return False

    def random_spd(self, seed: int = None) -> 'R8PBLMatrix':
        """
        生成随机对称正定带状矩阵。

        方法：
        1. 随机填充下次对角线 a[k, j] ∈ [0, 1]
        2. 设置对角元保证严格对角优势:
           a[0, j] = Σ_{k=1}^{ML} (|a[k, j]| + |a[k, j-k]|) + δ

        参数：
            seed: 随机种子

        返回：
            self (修改后)
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        # 随机填充下次对角线
        for k in range(1, self.ML + 1):
            self.data[k, :self.N - k] = rng.uniform(0, 1, self.N - k)

        # 设置对角元保证对角优势
        for j in range(self.N):
            row_sum = 0.0
            for k in range(1, self.ML + 1):
                if j >= k:
                    row_sum += abs(self.data[k, j - k])
                if j + k < self.N:
                    row_sum += abs(self.data[k, j])
            self.data[0, j] = row_sum + 1.0  # 严格对角优势

        return self


class JacobiSolver:
    """
    Jacobi迭代法求解器。

    用于求解线性系统 Ax = b 和本征值问题 Av = λv。

    1. 线性系统Jacobi迭代：
       x^{(k+1)}_i = (b_i - Σ_{j≠i} A_{ij} x^{(k)}_j) / A_{ii}
       矩阵形式: x^{(k+1)} = D⁻¹(b - (L+U)x^{(k)})

       收敛条件: ρ(D⁻¹(L+U)) < 1
       对于对角优势矩阵保证收敛。

    2. 本征值问题的Jacobi旋转法：
       通过一系列正交相似变换将对称矩阵化为对角阵
       A^{(k+1)} = P_k^T A^{(k)} P_k
       其中 P_k 是Givens旋转矩阵

       对于3×3中微子哈密顿量，通常10次迭代内收敛。
    """

    def __init__(self, max_iter: int = 10000, tol: float = 1e-12):
        """
        初始化Jacobi求解器。

        参数：
            max_iter: 最大迭代次数
            tol: 收敛容差
        """
        self.max_iter = max_iter
        self.tol = tol

    def jacobi_linear_solve(self, A: np.ndarray, b: np.ndarray,
                            x0: np.ndarray = None) -> Tuple[np.ndarray, dict]:
        """
        Jacobi迭代求解线性系统 Ax = b。

        算法：
        1. 分解 A = D + L + U (对角+下三角+上三角)
        2. 迭代: x^{(k+1)} = D⁻¹(b - (L+U)x^{(k)})
        3. 检查收敛: ||x^{(k+1)} - x^{(k)}|| < tol

        参数：
            A: 系数矩阵 (N×N)
            b: 右端向量 (N,)
            x0: 初始猜测 (N,)

        返回：
            x: 解向量
            info: 迭代信息字典
        """
        N = len(b)
        if A.shape != (N, N):
            raise ValueError(f"矩阵维度不匹配: A={A.shape}, b={b.shape}")

        # 提取对角元
        d = np.diag(A).copy()
        if np.any(np.abs(d) < 1e-15):
            raise ValueError("对角元接近零，Jacobi迭代不适用")

        # 初始化
        if x0 is None:
            x = np.zeros(N)
        else:
            x = x0.copy()

        # Jacobi迭代矩阵: T = -D⁻¹(L+U) = I - D⁻¹A
        D_inv = np.diag(1.0 / d)
        T = np.eye(N) - D_inv @ A
        c = D_inv @ b

        # 迭代
        converged = False
        residual_norms = []

        for iteration in range(self.max_iter):
            x_new = T @ x + c

            # 检查收敛
            diff = np.linalg.norm(x_new - x)
            residual = np.linalg.norm(A @ x_new - b)
            residual_norms.append(residual)

            if diff < self.tol:
                converged = True
                x = x_new
                break

            x = x_new

        info = {
            'converged': converged,
            'iterations': iteration + 1 if converged else self.max_iter,
            'final_residual': residual_norms[-1] if residual_norms else float('inf'),
            'residual_history': residual_norms,
            'spectral_radius': np.max(np.abs(np.linalg.eigvals(T))),
        }

        return x, info

    def jacobi_eigenvalues(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Jacobi旋转法求解对称矩阵的本征值问题。

        算法：
        1. 找到非对角元中绝对值最大的元素 A[p, q]
        2. 构造Givens旋转矩阵 P(p, q, θ) 使得 A'[p,q] = 0
           θ = 0.5 × arctan(2A[p,q] / (A[q,q] - A[p,p]))
        3. 更新 A' = P^T A P
        4. 重复直到所有非对角元 < tol

        参数：
            A: N×N实对称矩阵

        返回：
            eigenvalues: 本征值 (N,)
            eigenvectors: 本征向量 (N×N, 列向量)
            info: 收敛信息
        """
        N = A.shape[0]
        if A.shape != (N, N):
            raise ValueError(f"矩阵必须是方阵: {A.shape}")

        # 检查对称性
        sym_error = np.linalg.norm(A - A.T)
        if sym_error > 1e-10:
            A = 0.5 * (A + A.T)  # 强制对称

        S = A.copy()
        V = np.eye(N)  # 累积旋转矩阵

        converged = False
        off_diag_norms = []

        for iteration in range(self.max_iter):
            # 计算非对角元范数
            off_diag = S - np.diag(np.diag(S))
            off_norm = np.linalg.norm(off_diag, 'fro')
            off_diag_norms.append(off_norm)

            if off_norm < self.tol:
                converged = True
                break

            # 找到最大非对角元
            max_idx = np.unravel_index(np.argmax(np.abs(off_diag)), S.shape)
            p, q = max_idx

            if p == q:
                break

            # 计算旋转角
            if abs(S[q, q] - S[p, p]) < 1e-15:
                theta = np.pi / 4
            else:
                theta = 0.5 * np.arctan(2 * S[p, q] / (S[q, q] - S[p, p]))

            # 构造Givens旋转
            c_rot = np.cos(theta)
            s_rot = np.sin(theta)

            # 应用旋转: S' = P^T S P
            # 只更新受影响的行和列
            S_new = S.copy()

            # 第p行和第q行
            for j in range(N):
                if j != p and j != q:
                    S_new[p, j] = c_rot * S[p, j] - s_rot * S[q, j]
                    S_new[j, p] = S_new[p, j]
                    S_new[q, j] = s_rot * S[p, j] + c_rot * S[q, j]
                    S_new[j, q] = S_new[q, j]

            # (p,p), (q,q), (p,q) 元素
            S_new[p, p] = c_rot**2 * S[p, p] - 2 * s_rot * c_rot * S[p, q] + s_rot**2 * S[q, q]
            S_new[q, q] = s_rot**2 * S[p, p] + 2 * s_rot * c_rot * S[p, q] + c_rot**2 * S[q, q]
            S_new[p, q] = 0.0
            S_new[q, p] = 0.0

            S = S_new

            # 更新本征向量矩阵
            V_new = V.copy()
            for i in range(N):
                V_new[i, p] = c_rot * V[i, p] - s_rot * V[i, q]
                V_new[i, q] = s_rot * V[i, p] + c_rot * V[i, q]
            V = V_new

        eigenvalues = np.diag(S)
        # 排序
        sort_idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = V[:, sort_idx]

        info = {
            'converged': converged,
            'iterations': iteration + 1 if converged else self.max_iter,
            'final_off_diag_norm': off_diag_norms[-1] if off_diag_norms else float('inf'),
        }

        return eigenvalues, eigenvectors, info

    def hermitian_eigenvalues(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        求解厄米特矩阵的本征值问题（中微子哈密顿量）。

        对于复厄米特矩阵 H = H†：
        1. 分解为实部 H_R 和虚部 H_I: H = H_R + i H_I
        2. 构造2N×2N实对称矩阵:
           M = [[H_R, H_I], [-H_I, H_R]]
        3. 对M使用Jacobi旋转法
        4. 提取本征值和本征向量

        或者直接使用numpy/scipy的厄米特求解器。

        参数：
            H: N×N厄米特矩阵（复数）

        返回：
            eigenvalues: 本征值 (N,) 实数
            eigenvectors: 本征向量 (N×N) 复数
            info: 收敛信息
        """
        N = H.shape[0]

        # 验证厄米特性
        hermitian_error = np.linalg.norm(H - H.conj().T)
        if hermitian_error > 1e-10:
            H = 0.5 * (H + H.conj().T)

        # 使用numpy的厄米特求解器
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        info = {
            'converged': True,
            'hermitian_error': hermitian_error,
            'eigenvalue_range': (np.min(eigenvalues), np.max(eigenvalues)),
        }

        return eigenvalues, eigenvectors, info


def neutrino_hamiltonian_band(H_flavor: np.ndarray, ML: int = 2) -> R8PBLMatrix:
    """
    将中微子哈密顿量转换为带状矩阵（仅实部）。

    物理背景：
    在特定基底和离散化方案下，中微子传播的哈密顿量可以具有带状结构。

    参数：
        H_flavor: 味基底中的哈密顿量 (3×3复数)
        ML: 半带宽

    返回：
        band_matrix: R8PBL格式的带状矩阵
    """
    # 取实部构造对称矩阵
    H_real = np.real(H_flavor)
    H_sym = 0.5 * (H_real + H_real.T)

    band = R8PBLMatrix.from_dense(H_sym, ML=min(ML, 2))

    return band


def jacobi_iteration_single(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    执行单步Jacobi迭代。

    x_new_i = (b_i - Σ_{j≠i} A_{ij} x_j) / A_{ii}

    参数：
        A: 系数矩阵
        b: 右端向量
        x: 当前迭代值

    返回：
        x_new: 新的迭代值
    """
    N = len(b)
    d = np.diag(A)
    x_new = np.zeros(N)

    for i in range(N):
        if abs(d[i]) < 1e-15:
            x_new[i] = x[i]
        else:
            off_diag_sum = np.dot(A[i, :], x) - A[i, i] * x[i]
            x_new[i] = (b[i] - off_diag_sum) / d[i]

    return x_new
