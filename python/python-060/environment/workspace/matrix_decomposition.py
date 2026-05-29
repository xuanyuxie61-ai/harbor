"""
矩阵分解模块

本模块实现平流层化学模型中所需的矩阵分解算法，包括：
- Cholesky 分解 (对称正定矩阵)
- 协方差矩阵的构建与分解
- 预处理矩阵构造

科学背景:
在大气化学模型的统计分析和不确定性量化中，经常需要处理
对称正定(SPD)矩阵，例如:
1. 参数协方差矩阵
2. 有限元/有限差分离散化的刚度矩阵
3. Gauss-Newton 优化的正规方程矩阵

科学公式:
1. Cholesky 分解:
   对于 SPD 矩阵 A，存在唯一下三角矩阵 L 使得:
   A = L L^T
   其中 L_ii > 0

2. 前向/后向替换求解:
   L y = b  =>  y_i = (b_i - Σ_{j<i} L_ij y_j) / L_ii
   L^T x = y  =>  x_i = (y_i - Σ_{j>i} L_ji x_j) / L_ii

3. 矩阵条件数估计:
   κ(A) = ||A|| * ||A^{-1}||
   通过 Cholesky 分解可估计: κ ≈ max(diag(L)) / min(diag(L))

4. 不完全 Cholesky 预处理:
   仅保留 L 中与 A 相同位置的元素

融入原项目: 026_asa007 (Cholesky 分解)
"""

import numpy as np
from typing import Tuple, Optional


class CholeskyDecomposition:
    """
    Cholesky 分解器
    融入 asa007 的核心算法
    """

    def __init__(self, eta: float = 1e-12):
        """
        Parameters
        ----------
        eta : float
            数值稳定性阈值
        """
        self.eta = eta

    def decompose(self, A: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Cholesky 分解: A = L L^T

        Parameters
        ----------
        A : ndarray
            对称正定矩阵 (n, n)

        Returns
        -------
        L : ndarray
            下三角矩阵
        nullty : int
            秩亏量 (0 表示满秩)
        ifault : int
            错误指示 (0: 无错误, 1: n<1, 2: 非半正定)
        """
        n = A.shape[0]
        if n <= 0:
            return np.array([]), 0, 1

        if A.shape[0] != A.shape[1]:
            raise ValueError("A 必须是方阵")

        L = np.zeros((n, n))
        nullty = 0
        ifault = 0

        for icol in range(n):
            # 对角线元素
            sum_sq = 0.0
            for irow in range(icol):
                sum_sq += L[icol, irow] ** 2

            diag_val = A[icol, icol] - sum_sq

            # 检查正定性
            if diag_val < -self.eta * abs(A[icol, icol]):
                ifault = 2
                return L, nullty, ifault
            elif abs(diag_val) <= self.eta * abs(A[icol, icol]):
                L[icol, icol] = 0.0
                nullty += 1
            else:
                L[icol, icol] = np.sqrt(max(diag_val, 0.0))

            # 非对角线元素
            for jcol in range(icol + 1, n):
                sum_prod = 0.0
                for irow in range(icol):
                    sum_prod += L[jcol, irow] * L[icol, irow]

                if L[icol, icol] > self.eta:
                    L[jcol, icol] = (A[jcol, icol] - sum_prod) / L[icol, icol]
                else:
                    L[jcol, icol] = 0.0

        return L, nullty, ifault

    def solve(self, L: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        使用 Cholesky 分解求解 A x = b
        即 L L^T x = b

        Parameters
        ----------
        L : ndarray
            Cholesky 因子 (下三角)
        b : ndarray
            右端项

        Returns
        -------
        x : ndarray
            解向量
        """
        n = L.shape[0]
        if len(b) != n:
            raise ValueError("b 长度与 L 不匹配")

        # 前向替换: L y = b
        y = np.zeros(n)
        for i in range(n):
            if abs(L[i, i]) < 1e-30:
                y[i] = 0.0
            else:
                y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

        # 后向替换: L^T x = y
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            if abs(L[i, i]) < 1e-30:
                x[i] = 0.0
            else:
                x[i] = (y[i] - np.dot(L[i + 1:, i], x[i + 1:])) / L[i, i]

        return x

    def inverse(self, L: np.ndarray) -> np.ndarray:
        """
        通过 Cholesky 分解计算 A^{-1}
        """
        n = L.shape[0]
        A_inv = np.zeros((n, n))

        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            A_inv[:, j] = self.solve(L, e_j)

        return A_inv

    def log_determinant(self, L: np.ndarray) -> float:
        """
        计算 log(det(A)) = 2 * Σ log(L_ii)
        """
        diag = np.diag(L)
        if np.any(diag <= 0):
            return -np.inf
        return 2.0 * np.sum(np.log(diag))

    def condition_number_estimate(self, L: np.ndarray) -> float:
        """
        估计矩阵条件数
        κ ≈ max(diag(L)) / min(diag(L))
        """
        diag = np.diag(L)
        diag = diag[diag > 1e-30]
        if len(diag) == 0:
            return np.inf
        return np.max(diag) / np.min(diag)


class CovarianceMatrixHandler:
    """
    协方差矩阵处理器
    用于化学参数的不确定性分析
    """

    def __init__(self):
        self.cholesky = CholeskyDecomposition()

    def build_from_correlation(self, sigmas: np.ndarray,
                                correlation: np.ndarray) -> np.ndarray:
        """
        从标准差和相关性矩阵构建协方差矩阵
        Σ_ij = ρ_ij * σ_i * σ_j
        """
        n = len(sigmas)
        if correlation.shape != (n, n):
            raise ValueError("correlation 矩阵维度不匹配")

        Sigma = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Sigma[i, j] = correlation[i, j] * sigmas[i] * sigmas[j]

        # 确保对称
        Sigma = 0.5 * (Sigma + Sigma.T)

        # 确保正定性
        eigvals = np.linalg.eigvalsh(Sigma)
        if np.min(eigvals) < 1e-14:
            Sigma += (1e-12 - np.min(eigvals)) * np.eye(n)

        return Sigma

    def sample_multivariate_normal(self, mu: np.ndarray,
                                    Sigma: np.ndarray,
                                    n_samples: int = 1,
                                    seed: int = 42) -> np.ndarray:
        """
        使用 Cholesky 分解生成多元正态样本
        X = μ + L Z, Z ~ N(0, I)
        """
        L, nullty, ifault = self.cholesky.decompose(Sigma)

        if ifault != 0:
            # 添加正则化
            Sigma_reg = Sigma + 1e-10 * np.eye(Sigma.shape[0])
            L, _, _ = self.cholesky.decompose(Sigma_reg)

        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n_samples, len(mu)))
        X = mu + Z @ L.T
        return X

    def mahalanobis_distance(self, x: np.ndarray, mu: np.ndarray,
                              Sigma: np.ndarray) -> float:
        """
        计算 Mahalanobis 距离
        D² = (x - μ)^T Σ^{-1} (x - μ)
        """
        L, nullty, ifault = self.cholesky.decompose(Sigma)
        if ifault != 0:
            Sigma_reg = Sigma + 1e-10 * np.eye(Sigma.shape[0])
            L, _, _ = self.cholesky.decompose(Sigma_reg)

        diff = x - mu
        y = self.cholesky.solve(L, diff)
        return np.dot(y, y)


class PreconditionerBuilder:
    """
    预处理矩阵构造器
    用于迭代求解器的预处理
    """

    def __init__(self):
        self.cholesky = CholeskyDecomposition()

    def jacobi_preconditioner(self, A: np.ndarray) -> np.ndarray:
        """
        Jacobi 预处理: M = diag(A)^{-1}
        """
        diag = np.diag(A)
        diag = np.where(np.abs(diag) > 1e-30, diag, 1.0)
        return np.diag(1.0 / diag)

    def incomplete_cholesky(self, A: np.ndarray,
                             fill_level: int = 0) -> np.ndarray:
        """
        不完全 Cholesky 预处理
        仅保留与 A 同位置的元素
        """
        n = A.shape[0]
        L = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1):
                if A[i, j] == 0 and fill_level == 0:
                    continue

                if i == j:
                    sum_sq = np.sum(L[i, :j] ** 2)
                    val = A[i, i] - sum_sq
                    if val > 1e-14:
                        L[i, j] = np.sqrt(val)
                else:
                    sum_prod = np.sum(L[i, :j] * L[j, :j])
                    if L[j, j] > 1e-30:
                        L[i, j] = (A[i, j] - sum_prod) / L[j, j]

        return L

    def ssor_preconditioner(self, A: np.ndarray,
                             omega: float = 1.0) -> np.ndarray:
        """
        SSOR 预处理矩阵
        M = (D + omega L) D^{-1} (D + omega L^T) / (omega(2-omega))
        """
        n = A.shape[0]
        D = np.diag(np.diag(A))
        L_strict = np.tril(A, -1)

        D_inv = np.diag(1.0 / (np.diag(A) + 1e-30))

        M = (D + omega * L_strict) @ D_inv @ (D + omega * L_strict.T)
        M = M / (omega * (2.0 - omega) + 1e-30)

        return M
