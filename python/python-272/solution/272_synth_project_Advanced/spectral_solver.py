"""
谱方法求解器模块
实现 Weyl Hamiltonian 本征值问题的高效求解

核心算法：
1. Golub-Welsch 算法 (高斯求积与本征值问题的联系)
2. 隐式 QR 算法
3. 分治法 (divide and conquer)
4. Lanczos 迭代 (大规模稀疏问题)

物理应用：
求解 H(k)|u_n> = E_n(k)|u_n> 获得能带结构
"""

import numpy as np
from typing import Tuple, List, Dict


class SpectralSolver:
    """
    谱方法求解器

    用于求解 Weyl Hamiltonian 的本征值问题
    """

    def __init__(self, method: str = 'qr'):
        """
        初始化

        Parameters:
        -----------
        method : str
            求解方法 ('qr', 'divide_conquer', 'lanczos', 'golub_welsch')
        """
        self.method = method

    def solve_eigenproblem(self, H: np.ndarray,
                           compute_vectors: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解一般 Hermitian 矩阵本征问题

        Parameters:
        -----------
        H : np.ndarray
            Hermitian 矩阵
        compute_vectors : bool
            是否计算本征矢

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            eigenvalues, eigenvectors
        """
        if self.method == 'qr':
            return self._qr_method(H, compute_vectors)
        elif self.method == 'divide_conquer':
            return self._divide_conquer(H, compute_vectors)
        elif self.method == 'lanczos':
            return self._lanczos_method(H, compute_vectors)
        elif self.method == 'golub_welsch':
            return self._golub_welsch(H, compute_vectors)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _qr_method(self, H: np.ndarray,
                   compute_vectors: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        QR 算法求解本征问题

        通过反复 QR 分解迭代，矩阵收敛到 Schur 形式
        """
        n = H.shape[0]
        A = H.copy()
        Q_total = np.eye(n, dtype=complex)

        max_iter = 100
        tol = 1e-12

        for _ in range(max_iter):
            # QR 分解
            Q, R = np.linalg.qr(A)
            A = R @ Q

            if compute_vectors:
                Q_total = Q_total @ Q

            # 检查收敛 (次对角元)
            off_diag = np.sum(np.abs(np.diag(A, 1))) + np.sum(np.abs(np.diag(A, -1)))
            if off_diag < tol:
                break

        eigenvalues = np.real(np.diag(A))
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]

        if compute_vectors:
            eigenvectors = Q_total[:, idx]
        else:
            eigenvectors = np.eye(n)

        return eigenvalues, eigenvectors

    def _divide_conquer(self, H: np.ndarray,
                        compute_vectors: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        分治法求解对称三对角矩阵本征问题

        1. 先三对角化 (Householder)
        2. 分治法求解三对角矩阵
        """
        # Householder 三对角化
        T, Q = self._householder_tridiag(H)

        # 分治法求解三对角矩阵
        n = T.shape[0]
        if n <= 2:
            return np.linalg.eigh(H)

        # 分割
        mid = n // 2
        T1 = T[:mid, :mid]
        T2 = T[mid:, mid:]

        # 递归求解 (简化为直接求解)
        e1, v1 = np.linalg.eigh(T1)
        e2, v2 = np.linalg.eigh(T2)

        # 合并 (简化版本)
        eigenvalues = np.sort(np.concatenate([e1, e2]))

        if compute_vectors:
            _, eigenvectors = np.linalg.eigh(H)
        else:
            eigenvectors = np.eye(n)

        return eigenvalues, eigenvectors

    def _householder_tridiag(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Householder 三对角化

        将对称矩阵 A 转化为三对角矩阵 T = Q^T A Q
        """
        n = A.shape[0]
        T = A.copy()
        Q = np.eye(n, dtype=complex)

        for k in range(n - 2):
            # 构造 Householder 向量
            x = T[k+1:, k].copy()
            alpha = np.linalg.norm(x)
            if x[0] > 0:
                alpha = -alpha

            e1 = np.zeros_like(x)
            e1[0] = 1.0
            v = x - alpha * e1
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-15:
                v = v / v_norm
            else:
                continue

            # Householder 变换
            H_k = np.eye(n, dtype=complex)
            H_k[k+1:, k+1:] -= 2 * np.outer(v, np.conj(v))

            T = H_k @ T @ H_k
            Q = Q @ H_k

        return T, Q

    def _lanczos_method(self, H: np.ndarray,
                        compute_vectors: bool,
                        n_iter: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lanczos 迭代法 (适合大规模稀疏矩阵)

        构造 Krylov 子空间，将大矩阵投影为小三对角矩阵
        """
        n = H.shape[0]
        if n_iter is None:
            n_iter = min(n, 30)

        # Lanczos 过程
        alpha = np.zeros(n_iter)
        beta = np.zeros(n_iter)
        V = np.zeros((n, n_iter), dtype=complex)

        # 初始向量
        v = np.random.randn(n) + 1j * np.random.randn(n)
        v = v / np.linalg.norm(v)
        V[:, 0] = v

        w = H @ v
        alpha[0] = np.real(np.conj(v) @ w)
        w = w - alpha[0] * v

        for j in range(1, n_iter):
            beta[j-1] = np.linalg.norm(w)
            if beta[j-1] < 1e-15:
                n_iter = j
                break

            V[:, j] = w / beta[j-1]
            w = H @ V[:, j] - beta[j-1] * V[:, j-1]
            alpha[j] = np.real(np.conj(V[:, j]) @ w)
            w = w - alpha[j] * V[:, j]

            # 完全重正交化
            for i in range(j+1):
                w = w - np.conj(V[:, i] @ w) * V[:, i]

        # 构建三对角矩阵
        T = np.diag(alpha[:n_iter]) + np.diag(beta[:n_iter-1], 1) + np.diag(beta[:n_iter-1], -1)

        # 求解小本征问题
        e_small, v_small = np.linalg.eigh(T)

        # 转换回原空间
        eigenvalues = e_small
        if compute_vectors:
            eigenvectors = V[:, :n_iter] @ v_small
            # 补齐
            if n_iter < n:
                full_vectors = np.zeros((n, n), dtype=complex)
                full_vectors[:, :n_iter] = eigenvectors
                _, extra = np.linalg.eigh(H)
                full_vectors[:, n_iter:] = extra[:, :n-n_iter]
                eigenvectors = full_vectors
        else:
            eigenvectors = np.eye(n)

        return eigenvalues, eigenvectors

    def _golub_welsch(self, H: np.ndarray,
                      compute_vectors: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Golub-Welsch 算法

        原本用于计算高斯求积节点和权重，
        等价于求解三对角矩阵的本征问题
        """
        # 先三对角化
        T, _ = self._householder_tridiag(H)

        # 提取对角和次对角
        n = T.shape[0]
        diag = np.diag(T)
        off_diag = np.diag(T, 1)

        # Golub-Welsch: 本征值即为求积节点
        # 构建伴随矩阵
        J = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

        eigenvalues, eigenvectors = np.linalg.eigh(J)

        if compute_vectors:
            _, eigenvectors = np.linalg.eigh(H)
        else:
            eigenvectors = np.eye(n)

        return eigenvalues, eigenvectors

    def compute_dos(self, energies: np.ndarray,
                    energy_range: Tuple[float, float],
                    n_bins: int = 100,
                    eta: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算态密度 (DOS)

        ρ(E) = (1/N) Σ_n δ(E - E_n)

        使用 Lorentzian 展宽：
        δ(E) ≈ (1/π) η / (E² + η²)

        Parameters:
        -----------
        energies : np.ndarray
            能量本征值
        energy_range : Tuple[float, float]
            能量范围
        n_bins : int
            能量网格点数
        eta : float
            Lorentzian 展宽参数

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            energy_grid, dos
        """
        E_min, E_max = energy_range
        E_grid = np.linspace(E_min, E_max, n_bins)
        dos = np.zeros(n_bins)

        for E in energies:
            # Lorentzian 展宽
            dos += (1/np.pi) * eta / ((E_grid - E)**2 + eta**2)

        dos /= len(energies)
        return E_grid, dos

    def compute_green_function(self, H: np.ndarray,
                               energy: complex,
                               eta: float = 0.01) -> np.ndarray:
        """
        计算 Green 函数 G(E) = (E + iη - H)^{-1}

        Parameters:
        -----------
        H : np.ndarray
            Hamiltonian
        energy : complex
            能量
        eta : float
            虚部展宽

        Returns:
        --------
        np.ndarray
            Green 函数矩阵
        """
        z = energy + 1j * eta
        G = np.linalg.inv(z * np.eye(H.shape[0]) - H)
        return G


def golub_welsch_nodes_weights(n: int, weight_func: str = 'hermite') -> Tuple[np.ndarray, np.ndarray]:
    """
    Golub-Welsch 算法计算高斯求积节点和权重

    对于正交多项式 p_n(x) 的三项递推：
    x·p_n = a_n·p_{n+1} + b_n·p_n + c_n·p_{n-1}

    节点为三对角矩阵的本征值，权重与本征矢第一分量相关

    Parameters:
    -----------
    n : int
        求积点数
    weight_func : str
        权函数 ('hermite', 'legendre', 'laguerre', 'chebyshev')

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        nodes: 求积节点
        weights: 求积权重
    """
    if weight_func == 'hermite':
        # Hermite 多项式: a_n = sqrt((n+1)/2), b_n = 0
        b = np.zeros(n)
        a = np.sqrt(np.arange(1, n) / 2.0)
    elif weight_func == 'legendre':
        # Legendre 多项式
        b = np.zeros(n)
        i = np.arange(1, n)
        a = i / np.sqrt(4 * i**2 - 1)
    elif weight_func == 'laguerre':
        # Laguerre 多项式
        b = 2 * np.arange(n) + 1
        a = np.arange(1, n)
    elif weight_func == 'chebyshev':
        # Chebyshev 多项式
        b = np.zeros(n)
        a = 0.5 * np.ones(n - 1)
        a[0] = 1.0 / np.sqrt(2)
    else:
        raise ValueError(f"Unknown weight function: {weight_func}")

    # 构建三对角矩阵
    J = np.diag(b) + np.diag(a, 1) + np.diag(a, -1)

    # 求解本征问题
    nodes, V = np.linalg.eigh(J)

    # 权重
    if weight_func == 'hermite':
        mu0 = np.sqrt(np.pi)
    elif weight_func == 'legendre':
        mu0 = 2.0
    elif weight_func == 'laguerre':
        mu0 = 1.0
    elif weight_func == 'chebyshev':
        mu0 = np.pi
    else:
        mu0 = 1.0

    weights = mu0 * V[0, :]**2

    return nodes, weights
