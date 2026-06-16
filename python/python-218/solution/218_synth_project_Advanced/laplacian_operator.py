"""
laplacian_operator.py
=====================
离散 Laplacian 算子及其在变分不等式中的应用.

数学背景
--------
连续 Laplacian: Δu = Σ_i ∂²u/∂x_i²

离散化 (五点差分, 二维):
    (Δu)_{i,j} ≈ (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}) / h²

在障碍问题 (Obstacle Problem, VI 的经典应用) 中:
    -Δu ≥ f  在 Ω 中
    u ≥ φ    在 Ω 中  (φ 为障碍函数)
    (-Δu - f)(u - φ) = 0

这是一个典型的无穷维变分不等式, 离散化后成为有限维 VI.

本模块实现:
    1. 一维/二维/三维离散 Laplacian 的稀疏矩阵构建
    2. 带 Dirichlet/Neumann 边界条件的 Laplacian
    3. 分数阶 Laplacian (-Δ)^s (0 < s < 1) 的谱近似

关键公式
--------
分数阶 Laplacian 的特征值:
    (-Δ)^s φ_k = λ_k^s φ_k
其中 λ_k, φ_k 为标准 Laplacian 的特征对.

作者: DA synthesis project
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from typing import Tuple, Optional


class DiscreteLaplacian:
    """
    离散 Laplacian 算子的稀疏矩阵表示.

    一维: 三对角 [-1, 2, -1] / h²
    二维: 五点差分 (十字形模板)
    三维: 七点差分
    """

    @staticmethod
    def build_1d(n: int, h: float = 1.0) -> sparse.csr_matrix:
        """
        一维离散 Laplacian (负 Laplacian, 正定).

        矩阵形式:
            D = (1/h²) · tridiag(-1, 2, -1)

        边界条件: Dirichlet (u_0 = u_{n+1} = 0).

        特征值 (解析):
            λ_k = (2/h²)(1 - cos(kπ/(n+1))),  k = 1, ..., n
        """
        main_diag = 2.0 * np.ones(n) / h**2
        off_diag = -1.0 * np.ones(n - 1) / h**2
        D = sparse.diags(
            [off_diag, main_diag, off_diag],
            offsets=[-1, 0, 1],
            shape=(n, n),
            format='csr',
        )
        return D

    @staticmethod
    def build_2d(nx: int, ny: int, hx: float = 1.0, hy: float = 1.0) -> sparse.csr_matrix:
        """
        二维离散 Laplacian (五点差分).

        网格: (nx+2) × (ny+2), 内部点 nx × ny.
        未知数编号: (i,j) → i + j*nx (按列优先).

        对内部点 (i,j):
            (Δu)_{i,j} = (u_{i-1,j} - 2u_{i,j} + u_{i+1,j})/hx²
                        + (u_{i,j-1} - 2u_{i,j} + u_{i,j+1})/hy²

        Returns
        -------
        D : sparse matrix (nx*ny, nx*ny)
            负 Laplacian (正定)
        """
        N = nx * ny
        diag_vals = 2.0 / hx**2 + 2.0 / hy**2
        rows, cols, vals = [], [], []

        for j in range(ny):
            for i in range(nx):
                p = i + j * nx
                rows.append(p)
                cols.append(p)
                vals.append(diag_vals)

                # 西邻 (i-1, j)
                if i > 0:
                    rows.append(p)
                    cols.append(p - 1)
                    vals.append(-1.0 / hx**2)
                # 东邻 (i+1, j)
                if i < nx - 1:
                    rows.append(p)
                    cols.append(p + 1)
                    vals.append(-1.0 / hx**2)
                # 南邻 (i, j-1)
                if j > 0:
                    rows.append(p)
                    cols.append(p - nx)
                    vals.append(-1.0 / hy**2)
                # 北邻 (i, j+1)
                if j < ny - 1:
                    rows.append(p)
                    cols.append(p + nx)
                    vals.append(-1.0 / hy**2)

        D = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
        return D

    @staticmethod
    def eigenvalues_1d(n: int, h: float = 1.0) -> np.ndarray:
        """
        一维离散 Laplacian 的解析特征值.

        λ_k = (2/h²)(1 - cos(kπ/(n+1))),  k = 1, ..., n

        这些特征值在谱方法中用于构建分数阶 Laplacian.
        """
        k = np.arange(1, n + 1)
        return (2.0 / h**2) * (1.0 - np.cos(k * np.pi / (n + 1)))

    @staticmethod
    def eigenvectors_1d(n: int) -> np.ndarray:
        """
        一维离散 Laplacian 的特征向量矩阵.

        Φ_{jk} = sqrt(2/(n+1)) · sin(jkπ/(n+1))
        """
        j = np.arange(1, n + 1)[:, np.newaxis]
        k = np.arange(1, n + 1)[np.newaxis, :]
        Phi = np.sqrt(2.0 / (n + 1)) * np.sin(j * k * np.pi / (n + 1))
        return Phi


class FractionalLaplacian:
    """
    分数阶 Laplacian (-Δ)^s, 0 < s < 1.

    定义 (谱方法):
        (-Δ)^s u = Σ_k λ_k^s · ⟨u, φ_k⟩ · φ_k

    其中 λ_k, φ_k 为标准 Laplacian 的特征对.

    物理应用:
        - 反常扩散: ∂u/∂t = -(-Δ)^s u
        - 分数阶障碍问题: min(u - φ, (-Δ)^s u - f) = 0
        - Lévy 过程的无穷小生成元
    """

    def __init__(self, n: int, s: float, h: float = 1.0):
        """
        Parameters
        ----------
        n : int
            离散点数
        s : float
            分数阶指数, 0 < s < 1
        h : float
            网格间距
        """
        if not (0 < s < 1):
            raise ValueError(f"分数阶指数 s 必须在 (0,1) 内, 得到 s={s}")
        self.n = n
        self.s = s
        self.h = h
        # 预计算特征分解
        self.eigvals = DiscreteLaplacian.eigenvalues_1d(n, h)
        self.eigvecs = DiscreteLaplacian.eigenvectors_1d(n)
        # 分数阶特征值
        self.frac_eigvals = self.eigvals ** s

    def apply(self, u: np.ndarray) -> np.ndarray:
        """
        计算 (-Δ)^s u 通过谱分解.

        (-Δ)^s u = Φ · diag(λ^s) · Φ^T · u
        """
        coeffs = self.eigvecs.T @ u
        return self.eigvecs @ (self.frac_eigvals * coeffs)

    def build_matrix(self) -> np.ndarray:
        """构建分数阶 Laplacian 的稠密矩阵."""
        return self.eigvecs @ np.diag(self.frac_eigvals) @ self.eigvecs.T

    def condition_number(self) -> float:
        """条件数 = λ_max^s / λ_min^s."""
        return float((self.eigvals[-1] / self.eigvals[0]) ** self.s)


class ObstacleProblemVI:
    """
    障碍问题 (Obstacle Problem): VI 的经典模型.

    求 u ∈ K = {v ∈ H¹₀(Ω) : v ≥ φ a.e.} 使得
        ∫_Ω ∇u · ∇(v - u) dx ≥ ∫_Ω f(v - u) dx,  ∀ v ∈ K

    离散化后:
        求 u ≥ φ 使得 (u - φ)^T (Du - f) = 0,  Du - f ≥ 0.

    等价 NCP:
        min(u - φ, Du - f) = 0

    其中 D 为离散 Laplacian, φ 为障碍函数, f 为源项.
    """

    def __init__(
        self,
        n: int,
        obstacle: np.ndarray,
        source: np.ndarray,
        h: float = 1.0,
    ):
        """
        Parameters
        ----------
        n : int
            网格点数
        obstacle : ndarray (n,)
            障碍函数 φ 的离散值
        source : ndarray (n,)
            源项 f 的离散值
        h : float
            网格间距
        """
        self.n = n
        self.obstacle = obstacle.copy()
        self.source = source.copy()
        self.h = h
        self.D = DiscreteLaplacian.build_1d(n, h)

    def evaluate_F(self, u: np.ndarray) -> np.ndarray:
        """计算 F(u) = Du - f."""
        return self.D @ u - self.source

    def feasibility_residual(self, u: np.ndarray) -> float:
        """可行性残差: ||min(u - φ, 0)||."""
        violation = np.minimum(u - self.obstacle, 0.0)
        return float(np.linalg.norm(violation))

    def complementarity_residual(self, u: np.ndarray) -> float:
        """互补残差: ||min(u - φ, F(u))||."""
        shifted = u - self.obstacle
        F_u = self.evaluate_F(u)
        return float(np.linalg.norm(np.minimum(shifted, F_u)))
