"""
schroedinger_embedding.py
基于非线性薛定谔方程的谱流形嵌入
融合原项目: 1061_schroedinger_nonlinear_pde

核心科学思想:
将高维数据降维问题转化为非线性薛定谔方程(NLSE)的谱分析问题。
在数据流形上定义等效势能场，通过求解NLSE的特征函数，
获得数据的非线性谱嵌入表示。

数学模型:
考虑非线性薛定谔方程:
    i ∂ψ/∂t = -½ Δψ + V(x)ψ + γ|ψ|²ψ

在离散数据流形上，Laplacian Δ 由图Laplacian L 近似。
定义等效势能:
    V(x_i) = Σ_j K(x_i, x_j) / Σ_j K(x_i, x_j)

通过虚时间演化 (τ = it)，方程变为扩散型:
    ∂ψ/∂τ = ½ L ψ - V ψ - γ|ψ|²ψ

稳态解对应于广义特征值问题:
    L φ = λ D φ
    其中 D 为度矩阵

有限差分离散化:
    ∂ψ/∂t ≈ (ψ^{n+1} - ψ^n) / Δt
    采用Crank-Nicolson格式保证数值稳定性
"""

import numpy as np
from typing import Tuple, Optional
from linear_algebra_core import jacobi_eigenvalue, solve_cholesky, cholesky_factor


def nonlinear_schroedinger_parameters() -> dict:
    """NLSE物理参数"""
    return {
        'alpha': 1.0,      # 色散系数
        'c1': 1.0,         # 线性耦合
        'c2': 0.5,         # 非线性耦合
        'delta': 0.1,      # 耗散系数
        'gamma': -0.5,     # 非线性强度
        't0': 0.0,         # 初始时间
        'xmin': -5.0,      # 空间下限
        'xmax': 5.0        # 空间上限
    }


def build_effective_potential(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    基于数据密度构建等效势能场
    V_i = -log(ρ_i + ε)
    其中 ρ_i = Σ_j exp(-||x_i - x_j||² / (2σ²))
    """
    # TODO [Hole 1]: 实现基于数据密度的等效势能场构建
    # 科学公式: V_i = -log(ρ_i + ε), 其中 ρ_i = Σ_j exp(-||x_i - x_j||² / (2σ²))
    # 需要计算每个数据点的高斯核密度，取负对数后归一化
    raise NotImplementedError("Hole 1: build_effective_potential 待实现")


def schroedinger_deriv(u: np.ndarray, L: np.ndarray, V: np.ndarray,
                        gamma: float = -0.5) -> np.ndarray:
    """
    非线性薛定谔方程右端项 (虚时间演化)
    du/dτ = L u - V u - γ|u|² u
    """
    nonlin = gamma * np.abs(u) ** 2 * u
    dudt = -0.5 * (L @ u) - V * u - nonlin
    return dudt


def finite_difference_evolve(u0: np.ndarray, L: np.ndarray, V: np.ndarray,
                              dt: float = 0.01, n_steps: int = 1000,
                              gamma: float = -0.5) -> np.ndarray:
    """
    有限差分法求解NLSE虚时间演化
    采用显式-隐式混合格式 (IMEX)
    """
    N = len(u0)
    u = u0.copy()
    # 预计算矩阵
    A = np.eye(N) + 0.5 * dt * L
    B = np.eye(N) - 0.5 * dt * L
    for step in range(n_steps):
        # 非线性项显式处理
        nonlin = gamma * np.abs(u) ** 2 * u
        rhs = B @ u - dt * (V * u + nonlin)
        # 使用numpy求解线性系统
        u_new = np.linalg.solve(A, rhs)
        # 归一化保持概率守恒
        norm = np.linalg.norm(u_new)
        if norm > 1e-15:
            u_new = u_new / norm
        # 收敛检查
        if step > 0 and step % 20 == 0:
            residual = np.linalg.norm(u_new - u_old)
            if residual < 1e-7:
                u = u_new
                break
        u_old = u_new.copy()
        u = u_new
    return u


def schroedinger_spectral_embedding(data: np.ndarray, L: np.ndarray,
                                     n_components: int = 3,
                                     sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于NLSE谱分析的流形嵌入
    求解广义特征值问题: L φ = λ D φ
    取前n_components个最小非零特征值对应的特征向量
    """
    N = len(data)
    V = build_effective_potential(data, sigma)
    # 构建度矩阵
    D = np.diag(np.sum(np.exp(-np.linalg.norm(data[:, None] - data[None, :], axis=2) ** 2 / (2.0 * sigma ** 2)), axis=1))
    # 正则化
    D_reg = D + 1e-6 * np.eye(N)
    # 转换为标准特征值问题: D^{-1/2} L D^{-1/2} ψ = λ ψ
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_reg)))
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt
    # 使用稳定特征值分解求解
    eigvals, eigvecs = np.linalg.eigh(L_sym)
    # 取最小的非零特征值 (排序后是最后几个，因为Jacobi按降序排)
    # 实际上我们要最小的，所以从后往前取
    idx = np.argsort(eigvals)
    # 跳过第一个（接近0的）
    selected = idx[1:n_components + 1]
    eigenvalues = eigvals[selected]
    # 变换回原始坐标: φ = D^{-1/2} ψ
    embedding = D_inv_sqrt @ eigvecs[:, selected]
    return embedding, eigenvalues


def nonlinear_spectral_coordinates(data: np.ndarray, n_components: int = 3,
                                    sigma: float = 1.0, gamma: float = -0.5,
                                    n_iterations: int = 50) -> np.ndarray:
    """
    非线性谱坐标: 通过迭代非线性薛定谔演化提取坐标
    """
    N, D = data.shape
    from neighbor_graph import build_knn_graph, graph_laplacian
    edges, weights = build_knn_graph(data, k=min(10, N - 1))
    L = graph_laplacian(edges, weights, N, normalize=True)
    V = build_effective_potential(data, sigma)
    coordinates = []
    for comp in range(n_components):
        # 随机初始条件 (正交化)
        np.random.seed(42 + comp)
        u0 = np.random.randn(N)
        u0 = u0 / np.linalg.norm(u0)
        for prev in coordinates:
            u0 = u0 - np.dot(u0, prev) * prev
            u0 = u0 / (np.linalg.norm(u0) + 1e-15)
        # NLSE演化 (减少步数加速)
        u = finite_difference_evolve(u0, L, V, dt=0.02, n_steps=80, gamma=gamma)
        # 再次正交化
        for prev in coordinates:
            u = u - np.dot(u, prev) * prev
            u = u / (np.linalg.norm(u) + 1e-15)
        coordinates.append(u)
    return np.array(coordinates).T


def schroedinger_energy(u: np.ndarray, L: np.ndarray, V: np.ndarray,
                         gamma: float = -0.5) -> float:
    """
    NLSE能量泛函:
        E[ψ] = ½ <ψ, L ψ> + <ψ, V ψ> + (γ/2) <|ψ|⁴>
    """
    # TODO [Hole 2]: 实现NLSE能量泛函计算
    # 科学公式: E[ψ] = ½ <ψ, L ψ> + <ψ, V ψ> + (γ/2) <|ψ|⁴>
    # 注意: <ψ, L ψ> 表示 ψ^T (L @ ψ), <|ψ|⁴> 表示 Σ_i |ψ_i|⁴
    raise NotImplementedError("Hole 2: schroedinger_energy 待实现")
