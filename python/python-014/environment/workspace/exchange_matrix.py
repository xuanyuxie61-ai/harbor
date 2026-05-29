"""
exchange_matrix.py
==================
交换相互作用矩阵的构造、存储与操作模块。
融合来源：laplacian_matrix（1D离散拉普拉斯算子构造）、r8ss（skyline对称稀疏格式）。

本模块实现：
- 1D/2D/3D 离散 Laplacian 型交换矩阵（Dirichlet/Neumann/Periodic 边界）
- Skyline 压缩存储，适配大规模阻挫自旋系统
- 矩阵-向量乘法与 Cholesky 预条件子（简化版）
- 能量泛函的二次型表示：E = 0.5 * Σ_{ij} J_{ij} S_i · S_j
"""

import numpy as np
from typing import Tuple, Optional
from utils import build_skyline_from_tridiagonal, skyline_mv, EPS_MACHINE


def exchange_laplacian_1d(n: int, h: float, bc: str = "dirichlet") -> np.ndarray:
    """
    一维离散 Laplacian 型交换矩阵。
    对应连续算子：-(d^2/dx^2) 的有限差分，用于模拟铁磁链中的交换作用。

    能量泛函离散化：
        E ≈ (1/2h) Σ_i (S_i - S_{i+1})^2  =>  矩阵形式 E = S^T L S / 2

    参数
    ----
    n : int
        内部格点数，n >= 3。
    h : float
        格点间距。
    bc : str
        边界条件："dirichlet", "neumann", "periodic"。

    返回
    ----
    L : np.ndarray, shape (n, n)
        拉普拉斯矩阵。
    """
    if n < 3:
        raise ValueError("n must be >= 3")
    if h <= 0.0:
        raise ValueError("h must be positive")

    L = np.zeros((n, n), dtype=float)
    inv_h2 = 1.0 / (h * h)

    if bc == "dirichlet":
        # 两端固定，矩阵为标准的 tridiagonal( -1, 2, -1 )
        L[0, 0] = 2.0 * inv_h2
        L[0, 1] = -1.0 * inv_h2
        for i in range(1, n - 1):
            L[i, i - 1] = -1.0 * inv_h2
            L[i, i] = 2.0 * inv_h2
            L[i, i + 1] = -1.0 * inv_h2
        L[n - 1, n - 2] = -1.0 * inv_h2
        L[n - 1, n - 1] = 2.0 * inv_h2

    elif bc == "neumann":
        # 零流边界，首/尾对角元为 1
        L[0, 0] = 1.0 * inv_h2
        L[0, 1] = -1.0 * inv_h2
        for i in range(1, n - 1):
            L[i, i - 1] = -1.0 * inv_h2
            L[i, i] = 2.0 * inv_h2
            L[i, i + 1] = -1.0 * inv_h2
        L[n - 1, n - 2] = -1.0 * inv_h2
        L[n - 1, n - 1] = 1.0 * inv_h2

    elif bc == "periodic":
        # 周期边界
        L[0, 0] = 2.0 * inv_h2
        L[0, 1] = -1.0 * inv_h2
        L[0, n - 1] = -1.0 * inv_h2
        for i in range(1, n - 1):
            L[i, i - 1] = -1.0 * inv_h2
            L[i, i] = 2.0 * inv_h2
            L[i, i + 1] = -1.0 * inv_h2
        L[n - 1, 0] = -1.0 * inv_h2
        L[n - 1, n - 2] = -1.0 * inv_h2
        L[n - 1, n - 1] = 2.0 * inv_h2
    else:
        raise ValueError(f"Unknown boundary condition: {bc}")

    return L


def exchange_laplacian_2d(nx: int, ny: int, hx: float, hy: float, bc: str = "dirichlet") -> np.ndarray:
    """
    二维离散 Laplacian 型交换矩阵，通过 Kronecker 和构造：
        L = I_y ⊗ L_x + L_y ⊗ I_x
    总维度 N = nx * ny，当 N 很大时建议用稀疏格式。
    """
    if nx < 2 or ny < 2:
        raise ValueError("nx, ny must be >= 2")
    Lx = exchange_laplacian_1d(nx, hx, bc)
    Ly = exchange_laplacian_1d(ny, hy, bc)
    Ix = np.eye(nx)
    Iy = np.eye(ny)
    L = np.kron(Iy, Lx) + np.kron(Ly, Ix)
    return L


def exchange_skyline_1d(n: int, h: float, bc: str = "dirichlet") -> Tuple[int, np.ndarray, np.ndarray]:
    """
    将一维 Laplacian 转为 skyline 格式，返回 (na, diag_idx, a)。
    融合来源：r8ss_dif2（R8SS 差分矩阵构造）。
    """
    L = exchange_laplacian_1d(n, h, bc)
    # 对于三对角对称矩阵，lower = upper = 次对角线
    lower = np.diag(L, -1).copy()
    diag = np.diag(L).copy()
    return build_skyline_from_tridiagonal(lower, diag, lower)


def apply_exchange_operator(J: np.ndarray, spins: np.ndarray) -> np.ndarray:
    """
    计算有效交换场：H_i = Σ_j J_{ij} S_j。
    对于三维自旋，spins 形状为 (N, 3)。
    """
    if spins.ndim == 1:
        # 标量/Ising 情形
        return J @ spins
    elif spins.ndim == 2 and spins.shape[1] == 3:
        # Heisenberg 矢量情形：逐分量矩阵乘法
        hx = J @ spins[:, 0]
        hy = J @ spins[:, 1]
        hz = J @ spins[:, 2]
        return np.column_stack([hx, hy, hz])
    else:
        raise ValueError("spins must be 1D (Ising) or 2D with shape (N, 3) (Heisenberg)")


def exchange_energy(J: np.ndarray, spins: np.ndarray) -> float:
    """
    计算交换能量：
        E = (1/2) Σ_{i,j} J_{ij} S_i · S_j
    对于反铁磁耦合 J>0，基态为相邻反平行排列。
    """
    H = apply_exchange_operator(J, spins)
    if spins.ndim == 1:
        e = 0.5 * np.dot(spins, H)
    else:
        e = 0.5 * np.sum(spins * H)
    return float(e)


def add_disorder(J: np.ndarray, std: float, seed: Optional[int] = None) -> np.ndarray:
    """向耦合矩阵添加高斯键无序（Edwards-Anderson 型自旋玻璃）。"""
    if seed is not None:
        np.random.seed(seed)
    N = J.shape[0]
    disorder = np.random.normal(0.0, std, size=(N, N))
    disorder = (disorder + disorder.T) * 0.5  # 保证对称
    np.fill_diagonal(disorder, 0.0)
    return J + disorder
