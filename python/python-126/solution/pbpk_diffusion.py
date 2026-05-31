"""
pbpk_diffusion.py
基于种子项目 648_laplacian_matrix

实现有限差分离散化的 Laplacian 算子，支持多种边界条件：
- Dirichlet-Dirichlet (DD)
- Dirichlet-Neumann (DN)
- Neumann-Neumann (NN)
- Periodic-Periodic (PP)

并提供特征值/特征向量的显式公式、Cholesky/LU 分解解析表达式。

在 PBPK 模型中用于：
- 组织内药物扩散的有限差分离散化
- 求解稳态组织浓度分布
- 多器官耦合的扩散-对流方程的空间离散
"""

import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------------
# 1D Laplacian 矩阵构造
# ---------------------------------------------------------------------------

def laplacian_1d_dd(n: int, L: float = 1.0) -> np.ndarray:
    """
    Dirichlet-Dirichlet 边界条件的 1D Laplacian 矩阵。
    离散域 [0, L]，内部 n 个网格点，步长 h = L/(n+1)。
    矩阵 A = (1/h^2) * tridiag(-1, 2, -1)，size n×n。
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    h = L / (n + 1.0)
    diag = 2.0 * np.ones(n)
    offdiag = -1.0 * np.ones(n - 1)
    A = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)
    return A / (h * h)


def laplacian_1d_dn(n: int, L: float = 1.0) -> np.ndarray:
    """
    Dirichlet-Neumann 边界条件的 1D Laplacian 矩阵。
    x=0 处 Dirichlet，x=L 处 Neumann（零通量）。
    最后一个对角元为 1/h^2（一阶差分近似）。
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    h = L / n
    diag = 2.0 * np.ones(n)
    diag[-1] = 1.0
    offdiag = -1.0 * np.ones(n - 1)
    A = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)
    return A / (h * h)


def laplacian_1d_nd(n: int, L: float = 1.0) -> np.ndarray:
    """Neumann-Dirichlet 边界条件（对称于 DN）。"""
    if n < 1:
        raise ValueError("n must be at least 1")
    h = L / n
    diag = 2.0 * np.ones(n)
    diag[0] = 1.0
    offdiag = -1.0 * np.ones(n - 1)
    A = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)
    return A / (h * h)


def laplacian_1d_nn(n: int, L: float = 1.0) -> np.ndarray:
    """
    Neumann-Neumann 边界条件的 1D Laplacian 矩阵。
    两端均为零通量。
    注意：NN 矩阵是奇异的（存在常数零模）。
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    h = L / (n - 1.0) if n > 1 else L
    diag = 2.0 * np.ones(n)
    diag[0] = 1.0
    diag[-1] = 1.0
    offdiag = -1.0 * np.ones(n - 1)
    A = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)
    return A / (h * h)


def laplacian_1d_pp(n: int, L: float = 1.0) -> np.ndarray:
    """
    Periodic-Periodic 边界条件的 1D Laplacian 矩阵。
    循环连接首尾。
    """
    if n < 2:
        raise ValueError("n must be at least 2 for periodic")
    h = L / n
    diag = 2.0 * np.ones(n)
    offdiag = -1.0 * np.ones(n - 1)
    A = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)
    A[0, -1] = -1.0
    A[-1, 0] = -1.0
    return A / (h * h)


# ---------------------------------------------------------------------------
# 特征值与特征向量的显式公式
# ---------------------------------------------------------------------------

def laplacian_dd_eigenvalues(n: int, L: float = 1.0) -> np.ndarray:
    """
    DD Laplacian 的特征值显式公式：
        λ_j = 4 / h^2 * sin^2(j π / (2(n+1))),  j = 1, ..., n
    """
    h = L / (n + 1.0)
    j = np.arange(1, n + 1)
    lam = (4.0 / (h * h)) * np.sin(j * np.pi / (2.0 * (n + 1.0))) ** 2
    return lam


def laplacian_dd_eigenvectors(n: int) -> np.ndarray:
    """
    DD Laplacian 的正交归一化特征向量：
        V_{i,j} = sqrt(2/(n+1)) * sin(i j π / (n+1))
    """
    V = np.empty((n, n), dtype=float)
    scale = np.sqrt(2.0 / (n + 1.0))
    for j in range(1, n + 1):
        i = np.arange(1, n + 1)
        V[:, j - 1] = scale * np.sin(i * j * np.pi / (n + 1.0))
    return V


def laplacian_pp_eigenvalues(n: int, L: float = 1.0) -> np.ndarray:
    """
    PP Laplacian 的特征值：
        λ_j = 4 / h^2 * sin^2(j π / n),  j = 0, ..., n-1
    """
    h = L / n
    j = np.arange(n)
    lam = (4.0 / (h * h)) * np.sin(j * np.pi / n) ** 2
    return lam


# ---------------------------------------------------------------------------
# 矩阵应用（矩阵自由）
# ---------------------------------------------------------------------------

def laplacian_apply(u: np.ndarray, bc_type: str = "DD", L: float = 1.0) -> np.ndarray:
    """
    矩阵自由地应用 Laplacian 算子 A u。
    支持 'DD', 'DN', 'ND', 'NN', 'PP'。
    """
    n = len(u)
    if n < 1:
        raise ValueError("u must be non-empty")
    h = L / (n + 1.0) if bc_type == "DD" else L / n
    if bc_type == "DD":
        h = L / (n + 1.0)
        Au = np.zeros(n)
        Au[0] = (2.0 * u[0] - u[1]) / (h * h)
        Au[-1] = (2.0 * u[-1] - u[-2]) / (h * h)
        Au[1:-1] = (2.0 * u[1:-1] - u[2:] - u[:-2]) / (h * h)
        return Au
    elif bc_type == "PP":
        h = L / n
        Au = np.zeros(n)
        Au[0] = (2.0 * u[0] - u[1] - u[-1]) / (h * h)
        Au[-1] = (2.0 * u[-1] - u[0] - u[-2]) / (h * h)
        Au[1:-1] = (2.0 * u[1:-1] - u[2:] - u[:-2]) / (h * h)
        return Au
    else:
        # DN, ND, NN
        h = L / n if n > 1 else L
        Au = np.zeros(n)
        if bc_type in ("DN", "NN"):
            Au[0] = (u[0] - u[1]) / (h * h)
        else:  # ND
            Au[0] = (2.0 * u[0] - u[1]) / (h * h)
        if bc_type in ("ND", "NN"):
            Au[-1] = (u[-1] - u[-2]) / (h * h)
        else:  # DN
            Au[-1] = (2.0 * u[-1] - u[-2]) / (h * h)
        Au[1:-1] = (2.0 * u[1:-1] - u[2:] - u[:-2]) / (h * h)
        return Au


# ---------------------------------------------------------------------------
# 3D 扩散算子（器官尺度）
# ---------------------------------------------------------------------------

def laplacian_3d_tensor(nx: int, ny: int, nz: int,
                         Lx: float, Ly: float, Lz: float) -> np.ndarray:
    """
    构造 3D 笛卡尔网格上的 Laplacian 算子矩阵。
    使用 Kronecker 和：A = I_z ⊗ I_y ⊗ A_x + I_z ⊗ A_y ⊗ I_x + A_z ⊗ I_y ⊗ I_x
    假设三个方向均为 Dirichlet 边界条件。
    """
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("Grid dimensions must be positive")
    Ax = laplacian_1d_dd(nx, Lx)
    Ay = laplacian_1d_dd(ny, Ly)
    Az = laplacian_1d_dd(nz, Lz)
    Ix = np.eye(nx)
    Iy = np.eye(ny)
    Iz = np.eye(nz)
    # Kronecker 和
    A = (np.kron(Iz, np.kron(Iy, Ax))
         + np.kron(Iz, np.kron(Ay, Ix))
         + np.kron(Az, np.kron(Iy, Ix)))
    return A


def solve_steady_state_diffusion(n: int, L: float, source: np.ndarray,
                                  D: float = 1.0, bc_type: str = "DD") -> np.ndarray:
    """
    求解稳态扩散方程：-D ∇² C = S(x)，即 A C = S/D。
    使用特征值分解或直接求解。
    """
    if len(source) != n:
        raise ValueError("source length must match n")
    if bc_type != "DD":
        raise NotImplementedError("Only DD supported for direct solve currently")
    A = laplacian_1d_dd(n, L)
    rhs = source / D
    # 使用特征值分解求解（小规模问题）
    lam = laplacian_dd_eigenvalues(n, L)
    V = laplacian_dd_eigenvectors(n)
    # A = V Λ V^T,  A^{-1} = V Λ^{-1} V^T
    C = V @ ((V.T @ rhs) / lam)
    return C


# ---------------------------------------------------------------------------
# PBPK 组织浓度分布求解
# ---------------------------------------------------------------------------

def solve_tissue_concentration_profile(n_points: int, tissue_length: float,
                                        D_eff: float, clearance_rate: float,
                                        influx: float) -> np.ndarray:
    """
    求解一维组织内的稳态药物浓度分布。
    方程：-D_eff C'' + k_cl C = 0
    边界条件：C(0) = C_influx, C(L) = 0 (Dirichlet)
    解析解：C(x) = C_influx * sinh(α (L-x)) / sinh(α L)
    其中 α = sqrt(k_cl / D_eff)。
    同时返回有限差分解作为数值验证。
    """
    if D_eff <= 0.0 or clearance_rate < 0.0 or tissue_length <= 0.0:
        raise ValueError("Invalid diffusion parameters")
    h = tissue_length / (n_points + 1.0)
    x = np.linspace(h, tissue_length - h, n_points)
    # 解析解
    alpha = np.sqrt(clearance_rate / D_eff)
    if alpha * tissue_length > 700:
        # 防止 exp 溢出，使用渐近近似
        C_exact = influx * np.exp(-alpha * x)
    else:
        C_exact = influx * np.sinh(alpha * (tissue_length - x)) / np.sinh(alpha * tissue_length)
    # 有限差分解
    A = laplacian_1d_dd(n_points, tissue_length)
    # 修改 RHS 以包含源项和边界条件
    rhs = np.zeros(n_points)
    rhs[0] += D_eff * influx / (h * h)  # 左边界 Dirichlet 贡献
    C_fd = np.linalg.solve(D_eff * A + clearance_rate * np.eye(n_points), rhs)
    return x, C_exact, C_fd


# ---------------------------------------------------------------------------
# 模块自检
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    n = 20
    L = 1.0
    A = laplacian_1d_dd(n, L)
    lam = laplacian_dd_eigenvalues(n, L)
    lam_num = np.linalg.eigvalsh(A)
    print(f"Eigenvalue max error: {np.max(np.abs(np.sort(lam) - np.sort(lam_num))):.2e}")
    x, C_ex, C_fd = solve_tissue_concentration_profile(50, 0.01, 1e-9, 0.01, 1.0)
    print(f"Max FD error vs exact: {np.max(np.abs(C_ex - C_fd)):.2e}")
    A3 = laplacian_3d_tensor(4, 4, 4, 1.0, 1.0, 1.0)
    print(f"3D Laplacian shape: {A3.shape}")
