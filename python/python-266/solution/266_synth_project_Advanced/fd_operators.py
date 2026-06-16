"""
fd_operators.py — 高阶有限差分算子与 Bloch 边界条件
====================================================

本模块实现 Kohn-Sham 方程的高阶有限差分离散化。
融合种子项目:
  - 271_dg1d_advection: Vandermonde 矩阵, 微分矩阵, Jacobi 多项式
  - 362_fd1d_heat_steady: 稳态 FD 模板, 三对角稀疏求解
  - 225_cpr: Chebyshev 节点 (用于谱方法对比)

核心物理:
  Kohn-Sham 动能算子的 FD 离散:
    T_FD = -(ℏ²/2m) · Δ_FD

  其中 Δ_FD 为 2p 阶精度的 Laplacian 有限差分:
    (Δ_FD ψ)_i = (1/dx²) Σ_{j=-p}^{p} c_j ψ_{i+j}

  Bloch 边界条件:
    ψ(x + a) = e^{ika} ψ(x)
    ψ'(x + a) = e^{ika} ψ'(x)

  在 FD 网格上实现:
    ψ_{N+j} = e^{ika} ψ_j   for j = 1,...,p

  这导致 Hamilton 矩阵的非零元素从三对角扩展为带状矩阵,
  且在右上角和左下角出现 e^{±ika} 的非零元素。
"""

import numpy as np
from scipy import sparse
from typing import Tuple, Optional
from physical_constants import fd_coefficients_2nd, PI, TWO_PI


# ============================================================
# FD Laplacian 矩阵构建
# ============================================================

def build_laplacian_matrix(n_grid: int, dx: float, fd_order: int,
                           k_point: float = 0.0,
                           a: float = 1.0) -> np.ndarray:
    """
    构建带 Bloch 边界条件的 FD Laplacian 矩阵。

    对于 2p 阶精度的中心差分, Laplacian 矩阵为:
      L[i,j] = c_{j-i} / dx²

    其中 c_j 为 FD 系数 (物理常数模块中计算)。

    Bloch 边界条件的实现:
    对于超出边界的索引, 使用:
      ψ_{i+N} = e^{ika} ψ_i
      ψ_{i-N} = e^{-ika} ψ_i

    因此矩阵元素:
      L[i, j] += c_{j-i+N} / dx² · e^{-ika}   (左环绕)
      L[i, j] += c_{j-i-N} / dx² · e^{ika}    (右环绕)

    物理约束:
    - 矩阵必须是 Hermitian (实对称当 k=0)
    - 迹 = N · c_0/dx² = -2N/dx² · Σ_{j>0} c_j
    - 本征值 ≤ 0 (Laplacian 为负定算子)

    Parameters
    ----------
    n_grid : int
        网格点数 N
    dx : float
        网格间距
    fd_order : int
        FD 半带宽 p
    k_point : float
        k 点 (默认 Γ 点 k=0)
    a : float
        原胞长度

    Returns
    -------
    L : np.ndarray, shape (N, N)
        Laplacian 矩阵 (稠密形式)
    """
    if n_grid < 2 * fd_order + 1:
        raise ValueError(
            f"网格点数 {n_grid} 不足以支持 FD 半带宽 {fd_order}. "
            f"需要至少 {2 * fd_order + 1} 点.")

    c = fd_coefficients_2nd(fd_order)
    p = fd_order

    # Bloch 相位
    phase = np.exp(1j * k_point * a)
    phase_conj = np.conj(phase)

    L = np.zeros((n_grid, n_grid), dtype=complex)

    for i in range(n_grid):
        for j_idx in range(-p, p + 1):
            j_phys = i + j_idx  # 物理索引 (可能超出 [0, N))

            if 0 <= j_phys < n_grid:
                # 直接贡献
                L[i, j_phys] += c[p + j_idx] / (dx ** 2)
            elif j_phys >= n_grid:
                # 右环绕: ψ_{j_phys} = e^{ika} ψ_{j_phys - N}
                j_wrapped = j_phys - n_grid
                L[i, j_wrapped] += c[p + j_idx] / (dx ** 2) * phase
            else:
                # 左环绕: ψ_{j_phys} = e^{-ika} ψ_{j_phys + N}
                j_wrapped = j_phys + n_grid
                L[i, j_wrapped] += c[p + j_idx] / (dx ** 2) * phase_conj

    return L


def build_kinetic_matrix(n_grid: int, dx: float, fd_order: int,
                          k_point: float = 0.0,
                          a: float = 1.0) -> np.ndarray:
    """
    构建动能算子矩阵 T = -(1/2) L (原子单位 ℏ = m_e = 1)。

    T = -(1/2) · Δ_FD

    本征值为正: ε_k = k²_mod / 2 ≥ 0

    Parameters
    ----------
    n_grid : int
        网格点数
    dx : float
        网格间距
    fd_order : int
        FD 半带宽
    k_point : float
        k 点
    a : float
        原胞长度

    Returns
    -------
    T : np.ndarray, shape (N, N)
        动能矩阵
    """
    L = build_laplacian_matrix(n_grid, dx, fd_order, k_point, a)
    return -0.5 * L


# ============================================================
# 稀疏矩阵版本 (大规模计算)
# ============================================================

def build_laplacian_sparse(n_grid: int, dx: float, fd_order: int,
                            k_point: float = 0.0,
                            a: float = 1.0) -> sparse.csc_matrix:
    """
    构建稀疏格式的 Laplacian 矩阵。

    对于大规模网格 (N > 1000), 稀疏格式可显著节省内存。
    使用 CSR/CSC 格式存储带状结构。

    Parameters
    ----------
    n_grid : int
        网格点数
    dx : float
        网格间距
    fd_order : int
        FD 半带宽
    k_point : float
        k 点
    a : float
        原胞长度

    Returns
    -------
    L_sparse : scipy.sparse.csc_matrix
        稀疏 Laplacian 矩阵
    """
    c = fd_coefficients_2nd(fd_order)
    p = fd_order
    phase = np.exp(1j * k_point * a)
    phase_conj = np.conj(phase)

    rows = []
    cols = []
    vals = []

    for i in range(n_grid):
        for j_idx in range(-p, p + 1):
            j_phys = i + j_idx
            coeff = c[p + j_idx] / (dx ** 2)

            if 0 <= j_phys < n_grid:
                rows.append(i)
                cols.append(j_phys)
                vals.append(coeff)
            elif j_phys >= n_grid:
                j_wrapped = j_phys - n_grid
                rows.append(i)
                cols.append(j_wrapped)
                vals.append(coeff * phase)
            else:
                j_wrapped = j_phys + n_grid
                rows.append(i)
                cols.append(j_wrapped)
                vals.append(coeff * phase_conj)

    L_sparse = sparse.csc_matrix(
        (vals, (rows, cols)), shape=(n_grid, n_grid))
    return L_sparse


# ============================================================
# Vandermonde 矩阵与谱微分 (源自 271_dg1d_advection)
# ============================================================

def vandermonde_1d(n_points: int, r: np.ndarray) -> np.ndarray:
    """
    构建 1D Vandermonde 矩阵 (源自 271_dg1d_advection 的 Vandermonde1D)。

    对于 Legendre 多项式基 {P_0, P_1, ..., P_N}:
      V[i,j] = P_j(r_i)

    其中 P_j 为 j 阶 Legendre 多项式, r_i 为节点坐标。

    Vandermonde 矩阵用于:
    1. 将 Lagrange 插值系数转换为 modal 系数
    2. 构建微分矩阵 D = V_r · V^{-1}
    3. 构建 lifting 算子 LIFT = V·V^T·E

    Parameters
    ----------
    n_points : int
        多项式阶数 N (使用 N+1 个基函数)
    r : np.ndarray, shape (n_nodes,)
        节点坐标 (在 [-1, 1] 上)

    Returns
    -------
    V : np.ndarray, shape (n_nodes, N+1)
        Vandermonde 矩阵
    """
    N = n_points
    r = np.asarray(r)
    V = np.zeros((len(r), N + 1))

    # Legendre 多项式递推:
    # (j+1) P_{j+1}(r) = (2j+1) r P_j(r) - j P_{j-1}(r)
    V[:, 0] = 1.0
    if N >= 1:
        V[:, 1] = r
    for j in range(1, N):
        V[:, j + 1] = ((2 * j + 1) * r * V[:, j] -
                        j * V[:, j - 1]) / (j + 1)

    # 正交归一化: P_j → √((2j+1)/2) · P_j
    for j in range(N + 1):
        V[:, j] *= np.sqrt((2.0 * j + 1.0) / 2.0)

    return V


def differentiation_matrix_1d(n_order: int,
                                r: np.ndarray) -> np.ndarray:
    """
    构建 1D 谱微分矩阵 (源自 271_dg1d_advection 的 Dmatrix1D)。

    D = V_r · V^{-1}

    其中 V_r 为 Vandermonde 矩阵的径向导数。

    对于 Gauss-Lobatto 节点, D 矩阵的精度为 O(h^N)。

    Parameters
    ----------
    n_order : int
        多项式阶数
    r : np.ndarray
        节点坐标

    Returns
    -------
    D : np.ndarray, shape (n_nodes, n_nodes)
        微分矩阵
    """
    V = vandermonde_1d(n_order, r)

    # V_r: Vandermonde 矩阵的导数
    V_r = np.zeros_like(V)
    if n_order >= 1:
        V_r[:, 1] = 0.0
    for j in range(1, n_order + 1):
        # dP_j/dr 的递推
        V_r[:, j] = (np.sqrt(2.0 * j + 1.0) *
                      (r * V[:, j - 1] * np.sqrt(2.0 * j - 1.0) / np.sqrt(2.0 * j + 1.0)
                       + V[:, j - 1] * np.sqrt(2.0 * j - 1.0) / np.sqrt(2.0 * j + 1.0)))

    # 使用更稳定的方法: 直接计算
    # D[i,j] = Σ_k (V_r)_{ik} (V^{-1})_{kj}
    try:
        V_inv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        V_inv = np.linalg.pinv(V)

    D = V_r @ V_inv
    return D


def gauss_lobatto_nodes(n_points: int) -> np.ndarray:
    """
    计算 Gauss-Lobatto 节点 (源自 271_dg1d_advection 的 JacobiGL)。

    Gauss-Lobatto 节点是以下方程的根:
      (1 - r²) P'_N(r) = 0

    即 r = ±1 加上 P'_N(r) 的 N-1 个根。

    这些节点用于 DG 方法中的谱元离散。

    Parameters
    ----------
    n_points : int
        节点数 N+1 (多项式阶数 N)

    Returns
    -------
    r : np.ndarray, shape (N+1,)
        Gauss-Lobatto 节点, 在 [-1, 1] 上
    """
    if n_points < 2:
        raise ValueError(f"至少需要 2 个 GL 节点, 得到 {n_points}")

    N = n_points - 1
    r = np.zeros(n_points)
    r[0] = -1.0
    r[-1] = 1.0

    if N == 1:
        return r

    # 内部节点为 P'_N(r) 的根
    # 使用 Chebyshev 节点作为初始猜测, 然后 Newton 迭代
    for i in range(1, N):
        r[i] = -np.cos(PI * i / N)

    # Newton 迭代精化
    for _ in range(20):
        P, dP = _legendre_and_derivative(N, r[1:-1])
        # P'_N(r) = 0 的 Newton 步
        d2P = _legendre_second_derivative(N, r[1:-1])
        dr = -dP / d2P
        r[1:-1] += dr
        if np.max(np.abs(dr)) < 1e-15:
            break

    return np.sort(r)


def _legendre_and_derivative(N: int, r: np.ndarray
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """Legendre 多项式 P_N(r) 及其导数 P'_N(r)"""
    P = np.zeros_like(r)
    dP = np.zeros_like(r)

    if N == 0:
        P[:] = 1.0
        return P, dP

    P0 = np.ones_like(r)
    P1 = r.copy()
    P = P1.copy()

    for j in range(1, N):
        P2 = ((2 * j + 1) * r * P1 - j * P0) / (j + 1)
        P0 = P1.copy()
        P1 = P2.copy()

    P = P1.copy()

    # 导数: P'_N(r) = N(r P_N - P_{N-1}) / (r² - 1)
    # 或用递推: P'_N = (N+1)(P_{N-1} - r P_N)/(1-r²)  (对 |r|<1)
    mask = np.abs(r) < 1.0 - 1e-14
    if np.any(mask):
        dP[mask] = N * (P0[mask] - r[mask] * P[mask]) / (1.0 - r[mask] ** 2)

    return P, dP


def _legendre_second_derivative(N: int, r: np.ndarray) -> np.ndarray:
    """Legendre 多项式二阶导数 P''_N(r)"""
    P, dP = _legendre_and_derivative(N, r)
    # P''_N = (2r P'_N - N(N+1) P_N) / (1 - r²)
    mask = np.abs(r) < 1.0 - 1e-14
    d2P = np.zeros_like(r)
    if np.any(mask):
        d2P[mask] = (2.0 * r[mask] * dP[mask] -
                     N * (N + 1) * P[mask]) / (1.0 - r[mask] ** 2)
    return d2P


# ============================================================
# FD 色散关系分析
# ============================================================

def compute_fd_dispersion(fd_order: int, n_samples: int = 1000
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算 FD 算子的色散关系。

    对于平面波 ψ = e^{ikx}:
      (-d²/dx²) e^{ikx} = k² e^{ikx}
      (-Δ_FD) e^{ikx} = k²_mod(k) e^{ikx}

    k²_mod = -(1/dx²) Σ_{j=-p}^{p} c_j e^{ij·kdx}
           = -(1/dx²) [c_0 + 2 Σ_{j=1}^{p} c_j cos(j·kdx)]

    Parameters
    ----------
    fd_order : int
        FD 半带宽
    n_samples : int
        采样点数

    Returns
    -------
    kdx : np.ndarray
        k·dx 值, 在 [0, π] 上
    k2_mod : np.ndarray
        k²_mod · dx² (修正波数平方)
    k2_exact : np.ndarray
        (k·dx)² (精确波数平方)
    """
    kdx = np.linspace(0, PI, n_samples)
    c = fd_coefficients_2nd(fd_order)
    p = fd_order

    k2_mod = np.zeros_like(kdx)
    for j in range(-p, p + 1):
        k2_mod += -c[p + j] * np.cos(j * kdx)

    k2_exact = kdx ** 2

    return kdx, k2_mod, k2_exact


def fd_stability_limit(fd_order: int) -> float:
    """
    计算 FD 算子的稳定性极限 (最大可解析波数)。

    定义: 当 k²_mod(k_max) 偏离 k² 不超过 1% 时的最大 kdx。

    对于 2p 阶 FD:
      kdx_max ≈ π · (1 - C/p)  (近似)
    其中 C ~ 0.1-0.3 取决于 p。

    物理意义: 在能带计算中, 需要 dx 足够小使得
    最高能带的波数也在 FD 的精确范围内。

    Parameters
    ----------
    fd_order : int
        FD 半带宽

    Returns
    -------
    kdx_max : float
        最大可解析 kdx (精度 < 1%)
    """
    kdx, k2_mod, k2_exact = compute_fd_dispersion(fd_order, 10000)

    # 找到相对误差超过 1% 的第一个点
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error = np.abs(k2_mod - k2_exact) / np.maximum(k2_exact, 1e-30)

    # 忽略 kdx=0 附近
    valid = kdx > 0.01
    if not np.any(valid):
        return PI

    error_valid = rel_error[valid]
    kdx_valid = kdx[valid]

    above_threshold = error_valid > 0.01
    if not np.any(above_threshold):
        return PI

    idx_first = np.argmax(above_threshold)
    return kdx_valid[idx_first]
