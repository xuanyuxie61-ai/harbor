"""
sparse_matrix_utils.py
======================
基于 ge_to_crs (458_ge_to_crs) 与 ge_to_st (459_ge_to_st) 的稀疏矩阵格式转换，
为海气耦合模式中的大型稀疏线性系统提供高效存储与运算支持。

科学背景
--------
海洋环流数值模式（如有限元/有限差分离散）产生的系统矩阵通常是稀疏的，
维度可达 10^6 × 10^6，但非零元仅占 O(10^{-4}) 比例。压缩稀疏行 (CRS) 格式
与稀疏三元组 (ST) 格式是海洋模式中求解 Poisson/Helmholtz 方程的标准存储方式。

核心公式
--------
1. 海洋 Sverdrup 平衡的离散化：
   
   β * ∂ψ/∂x = curl(τ) / (ρ_0 * H)

   在经度-纬度网格上离散后，得到稀疏线性系统 A * ψ = b，
   其中 A 的每一行仅有 3–5 个非零元（中心差分格式）。

2. CRS 格式：
   对于 n×n 矩阵，存储三个数组：
   - val[nz]    : 非零元值
   - col[nz]    : 非零元的列索引
   - row_ptr[n+1] : 第 i 行的非零元在 val/col 中的起始位置

   第 i 行的非零元为 val[row_ptr[i] : row_ptr[i+1]]，
   对应列索引为 col[row_ptr[i] : row_ptr[i+1]]。

3. ST (Sparse Triplet) 格式：
   直接存储非零元的 (row, col, value) 三元组列表。

4. 热含量扩散方程的隐式离散：
   
   (I - Δt * D * ∇²) T^{n+1} = T^n + Δt * Q

   其中 ∇² 的离散矩阵是稀疏对称正定的，适合 CRS 存储。
"""

import numpy as np
from typing import Tuple, List


def ge_to_crs(A: np.ndarray) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    将稠密一般矩阵 (GE) 转换为压缩稀疏行 (CRS) 格式。

    参数
    ----
    A : np.ndarray, shape (n, n)
        输入稠密矩阵。

    返回
    ----
    n : int
        矩阵阶数。
    nz : int
        非零元个数。
    row_ptr : np.ndarray, shape (n+1,)
        行指针数组。
    col : np.ndarray, shape (nz,)
        列索引数组。
    val : np.ndarray, shape (nz,)
        非零元值数组。
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D array")

    n = A.shape[0]
    nz = np.count_nonzero(A)
    row_ptr = np.zeros(n + 1, dtype=int)
    col = np.zeros(nz, dtype=int)
    val = np.zeros(nz, dtype=float)

    row_ptr[0] = 0
    k = 0
    for i in range(n):
        for j in range(n):
            if A[i, j] != 0.0:
                col[k] = j
                val[k] = A[i, j]
                k += 1
        row_ptr[i + 1] = k

    return n, nz, row_ptr, col, val


def ge_to_st(A: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    将稠密一般矩阵 (GE) 转换为稀疏三元组 (ST) 格式。

    参数
    ----
    A : np.ndarray, shape (m, n)
        输入稠密矩阵（可为非方阵）。

    返回
    ----
    nz : int
        非零元个数。
    ist : np.ndarray, shape (nz,)
        行索引。
    jst : np.ndarray, shape (nz,)
        列索引。
    Ast : np.ndarray, shape (nz,)
        非零元值。
    """
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")

    m, n = A.shape
    nz = np.count_nonzero(A)
    ist = np.zeros(nz, dtype=int)
    jst = np.zeros(nz, dtype=int)
    Ast = np.zeros(nz, dtype=float)

    k = 0
    for j in range(n):
        for i in range(m):
            if A[i, j] != 0.0:
                ist[k] = i
                jst[k] = j
                Ast[k] = A[i, j]
                k += 1

    return nz, ist, jst, Ast


def crs_matvec(n: int, row_ptr: np.ndarray, col: np.ndarray,
               val: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    CRS 格式稀疏矩阵与向量乘法 y = A @ x。

    参数
    ----
    n : int
        矩阵维度。
    row_ptr, col, val : np.ndarray
        CRS 格式数组。
    x : np.ndarray, shape (n,)
        输入向量。

    返回
    ----
    y : np.ndarray, shape (n,)
        结果向量。
    """
    if x.shape[0] != n:
        raise ValueError("Dimension mismatch between matrix and vector")
    y = np.zeros(n, dtype=float)
    for i in range(n):
        for idx in range(row_ptr[i], row_ptr[i + 1]):
            y[i] += val[idx] * x[col[idx]]
    return y


def build_laplacian_2d_crs(nx: int, ny: int, dx: float, dy: float,
                           boundary: str = "dirichlet") -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    构建二维五点差分 Laplacian 算子的 CRS 稀疏矩阵。

    离散公式（Dirichlet 边界）：
    ∇² T ≈ (T_{i-1,j} - 2T_{i,j} + T_{i+1,j}) / dx²
          + (T_{i,j-1} - 2T_{i,j} + T_{i,j+1}) / dy²

    参数
    ----
    nx, ny : int
        x, y 方向网格点数。
    dx, dy : float
        网格间距。
    boundary : str
        边界条件类型，"dirichlet" 或 "neumann"。

    返回
    ----
    n, nz, row_ptr, col, val : CRS 格式矩阵。
    """
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be at least 2")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive")

    n = nx * ny
    # 预估非零元：内部点5个，边界点3-4个
    nz_max = 5 * n
    row_ptr = np.zeros(n + 1, dtype=int)
    col = np.zeros(nz_max, dtype=int)
    val = np.zeros(nz_max, dtype=float)

    idx = 0
    for j in range(ny):
        for i in range(nx):
            row = j * nx + i
            start = idx

            cx = 1.0 / (dx * dx)
            cy = 1.0 / (dy * dy)

            # 中心点
            diag = 0.0

            # 左邻居
            if i > 0:
                col[idx] = row - 1
                val[idx] = cx
                idx += 1
                diag -= cx
            elif boundary == "dirichlet":
                diag -= cx  # 镜像点贡献

            # 右邻居
            if i < nx - 1:
                col[idx] = row + 1
                val[idx] = cx
                idx += 1
                diag -= cx
            elif boundary == "dirichlet":
                diag -= cx

            # 下邻居
            if j > 0:
                col[idx] = row - nx
                val[idx] = cy
                idx += 1
                diag -= cy
            elif boundary == "dirichlet":
                diag -= cy

            # 上邻居
            if j < ny - 1:
                col[idx] = row + nx
                val[idx] = cy
                idx += 1
                diag -= cy
            elif boundary == "dirichlet":
                diag -= cy

            # 对角元
            col[idx] = row
            val[idx] = diag
            idx += 1

            row_ptr[row] = start

    row_ptr[n] = idx
    nz = idx
    col = col[:nz]
    val = val[:nz]
    return n, nz, row_ptr, col, val


def build_sverdrup_matrix_crs(nx: int, ny: int, dx: float,
                              beta: float = 2.28e-11) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    构建 Sverdrup 平衡的离散稀疏矩阵。

    方程：β * ∂ψ/∂x = curl(τ) / (ρ_0 * H)
    采用一阶迎风格式离散 ∂ψ/∂x：
    
    ∂ψ/∂x ≈ (ψ_{i,j} - ψ_{i-1,j}) / dx   (若 β > 0)

    参数
    ----
    nx, ny : int
        网格维度。
    dx : float
        经度方向网格间距（米）。
    beta : float
        Coriolis 参数梯度，默认 2.28e-11 s^{-1}m^{-1}。

    返回
    ----
    CRS 格式矩阵。
    """
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be at least 2")
    if dx <= 0.0:
        raise ValueError("dx must be positive")

    n = nx * ny
    nz_max = 3 * n
    row_ptr = np.zeros(n + 1, dtype=int)
    col = np.zeros(nz_max, dtype=int)
    val = np.zeros(nz_max, dtype=float)

    idx = 0
    for j in range(ny):
        for i in range(nx):
            row = j * nx + i
            start = idx

            if i > 0:
                col[idx] = row - 1
                val[idx] = -beta / dx
                idx += 1

            col[idx] = row
            val[idx] = beta / dx
            idx += 1

            row_ptr[row] = start

    row_ptr[n] = idx
    nz = idx
    col = col[:nz]
    val = val[:nz]
    return n, nz, row_ptr, col, val


def crs_to_dense(n: int, row_ptr: np.ndarray, col: np.ndarray,
                 val: np.ndarray) -> np.ndarray:
    """
    将 CRS 格式矩阵转换回稠密矩阵（用于小规模验证）。
    """
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for k in range(row_ptr[i], row_ptr[i + 1]):
            A[i, col[k]] += val[k]
    return A
