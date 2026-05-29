"""
滞弹性压力方程稀疏矩阵求解器 (Anelastic Pressure Solver)

集成种子项目:
- 998_r8st: 稀疏矩阵 COO 格式存储与共轭梯度 (CG) 求解器

科学背景:
  中尺度对流系统的滞弹性近似下, 压力场满足椭圆方程:
    ∇·(ρ₀ ∇φ) = ∇·(ρ₀ B k̂) + 2 * J(u,v)
  其中 B 为浮力, J(u,v) 为水平变形项.

  在离散形式下, 这转化为大型稀疏线性系统:
    A p = b
  其中 A 为对称正定 (SPD) 稀疏矩阵, 使用 CG 迭代求解.

核心公式 (离散化):
  对 2D 域 (x-z 剖面), 使用二阶中心差分:
    ∂/∂x (ρ ∂φ/∂x) ≈ [ρ_{i+1/2}(φ_{i+1}-φ_i) - ρ_{i-1/2}(φ_i-φ_{i-1})] / dx²
    ∂/∂z (ρ ∂φ/∂z) ≈ [ρ_{j+1/2}(φ_{j+1}-φ_j) - ρ_{j-1/2}(φ_j-φ_{j-1})] / dz²
"""

import numpy as np
from typing import Tuple, List


class SparseMatrixCOO:
    """
    稀疏矩阵 COO (Coordinate / Triplet) 格式 (基于 998_r8st).
    """

    def __init__(self, nrow: int, ncol: int, nnz: int = 0):
        self.nrow = nrow
        self.ncol = ncol
        self.row = []
        self.col = []
        self.val = []
        self._nnz_max = nnz if nnz > 0 else nrow * ncol

    def append(self, i: int, j: int, v: float):
        """添加一个非零元."""
        if abs(v) > 1e-20:
            self.row.append(i)
            self.col.append(j)
            self.val.append(v)

    def to_dense(self) -> np.ndarray:
        """转为稠密矩阵 (仅用于小规模验证)."""
        A = np.zeros((self.nrow, self.ncol))
        for i, j, v in zip(self.row, self.col, self.val):
            A[i, j] += v
        return A

    def mv(self, x: np.ndarray) -> np.ndarray:
        """
        稀疏矩阵-向量乘法 y = A @ x, O(nnz) 复杂度.
        """
        y = np.zeros(self.nrow, dtype=float)
        for i, j, v in zip(self.row, self.col, self.val):
            y[i] += v * x[j]
        return y

    def check(self) -> bool:
        """边界检查."""
        for i, j in zip(self.row, self.col):
            if i < 0 or i >= self.nrow or j < 0 or j >= self.ncol:
                return False
        return True


def build_anelastic_laplacian_2d(nx: int, nz: int, dx: float, dz: float,
                                 rho: np.ndarray) -> SparseMatrixCOO:
    """
    构建 2D x-z 剖面上滞弹性压力方程的离散 Laplacian 算子矩阵.

    方程: ∂/∂x(ρ ∂p/∂x) + ∂/∂z(ρ ∂p/∂z) = rhs

    使用交错网格上的密度 ρ_{i+1/2} 近似为 0.5*(ρ_i + ρ_{i+1}).
    边界条件: Neumann (零法向梯度) 或 Dirichlet (固定值).
    这里侧边界用 Dirichlet, 上下边界用 Neumann.
    """
    n = nx * nz
    A = SparseMatrixCOO(n, n, nnz=n * 5)

    def idx(i, j):
        return j * nx + i

    dx2 = dx * dx
    dz2 = dz * dz

    for j in range(nz):
        for i in range(nx):
            k = idx(i, j)
            # 密度在界面上的值
            rho_e = 0.5 * (rho[j, min(i+1, nx-1)] + rho[j, i])
            rho_w = 0.5 * (rho[j, max(i-1, 0)] + rho[j, i])
            rho_n = 0.5 * (rho[min(j+1, nz-1), i] + rho[j, i])
            rho_s = 0.5 * (rho[max(j-1, 0), i] + rho[j, i])

            coeff = 0.0
            # x 方向
            if i > 0:
                A.append(k, idx(i-1, j), rho_w / dx2)
                coeff -= rho_w / dx2
            if i < nx - 1:
                A.append(k, idx(i+1, j), rho_e / dx2)
                coeff -= rho_e / dx2
            # z 方向
            if j > 0:
                A.append(k, idx(i, j-1), rho_s / dz2)
                coeff -= rho_s / dz2
            if j < nz - 1:
                A.append(k, idx(i, j+1), rho_n / dz2)
                coeff -= rho_n / dz2

            # 对角元
            A.append(k, k, coeff)

    return A


def conjugate_gradient(A: SparseMatrixCOO, b: np.ndarray, x0: np.ndarray = None,
                       tol: float = 1e-8, max_iter: int = None) -> Tuple[np.ndarray, int, float]:
    """
    共轭梯度法求解 Ax = b (基于 998_r8st/r8st_cg).

    要求 A 对称正定 (SPD).
    返回 (x, iter_count, residual_norm).
    """
    n = A.nrow
    if max_iter is None:
        max_iter = n
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    r = b - A.mv(x)
    p = r.copy()
    rs_old = np.dot(r, r)

    # 边界处理: 零初始猜测的初残差
    if rs_old < 1e-30:
        return x, 0, 0.0

    for k in range(max_iter):
        Ap = A.mv(p)
        pAp = np.dot(p, Ap)
        if abs(pAp) < 1e-30:
            break
        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol * (np.linalg.norm(b) + 1.0):
            return x, k + 1, np.sqrt(rs_new)
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x, max_iter, np.sqrt(rs_old)


def jacobi_iteration(A: SparseMatrixCOO, b: np.ndarray, x0: np.ndarray = None,
                     tol: float = 1e-8, max_iter: int = 1000) -> np.ndarray:
    """
    Jacobi 迭代 (基于 998_r8st/r8st_jac_sl), 作为 CG 的后备方案.
    """
    n = A.nrow
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    # 提取对角元
    diag = np.zeros(n)
    for i, j, v in zip(A.row, A.col, A.val):
        if i == j:
            diag[i] = v
    # 保护零对角
    diag = np.where(np.abs(diag) < 1e-20, 1.0, diag)

    for _ in range(max_iter):
        x_new = (b - A.mv(x) + diag * x) / diag
        if np.linalg.norm(x_new - x) < tol * (np.linalg.norm(x_new) + 1.0):
            return x_new
        x = x_new
    return x


def solve_anelastic_pressure(nx: int, nz: int, dx: float, dz: float,
                             rho: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    求解滞弹性压力方程并返回压力场 p(x,z).

    输入:
      nx, nz: 格点数
      dx, dz: 格距
      rho: 基态密度场 (nz, nx)
      rhs: 右端项 (nz, nx)
    输出:
      p: 压力扰动场 (nz, nx)
    """
    A = build_anelastic_laplacian_2d(nx, nz, dx, dz, rho)
    b = rhs.flatten()

    # 均值移除使系统相容 (Neumann 边界导致奇异)
    b -= np.mean(b)

    x0 = np.zeros_like(b)
    x, iters, res = conjugate_gradient(A, b, x0, tol=1e-10, max_iter=min(5000, A.nrow))

    # 若 CG 不收敛, 回退到 Jacobi
    if res > 1e-4:
        x = jacobi_iteration(A, b, x, tol=1e-8, max_iter=2000)

    p = x.reshape((nz, nx))
    # 零均值规范化
    p -= np.mean(p)
    return p
