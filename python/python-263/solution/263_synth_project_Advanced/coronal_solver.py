# -*- coding: utf-8 -*-
"""
coronal_solver.py
-----------------
隐式 MHD 求解器: 重启 GMRES (760_mgmres) + 下三角 LU (984_r8lt).

物理需求
--------
日冕 MHD 含强各向异性热传导 (kappa_|| / kappa_perp ~ 1e10), 显式格式
CFL 时步 dt < h^2 / (2 kappa) ~ 1e-6 s, 不可接受. 采用隐式-显式 (IMEX)
分裂:
    显式: 对流项、Lorentz 力、重力 (CFL 受 Alfvén 速度约束)
    隐式: 热传导项 (parabolic stiffness)

隐式子步求解线性系统:
    (I - dt theta L) T^{n+1} = (I + dt (1-theta) L) T^n + dt S
其中 L = d/ds kappa d/ds 为椭圆算子.

GMRES 用于大型稀疏系统, 下三角 LU 用于预处理或小型子系统.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import gmres as scipy_gmres


# ============================================================
# 稀疏矩阵向量乘 (CRS 风格, 致敬 760_mgmres)
# ============================================================
def sparse_matvec(A: sparse.csr_matrix, x: np.ndarray) -> np.ndarray:
    """CRS 格式稀疏矩阵-向量乘."""
    return A.dot(x)


# ============================================================
# 重启 GMRES (核心算法来自 760_mgmres)
# ============================================================
def restarted_gmres(A: sparse.csr_matrix, b: np.ndarray, x0: np.ndarray,
                    mr: int = 30, tol_abs: float = 1.0e-10,
                    tol_rel: float = 1.0e-8, itr_max: int = 100,
                    precond: sparse.csr_matrix | None = None):
    """重启 GMRES 求解 A x = b.

    Parameters
    ----------
    A : sparse csr matrix
    b : rhs
    x0 : initial guess
    mr : restart dimension (Krylov subspace size)
    tol_abs, tol_rel : 绝对/相对容差
    itr_max : 最大重启次数
    precond : 预处理矩阵 M (解 M z = r 代替 r)

    Returns
    -------
    x : solution
    info : dict with 'converged', 'residual', 'iterations'
    """
    n = A.shape[0]
    x = x0.copy()
    r = b - A.dot(x)
    if precond is not None:
        r = precond.dot(r)
    beta = np.linalg.norm(r)
    if beta < tol_abs:
        return x, dict(converged=True, residual=float(beta), iterations=0)

    tol = max(tol_abs, tol_rel * beta)
    total_iter = 0

    for restart in range(itr_max):
        # Arnoldi 过程
        V = np.zeros((n, mr + 1))
        H = np.zeros((mr + 1, mr))
        V[:, 0] = r / beta
        cs = np.zeros(mr)
        sn = np.zeros(mr)
        e1 = np.zeros(mr + 1)
        e1[0] = beta

        breakdown = False
        j_final = 0
        for j in range(mr):
            w = A.dot(V[:, j])
            if precond is not None:
                w = precond.dot(w)
            # Arnoldi 正交化 (modified Gram-Schmidt)
            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i], w)
                w = w - H[i, j] * V[:, i]
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] < 1.0e-14:
                breakdown = True
                j_final = j + 1
                break
            V[:, j + 1] = w / H[j + 1, j]

            # 应用之前的 Givens 旋转
            for i in range(j):
                temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = temp

            # 计算新 Givens 旋转
            denom = np.sqrt(H[j, j]**2 + H[j + 1, j]**2)
            if denom < 1.0e-14:
                cs[j] = 1.0
                sn[j] = 0.0
            else:
                cs[j] = H[j, j] / denom
                sn[j] = H[j + 1, j] / denom

            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0

            e1[j + 1] = -sn[j] * e1[j]
            e1[j] = cs[j] * e1[j]

            j_final = j + 1
            if abs(e1[j + 1]) < tol:
                break

        # 回代 H y = e1
        y = np.linalg.lstsq(H[:j_final, :j_final], e1[:j_final], rcond=None)[0]
        x = x + V[:, :j_final].dot(y)

        total_iter += j_final
        r = b - A.dot(x)
        if precond is not None:
            r = precond.dot(r)
        beta = np.linalg.norm(r)
        if beta < tol:
            return x, dict(converged=True, residual=float(beta),
                          iterations=total_iter)
        if breakdown:
            break

    return x, dict(converged=False, residual=float(beta),
                  iterations=total_iter)


# ============================================================
# 下三角矩阵工具 (来自 984_r8lt)
# ============================================================
class R8LT:
    """Lower-Triangular matrix utilities (adapted to numpy).

    主要用于隐式时间步中的 ILU 预处理子.
    """

    def __init__(self, diag: np.ndarray, subdiag: np.ndarray):
        """n x n 下三角矩阵: diag (n,), subdiag (n-1)."""
        self.diag = diag.copy()
        self.subdiag = subdiag.copy()
        self.n = diag.size

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """L x."""
        y = self.diag * x
        y[1:] += self.subdiag * x[:-1]
        return y

    def solve(self, b: np.ndarray) -> np.ndarray:
        """前代法解 L x = b."""
        x = np.zeros_like(b)
        x[0] = b[0] / (self.diag[0] + 1.0e-30)
        for i in range(1, self.n):
            x[i] = (b[i] - self.subdiag[i - 1] * x[i - 1]) / (
                self.diag[i] + 1.0e-30
            )
        return x

    def determinant(self) -> float:
        """det(L) = prod(diag)."""
        return float(np.prod(self.diag))

    def to_dense(self) -> np.ndarray:
        L = np.diag(self.diag)
        for i in range(self.n - 1):
            L[i + 1, i] = self.subdiag[i]
        return L


# ============================================================
# ILU(0) 预处理子构造
# ============================================================
def ilu0_preconditioner(A: sparse.csr_matrix) -> sparse.csr_matrix:
    """ILU(0) 预处理: 保持稀疏结构, 近似分解 A ~ L U.
    返回 M^{-1} 近似 (以 csr 形式)."""
    try:
        from scipy.sparse.linalg import spilu
        ilu = spilu(A.tocsc(), drop_tol=1.0e-4)
        n = A.shape[0]
        e = np.ones(n)
        # 构造 M^{-1} 近似为 LinearOperator 转 dense (小规模问题)
        Minv_dense = np.zeros((n, n))
        for j in range(n):
            ej = np.zeros(n)
            ej[j] = 1.0
            Minv_dense[:, j] = ilu.solve(ej)
        return sparse.csr_matrix(Minv_dense)
    except Exception:
        return sparse.eye(A.shape[0], format="csr")


# ============================================================
# 隐式热传导步
# ============================================================
def implicit_heat_step(T: np.ndarray, dt: float, theta: float,
                       z_grid: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Crank-Nicolson (theta=0.5) 或全隐式 (theta=1) 热传导步."""
    from fd_operators import diffusion_matrix
    L = diffusion_matrix(z_grid, kappa)
    n = z_grid.size
    I = sparse.eye(n, format="csr")
    A_lhs = I - dt * theta * L
    b_rhs = (I + dt * (1.0 - theta) * L).dot(T)
    # 固定边界
    A_lhs_dense = A_lhs.toarray()
    A_lhs_dense[0, :] = 0.0
    A_lhs_dense[0, 0] = 1.0
    A_lhs_dense[-1, :] = 0.0
    A_lhs_dense[-1, -1] = 1.0
    b_rhs[0] = T[0]
    b_rhs[-1] = T[-1]

    x0 = T.copy()
    A_sparse = sparse.csr_matrix(A_lhs_dense)
    # 使用 GMRES
    sol, info = restarted_gmres(A_sparse, b_rhs, x0, mr=20,
                               tol_abs=1.0e-8, tol_rel=1.0e-6,
                               itr_max=20)
    return sol
