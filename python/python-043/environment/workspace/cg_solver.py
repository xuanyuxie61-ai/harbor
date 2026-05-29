"""
共轭梯度求解器 (cg_solver.py)
==============================
基于种子项目 152_cg_rc 的 reverse-communication 共轭梯度思想，
实现用于地核发电机大型稀疏线性系统求解的 CG 与预条件 CG。

地核发电机模拟中，每个时间步需多次求解泊松型方程：
    (I - dt*eta*Laplacian) * x = b
其中 Laplacian 在径向方向经离散化后形成对称正定三对角/带状矩阵。

本模块提供：
  - 标准 CG 求解器
  - 不完全 Cholesky (IC) 预条件 CG
  - 适用于地核径向扩散算子的专用求解器
"""

import numpy as np
from typing import Callable, Optional, Tuple


def conjugate_gradient(A_matvec: Callable[[np.ndarray], np.ndarray],
                       b: np.ndarray,
                       x0: Optional[np.ndarray] = None,
                       tol: float = 1e-10,
                       maxiter: int = 2000) -> Tuple[np.ndarray, int, float]:
    """
    标准共轭梯度法求解 A x = b。

    参数:
      A_matvec : 矩阵-向量乘积函数 y = A @ x
      b        : 右端项
      x0       : 初始猜测
      tol      : 相对残差收敛容限
      maxiter  : 最大迭代次数

    返回:
      x       : 近似解
      iters   : 实际迭代次数
      resid   : 最终相对残差 ||r||/||b||

    算法（Fletcher-Reeves 风格）:
      r_0 = b - A x_0
      p_0 = r_0
      for k = 0, 1, ...:
          alpha_k = (r_k^T r_k) / (p_k^T A p_k)
          x_{k+1} = x_k + alpha_k p_k
          r_{k+1} = r_k - alpha_k A p_k
          beta_k  = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
          p_{k+1} = r_{k+1} + beta_k p_k
    """
    b = np.asarray(b, dtype=float)
    n = b.size
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()

    r = b - A_matvec(x)
    p = r.copy()
    rsold = float(np.dot(r, r))
    norm_b = float(np.linalg.norm(b))
    if norm_b < 1e-30:
        norm_b = 1.0

    for k in range(maxiter):
        Ap = A_matvec(p)
        pAp = float(np.dot(p, Ap))
        if abs(pAp) < 1e-30:
            break
        alpha = rsold / pAp
        x += alpha * p
        r -= alpha * Ap
        rsnew = float(np.dot(r, r))
        resid = np.sqrt(rsnew) / norm_b
        if resid < tol:
            return x, k + 1, resid
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

    return x, maxiter, np.sqrt(float(np.dot(r, r))) / norm_b


def incomplete_cholesky_prec(A: np.ndarray, drop_tol: float = 1e-12) -> np.ndarray:
    """
    计算不完全 Cholesky 预条件子 M = L L^T 的近似，
    返回 L（下三角矩阵）。仅保留原矩阵非零模式内的填充。

    参数:
      A        : 对称正定稀疏矩阵（稠密格式，小系统用）
      drop_tol : 丢弃小元素的阈值

    算法（IC(0) 变体）:
      for j = 0..n-1:
          L[j,j] = sqrt(A[j,j] - sum_{k<j} L[j,k]^2)
          for i = j+1..n-1:
              if A[i,j] != 0:
                  L[i,j] = (A[i,j] - sum_{k<j} L[i,k] L[j,k]) / L[j,j]
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    L = np.zeros((n, n), dtype=float)
    for j in range(n):
        diag_val = A[j, j] - np.dot(L[j, :j], L[j, :j])
        if diag_val <= 0.0:
            diag_val = abs(diag_val) + 1e-10
        L[j, j] = np.sqrt(diag_val)
        for i in range(j + 1, n):
            if abs(A[i, j]) > drop_tol:
                offdiag = A[i, j] - np.dot(L[i, :j], L[j, :j])
                L[i, j] = offdiag / L[j, j]
    return L


def pcg_solver(A_matvec: Callable[[np.ndarray], np.ndarray],
               b: np.ndarray,
               preconditioner_solve: Callable[[np.ndarray], np.ndarray],
               x0: Optional[np.ndarray] = None,
               tol: float = 1e-10,
               maxiter: int = 2000) -> Tuple[np.ndarray, int, float]:
    """
    预条件共轭梯度法 (PCG)。

    参数:
      preconditioner_solve : 函数 z = M^{-1} r，其中 M 是预条件矩阵
      其余同 conjugate_gradient。

    算法:
      r_0 = b - A x_0
      z_0 = M^{-1} r_0
      p_0 = z_0
      for k = 0, 1, ...:
          alpha_k = (r_k^T z_k) / (p_k^T A p_k)
          x_{k+1} = x_k + alpha_k p_k
          r_{k+1} = r_k - alpha_k A p_k
          z_{k+1} = M^{-1} r_{k+1}
          beta_k  = (r_{k+1}^T z_{k+1}) / (r_k^T z_k)
          p_{k+1} = z_{k+1} + beta_k p_k
    """
    b = np.asarray(b, dtype=float)
    n = b.size
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()

    r = b - A_matvec(x)
    z = preconditioner_solve(r)
    p = z.copy()
    rzold = float(np.dot(r, z))
    norm_b = float(np.linalg.norm(b))
    if norm_b < 1e-30:
        norm_b = 1.0

    for k in range(maxiter):
        Ap = A_matvec(p)
        pAp = float(np.dot(p, Ap))
        if abs(pAp) < 1e-30:
            break
        alpha = rzold / pAp
        x += alpha * p
        r -= alpha * Ap
        resid = float(np.linalg.norm(r)) / norm_b
        if resid < tol:
            return x, k + 1, resid
        z = preconditioner_solve(r)
        rznew = float(np.dot(r, z))
        beta = rznew / rzold
        p = z + beta * p
        rzold = rznew

    return x, maxiter, float(np.linalg.norm(r)) / norm_b


# ---------------------------------------------------------------------------
# 地核径向扩散算子专用求解器
#    算子: (I - theta * dt * eta * d^2/dr^2) ，带有变系数和边界条件
# ---------------------------------------------------------------------------
def build_radial_diffusion_operator(n: int, dr: float, dt: float, eta: float,
                                     theta_cn: float = 0.5) -> np.ndarray:
    """
    构建径向扩散算子的 Crank-Nicolson 离散矩阵：
        A = I - theta * dt * eta * L_r
    其中 L_r 为二阶中心差分 Laplacian（一维径向近似）。

    返回稠密矩阵 A (n x n)。
    """
    A = np.zeros((n, n), dtype=float)
    coeff = theta_cn * dt * eta / (dr * dr)
    for i in range(n):
        A[i, i] = 1.0 + 2.0 * coeff
        if i > 0:
            A[i, i - 1] = -coeff
        if i < n - 1:
            A[i, i + 1] = -coeff
    # 边界条件：Dirichlet (值固定)
    A[0, 0] = 1.0
    A[0, 1] = 0.0
    A[n - 1, n - 1] = 1.0
    A[n - 1, n - 2] = 0.0
    return A


def solve_radial_diffusion_cg(rhs: np.ndarray, n: int, dr: float, dt: float,
                               eta: float, theta_cn: float = 0.5,
                               tol: float = 1e-10, maxiter: int = 2000) -> np.ndarray:
    """
    使用 PCG 求解径向扩散方程的离散系统 A x = rhs。
    """
    A = build_radial_diffusion_operator(n, dr, dt, eta, theta_cn)

    def matvec(v):
        return A @ v

    # 使用对角预条件
    diag = np.diag(A).copy()
    diag[diag == 0.0] = 1.0

    def prec_solve(r):
        return r / diag

    x, iters, resid = pcg_solver(matvec, rhs, prec_solve, tol=tol, maxiter=maxiter)
    return x


# ---------------------------------------------------------------------------
# 自测试
# ---------------------------------------------------------------------------
def _self_test():
    # 测试 1：简单对角系统
    n = 50
    A = np.diag(np.arange(1, n + 1, dtype=float))
    b = np.ones(n, dtype=float)
    x, iters, resid = conjugate_gradient(lambda v: A @ v, b, tol=1e-12)
    assert resid < 1e-10
    x_exact = 1.0 / np.arange(1, n + 1, dtype=float)
    assert np.linalg.norm(x - x_exact) < 1e-8

    # 测试 2：径向扩散
    n = 32
    dr = 1.0 / (n - 1)
    dt = 0.01
    eta = 1.0
    rhs = np.ones(n, dtype=float)
    rhs[0] = 0.0
    rhs[-1] = 0.0
    x = solve_radial_diffusion_cg(rhs, n, dr, dt, eta)
    assert not np.isnan(x).any()
    assert not np.isinf(x).any()

    print(f"cg_solver: CG converged in {iters} iterations, resid={resid:.4e}")
    print("cg_solver: self-test passed.")


if __name__ == "__main__":
    _self_test()
