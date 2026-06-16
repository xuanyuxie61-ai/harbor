"""
linear_solvers.py — 迭代线性方程组求解器
============================================
来源项目: 452_gauss_seidel_poisson_1d (Gauss-Seidel 迭代)

本模块实现多种迭代法求解线性系统 Ax = b:
  1. Gauss-Seidel 迭代 (来源: 452)
  2. SOR (逐次超松弛法)
  3. Jacobi 迭代
  4. 共轭梯度法 (CG)
  5. 预处理共轭梯度法 (PCG)

Gauss-Seidel 迭代 (来源: 452_gauss_seidel_poisson_1d):
  A = L + D + U (下三角 + 对角 + 上三角)
  (L + D) x^{k+1} = b - U x^k
  x_i^{k+1} = (b_i - Σ_{j<i} a_{ij} x_j^{k+1} - Σ_{j>i} a_{ij} x_j^k) / a_{ii}

收敛条件: A 严格对角占优 或 对称正定

SOR 方法:
  x_i^{k+1} = (1-ω) x_i^k + (ω/a_{ii}) (b_i - Σ_{j<i} a_{ij} x_j^{k+1} - Σ_{j>i} a_{ij} x_j^k)
  最优松弛参数: ω_opt = 2 / (1 + sqrt(1 - ρ(J)^2))
  其中 ρ(J) 为 Jacobi 迭代矩阵谱半径.

共轭梯度法 (对称正定系统):
  r_0 = b - Ax_0
  p_0 = r_0
  for k = 0, 1, ...
    α_k = (r_k^T r_k) / (p_k^T A p_k)
    x_{k+1} = x_k + α_k p_k
    r_{k+1} = r_k - α_k A p_k
    β_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
    p_{k+1} = r_{k+1} + β_k p_k
"""

import numpy as np
from typing import Tuple, Optional, Callable, List, Dict
from config import EPS_NUM


# ============================================================
#  Gauss-Seidel 迭代 (来源: 452_gauss_seidel_poisson_1d)
# ============================================================
def gauss_seidel(A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None,
                 max_iter: int = 1000, tol: float = 1e-8) -> dict:
    """Gauss-Seidel 迭代法求解 Ax = b

    算法 (来源: 452):
      分解 A = L + D + U
      对 i = 1, ..., n:
        x_i^{k+1} = (b_i - Σ_{j<i} a_{ij} x_j^{k+1} - Σ_{j>i} a_{ij} x_j^k) / a_{ii}

    收敛性:
      - 若 A 对称正定, 则 Gauss-Seidel 收敛
      - 若 A 严格对角占优, 则收敛
      - 谱半径 ρ(T_GS) < 1 是充要条件

    对于 1D Poisson 方程, Gauss-Seidel 的收敛速率:
      ρ(T_GS) = cos²(πh) ≈ 1 - π²h²  (h = 1/n)

    Args:
        A: 系数矩阵 (n, n)
        b: 右端向量 (n,)
        x0: 初始猜测 (默认为零向量)
        max_iter: 最大迭代次数
        tol: 收敛容差 (相对残差)

    Returns:
        dict: x (解), iterations, residual_history, converged
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    diag = np.diag(A).copy()
    if np.any(np.abs(diag) < EPS_NUM):
        raise ValueError("对角元素含零, Gauss-Seidel 不适用")

    residual_history = []
    b_norm = np.linalg.norm(b) + EPS_NUM

    for k in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j] if j < i else A[i, j] * x_old[j]
            x[i] = (b[i] - sigma) / diag[i]

        # 残差: r = b - Ax
        r = b - A @ x
        r_norm = np.linalg.norm(r)
        residual_history.append(r_norm)

        # 收敛判据
        if r_norm / b_norm < tol:
            return {
                'x': x,
                'iterations': k + 1,
                'residual_history': residual_history,
                'converged': True,
                'final_residual': r_norm,
            }

    return {
        'x': x,
        'iterations': max_iter,
        'residual_history': residual_history,
        'converged': False,
        'final_residual': residual_history[-1] if residual_history else float('inf'),
    }


# ============================================================
#  SOR 方法 (逐次超松弛)
# ============================================================
def sor(A: np.ndarray, b: np.ndarray, omega: float = 1.5,
        x0: Optional[np.ndarray] = None,
        max_iter: int = 1000, tol: float = 1e-8) -> dict:
    """逐次超松弛法 (SOR) 求解 Ax = b

    SOR 迭代:
      x_i^{k+1} = (1-ω) x_i^k + (ω/a_{ii}) (b_i - Σ_{j<i} a_{ij} x_j^{k+1} - Σ_{j>i} a_{ij} x_j^k)

    ω = 1: 退化为 Gauss-Seidel
    ω ∈ (1, 2): 超松弛 (加速收敛)
    ω ∈ (0, 1): 亚松弛 (改善稳定性)

    最优 ω (对模型问题 -u'' = f):
      ω_opt = 2 / (1 + sin(πh))  ≈  2 / (1 + πh)  for small h

    收敛加速比 (vs Gauss-Seidel):
      R_SOR / R_GS ≈ 1 + πh  (当 h → 0 时趋近于 1, 但有限 h 下显著)
    """
    if omega <= 0 or omega >= 2:
        raise ValueError(f"松弛参数 ω 必须在 (0, 2) 内, got {omega}")

    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    diag = np.diag(A).copy()
    if np.any(np.abs(diag) < EPS_NUM):
        raise ValueError("对角元素含零, SOR 不适用")

    residual_history = []
    b_norm = np.linalg.norm(b) + EPS_NUM

    for k in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x[j] if j < i else A[i, j] * x_old[j]
            x_gs = (b[i] - sigma) / diag[i]
            x[i] = (1.0 - omega) * x_old[i] + omega * x_gs

        r = b - A @ x
        r_norm = np.linalg.norm(r)
        residual_history.append(r_norm)

        if r_norm / b_norm < tol:
            return {
                'x': x,
                'iterations': k + 1,
                'residual_history': residual_history,
                'converged': True,
                'final_residual': r_norm,
            }

    return {
        'x': x,
        'iterations': max_iter,
        'residual_history': residual_history,
        'converged': False,
        'final_residual': residual_history[-1] if residual_history else float('inf'),
    }


# ============================================================
#  共轭梯度法 (CG)
# ============================================================
def conjugate_gradient(A: np.ndarray, b: np.ndarray,
                       x0: Optional[np.ndarray] = None,
                       max_iter: int = 1000, tol: float = 1e-8) -> dict:
    """共轭梯度法 (Conjugate Gradient) 求解对称正定系统 Ax = b

    算法:
      r_0 = b - Ax_0
      p_0 = r_0
      for k = 0, 1, 2, ...
        α_k = ||r_k||² / (p_k^T A p_k)
        x_{k+1} = x_k + α_k p_k
        r_{k+1} = r_k - α_k A p_k
        if ||r_{k+1}|| / ||b|| < tol: stop
        β_k = ||r_{k+1}||² / ||r_k||²
        p_{k+1} = r_{k+1} + β_k p_k

    性质:
      - 对 n×n SPD 矩阵, 最多 n 步收敛 (精确算术)
      - 条件数 κ 决定收敛速率: ||e_k||_A ≤ 2 ((√κ-1)/(√κ+1))^k ||e_0||_A
      - 每步仅需一次矩阵-向量乘法
    """
    n = len(b)
    x = x0.copy() if x0 is not None else np.zeros(n)

    r = b - A @ x
    p = r.copy()
    rs_old = np.dot(r, r)
    b_norm = np.linalg.norm(b) + EPS_NUM

    residual_history = []

    for k in range(max_iter):
        Ap = A @ p
        pAp = np.dot(p, Ap)

        if abs(pAp) < EPS_NUM:
            break

        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = np.dot(r, r)
        r_norm = np.sqrt(rs_new)
        residual_history.append(r_norm)

        if r_norm / b_norm < tol:
            return {
                'x': x,
                'iterations': k + 1,
                'residual_history': residual_history,
                'converged': True,
                'final_residual': r_norm,
            }

        beta = rs_new / (rs_old + EPS_NUM)
        p = r + beta * p
        rs_old = rs_new

    return {
        'x': x,
        'iterations': max_iter,
        'residual_history': residual_history,
        'converged': False,
        'final_residual': residual_history[-1] if residual_history else float('inf'),
    }


# ============================================================
#  预处理共轭梯度法 (PCG)
# ============================================================
def preconditioned_cg(A: np.ndarray, b: np.ndarray,
                      precond: Optional[np.ndarray] = None,
                      x0: Optional[np.ndarray] = None,
                      max_iter: int = 1000, tol: float = 1e-8) -> dict:
    """预处理共轭梯度法 (PCG)

     preconditioner M ≈ A^{-1} (近似逆):
      求解 M^{-1} A x = M^{-1} b (条件数降低)

    算法:
      r_0 = b - Ax_0
      z_0 = M^{-1} r_0
      p_0 = z_0
      for k = 0, 1, ...
        α_k = (r_k^T z_k) / (p_k^T A p_k)
        x_{k+1} = x_k + α_k p_k
        r_{k+1} = r_k - α_k A p_k
        z_{k+1} = M^{-1} r_{k+1}
        β_k = (r_{k+1}^T z_{k+1}) / (r_k^T z_k)
        p_{k+1} = z_{k+1} + β_k p_k

    常用预条件器:
      - Jacobi: M = diag(A)
      - SSOR: M = (D+ωL)D^{-1}(D+ωU) / (ω(2-ω))
      - ILU: 不完全 LU 分解
    """
    n = len(b)
    x = x0.copy() if x0 is not None else np.zeros(n)

    # 预条件器 (默认 Jacobi)
    if precond is None:
        diag_A = np.diag(A)
        M_inv = lambda v: v / (diag_A + EPS_NUM)
    else:
        M_inv = lambda v: precond @ v

    r = b - A @ x
    z = M_inv(r)
    p = z.copy()
    rz_old = np.dot(r, z)
    b_norm = np.linalg.norm(b) + EPS_NUM

    residual_history = []

    for k in range(max_iter):
        Ap = A @ p
        pAp = np.dot(p, Ap)

        if abs(pAp) < EPS_NUM:
            break

        alpha = rz_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        r_norm = np.linalg.norm(r)
        residual_history.append(r_norm)

        if r_norm / b_norm < tol:
            return {
                'x': x,
                'iterations': k + 1,
                'residual_history': residual_history,
                'converged': True,
                'final_residual': r_norm,
            }

        z = M_inv(r)
        rz_new = np.dot(r, z)
        beta = rz_new / (rz_old + EPS_NUM)
        p = z + beta * p
        rz_old = rz_new

    return {
        'x': x,
        'iterations': max_iter,
        'residual_history': residual_history,
        'converged': False,
        'final_residual': residual_history[-1] if residual_history else float('inf'),
    }


# ============================================================
#  1D Poisson 问题测试 (来源: 452)
# ============================================================
def solve_poisson_1d_gs(n: int = 50, max_iter: int = 5000, tol: float = 1e-8) -> dict:
    """求解 1D Poisson 方程 (Gauss-Seidel)

    -u'' = f on [0, 1],  u(0) = u(1) = 0
    f(x) = -x(x+3)exp(x)
    精确解: u(x) = x(x-1)exp(x)

    有限差分离散:
      (-u_{i-1} + 2u_i - u_{i+1}) / h² = f_i
      → Au = hf  (A 为三对角矩阵)

    Returns:
        dict: x, u_numerical, u_exact, error, solver_info
    """
    h = 1.0 / (n + 1)
    x = np.linspace(0, 1, n + 2)[1:-1]  # 内部节点

    # 组装矩阵 A (三对角)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 2.0
        if i > 0:
            A[i, i - 1] = -1.0
        if i < n - 1:
            A[i, i + 1] = -1.0
    A /= h ** 2

    # 右端项
    f = lambda xi: -xi * (xi + 3) * np.exp(xi)
    b = np.array([f(xi) for xi in x])

    # Gauss-Seidel 求解
    result = gauss_seidel(A, b, max_iter=max_iter, tol=tol)

    # 精确解
    u_exact = np.array([xi * (xi - 1) * np.exp(xi) for xi in x])
    error = np.max(np.abs(result['x'] - u_exact))

    return {
        'x': x,
        'u_numerical': result['x'],
        'u_exact': u_exact,
        'max_error': error,
        'solver_info': result,
    }
