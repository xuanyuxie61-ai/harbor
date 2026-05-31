"""
linear_solver_cgne.py
共轭梯度法求解法方程 (CG on Normal Equations)
应用于信用风险中大规模线性系统的隐含相关性校准

原项目映射: 151_cg_ne
科学问题: 在信用组合模型中，经常需要从市场价格反推隐含相关性。
设市场观测到的 CDO 分券价格为向量 P_market，
模型价格 P_model(rho) 对相关性参数 rho 的敏感性构成 Jacobi 矩阵 J。
校准问题转化为最小二乘:
    min || J * delta_rho - (P_market - P_model) ||^2
这正是一个法方程求解问题: (J^T J) delta_rho = J^T * residual
由于 J 通常病态 (ill-conditioned)，直接求逆不稳定，
故采用 CG-NE 迭代求解。
"""

import numpy as np
from typing import Tuple, Optional, Callable


def cg_ne_solve(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    max_iter: Optional[int] = None,
    tol: float = 1e-10
) -> Tuple[np.ndarray, int, float]:
    """
    共轭梯度法求解法方程 A^T A x = A^T b

    算法步骤:
        r_0 = b - A x_0
        z_0 = A^T r_0
        d_0 = z_0
        对 k = 0, 1, 2, ...:
            alpha_k = (z_k^T z_k) / ((A d_k)^T (A d_k))
            x_{k+1} = x_k + alpha_k * d_k
            r_{k+1} = r_k - alpha_k * A d_k
            z_{k+1} = A^T r_{k+1}
            beta_k = (z_{k+1}^T z_{k+1}) / (z_k^T z_k)
            d_{k+1} = z_{k+1} + beta_k * d_k

    Parameters:
        A: 系数矩阵 (m x n)
        b: 右端项 (m,)
        x0: 初始猜测 (n,)
        max_iter: 最大迭代次数，默认 n
        tol: 残差收敛阈值

    Returns:
        x: 解向量
        iter_count: 实际迭代次数
        residual_norm: 最终残差范数
    """
    m, n = A.shape
    if max_iter is None:
        max_iter = n

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = x0.copy()

    r = b - A @ x
    z = A.T @ r
    d = z.copy()
    rz_old = np.dot(z, z)

    for k in range(max_iter):
        Ad = A @ d
        denom = np.dot(Ad, Ad)
        if denom < 1e-30:
            break
        alpha = rz_old / denom
        x += alpha * d
        r -= alpha * Ad
        z = A.T @ r
        rz_new = np.dot(z, z)
        residual_norm = np.sqrt(rz_new)
        if residual_norm < tol:
            return x, k + 1, residual_norm
        beta = rz_new / rz_old
        d = z + beta * d
        rz_old = rz_new

    return x, max_iter, np.sqrt(rz_old)


def cg_ne_solve_with_regularization(
    A: np.ndarray,
    b: np.ndarray,
    lam: float = 1e-6,
    x0: Optional[np.ndarray] = None,
    max_iter: Optional[int] = None,
    tol: float = 1e-10
) -> Tuple[np.ndarray, int, float]:
    """
    带 Tikhonov 正则化的 CG-NE:
        min ||A x - b||^2 + lam * ||x||^2
    等价于求解:
        (A^T A + lam I) x = A^T b

    在信用风险中，当不同期限的 CDO 分券价格提供的信息不足时
    (即 A 不满秩或病态)，正则化确保解的唯一性和稳定性。
    """
    m, n = A.shape
    if max_iter is None:
        max_iter = n
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = x0.copy()

    r = b - A @ x
    z = A.T @ r - lam * x
    d = z.copy()
    rz_old = np.dot(z, z)

    for k in range(max_iter):
        Ad = A @ d
        denom = np.dot(Ad, Ad) + lam * np.dot(d, d)
        if denom < 1e-30:
            break
        alpha = rz_old / denom
        x += alpha * d
        r -= alpha * Ad
        z = A.T @ r - lam * x
        rz_new = np.dot(z, z)
        residual_norm = np.sqrt(rz_new)
        if residual_norm < tol:
            return x, k + 1, residual_norm
        beta = rz_new / rz_old
        d = z + beta * d
        rz_old = rz_new

    return x, max_iter, np.sqrt(rz_old)


def helmert_matrix(n: int) -> np.ndarray:
    """
    构造 Helmert 正交矩阵
    H(1,j) = 1/sqrt(n)
    H(i,j) = 1/sqrt(i*(i-1))  for j < i
    H(i,i) = -(i-1)/sqrt(i*(i-1))

    在信用风险中可用于构造正交的行业/国家分类基
    """
    H = np.zeros((n, n), dtype=float)
    H[0, :] = 1.0 / np.sqrt(n)
    for i in range(1, n):
        H[i, :i] = 1.0 / np.sqrt(i * (i + 1))
        H[i, i] = -i / np.sqrt(i * (i + 1))
    return H


def lesp_matrix(m: int, n: int) -> np.ndarray:
    """
    构造 LESP 三对角测试矩阵
    A(i,i-1) = 1/i, A(i,i) = -(2i+3), A(i,i+1) = i+1

    用于测试 CG-NE 在病态矩阵上的收敛性
    """
    A = np.zeros((m, n), dtype=float)
    for i in range(min(m, n)):
        A[i, i] = -(2.0 * (i + 1) + 3.0)
    for i in range(1, min(m, n)):
        A[i, i - 1] = 1.0 / (i + 1)
    for i in range(min(m, n - 1)):
        A[i, i + 1] = (i + 1) + 1.0
    return A


def test_cg_ne():
    """测试 CG-NE 求解器"""
    np.random.seed(42)
    n = 20
    # 使用 Helmert 矩阵构造一个良态测试问题
    H = helmert_matrix(n)
    x_true = np.random.randn(n)
    b = H @ x_true
    x_sol, iters, res = cg_ne_solve(H, b, tol=1e-12)
    err = np.linalg.norm(x_sol - x_true)
    assert err < 1e-8, f"CG-NE 解误差过大: {err}"
    print(f"cg_ne test passed. iters={iters}, residual={res:.2e}, error={err:.2e}")


if __name__ == "__main__":
    test_cg_ne()
