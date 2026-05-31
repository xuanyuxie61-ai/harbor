"""
utils.py
通用数值工具与数学辅助函数
为信用风险违约相关性建模提供底层数值支撑
"""

import numpy as np
import math
from typing import Tuple, Optional


def gauss_legendre_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 n 点 Gauss-Legendre 求积节点与权重
    用于奇异积分主值计算及高精度数值积分

    Parameters:
        n: 求积点数

    Returns:
        x: 节点 (在 [-1, 1] 上)
        w: 权重
    """
    if n <= 0:
        raise ValueError("求积点数 n 必须为正整数")
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def logistic_transform(z: np.ndarray, a: float = 0.0, b: float = 1.0) -> np.ndarray:
    """
    将实数域映射到 (a, b) 区间，用于保证相关性参数等处于合法区间
    g(z) = a + (b-a) / (1 + exp(-z))

    Parameters:
        z: 实数输入
        a: 下界
        b: 上界

    Returns:
        映射后的值
    """
    return a + (b - a) / (1.0 + np.exp(-z))


def inverse_logistic_transform(y: np.ndarray, a: float = 0.0, b: float = 1.0) -> np.ndarray:
    """
    logistic_transform 的反函数
    """
    eps = 1e-12
    y_clip = np.clip(y, a + eps, b - eps)
    return -np.log((b - a) / (y_clip - a) - 1.0)


def safe_sqrt(x: np.ndarray) -> np.ndarray:
    """
    安全开方，对负数返回 0 并避免警告
    """
    return np.sqrt(np.maximum(x, 0.0))


def normal_cdf(x: np.ndarray) -> np.ndarray:
    """
    标准正态累积分布函数
    采用误差函数实现，具有数值稳定性
    """
    return 0.5 * (1.0 + np.vectorize(lambda t: math.erf(t / np.sqrt(2.0)))(x))


def normal_pdf(x: np.ndarray) -> np.ndarray:
    """
    标准正态概率密度函数
    phi(x) = (1/sqrt(2*pi)) * exp(-x^2/2)
    """
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def is_positive_definite(M: np.ndarray, tol: float = 1e-10) -> bool:
    """
    判断矩阵是否正定（通过检查所有特征值是否为正）
    用于验证相关性矩阵的合法性
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return False
    eigvals = np.linalg.eigvalsh(M)
    return bool(np.all(eigvals > tol))


def nearest_correlation_matrix(A: np.ndarray, max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
    """
    通过交替投影计算最近的相关性矩阵
    Higham (2002) 算法，确保输出矩阵对称、半正定、对角线为 1

    Parameters:
        A: 输入对称矩阵
        max_iter: 最大迭代次数
        tol: 收敛容差

    Returns:
        最近的相关性矩阵
    """
    n = A.shape[0]
    R = np.copy(A)
    dS = np.zeros_like(A)
    Y = np.copy(A)
    for k in range(max_iter):
        R = Y - dS
        # 投影到半正定锥
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals = np.maximum(eigvals, 0.0)
        X = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # 投影到单位对角线
        dS = X - R
        np.fill_diagonal(dS, 0.0)
        Y = np.copy(X)
        np.fill_diagonal(Y, 1.0)
        err = np.max(np.abs(Y - X))
        if err < tol:
            break
    np.fill_diagonal(Y, 1.0)
    return Y


def cholesky_with_pivot(A: np.ndarray, tol: float = 1e-12) -> Optional[np.ndarray]:
    """
    带主元的 Cholesky 分解，用于相关性矩阵分解
    若矩阵接近奇异，则对接近零的特征值进行截断

    Parameters:
        A: 对称半正定矩阵
        tol: 截断容差

    Returns:
        下三角矩阵 L 或 None
    """
    try:
        L = np.linalg.cholesky(A)
        return L
    except np.linalg.LinAlgError:
        # 使用特征值截断
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, tol)
        L = eigvecs @ np.diag(np.sqrt(eigvals))
        return L


def finite_difference_1d_second_derivative(u: np.ndarray, dx: float) -> np.ndarray:
    """
    一维二阶中心差分
    d^2u/dx^2 的离散近似: (u_{i-1} - 2u_i + u_{i+1}) / dx^2
    边界处使用前向/后向差分
    """
    n = len(u)
    if n < 3:
        raise ValueError("数组长度至少为 3")
    d2u = np.zeros_like(u)
    d2u[1:-1] = (u[:-2] - 2.0 * u[1:-1] + u[2:]) / (dx * dx)
    # 边界：一阶精度
    d2u[0] = (2.0 * u[0] - 5.0 * u[1] + 4.0 * u[2] - u[3]) / (dx * dx)
    d2u[-1] = (2.0 * u[-1] - 5.0 * u[-2] + 4.0 * u[-3] - u[-4]) / (dx * dx)
    return d2u


def tridiagonal_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    求解三对角线性系统 Ax = d
    a: 下对角线 (长度 n-1)
    b: 主对角线 (长度 n)
    c: 上对角线 (长度 n-1)
    d: 右端项 (长度 n)
    采用 Thomas 算法，O(n) 复杂度
    """
    n = len(b)
    if len(a) != n - 1 or len(c) != n - 1 or len(d) != n:
        raise ValueError("三对角矩阵维度不匹配")
    cp = np.zeros(n - 1, dtype=float)
    dp = np.zeros(n, dtype=float)
    x = np.zeros(n, dtype=float)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * cp[i - 1]
        if abs(denom) < 1e-15:
            denom = 1e-15
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom
    dp[-1] = (d[-1] - a[-1] * dp[-2]) / (b[-1] - a[-1] * cp[-2])

    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x
