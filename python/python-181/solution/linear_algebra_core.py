"""
linear_algebra_core.py
核心线性代数与数值计算工具库
融合原项目: 983_r8lib

提供矩阵运算、特征值分解、数值稳定性保障等基础功能，
服务于高维流形降维中的各类线性代数需求。
"""

import numpy as np
from typing import Tuple, Optional


def cholesky_factor(a: np.ndarray) -> np.ndarray:
    """
    计算对称正定矩阵的Cholesky分解 A = L * L^T
    输入: a (n, n) 对称正定矩阵
    输出: l (n, n) 下三角矩阵
    """
    n = a.shape[0]
    l = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        s = 0.0
        for k in range(j):
            s += l[j, k] ** 2
        d = a[j, j] - s
        if d <= 1e-14:
            d = 1e-14  # 数值稳定性修正
        l[j, j] = np.sqrt(d)
        for i in range(j + 1, n):
            s = 0.0
            for k in range(j):
                s += l[i, k] * l[j, k]
            l[i, j] = (a[i, j] - s) / l[j, j]
    return l


def solve_cholesky(l: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    利用Cholesky分解求解线性方程组 A x = b
    先解 L y = b, 再解 L^T x = y
    """
    n = l.shape[0]
    y = np.zeros_like(b, dtype=np.float64)
    x = np.zeros_like(b, dtype=np.float64)
    # 前向替换
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= l[i, j] * y[j]
        y[i] = s / l[i, i]
    # 后向替换
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= l[j, i] * x[j]
        x[i] = s / l[i, i]
    return x


def jacobi_eigenvalue(a: np.ndarray, max_iter: int = 1000, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Jacobi方法计算实对称矩阵的特征值和特征向量
    输入: a (n, n) 实对称矩阵
    输出: (eigenvalues, eigenvectors)
    """
    n = a.shape[0]
    v = np.eye(n, dtype=np.float64)
    a_work = a.copy()
    for _iter in range(max_iter):
        # 寻找最大非对角元
        max_val = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a_work[i, j]) > max_val:
                    max_val = abs(a_work[i, j])
                    p, q = i, j
        if max_val < tol:
            break
        # 计算旋转角
        if abs(a_work[p, p] - a_work[q, q]) < 1e-15:
            theta = np.pi / 4.0
        else:
            theta = 0.5 * np.arctan2(2.0 * a_work[p, q], a_work[q, q] - a_work[p, p])
        c = np.cos(theta)
        s = np.sin(theta)
        # 执行旋转
        app = c * c * a_work[p, p] - 2.0 * c * s * a_work[p, q] + s * s * a_work[q, q]
        aqq = s * s * a_work[p, p] + 2.0 * c * s * a_work[p, q] + c * c * a_work[q, q]
        apq = 0.0
        a_work[p, p] = app
        a_work[q, q] = aqq
        a_work[p, q] = apq
        a_work[q, p] = apq
        for i in range(n):
            if i != p and i != q:
                a_ip = a_work[i, p]
                a_iq = a_work[i, q]
                a_work[i, p] = c * a_ip - s * a_iq
                a_work[p, i] = a_work[i, p]
                a_work[i, q] = s * a_ip + c * a_iq
                a_work[q, i] = a_work[i, q]
        # 更新特征向量
        for i in range(n):
            v_ip = v[i, p]
            v_iq = v[i, q]
            v[i, p] = c * v_ip - s * v_iq
            v[i, q] = s * v_ip + c * v_iq
    eigenvalues = np.diag(a_work)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], v[:, idx]


def matrix_power(a: np.ndarray, p: int) -> np.ndarray:
    """计算矩阵整数幂 A^p"""
    if p == 0:
        return np.eye(a.shape[0], dtype=np.float64)
    if p == 1:
        return a.copy()
    if p % 2 == 0:
        half = matrix_power(a, p // 2)
        return half @ half
    else:
        return a @ matrix_power(a, p - 1)


def frobenius_norm(a: np.ndarray) -> float:
    """Frobenius范数 ||A||_F = sqrt(sum_{i,j} a_{ij}^2)"""
    return np.sqrt(np.sum(a ** 2))


def householder_qr(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Householder QR分解
    返回 Q (正交矩阵), R (上三角矩阵)
    """
    m, n = a.shape
    q = np.eye(m, dtype=np.float64)
    r = a.copy()
    for k in range(min(m, n)):
        x = r[k:, k].copy()
        norm_x = np.linalg.norm(x)
        if norm_x < 1e-15:
            continue
        alpha = -np.sign(x[0]) * norm_x
        u = x.copy()
        u[0] -= alpha
        norm_u = np.linalg.norm(u)
        if norm_u < 1e-15:
            continue
        u = u / norm_u
        # H = I - 2 u u^T
        r[k:, k:] -= 2.0 * np.outer(u, u @ r[k:, k:])
        q[:, k:] -= 2.0 * np.outer(q[:, k:] @ u, u)
    return q, r


def givens_rotation(a: float, b: float) -> Tuple[float, float]:
    """
    计算Givens旋转参数 (c, s) 使得
    [ c  s ] [a]   [r]
    [-s  c ] [b] = [0]
    """
    if b == 0.0:
        c = 1.0
        s = 0.0
    elif abs(b) > abs(a):
        tau = -a / b
        s = 1.0 / np.sqrt(1.0 + tau ** 2)
        c = s * tau
    else:
        tau = -b / a
        c = 1.0 / np.sqrt(1.0 + tau ** 2)
        s = c * tau
    return c, s


def tridiagonal_solve(d: np.ndarray, e: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    求解对称三对角方程组 T x = b
    d: 对角线元素 (n,)
    e: 次对角线元素 (n-1,)
    """
    n = len(d)
    x = b.copy()
    cp = d.copy()
    ep = e.copy()
    # 前向消去
    for i in range(1, n):
        ratio = ep[i - 1] / cp[i - 1]
        cp[i] -= ratio * ep[i - 1]
        x[i] -= ratio * x[i - 1]
    # 后向替换
    x[n - 1] /= cp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (x[i] - ep[i] * x[i + 1]) / cp[i]
    return x


def normalize_vector(v: np.ndarray, ord: int = 2) -> np.ndarray:
    """向量归一化"""
    norm = np.linalg.norm(v, ord=ord)
    if norm < 1e-15:
        return v
    return v / norm


def condition_number(a: np.ndarray) -> float:
    """矩阵条件数估计 (2-范数)"""
    s = np.linalg.svd(a, compute_uv=False)
    if s[-1] < 1e-15:
        return np.inf
    return s[0] / s[-1]


def safe_inverse(a: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """安全矩阵求逆，对接近奇异的矩阵进行正则化"""
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    s_inv = np.where(s > rcond * s[0], 1.0 / s, 0.0)
    return vt.T @ np.diag(s_inv) @ u.T
