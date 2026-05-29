# -*- coding: utf-8 -*-
"""
utils.py
========
通用数值工具与鲁棒性辅助函数。

包含：
- 向量/矩阵范数与残差计算
- 校验和验证（改编自 ISBN 校验思想）
- 边界条件检查与数值安全函数
- 打印与格式化工具
"""

import numpy as np
import math


def vec_norm(v, ord=2):
    """计算向量的 L2（或指定阶）范数，带边界检查。"""
    v = np.asarray(v, dtype=float).flatten()
    if v.size == 0:
        return 0.0
    if ord == 2:
        return np.linalg.norm(v, 2)
    elif ord == 1:
        return np.linalg.norm(v, 1)
    elif ord == np.inf:
        return np.linalg.norm(v, np.inf)
    else:
        return np.linalg.norm(v, ord)


def mat_norm(A, ord=2):
    """计算矩阵范数，处理空矩阵边界。"""
    A = np.asarray(A, dtype=float)
    if A.size == 0:
        return 0.0
    return np.linalg.norm(A, ord)


def residual_dense(A, x, b):
    """ dense 矩阵残差 r = b - A @ x """
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float).flatten()
    b = np.asarray(b, dtype=float).flatten()
    if A.shape[0] != b.size or A.shape[1] != x.size:
        raise ValueError("Dimension mismatch in residual_dense.")
    return b - A @ x


def residual_tridiag(a, x, b):
    """
    三对角矩阵（R83 竖直折叠存储 a[3,n]）残差 r = b - A @ x。
    a[0,:] 为上对角线，a[1,:] 为主对角线，a[2,:] 为下对角线。
    """
    a = np.asarray(a, dtype=float)
    x = np.asarray(x, dtype=float).flatten()
    b = np.asarray(b, dtype=float).flatten()
    n = x.size
    if a.shape != (3, n):
        raise ValueError("R83 matrix must have shape (3, n).")
    ax = np.zeros(n, dtype=float)
    ax[0] = a[1, 0] * x[0] + a[0, 1] * x[1] if n > 1 else a[1, 0] * x[0]
    for i in range(1, n - 1):
        ax[i] = a[2, i - 1] * x[i - 1] + a[1, i] * x[i] + a[0, i + 1] * x[i + 1]
    if n > 1:
        ax[n - 1] = a[2, n - 2] * x[n - 2] + a[1, n - 1] * x[n - 1]
    return b - ax


def residual_banded(mu, a, x, b):
    """
    对称带状矩阵（R8PBU 存储 a[mu+1, n]）残差。
    第 mu+1 行为主对角线，第 mu 行为第一条上对角线，...，第 1 行为第 mu 条上对角线。
    """
    a = np.asarray(a, dtype=float)
    x = np.asarray(x, dtype=float).flatten()
    b = np.asarray(b, dtype=float).flatten()
    n = x.size
    if a.shape != (mu + 1, n):
        raise ValueError("R8PBU matrix shape mismatch.")
    ax = np.zeros(n, dtype=float)
    for i in range(n):
        ax[i] = a[mu, i] * x[i]
    for k in range(1, mu + 1):
        for j in range(mu + 1 - k, n):
            i_eq = k + j - mu - 1
            ax[i_eq] += a[mu - k, j] * x[j]
            ax[j] += a[mu - k, j] * x[i_eq]
    return b - ax


def residual_sd(ndiag, offset, a, x, b):
    """
    对称对角存储（R8SD）残差。
    a[n, ndiag], offset[ndiag] 为非零对角线的偏移量。
    """
    a = np.asarray(a, dtype=float)
    x = np.asarray(x, dtype=float).flatten()
    b = np.asarray(b, dtype=float).flatten()
    n = x.size
    ax = np.zeros(n, dtype=float)
    for i in range(n):
        for jd in range(ndiag):
            off = offset[jd]
            if off >= 0:
                j = i + off
                if 0 <= j < n:
                    ax[i] += a[i, jd] * x[j]
                    if off != 0:
                        ax[j] += a[i, jd] * x[i]
    return b - ax


def residual_sparse(nz_num, row, col, a_val, x, b):
    """
    稀疏三元组（COO）格式残差。
    """
    x = np.asarray(x, dtype=float).flatten()
    b = np.asarray(b, dtype=float).flatten()
    n = x.size
    ax = np.zeros(n, dtype=float)
    for k in range(nz_num):
        i = row[k]
        j = col[k]
        ax[i] += a_val[k] * x[j]
    return b - ax


def checksum_vector(x, base=11):
    """
    基于 ISBN 模校验思想的向量校验和。
    对解向量生成一个标量校验和，用于验证迭代一致性。
    公式：
        S = sum_{i=1}^{n} ( (n - i + 1) * x_i )  mod  base
    """
    x = np.asarray(x, dtype=float).flatten()
    if x.size == 0:
        return 0
    n = x.size
    weights = np.arange(n, 0, -1, dtype=float)
    s = float(np.dot(weights, x))
    # 处理负数模
    cs = ((s % base) + base) % base
    return int(cs)


def verify_checksum(x, expected, base=11, tol=1e-6):
    """校验解向量的校验和是否在容差范围内一致。"""
    cs = checksum_vector(x, base)
    return abs(cs - expected) < tol or abs(cs - expected - base) < tol or abs(cs - expected + base) < tol


def safe_divide(a, b, default=0.0):
    """安全除法，避免除以零。"""
    if abs(b) < 1e-30:
        return default
    return a / b


def condition_number_estimate(A, method='power', max_iter=50):
    """
    用幂法/反幂法估计对称正定矩阵的条件数 κ(A) = λ_max / λ_min。
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    # 幂法估计最大特征值
    x = np.random.randn(n)
    x = x / vec_norm(x)
    for _ in range(max_iter):
        y = A @ x
        norm_y = vec_norm(y)
        if norm_y < 1e-30:
            break
        x = y / norm_y
    lam_max = float(x @ (A @ x))

    # 反幂法估计最小特征值（简单实现，用若干步最速下降近似）
    x2 = np.random.randn(n)
    x2 = x2 / vec_norm(x2)
    # 用 CG 一步近似逆矩阵作用：对 SPD 矩阵，用 A^{-1} 的最大特征值 = 1/λ_min
    # 这里简化为用带正则化的幂法
    eps_reg = 1e-12
    for _ in range(max_iter):
        try:
            y2 = np.linalg.solve(A + eps_reg * np.eye(n), x2)
        except np.linalg.LinAlgError:
            y2 = x2
        norm_y2 = vec_norm(y2)
        if norm_y2 < 1e-30:
            break
        x2 = y2 / norm_y2
    lam_min = float(x2 @ (A @ x2))
    if lam_min <= 0:
        lam_min = eps_reg
    kappa = lam_max / lam_min
    return kappa, lam_max, lam_min


def print_header(title):
    """打印带分隔线的标题。"""
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_vec(name, v, max_show=8):
    """安全打印向量片段。"""
    v = np.asarray(v, dtype=float).flatten()
    n = v.size
    if n <= max_show:
        s = ", ".join(f"{vi:.6e}" for vi in v)
    else:
        s = ", ".join(f"{vi:.6e}" for vi in v[:max_show // 2])
        s += ", ... , "
        s += ", ".join(f"{vi:.6e}" for vi in v[-max_show // 2:])
    print(f"  {name}[{n}] = [{s}]")


def is_spd(A, tol=1e-10):
    """检查矩阵是否对称正定（通过Cholesky分解）。"""
    A = np.asarray(A, dtype=float)
    if A.shape[0] != A.shape[1]:
        return False
    if not np.allclose(A, A.T, atol=tol):
        return False
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
