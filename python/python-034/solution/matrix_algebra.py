"""
matrix_algebra.py
=================
对称正定（SPD）矩阵的打包存储与 Cholesky 分解。

原项目映射：991_r8pp（对称正定打包矩阵运算）

在格点 QCD 中，协方差矩阵与关联矩阵通常具有 SPD 结构。
本模块提供打包格式的 Cholesky 分解 A = R^T R，用于：
1. 变分分析中关联矩阵的预处理；
2. 高斯随机场的生成（协方差矩阵平方根）。
"""

import numpy as np


def r8pp_fa(n: int, a: np.ndarray) -> tuple:
    """
    对 SPD 矩阵进行 Cholesky 分解，矩阵以打包（packed）格式存储。

    打包格式：仅存储上三角部分，按列优先顺序展平为长度为 N*(N+1)/2 的向量：
        [A11, A12, A22, A13, A23, A33, A14, ..., ANN]

    数学公式：
        对 SPD 矩阵 A，存在上三角矩阵 R 使得
            A = R^T R
        其中 R 的对角元满足
            R_{jj} = sqrt( A_{jj} - sum_{k=1}^{j-1} R_{kj}^2 )
        非对角元满足
            R_{ij} = ( A_{ij} - sum_{k=1}^{i-1} R_{ki} R_{kj} ) / R_{ii}

    Parameters
    ----------
    n : int
        矩阵阶数。
    a : np.ndarray
        长度为 n*(n+1)//2 的打包 SPD 矩阵。

    Returns
    -------
    r : np.ndarray
        打包格式的上三角因子 R。
    info : int
        0 表示正常返回；k > 0 表示第 k 阶顺序主子式非正定。
    """
    if a.size != n * (n + 1) // 2:
        raise ValueError("Packed array length mismatch.")

    r = a.copy()
    info = 0
    jj = 0

    for j in range(1, n + 1):
        s = 0.0
        kj = jj
        kk = 0
        for k in range(1, j):
            kj += 1
            t = r[kj - 1]
            for i in range(1, k):
                t -= r[kk + i - 1] * r[jj + i - 1]
            kk += k
            if abs(r[kk - 1]) < 1e-15:
                info = j
                return r, info
            t = t / r[kk - 1]
            r[kj - 1] = t
            s += t * t

        jj += j
        s = r[jj - 1] - s
        if s <= 0.0:
            info = j
            return r, info
        r[jj - 1] = np.sqrt(s)

    return r, info


def r8pp_sl(n: int, r: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    利用打包 Cholesky 因子 R 求解线性方程组 A x = b，
    其中 A = R^T R。

    求解分为两步前向/后向替换：
        1. 解 R^T y = b  （前向替换）
        2. 解 R x = y    （后向替换）

    Parameters
    ----------
    n : int
        矩阵阶数。
    r : np.ndarray
        由 r8pp_fa 生成的打包上三角因子。
    b : np.ndarray
        右端项向量。

    Returns
    -------
    x : np.ndarray
        解向量。
    """
    x = b.copy().astype(float)

    # 前向替换: R^T y = b
    kk = 0
    for k in range(1, n + 1):
        x[k - 1] = (x[k - 1] - np.dot(r[kk:kk + k - 1], x[:k - 1])) / r[kk + k - 1]
        kk += k

    # 后向替换: R x = y
    for k in range(n, 0, -1):
        kk = k * (k - 1) // 2
        x[k - 1] = x[k - 1] / r[kk + k - 1]
        for i in range(1, k):
            x[i - 1] -= r[kk + i - 1] * x[k - 1]

    return x


def dense_to_packed(a_dense: np.ndarray) -> np.ndarray:
    """将稠密上三角矩阵转为打包格式。"""
    n = a_dense.shape[0]
    a = np.zeros(n * (n + 1) // 2)
    idx = 0
    for j in range(n):
        for i in range(j + 1):
            a[idx] = a_dense[i, j]
            idx += 1
    return a


def packed_to_dense(n: int, a: np.ndarray) -> np.ndarray:
    """将打包格式转为稠密上三角矩阵（对称填充）。"""
    a_dense = np.zeros((n, n))
    idx = 0
    for j in range(n):
        for i in range(j + 1):
            a_dense[i, j] = a[idx]
            if i != j:
                a_dense[j, i] = a[idx]
            idx += 1
    return a_dense


def spd_sample(n: int, cov_packed: np.ndarray) -> np.ndarray:
    """
    利用 Cholesky 因子从多维高斯分布 N(0, A) 中采样。

    算法：若 A = R^T R，则 x = R^T z，其中 z ~ N(0, I)。
    """
    r, info = r8pp_fa(n, cov_packed)
    if info != 0:
        raise RuntimeError(f"Cholesky factorization failed at step {info}.")
    z = np.random.randn(n)
    x = np.zeros(n)
    # x = R^T z
    kk = 0
    for k in range(1, n + 1):
        x[k - 1] = np.dot(r[kk:kk + k], z[:k])
        kk += k
    return x
