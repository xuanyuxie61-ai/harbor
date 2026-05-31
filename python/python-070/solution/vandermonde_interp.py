"""
vandermonde_interp.py
Vandermonde 矩阵快速求解与多维插值模块
基于 Bjorck-Pereyra O(n^2) 算法

在渔业生态建模中用于：
1. 年龄-体长关系的多项式插值
2. 渔获量历史数据的平滑重构
3. 空间分布场的双变量 Vandermonde 插值
"""

import numpy as np
from utils import NumericalConfig


def pvand(n, alpha, b):
    """
    Bjorck-Pereyra 算法求解 Vandermonde 线性系统 A * x = b
    其中 A_{ij} = alpha_j^{i-1}, i,j = 1,...,n
    时间复杂度 O(n^2)，远优于一般高斯消元 O(n^3)

    算法步骤：
    1. 前向消去（对应于 Newton 差商）：
       x_j = x_j - alpha_k * x_{j-1},  j = n,...,k+1
    2. 后向替换（对应于 Lagrange 插值）：
       x_j = x_j / (alpha_j - alpha_{j-k}),  j = k+1,...,n
       x_j = x_j - x_{j+1},  j = k,...,n-1

    Parameters
    ----------
    n : int
        矩阵阶数
    alpha : array_like, shape (n,)
        Vandermonde 节点，要求互不相同
    b : array_like, shape (n,)
        右端项向量

    Returns
    -------
    x : ndarray, shape (n,)
        解向量
    """
    alpha = np.asarray(alpha, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(alpha) != n or len(b) != n:
        raise ValueError("Dimension mismatch in pvand")

    # 检查节点互异性
    for i in range(n):
        for j in range(i + 1, n):
            if abs(alpha[i] - alpha[j]) < NumericalConfig.TOL:
                raise ValueError(f"Vandermonde nodes must be distinct: alpha[{i}]={alpha[i]}, alpha[{j}]={alpha[j]}")

    x = b.copy()

    # 前向消去
    for k in range(n - 1):
        for j in range(n - 1, k, -1):
            x[j] = x[j] - alpha[k] * x[j - 1]

    # 后向替换
    for k in range(n - 2, -1, -1):
        for j in range(k + 1, n):
            denom = alpha[j] - alpha[j - k - 1]
            if abs(denom) < NumericalConfig.EPS:
                denom = NumericalConfig.EPS
            x[j] = x[j] / denom
        for j in range(k, n - 1):
            x[j] = x[j] - x[j + 1]

    return x


def dvand(n, alpha, b):
    """
    Bjorck-Pereyra 算法求解转置 Vandermonde 系统 A^T * x = b
    其中 A_{ij} = alpha_j^{i-1}

    Parameters
    ----------
    n : int
        矩阵阶数
    alpha : array_like, shape (n,)
        Vandermonde 节点
    b : array_like, shape (n,)
        右端项向量

    Returns
    -------
    x : ndarray, shape (n,)
        解向量
    """
    alpha = np.asarray(alpha, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(alpha) != n or len(b) != n:
        raise ValueError("Dimension mismatch in dvand")

    x = b.copy()

    # 后向替换（转置的逆序）
    for k in range(n - 1):
        for j in range(n - 1, k, -1):
            denom = alpha[j] - alpha[j - k - 1]
            if abs(denom) < NumericalConfig.EPS:
                denom = NumericalConfig.EPS
            x[j] = x[j] / denom
        for j in range(k, n - 1):
            x[j] = x[j] - x[j + 1]

    # 前向消去
    for k in range(n - 2, -1, -1):
        for j in range(k + 1, n):
            x[j] = x[j] - alpha[k] * x[j - 1]

    return x


def bidim_vandermonde_solve(n, alpha, beta, b):
    """
    求解二维 Vandermonde 系统
    利用嵌套的一维 Bjorck-Pereyra 算法
    矩阵结构：A = V_x \otimes V_y（Kronecker 积）

    在渔业模型中用于双变量插值：
    例如年龄-体重关系的二维重构

    Parameters
    ----------
    n : int
        每维节点数（总未知数为 n*n）
    alpha : array_like, shape (n,)
        x 方向 Vandermonde 节点
    beta : array_like, shape (n,)
        y 方向 Vandermonde 节点
    b : array_like, shape (n*n,)
        右端项，按行优先排列

    Returns
    -------
    x : ndarray, shape (n*n,)
        解向量
    """
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    b = np.asarray(b, dtype=float)

    # 先对每一行（固定 y）求解 x 方向
    temp = np.zeros((n, n), dtype=float)
    for i in range(n):
        rhs = b[i * n:(i + 1) * n]
        temp[i, :] = pvand(n, alpha, rhs)

    # 再对每一列（固定 x）求解 y 方向
    x = np.zeros(n * n, dtype=float)
    for j in range(n):
        rhs = temp[:, j]
        sol = pvand(n, beta, rhs)
        x[j * n:(j + 1) * n] = sol

    return x


def vandermonde_interp_1d(alpha, y, x_eval):
    """
    基于 Vandermonde 系统的多项式插值
    求满足 p(alpha_i) = y_i 的多项式 p 在 x_eval 处的值

    数学上，先求展开系数 c 使得 p(x) = \sum_{j=0}^{n-1} c_j x^j
    通过求解 V * c = y 得到，其中 V_{ij} = alpha_j^{i-1}

    注：对于 n > 10 的节点，Vandermonde 矩阵条件数极大，
    插值可能不稳定。此时建议使用分段低次插值或样条。

    Parameters
    ----------
    alpha : array_like
        插值节点
    y : array_like
        节点处的函数值
    x_eval : array_like
        待求值点

    Returns
    -------
    values : ndarray
        插值结果
    """
    alpha = np.asarray(alpha, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(alpha)

    # 构建 Vandermonde 矩阵 V_{ij} = alpha_j^{i-1}
    V = np.vander(alpha, N=n, increasing=True)
    c = np.linalg.solve(V, y)

    x_eval = np.asarray(x_eval, dtype=float)
    values = np.zeros_like(x_eval, dtype=float)
    for j in range(n):
        values += c[j] * (x_eval ** j)
    return values
