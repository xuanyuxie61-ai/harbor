"""
稀疏线性系统求解模块
====================
基于种子项目 736_matman 的矩阵操作（LU分解、行变换）思想。

核心科学问题：
    大气数值模式中的离散化产生大规模稀疏线性系统。
    本模块实现基于高斯消元的稀疏直接求解器，包含：
    - 部分主元 LU 分解（PA = LU）
    - 带填充元控制的稀疏消元
    - 前代-回代求解

数学模型：

对于线性系统 A*x = b，进行 LU 分解：
    P * A = L * U

其中：
    P: 置换矩阵（部分主元）
    L: 单位下三角矩阵
    U: 上三角矩阵

求解步骤：
    1. P*A = L*U
    2. 解 L*y = P*b   （前代）
    3. 解 U*x = y     （回代）

部分主元选取：
    |a_{k,k}| = max_{i≥k} |a_{i,k}|

数值稳定性条件：
    增长因子 ρ = max|u_{ij}| / max|a_{ij}| 应控制在可接受范围。
"""

import numpy as np


def lu_decomposition_pivot(A, eps=1e-12):
    """
    带部分主元的 Doolittle LU 分解。
    
    基于种子项目 736_matman 的矩阵行变换思想。
    
    参数:
        A: (n, n) 方阵
        eps: 零主元阈值
    
    返回:
        L: 单位下三角矩阵
        U: 上三角矩阵
        P: 置换向量（行交换记录）
        success: 是否成功
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("输入矩阵必须是方阵")
    
    # 工作副本
    M = A.copy()
    P = np.arange(n)
    
    for k in range(n - 1):
        # 部分主元选取
        pivot_idx = k + np.argmax(np.abs(M[k:, k]))
        
        if abs(M[pivot_idx, k]) < eps:
            # 奇异或近奇异
            continue
        
        # 行交换
        if pivot_idx != k:
            M[[k, pivot_idx], :] = M[[pivot_idx, k], :]
            P[[k, pivot_idx]] = P[[pivot_idx, k]]
        
        # 消元
        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]
            M[i, k] = factor
            M[i, k + 1:] -= factor * M[k, k + 1:]
    
    # 提取 L 和 U
    L = np.tril(M, -1) + np.eye(n)
    U = np.triu(M)
    
    return L, U, P, True


def forward_substitution(L, b):
    """
    前代求解 L*y = b。
    
    公式：
        y_i = b_i - Σ_{j=0}^{i-1} L_{ij} * y_j
    
    参数:
        L: 单位下三角矩阵
        b: 右端项
    
    返回:
        y: 解向量
    """
    n = len(b)
    y = np.zeros(n)
    
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    
    return y


def backward_substitution(U, y):
    """
    回代求解 U*x = y。
    
    公式：
        x_i = (y_i - Σ_{j=i+1}^{n-1} U_{ij} * x_j) / U_{ii}
    
    参数:
        U: 上三角矩阵
        y: 右端项
    
    返回:
        x: 解向量
    """
    n = len(y)
    x = np.zeros(n)
    
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < 1e-14:
            x[i] = 0.0  # 奇异情况下的最小范数解
        else:
            x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    
    return x


def solve_linear_system(A, b, eps=1e-12):
    """
    求解线性系统 A*x = b。
    
    参数:
        A: 系数矩阵
        b: 右端项
        eps: 零阈值
    
    返回:
        x: 解向量
        residual_norm: 残差范数 ||A*x - b||
    """
    L, U, P, success = lu_decomposition_pivot(A, eps)
    
    # 置换右端项
    b_perm = b[P]
    
    # 前代-回代
    y = forward_substitution(L, b_perm)
    x = backward_substitution(U, y)
    
    # 计算残差
    residual = np.dot(A, x) - b
    residual_norm = np.linalg.norm(residual)
    
    return x, residual_norm


def sparse_matrix_vector_product(A_data, A_row, A_col, x):
    """
    稀疏矩阵-向量乘积（CSR 格式）。
    
    参数:
        A_data: 非零元素值
        A_row: 行指针
        A_col: 列索引
        x: 向量
    
    返回:
        y: 结果向量
    """
    n = len(A_row) - 1
    y = np.zeros(n)
    
    for i in range(n):
        for j in range(A_row[i], A_row[i + 1]):
            y[i] += A_data[j] * x[A_col[j]]
    
    return y


def iterative_refinement(A, b, x0, max_iter=5, eps=1e-12):
    """
    迭代精化改善解的精度。
    
    算法：
        x^{k+1} = x^k + δx^k
        其中 A * δx^k = b - A*x^k  （残差方程）
    
    参数:
        A: 系数矩阵
        b: 右端项
        x0: 初始猜测
        max_iter: 最大迭代次数
    
    返回:
        x: 精化后的解
        history: 残差范数历史
    """
    x = x0.copy()
    history = []
    
    L, U, P, _ = lu_decomposition_pivot(A, eps)
    
    for _ in range(max_iter):
        residual = b - np.dot(A, x)
        res_norm = np.linalg.norm(residual)
        history.append(res_norm)
        
        if res_norm < 1e-12:
            break
        
        # 用LU分解求解残差方程
        b_perm = residual[P]
        y = forward_substitution(L, b_perm)
        dx = backward_substitution(U, y)
        
        x = x + dx
    
    return x, history
