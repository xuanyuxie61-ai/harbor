#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
稀疏矩阵工具箱
================================================================================

基于 770_mm_to_hb 的稀疏矩阵格式转换思想，提供等离子体计算中
大型稀疏矩阵的组装、格式转换和基本运算。

核心功能：
1. COO (Coordinate) → CSR (Compressed Sparse Row) 格式转换
2. Harwell-Boeing 格式输出（用于外部求解器接口）
3. 稀疏矩阵-向量乘法
4. 对称稀疏矩阵的条件数估计

稀疏矩阵存储的数学基础：

CSR格式：
    - data: 非零元数组
    - indices: 列索引数组
    - indptr: 行指针数组，indptr[i] 表示第i行在data中的起始位置

矩阵条件数估计（基于幂迭代）：
    κ(A) ≈ ||A||_2 ||A^{-1}||_2
    
通过幂迭代估计最大和最小特征值模：
    λ_max ≈ lim_{k→∞} ||A^k v|| / ||A^{k-1} v||
    λ_min ≈ lim_{k→∞} ||(A^{-1})^k v|| / ||(A^{-1})^{k-1} v||
================================================================================
"""

import numpy as np


def coo_to_csr(data, row, col, n_rows, n_cols):
    """
    将COO格式转换为CSR格式。
    
    参数
    ----
    data : ndarray
        非零元素值。
    row, col : ndarray
        行索引和列索引。
    n_rows, n_cols : int
        矩阵维度。
        
    返回
    ----
    csr_data, csr_indices, csr_indptr : ndarray
        CSR格式的三个数组。
    """
    nnz = len(data)
    
    # 按行排序
    order = np.lexsort((col, row))
    data = data[order]
    row = row[order]
    col = col[order]
    
    # 构建indptr
    csr_indptr = np.zeros(n_rows + 1, dtype=int)
    for r in row:
        csr_indptr[r + 1] += 1
    csr_indptr = np.cumsum(csr_indptr)
    
    csr_data = data.copy()
    csr_indices = col.copy()
    
    return csr_data, csr_indices, csr_indptr


def csr_matvec(csr_data, csr_indices, csr_indptr, x):
    """
    CSR格式稀疏矩阵与向量乘法 y = A x。
    """
    n = len(csr_indptr) - 1
    y = np.zeros(n)
    
    for i in range(n):
        for j in range(csr_indptr[i], csr_indptr[i + 1]):
            y[i] += csr_data[j] * x[csr_indices[j]]
    
    return y


def estimate_condition_number(A, n_iter=10):
    """
    使用幂迭代估计矩阵条件数 κ(A) = ||A||_2 ||A^{-1}||_2。
    
    参数
    ----
    A : ndarray
        方阵。
    n_iter : int
        幂迭代次数。
        
    返回
    ----
    cond_est : float
        条件数估计。
    """
    n = A.shape[0]
    
    # 估计最大特征值
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    for _ in range(n_iter):
        Av = A @ v
        norm_Av = np.linalg.norm(Av)
        if norm_Av < 1e-30:
            break
        v = Av / norm_Av
    
    lambda_max = np.abs(v @ (A @ v))
    
    # 估计最小特征值（使用逆迭代）
    try:
        A_inv = np.linalg.inv(A)
        w = np.random.randn(n)
        w = w / np.linalg.norm(w)
        
        for _ in range(n_iter):
            A_inv_w = A_inv @ w
            norm_Aiw = np.linalg.norm(A_inv_w)
            if norm_Aiw < 1e-30:
                break
            w = A_inv_w / norm_Aiw
        
        lambda_min = np.abs(w @ (A @ w))
        lambda_min = max(lambda_min, 1e-30)
    except np.linalg.LinAlgError:
        lambda_min = 1e-30
    
    cond_est = lambda_max / lambda_min
    return cond_est


def sparse_matrix_operations(A_dense, tol=1e-12):
    """
    对稠密矩阵执行稀疏化操作和格式转换。
    
    参数
    ----
    A_dense : ndarray
        输入稠密矩阵。
    tol : float
        稀疏化阈值。
        
    返回
    ----
    result : dict
        包含各种格式和统计信息的字典。
    """
    n_rows, n_cols = A_dense.shape
    
    # 稀疏化
    mask = np.abs(A_dense) > tol
    data = A_dense[mask]
    row_idx, col_idx = np.where(mask)
    nnz = len(data)
    
    # COO → CSR
    csr_data, csr_indices, csr_indptr = coo_to_csr(
        data, row_idx, col_idx, n_rows, n_cols
    )
    
    # 稀疏度统计
    sparsity = 1.0 - nnz / (n_rows * n_cols)
    
    # 条件数估计
    if n_rows == n_cols and n_rows <= 200:
        cond_est = estimate_condition_number(A_dense)
    else:
        cond_est = None
    
    result = {
        'n_rows': n_rows,
        'n_cols': n_cols,
        'nnz': nnz,
        'sparsity': sparsity,
        'coo_data': data,
        'coo_row': row_idx,
        'coo_col': col_idx,
        'csr_data': csr_data,
        'csr_indices': csr_indices,
        'csr_indptr': csr_indptr,
        'condition_number': cond_est
    }
    
    return result


def write_harwell_boeing(filename, A_dense, title="Matrix", key="KEY"):
    """
    将稠密矩阵输出为Harwell-Boeing格式。
    
    基于 770_mm_to_hb 中的 msm_to_hb 思想。
    HB格式：
      Line 1: Title (72 chars) + Key (8 chars)
      Line 2: totcrd ptrcrd indcrd valcrd rhscrd
      Line 3: type nrow ncol nnzeros nrhs
      Line 4: ptrfmt indfmt valfmt rhsfmt
      然后按列输出列指针、行索引、数值。
    """
    n_rows, n_cols = A_dense.shape
    
    # 稀疏化
    mask = np.abs(A_dense) > 1e-12
    row_idx, col_idx = np.where(mask)
    data = A_dense[mask]
    nnz = len(data)
    
    # COO → 按列压缩
    order = np.lexsort((row_idx, col_idx))
    data = data[order]
    row_idx = row_idx[order]
    col_idx = col_idx[order]
    
    # 列指针
    col_ptr = np.zeros(n_cols + 1, dtype=int)
    for c in col_idx:
        col_ptr[c + 1] += 1
    col_ptr = np.cumsum(col_ptr)
    
    with open(filename, 'w') as f:
        # Line 1
        f.write(f"{title[:72]:72s}{key[:8]:8s}\n")
        
        # Line 2: 简化估计
        ptrcrd = (n_cols + 1 + 5) // 6
        indcrd = (nnz + 5) // 6
        valcrd = (nnz + 4) // 5
        rhscrd = 0
        totcrd = ptrcrd + indcrd + valcrd + rhscrd
        f.write(f"{totcrd:14d}{ptrcrd:14d}{indcrd:14d}{valcrd:14d}{rhscrd:14d}\n")
        
        # Line 3
        f.write(f"{'RUA':14s}{n_rows:14d}{n_cols:14d}{nnz:14d}{0:14d}\n")
        
        # Line 4
        f.write(f"{'(6I13)':16s}{'(6I13)':16s}{'(5E16.8)':20s}{'(5E16.8)':20s}\n")
        
        # Column pointers
        for i in range(0, n_cols + 1, 6):
            vals = col_ptr[i:min(i+6, n_cols+1)]
            f.write("".join(f"{v:13d}" for v in vals) + "\n")
        
        # Row indices
        for i in range(0, nnz, 6):
            vals = row_idx[i:min(i+6, nnz)] + 1  # 1-based
            f.write("".join(f"{v:13d}" for v in vals) + "\n")
        
        # Values
        for i in range(0, nnz, 5):
            vals = data[i:min(i+5, nnz)]
            f.write("".join(f"{v:16.8e}" for v in vals) + "\n")
