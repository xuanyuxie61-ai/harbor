#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def coo_to_csr(data, row, col, n_rows, n_cols):
    nnz = len(data)
    

    order = np.lexsort((col, row))
    data = data[order]
    row = row[order]
    col = col[order]
    

    csr_indptr = np.zeros(n_rows + 1, dtype=int)
    for r in row:
        csr_indptr[r + 1] += 1
    csr_indptr = np.cumsum(csr_indptr)
    
    csr_data = data.copy()
    csr_indices = col.copy()
    
    return csr_data, csr_indices, csr_indptr


def csr_matvec(csr_data, csr_indices, csr_indptr, x):
    n = len(csr_indptr) - 1
    y = np.zeros(n)
    
    for i in range(n):
        for j in range(csr_indptr[i], csr_indptr[i + 1]):
            y[i] += csr_data[j] * x[csr_indices[j]]
    
    return y


def estimate_condition_number(A, n_iter=10):
    n = A.shape[0]
    

    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    for _ in range(n_iter):
        Av = A @ v
        norm_Av = np.linalg.norm(Av)
        if norm_Av < 1e-30:
            break
        v = Av / norm_Av
    
    lambda_max = np.abs(v @ (A @ v))
    

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
    n_rows, n_cols = A_dense.shape
    

    mask = np.abs(A_dense) > tol
    data = A_dense[mask]
    row_idx, col_idx = np.where(mask)
    nnz = len(data)
    

    csr_data, csr_indices, csr_indptr = coo_to_csr(
        data, row_idx, col_idx, n_rows, n_cols
    )
    

    sparsity = 1.0 - nnz / (n_rows * n_cols)
    

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
    n_rows, n_cols = A_dense.shape
    

    mask = np.abs(A_dense) > 1e-12
    row_idx, col_idx = np.where(mask)
    data = A_dense[mask]
    nnz = len(data)
    

    order = np.lexsort((row_idx, col_idx))
    data = data[order]
    row_idx = row_idx[order]
    col_idx = col_idx[order]
    

    col_ptr = np.zeros(n_cols + 1, dtype=int)
    for c in col_idx:
        col_ptr[c + 1] += 1
    col_ptr = np.cumsum(col_ptr)
    
    with open(filename, 'w') as f:

        f.write(f"{title[:72]:72s}{key[:8]:8s}\n")
        

        ptrcrd = (n_cols + 1 + 5) // 6
        indcrd = (nnz + 5) // 6
        valcrd = (nnz + 4) // 5
        rhscrd = 0
        totcrd = ptrcrd + indcrd + valcrd + rhscrd
        f.write(f"{totcrd:14d}{ptrcrd:14d}{indcrd:14d}{valcrd:14d}{rhscrd:14d}\n")
        

        f.write(f"{'RUA':14s}{n_rows:14d}{n_cols:14d}{nnz:14d}{0:14d}\n")
        

        f.write(f"{'(6I13)':16s}{'(6I13)':16s}{'(5E16.8)':20s}{'(5E16.8)':20s}\n")
        

        for i in range(0, n_cols + 1, 6):
            vals = col_ptr[i:min(i+6, n_cols+1)]
            f.write("".join(f"{v:13d}" for v in vals) + "\n")
        

        for i in range(0, nnz, 6):
            vals = row_idx[i:min(i+6, nnz)] + 1
            f.write("".join(f"{v:13d}" for v in vals) + "\n")
        

        for i in range(0, nnz, 5):
            vals = data[i:min(i+5, nnz)]
            f.write("".join(f"{v:16.8e}" for v in vals) + "\n")
