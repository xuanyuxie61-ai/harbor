"""
sparse_io.py
============
稀疏矩阵I/O与数据结构模块（融合 507_hb_io）

功能：
- Harwell-Boeing稀疏矩阵格式的读写封装
- CSR/CSC格式转换
- 块稀疏矩阵（PCE-Galerkin系统）的存储优化

数学背景：
- PCE-Galerkin离散化产生块稀疏矩阵 A = [A_{ij}]，其中每个块对应空间算子
- HB格式: 标题行 + 指针 + 行索引 + 数值
- CSR格式: data, indices, indptr
"""

import numpy as np


def dense_to_csr(A):
    """
    将稠密矩阵转为CSR格式。
    """
    A = np.asarray(A, dtype=float)
    n_rows, n_cols = A.shape
    data = []
    indices = []
    indptr = [0]
    
    for i in range(n_rows):
        row_nnz = 0
        for j in range(n_cols):
            if abs(A[i, j]) > 1e-15:
                data.append(A[i, j])
                indices.append(j)
                row_nnz += 1
        indptr.append(indptr[-1] + row_nnz)
    
    return {
        'data': np.array(data),
        'indices': np.array(indices, dtype=int),
        'indptr': np.array(indptr, dtype=int),
        'shape': (n_rows, n_cols)
    }


def csr_to_dense(csr):
    """
    CSR转稠密矩阵。
    """
    n_rows, n_cols = csr['shape']
    A = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for idx in range(csr['indptr'][i], csr['indptr'][i + 1]):
            j = csr['indices'][idx]
            A[i, j] = csr['data'][idx]
    return A


def write_hb_simple(filename, A, title="SPARSE_MATRIX"):
    """
    简化版Harwell-Boeing格式写入。
    仅处理实数、非对称、 assembled矩阵。
    
    HB格式结构（简化）：
    Line 1: Title(72) + Key(8)
    Line 2: TOTCRD PTRCRD INDCRD VALCRD RHSCRD
    Line 3: MXTYPE NROW NCOL NNZERO NELTVL
    Line 4+: 列指针 ( ncol+1 个整数)
    Line 4+: 行索引 ( nnzero 个整数)
    Line 4+: 数值 ( nnzero 个实数)
    """
    A = np.asarray(A, dtype=float)
    n_rows, n_cols = A.shape
    
    # 构造COO格式
    rows, cols = np.where(np.abs(A) > 1e-15)
    vals = A[rows, cols]
    nnz = len(vals)
    
    # 按列排序
    order = np.lexsort((rows, cols))
    rows = rows[order]
    cols = cols[order]
    vals = vals[order]
    
    # 列指针
    colptr = np.zeros(n_cols + 1, dtype=int)
    colptr[n_cols] = nnz
    for c in range(n_cols):
        colptr[c] = np.searchsorted(cols, c, side='left')
    
    with open(filename, 'w') as f:
        # 标题行
        f.write(f"{title:<72}{'EXPM1':>8}\n")
        # 数据行数（粗略估计）
        f.write(f"{0:14d}{0:14d}{0:14d}{0:14d}{0:14d}\n")
        # 矩阵信息
        f.write(f"{'RUA':>3}{n_rows:14d}{n_cols:14d}{nnz:14d}{0:14d}\n")
        # 列指针
        for i, cp in enumerate(colptr):
            f.write(f"{cp:8d}")
            if (i + 1) % 10 == 0:
                f.write("\n")
        if len(colptr) % 10 != 0:
            f.write("\n")
        # 行索引
        for i, r in enumerate(rows):
            f.write(f"{r + 1:8d}")  # 1-based
            if (i + 1) % 10 == 0:
                f.write("\n")
        if len(rows) % 10 != 0:
            f.write("\n")
        # 数值
        for i, v in enumerate(vals):
            f.write(f"{v:16.8e}")
            if (i + 1) % 5 == 0:
                f.write("\n")
        if len(vals) % 5 != 0:
            f.write("\n")


def build_pce_block_sparse(spatial_A, pce_degree, alpha_mu, alpha_sigma):
    """
    为空间算子和随机耦合构建块稀疏PCE-Galerkin矩阵。
    
    系统方程: ∂u/∂t = -L u - α(ξ) N(u)
    PCE展开后，每个空间自由度对应N_pce个系数。
    
    这里简化为每个空间单元一个自由度，空间算子L为对角阵。
    返回 (n_elem * n_pce, n_elem * n_pce) 块稀疏矩阵的稠密表示。
    """
    n_elem = spatial_A.shape[0]
    n_pce = pce_degree + 1
    N = n_elem * n_pce
    
    from pce_basis import build_pce_galerkin_matrix
    A_pce = build_pce_galerkin_matrix(pce_degree, alpha_mu, alpha_sigma)
    
    # 块积: A_total = I ⊗ A_pce + spatial_A ⊗ I
    # 简化为直接Kronecker积
    I_pce = np.eye(n_pce)
    I_spatial = np.eye(n_elem)
    
    A_total = np.kron(I_spatial, A_pce) + np.kron(spatial_A, I_pce)
    return A_total
