# -*- coding: utf-8 -*-
"""
sparse_matrix_io.py
稀疏矩阵格式转换与I/O工具

融合种子项目:
  - 771_mm_to_msm: Matrix Market 格式读取
  - 1157_st_to_hb: ST 格式到 Harwell-Boeing 格式转换
  - 1158_st_to_mm: ST 格式到 Matrix Market 格式转换

科学背景:
  壳体有限元刚度矩阵规模巨大但稀疏，通常采用 Coordinate (COO)
  或 Compressed Sparse Row (CSR) 格式存储。
  本模块提供多种稀疏矩阵格式的相互转换，便于与外部求解器
  (如 ARPACK, FEAST, PARDISO) 对接。
"""

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from typing import Tuple


def st_to_coo(st_data: str) -> coo_matrix:
    """
    从 ST (Sparse Triplet) 字符串解析为 COO 稀疏矩阵

    ST 格式: 每行 "i j value"，0-based 或 1-based 索引
    """
    rows = []
    cols = []
    data = []
    min_idx = float('inf')
    lines = st_data.strip().splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        i, j, v = int(parts[0]), int(parts[1]), float(parts[2])
        rows.append(i)
        cols.append(j)
        data.append(v)
        min_idx = min(min_idx, i, j)
    if not rows:
        raise ValueError("ST 数据为空")
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(data)
    # 检测 1-based 并转换为 0-based
    if min_idx == 1:
        rows -= 1
        cols -= 1
    elif min_idx != 0:
        raise ValueError(f"索引基值异常: {min_idx}")
    n = max(np.max(rows), np.max(cols)) + 1
    return coo_matrix((data, (rows, cols)), shape=(n, n))


def coo_to_st(mat: coo_matrix, one_based: bool = False) -> str:
    """
    将 COO 稀疏矩阵输出为 ST 格式字符串
    """
    mat = mat.tocoo()
    lines = []
    offset = 1 if one_based else 0
    for i, j, v in zip(mat.row, mat.col, mat.data):
        lines.append(f"{i + offset} {j + offset} {v:.16e}")
    return "\n".join(lines)


def mm_to_coo(mm_data: str) -> coo_matrix:
    """
    从 Matrix Market 格式字符串解析为 COO 稀疏矩阵

    Matrix Market 格式头:
      %%MatrixMarket matrix coordinate real general
      rows cols nnz
    """
    lines = mm_data.strip().splitlines()
    idx = 0
    # 跳过注释
    while idx < len(lines) and lines[idx].startswith('%'):
        idx += 1
    if idx >= len(lines):
        raise ValueError("Matrix Market 数据缺失维度行")
    header_parts = lines[idx].split()
    idx += 1
    if len(header_parts) < 3:
        raise ValueError("Matrix Market 维度行格式错误")
    nrows, ncols, nnz = int(header_parts[0]), int(header_parts[1]), int(header_parts[2])
    rows = []
    cols = []
    data = []
    for line in lines[idx:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        i, j, v = int(parts[0]), int(parts[1]), float(parts[2])
        rows.append(i - 1)  # MM 是 1-based
        cols.append(j - 1)
        data.append(v)
    return coo_matrix((data, (rows, cols)), shape=(nrows, ncols))


def coo_to_mm(mat: coo_matrix, symmetry: str = "general") -> str:
    """
    将 COO 稀疏矩阵输出为 Matrix Market 格式字符串
    """
    mat = mat.tocoo()
    nrows, ncols = mat.shape
    nnz = mat.nnz
    lines = ["%%MatrixMarket matrix coordinate real general"]
    lines.append(f"{nrows} {ncols} {nnz}")
    for i, j, v in zip(mat.row, mat.col, mat.data):
        lines.append(f"{i + 1} {j + 1} {v:.16e}")
    return "\n".join(lines)


def st_to_hb(st_data: str, title: str = "SHELL", key: str = "SHELL") -> str:
    """
    将 ST 格式转换为 Harwell-Boeing (HB) 格式字符串

    HB 格式为早期 Fortran 稀疏求解器标准格式，
    包含:
      - 标题行 (72字符标题 + 8字符密钥)
      - 维度行 (Totcrd, Ptrcrd, Indcrd, Valcrd, Rhscrd)
      - 矩阵类型行 (Mxtype, Nrow, Ncol, Nnzero, Neltvl)
      - 指针数组 (Colptr)
      - 行索引数组 (Rowind)
      - 数值数组 (Values)
    """
    mat = st_to_coo(st_data).tocsr()
    nrows, ncols = mat.shape
    nnz = mat.nnz
    # 构造指针和行索引
    colptr = mat.indptr + 1  # 1-based
    rowind = mat.indices + 1
    values = mat.data
    # 简化输出: 仅输出 header + 数据
    lines = []
    lines.append(f"{title:<72}{key:<8}")
    lines.append(f"{'RUA':<3}             {nrows:14d}{ncols:14d}{nnz:14d}{0:14d}")
    # 指针数组 (每行5个整数)
    def format_ints(arr, per_line=5):
        s = ""
        for i, v in enumerate(arr):
            if i % per_line == 0 and i > 0:
                s += "\n"
            s += f"{v:8d}"
        return s

    lines.append(format_ints(colptr, 5))
    lines.append(format_ints(rowind, 5))
    # 数值 (每行3个，科学计数法)
    def format_vals(arr, per_line=3):
        s = ""
        for i, v in enumerate(arr):
            if i % per_line == 0 and i > 0:
                s += "\n"
            s += f"{v:24.16e}"
        return s

    lines.append(format_vals(values, 3))
    return "\n".join(lines)


def coo_bandwidth(mat: coo_matrix) -> Tuple[int, int, int]:
    """
    计算稀疏矩阵的带宽 (基于 417_fem3d_pack 的 bandwidth_mesh 思想)

    下带宽 ML: 某行最左非零元到对角元的距离
    上带宽 MU: 某行最右非零元到对角元的距离
    半带宽 M = ML + 1 + MU

    Returns
    -------
    ml, mu, m : int
    """
    mat = mat.tocsr()
    n = mat.shape[0]
    ml = 0
    mu = 0
    for i in range(n):
        row_start = mat.indptr[i]
        row_end = mat.indptr[i + 1]
        if row_start == row_end:
            continue
        cols = mat.indices[row_start:row_end]
        left = np.min(cols)
        right = np.max(cols)
        ml = max(ml, i - left)
        mu = max(mu, right - i)
    return ml, mu, ml + 1 + mu


def matrix_profile_reduction(mat: coo_matrix) -> np.ndarray:
    """
    基于 Reverse Cuthill-McKee (RCM) 算法的矩阵轮廓缩减

    有限元中通过节点重编号可以显著降低刚度矩阵带宽，
    从而提高直接求解器效率。

    Returns
    -------
    perm : (n,) ndarray
        新节点到旧节点的排列
    """
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    mat = mat.tocsr()
    # 构造无向图 (对称化)
    sym = mat + mat.T
    perm = reverse_cuthill_mckee(sym, symmetric_mode=True)
    return perm
