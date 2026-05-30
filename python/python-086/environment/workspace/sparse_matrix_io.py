# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from typing import Tuple


def st_to_coo(st_data: str) -> coo_matrix:
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

    if min_idx == 1:
        rows -= 1
        cols -= 1
    elif min_idx != 0:
        raise ValueError(f"索引基值异常: {min_idx}")
    n = max(np.max(rows), np.max(cols)) + 1
    return coo_matrix((data, (rows, cols)), shape=(n, n))


def coo_to_st(mat: coo_matrix, one_based: bool = False) -> str:
    mat = mat.tocoo()
    lines = []
    offset = 1 if one_based else 0
    for i, j, v in zip(mat.row, mat.col, mat.data):
        lines.append(f"{i + offset} {j + offset} {v:.16e}")
    return "\n".join(lines)


def mm_to_coo(mm_data: str) -> coo_matrix:
    lines = mm_data.strip().splitlines()
    idx = 0

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
        rows.append(i - 1)
        cols.append(j - 1)
        data.append(v)
    return coo_matrix((data, (rows, cols)), shape=(nrows, ncols))


def coo_to_mm(mat: coo_matrix, symmetry: str = "general") -> str:
    mat = mat.tocoo()
    nrows, ncols = mat.shape
    nnz = mat.nnz
    lines = ["%%MatrixMarket matrix coordinate real general"]
    lines.append(f"{nrows} {ncols} {nnz}")
    for i, j, v in zip(mat.row, mat.col, mat.data):
        lines.append(f"{i + 1} {j + 1} {v:.16e}")
    return "\n".join(lines)


def st_to_hb(st_data: str, title: str = "SHELL", key: str = "SHELL") -> str:
    mat = st_to_coo(st_data).tocsr()
    nrows, ncols = mat.shape
    nnz = mat.nnz

    colptr = mat.indptr + 1
    rowind = mat.indices + 1
    values = mat.data

    lines = []
    lines.append(f"{title:<72}{key:<8}")
    lines.append(f"{'RUA':<3}             {nrows:14d}{ncols:14d}{nnz:14d}{0:14d}")

    def format_ints(arr, per_line=5):
        s = ""
        for i, v in enumerate(arr):
            if i % per_line == 0 and i > 0:
                s += "\n"
            s += f"{v:8d}"
        return s

    lines.append(format_ints(colptr, 5))
    lines.append(format_ints(rowind, 5))

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
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    mat = mat.tocsr()

    sym = mat + mat.T
    perm = reverse_cuthill_mckee(sym, symmetric_mode=True)
    return perm
