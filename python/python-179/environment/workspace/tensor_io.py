"""
tensor_io.py
张量数据读写模块（Matrix Market 风格扩展）
==========================================
对应原项目 769_mm_io 的矩阵市场格式 I/O，扩展至高阶张量坐标存储。
支持将稠密张量以坐标 (i1,i2,...,id, value) 形式读写，兼容稀疏切片。
"""

import numpy as np
from typing import Tuple, List


# ---------------------------------------------------------------------------
# 张量 ↔ 坐标格式转换
# ---------------------------------------------------------------------------

def tensor_to_coordinate(tensor: np.ndarray, tol: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    将稠密 d 阶张量转换为坐标格式 (indices, values)。

    参数
    ----
    tensor : np.ndarray, shape (n1, n2, ..., nd)
        输入稠密张量 A。
    tol : float
        零值阈值；仅保留 |A[i1,...,id]| > tol 的项。

    返回
    ----
    indices : np.ndarray, shape (nnz, d)
        每项的非零坐标，每行为 (i1, i2, ..., id)，0-based。
    values  : np.ndarray, shape (nnz,)
        对应非零值。

    数学背景
    --------
    坐标格式 (COO) 是稀疏张量的标准存储方式。对于 d 阶张量 A∈ℝ^{n1×...×nd}，
    其 Frobenius 范数定义为
        ‖A‖_F = sqrt( Σ_{i1,...,id} A_{i1...id}² )
    坐标格式在 nnz << ∏ nk 时显著节省存储。
    """
    tensor = np.asarray(tensor)
    shape = tensor.shape
    d = len(shape)
    flat = tensor.ravel()
    mask = np.abs(flat) > tol
    values = flat[mask]
    # 将扁平索引转换为多维坐标
    flat_idx = np.flatnonzero(mask)
    indices = np.zeros((values.size, d), dtype=int)
    tmp = flat_idx.copy()
    strides = [np.prod(shape[k+1:], dtype=np.int64) for k in range(d)]
    for k in range(d):
        indices[:, k] = tmp // strides[k]
        tmp = tmp % strides[k]
    return indices, values


def coordinate_to_tensor(indices: np.ndarray, values: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    将坐标格式还原为稠密张量。
    """
    tensor = np.zeros(shape, dtype=float)
    if indices.size == 0:
        return tensor
    # 使用 np.ravel_multi_index 将多维索引转为一维
    flat_idx = np.ravel_multi_index(indices.T, shape, mode='clip')
    np.add.at(tensor.ravel(), flat_idx, values)
    return tensor


# ---------------------------------------------------------------------------
# 文件 I/O（类 Matrix Market 头部 + 坐标体）
# ---------------------------------------------------------------------------

def write_tensor_mm(path: str, tensor: np.ndarray, tol: float = 0.0):
    """
    将张量写入类 Matrix Market 文件。
    头部格式:
        %%TensorMarket tensor coordinate real general
        % <comment lines>
        d  n1 n2 ... nd  nnz
        i1 i2 ... id   value
    """
    tensor = np.asarray(tensor)
    shape = tensor.shape
    d = len(shape)
    indices, values = tensor_to_coordinate(tensor, tol=tol)
    nnz = values.size
    with open(path, 'w', encoding='utf-8') as f:
        f.write("%%TensorMarket tensor coordinate real general\n")
        f.write(f"% Order={d}  Shape={'x'.join(str(s) for s in shape)}\n")
        f.write(f"{d}  {' '.join(str(s) for s in shape)}  {nnz}\n")
        for idx, val in zip(indices, values):
            idx_str = ' '.join(str(i + 1) for i in idx)  # 1-based
            f.write(f"{idx_str}  {val:.16e}\n")


def read_tensor_mm(path: str) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """
    从类 Matrix Market 文件读取张量。
    返回 (tensor, shape)。
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 跳过注释行
    data_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith('%')]
    if not data_lines:
        raise ValueError("Empty or comment-only file.")
    header = data_lines[0].split()
    d = int(header[0])
    shape = tuple(int(x) for x in header[1:1+d])
    nnz = int(header[1+d])
    indices = []
    values = []
    for line in data_lines[1:]:
        parts = line.split()
        if len(parts) < d + 1:
            continue
        idx = [int(p) - 1 for p in parts[:d]]  # 转 0-based
        val = float(parts[d])
        indices.append(idx)
        values.append(val)
    indices = np.array(indices, dtype=int)
    values = np.array(values, dtype=float)
    tensor = coordinate_to_tensor(indices, values, shape)
    return tensor, shape


# ---------------------------------------------------------------------------
# 对称张量半存储（类 Matrix Market symmetry 扩展）
# ---------------------------------------------------------------------------

def write_symmetric_tensor_mm(path: str, tensor: np.ndarray):
    """
    对于对称矩阵（2阶张量），仅存储下三角部分。
    对应原 mm_io 中 symmetric 的处理思想。
    """
    tensor = np.asarray(tensor)
    if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError("Symmetric storage only for square matrices.")
    n = tensor.shape[0]
    indices = []
    values = []
    for i in range(n):
        for j in range(i + 1):
            indices.append([i, j])
            values.append(tensor[i, j])
    indices = np.array(indices, dtype=int)
    values = np.array(values, dtype=float)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("%%TensorMarket matrix coordinate real symmetric\n")
        f.write(f"2  {n} {n}  {len(values)}\n")
        for idx, val in zip(indices, values):
            f.write(f"{idx[0]+1} {idx[1]+1}  {val:.16e}\n")


def read_symmetric_tensor_mm(path: str) -> np.ndarray:
    """
    读取对称矩阵并还原完整矩阵。
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith('%')]
    header = data_lines[0].split()
    n = int(header[1])
    tensor = np.zeros((n, n), dtype=float)
    for line in data_lines[1:]:
        parts = line.split()
        i, j = int(parts[0]) - 1, int(parts[1]) - 1
        val = float(parts[2])
        tensor[i, j] = val
        if i != j:
            tensor[j, i] = val
    return tensor
