
import numpy as np
from typing import Tuple, List






def tensor_to_coordinate(tensor: np.ndarray, tol: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    tensor = np.asarray(tensor)
    shape = tensor.shape
    d = len(shape)
    flat = tensor.ravel()
    mask = np.abs(flat) > tol
    values = flat[mask]

    flat_idx = np.flatnonzero(mask)
    indices = np.zeros((values.size, d), dtype=int)
    tmp = flat_idx.copy()
    strides = [np.prod(shape[k+1:], dtype=np.int64) for k in range(d)]
    for k in range(d):
        indices[:, k] = tmp // strides[k]
        tmp = tmp % strides[k]
    return indices, values


def coordinate_to_tensor(indices: np.ndarray, values: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    tensor = np.zeros(shape, dtype=float)
    if indices.size == 0:
        return tensor

    flat_idx = np.ravel_multi_index(indices.T, shape, mode='clip')
    np.add.at(tensor.ravel(), flat_idx, values)
    return tensor






def write_tensor_mm(path: str, tensor: np.ndarray, tol: float = 0.0):
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
            idx_str = ' '.join(str(i + 1) for i in idx)
            f.write(f"{idx_str}  {val:.16e}\n")


def read_tensor_mm(path: str) -> Tuple[np.ndarray, Tuple[int, ...]]:
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

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
        idx = [int(p) - 1 for p in parts[:d]]
        val = float(parts[d])
        indices.append(idx)
        values.append(val)
    indices = np.array(indices, dtype=int)
    values = np.array(values, dtype=float)
    tensor = coordinate_to_tensor(indices, values, shape)
    return tensor, shape






def write_symmetric_tensor_mm(path: str, tensor: np.ndarray):
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
