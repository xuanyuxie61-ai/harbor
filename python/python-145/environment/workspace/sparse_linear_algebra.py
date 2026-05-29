"""
sparse_linear_algebra.py
========================
博士级稀疏线性代数：三元组格式矩阵 I/O 与稀疏求解器封装

本模块实现稀疏矩阵的三元组（Coordinate / ST）格式操作，
用于处理大规模有限元离散化产生的稀疏线性系统。

数学背景
--------
对于 N × N 稀疏矩阵 A，三元组格式存储为三个等长数组:
    I[k] : 第 k 个非零元的行索引
    J[k] : 第 k 个非零元的列索引
    V[k] : 第 k 个非零元的数值

该格式是 FEM、有限差分与图论算法的标准中间表示。

在金融工程中，大型期限结构模型的状态空间维度可达 10^4 ~ 10^6，
稀疏矩阵存储将内存从 O(N^2) 降至 O(nnz)，其中 nnz 为非零元个数。

操作:
  1. 读取 / 写入 ST 格式文件
  2. COO -> CSR 转换
  3. 稀疏矩阵向量乘法
  4. 稀疏直接求解（LU 分解）
  5. 带宽估计与重排序
"""

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve, splu


def st_to_coo(ist, jst, ast, shape):
    """
    将三元组格式转换为 scipy COO 稀疏矩阵。

    Parameters
    ----------
    ist : np.ndarray
        行索引（0-based 或 1-based 均可自动检测）。
    jst : np.ndarray
        列索引。
    ast : np.ndarray
        非零值。
    shape : tuple (m, n)
        矩阵形状。

    Returns
    -------
    scipy.sparse.coo_matrix
    """
    ist = np.asarray(ist, dtype=int)
    jst = np.asarray(jst, dtype=int)
    ast = np.asarray(ast, dtype=float)

    # 自动检测 1-based 索引
    if ist.min() == 1 or jst.min() == 1:
        ist = ist - 1
        jst = jst - 1

    # 边界裁剪
    ist = np.clip(ist, 0, shape[0] - 1)
    jst = np.clip(jst, 0, shape[1] - 1)

    return sp.coo_matrix((ast, (ist, jst)), shape=shape)


def coo_to_st(mat):
    """
    将 COO 稀疏矩阵转换为三元组格式。

    Parameters
    ----------
    mat : scipy.sparse.coo_matrix

    Returns
    -------
    ist, jst, ast : np.ndarray
        0-based 索引的三元组。
    """
    mat = mat.tocoo()
    return mat.row, mat.col, mat.data


def write_st_file(filename, m, n, nst, ist, jst, ast):
    """
    将三元组矩阵写入文本文件。

    格式:
        每行:  i  j  value
    头部注释包含矩阵维度与非零元个数。

    Parameters
    ----------
    filename : str
        输出文件名。
    m, n : int
        矩阵维度。
    nst : int
        非零元个数。
    ist, jst : np.ndarray
        索引（0-based，输出时转为 1-based 以兼容部分传统格式）。
    ast : np.ndarray
        数值。
    """
    with open(filename, 'w') as f:
        f.write(f"# ST sparse matrix: {m} x {n}, nnz={nst}\n")
        for k in range(nst):
            f.write(f"{ist[k]+1:8d}  {jst[k]+1:8d}  {ast[k]:20.12e}\n")


def read_st_file(filename):
    """
    从文本文件读取三元组矩阵。

    Parameters
    ----------
    filename : str
        输入文件名。

    Returns
    -------
    m, n : int
        矩阵维度（从注释解析，若失败则从数据推断）。
    nst : int
        非零元个数。
    ist, jst : np.ndarray
        行、列索引（0-based）。
    ast : np.ndarray
        数值。
    """
    rows = []
    with open(filename, 'r') as f:
        header_m, header_n, header_nnz = None, None, None
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                # 尝试从注释头解析维度
                if 'x' in line and 'nnz' in line:
                    parts = line.replace('#', '').split(',')
                    for part in parts:
                        if 'x' in part and 'nnz' not in part:
                            dim_part = part.strip().split('x')
                            if len(dim_part) == 2:
                                try:
                                    header_m = int(dim_part[0].strip().split()[-1])
                                    header_n = int(dim_part[1].strip().split()[0])
                                except ValueError:
                                    pass
                        elif 'nnz' in part:
                            try:
                                header_nnz = int(part.split('=')[1].strip())
                            except (ValueError, IndexError):
                                pass
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    i = int(parts[0])
                    j = int(parts[1])
                    v = float(parts[2])
                    rows.append((i, j, v))
                except ValueError:
                    continue

    if len(rows) == 0:
        raise ValueError(f"read_st_file: 文件 {filename} 中未找到有效数据")

    ist = np.array([r[0] for r in rows], dtype=int)
    jst = np.array([r[1] for r in rows], dtype=int)
    ast = np.array([r[2] for r in rows], dtype=float)

    # 自动检测 1-based 索引
    if ist.min() == 1 or jst.min() == 1:
        ist = ist - 1
        jst = jst - 1

    nst = len(rows)
    if header_m is not None and header_n is not None:
        m, n = header_m, header_n
    else:
        m = ist.max() + 1
        n = jst.max() + 1

    return m, n, nst, ist, jst, ast


def estimate_bandwidth(A):
    """
    估计稀疏矩阵的半带宽。

    定义:
        bandwidth = max_{A[i,j] != 0} |i - j|

    Parameters
    ----------
    A : scipy.sparse.spmatrix

    Returns
    -------
    int
        半带宽。
    """
    A = A.tocoo()
    if A.nnz == 0:
        return 0
    return int(np.max(np.abs(A.row - A.col)))


def sparse_matvec(A, x):
    """
    稀疏矩阵-向量乘法，带维度检查。

    Parameters
    ----------
    A : scipy.sparse.spmatrix
    x : np.ndarray

    Returns
    -------
    np.ndarray
    """
    x = np.asarray(x, dtype=float)
    if x.shape[0] != A.shape[1]:
        raise ValueError("sparse_matvec: 维度不匹配")
    return A @ x


def solve_sparse_system(A, b, use_lu=False):
    """
    求解稀疏线性系统 Ax = b。

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        系数矩阵。
    b : np.ndarray
        右端项。
    use_lu : bool
        是否使用 LU 分解（适合多次求解相同矩阵）。

    Returns
    -------
    x : np.ndarray
        解向量。
    info : dict
        求解信息。
    """
    b = np.asarray(b, dtype=float)
    if b.shape[0] != A.shape[1]:
        raise ValueError("solve_sparse_system: 维度不匹配")

    info = {'method': 'spsolve', 'nnz': A.nnz, 'shape': A.shape}

    if use_lu:
        lu = splu(A.tocsc())
        x = lu.solve(b)
        info['method'] = 'splu'
        info['fill_factor'] = lu.nnz / max(1, A.nnz)
    else:
        x = spsolve(A, b)

    # 残差检验
    if x is not None:
        residual = np.linalg.norm(A @ x - b)
        info['residual'] = residual
    else:
        info['residual'] = np.inf

    return x, info
