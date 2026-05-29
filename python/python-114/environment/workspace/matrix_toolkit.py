"""
matrix_toolkit.py
线性代数与矩阵运算工具模块

融合原项目:
  - 738_matrix_assemble_parfor: Hilbert矩阵组装 → 力场核矩阵组装
  - 740_matrix_chain_dynamic: 矩阵链动态规划 → 多体相互作用乘法顺序优化
  - 771_mm_to_msm: Matrix Market读取 → 稀疏力常数矩阵I/O
  - 979_r8gb: 带状矩阵LU分解 → 约束动力学线性求解

科学背景:
  在粗粒化分子动力学中，力场常可用核矩阵表示:
    K_{ij} = H_{ij} · f(|r_i - r_j|)
  其中 H_{ij} 为Hilbert型长程核 (1/(i+j))，f(r) 为截断函数。

  矩阵链乘法优化用于多体相互作用:
    计算 A1·A2·...·An 时，乘法顺序影响运算量。
    最优顺序通过动态规划求解，复杂度 O(n³)。
"""

import numpy as np
import io


def assemble_hilbert_kernel_matrix(m: int, n: int,
                                    cutoff_distance: float = 5.0,
                                    coords: np.ndarray = None) -> np.ndarray:
    """
    基于 matrix_assemble_parfor 思想的Hilbert核矩阵组装

    Hilbert矩阵:
        H_{ij} = 1 / (i + j - 1)

    在分子力学中，加入距离截断:
        K_{ij} = H_{ij} · exp(-|r_i - r_j|² / σ²)   if |r_i-r_j| < cutoff
               = 0                                    otherwise

    参数:
        m, n: 矩阵维度
        cutoff_distance: 截断距离 (nm)
        coords: 坐标数组 (可选) shape (max(m,n), 3)

    Returns:
        K: (m, n) 核矩阵
    """
    if m <= 0 or n <= 0:
        raise ValueError("Dimensions must be positive")

    K = np.zeros((m, n), dtype=float)
    sigma = cutoff_distance / 3.0  # 高斯衰减宽度

    for i in range(m):
        for j in range(n):
            h_ij = 1.0 / (i + j + 1.0)
            if coords is not None and coords.shape[0] > max(i, j):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < cutoff_distance:
                    K[i, j] = h_ij * np.exp(-dist ** 2 / (sigma ** 2))
                else:
                    K[i, j] = 0.0
            else:
                K[i, j] = h_ij

    return K


def matrix_chain_optimal_order(dims: list) -> tuple:
    """
    基于 matrix_chain_dynamic 的动态规划求解矩阵链最优乘法顺序

    参数:
        dims: list of int, 维度信息
              矩阵 A_i 的维度为 dims[i] × dims[i+1]

    Returns:
        min_cost: 最小标量乘法次数
        split_table: 最优分割表
    """
    n = len(dims) - 1
    if n < 1:
        return 0, []
    if any(d <= 0 for d in dims):
        raise ValueError("All dimensions must be positive")

    # m[i,j] = 计算 A[i..j] 的最小代价
    m = np.full((n, n), np.inf, dtype=float)
    s = np.zeros((n, n), dtype=int)

    for i in range(n):
        m[i, i] = 0.0

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            for k in range(i, j):
                cost = m[i, k] + m[k + 1, j] + dims[i] * dims[k + 1] * dims[j + 1]
                if cost < m[i, j]:
                    m[i, j] = cost
                    s[i, j] = k

    return int(m[0, n - 1]), s


def read_matrix_market_string(text: str) -> dict:
    """
    基于 mm_to_msm 思想的简化Matrix Market格式解析器

    支持 coordinate 格式的实数稀疏矩阵
    """
    lines = text.strip().splitlines()
    if not lines:
        raise ValueError("Empty matrix market data")

    header = lines[0].strip()
    parts = header.split()
    if len(parts) < 5 or parts[0] != '%%MatrixMarket' or parts[1] != 'matrix':
        raise ValueError("Invalid Matrix Market header")

    rep = parts[2].lower()
    field = parts[3].lower()
    symm = parts[4].lower()

    # 跳过注释
    idx = 1
    while idx < len(lines) and lines[idx].strip().startswith('%'):
        idx += 1

    if rep == 'coordinate':
        sizeinfo = [int(x) for x in lines[idx].strip().split()]
        idx += 1
        if len(sizeinfo) != 3:
            raise ValueError("Invalid size line for coordinate format")
        rows, cols, entries = sizeinfo

        row_idx = []
        col_idx = []
        data = []

        for e in range(entries):
            if idx >= len(lines):
                break
            vals = lines[idx].strip().split()
            idx += 1
            if len(vals) >= 3:
                row_idx.append(int(vals[0]) - 1)  # 1-based to 0-based
                col_idx.append(int(vals[1]) - 1)
                data.append(float(vals[2]))

        A = np.zeros((rows, cols), dtype=float)
        for r, c, d in zip(row_idx, col_idx, data):
            A[r, c] = d
            if symm == 'symmetric' and r != c:
                A[c, r] = d

        return {
            'A': A,
            'rows': rows,
            'cols': cols,
            'entries': len(data),
            'rep': rep,
            'field': field,
            'symm': symm
        }
    else:
        raise NotImplementedError("Only coordinate format is supported")


def write_matrix_market_string(A: np.ndarray, title: str = "force_constant") -> str:
    """
    将矩阵写入Matrix Market coordinate格式字符串
    """
    rows, cols = A.shape
    lines = [f"%%MatrixMarket matrix coordinate real general"]
    lines.append(f"% {title}")

    nnz = 0
    entries = []
    for i in range(rows):
        for j in range(cols):
            if abs(A[i, j]) > 1e-14:
                entries.append((i + 1, j + 1, A[i, j]))
                nnz += 1

    lines.append(f"{rows} {cols} {nnz}")
    for i, j, v in entries:
        lines.append(f"{i} {j} {v:.8e}")

    return '\n'.join(lines)


def solve_constraint_dynamics_banded(hessian: np.ndarray,
                                      gradient: np.ndarray,
                                      constraints: np.ndarray) -> np.ndarray:
    """
    使用带状矩阵求解约束动力学线性系统

    系统:
        [ H   C^T ] [ dx ]   [ -g ]
        [ C   0   ] [ λ  ] = [  0 ]

    其中 H 为Hessian，C 为约束Jacobian，g 为梯度
    """
    n = hessian.shape[0]
    m = constraints.shape[0] if constraints.ndim > 1 else 1

    if constraints.ndim == 1:
        constraints = constraints.reshape(1, -1)

    # 构建增广矩阵
    aug_size = n + m
    KKT = np.zeros((aug_size, aug_size), dtype=float)
    KKT[:n, :n] = hessian
    KKT[:n, n:n + m] = constraints.T
    KKT[n:n + m, :n] = constraints

    rhs = np.zeros(aug_size, dtype=float)
    rhs[:n] = -gradient

    try:
        sol = np.linalg.solve(KKT, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]

    return sol[:n]


def condition_number_estimate(A: np.ndarray) -> float:
    """
    估算矩阵条件数
    """
    s = np.linalg.svd(A, compute_uv=False)
    if s[-1] < 1e-15:
        return np.inf
    return s[0] / s[-1]
