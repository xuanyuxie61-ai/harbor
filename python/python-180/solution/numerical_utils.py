"""
numerical_utils.py
数值工具与线性代数辅助

融合种子项目:
  - 404_fem2d_heat_rectangle: 带状矩阵存储与求解 (LINPACK DGB 风格)
  - 907_praxis: 无梯度优化的辅助数值工具 (hypot, svd 排序)

科学背景:
  SPDE 隐式时间步进产生的大型稀疏/带状线性系统:
      (M + theta * dt * A) u^{n+1} = M u^n + dt * f + boundary terms
  其中 M 为质量矩阵，A 为刚度矩阵。采用 band storage 可显著降低存储量
  从 O(N^2) 到 O(N*(ml+mu+1))。
"""

import numpy as np
from typing import Tuple


def r8_hypot(x: float, y: float) -> float:
    """
    安全计算 sqrt(x^2 + y^2)，避免上溢/下溢。
    来自 praxis 项目 (r8_hypot.m)。
    """
    ax = abs(x)
    ay = abs(y)
    if ax < ay:
        ax, ay = ay, ax
    if ax == 0.0:
        return 0.0
    t = ay / ax
    return ax * np.sqrt(1.0 + t * t)


def band_solve(a_band: np.ndarray,
               ml: int,
               mu: int,
               b: np.ndarray) -> np.ndarray:
    """
    带状矩阵求解: a_band 为 LINPACK DGB 存储格式。

    存储映射:
        全矩阵 A(i,j) -> band[k, j],  k = i - j + ml + mu
        band 行数 = 2*ml + mu + 1
    """
    if a_band.ndim != 2 or b.ndim != 1:
        raise ValueError("Invalid dimensions")
    n = b.shape[0]
    rows = a_band.shape[0]
    if rows != 2 * ml + mu + 1:
        raise ValueError(f"Expected {2*ml+mu+1} rows, got {rows}")

    # 转为 scipy 稀疏或直接 dense solve 以简化
    # 为了鲁棒性，这里展开为 dense 矩阵后调用 numpy.linalg.solve
    A_full = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        i1 = max(0, j - mu)
        i2 = min(n - 1, j + ml)
        for i in range(i1, i2 + 1):
            k = i - j + ml + mu
            A_full[i, j] = a_band[k, j]

    # 边界鲁棒性：检查条件数
    cond_est = np.linalg.cond(A_full)
    if cond_est > 1e14:
        # 病态矩阵，使用伪逆
        x = np.linalg.lstsq(A_full, b, rcond=1e-14)[0]
    else:
        x = np.linalg.solve(A_full, b)
    return x


def assemble_band_storage(A_full: np.ndarray,
                          ml: int,
                          mu: int) -> np.ndarray:
    """
    将 full 矩阵转换为 LINPACK DGB 带状存储。
    """
    n = A_full.shape[0]
    a_band = np.zeros((2 * ml + mu + 1, n), dtype=np.float64)
    for j in range(n):
        i1 = max(0, j - mu)
        i2 = min(n - 1, j + ml)
        for i in range(i1, i2 + 1):
            k = i - j + ml + mu
            a_band[k, j] = A_full[i, j]
    return a_band


def svsort(n: int, d: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    按奇异值/特征值降序排序，同时重排向量。
    来自 praxis 项目 (svsort.m)。
    """
    if len(d) < n or v.shape[0] < n:
        raise ValueError("Dimensions mismatch in svsort")
    idx = np.argsort(-d[:n])
    d_out = d[:n].copy()
    v_out = v[:n, :n].copy()
    d_out = d_out[idx]
    v_out = v_out[:, idx]
    return d_out, v_out


def apply_dirichlet_bc(A: np.ndarray,
                       b: np.ndarray,
                       bc_nodes: np.ndarray,
                       bc_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    强施加 Dirichlet 边界条件。
    对于每个边界节点 i:
        A[i, :] = 0, A[:, i] = 0, A[i, i] = 1
        b[i] = bc_values[i]
    """
    A = A.copy()
    b = b.copy()
    for idx, val in zip(bc_nodes, bc_values):
        i = int(idx)
        A[i, :] = 0.0
        A[:, i] = 0.0
        A[i, i] = 1.0
        b[i] = val
    return A, b


def apply_neumann_bc_rhs(b: np.ndarray,
                         dx: float,
                         neumann_nodes: np.ndarray,
                         flux_values: np.ndarray) -> np.ndarray:
    """
    弱形式 Neumann 边界条件施加到 RHS:
        b[i] += flux * dx   (一维情形)
    """
    b = b.copy()
    for idx, flux in zip(neumann_nodes, flux_values):
        i = int(idx)
        b[i] += flux * dx
    return b
