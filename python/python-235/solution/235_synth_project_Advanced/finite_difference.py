"""
finite_difference.py — 高阶有限差分算子
========================================
种子项目映射:
  982_r8ge_np (LU分解) → 隐式格式线性系统求解
  002_advection_pde (PDE设置) → 差分模板参数化
  095_bisection_integer (二分法) → 自适应阶数选择

差分格式:
  2阶: (f_{i+1} - 2f_i + f_{i-1})/h²
  4阶: (-f_{i+2} + 16f_{i+1} - 30f_i + 16f_{i-1} - f_{i-2})/(12h²)
  6阶: (f_{i+3} - 12f_{i+2} + 150f_{i+1} - 280f_i + 150f_{i-1} - 12f_{i-2} + f_{i-3})/(180h²)
"""
import numpy as np


STENCILS = {
    2: (np.array([1.0, -2.0, 1.0]), np.array([-1, 0, 1])),
    4: (np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / 12.0,
        np.array([-2, -1, 0, 1, 2])),
    6: (np.array([1.0, -12.0, 150.0, -280.0, 150.0, -12.0, 1.0]) / 180.0,
        np.array([-3, -2, -1, 0, 1, 2, 3])),
}


def fd_second_deriv(f, h, order=4, boundary="periodic"):
    """
    计算二阶导数 D²f 的高阶有限差分近似.

    Parameters
    ----------
    f : ndarray, shape (N,)
    h : float, 格点间距
    order : int, 差分阶数 (2, 4, 6)
    boundary : str, "periodic" 或 "dirichlet"
    """
    if order not in STENCILS:
        raise ValueError(f"order={order} 不支持, 可选: {list(STENCILS.keys())}")

    stencil, offsets = STENCILS[order]
    N = len(f)
    result = np.zeros(N)

    for coeff, offset in zip(stencil, offsets):
        if boundary == "periodic":
            shifted = np.roll(f, -offset)
        else:
            shifted = np.zeros(N)
            if offset >= 0:
                shifted[:N - offset] = f[offset:]
            else:
                shifted[-offset:] = f[:N + offset]
        result += coeff * shifted

    return result / h**2


def build_diff_matrix(N, h, order=4, boundary="periodic"):
    """构建差分算子的稠密矩阵形式."""
    stencil, offsets = STENCILS.get(order, STENCILS[4])
    A = np.zeros((N, N))
    for i in range(N):
        for coeff, offset in zip(stencil, offsets):
            if boundary == "periodic":
                j = (i + offset) % N
            else:
                j = i + offset
                if j < 0 or j >= N:
                    continue
            A[i, j] += coeff
    return A / h**2


def lu_solve_no_pivot(A, b):
    """
    无主元LU分解求解 Ax=b.
    种子项目 982_r8ge_np 直接映射.
    仅适用于对角占优矩阵.
    """
    n = len(b)
    LU = A.copy().astype(float)
    x = b.copy().astype(float)

    # LU分解
    for k in range(n - 1):
        if abs(LU[k, k]) < 1e-30:
            raise ValueError(f"主元 LU[{k},{k}] 接近零, 需要选主元")
        for i in range(k + 1, n):
            LU[i, k] /= LU[k, k]
            LU[i, k + 1:] -= LU[i, k] * LU[k, k + 1:]

    # 前代 Ly = b
    for i in range(1, n):
        x[i] -= np.dot(LU[i, :i], x[:i])

    # 回代 Ux = y
    for i in range(n - 1, -1, -1):
        x[i] -= np.dot(LU[i, i + 1:], x[i + 1:])
        x[i] /= LU[i, i]

    return x


def modified_wavenumber(k, h, order=4):
    """
    修正波数 k̃(k): D² exp(ikx) = -k̃² exp(ikx).
    衡量差分的色散误差.
    """
    kh = k * h
    if order == 2:
        k_sq = 2.0 * (1.0 - np.cos(kh)) / h**2
    elif order == 4:
        k_sq = (30.0 - 32.0 * np.cos(kh) + 2.0 * np.cos(2 * kh)) / (12.0 * h**2)
    elif order == 6:
        k_sq = (210.0 - 270.0 * np.cos(kh) + 75.0 * np.cos(2 * kh)
                - 10.0 * np.cos(3 * kh)) / (180.0 * h**2)
    else:
        raise ValueError(f"order={order} 不支持")
    return np.sqrt(np.maximum(k_sq, 0.0))
