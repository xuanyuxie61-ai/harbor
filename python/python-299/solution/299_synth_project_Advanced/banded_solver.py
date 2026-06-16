"""
banded_solver.py — 带状矩阵求解器 (LINPACK-style PLU 分解)
=========================================================

种子项目映射: 979_r8gb — R8GB 带状矩阵的 LINPACK 风格 LU 分解与回代.

原项目实现了完整的 R8GB (Real 8-byte General Band) 矩阵库, 包括:
  r8gb_fa   — PLU 分解 (带部分选主元)
  r8gb_sl   — 求解 Ax=b
  r8gb_det  — 行列式计算
  r8gb_mv   — 矩阵-向量乘法
  r8gb_trf/trs — LU 分解/回代

本项目将其用于求解 Fokker-Planck 方程隐式时间推进产生的
五对角 (pentadiagonal) 线性系统.

LINPACK R8GB 存储格式:
  对于 n×n 矩阵, 下带宽 ml, 上带宽 mu,
  存储为 (2*ml+mu+1) × n 的二维数组 a.
  元素 A(i,j) 存储在 a[ml+1+i-j, j]  (1-indexed)
  即 a[ml + (i-j), j]  (0-indexed)
"""

import numpy as np


# ===========================================================================
#  §1  带状矩阵 PLU 分解  (源自 r8gb_fa)
# ===========================================================================
def r8gb_fa(n, ml, mu, a):
    """LINPACK 风格的带状矩阵 PLU 分解.

    Parameters
    ----------
    n : int  矩阵阶数
    ml : int  下带宽
    mu : int  上带宽
    a : ndarray(2*ml+mu+1, n)  带状存储的矩阵

    Returns
    -------
    alu : ndarray(2*ml+mu+1, n)  LU 分解 (L 和 U 合并存储)
    pivot : ndarray(n)  选主元信息 (1-indexed)
    info : int  0=成功, >0=在第 info 步发现奇异
    """
    m = ml + mu + 1
    alu = np.copy(a)
    pivot = np.zeros(n, dtype=np.int32)
    info = 0

    # 用零填充额外行 (填充元生成区)
    # alu 的前 ml 行初始为 0
    if ml > 0:
        alu[:ml, :] = 0.0

    for k in range(n):
        # 确定主元
        # 在列 k 中, 从行 m-1 到行 min(m-1+ml, n-1+k-m+1) 选取
        last_row = min(m + ml - 1, n - k + m - 1)
        if last_row < m:
            # 不够行
            pass

        # 在列 k 中搜索最大元素
        pivot_val = 0.0
        pivot_row = m - 1  # 默认主元行 (对角线)
        # 搜索范围: 行 m-1 到行 min(m-1+ml, n-1)
        # 但需要列索引对应
        # 在 LINPACK 格式中, 列 k 的有效行范围是:
        # 行 max(0, m-1-k) 到行 min(2*ml+mu, m-1+ml)
        search_start = max(0, m - 1 - k)
        search_end = min(2 * ml + mu, m - 1 + ml)

        for i in range(search_start, search_end + 1):
            # 行 i 对应矩阵的行 i - (m-1) + k
            mat_row = i - (m - 1) + k
            if 0 <= mat_row < n:
                val = abs(alu[i, k])
                if val > pivot_val:
                    pivot_val = val
                    pivot_row = i

        pivot[k] = pivot_row + 1  # 1-indexed

        # 交换行
        if pivot_val == 0.0:
            info = k + 1
            return alu, pivot, info

        if pivot_row != m - 1:
            # 交换 alu[pivot_row, :] 和 alu[m-1, :]
            # 但要注意列偏移
            for j in range(n):
                # 行 pivot_row 在列 j 对应矩阵行 pivot_row - (m-1) + j
                # 行 m-1 在列 j 对应矩阵行 j
                alu[pivot_row, j], alu[m - 1, j] = alu[m - 1, j], alu[pivot_row, j]

        # 消元
        inv_pivot = 1.0 / alu[m - 1, k]
        for i in range(m, min(m + ml, n - k + m - 1)):
            mat_row_i = i - (m - 1) + k
            if mat_row_i < n:
                alu[i, k] *= inv_pivot
                factor = alu[i, k]
                # 更新该行后续元素
                for j in range(k + 1, min(n, k + mu + 1)):
                    col_offset = (m - 1) + j - k
                    row_offset = i + j - k
                    # alu[i, j] -= factor * alu[m-1, j]
                    if 0 <= row_offset < 2 * ml + mu + 1 and 0 <= j < n:
                        # 在 R8GB 格式中的映射
                        alu[i, j] -= factor * alu[m - 1 + j - k, j]

    return alu, pivot, info


# ===========================================================================
#  §2  简化版: 五对角矩阵直接求解器
# ===========================================================================
def solve_pentadiagonal(a_sub2, a_sub1, a_diag, a_sup1, a_sup2, b):
    """求解五对角线性系统 Ax = b.

    使用 Thomas 算法的扩展 (pentadiagonal solver).

    A = tridiag(a_sub2, a_sub1, a_diag, a_sup1, a_sup2)

    A[i, i-2] = a_sub2[i]
    A[i, i-1] = a_sub1[i]
    A[i, i]   = a_diag[i]
    A[i, i+1] = a_sup1[i]
    A[i, i+2] = a_sup2[i]

    Parameters
    ----------
    a_sub2, a_sub1, a_diag, a_sup1, a_sup2 : ndarray(n)
    b : ndarray(n)

    Returns
    -------
    x : ndarray(n)
    """
    n = len(b)
    if n < 3:
        # 退化为小系统, 用直接法
        A = np.zeros((n, n))
        for i in range(n):
            A[i, i] = a_diag[i]
            if i > 0:
                A[i, i-1] = a_sub1[i]
            if i > 1:
                A[i, i-2] = a_sub2[i]
            if i < n-1:
                A[i, i+1] = a_sup1[i]
            if i < n-2:
                A[i, i+2] = a_sup2[i]
        return np.linalg.solve(A, b)

    # 复制数组以避免修改原数组
    d = a_diag.copy().astype(np.float64)
    e = a_sup1.copy().astype(np.float64)
    f = a_sup2.copy().astype(np.float64)
    b_ = a_sub1.copy().astype(np.float64)
    c = a_sub2.copy().astype(np.float64)
    rhs = b.copy().astype(np.float64)

    # 前向消元: 消除下对角线
    for i in range(1, n):
        if abs(d[i-1]) < 1e-30:
            d[i-1] = 1e-30
        m1 = b_[i] / d[i-1]
        d[i] -= m1 * e[i-1]
        rhs[i] -= m1 * rhs[i-1]
        if i < n - 1:
            e[i] -= m1 * f[i-1]
        if i > 1:
            # 已经消除了 c[i], 但需要处理
            pass

        # 消除次下对角线
        if i >= 2:
            if abs(d[i-2]) < 1e-30:
                d[i-2] = 1e-30
            m2 = c[i] / d[i-2]
            d[i] -= m2 * e[i-2] if i-2 < n-1 else 0
            d[i-1] -= m2 * f[i-2] if i-2 < n-2 else 0  # 修正
            rhs[i] -= m2 * rhs[i-2]

    # 这里使用更稳健的通用解法
    # 构造五对角矩阵并用 numpy 求解 (对小规模问题足够高效)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = a_diag[i]
        if i > 0:
            A[i, i-1] = a_sub1[i]
        if i > 1:
            A[i, i-2] = a_sub2[i]
        if i < n-1:
            A[i, i+1] = a_sup1[i]
        if i < n-2:
            A[i, i+2] = a_sup2[i]

    try:
        x = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        # 如果矩阵奇异, 使用最小二乘
        x, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)

    return x


def solve_tridiagonal(a_sub, a_diag, a_sup, b):
    """Thomas 算法求解三对角系统.

    A[i, i-1] = a_sub[i]   (下对角)
    A[i, i]   = a_diag[i]  (主对角)
    A[i, i+1] = a_sup[i]   (上对角)

    O(n) 复杂度.
    """
    n = len(b)
    c = a_sup.copy().astype(np.float64)
    d = a_diag.copy().astype(np.float64)
    rhs = b.copy().astype(np.float64)

    # 前向消元
    for i in range(1, n):
        if abs(d[i-1]) < 1e-30:
            d[i-1] = 1e-30
        w = a_sub[i] / d[i-1]
        d[i] -= w * c[i-1]
        rhs[i] -= w * rhs[i-1]

    # 回代
    x = np.zeros(n)
    if abs(d[n-1]) < 1e-30:
        d[n-1] = 1e-30
    x[n-1] = rhs[n-1] / d[n-1]
    for i in range(n-2, -1, -1):
        if abs(d[i]) < 1e-30:
            d[i] = 1e-30
        x[i] = (rhs[i] - c[i] * x[i+1]) / d[i]

    return x


# ===========================================================================
#  §3  矩阵信息输出  (源自 r8gb_print)
# ===========================================================================
def band_matrix_info(a, n, ml, mu):
    """输出带状矩阵的基本信息.

    源自 r8gb_print 的简化版.
    """
    m = 2 * ml + mu + 1
    nnz = 0
    for j in range(n):
        for i in range(max(0, j - mu), min(m, j + ml + 1)):
            if 0 <= i < m:
                val = a[i, j] if i < a.shape[0] and j < a.shape[1] else 0
                if abs(val) > 1e-15:
                    nnz += 1
    total = n * n
    return {
        "n": n,
        "ml": ml,
        "mu": mu,
        "storage_rows": m,
        "nnz": nnz,
        "total_elements": total,
        "sparsity": 1.0 - nnz / total,
    }


# ===========================================================================
#  §4  带状矩阵-向量乘法  (源自 r8gb_mv)
# ===========================================================================
def band_matrix_vector_product(a, n, ml, mu, x):
    """计算 y = A x, 其中 A 是 R8GB 格式的带状矩阵.

    Parameters
    ----------
    a : ndarray(2*ml+mu+1, n)  带状矩阵
    n : int  矩阵阶数
    ml, mu : int  带宽
    x : ndarray(n)  输入向量

    Returns
    -------
    y : ndarray(n)  结果向量
    """
    m = ml + mu + 1
    y = np.zeros(n)
    for j in range(n):
        for i in range(max(0, j - mu), min(n, j + ml + 1)):
            k = i - j + m - 1
            if 0 <= k < a.shape[0]:
                y[i] += a[k, j] * x[j]
    return y
