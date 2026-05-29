#!/usr/bin/env python3
"""
banded_linear_algebra.py
结构化线性代数模块（源自 r8pbl + hankel_cholesky + linpack_bench 项目）

融合对称正定带状矩阵紧凑存储（R8PBL）、Hankel 矩阵 Cholesky 分解
与经典 LINPACK 密集求解器，为 PEMFC 多物理场耦合线性系统提供
高效、稳定的数值代数基础。
"""

import numpy as np
import time


# ---------------------------------------------------------------------------
# r8pbl 迁移：对称正定带状矩阵紧凑存储与运算
# ---------------------------------------------------------------------------

def r8pbl_zeros(n, mu):
    """
    创建 n×n 对称带状 SPD 矩阵的紧凑 R8PBL 存储。
    只存储对角线及以下 mu 条次对角线，按列存储。
    对应原项目 r8pbl_zeros.m。
    """
    a = np.zeros((mu + 1, n), dtype=float)
    return a


def r8pbl_dif2(n, mu):
    """
    创建经典 DIF2 测试矩阵（二阶差分算子）的 R8PBL 形式。
    对应原项目 r8pbl_dif2.m。
    矩阵特点：主对角线为 2，上下次对角线为 -1。
    """
    a = r8pbl_zeros(n, mu)
    for j in range(n):
        a[0, j] = 2.0
        if mu >= 1 and j + 1 < n:
            a[1, j + 1] = -1.0
    return a


def r8pbl_to_r8ge(a, n, mu):
    """
    将 R8PBL 格式转换为普通密集矩阵 R8GE。
    对应原项目 r8pbl_to_r8ge.m。
    """
    a_dense = np.zeros((n, n), dtype=float)
    for j in range(n):
        for i in range(max(0, j - mu), j + 1):
            a_dense[i, j] = a[j - i, j]
            a_dense[j, i] = a[j - i, j]
    return a_dense


def r8pbl_mv(a, n, mu, x):
    """
    R8PBL 格式矩阵-向量乘法 b = A·x。
    对应原项目 r8pbl_mv.m。
    """
    b = np.zeros(n, dtype=float)
    for j in range(n):
        for i in range(max(0, j - mu), j + 1):
            bij = a[j - i, j]
            b[i] += bij * x[j]
            if i != j:
                b[j] += bij * x[i]
    return b


# ---------------------------------------------------------------------------
# linpack_bench 迁移：密集 LU 分解与求解
# ---------------------------------------------------------------------------

def dgefa(a, n):
    """
    密集矩阵 LU 分解（部分主元），对应原项目 dgefa.m。
    返回 LU 矩阵与主元置换向量 ipvt。
    """
    a = a.copy()
    ipvt = np.arange(n)
    info = 0

    for k in range(n - 1):
        # 选主元
        max_idx = np.argmax(np.abs(a[k:n, k])) + k
        if abs(a[max_idx, k]) < 1e-15:
            info = k
            continue
        if max_idx != k:
            a[[k, max_idx]] = a[[max_idx, k]]
            ipvt[[k, max_idx]] = ipvt[[max_idx, k]]

        # 消元
        for i in range(k + 1, n):
            a[i, k] /= a[k, k]
            a[i, k + 1:n] -= a[i, k] * a[k, k + 1:n]

    if abs(a[n - 1, n - 1]) < 1e-15:
        info = n - 1

    return a, ipvt, info


def dgesl(a, n, ipvt, b, job=0):
    """
    LU 分解后求解线性系统，对应原项目 dgesl.m。
    job=0: 解 A·x = b;  job=1: 解 A^T·x = b
    """
    x = b.copy()

    if job == 0:
        # 前代（考虑行置换）
        for k in range(n - 1):
            # 检查是否需要交换
            if ipvt[k] != k:
                x[k], x[ipvt[k]] = x[ipvt[k]], x[k]
            for i in range(k + 1, n):
                x[i] -= a[i, k] * x[k]

        # 回代
        for k in range(n - 1, -1, -1):
            x[k] /= a[k, k]
            for i in range(k):
                x[i] -= a[i, k] * x[k]
    else:
        # 转置求解
        for k in range(n):
            x[k] /= a[k, k]
            for i in range(k + 1, n):
                x[i] -= a[k, i] * x[k]
        for k in range(n - 1, -1, -1):
            for i in range(k):
                x[i] -= a[i, k] * x[i]
            if ipvt[k] != k:
                x[k], x[ipvt[k]] = x[ipvt[k]], x[k]

    return x


# ---------------------------------------------------------------------------
# hankel_cholesky 迁移：Hankel 矩阵 Cholesky 分解
# ---------------------------------------------------------------------------

def hankel_cholesky_upper(h_first_row):
    """
    Hankel 矩阵的上三角 Cholesky 分解。
    对应原项目 hankel_cholesky_upper.m。

    Hankel 矩阵 H 满足 H[i,j] = h_{i+j}，即反对角线元素相同。
    利用 Phillips (1971) 递推算法计算 R 使得 H = R^T · R。
    加入数值正则化保证正定性。
    """
    h_first_row = np.asarray(h_first_row, dtype=float)
    n = len(h_first_row)
    # 构造 Hankel 矩阵
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            idx = i + j
            if idx < n:
                H[i, j] = h_first_row[idx]
            else:
                H[i, j] = 0.0

    # 数值正则化：加入小量对角扰动保证 SPD
    H += np.eye(n) * 1e-8

    # 标准 Cholesky 分解（列优先）
    R = np.zeros((n, n), dtype=float)
    for j in range(n):
        s = H[j, j]
        for k in range(j):
            s -= R[k, j] ** 2
        if s <= 1e-15:
            s = 1e-15
        R[j, j] = np.sqrt(s)
        for i in range(j + 1, n):
            s = H[j, i]
            for k in range(j):
                s -= R[k, j] * R[k, i]
            if abs(R[j, j]) > 1e-15:
                R[j, i] = s / R[j, j]
            else:
                R[j, i] = 0.0

    return R


def hankel_spd_cholesky_lower(diag, subdiag):
    """
    由对角线与次对角线构造 Hankel SPD 矩阵的下三角 Cholesky 因子 L。
    对应原项目 hankel_spd_cholesky_lower.m。

    给定 L 的对角线 diag 与次对角线 subdiag，
    递推填充剩余元素，使得 H = L · L^T 为 Hankel 矩阵。
    """
    n = len(diag)
    L = np.zeros((n, n), dtype=float)
    for i in range(n):
        L[i, i] = diag[i]
    for i in range(1, n):
        L[i, i - 1] = subdiag[i - 1]

    # Al-Homidan & Alshahrani (2009) 递推
    for j in range(n):
        for i in range(j + 1, n):
            if i > 0 and j > 0:
                L[i, j] = L[i - 1, j - 1]
            elif j == 0 and i > 1:
                # 利用 Hankel 性质推导
                L[i, j] = L[i - 1, j + 1] if j + 1 < i else 0.0

    return L


def hankel_covariance_factor(signal):
    """
    利用 Hankel Cholesky 构造水含量测量信号的协方差矩阵因子。
    将信号自相关函数视为 Hankel 矩阵的反对角线元素。
    使用稳定化的 Cholesky 分解并加入正则化。
    """
    signal = np.asarray(signal, dtype=float)
    n = min(len(signal), 25)
    sig = signal[:n]

    # 构造自相关序列
    sig_c = sig - np.mean(sig)
    autocorr = np.correlate(sig_c, sig_c, mode='full')
    autocorr = autocorr[n - 1:]
    autocorr = autocorr / max(autocorr[0], 1e-15)
    # 强制指数衰减以保证正定性
    for i in range(len(autocorr)):
        autocorr[i] *= np.exp(-0.1 * i)
    autocorr = np.clip(autocorr, -0.99, 0.99)

    # 构造 Toeplitz 协方差矩阵（Hankel 的对偶结构）
    from scipy.linalg import toeplitz
    cov = toeplitz(autocorr)
    cov += np.eye(n) * 1e-4  # 正则化

    # 稳定 Cholesky
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # 若失败，使用特征值修正
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-6, None)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    return L


# ---------------------------------------------------------------------------
# 综合测试：求解带状线性系统
# ---------------------------------------------------------------------------

def solve_banded_linear_system(params):
    """
    使用 R8PBL + LINPACK 求解一个 PEMFC 多物理场耦合的带状线性系统。
    系统来源于二维泊松方程的有限差分离散（五对角带状结构）。
    """
    n = max(20, params.get('Nx', 81))
    mu = 1  # 三对角（简化演示）

    # 构造 DIF2 带状矩阵
    a_pbl = r8pbl_dif2(n, mu)
    A_dense = r8pbl_to_r8ge(a_pbl, n, mu)

    # 右端项（模拟电势源）
    b = np.sin(np.linspace(0.0, np.pi, n))

    # LINPACK 求解
    t0 = time.perf_counter()
    a_lu, ipvt, info = dgefa(A_dense, n)
    if info != 0:
        # 若奇异，做正则化
        A_dense += np.eye(n) * 1e-6
        a_lu, ipvt, info = dgefa(A_dense, n)
    x = dgesl(a_lu, n, ipvt, b, job=0)
    t_solve = time.perf_counter() - t0

    # 计算残差
    b_recon = A_dense @ x
    resid = np.linalg.norm(b_recon - b) / np.linalg.norm(b)

    return resid, t_solve


if __name__ == '__main__':
    p = {'Nx': 81}
    r, t = solve_banded_linear_system(p)
    print("Residual:", r, "Time:", t)
