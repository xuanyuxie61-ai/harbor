"""
matrix_solvers.py
=================
海洋数值模拟中的稀疏线性代数工具箱，包含：
- 双共轭梯度法 (BiCG) 求解大型稀疏线性系统
- 对称 Toeplitz 矩阵的快速求逆与乘法（用于时空相关分析）
- 非对称稀疏矩阵 (NCF) 存储格式与运算
- 泊松方程的五点差分格式矩阵生成

数学基础
--------
1. BiCG 算法（Barrett et al., 1994）：
   对非对称矩阵 A，同时构造两个 Krylov 子空间：
       K_m(A, r0)  与  K_m(A^T, r̃0)
   迭代格式：
       x_{k+1} = x_k + α_k p_k
       r_{k+1} = r_k - α_k A p_k
       r̃_{k+1} = r̃_k - α_k A^T p̃_k
   其中 α_k = (r_k, r̃_k) / (A p_k, p̃_k)

2. 对称 Toeplitz 矩阵逆（Gohberg-Semencul 公式）：
   对正定对称 Toeplitz 矩阵 T_n，其逆可由第一行向量 v 的 Yule-Walker 解显式表达：
       (T_n^{-1})_{ij} = (1/v_n)[ v_{n+1-i}v_{n+1-j} - v_{i-1}v_{j-1} ]
   其中 v 满足 Yule-Walker 方程 T_{n-1} v = -a。

3. NCF (NSPCG Coordinate Format) 稀疏矩阵：
   以 (row, col, val) 三元组存储非零元，适用于非对称稀疏结构。
"""

import numpy as np


# ---------------------------------------------------------------------------
# BiCG 求解器（源自 085_bicg）
# ---------------------------------------------------------------------------

def bicg_solve(A, b, x0=None, tol=1e-8, max_iter=1000):
    """
    双共轭梯度法求解 A x = b。

    参数
    ----
    A : ndarray (n, n) 或 scipy.sparse 兼容矩阵
    b : ndarray (n,)
    x0 : ndarray (n,), 初始猜测
    tol : float
        相对残差容差
    max_iter : int
        最大迭代次数

    返回
    ----
    x : ndarray (n,)
    """
    n = b.shape[0]
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    bnrm = np.linalg.norm(b)
    if bnrm == 0.0:
        bnrm = 1.0

    r = b - A.dot(x)
    error = np.linalg.norm(r) / bnrm
    if error < tol:
        return x

    r_tld = r.copy()
    rho_old = 1.0
    p = np.zeros(n)
    p_tld = np.zeros(n)

    for it in range(1, max_iter + 1):
        z = r.copy()
        z_tld = r_tld.copy()
        rho = np.dot(z, r_tld)
        if abs(rho) < 1e-30:
            break

        if it == 1:
            p = z
            p_tld = z_tld
        else:
            beta = rho / rho_old
            p = z + beta * p
            p_tld = z_tld + beta * p_tld

        q = A.dot(p)
        q_tld = A.T.dot(p_tld)
        denom = np.dot(p_tld, q)
        if abs(denom) < 1e-30:
            break
        alpha = rho / denom

        x = x + alpha * p
        r = r - alpha * q
        r_tld = r_tld - alpha * q_tld

        error = np.linalg.norm(r) / bnrm
        if error <= tol:
            break
        rho_old = rho

    return x


# ---------------------------------------------------------------------------
# 对称 Toeplitz 矩阵运算（源自 999_r8sto）
# ---------------------------------------------------------------------------

def r8sto_yw_sl(n, a):
    """
    求解 Yule-Walker 方程 T_n x = -a，其中 T_n 为对称 Toeplitz 矩阵，
    第一行为 [1, a1, a2, ..., an]。

    采用 Levinson-Durbin 递推算法，复杂度 O(n²)。
    """
    if n == 0:
        return np.array([])
    x = np.zeros(n)
    x[0] = -a[0]
    if n == 1:
        return x

    alpha = -a[0]
    beta = 1.0

    for i in range(1, n):
        # 反射系数
        num = -(a[i] + np.dot(a[:i][::-1], x[:i]))
        den = beta
        if abs(den) < 1e-30:
            den = 1e-30
        kappa = num / den

        # 更新解
        x_old = x[:i].copy()
        x[:i] = x_old + kappa * x_old[::-1]
        x[i] = kappa

        # 更新 beta
        beta = beta * (1.0 - kappa ** 2)
        if abs(beta) < 1e-30:
            beta = 1e-30

    return x


def r8sto_inverse(n, a_row):
    """
    计算对称正定 Toeplitz 矩阵的逆矩阵（Golub & Van Loan, 4.7.3）。

    参数
    ----
    n : int
        矩阵阶数
    a_row : ndarray (n,)
        第一行元素 [a0, a1, ..., a_{n-1}]

    返回
    ----
    B : ndarray (n, n)
        逆矩阵（一般稠密格式）
    """
    if n < 1:
        raise ValueError("n >= 1")
    if a_row.shape[0] < n:
        raise ValueError("a_row 长度不足")

    a0 = a_row[0]
    if abs(a0) < 1e-30:
        raise ValueError("对角元 a0 不能为零")

    if n == 1:
        return np.array([[1.0 / a0]])

    a2 = a_row[1:n] / a0
    v = r8sto_yw_sl(n - 1, a2)

    # v(n) = 1 / (1 + a2 * v)
    v_n = 1.0 / (1.0 + np.dot(a2, v))
    # 翻转并缩放
    v_rev = v[::-1]
    v_full = np.zeros(n)
    v_full[0] = v_n
    v_full[1:n] = v_n * v_rev

    B = np.zeros((n, n))
    B[0, :] = v_full[::-1]
    B[n - 1, :] = v_full
    B[1:n - 1, 0] = v_full[n - 1:1:-1]
    B[1:n - 1, n - 1] = v_full[1:n - 1]

    # 填充内部
    for i in range(1, (n + 1) // 2 + 1):
        for j in range(i, n - i + 1):
            val = B[i - 1, j - 1] + (v_full[n - j] * v_full[n - i] - v_full[i - 1] * v_full[j - 1]) / v_full[n - 1]
            B[i - 1, j - 1] = val
            B[j - 1, i - 1] = val
            B[n - i, n - j] = val
            B[n - j, n - i] = val

    B /= a0
    return B


def toeplitz_solve(n, a_row, b):
    """
    求解 Toeplitz 线性系统 T x = b。
    """
    Tinv = r8sto_inverse(n, a_row)
    return Tinv.dot(b)


# ---------------------------------------------------------------------------
# NCF 稀疏矩阵运算（源自 986_r8ncf）
# ---------------------------------------------------------------------------

class SparseNCF:
    """
    NSPCG 坐标格式稀疏矩阵：存储非零元的 (row, col, val)。
    """

    def __init__(self, m, n, nz_num, rowcol, a):
        """
        参数
        ----
        m, n : int
            矩阵维度
        nz_num : int
            非零元个数
        rowcol : ndarray (2, nz_num)
            行/列索引（0-based）
        a : ndarray (nz_num,)
            非零元值
        """
        self.m = m
        self.n = n
        self.nz_num = nz_num
        self.rowcol = rowcol.astype(int)
        self.a = a

    def mv(self, x):
        """
        矩阵-向量乘法 y = A @ x。
        """
        if x.shape[0] != self.n:
            raise ValueError("维度不匹配")
        y = np.zeros(self.m)
        for k in range(self.nz_num):
            i = self.rowcol[0, k]
            j = self.rowcol[1, k]
            y[i] += self.a[k] * x[j]
        return y

    def mtv(self, x):
        """
        转置矩阵-向量乘法 y = A^T @ x。
        """
        if x.shape[0] != self.m:
            raise ValueError("维度不匹配")
        y = np.zeros(self.n)
        for k in range(self.nz_num):
            i = self.rowcol[0, k]
            j = self.rowcol[1, k]
            y[j] += self.a[k] * x[i]
        return y

    def to_dense(self):
        """
        转为稠密矩阵（仅用于小规模测试）。
        """
        A = np.zeros((self.m, self.n))
        for k in range(self.nz_num):
            i = self.rowcol[0, k]
            j = self.rowcol[1, k]
            A[i, j] += self.a[k]
        return A


# ---------------------------------------------------------------------------
# 泊松方程 stencil（源自 PIC stencil 与 bicg 结合）
# ---------------------------------------------------------------------------

def create_poisson_stencil(nx, nz, dx, dz):
    """
    构建二维泊松方程 ∇²ψ = f 的五点差分矩阵 A（含 Dirichlet 边界）。

    离散格式：
        (ψ_{i+1,j} - 2ψ_{i,j} + ψ_{i-1,j}) / dx²
      + (ψ_{i,j+1} - 2ψ_{i,j} + ψ_{i,j-1}) / dz² = f_{i,j}

    边界节点直接设 ψ = 0（Dirichlet）。
    """
    N = nx * nz
    A = np.zeros((N, N))

    coeff_x = 1.0 / (dx ** 2)
    coeff_z = 1.0 / (dz ** 2)
    coeff_c = -2.0 * (coeff_x + coeff_z)

    for j in range(nz):
        for i in range(nx):
            idx = j * nx + i
            # 边界点 Dirichlet
            if i == 0 or i == nx - 1 or j == 0 or j == nz - 1:
                A[idx, idx] = 1.0
                continue

            A[idx, idx] = coeff_c
            A[idx, idx - 1] = coeff_x      # i-1
            A[idx, idx + 1] = coeff_x      # i+1
            A[idx, idx - nx] = coeff_z     # j-1
            A[idx, idx + nx] = coeff_z     # j+1

    return A


def create_sparse_poisson_ncf(nx, nz, dx, dz):
    """
    以 NCF 格式构建泊松方程稀疏矩阵。
    """
    N = nx * nz
    rowcol_list = []
    a_list = []

    coeff_x = 1.0 / (dx ** 2)
    coeff_z = 1.0 / (dz ** 2)
    coeff_c = -2.0 * (coeff_x + coeff_z)

    for j in range(nz):
        for i in range(nx):
            idx = j * nx + i
            if i == 0 or i == nx - 1 or j == 0 or j == nz - 1:
                rowcol_list.append([idx, idx])
                a_list.append(1.0)
                continue

            rowcol_list.append([idx, idx])
            a_list.append(coeff_c)
            rowcol_list.append([idx, idx - 1])
            a_list.append(coeff_x)
            rowcol_list.append([idx, idx + 1])
            a_list.append(coeff_x)
            rowcol_list.append([idx, idx - nx])
            a_list.append(coeff_z)
            rowcol_list.append([idx, idx + nx])
            a_list.append(coeff_z)

    rowcol = np.array(rowcol_list).T
    a = np.array(a_list)
    return SparseNCF(N, N, len(a_list), rowcol, a)
