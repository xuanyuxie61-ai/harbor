"""
matrix_algebra.py
稀疏矩阵运算与格式转换工具。

核心物理模型：
  托卡马克 MHD 稳定性分析与有限元离散产生大型稀疏矩阵。
  本模块实现矩阵格式转换与求解，对应原项目的：
    - mm_to_msm: Matrix Market → 稠密/稀疏矩阵
    - msm_to_hb: MATLAB 稀疏矩阵 → Harwell-Boeing 格式
    - r8utt: 上三角 Toeplitz 矩阵快速运算

  1. 刚度矩阵结构：
     Grad-Shafranov 方程有限元离散后得到形如

         A x = b

     的线性系统，其中 A 为对称正定稀疏矩阵。

  2. 上三角 Toeplitz 矩阵（UTT）：
     在环向 Fourier 模分析中，刚度矩阵经消元后呈现
     UTT 结构，可快速求解：

         A = [ a0  a1  a2  ...  a_{n-1} ]
             [ 0   a0  a1  ...  a_{n-2} ]
             [ ...                     ]
             [ 0   0   0   ...   a0    ]

     行列式: det(A) = a0^n
     求解:  向后回代

  3. 矩阵市场（MM）格式：
     用于交换大型稀疏矩阵：
         %%MatrixMarket matrix coordinate real general
         M N NNZ
         I1 J1 A1
         I2 J2 A2
         ...

  4. Harwell-Boeing（HB）格式：
     压缩列存储（CCS）的文本表示，包含指针数组、
     行索引数组与数值数组。
"""

import numpy as np


# ============================================================
# 1. 上三角 Toeplitz 矩阵 (r8utt)
# ============================================================

def r8utt_det(n, a):
    """
    计算上三角 Toeplitz 矩阵行列式。

    公式
    ----
        det(A) = a[0]^n

    参数
    ------
    n : int
        矩阵阶数。
    a : ndarray, shape (n,)
        第一行元素。

    返回
    ------
    det : float
        行列式值。
    """
    if n < 1:
        raise ValueError("矩阵阶数必须 ≥ 1")
    a = np.asarray(a)
    if len(a) < n:
        raise ValueError("数组长度不足")
    return float(a[0] ** n)


def r8utt_solve(n, a, b):
    """
    求解上三角 Toeplitz 线性系统 A x = b。

    算法
    ----
    向后回代：
        x[n-1] = b[n-1] / a[0]
        for j = n-2 downto 0:
            x[j] = (b[j] - Σ_{k=j+1}^{n-1} a[k-j] x[k]) / a[0]

    参数
    ------
    n : int
    a : ndarray, shape (n,)
        第一行。
    b : ndarray, shape (n,)
        右端项。

    返回
    ------
    x : ndarray, shape (n,)
        解向量。
    """
    if n < 1:
        raise ValueError("矩阵阶数必须 ≥ 1")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < n or len(b) < n:
        raise ValueError("数组长度不足")
    if abs(a[0]) < 1e-15:
        raise ValueError("对角元 a[0] 为零，矩阵奇异")

    x = b.copy()[:n]
    for j in range(n - 1, -1, -1):
        x[j] /= a[0]
        for i in range(j):
            x[i] -= a[j - i] * x[j]
    return x


# ============================================================
# 2. Matrix Market 格式读写 (mm_to_msm)
# ============================================================

def read_matrix_market(filename):
    """
    读取 Matrix Market 格式文件为稠密 numpy 数组。

    格式
    ----
        %%MatrixMarket matrix coordinate real general
        M N NNZ
        I J VALUE
        ...

    参数
    ------
    filename : str
        文件路径。

    返回
    ------
    A : ndarray, shape (M, N)
        稠密矩阵。
    info : dict
        元数据。
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # 跳过注释
    idx = 0
    while idx < len(lines) and lines[idx].strip().startswith('%'):
        idx += 1

    if idx >= len(lines):
        raise ValueError("文件为空或格式错误")

    header = lines[idx].strip().split()
    if len(header) < 3:
        raise ValueError("Matrix Market 头格式错误")

    M, N, nnz = int(header[0]), int(header[1]), int(header[2])
    A = np.zeros((M, N), dtype=float)

    for line in lines[idx + 1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        i, j, val = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
        if 0 <= i < M and 0 <= j < N:
            A[i, j] = val

    info = {"rows": M, "cols": N, "nnz": nnz, "format": "coordinate"}
    return A, info


def write_matrix_market(filename, A, title="Matrix"):
    """
    将稠密矩阵写入 Matrix Market 坐标格式。

    参数
    ------
    filename : str
    A : ndarray
    title : str
    """
    A = np.asarray(A, dtype=float)
    M, N = A.shape
    nnz = np.count_nonzero(A)
    with open(filename, 'w') as f:
        f.write(f"%%MatrixMarket matrix coordinate real general\n")
        f.write(f"%{title}\n")
        f.write(f"{M} {N} {nnz}\n")
        for i in range(M):
            for j in range(N):
                if abs(A[i, j]) > 1e-15:
                    f.write(f"{i + 1} {j + 1} {A[i, j]:.16e}\n")


# ============================================================
# 3. Harwell-Boeing 格式转换 (msm_to_hb)
# ============================================================

def matrix_to_hb_format(A, title="Title", key="Key", ifmt=8, job=2):
    """
    将稠密矩阵转换为 Harwell-Boeing 格式的字符串表示。

    HB 格式结构（简化版，仅支持实非对称矩阵 RUA）：
        Line 1: Title(72) Key(8)
        Line 2: Totcrd(14) Ptrcrd(14) Indcrd(14) Valcrd(14) Rhscrd(14)
        Line 3: MXTYPE(3) NROW(14) NCOL(14) NNZERO(14) NELTVL(14)
        Line 4: PTRFMT(16) INDFMT(16) VALFMT(20) RHSFMT(20)
        指针行: IA
        索引行: JA
        数值行: A

    参数
    ------
    A : ndarray
    title, key : str
    ifmt : int
        数值格式控制。
    job : int
        写入内容控制。

    返回
    ------
    hb_string : str
        HB 格式文本。
    """
    A = np.asarray(A, dtype=float)
    nrow, ncol = A.shape

    # 转为 CCS（压缩列存储）
    col_ptr = [1]
    row_ind = []
    values = []
    for j in range(ncol):
        for i in range(nrow):
            if abs(A[i, j]) > 1e-15:
                row_ind.append(i + 1)
                values.append(A[i, j])
        col_ptr.append(len(row_ind) + 1)

    nnzero = len(values)
    neltvl = 0

    # 行数估算
    ptr_per_line = 8
    ind_per_line = 8
    val_per_line = 4
    ptrcrd = (len(col_ptr) + ptr_per_line - 1) // ptr_per_line
    indcrd = (len(row_ind) + ind_per_line - 1) // ind_per_line
    valcrd = (len(values) + val_per_line - 1) // val_per_line
    rhscrd = 0
    totcrd = ptrcrd + indcrd + valcrd + rhscrd

    lines = []
    lines.append(f"{title[:72]:72}{key[:8]:8}")
    lines.append(f"{totcrd:14d}{ptrcrd:14d}{indcrd:14d}{valcrd:14d}{rhscrd:14d}")
    lines.append(f"{'RUA':3}{nrow:14d}{ncol:14d}{nnzero:14d}{neltvl:14d}")
    lines.append(f"{'(8I10)':16}{'(8I10)':16}{'(4E20.13)':20}{'(4E20.13)':20}")

    # 列指针
    for k in range(0, len(col_ptr), ptr_per_line):
        chunk = col_ptr[k:k + ptr_per_line]
        lines.append("".join(f"{v:10d}" for v in chunk))

    # 行索引
    for k in range(0, len(row_ind), ind_per_line):
        chunk = row_ind[k:k + ind_per_line]
        lines.append("".join(f"{v:10d}" for v in chunk))

    # 数值
    for k in range(0, len(values), val_per_line):
        chunk = values[k:k + val_per_line]
        lines.append("".join(f"{v:20.13e}" for v in chunk))

    return "\n".join(lines) + "\n"


def write_hb_file(filename, A, title="Title", key="Key"):
    """
    将矩阵写入 Harwell-Boeing 文件。

    参数
    ------
    filename : str
    A : ndarray
    title, key : str
    """
    hb_text = matrix_to_hb_format(A, title=title, key=key)
    with open(filename, 'w') as f:
        f.write(hb_text)


# ============================================================
# 4. 托卡马克专用矩阵运算
# ============================================================

def assemble_global_stiffness(vertices, triangles):
    """
    从三角形网格组装全局刚度矩阵。

    参数
    ------
    vertices : ndarray, shape (n_v, 2)
    triangles : ndarray, shape (n_t, 3)

    返回
    ------
    K : ndarray, shape (n_v, n_v)
        全局刚度矩阵。
    """
    from quadrature_engine import assemble_stiffness_triangle

    n_v = vertices.shape[0]
    K = np.zeros((n_v, n_v))

    for tri in triangles:
        v1, v2, v3 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        K_loc = assemble_stiffness_triangle(v1, v2, v3)
        for i in range(3):
            for j in range(3):
                K[tri[i], tri[j]] += K_loc[i, j]

    return K


def solve_stiffness_system(K, b, tol=1e-12):
    """
    使用共轭梯度法求解对称正定刚度方程组。

    参数
    ------
    K : ndarray
        刚度矩阵（假设对称正定）。
    b : ndarray
        右端项。
    tol : float
        收敛容差。

    返回
    ------
    x : ndarray
        解向量。
    info : dict
        迭代信息。
    """
    b = np.asarray(b, dtype=float)
    n = len(b)
    x = np.zeros(n)
    r = b - K @ x
    p = r.copy()
    rs_old = np.dot(r, r)

    max_iter = min(n * 10, 5000)
    for it in range(max_iter):
        Ap = K @ p
        alpha = rs_old / (np.dot(p, Ap) + 1e-30)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    info = {"iterations": it + 1, "residual": float(np.sqrt(rs_new))}
    return x, info


def condition_number_estimate(A):
    """
    估计矩阵条件数 κ(A) = ||A|| · ||A^{-1}||。

    使用幂迭代估计最大与最小奇异值。

    参数
    ------
    A : ndarray

    返回
    ------
    cond : float
        条件数估计。
    """
    A = np.asarray(A)
    # 最大特征值估计
    x = np.random.randn(A.shape[1])
    for _ in range(20):
        x = A.T @ (A @ x)
        x /= np.linalg.norm(x)
    sigma_max = np.sqrt(np.linalg.norm(A @ x))

    # 最小特征值估计（对逆矩阵用幂迭代）
    try:
        A_inv = np.linalg.inv(A + 1e-12 * np.eye(A.shape[0]))
        y = np.random.randn(A.shape[0])
        for _ in range(20):
            y = A_inv.T @ (A_inv @ y)
            y /= np.linalg.norm(y)
        sigma_min = np.sqrt(np.linalg.norm(A_inv @ y))
        if sigma_min < 1e-15:
            sigma_min = 1e-15
        return float(sigma_max / sigma_min)
    except np.linalg.LinAlgError:
        return float('inf')
