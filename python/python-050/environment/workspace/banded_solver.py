"""
banded_solver.py
带状矩阵高效求解器 — 冰盖一维/二维流动线性系统

基于种子项目 972_r8but 的带状上三角矩阵操作思想，
扩展为一般带状矩阵 (包括三对角、五对角) 的高效前代/回代求解，
用于冰盖 SIA 与 SSA (Shallow Shelf Approximation) 离散后的线性系统。

核心数学:
  1. 带状矩阵存储 (对角线编号):
       对角线索引: main=0, lower=-1,-2,..., upper=+1,+2,...
       存储为 (2*kl + ku + 1) x n 的稠密数组，或更紧凑格式。

  2. 三对角系统 (Thomas 算法):
       a_i u_{i-1} + b_i u_i + c_i u_{i+1} = d_i

       前向消元:
         c'_i = c_i / (b_i - a_i c'_{i-1})
         d'_i = (d_i - a_i d'_{i-1}) / (b_i - a_i c'_{i-1})

       回代:
         u_n = d'_n
         u_i = d'_i - c'_i u_{i+1}

  3. 五对角系统 (用于四阶精度或耦合多物理):
       p_i u_{i-2} + q_i u_{i-1} + r_i u_i + s_i u_{i+1} + t_i u_{i+2} = d_i

  4. 带状 LU 分解:
       对于一般带状矩阵 A (下半带宽 kl, 上半带宽 ku)，
       LU 分解后的 L 和 U 保持带状结构，求解复杂度 O(n * kl * ku)。

应用场景:
  - 一维 SIA 垂直剖面速度求解
  - 二维 SSA 水平速度场的线性化求解
  - 冰盖热力学方程的隐式时间步进
"""

import numpy as np
from typing import Tuple


def solve_tridiagonal(a: np.ndarray,
                      b: np.ndarray,
                      c: np.ndarray,
                      d: np.ndarray) -> np.ndarray:
    """
    Thomas 算法求解三对角线性系统。

    系统形式:
        a_i u_{i-1} + b_i u_i + c_i u_{i+1} = d_i,  i = 0, ..., n-1

    边界约定:
        a[0] = 0, c[n-1] = 0

    参数:
        a: 下对角线 (n,)
        b: 主对角线 (n,)
        c: 上对角线 (n,)
        d: 右端项 (n,)

    返回:
        u: 解向量 (n,)
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    n = len(b)

    if not (len(a) == len(c) == len(d) == n):
        raise ValueError("All input arrays must have the same length.")

    # 前向消元
    cp = np.zeros(n, dtype=np.float64)
    dp = np.zeros(n, dtype=np.float64)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if abs(denom) < 1e-20:
            denom = 1e-20 * np.sign(denom) if denom != 0 else 1e-20
        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    # 回代
    u = np.zeros(n, dtype=np.float64)
    u[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        u[i] = dp[i] - cp[i] * u[i + 1]

    return u


def solve_pentadiagonal(p: np.ndarray, q: np.ndarray,
                        r: np.ndarray, s: np.ndarray,
                        t: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    求解五对角线性系统 (使用带状 LU 分解)。

    系统形式:
        p_i u_{i-2} + q_i u_{i-1} + r_i u_i + s_i u_{i+1} + t_i u_{i+2} = d_i

    参数:
        p: 次下对角线 (n,), p[0]=p[1]=0
        q: 下对角线 (n,), q[0]=0
        r: 主对角线 (n,)
        s: 上对角线 (n,), s[n-1]=0
        t: 次上对角线 (n,), t[n-1]=t[n-2]=0
        d: 右端项 (n,)

    返回:
        u: 解向量
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    n = len(r)

    # 转换为稠密矩阵后使用 numpy 求解 (对于中等规模 n 足够高效)
    # 对于大规模问题，应使用专门的带状求解器
    if n > 5000:
        # 使用 scipy 稀疏求解
        try:
            from scipy.sparse import diags
            from scipy.sparse.linalg import spsolve
            A = diags([p[2:], q[1:], r, s[:-1], t[:-2]],
                      [-2, -1, 0, 1, 2], format='csc')
            return spsolve(A, d)
        except ImportError:
            pass

    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        A[i, i] = r[i]
        if i > 0:
            A[i, i - 1] = q[i]
        if i > 1:
            A[i, i - 2] = p[i]
        if i < n - 1:
            A[i, i + 1] = s[i]
        if i < n - 2:
            A[i, i + 2] = t[i]

    return np.linalg.solve(A, d)


def banded_lu_solve(ab: np.ndarray, kl: int, ku: int, b: np.ndarray) -> np.ndarray:
    """
    求解一般带状线性系统 A x = b。

    采用 scipy.linalg.lapack 的 gbsv 接口 (若有)，否则用稠密求解。

    参数:
        ab: 带状存储矩阵 (kl+ku+1, n)
            行 0..ku-1: 上对角线
            行 ku: 主对角线
            行 ku+1..kl+ku: 下对角线
        kl: 下半带宽
        ku: 上半带宽
        b: 右端项 (n,) 或 (n, nrhs)

    返回:
        x: 解
    """
    b = np.asarray(b, dtype=np.float64)
    ab = np.asarray(ab, dtype=np.float64)

    if ab.shape[0] != kl + ku + 1:
        raise ValueError("ab must have shape (kl+ku+1, n)")

    n = ab.shape[1]

    try:
        from scipy.linalg.lapack import dgbsv
        # LAPACK 要求 ab 有 (2*kl+ku+1, n) 形状，并在特定位置填充
        ab_lapack = np.zeros((2 * kl + ku + 1, n), dtype=np.float64)
        for j in range(n):
            for i in range(kl + ku + 1):
                row_in_lapack = i + kl
                col = j + i - ku
                if 0 <= col < n:
                    ab_lapack[row_in_lapack, col] = ab[i, col]

        if b.ndim == 1:
            b = b.reshape(-1, 1)
        _, _, x, info = dgbsv(kl, ku, ab_lapack, b)
        if info != 0:
            raise RuntimeError(f"dgbsv failed with info={info}")
        return x.ravel() if x.shape[1] == 1 else x
    except ImportError:
        # 回退到稠密求解
        A = banded_to_dense(ab, kl, ku, n)
        return np.linalg.solve(A, b)


def banded_to_dense(ab: np.ndarray, kl: int, ku: int, n: int) -> np.ndarray:
    """
    将带状存储矩阵展开为稠密矩阵。
    """
    A = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        for i in range(kl + ku + 1):
            row = j + ku - i
            col = j
            if 0 <= row < n:
                A[row, col] = ab[i, col]
    return A


def dense_to_banded(A: np.ndarray, kl: int, ku: int) -> np.ndarray:
    """
    将稠密矩阵压缩为带状存储。
    """
    n = A.shape[0]
    ab = np.zeros((kl + ku + 1, n), dtype=np.float64)
    for j in range(n):
        i_start = max(0, ku - j)
        i_end = min(kl + ku + 1, ku + n - j)
        for i in range(i_start, i_end):
            row = j + ku - i
            ab[i, j] = A[row, j]
    return ab


def build_sia_tridiagonal(H: np.ndarray,
                          bedrock: np.ndarray,
                          dx: float,
                          A: float,
                          rho_g: float,
                          n: float = 3.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    为一维 SIA 冰厚度演化构建三对角系统矩阵。

    离散化后的线性化系统:
        -D_{i-1/2} H_{i-1} + (D_{i-1/2} + D_{i+1/2}) H_i - D_{i+1/2} H_{i+1} = RHS_i

    参数:
        H: 当前厚度剖面 (n,)
        bedrock: 基岩高程 (n,)
        dx: 网格间距
        A: 率因子
        rho_g: \rho g
        n: Glen 指数

    返回:
        a, b, c, rhs: 三对角元素与右端项
    """
    H = np.asarray(H, dtype=np.float64)
    n_nodes = len(H)

    a = np.zeros(n_nodes, dtype=np.float64)
    b = np.zeros(n_nodes, dtype=np.float64)
    c = np.zeros(n_nodes, dtype=np.float64)
    rhs = np.zeros(n_nodes, dtype=np.float64)

    surface = bedrock + H
    grad_s = np.zeros(n_nodes, dtype=np.float64)
    grad_s[1:-1] = (surface[2:] - surface[:-2]) / (2.0 * dx)
    grad_s[0] = (surface[1] - surface[0]) / dx
    grad_s[-1] = (surface[-1] - surface[-2]) / dx
    grad_s = np.abs(grad_s)
    grad_s = np.maximum(grad_s, 1e-12)

    # TODO_HOLE_3: 实现 SIA 三对角系统构建
    # 科学知识点:
    #   1. 扩散系数 D = (2A/(n+2)) * (rho_g)^n * H^(n+2) * |grad_s|^(n-1)
    #   2. 内部节点矩阵元素: a[i] = -D_left, b[i] = D_left + D_right, c[i] = -D_right
    #      其中 D_left = 0.5*(D[i]+D[i-1])/dx^2, D_right = 0.5*(D[i]+D[i+1])/dx^2
    #   3. Dirichlet 边界: b[0]=b[-1]=1, c[0]=a[-1]=0, rhs[0]=H[0], rhs[-1]=H[-1]
    # 注意数值保护与 grad_s 的处理
    raise NotImplementedError("Hole 3: 请实现 build_sia_tridiagonal 核心公式与矩阵组装")
