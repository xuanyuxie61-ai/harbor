# -*- coding: utf-8 -*-
"""
test_problems.py
================
稀疏线性系统测试问题构造器。

融合种子项目：
- 149_cg        : DIF2, 随机 SPD 矩阵
- 506_hankel_spd: Hankel SPD 矩阵与 Cholesky 分解
- 898_polynomials: 多元多项式（Rosenbrock, Camel, Heart 等）
- 1161_steinerberger: Steinerberger 函数构造的病态右端项
- 141_cavity_flow_display: 顶盖驱动方腔流（转化为离散系统）
- 1214_test_interp_nd    : Genz 多维测试函数（构造高维右端项）
"""

import numpy as np
import math
from sparse_matrix import dif2_r8ge, SparseMatrixOperator
from special_functions import steinerberger_function
from random_tools import random_spd_matrix, random_spd_with_clustered_spectrum


# ---------------------------------------------------------------------------
# Hankel SPD 矩阵（506_hankel_spd）
# ---------------------------------------------------------------------------

def hankel_spd_cholesky_lower(n, lii, liim1):
    """
    构造 Hankel 对称正定矩阵的下三角 Cholesky 因子 L，使得 H = L @ L.T。
    其中 H 满足 H(i+j) = h(k-1)（沿反对角线恒定）。

    算法（Al-Homidan & Alshahrani, 2009）：
        L[i,i]   = lii[i]
        L[i+1,i] = liim1[i]
        对 i >= 3, j < i-1:
            令 q = floor((i+j)/2), r = ceil((i+j)/2)
            α = Σ_{s=1}^q L[q,s] L[r,s]
            β = Σ_{t=1}^{j-1} L[i,t] L[j,t]
            L[i,j] = (α - β) / L[j,j]

    参数
    ----
    n : int
    lii : ndarray, shape (n,)
        对角元 L[i,i]。
    liim1 : ndarray, shape (n-1,)
        次对角元 L[i+1,i]。

    返回
    ----
    L : ndarray, shape (n, n)
        下三角 Cholesky 因子。
    H : ndarray, shape (n, n)
        Hankel SPD 矩阵 H = L @ L.T。
    """
    if len(lii) != n:
        raise ValueError("lii length must equal n.")
    if len(liim1) != n - 1:
        raise ValueError("liim1 length must equal n-1.")
    L = np.zeros((n, n), dtype=float)
    for i in range(n):
        L[i, i] = lii[i]
    for i in range(n - 1):
        L[i + 1, i] = liim1[i]

    for i in range(2, n):
        for j in range(i - 1):
            if (i + j) % 2 == 0:
                q = (i + j) // 2
                r = q
            else:
                q = (i + j - 1) // 2
                r = q + 1

            alpha = 0.0
            for s in range(q):
                alpha += L[q, s] * L[r, s]

            beta = 0.0
            for t in range(j):
                beta += L[i, t] * L[j, t]

            if abs(L[j, j]) < 1e-30:
                L[i, j] = 0.0
            else:
                L[i, j] = (alpha - beta) / L[j, j]

    H = L @ L.T
    return L, H


def hankel_spd_from_moments(n, moments):
    """
    从矩序列构造 Hankel SPD 矩阵。
    H[i,j] = moments[i+j]，要求矩序列正定。
    """
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            idx = i + j
            if idx < len(moments):
                H[i, j] = moments[idx]
            else:
                H[i, j] = 0.0
    # 若不正定，做正则化
    lam = np.linalg.eigvalsh(H)
    if lam[0] <= 0:
        H += (-lam[0] + 1e-6) * np.eye(n)
    return H


# ---------------------------------------------------------------------------
# 各向异性扩散矩阵（2D/3D 有限差分离散）
# ---------------------------------------------------------------------------

def anisotropic_diffusion_2d(nx, ny, epsilon_x, epsilon_y, hx=None, hy=None):
    """
    二维各向异性扩散方程的有限差分离散：
        -∂/∂x (ε_x ∂u/∂x) - ∂/∂y (ε_y ∂u/∂y) = f
    在 [0,1]×[0,1] 上均匀网格，Dirichlet 边界条件 u=0。

    离散格式（五点 stencil）：
        (2ε_x/hx^2 + 2ε_y/hy^2) u_{i,j}
        - ε_x/hx^2 (u_{i-1,j} + u_{i+1,j})
        - ε_y/hy^2 (u_{i,j-1} + u_{i,j+1})

    返回
    ----
    A : ndarray, shape (N, N), N = nx*ny
        SPD 稀疏矩阵（稠密存储，仅用于中小规模）。
    """
    if hx is None:
        hx = 1.0 / (nx + 1)
    if hy is None:
        hy = 1.0 / (ny + 1)
    N = nx * ny
    A = np.zeros((N, N), dtype=float)
    cx = epsilon_x / (hx ** 2)
    cy = epsilon_y / (hy ** 2)
    diag = 2.0 * cx + 2.0 * cy

    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            A[idx, idx] = diag
            if i > 0:
                A[idx, idx - 1] = -cx
            if i < nx - 1:
                A[idx, idx + 1] = -cx
            if j > 0:
                A[idx, idx - nx] = -cy
            if j < ny - 1:
                A[idx, idx + nx] = -cy
    return A


def helmholtz_2d(nx, ny, k, hx=None, hy=None):
    """
    二维 Helmholtz 方程离散：
        -Δu - k^2 u = f
    注意：当 k 较大时矩阵不定，此处做正则化使其 SPD（仅用于预处理测试）：
        A = -Δ + k^2 I （即将符号反转，构造正定系统）。
    """
    if hx is None:
        hx = 1.0 / (nx + 1)
    if hy is None:
        hy = 1.0 / (ny + 1)
    N = nx * ny
    A = np.zeros((N, N), dtype=float)
    cx = 1.0 / (hx ** 2)
    cy = 1.0 / (hy ** 2)
    diag = 2.0 * cx + 2.0 * cy + k ** 2

    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            A[idx, idx] = diag
            if i > 0:
                A[idx, idx - 1] = -cx
            if i < nx - 1:
                A[idx, idx + 1] = -cx
            if j > 0:
                A[idx, idx - nx] = -cy
            if j < ny - 1:
                A[idx, idx + nx] = -cy
    return A


# ---------------------------------------------------------------------------
# 基于多元多项式的非线性右端项构造（898_polynomials）
# ---------------------------------------------------------------------------

def rosenbrock_rhs(m, x_grid):
    """
    Rosenbrock 函数构造右端项：
        f(x) = Σ_{i=1}^{m-1} [100 (x_i - x_{i+1})^2 + (x_i - 1)^2]
    用于测试非线性优化问题的线性化子系统。
    """
    x = np.asarray(x_grid, dtype=float)
    if x.ndim == 1 and x.size == m:
        val = 0.0
        for i in range(m - 1):
            val += 100.0 * (x[i] - x[i + 1]) ** 2 + (x[i] - 1.0) ** 2
        return val
    else:
        # 批量
        n = x.shape[1] if x.ndim > 1 else 1
        val = np.zeros(n, dtype=float)
        for i in range(m - 1):
            val += 100.0 * (x[i, :] - x[i + 1, :]) ** 2 + (x[i, :] - 1.0) ** 2
        return val


def camel_rhs(x):
    """Six-hump camel 函数（2D）构造的右端项。"""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x1, x2 = x[0], x[1]
    else:
        x1, x2 = x[0, :], x[1, :]
    return (4.0 * x1 ** 2
            - 2.1 * x1 ** 4
            + (1.0 / 3.0) * x1 ** 6
            + x1 * x2
            - 4.0 * x2 ** 2
            + 4.0 * x2 ** 4)


# ---------------------------------------------------------------------------
# 顶盖驱动方腔流离散（141_cavity_flow_display 的数学模型）
# ---------------------------------------------------------------------------

def cavity_flow_stokes_matrix(nx, ny, nu=1.0, hx=None, hy=None):
    """
    简化 Stokes 方程的有限差分离散（速度-压力形式的对称正定子系统）。
    取速度场的 Laplacian 部分：
        A = [ -νΔ   0   ]
            [  0   -νΔ  ]
    对二维方腔流，仅提取速度分量的 SPD 块。
    这里简化为两个解耦的各向同性扩散算子。
    """
    if hx is None:
        hx = 1.0 / (nx + 1)
    if hy is None:
        hy = 1.0 / (ny + 1)
    N = nx * ny
    A = np.zeros((N, N), dtype=float)
    cx = nu / (hx ** 2)
    cy = nu / (hy ** 2)
    diag = 2.0 * cx + 2.0 * cy
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            A[idx, idx] = diag
            if i > 0:
                A[idx, idx - 1] = -cx
            if i < nx - 1:
                A[idx, idx + 1] = -cx
            if j > 0:
                A[idx, idx - nx] = -cy
            if j < ny - 1:
                A[idx, idx + nx] = -cy
    return A


def cavity_flow_rhs(nx, ny):
    """
    顶盖驱动方腔流的简化右端项：上边界 (y=1) 有单位切向速度驱动。
    在内部点产生强迫项。
    """
    N = nx * ny
    b = np.zeros(N, dtype=float)
    # 驱动效应通过边界条件引入，这里用指数衰减近似
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            y = (j + 1) / (ny + 1)
            b[idx] = math.exp(-10.0 * (1.0 - y))
    return b


# ---------------------------------------------------------------------------
# 综合测试问题生成器
# ---------------------------------------------------------------------------

def build_test_problem(problem_id, n, extra=None):
    """
    统一接口生成测试问题 (A, b, x_exact)。

    problem_id:
        'dif2'        : 1D DIF2 (n×n)
        'hankel'      : Hankel SPD (n×n)
        'aniso2d'     : 2D 各向异性扩散，n = nx*ny
        'helmholtz2d' : 2D Helmholtz SPD 正则化，n = nx*ny
        'random_spd'  : 随机 SPD
        'clustered'   : 聚类谱 SPD
        'cavity'      : 方腔流简化模型，n = nx*ny
        'steinerberger': Steinerberger 函数驱动的 DIF2 系统
    """
    extra = extra if extra is not None else {}
    rng = np.random.default_rng(extra.get('seed', 42))

    if problem_id == 'dif2':
        A = dif2_r8ge(n)
        x_exact = rng.random(n)
        b = A @ x_exact
        return A, b, x_exact

    elif problem_id == 'hankel':
        lii = np.ones(n, dtype=float)
        liim1 = 0.5 * np.ones(n - 1, dtype=float)
        _, A = hankel_spd_cholesky_lower(n, lii, liim1)
        x_exact = np.sin(np.linspace(0, math.pi, n))
        b = A @ x_exact
        return A, b, x_exact

    elif problem_id == 'aniso2d':
        nx = extra.get('nx', int(math.sqrt(n)))
        ny = extra.get('ny', n // nx)
        n = nx * ny
        eps_x = extra.get('eps_x', 1.0)
        eps_y = extra.get('eps_y', 0.01)
        A = anisotropic_diffusion_2d(nx, ny, eps_x, eps_y)
        x_exact = rng.random(n)
        b = A @ x_exact
        return A, b, x_exact

    elif problem_id == 'helmholtz2d':
        nx = extra.get('nx', int(math.sqrt(n)))
        ny = extra.get('ny', n // nx)
        n = nx * ny
        k = extra.get('k', 10.0)
        A = helmholtz_2d(nx, ny, k)
        x_exact = rng.random(n)
        b = A @ x_exact
        return A, b, x_exact

    elif problem_id == 'random_spd':
        A, _, _ = random_spd_matrix(n, seed=extra.get('seed', 42))
        x_exact = rng.random(n)
        b = A @ x_exact
        return A, b, x_exact

    elif problem_id == 'clustered':
        clusters = extra.get('clusters', [(0.1, 0.05, n // 4), (10.0, 1.0, n // 4), (100.0, 5.0, n // 2)])
        A, _, _ = random_spd_with_clustered_spectrum(n, clusters, seed=extra.get('seed', 42))
        x_exact = rng.random(n)
        b = A @ x_exact
        return A, b, x_exact

    elif problem_id == 'cavity':
        nx = extra.get('nx', int(math.sqrt(n)))
        ny = extra.get('ny', n // nx)
        n = nx * ny
        A = cavity_flow_stokes_matrix(nx, ny, nu=extra.get('nu', 1.0))
        b = cavity_flow_rhs(nx, ny)
        x_exact = np.linalg.solve(A, b)
        return A, b, x_exact

    elif problem_id == 'steinerberger':
        A = dif2_r8ge(n)
        x_grid = np.linspace(0.0, 1.0, n)
        sb_param = extra.get('sb_param', 20)
        b = steinerberger_function(sb_param, x_grid)
        # 使右端项与离散 Laplacian 一致：b_i = f(x_i) * h^2
        h = 1.0 / (n + 1)
        b = b * (h ** 2)
        x_exact = np.linalg.solve(A, b)
        return A, b, x_exact

    else:
        raise ValueError(f"Unknown problem_id: {problem_id}")
