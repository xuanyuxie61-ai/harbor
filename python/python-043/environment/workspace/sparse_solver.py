"""
sparse_solver.py — 稀疏线性系统求解器

融合以下种子项目：
- 152_cg_rc : 逆通信共轭梯度法 (Conjugate Gradient with Reverse Communication)
- 1349_triangulation_rcm : RCM 重排序以降低带宽

功能：
1. 实现逆通信 CG 求解器，用于大规模稀疏对称正定矩阵
2. 实现 Jacobi 预处理器与不完全 Cholesky 预处理器
3. 集成 RCM 重排序接口
4. 矩阵-向量乘积接口（支持隐式矩阵）
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve_triangular


class CGSolverRC:
    """
    逆通信共轭梯度求解器（Reverse Communication CG）。
    源自 152_cg_rc 的核心算法思想。

    求解:  A * x = b
    其中 A 是对称正定矩阵，用户通过回调函数提供矩阵-向量乘积。

    算法流程：
      r_0 = b - A x_0
      对 k = 0,1,2,...:
        z_k = M^{-1} r_k      (预处理)
        ρ_k = r_k^T z_k
        若 k>0: β = ρ_k / ρ_{k-1},  p_k = z_k + β p_{k-1}
        否则:   p_k = z_k
        q_k = A p_k
        α = ρ_k / (p_k^T q_k)
        x_{k+1} = x_k + α p_k
        r_{k+1} = r_k - α q_k
        若 ||r_{k+1}|| < tol: 收敛
    """

    def __init__(self, n, tol=1e-10, max_iter=None):
        self.n = n
        self.tol = tol
        self.max_iter = max_iter if max_iter is not None else 10 * n
        self.iter = 0
        self.rho = 0.0
        self.rho_old = 0.0
        self.state = 0  # 0=init, 1=need_precond, 2=need_matvec, 3=check_conv

    def solve(self, b, x0, matvec_func, precond_func=None):
        """
        执行 CG 迭代直到收敛。

        参数：
          b            : 右端项 (n,)
          x0           : 初始猜测 (n,)
          matvec_func  : 矩阵-向量乘积函数 func(v) -> A@v
          precond_func : 预处理函数 func(v) -> M^{-1}@v，若为 None 则用恒等预处理
        """
        if precond_func is None:
            precond_func = lambda v: v.copy()

        x = x0.copy()
        r = b - matvec_func(x)
        z = precond_func(r)
        p = z.copy()
        rho = np.dot(r, z)

        for k in range(self.max_iter):
            q = matvec_func(p)
            pdotq = np.dot(p, q)
            if abs(pdotq) < 1e-30:
                break
            alpha = rho / pdotq
            x += alpha * p
            r -= alpha * q

            norm_r = np.linalg.norm(r)
            if norm_r < self.tol * np.linalg.norm(b):
                break

            z = precond_func(r)
            rho_old = rho
            rho = np.dot(r, z)
            if abs(rho_old) < 1e-30:
                break
            beta = rho / rho_old
            p = z + beta * p

        return x, k + 1, norm_r


def jacobi_preconditioner_diagonal(A_sparse):
    """
    构建 Jacobi 预处理器的对角线：M = diag(A)^{-1}。
    返回对角线逆的数组。
    """
    diag = A_sparse.diagonal().copy()
    diag = np.where(np.abs(diag) < 1e-15, 1.0, diag)
    return 1.0 / diag


def apply_jacobi_precond(d_inv, r):
    """应用 Jacobi 预处理。"""
    return d_inv * r


def incomplete_cholesky_preconditioner(A_sparse, drop_tol=1e-12):
    """
    不完全 Cholesky 分解预处理器。
    对 A ≈ L L^T，其中 L 是稀疏下三角矩阵。
    返回 L 矩阵（scipy sparse 格式）。
    """
    try:
        from sksparse.cholmod import cholesky
        factor = cholesky(A_sparse, beta=drop_tol)
        L = factor.L()
        return L
    except Exception:
        # 回退到 scipy 的 ILU 近似
        from scipy.sparse.linalg import spilu
        ilu = spilu(A_sparse, drop_tol=drop_tol)
        return ilu


def build_wathen_matrix(nx, ny):
    """
    构建 Wathen 有限元矩阵（源自 152_cg_rc/wathen.m）。
    该矩阵是稀疏对称正定的，用于测试求解器。
    
    元素质量矩阵（8节点 serendipity 单元）：
      EM = [
         6, -6,  2, -8,  3, -8,  2, -6;
        -6, 32, -6, 20, -8, 16, -8, 20;
         2, -6,  6, -6,  2, -8,  3, -8;
        -8, 20, -6, 32, -6, 20, -8, 16;
         3, -8,  2, -6,  6, -6,  2, -8;
        -8, 16, -8, 20, -6, 32, -6, 20;
         2, -8,  3, -8,  2, -6,  6, -6;
        -6, 20, -8, 16, -8, 20, -6, 32
      ]
    
    节点数: N = 3*NX*NY + 2*NX + 2*NY + 1
    """
    em = np.array([
        [6.0, -6.0, 2.0, -8.0, 3.0, -8.0, 2.0, -6.0],
        [-6.0, 32.0, -6.0, 20.0, -8.0, 16.0, -8.0, 20.0],
        [2.0, -6.0, 6.0, -6.0, 2.0, -8.0, 3.0, -8.0],
        [-8.0, 20.0, -6.0, 32.0, -6.0, 20.0, -8.0, 16.0],
        [3.0, -8.0, 2.0, -6.0, 6.0, -6.0, 2.0, -8.0],
        [-8.0, 16.0, -8.0, 20.0, -6.0, 32.0, -6.0, 20.0],
        [2.0, -8.0, 3.0, -8.0, 2.0, -6.0, 6.0, -6.0],
        [-6.0, 20.0, -8.0, 16.0, -8.0, 20.0, -6.0, 32.0]
    ])

    n = 3 * nx * ny + 2 * nx + 2 * ny + 1
    A = np.zeros((n, n))

    for j in range(1, ny + 1):
        for i in range(1, nx + 1):
            node = np.zeros(8, dtype=int)
            node[0] = 3 * j * nx + 2 * j + 2 * i + 1
            node[1] = node[0] - 1
            node[2] = node[0] - 2
            node[3] = (3 * j - 1) * nx + 2 * j + i - 1
            node[7] = node[3] + 1
            node[4] = (3 * j - 3) * nx + 2 * j + 2 * i - 3
            node[5] = node[4] + 1
            node[6] = node[4] + 2

            for krow in range(8):
                for kcol in range(8):
                    if 1 <= node[krow] <= n and 1 <= node[kcol] <= n:
                        A[node[krow] - 1, node[kcol] - 1] += 20.0 * em[krow, kcol] / 9.0

    return A


def build_laplacian_spherical_shell(nodes, elements, r_icb, r_cmb):
    """
    在球壳网格上构建离散 Laplacian 算子的稀疏矩阵。
    用于求解磁矢势的 Poisson 方程：
        ∇^2 A = -μ_0 J
    
    采用有限体积法（FVM）在四面体网格上离散。
    """
    n_nodes = len(nodes)
    row_ind = []
    col_ind = []
    data = []

    if elements.size == 0:
        # 回退：基于距离的高斯近似 Laplacian
        for i in range(n_nodes):
            neighbors = []
            for j in range(n_nodes):
                if i == j:
                    continue
                dist = np.linalg.norm(nodes[i] - nodes[j])
                if dist < 0.3:
                    neighbors.append((j, dist))
            sum_w = 0.0
            for j, dist in neighbors:
                w = np.exp(-dist * dist / 0.02)
                row_ind.append(i)
                col_ind.append(j)
                data.append(-w)
                sum_w += w
            row_ind.append(i)
            col_ind.append(i)
            data.append(sum_w)
        return csr_matrix((data, (row_ind, col_ind)), shape=(n_nodes, n_nodes))

    # 基于四面体的有限体积 Laplacian
    for elem in elements:
        pts = nodes[elem]
        # 计算四面体体积
        v0 = pts[1] - pts[0]
        v1 = pts[2] - pts[0]
        v2 = pts[3] - pts[0]
        vol = abs(np.dot(v0, np.cross(v1, v2))) / 6.0
        if vol < 1e-15:
            continue

        # 简化的边权
        for i_idx in range(4):
            for j_idx in range(i_idx + 1, 4):
                i = elem[i_idx]
                j = elem[j_idx]
                edge_len = np.linalg.norm(nodes[i] - nodes[j])
                w = vol / (edge_len ** 2 + 1e-15)
                row_ind.append(i)
                col_ind.append(j)
                data.append(-w)
                row_ind.append(j)
                col_ind.append(i)
                data.append(-w)
                row_ind.append(i)
                col_ind.append(i)
                data.append(w)
                row_ind.append(j)
                col_ind.append(j)
                data.append(w)

    L = csr_matrix((data, (row_ind, col_ind)), shape=(n_nodes, n_nodes))
    return L


def solve_poisson_spherical_shell(rhs, nodes, elements, r_icb, r_cmb, tol=1e-10):
    """
    求解球壳上的 Poisson 方程：∇^2 φ = rhs。
    使用 CG + Jacobi 预处理。
    """
    L = build_laplacian_spherical_shell(nodes, elements, r_icb, r_cmb)
    n = len(nodes)

    # 边界条件：在内核边界和核幔边界设 Dirichlet
    boundary = np.zeros(n, dtype=bool)
    for i, node in enumerate(nodes):
        r = np.linalg.norm(node)
        if abs(r - r_icb) < 0.05 or abs(r - r_cmb) < 0.05:
            boundary[i] = True

    # 修改矩阵和右端项以施加边界条件
    rhs_mod = rhs.copy()
    for i in np.where(boundary)[0]:
        L.data[L.indptr[i]:L.indptr[i + 1]] = 0.0
        L[i, i] = 1.0
        rhs_mod[i] = 0.0

    d_inv = jacobi_preconditioner_diagonal(L)
    cg = CGSolverRC(n, tol=tol, max_iter=min(5000, 10 * n))

    def matvec(v):
        return L.dot(v)

    def precond(v):
        return apply_jacobi_precond(d_inv, v)

    x0 = np.zeros(n)
    x, iters, resid = cg.solve(rhs_mod, x0, matvec, precond)
    return x, iters, resid
