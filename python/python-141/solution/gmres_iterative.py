"""
重启GMRES迭代求解器模块
=========================
基于种子项目 760_mgmres 的核心算法改造。

在金融工程中，Heston随机波动率PDE经ADI（交替方向隐式）离散化后，
每一步需要求解大型稀疏线性系统。GMRES（广义最小残差法）是求解
非对称稀疏系统的首选Krylov子空间方法。

数学背景:
---------
对于线性系统 A·x = b，GMRES在Krylov子空间:
    K_m(A, r_0) = span{r_0, A r_0, A² r_0, ..., A^{m-1} r_0}
中寻找使残差 ||b - A·x||₂ 最小的近似解。

Arnoldi迭代构造正交基 V_m = [v_1, ..., v_m] 和上Hessenberg矩阵 H_m:
    A V_m = V_{m+1} H̃_m

最小化问题转化为:
    min ||β e_1 - H̃_m · y||₂
    x_m = x_0 + V_m · y

通过Givens旋转将 H̃_m 逐步QR分解，可高效更新残差。
当内存受限时，采用 restarted GMRES：每m步以当前解为初值重新开始。

参考:
    Saad, "Iterative Methods for Sparse Linear Systems", 2nd ed., SIAM, 2003.
"""

import numpy as np


def mult_givens(c, s, k, g):
    """
    对向量g的第k和k+1个元素施加Givens旋转。

    旋转矩阵:
        G = [ c  -s ]
            [ s   c ]

    [g_k    ]   [ c  -s ] [g_k    ]
    [g_{k+1}] = [ s   c ] [g_{k+1}]
    """
    g = np.asarray(g, dtype=np.float64).copy()
    g1 = c * g[k] - s * g[k + 1]
    g2 = s * g[k] + c * g[k + 1]
    g[k] = g1
    g[k + 1] = g2
    return g


def sparse_mv(a, ia, ja, x, n, nz_num):
    """
    稀疏矩阵-向量乘法: w = A · x

    A通过COO-like三元组存储:
        a[k]  = A[ia[k], ja[k]]
    """
    w = np.zeros(n, dtype=np.float64)
    for k in range(nz_num):
        i = ia[k]
        j = ja[k]
        w[i] += a[k] * x[j]
    return w


def restarted_gmres(a, ia, ja, x0, rhs, n, nz_num, itr_max, mr, tol_abs, tol_rel, verbose=0):
    """
    重启GMRES求解器。

    参数:
    ------
    a, ia, ja : 稀疏矩阵的COO三元组
    x0        : ndarray, 初始猜测
    rhs       : ndarray, 右端项
    n         : int, 系统阶数
    nz_num    : int, 非零元个数
    itr_max   : int, 最大外迭代次数
    mr        : int, 每次重启的最大内迭代次数 (0 < mr <= n)
    tol_abs   : float, 绝对残差容限
    tol_rel   : float, 相对残差容限 (相对于初始残差)
    verbose   : int, 输出级别 (0=静默, 1=每步输出)

    返回:
    ------
    x         : ndarray, 解向量
    converged : bool, 是否收敛
    itr_used  : int, 总迭代次数
    final_res : float, 最终残差范数
    """
    if n <= 0:
        raise ValueError("n必须为正整数")
    if mr <= 0 or mr > n:
        raise ValueError("mr必须满足 0 < mr <= n")
    if nz_num < 0:
        raise ValueError("nz_num不能为负")

    delta = 0.001
    x = np.asarray(x0, dtype=np.float64).copy()
    rhs = np.asarray(rhs, dtype=np.float64)

    if len(x) != n or len(rhs) != n:
        raise ValueError("向量维度必须与n一致")

    itr_used = 0
    rho_tol = None

    for itr in range(1, itr_max + 1):
        # 计算残差 r = b - A x
        r = rhs - sparse_mv(a, ia, ja, x, n, nz_num)
        rho = np.linalg.norm(r)

        if verbose:
            print(f"  ITR = {itr:8d}  Residual = {rho:.6e}")

        if itr == 1:
            rho_tol = rho * tol_rel

        # 初始化Krylov基
        v = np.zeros((n, mr + 1), dtype=np.float64)
        if rho < 1e-30:
            converged = True
            final_res = rho
            return x, converged, itr_used, final_res
        v[:, 0] = r / rho

        g = np.zeros(mr + 1, dtype=np.float64)
        g[0] = rho
        h = np.zeros((mr + 1, mr), dtype=np.float64)
        c = np.zeros(mr, dtype=np.float64)
        s = np.zeros(mr, dtype=np.float64)

        k_copy = 0
        for k in range(mr):
            k_copy = k
            # Arnoldi步: v_{k+1} = A v_k
            v[:, k + 1] = sparse_mv(a, ia, ja, v[:, k], n, nz_num)
            av_norm = np.linalg.norm(v[:, k + 1])

            # Gram-Schmidt正交化
            for j in range(k + 1):
                h[j, k] = np.dot(v[:, j], v[:, k + 1])
                v[:, k + 1] -= h[j, k] * v[:, j]

            h[k + 1, k] = np.linalg.norm(v[:, k + 1])

            # 重正交化（数值稳定性）
            if av_norm + delta * h[k + 1, k] <= av_norm + 1e-15:
                for j in range(k + 1):
                    htmp = np.dot(v[:, j], v[:, k + 1])
                    h[j, k] += htmp
                    v[:, k + 1] -= htmp * v[:, j]
                h[k + 1, k] = np.linalg.norm(v[:, k + 1])

            if h[k + 1, k] > 1e-30:
                v[:, k + 1] /= h[k + 1, k]

            # 应用Givens旋转消去h[k+1,k]
            if k > 0:
                y = h[:k + 2, k].copy()
                for j in range(k):
                    y = mult_givens(c[j], s[j], j, y)
                h[:k + 2, k] = y

            mu = np.sqrt(h[k, k] ** 2 + h[k + 1, k] ** 2)
            if mu < 1e-30:
                mu = 1.0
            c[k] = h[k, k] / mu
            s[k] = -h[k + 1, k] / mu
            h[k, k] = c[k] * h[k, k] - s[k] * h[k + 1, k]
            h[k + 1, k] = 0.0
            g[:k + 2] = mult_givens(c[k], s[k], k, g[:k + 2])

            rho = abs(g[k + 1])
            itr_used += 1

            if verbose:
                print(f"  K =   {k + 1:8d}  Residual = {rho:.6e}")

            if rho <= rho_tol and rho <= tol_abs:
                break

        # 求解上三角系统 H(1:k+1, 1:k+1) · y = g(1:k+1)
        k_solve = k_copy
        if k_solve >= mr:
            k_solve = mr - 1
        y = np.zeros(k_solve + 1, dtype=np.float64)
        y[k_solve] = g[k_solve] / h[k_solve, k_solve]
        for i in range(k_solve - 1, -1, -1):
            y[i] = (g[i] - np.dot(h[i, i + 1:k_solve + 1], y[i + 1:k_solve + 1])) / h[i, i]

        # 更新解
        x += v[:, :k_solve + 1] @ y

        if rho <= rho_tol and rho <= tol_abs:
            converged = True
            final_res = rho
            return x, converged, itr_used, final_res

    converged = False
    final_res = rho
    return x, converged, itr_used, final_res


def gmres_dense(A, b, x0=None, tol=1e-10, maxiter=None, restart=None):
    """
    针对稠密矩阵的GMRES包装器（用于中小规模系统调试）。
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = A.shape[0]
    if x0 is None:
        x0 = np.zeros(n, dtype=np.float64)
    if maxiter is None:
        maxiter = n
    if restart is None:
        restart = min(n, 50)

    # 转换为COO三元组
    rowind, colind = np.nonzero(np.abs(A) > 1e-15)
    nz_num = len(rowind)
    a_vals = A[rowind, colind]

    x, converged, itr_used, final_res = restarted_gmres(
        a_vals, rowind.astype(np.int64), colind.astype(np.int64),
        x0, b, n, nz_num, maxiter, restart, tol, tol, verbose=0
    )
    return x, converged, itr_used, final_res
