# -*- coding: utf-8 -*-
"""
linear_solvers.py
线性系统求解器：共轭梯度法、重启 GMRES、Cholesky 分解。
用于求解化学-输运耦合模型中产生的稀疏/稠密线性系统。

融合来源：
  - 149_cg: 共轭梯度法（CG）
  - 760_mgmres: 重启 GMRES
  - 026_asa007: Cholesky 分解
"""

import numpy as np
from numpy.linalg import norm


def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=None):
    r"""
    共轭梯度法（CG）求解对称正定线性系统：

        A \mathbf{x} = \mathbf{b}

    迭代格式：

        \mathbf{r}_0 = \mathbf{b} - A \mathbf{x}_0,
        \mathbf{p}_0 = \mathbf{r}_0,
        \alpha_k = \frac{\mathbf{r}_k^T \mathbf{r}_k}{\mathbf{p}_k^T A \mathbf{p}_k},
        \mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{p}_k,
        \mathbf{r}_{k+1} = \mathbf{r}_k - \alpha_k A \mathbf{p}_k,
        \beta_k = \frac{\mathbf{r}_{k+1}^T \mathbf{r}_{k+1}}{\mathbf{r}_k^T \mathbf{r}_k},
        \mathbf{p}_{k+1} = \mathbf{r}_{k+1} + \beta_k \mathbf{p}_k

    Parameters
    ----------
    A : ndarray or callable
        若 ndarray，则为系数矩阵；若 callable，则为矩阵-向量乘积函数 A(v)。
    b : ndarray, shape (n,)
    x0 : ndarray, optional
        初始猜测，默认零向量。
    tol : float
        相对残差容差。
    max_iter : int, optional
        最大迭代次数，默认 n。

    Returns
    -------
    x : ndarray
        近似解。
    residual : float
        最终残差范数。
    iters : int
        实际迭代次数。
    """
    b = np.asarray(b, dtype=float).ravel()
    n = b.size
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).ravel().copy()
    if max_iter is None:
        max_iter = n

    if callable(A):
        Av = A
    else:
        A_mat = np.asarray(A, dtype=float)
        def Av(v):
            return A_mat.dot(v)

    r = b - Av(x)
    p = r.copy()
    rsold = float(r.dot(r))
    bnorm = norm(b)
    if bnorm < 1e-15:
        bnorm = 1.0

    for k in range(max_iter):
        Ap = Av(p)
        pAp = float(p.dot(Ap))
        if abs(pAp) < 1e-20:
            break
        alpha = rsold / pAp
        x += alpha * p
        r -= alpha * Ap
        rsnew = float(r.dot(r))
        if np.sqrt(rsnew) / bnorm < tol:
            return x, np.sqrt(rsnew), k + 1
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

    return x, norm(r), max_iter


def mult_givens(c, s, k, g):
    r"""
    Givens 旋转应用于向量 g：

        \begin{bmatrix} c & -s \\ s & c \end{bmatrix}
        \begin{bmatrix} g_k \\ g_{k+1} \end{bmatrix}

    Parameters
    ----------
    c, s : float
        旋转参数。
    k : int
        0-based 起始索引。
    g : ndarray

    Returns
    -------
    g : ndarray
    """
    g = np.asarray(g, dtype=float).copy()
    g1 = c * g[k] - s * g[k + 1]
    g2 = s * g[k] + c * g[k + 1]
    g[k] = g1
    g[k + 1] = g2
    return g


def gmres_restart(Ax, b, x0=None, max_iter=100, restart=30, tol_abs=1e-10, tol_rel=1e-6):
    r"""
    重启 GMRES 求解一般线性系统：

        A \mathbf{x} = \mathbf{b}

    使用 Krylov 子空间 K_m = \text{span}\{r_0, A r_0, \dots, A^{m-1} r_0\}
    并通过 Arnoldi 过程构造 Hessenberg 矩阵 H_m。

    Parameters
    ----------
    Ax : callable
        矩阵-向量乘积函数，输入 ndarray，输出 ndarray。
    b : ndarray, shape (n,)
    x0 : ndarray, optional
    max_iter : int
        外迭代最大次数。
    restart : int
        Krylov 子空间维数 m（restart）。
    tol_abs, tol_rel : float
        绝对与相对残差容差。

    Returns
    -------
    x : ndarray
    residual : float
    iters : int
    """
    b = np.asarray(b, dtype=float).ravel()
    n = b.size
    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).ravel().copy()

    if restart > n:
        restart = n

    delta = 0.001
    r = b - Ax(x)
    rho = norm(r)
    bnorm = norm(b)
    if bnorm < 1e-15:
        bnorm = 1.0
    rho_tol = rho * tol_rel

    itr_used = 0

    for itr in range(max_iter):
        r = b - Ax(x)
        rho = norm(r)
        if rho <= rho_tol and rho <= tol_abs:
            break

        v = np.zeros((n, restart + 1))
        v[:, 0] = r / rho if rho > 1e-15 else r
        g = np.zeros(restart + 1)
        g[0] = rho
        H = np.zeros((restart + 1, restart))
        c = np.zeros(restart)
        s = np.zeros(restart)

        for k in range(restart):
            w = Ax(v[:, k])
            av = norm(w)
            for j in range(k + 1):
                H[j, k] = np.dot(v[:, j], w)
                w -= H[j, k] * v[:, j]
            H[k + 1, k] = norm(w)

            if av + delta * H[k + 1, k] == av:
                for j in range(k + 1):
                    htmp = np.dot(v[:, j], w)
                    H[j, k] += htmp
                    w -= htmp * v[:, j]
                H[k + 1, k] = norm(w)

            if H[k + 1, k] > 1e-15:
                v[:, k + 1] = w / H[k + 1, k]

            y = H[:k + 2, k].copy()
            for j in range(k):
                y = mult_givens(c[j], s[j], j, y)
            H[:k + 2, k] = y

            mu = np.sqrt(H[k, k] ** 2 + H[k + 1, k] ** 2)
            if mu < 1e-15:
                mu = 1.0
            c[k] = H[k, k] / mu
            s[k] = -H[k + 1, k] / mu
            H[k, k] = c[k] * H[k, k] - s[k] * H[k + 1, k]
            H[k + 1, k] = 0.0
            g = mult_givens(c[k], s[k], k, g)
            rho = abs(g[k + 1])
            itr_used += 1

            if rho <= rho_tol and rho <= tol_abs:
                break

        k_use = k if (rho <= rho_tol and rho <= tol_abs) else restart - 1
        y = np.zeros(k_use + 1)
        y[k_use] = g[k_use] / H[k_use, k_use]
        for i in range(k_use - 1, -1, -1):
            y[i] = (g[i] - np.dot(H[i, i + 1:k_use + 1], y[i + 1:k_use + 1])) / H[i, i]
        x += v[:, :k_use + 1].dot(y)

        if rho <= rho_tol and rho <= tol_abs:
            break

    return x, rho, itr_used


def cholesky_factor(a, n, eta=1e-09):
    r"""
    Cholesky 分解：对对称正定矩阵 A，计算上三角矩阵 U 使得

        A = U^T U

    输入 A 按行存储下三角部分为长度为 n(n+1)/2 的一维数组：
    a[0]=A_{11}, a[1]=A_{21}, a[2]=A_{22}, a[3]=A_{31}, ...

    Parameters
    ----------
    a : ndarray
        压缩存储的 SPD 矩阵元素。
    n : int
        矩阵阶数。
    eta : float
        数值稳定性参数。

    Returns
    -------
    u : ndarray
        Cholesky 因子（按列存储上三角部分）。
    nullty : int
        秩亏量。
    ifault : int
        0=成功, 1=n<1, 2=非半正定, 3=数组过小。
    """
    nn = len(a)
    req = n * (n + 1) // 2
    if n <= 0:
        return np.array([]), 0, 1
    if nn < req:
        return np.array([]), 0, 3

    u = np.zeros(req, dtype=float)
    nullty = 0
    j = 0
    k = 0
    ii = 0

    for icol in range(n):
        ii += icol + 1
        x = eta * eta * a[ii - 1]
        l = 0
        kk = 0
        for irow in range(icol + 1):
            kk += irow + 1
            k += 1
            w = a[k - 1]
            m = j
            for i in range(irow):
                l += 1
                w -= u[l - 1] * u[m]
                m += 1
            l += 1
            if irow == icol:
                break
            if abs(u[l - 1]) > 1e-15:
                u[k - 1] = w / u[l - 1]
            else:
                u[k - 1] = 0.0
                if abs(x * a[k - 1]) < w * w:
                    return np.array([]), 0, 2
        if abs(w) <= abs(eta * a[k - 1]):
            u[k - 1] = 0.0
            nullty += 1
        else:
            if w < 0.0:
                return np.array([]), 0, 2
            u[k - 1] = np.sqrt(w)
        j += icol + 1

    return u, nullty, 0


def cholesky_solve_dense(A, b):
    r"""
    使用稠密 Cholesky 分解求解 SPD 系统 A x = b。

    Parameters
    ----------
    A : ndarray, shape (n, n)
        对称正定矩阵。
    b : ndarray, shape (n,)

    Returns
    -------
    x : ndarray
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = A.shape[0]
    # 压缩下三角
    a = np.zeros(n * (n + 1) // 2)
    idx = 0
    for i in range(n):
        for j in range(i + 1):
            a[idx] = A[i, j]
            idx += 1
    u, nullty, ifault = cholesky_factor(a, n)
    if ifault != 0:
        # 回退到 numpy
        return np.linalg.solve(A, b)

    # 从压缩 u 重建上三角 U
    U = np.zeros((n, n))
    idx = 0
    for j in range(n):
        for i in range(j + 1):
            U[i, j] = u[idx]
            idx += 1

    # 解 U^T y = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= U[j, i] * y[j]
        if abs(U[i, i]) < 1e-15:
            y[i] = 0.0
        else:
            y[i] /= U[i, i]

    # 解 U x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U[i, j] * x[j]
        if abs(U[i, i]) < 1e-15:
            x[i] = 0.0
        else:
            x[i] /= U[i, i]
    return x
