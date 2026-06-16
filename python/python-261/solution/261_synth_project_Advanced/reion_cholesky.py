"""
reion_cholesky.py
=================
Cholesky 分解与统计推断

本模块实现 Cholesky 分解及其在再电离参数不确定性量化中的应用.
核心算法来源于 Algorithm AS 6 (Healy 1968), 并扩展为子矩阵 Cholesky
(Remark AS R44, Freeman 1982).

功能:
  1. cholesky_factor : 对称正定矩阵的 Cholesky 分解 A = U^T U
  2. submatrix_cholesky : 按指定顺序对子矩阵分解
  3. sample_from_covariance : 从协方差矩阵抽样 (用于参数不确定性)
  4. log_determinant : 通过 Cholesky 计算对数行列式
  5. solve_spd : 用 Cholesky 解 SPD 系统

物理应用:
  再电离参数的后验协方差矩阵 Sigma (Fisher 矩阵的逆) 是 SPD 的,
  Cholesky 分解用于:
    - 快速计算 det(Sigma) (用于证据/模型比较)
    - 从先验/后验中抽样 (用于 Monte Carlo 不确定性传播)
    - 解线性系统 (用于最大似然估计)

对应种子项目:
  - 025_asa006 (Cholesky 分解 → 再电离参数协方差)
"""

import numpy as np


def cholesky_factor(A):
    """Cholesky 分解 A = U^T U, U 为上三角矩阵.

    实现 Algorithm AS 6 (Healy 1968), 带秩缺陷检测.

    Parameters
    ----------
    A : array [N x N]
        对称正定 (或半正定) 矩阵

    Returns
    -------
    U : array [N x N]
        上三角 Cholesky 因子, A ≈ U^T U
    nullity : int
        秩亏数 (0 = 满秩)
    ifault : int
        错误标志:
          0 = 成功
          1 = N < 1
          2 = 非正半定
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A 必须为方阵")
    N = A.shape[0]
    if N < 1:
        return np.zeros((0, 0)), 0, 1

    eta = 1.0e-9
    U = np.zeros((N, N))
    nullity = 0
    ifault = 0

    j_col = 0
    k_idx = 0

    for icol in range(N):
        j_col += icol
        x = eta * eta * A[icol, icol]
        l_idx = 0

        for irow in range(icol + 1):
            k_idx += 1
            w = A[irow, icol]
            m_idx = j_col

            for i in range(irow):
                l_idx += 1
                w -= U[i, irow] * U[i, icol] if irow < N and icol < N else 0.0
                m_idx += 1

            l_idx += 1

            if irow == icol:
                break

            if abs(U[irow, irow]) > 1.0e-30:
                U[irow, icol] = w / U[irow, irow]
            else:
                U[irow, icol] = 0.0
                diag_irow = A[irow, irow]
                if abs(x * diag_irow) < w * w:
                    ifault = 2
                    return U, nullity, ifault

        # 对角元素
        diag_val = w
        if abs(diag_val) <= abs(eta * A[icol, icol]):
            U[icol, icol] = 0.0
            nullity += 1
        else:
            if diag_val < 0.0:
                ifault = 2
                return U, nullity, ifault
            U[icol, icol] = np.sqrt(diag_val)

        j_col = j_col  # keep track

    return U, nullity, ifault


def cholesky_factor_simple(A):
    """简化版 Cholesky 分解 (使用标准递归公式).

    A = L L^T, L 为下三角矩阵.

    Parameters
    ----------
    A : array [N x N]
        对称正定矩阵

    Returns
    -------
    L : array [N x N]
        下三角 Cholesky 因子
    """
    A = np.asarray(A, dtype=float)
    N = A.shape[0]
    L = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                val = A[i, i] - s
                if val <= 0:
                    # 数值容差: 加小量
                    val = max(val, 1.0e-30)
                L[i, j] = np.sqrt(val)
            else:
                if abs(L[j, j]) < 1.0e-30:
                    L[i, j] = 0.0
                else:
                    L[i, j] = (A[i, j] - s) / L[j, j]
    return L


def submatrix_cholesky(A, order_indices):
    """按指定顺序对 A 的子矩阵进行 Cholesky 分解.

    实现 Remark AS R44 (Freeman 1982).

    Parameters
    ----------
    A : array [M x M]
        原始 SPD 矩阵
    order_indices : array-like
        选取的行列索引 (按此顺序分解)

    Returns
    -------
    U : array [N x N]
        上三角因子
    nullity : int
    ifault : int
    det : float
        行列式
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(order_indices, dtype=int)
    N = len(b)

    U = np.zeros((N, N))
    eta = 1.0e-9
    ifault = 0
    nullity = 0
    det = 1.0

    if N < 1:
        return U, nullity, 1, det

    j_col = 0
    for icol in range(N):
        ii = b[icol]
        x = eta * eta * A[ii, ii]
        l_idx = 0

        for irow in range(icol + 1):
            k_idx_val = A[b[irow], b[irow]]
            w = A[b[irow], ii]
            m_idx = j_col

            for i in range(irow):
                l_idx += 1
                w -= U[i, irow] * U[i, icol]
                m_idx += 1

            l_idx += 1

            if irow == icol:
                break

            if abs(U[irow, irow]) > 1.0e-30:
                U[irow, icol] = w / U[irow, irow]
            else:
                if abs(x * A[b[irow], b[irow]]) < w * w:
                    ifault = 2
                    return U, nullity, ifault, det
                U[irow, icol] = 0.0

        # 对角
        if abs(w) <= abs(eta * A[b[irow], b[irow]]):
            U[icol, icol] = 0.0
            nullity += 1
        else:
            if w < 0:
                ifault = 2
                return U, nullity, ifault, det
            U[icol, icol] = np.sqrt(w)

        j_col += icol + 1
        det *= U[icol, icol] ** 2

    return U, nullity, ifault, det


def log_determinant(A):
    """通过 Cholesky 计算 SPD 矩阵的对数行列式.

    log det(A) = 2 * sum_i log(L_{ii})

    Parameters
    ----------
    A : array [N x N]
        SPD 矩阵

    Returns
    -------
    log_det : float
    """
    L = cholesky_factor_simple(A)
    diag = np.diag(L)
    diag = np.maximum(diag, 1.0e-300)
    return 2.0 * np.sum(np.log(diag))


def solve_spd(A, b):
    """用 Cholesky 分解解 SPD 系统 A x = b.

    步骤:
      1. A = L L^T
      2. 解 L y = b (前代)
      3. 解 L^T x = y (回代)

    Parameters
    ----------
    A : array [N x N]
    b : array [N]

    Returns
    -------
    x : array [N]
    """
    L = cholesky_factor_simple(A)
    N = L.shape[0]
    # 前代: L y = b
    y = np.zeros(N)
    for i in range(N):
        s = sum(L[i, j] * y[j] for j in range(i))
        y[i] = (b[i] - s) / max(L[i, i], 1.0e-30)
    # 回代: L^T x = y
    x = np.zeros(N)
    for i in range(N - 1, -1, -1):
        s = sum(L[j, i] * x[j] for j in range(i + 1, N))
        x[i] = (y[i] - s) / max(L[i, i], 1.0e-30)
    return x


def sample_from_covariance(mean, cov, n_samples, rng=None):
    """从多元高斯分布 N(mean, cov) 中抽样.

    使用 Cholesky 分解: x = mean + L @ z, z ~ N(0, I).

    Parameters
    ----------
    mean : array [N]
    cov : array [N x N]
    n_samples : int
    rng : numpy.random.Generator, optional

    Returns
    -------
    samples : array [n_samples, N]
    """
    if rng is None:
        rng = np.random.default_rng(42)
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    N = len(mean)

    # 使协方差严格 SPD (加小量)
    cov_reg = cov + 1.0e-10 * np.eye(N)
    # 对称化
    cov_reg = 0.5 * (cov_reg + cov_reg.T)

    L = cholesky_factor_simple(cov_reg)
    z = rng.standard_normal((N, n_samples))
    samples = mean[:, None] + L @ z
    return samples.T


def fisher_information_reionization(params, z_data, xHII_data,
                                    model_func, sigma_obs):
    """计算再电离参数的 Fisher 信息矩阵.

    F_{ij} = sum_k (1/sigma_k^2) *
             (dmu/dp_i)(z_k) * (dmu/dp_j)(z_k)

    Parameters
    ----------
    params : array [n_params]
    z_data : array [n_data]
    xHII_data : array [n_data]
    model_func : callable
        model(z, params) → xHII_model
    sigma_obs : array [n_data]

    Returns
    -------
    F : array [n_params x n_params]
    """
    n_params = len(params)
    n_data = len(z_data)
    delta = 1.0e-4

    # 数值梯度
    J = np.zeros((n_data, n_params))
    for j in range(n_params):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[j] += delta
        p_minus[j] -= delta
        m_plus = model_func(z_data, p_plus)
        m_minus = model_func(z_data, p_minus)
        J[:, j] = (m_plus - m_minus) / (2.0 * delta)

    # Fisher 矩阵
    W = np.diag(1.0 / np.maximum(sigma_obs**2, 1.0e-30))
    F = J.T @ W @ J
    # 对称化 + 正则化
    F = 0.5 * (F + F.T) + 1.0e-10 * np.eye(n_params)
    return F
