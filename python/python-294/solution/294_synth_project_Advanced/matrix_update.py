"""
matrix_update.py - Sherman-Morrison 矩阵更新

本模块实现 Sherman-Morrison 公式及其在激光等离子体仿真中的应用。

在 Vlasov-Maxwell 仿真中，介电常数随时间变化（由于电子密度
和速度的演化），导致系统矩阵需要频繁更新。Sherman-Morrison 公式
允许在低秩更新下高效求解线性系统，避免重复进行完整的 LU 分解。

Sherman-Morrison 公式:

  设 B = A - u * v^T, 其中 A 为 N×N 可逆矩阵, u, v 为 N 维向量。
  则 B 的逆为:
      B^{-1} = A^{-1} + alpha * (A^{-1} u) * (v^T A^{-1}) / (1 - v^T A^{-1} u)
  其中 alpha = 1, 且标量分母:
      beta = 1 - v^T * A^{-1} * u ≠ 0

  对于线性系统 B*x = b:
      1. 求解 A*w = u  ->  w = A^{-1} * u
      2. 求解 A*y = b  ->  y = A^{-1} * b
      3. 计算 alpha = (v^T * y) / (1 - v^T * w)
      4. x = y + alpha * w

  计算复杂度: O(N^2) (相比完整 LU 分解的 O(N^3))

物理应用场景:
  在激光等离子体相互作用中，Maxwell 方程的离散化导致:
      A(t) * E(t+dt) = rhs(t)
  其中 A(t) = I - dt^2 * c^2 * nabla^2 + omega_p^2(t) * dt^2 * I

  当等离子体密度变化较小时:
      A(t+dt) ≈ A(t) + delta_omega_p^2 * dt^2 * I
  这是一个秩-1 更新 (如果密度变化集中在一个空间区域),
  可以用 Sherman-Morrison 高效处理。
"""

import numpy as np


def lu_factorize(A):
    """LU 分解 (部分主元法)。

    实现 Doolittle LU 分解:
        P * A = L * U
    其中 P 为置换矩阵, L 为单位下三角矩阵, U 为上三角矩阵。

    分解复杂度: O(N^3/3)

    Parameters
    ----------
    A : ndarray (N, N)
        待分解矩阵

    Returns
    -------
    LU : ndarray (N, N)
        紧凑存储的 L 和 U (L 的对角线隐含为 1)
    piv : ndarray (N,)
        主元索引
    """
    n = len(A)
    LU = A.astype(float).copy()
    piv = np.arange(n)

    for k in range(n - 1):
        # 部分主元: 选最大元素
        max_idx = k + np.argmax(np.abs(LU[k:, k]))
        if max_idx != k:
            LU[[k, max_idx]] = LU[[max_idx, k]]
            piv[[k, max_idx]] = piv[[max_idx, k]]

        if abs(LU[k, k]) < 1.0e-30:
            continue

        # 消元
        for i in range(k + 1, n):
            LU[i, k] /= LU[k, k]
            LU[i, k + 1:] -= LU[i, k] * LU[k, k + 1:]

    return LU, piv


def lu_solve(LU, piv, b):
    """使用 LU 分解求解线性系统。

    求解 A*x = b, 其中 P*A = L*U:
        1. 应用置换: b' = P*b
        2. 前代: L*y = b'
        3. 回代: U*x = y

    Parameters
    ----------
    LU : ndarray (N, N)
        lu_factorize 返回的紧凑 LU 矩阵
    piv : ndarray (N,)
        主元索引
    b : ndarray (N,)
        右端向量

    Returns
    -------
    x : ndarray (N,)
        解向量
    """
    n = len(b)
    # 应用置换
    b_perm = b[piv].astype(float)

    # 前代: L * y = b_perm
    y = b_perm.copy()
    for i in range(1, n):
        y[i] -= np.dot(LU[i, :i], y[:i])

    # 回代: U * x = y
    x = y.copy()
    for i in range(n - 1, -1, -1):
        if abs(LU[i, i]) < 1.0e-30:
            x[i] = 0.0
        else:
            x[i] = (x[i] - np.dot(LU[i, i + 1:], x[i + 1:])) / LU[i, i]

    return x


def lu_transpose_solve(LU, piv, b):
    """求解 A^T * x = b。

    用于 Sherman-Morrison 公式中需要求解 A^T * w = v 的场景。

    A^T = U^T * L^T * P^T, 所以:
        1. 前代: U^T * y = b
        2. 回代: L^T * z = y
        3. 逆置换: x = P^T * z

    Parameters
    ----------
    LU : ndarray (N, N)
        LU 分解结果
    piv : ndarray (N,)
        主元索引
    b : ndarray (N,)
        右端向量

    Returns
    -------
    x : ndarray (N,)
        A^T * x = b 的解
    """
    n = len(b)
    # 前代: U^T * y = b
    y = b.astype(float).copy()
    for i in range(n):
        if abs(LU[i, i]) < 1.0e-30:
            y[i] = 0.0
        else:
            y[i] = (y[i] - np.dot(LU[:i, i], y[:i])) / LU[i, i]

    # 回代: L^T * z = y
    z = y.copy()
    for i in range(n - 2, -1, -1):
        z[i] -= np.dot(LU[i + 1:, i], z[i + 1:])

    # 逆置换
    x = np.zeros(n)
    x[piv] = z
    return x


def sherman_morrison_solve(LU, piv, u, v, b):
    """Sherman-Morrison 公式求解秩-1 更新系统。

    求解 (A - u * v^T) * x = b:

        Step 1: w = A^{-1} * u   (通过 LU 前代回代)
        Step 2: y = A^{-1} * b
        Step 3: beta = 1 - v^T * w
        Step 4: 如果 |beta| < eps, 更新矩阵奇异
        Step 5: alpha = v^T * y / beta
        Step 6: x = y + alpha * w

    复杂度: O(N^2) 相比完整重分解的 O(N^3)

    Parameters
    ----------
    LU : ndarray (N, N)
        原始矩阵 A 的 LU 分解
    piv : ndarray (N,)
        主元索引
    u : ndarray (N,)
        左更新向量
    v : ndarray (N,)
        右更新向量
    b : ndarray (N,)
        右端向量

    Returns
    -------
    x : ndarray (N,)
        解向量
    beta : float
        分母标量 (接近 0 表示更新使矩阵接近奇异)
    info : dict
        中间计算结果
    """
    # Step 1: A*w = u
    w = lu_solve(LU, piv, u)

    # Step 2: A*y = b
    y = lu_solve(LU, piv, b)

    # Step 3: beta = 1 - v^T * w
    beta = 1.0 - np.dot(v, w)

    # Step 4: 检查奇异性
    if abs(beta) < 1.0e-14:
        info = {'beta': beta, 'condition': 'near-singular', 'used_fallback': True}
        # 回退到直接构造更新矩阵求解
        n = len(b)
        A_updated = np.zeros((n, n))
        # 重建 A 并更新
        A_orig = reconstruct_from_lu(LU, piv)
        A_updated = A_orig - np.outer(u, v)
        try:
            x = np.linalg.solve(A_updated, b)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(A_updated, b, rcond=None)[0]
        return x, beta, info

    # Step 5: alpha = v^T * y / beta
    alpha = np.dot(v, y) / beta

    # Step 6: x = y + alpha * w
    x = y + alpha * w

    info = {
        'beta': beta,
        'alpha': alpha,
        'norm_w': np.linalg.norm(w),
        'norm_y': np.linalg.norm(y),
        'condition': 'ok',
        'used_fallback': False,
    }
    return x, beta, info


def sherman_morrison_mv(A, u, v, x):
    """Sherman-Morrison 矩阵-向量乘法。

    计算 b = (A - u * v^T) * x = A*x - u * (v^T * x)

    Parameters
    ----------
    A : ndarray (N, N)
        原始矩阵
    u : ndarray (N,)
        左更新向量
    v : ndarray (N,)
        右更新向量
    x : ndarray (N,)
        输入向量

    Returns
    -------
    b : ndarray (N,)
        结果向量
    """
    return A @ x - u * np.dot(v, x)


def reconstruct_from_lu(LU, piv):
    """从 LU 分解重建原始矩阵 A。

    Parameters
    ----------
    LU : ndarray (N, N)
        LU 分解结果
    piv : ndarray (N,)
        主元索引

    Returns
    -------
    A : ndarray (N, N)
        重建的矩阵
    """
    n = LU.shape[0]
    L = np.tril(LU, -1) + np.eye(n)
    U = np.triu(LU)
    A_permuted = L @ U

    # 逆置换
    A = np.zeros_like(A_permuted)
    A[piv, :] = A_permuted
    return A


def plasma_dielectric_update_system(N_x, omega_p_sq_old, omega_p_sq_new, dx):
    """构建等离子体介电常数更新向量。

    在 Vlasov-Maxwell 仿真中，等离子体频率平方的变化导致
    Maxwell 方程系统矩阵的秩-1 (或低秩) 更新。

    系统矩阵: A = I - c^2 * dt^2 * D2 + omega_p^2 * dt^2 * I
    更新: delta_A = (omega_p_new^2 - omega_p_old^2) * dt^2 * diag(n_profile)

    如果密度变化集中在一个位置 x_c (如临界密度面附近),
    则 delta_A 近似为秩-1:
        u = sqrt(|delta_omega_p^2|) * dt * e_c
        v = sign(delta_omega_p^2) * sqrt(|delta_omega_p^2|) * dt * e_c

    Parameters
    ----------
    N_x : int
        空间格点数
    omega_p_sq_old : ndarray
        旧等离子体频率平方分布
    omega_p_sq_new : ndarray
        新等离子体频率平方分布
    dx : float
        网格间距

    Returns
    -------
    u : ndarray
        左更新向量
    v : ndarray
        右更新向量
    max_change : float
        最大变化量
    """
    delta = omega_p_sq_new - omega_p_sq_old
    max_change = np.max(np.abs(delta))

    # 找到最大变化位置
    idx_max = np.argmax(np.abs(delta))

    # 构建秩-1 近似
    sign_delta = np.sign(delta[idx_max]) if delta[idx_max] != 0 else 1.0
    sqrt_abs_delta = np.sqrt(np.abs(delta))

    u = sqrt_abs_delta
    v = sign_delta * sqrt_abs_delta

    return u, v, max_change
