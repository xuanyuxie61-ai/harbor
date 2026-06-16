"""
mode_decomposition.py - 电磁模式 Gram-Schmidt 分解

本模块使用 Gram-Schmidt 正交化方法对等离子体中的电磁场模式
进行正交分解。

在激光等离子体相互作用中，电磁场可以分解为一系列正交模式:
    E(x, t) = sum_n a_n(t) * phi_n(x)

其中 phi_n(x) 为正交模式函数, a_n(t) 为模式振幅。

Gram-Schmidt 正交化:

  经典 Gram-Schmidt (CGS):
    对于输入向量集 {a_1, a_2, ..., a_k}:
      u_1 = a_1
      q_1 = u_1 / ||u_1||
      for j = 2, ..., k:
          u_j = a_j - sum_{i=1}^{j-1} (a_j . q_i) * q_i
          q_j = u_j / ||u_j||

  修正 Gram-Schmidt (MGS):
    数值上更稳定的变体:
      for j = 2, ..., k:
          v_j = a_j
          for i = 1, ..., j-1:
              v_j = v_j - (v_j . q_i) * q_i   <-- 逐步修正
          q_j = v_j / ||v_j||

  数值稳定性:
    CGS 的浮点误差 ~ O(epsilon * kappa(A)^2)
    MGS 的浮点误差 ~ O(epsilon * kappa(A))
    其中 kappa(A) 为条件数, epsilon 为机器精度

  对于近线性相关的模式集, MGS 显著优于 CGS。
  当 ||u_j|| < tol 时, 认为第 j 个向量线性相关, 跳过。

物理应用:
  1. 从模拟输出的电场时间序列中提取正交模式
  2. 分析模式之间的能量分配
  3. 检验模式是否满足理论预测的正交关系
  4. 构建降阶模型 (ROM) 的基函数
"""

import numpy as np


def classical_gram_schmidt(A, tol=1.0e-14):
    """经典 Gram-Schmidt 正交化 (CGS)。

    输入矩阵 A 的列向量 {a_1, ..., a_k}, 计算正交归一基 {q_1, ..., q_k}。

    算法:
      for j = 1, ..., k:
          v_j = a_j
          for i = 1, ..., j-1:
              v_j = v_j - dot(a_j, q_i) * q_i    <-- 使用原始 a_j
          if ||v_j|| < tol:
              线性相关, 跳过
          else:
              q_j = v_j / ||v_j||

    Parameters
    ----------
    A : ndarray (M, N)
        输入矩阵 (M 维空间中的 N 个向量)
    tol : float
        线性相关性检测容差

    Returns
    -------
    Q : ndarray (M, K)
        正交归一基 (K <= N, 去掉线性相关向量)
    R : ndarray (N, N)
        上三角矩阵 (满足 A = Q*R)
    rank : int
        有效秩 K
    info : dict
        正交性检验信息
    """
    M, N = A.shape
    Q = np.zeros((M, N))
    R = np.zeros((N, N))
    independent = []

    for j in range(N):
        v = A[:, j].copy()

        # 减去在所有已计算基向量上的投影
        for i in independent:
            R[i, j] = np.dot(A[:, j], Q[:, i])
            v -= R[i, j] * Q[:, i]

        norm_v = np.linalg.norm(v)
        R[j, j] = norm_v

        if norm_v > tol:
            Q[:, j] = v / norm_v
            independent.append(j)

    K = len(independent)
    Q_out = Q[:, independent[:K]].copy() if K > 0 else np.zeros((M, 0))
    R_out = R[:K, :].copy()

    # 正交性检验
    if K > 0:
        orth_err = np.max(np.abs(Q_out.T @ Q_out - np.eye(K)))
    else:
        orth_err = 0.0

    info = {
        'rank': K,
        'orthogonality_error': orth_err,
        'is_orthogonal': orth_err < 1.0e-10,
        'rejected_vectors': N - K,
    }

    return Q_out, R_out, K, info


def modified_gram_schmidt(A, tol=1.0e-14):
    """修正 Gram-Schmidt 正交化 (MGS)。

    与 CGS 的区别: 使用逐步修正的 v (而非原始 a_j) 计算投影。

    算法:
      for j = 1, ..., k:
          v_j = a_j
          for i = 1, ..., j-1:
              R[i,j] = dot(v_j, q_i)     <-- 使用当前的 v_j
              v_j = v_j - R[i,j] * q_i
          if ||v_j|| < tol:
              线性相关, 跳过
          else:
              q_j = v_j / ||v_j||

    MGS 的数值稳定性比 CGS 好一个数量级 (对条件数)。

    Parameters
    ----------
    A : ndarray (M, N)
        输入矩阵
    tol : float
        线性相关性检测容差

    Returns
    -------
    Q : ndarray (M, K)
        正交归一基
    R : ndarray (N, N)
        上三角矩阵
    rank : int
        有效秩
    info : dict
        正交性检验信息
    """
    M, N = A.shape
    V = A.astype(float).copy()
    Q = np.zeros((M, N))
    R = np.zeros((N, N))
    independent = []

    for j in range(N):
        # 逐步减去投影 (使用当前 V[:, j])
        for i in independent:
            R[i, j] = np.dot(V[:, j], Q[:, i])
            V[:, j] -= R[i, j] * Q[:, i]

        norm_v = np.linalg.norm(V[:, j])
        R[j, j] = norm_v

        if norm_v > tol:
            Q[:, j] = V[:, j] / norm_v
            independent.append(j)

    K = len(independent)
    Q_out = Q[:, independent[:K]].copy() if K > 0 else np.zeros((M, 0))
    R_out = R[:K, :].copy()

    if K > 0:
        orth_err = np.max(np.abs(Q_out.T @ Q_out - np.eye(K)))
    else:
        orth_err = 0.0

    info = {
        'rank': K,
        'orthogonality_error': orth_err,
        'is_orthogonal': orth_err < 1.0e-10,
        'rejected_vectors': N - K,
    }

    return Q_out, R_out, K, info


def compare_cgs_mgs(A, tol_values=None):
    """比较 CGS 和 MGS 在不同容差下的表现。

    Parameters
    ----------
    A : ndarray (M, N)
        输入矩阵
    tol_values : list of float
        测试容差列表

    Returns
    -------
    comparison : dict
        比较结果
    """
    if tol_values is None:
        tol_values = [1e-8, 1e-10, 1e-12, 1e-14, 1e-16]

    comparison = {}
    for tol in tol_values:
        Q_cgs, _, K_cgs, info_cgs = classical_gram_schmidt(A, tol)
        Q_mgs, _, K_mgs, info_mgs = modified_gram_schmidt(A, tol)

        comparison[tol] = {
            'cgs_rank': K_cgs,
            'cgs_orth_err': info_cgs['orthogonality_error'],
            'mgs_rank': K_mgs,
            'mgs_orth_err': info_mgs['orthogonality_error'],
            'mgs_better': info_mgs['orthogonality_error'] < info_cgs['orthogonality_error'],
        }

    return comparison


def decompose_field_modes(E_snapshots, method='mgs', tol=1.0e-12):
    """对电场时间序列进行模式分解。

    将电场快照矩阵 E(x, t_n) 分解为正交空间模式和时间系数:
        E = sum_k sigma_k * phi_k(x) * psi_k(t)

    这等价于计算 E 的 QR 分解 (通过 Gram-Schmidt)。

    Parameters
    ----------
    E_snapshots : ndarray (N_x, N_t)
        电场快照矩阵 (空间 x 时间)
    method : str
        正交化方法 ('cgs' 或 'mgs')
    tol : float
        正交化容差

    Returns
    -------
    modes : dict
        模式分解结果
    """
    if method == 'cgs':
        Q, R, rank, info = classical_gram_schmidt(E_snapshots, tol)
    elif method == 'mgs':
        Q, R, rank, info = modified_gram_schmidt(E_snapshots, tol)
    else:
        raise ValueError(f"未知方法: {method}")

    # 模式能量 (R 对角线的平方)
    mode_energies = np.diag(R) ** 2
    total_energy = np.sum(mode_energies)
    if total_energy > 0:
        energy_fractions = mode_energies / total_energy
    else:
        energy_fractions = np.zeros_like(mode_energies)

    # 时间系数
    temporal = R  # R 的每一行对应一个模式的时间演化

    modes = {
        'spatial_modes': Q,
        'temporal_coefficients': temporal,
        'rank': rank,
        'mode_energies': mode_energies[:rank],
        'energy_fractions': energy_fractions[:rank],
        'total_energy': total_energy,
        'info': info,
    }

    return modes


def electromagnetic_mode_basis(x, omega_p, L_x, n_modes=10):
    """构建电磁模式的理论基函数。

    在均匀等离子体腔 (0, L_x) 中, 电磁模式的解析形式为:
        phi_n(x) = sqrt(2/L_x) * sin(k_n * x)
    其中 k_n = n * pi / L_x (驻波边界条件)

    模式频率: omega_n = sqrt(omega_p^2 + k_n^2)

    Parameters
    ----------
    x : ndarray
        空间坐标
    omega_p : float
        等离子体频率
    L_x : float
        腔长
    n_modes : int
        模式数量

    Returns
    -------
    basis : ndarray (N_x, n_modes)
        模式基函数矩阵
    omega_modes : ndarray (n_modes,)
        模式频率
    k_modes : ndarray (n_modes,)
        模式波数
    """
    N_x = len(x)
    basis = np.zeros((N_x, n_modes))
    k_modes = np.zeros(n_modes)
    omega_modes = np.zeros(n_modes)

    for n in range(1, n_modes + 1):
        k_n = n * np.pi / L_x
        k_modes[n - 1] = k_n
        omega_modes[n - 1] = np.sqrt(omega_p ** 2 + k_n ** 2)
        basis[:, n - 1] = np.sqrt(2.0 / L_x) * np.sin(k_n * x)

    return basis, omega_modes, k_modes
