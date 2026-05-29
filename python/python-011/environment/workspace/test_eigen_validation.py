# -*- coding: utf-8 -*-
"""
test_eigen_validation.py
------------------------
具有预设特征结构的测试矩阵生成器，用于验证本征值求解器的数值精度。

对应种子项目：1206_test_eigen
核心算法：
  - Householder 变换构造正交矩阵
  - Stewart QR 法生成随机正交矩阵 Q
  - 谱相似构造 A = Q * diag(lambda) * Q^T（对称）
  - Schur 相似构造 A = Q^T * T * Q（非对称，T 上三角）
  - 直方图分箱统计

在超导问题中用于：验证 Bogoliubov-de Gennes 哈密顿量对角化器的精度。
"""

import numpy as np


def householder_column(a_vec, k):
    """
    构造 Householder 向量 v，使得 H = I - 2 v v^T / (v^T v) 能将
    a_vec[k+1:] 全部置零。

    标准公式：
        x = a_vec[k:]
        s = sign(x_0) * ||x||
        v_0 = x_0 + s,   v_i = x_i (i>0)
    """
    n = a_vec.size
    v = np.zeros(n)
    v[k:] = a_vec[k:].copy()
    norm_x = np.linalg.norm(v[k:])
    if norm_x < 1e-15:
        return v
    if v[k] >= 0:
        s = norm_x
    else:
        s = -norm_x
    v[k] += s
    # 归一化
    norm_v = np.linalg.norm(v[k:])
    if norm_v > 1e-15:
        v[k:] /= norm_v
    return v


def apply_householder_right(a, v):
    """
    右乘 Householder 反射器：A -> A * H，其中 H = I - 2 v v^T。
    等价于 A - 2 (A v) v^T。
    """
    av = a @ v
    return a - 2.0 * np.outer(av, v)


def random_orthogonal_matrix(n):
    """
    Stewart QR 法生成均匀分布的随机正交矩阵。
    对标准正态随机矩阵做 QR 分解并调整符号。
    """
    if n <= 0:
        return np.zeros((0, 0))
    X = np.random.randn(n, n)
    Q, R = np.linalg.qr(X)
    # 调整使得对角元为正以保证唯一性（接近均匀）
    diag_r = np.diag(R)
    signs = np.where(diag_r >= 0, 1.0, -1.0)
    Q = Q * signs[np.newaxis, :]
    return Q


def generate_symmetric_test_matrix(n, lambda_mean=0.0, lambda_std=1.0):
    """
    生成对称测试矩阵 A = Q * diag(lambda) * Q^T，
    其中 lambda ~ N(lambda_mean, lambda_std^2)，Q 随机正交。

    返回 A, Q, lambda_exact。
    由于 A 对称，其特征值精确等于 lambda。
    """
    if n <= 0:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0)
    lam = np.random.normal(lambda_mean, lambda_std, size=n)
    Q = random_orthogonal_matrix(n)
    A = Q @ np.diag(lam) @ Q.T
    return A, Q, lam


def generate_nonsymmetric_test_matrix(n, lambda_mean=0.0, lambda_std=1.0):
    """
    生成非对称测试矩阵 A = Q^T * T * Q，
    其中 T 为上三角矩阵，对角元 lambda ~ N(lambda_mean, lambda_std^2)，
    严格上三角元为随机数。

    A 的特征值精确等于 T 的对角元 lambda。
    """
    if n <= 0:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0)
    lam = np.random.normal(lambda_mean, lambda_std, size=n)
    T = np.triu(np.random.randn(n, n) * 0.5, k=1)
    np.fill_diagonal(T, lam)
    Q = random_orthogonal_matrix(n)
    A = Q.T @ T @ Q
    return A, Q, lam


def validate_eigensolver(A, lam_exact, method='numpy'):
    """
    验证本征值求解器：计算相对误差和最大偏差。

    返回字典包含 max_abs_err, mean_rel_err, residual_norm。
    """
    if A.size == 0:
        return {'max_abs_err': 0.0, 'mean_rel_err': 0.0, 'residual_norm': 0.0}
    if method == 'numpy':
        lam_computed = np.linalg.eigvals(A)
    else:
        raise ValueError("不支持的求解器方法。")
    # 排序后比较
    lam_exact_s = np.sort(np.real(lam_exact))
    lam_comp_s = np.sort(np.real(lam_computed))
    abs_err = np.abs(lam_exact_s - lam_comp_s)
    max_abs_err = np.max(abs_err)
    rel_err = safe_divide(abs_err, np.abs(lam_exact_s) + 1e-15)
    mean_rel_err = np.mean(rel_err)
    # 残差范数 ||A - Q Lambda Q^{-1}||
    if A.shape[0] <= 200:
        w, v = np.linalg.eig(A)
        try:
            v_inv = np.linalg.inv(v)
            recon = v @ np.diag(w) @ v_inv
            res_norm = np.linalg.norm(A - recon, ord='fro')
        except np.linalg.LinAlgError:
            res_norm = np.inf
    else:
        res_norm = None
    return {
        'max_abs_err': max_abs_err,
        'mean_rel_err': mean_rel_err,
        'residual_norm': res_norm
    }


def r8vec_bin(data, nbin, a_min=None, a_max=None):
    """
    将向量数据分入 nbin 个等宽 bin，返回每个 bin 的计数和中心值。
    """
    data = np.asarray(data, dtype=float)
    if a_min is None:
        a_min = np.min(data)
    if a_max is None:
        a_max = np.max(data)
    if a_max <= a_min:
        a_max = a_min + 1.0
    counts = np.zeros(nbin, dtype=int)
    width = (a_max - a_min) / nbin
    idx = np.floor((data - a_min) / width).astype(int)
    idx = np.clip(idx, 0, nbin - 1)
    for i in idx:
        counts[i] += 1
    centers = a_min + (np.arange(nbin) + 0.5) * width
    return centers, counts


def safe_divide(a, b, eps=1e-15):
    """局部安全除法。"""
    return np.where(np.abs(b) > eps, a / b, 0.0)
