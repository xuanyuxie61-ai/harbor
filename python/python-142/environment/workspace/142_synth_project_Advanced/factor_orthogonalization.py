"""
factor_orthogonalization.py
基于 Gram-Schmidt 正交化的宏观经济因子分解模块
应用于信用风险多因子模型中的因子载荷矩阵正交化

原项目映射: 480_gram_schmidt
科学问题: 在 CreditMetrics / Vasicek 多因子模型中，宏观经济因子往往高度共线。
通过修正 Gram-Schmidt (MGS) 正交化，将原始因子载荷矩阵转换为正交基，
消除多重共线性，提高违约相关性估计的数值稳定性。
"""

import numpy as np
from typing import Tuple, Optional


def modified_gram_schmidt(A: np.ndarray, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    修正 Gram-Schmidt 正交化
    将矩阵 A (n x m) 分解为 A = Q @ R，其中 Q 的列向量正交归一，R 为上三角矩阵

    数学公式:
        对 j = 1, ..., m:
            v_j = a_j
            对 i = 1, ..., j-1:
                R_{i,j} = <a_j, q_i>
                v_j = v_j - R_{i,j} * q_i
            R_{j,j} = ||v_j||
            若 R_{j,j} > tol:
                q_j = v_j / R_{j,j}

    Parameters:
        A: 输入矩阵 (n x m)，列代表宏观经济因子载荷
        tol: 线性相关判定容差

    Returns:
        Q: 正交归一矩阵 (n x m)
        R: 上三角矩阵 (m x m)
        rank: 有效秩
    """
    n, m = A.shape
    Q = np.zeros((n, m), dtype=float)
    R = np.zeros((m, m), dtype=float)
    rank = 0
    for j in range(m):
        v = A[:, j].copy()
        for i in range(rank):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
        norm_v = np.linalg.norm(v)
        if norm_v > tol:
            Q[:, rank] = v / norm_v
            R[rank, j] = norm_v
            rank += 1
    return Q[:, :rank], R[:rank, :], rank


def classical_gram_schmidt(A: np.ndarray, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    经典 Gram-Schmidt (CGS) 正交化
    数值稳定性较 MGS 差，但计算量略小，用于对比

    数学公式:
        对 j = 1, ..., m:
            对 i = 1, ..., j-1:
                R_{i,j} = <a_j, q_i>
            v_j = a_j - sum_{i<j} R_{i,j} * q_i
            R_{j,j} = ||v_j||
            q_j = v_j / R_{j,j}
    """
    n, m = A.shape
    Q = np.zeros((n, m), dtype=float)
    R = np.zeros((m, m), dtype=float)
    rank = 0
    for j in range(m):
        v = A[:, j].copy()
        for i in range(rank):
            R[i, j] = np.dot(Q[:, i], A[:, j])
        for i in range(rank):
            v -= R[i, j] * Q[:, i]
        norm_v = np.linalg.norm(v)
        if norm_v > tol:
            Q[:, rank] = v / norm_v
            R[rank, j] = norm_v
            rank += 1
    return Q[:, :rank], R[:rank, :], rank


def factor_covariance_from_loadings(B: np.ndarray, sigma_factor: np.ndarray) -> np.ndarray:
    """
    由因子载荷矩阵 B 和因子波动率 sigma_factor 计算资产间协方差矩阵

    信用风险多因子模型中，资产 i 的违约指示变量可表示为:
        X_i = sum_k B_{i,k} * F_k + sqrt(1 - sum_k B_{i,k}^2) * Z_i
    其中 F_k 为系统性因子，Z_i 为特质性冲击。

    资产 i 与 j 的违约相关性为:
        Corr(X_i, X_j) = sum_k B_{i,k} * B_{j,k}  (当因子标准正交时)

    Parameters:
        B: 因子载荷矩阵 (n_assets x n_factors)
        sigma_factor: 因子波动率 (n_factors,)

    Returns:
        Cov: 协方差矩阵 (n_assets x n_assets)
    """
    n_assets = B.shape[0]
    # 特质性方差
    idio_var = np.maximum(1.0 - np.sum(B**2, axis=1), 1e-12)
    # 系统性协方差
    sys_cov = B @ np.diag(sigma_factor**2) @ B.T
    # 加上特质性部分
    Cov = sys_cov + np.diag(idio_var)
    # 归一化确保对角线为 1
    d = np.sqrt(np.diag(Cov))
    D_inv = np.diag(1.0 / d)
    Corr = D_inv @ Cov @ D_inv
    return Corr


def orthogonalize_credit_factors(
    raw_loadings: np.ndarray,
    method: str = "mgs",
    tol: float = 1e-12
) -> np.ndarray:
    """
    对原始因子载荷矩阵进行正交化处理，返回正交因子载荷

    处理流程:
        1. 对原始载荷矩阵的转置做 Gram-Schmidt (因子间正交)
        2. 重新计算资产在正交因子空间中的投影系数
        3. 确保每行载荷的平方和不超过 1 (R^2 <= 1)

    Parameters:
        raw_loadings: 原始因子载荷 (n_assets x n_factors)
        method: "mgs" 或 "cgs"
        tol: 容差

    Returns:
        orth_loadings: 正交化后的因子载荷 (n_assets x rank)
    """
    n_assets, n_factors = raw_loadings.shape
    if method.lower() == "mgs":
        Q, R, rank = modified_gram_schmidt(raw_loadings.T, tol)
    else:
        Q, R, rank = classical_gram_schmidt(raw_loadings.T, tol)

    if rank == 0:
        return np.zeros((n_assets, 1))

    # Q 的列是原始因子空间的正交基 (n_factors x rank)
    # 资产在正交基上的投影: orth_loadings = raw_loadings @ Q
    orth_loadings = raw_loadings @ Q

    # 截断到合法区间
    row_norms = np.sqrt(np.sum(orth_loadings**2, axis=1))
    scale = np.where(row_norms > 1.0, 1.0 / row_norms, 1.0)
    orth_loadings = orth_loadings * scale[:, np.newaxis]

    return orth_loadings


def test_factor_orthogonalization():
    """内部测试: 验证正交化后相关性矩阵的半正定性"""
    np.random.seed(42)
    n_assets = 50
    n_factors = 10
    # 生成高度共线的随机载荷
    base = np.random.randn(n_assets, 3)
    raw = np.hstack([base + 0.1 * np.random.randn(n_assets, 3) for _ in range(n_factors // 3 + 1)])
    raw = raw[:, :n_factors]
    # 归一化每行
    raw = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-12) * 0.8

    orth = orthogonalize_credit_factors(raw, method="mgs")
    Corr = factor_covariance_from_loadings(orth, np.ones(orth.shape[1]))

    eigvals = np.linalg.eigvalsh(Corr)
    assert np.all(eigvals > -1e-10), "相关性矩阵存在负特征值!"
    assert np.allclose(np.diag(Corr), 1.0, atol=1e-6), "对角线不为 1!"
    print(f"factor_orthogonalization test passed. rank={orth.shape[1]}, min_eig={eigvals.min():.6f}")


if __name__ == "__main__":
    test_factor_orthogonalization()
