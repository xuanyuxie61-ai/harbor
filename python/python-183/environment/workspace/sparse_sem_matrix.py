r"""
sparse_sem_matrix.py
================================================================================
基于 Harwell-Boeing 稀疏矩阵思想的结构方程模型 (SEM) 稀疏精度矩阵构建模块

原项目映射: 510_hb_to_st — Harwell-Boeing 稀疏矩阵文件格式读写与 CSR 转换

科学背景
--------
在因果推断中，观测变量 $X=(X_1,\dots,X_p)^T$ 的协方差矩阵 $\Sigma$ 的逆矩阵
$\Theta = \Sigma^{-1}$（精度矩阵，Precision Matrix）刻画了条件独立性：
若 $\Theta_{ij}=0$，则在控制其他所有变量后 $X_i \perp X_j$。
因此稀疏精度矩阵的非零结构即为因果骨架（Causal Skeleton）。

核心公式
--------
1. 样本协方差矩阵（极大似然估计）:
   $$ \hat{\Sigma} = \frac{1}{n}\sum_{k=1}^{n}(x^{(k)}-\bar{x})(x^{(k)}-\bar{x})^T $$

2. Graphical Lasso 目标泛函（带 $L_1$ 惩罚的极大似然）:
   $$ \min_{\Theta \succ 0} \; -\log\det\Theta + \text{tr}(S\Theta) + \lambda\|\Theta\|_1 $$
   其中 $S$ 为样本协方差矩阵，$\lambda>0$ 为正则化参数。

3. 近端梯度更新（ISTA 风格）:
   $$ \Theta_{t+1} = \text{ST}_{\eta\lambda}\left(\Theta_t - \eta(\Theta_t^{-1} - S)\right) $$
   其中 ST 为软阈值算子：$\text{ST}_\tau(x)=\text{sign}(x)\max(|x|-\tau,0)$。

4. CSR (Compressed Sparse Row) 存储格式:
   将稠密精度矩阵转换为稀疏三元组 (row, col, val)，再压缩为 CSR
   以支持大规模因果网络分析。
r"""

import numpy as np
from typing import Tuple, List, Optional


def sample_covariance(X: np.ndarray) -> np.ndarray:
    r"""
    计算样本协方差矩阵。

    Parameters
    ----------
    X : ndarray, shape (n_samples, p)
        观测数据矩阵。

    Returns
    -------
    S : ndarray, shape (p, p)
        样本协方差矩阵 $S = \frac{1}{n} X_c^T X_c$。
    r"""
    n, p = X.shape
    if n < 2:
        raise ValueError("样本数 n 必须至少为 2 才能计算协方差。")
    Xc = X - np.mean(X, axis=0, keepdims=True)
    S = Xc.T @ Xc / n
    # 数值稳定性：确保对称
    S = 0.5 * (S + S.T)
    return S


def soft_threshold(M: np.ndarray, tau: float) -> np.ndarray:
    r"""
    逐元素软阈值算子。

    $$ \text{ST}_\tau(M_{ij}) = \text{sign}(M_{ij})\max(|M_{ij}|-\tau, 0) $$
    注意对角线元素不做阈值（保留方差信息）。
    r"""
    if tau < 0.0:
        raise ValueError("阈值 tau 必须非负。")
    res = np.sign(M) * np.maximum(np.abs(M) - tau, 0.0)
    np.fill_diagonal(res, np.diag(M))
    return res


def graphical_lasso(S: np.ndarray,
                    lam: float = 0.05,
                    max_iter: int = 200,
                    tol: float = 1e-6,
                    eta: float = 0.5,
                    verbose: bool = False) -> np.ndarray:
    r"""
    基于近端梯度下降（ISTA）的 Graphical Lasso 稀疏精度矩阵估计。

    目标：
    $$ \min_{\Theta \succ 0} -\log\det\Theta + \text{tr}(S\Theta) + \lambda\|\Theta\|_1 $$

    Parameters
    ----------
    S : ndarray, shape (p, p)
        样本协方差矩阵（需半正定）。
    lam : float
        $L_1$ 正则化强度 $\lambda$。
    max_iter : int
        最大迭代次数。
    tol : float
        收敛容差（Frobenius 范数变化）。
    eta : float
        梯度步长（需满足 $\eta \le 1/L$，这里取固定小步长）。

    Returns
    -------
    Theta : ndarray, shape (p, p)
        估计的稀疏精度矩阵。
    r"""
    p = S.shape[0]
    if p == 0:
        raise ValueError("协方差矩阵维度不能为 0。")
    # 初始化：对角逆
    Theta = np.diag(1.0 / (np.diag(S) + 1e-8))

    # TODO [Hole 1] 请补全 Graphical Lasso 的核心迭代步骤：
    # 1. 计算当前 Theta 的逆矩阵 Theta_inv（需处理奇异情况）
    # 2. 计算梯度 G = Theta_inv - S
    # 3. 应用软阈值算子：Theta_new = soft_threshold(Theta - eta * G, eta * lam)
    # 4. 投影到正定锥（谱截断，确保特征值 >= 1e-8）
    # 5. 计算 Frobenius 范数变化 diff，若 diff < tol 则收敛退出
    # 注意：对角线元素在软阈值中应保留原值（方差信息不可稀疏化）
    raise NotImplementedError("Hole 1: Graphical Lasso 迭代循环待实现")

    return Theta


def threshold_precision(Theta: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    r"""
    对精度矩阵进行硬阈值，提取因果骨架的非零结构。

    Parameters
    ----------
    Theta : ndarray
        精度矩阵。
    eps : float
        零值截断阈值。

    Returns
    -------
    Theta_sparse : ndarray
        阈值后的稀疏精度矩阵。
    r"""
    Theta_s = Theta.copy()
    Theta_s[np.abs(Theta_s) < eps] = 0.0
    np.fill_diagonal(Theta_s, np.diag(Theta))
    return Theta_s


def dense_to_csr(A: np.ndarray, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    将稠密矩阵转换为 CSR (Compressed Sparse Row) 格式。

    原项目 510_hb_to_st 核心思想：将 Harwell-Boeing 格式中的稀疏矩阵
    通过列指针 (colptr) 与行索引 (rowind) 进行压缩存储。
    这里实现通用的稠密→CSR 转换，适用于大规模因果骨架矩阵。

    CSR 定义：
    - data: 所有非零元值
    - indices: 每行中非零元的列号
    - indptr: 第 i 行的非零元在 data 中的起始位置

    Parameters
    ----------
    A : ndarray, shape (m, n)
        输入稠密矩阵。
    tol : float
        判定为零的容差。

    Returns
    -------
    data, indices, indptr : 1-D arrays
        CSR 格式的三个数组。
    r"""
    m, n = A.shape
    data_list: List[float] = []
    indices_list: List[int] = []
    indptr = np.zeros(m + 1, dtype=int)

    for i in range(m):
        row_nnz = 0
        for j in range(n):
            val = A[i, j]
            if np.abs(val) > tol:
                data_list.append(float(val))
                indices_list.append(j)
                row_nnz += 1
        indptr[i + 1] = indptr[i] + row_nnz

    data = np.array(data_list, dtype=float)
    indices = np.array(indices_list, dtype=int)
    return data, indices, indptr


def csr_to_dense(data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, ncols: int) -> np.ndarray:
    r"""
    将 CSR 格式还原为稠密矩阵。
    r"""
    m = indptr.shape[0] - 1
    A = np.zeros((m, ncols), dtype=float)
    for i in range(m):
        for idx in range(indptr[i], indptr[i + 1]):
            j = indices[idx]
            A[i, j] = data[idx]
    return A


def extract_causal_skeleton(Theta_sparse: np.ndarray) -> Tuple[List[Tuple[int, int, float]], int]:
    r"""
    从稀疏精度矩阵中提取因果骨架边列表。

    返回 (i, j, weight) 列表，仅含上三角非对角非零元，
    以及总节点数。
    r"""
    p = Theta_sparse.shape[0]
    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            w = Theta_sparse[i, j]
            if w != 0.0:
                edges.append((i, j, float(w)))
    return edges, p


def demo():
    r"""
    模块自测试：生成合成 SEM 数据，估计稀疏精度矩阵并输出 CSR。
    r"""
    np.random.seed(42)
    p = 12
    n = 500
    # 构造真实的稀疏精度矩阵（因果骨架）
    Theta_true = np.eye(p) * 2.0
    for i in range(p - 1):
        Theta_true[i, i + 1] = 0.4
        Theta_true[i + 1, i] = 0.4
    Theta_true[0, 3] = 0.3
    Theta_true[3, 0] = 0.3
    Theta_true[5, 8] = -0.25
    Theta_true[8, 5] = -0.25

    # 生成样本：从 $N(0, \Sigma)$ 采样，其中 $\Sigma = \Theta^{-1}$
    Sigma_true = np.linalg.inv(Theta_true)
    X = np.random.multivariate_normal(np.zeros(p), Sigma_true, size=n)

    S = sample_covariance(X)
    Theta_est = graphical_lasso(S, lam=0.08, max_iter=300, verbose=True)
    Theta_sparse = threshold_precision(Theta_est, eps=5e-3)

    edges, _ = extract_causal_skeleton(Theta_sparse)
    print(f"[sparse_sem_matrix] 估计的因果骨架边数: {len(edges)}")

    # CSR 转换测试
    data, indices, indptr = dense_to_csr(Theta_sparse)
    Theta_recover = csr_to_dense(data, indices, indptr, p)
    rec_err = np.linalg.norm(Theta_sparse - Theta_recover, 'fro')
    print(f"[sparse_sem_matrix] CSR 重构误差: {rec_err:.2e}")
    return Theta_sparse, edges


if __name__ == "__main__":
    demo()
