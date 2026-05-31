"""
svd_reduction.py
降阶模型与主成分分析模块

基于种子项目的核心算法：
- 1184_svd_basis: 从 PDE 解数据中提取主导 POD 模态
- 1187_svd_fingerprint: SVD 降维与低秩近似

在离子通道问题中的应用：
- 对分子动力学轨迹进行 PCA/SVD，提取离子传导的集体运动模式
- 构建降阶模型（ROM），用少量主成分描述高维构型空间
- 对电势场数据进行低秩近似，加速 PNP 方程求解

数学基础：
给定数据矩阵 A ∈ R^{m×n}（m 为空间点数，n 为时间帧数），
SVD 分解：
    A = U Σ V^T

其中 U 的列向量为空间模态（POD/PCA 主成分），
Σ 的对角元素为奇异值（对应各模态的能量贡献）。

降阶近似（秩 r）：
    A_r = U_r Σ_r V_r^T

压缩比：
    CR = (m·r + r + r·n) / (m·n)
"""

import numpy as np


def compute_pod_basis(data_matrix, n_modes, subtract_mean=True):
    """
    计算 POD（Proper Orthogonal Decomposition）基。

    Parameters
    ----------
    data_matrix : ndarray, shape (m, n)
        数据矩阵，每列为一个快照
    n_modes : int
        提取的主导模态数
    subtract_mean : bool
        是否减去时间平均

    Returns
    -------
    modes : ndarray, shape (m, n_modes)
        POD 模态（U 矩阵的前 n_modes 列）
    singular_values : ndarray
        奇异值
    coefficients : ndarray, shape (n_modes, n)
        各快照在各模态上的投影系数
    """
    A = data_matrix.copy()
    m, n = A.shape
    mean_vec = None

    if subtract_mean:
        mean_vec = np.mean(A, axis=1, keepdims=True)
        A = A - mean_vec

    # 经济型 SVD
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    n_modes = min(n_modes, len(s))
    modes = U[:, :n_modes]
    singular_values = s[:n_modes]
    coefficients = np.diag(singular_values) @ Vt[:n_modes, :]

    return modes, singular_values, coefficients, mean_vec


def low_rank_approximation(data_matrix, rank):
    """
    计算低秩近似（源自 svd_fingerprint.m / svd_bw.m 思想）。

    A_r = Σ_{k=1}^r σ_k u_k v_k^T
    """
    U, s, Vt = np.linalg.svd(data_matrix, full_matrices=False)
    A_r = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
    return A_r


def compression_ratio(m, n, rank):
    """
    计算 SVD 低秩近似的压缩比。

    CR = (m·r + r + r·n) / (m·n)
    """
    return (m * rank + rank + rank * n) / (m * n)


def cumulative_energy(singular_values):
    """
    计算累积能量占比：
        E_r = Σ_{k=1}^r σ_k^2 / Σ_{k=1}^n σ_k^2
    """
    total = np.sum(singular_values ** 2)
    cum = np.cumsum(singular_values ** 2) / total
    return cum


def reconstruct_field(modes, coefficients, mean_vec=None):
    """
    从 POD 模态和系数重构数据场。
    """
    recon = modes @ coefficients
    if mean_vec is not None:
        recon = recon + mean_vec
    return recon


class ReducedOrderModel:
    """
    降阶模型：用少量 POD 模态近似高维场演化。
    """
    def __init__(self, modes, singular_values, mean_vec=None):
        self.modes = modes
        self.singular_values = singular_values
        self.mean_vec = mean_vec
        self.n_modes = modes.shape[1]

    def project_to_reduced(self, field):
        """
        将完整场投影到降阶子空间：
            a = U_r^T (field - mean)
        """
        if self.mean_vec is not None:
            field = field - self.mean_vec.flatten()
        return self.modes.T @ field

    def reconstruct_from_reduced(self, coeffs):
        """
        从降阶系数重构完整场：
            field = U_r a + mean
        """
        recon = self.modes @ coeffs
        if self.mean_vec is not None:
            recon = recon + self.mean_vec.flatten()
        return recon

    def galilean_invariance_error(self, field, shifted_field):
        """
        评估降阶模型对平移不变性的保持能力。
        """
        a1 = self.project_to_reduced(field)
        a2 = self.project_to_reduced(shifted_field)
        return np.linalg.norm(a1 - a2) / (np.linalg.norm(a1) + 1e-30)


def analyze_trajectory_pca(trajectories, n_modes=5):
    """
    对离子轨迹数据进行 PCA 分析，提取集体运动模式。

    Parameters
    ----------
    trajectories : list of ndarray
        每个粒子的轨迹 (n_steps, 3)
    n_modes : int
        主成分数

    Returns
    -------
    modes, singular_values, cumulative_variance
    """
    # 拼接所有轨迹为 (n_particles * n_steps, 3)
    all_pos = np.vstack(trajectories)
    # 去均值
    mean_pos = np.mean(all_pos, axis=0)
    centered = all_pos - mean_pos

    # 协方差矩阵
    cov = centered.T @ centered / centered.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)

    # 按特征值降序排列
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    cumulative_variance = np.cumsum(eigvals) / np.sum(eigvals)
    return eigvecs[:, :n_modes], eigvals[:n_modes], cumulative_variance[:n_modes]
