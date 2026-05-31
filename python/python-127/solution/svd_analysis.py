"""
svd_analysis.py
===============
基于 SVD 的患者参数降维与电场分析模块

基于种子项目:
  - 1186_svd_faces: SVD 向量提取（人脸识别中的降维思想）

科学背景:
  不同患者的耳蜗几何、组织电导率、神经分布存在显著个体差异。
  通过收集大量患者数据构成数据矩阵 A ∈ ℝ^{M×N}，
  使用 SVD 提取主成分:
      A = U Σ V^T

  其中:
    - U 的列向量为主成分电场模式
    - Σ 的对角元为奇异值，反映各模式的能量占比
    - V 的列向量为患者在主成分空间中的坐标

  低秩近似:
      A_k = Σ_{i=1}^k σ_i u_i v_i^T

  可用于:
    1) 患者个性化参数压缩
    2) 电场模式降维与去噪
    3) 电极配置优化
"""

import numpy as np


class PatientSVDAnalyzer:
    """
    基于 SVD 的患者电场数据降维分析器。
    """

    def __init__(self):
        self.U = None
        self.S = None
        self.Vt = None
        self.mean_vector = None
        self.n_patients = 0
        self.n_features = 0

    def fit(self, data_matrix):
        """
        对数据矩阵进行 SVD 分解。

        Parameters
        ----------
        data_matrix : ndarray, shape (n_features, n_patients)
            每列是一个患者的特征向量（如电场采样值）
        """
        data_matrix = np.asarray(data_matrix, dtype=float)
        if data_matrix.ndim != 2:
            raise ValueError("data_matrix 必须为二维数组")

        self.n_features, self.n_patients = data_matrix.shape

        # 去均值
        self.mean_vector = np.mean(data_matrix, axis=1)
        A_centered = data_matrix - self.mean_vector[:, np.newaxis]

        # 经济型 SVD
        self.U, self.S, self.Vt = np.linalg.svd(A_centered, full_matrices=False)

    def get_principal_components(self, n_components):
        """
        获取前 n_components 个主成分。

        Parameters
        ----------
        n_components : int

        Returns
        -------
        pcs : ndarray, shape (n_features, n_components)
        """
        if self.U is None:
            raise RuntimeError("必须先调用 fit()")
        n = min(n_components, self.U.shape[1])
        return self.U[:, :n]

    def get_singular_values(self):
        """返回所有奇异值。"""
        if self.S is None:
            raise RuntimeError("必须先调用 fit()")
        return self.S.copy()

    def explained_variance_ratio(self):
        """
        各主成分解释的方差比例。

        Returns
        -------
        ratios : ndarray
        """
        if self.S is None:
            raise RuntimeError("必须先调用 fit()")
        total = np.sum(self.S**2)
        if total < 1e-15:
            return np.zeros_like(self.S)
        return (self.S**2) / total

    def cumulative_variance_ratio(self):
        """累积方差比例。"""
        return np.cumsum(self.explained_variance_ratio())

    def project(self, patient_vector, n_components):
        """
        将患者向量投影到主成分空间。

        Parameters
        ----------
        patient_vector : ndarray, shape (n_features,)
        n_components : int

        Returns
        -------
        coeffs : ndarray, shape (n_components,)
            主成分系数
        """
        if self.U is None:
            raise RuntimeError("必须先调用 fit()")
        patient_vector = np.asarray(patient_vector, dtype=float)
        centered = patient_vector - self.mean_vector
        pcs = self.get_principal_components(n_components)
        coeffs = pcs.T @ centered
        return coeffs

    def reconstruct(self, coeffs):
        """
        从主成分系数重构患者向量。

        Parameters
        ----------
        coeffs : ndarray, shape (n_components,)

        Returns
        -------
        reconstructed : ndarray, shape (n_features,)
        """
        if self.U is None:
            raise RuntimeError("必须先调用 fit()")
        coeffs = np.asarray(coeffs, dtype=float)
        n_components = len(coeffs)
        pcs = self.get_principal_components(n_components)
        reconstructed = pcs @ coeffs + self.mean_vector
        return reconstructed

    def low_rank_approximation(self, rank):
        """
        获取秩为 rank 的低秩近似矩阵。

        Parameters
        ----------
        rank : int

        Returns
        -------
        A_approx : ndarray, shape (n_features, n_patients)
        """
        if self.U is None:
            raise RuntimeError("必须先调用 fit()")
        k = min(rank, len(self.S))
        A_approx = (self.U[:, :k] * self.S[:k]) @ self.Vt[:k, :]
        A_approx += self.mean_vector[:, np.newaxis]
        return A_approx

    def compression_ratio(self, n_components):
        """
        计算压缩比。

        原始数据: n_features × n_patients
        压缩后: n_features × n_components (均值 + 主成分) + n_patients × n_components (系数)
        """
        if self.U is None:
            raise RuntimeError("必须先调用 fit()")
        original = self.n_features * self.n_patients
        compressed = self.n_features * n_components + self.n_patients * n_components + self.n_features
        return original / compressed


def generate_synthetic_patient_data(n_patients=50, n_features=200,
                                     n_modes=5, noise_level=0.05):
    """
    生成合成患者电场数据用于 SVD 分析。

    Parameters
    ----------
    n_patients : int
        患者数
    n_features : int
        特征数（空间采样点数）
    n_modes : int
        真实低秩模式数
    noise_level : float
        噪声水平

    Returns
    -------
    data : ndarray, shape (n_features, n_patients)
    true_modes : ndarray, shape (n_features, n_modes)
    """
    np.random.seed(42)
    # 生成正交模式
    true_modes = np.random.randn(n_features, n_modes)
    q, _ = np.linalg.qr(true_modes)
    true_modes = q[:, :n_modes]

    # 随机系数
    coeffs = np.random.randn(n_modes, n_patients)

    # 合成数据
    data = true_modes @ coeffs
    data += noise_level * np.random.randn(n_features, n_patients)

    return data, true_modes


def electrode_config_optimization_svd(electrode_responses, n_components=3):
    """
    使用 SVD 优化电极配置。

    给定不同电极配置下的响应矩阵，提取最优配置方向。

    Parameters
    ----------
    electrode_responses : ndarray, shape (n_configs, n_neurons)
        每行是一个电极配置在所有神经元上的响应
    n_components : int

    Returns
    -------
    optimal_direction : ndarray
        最优配置方向（第一主成分）
    importance_scores : ndarray
        各配置参数的重要性得分
    """
    responses = np.asarray(electrode_responses, dtype=float)
    analyzer = PatientSVDAnalyzer()
    analyzer.fit(responses.T)

    pcs = analyzer.get_principal_components(n_components)
    sv = analyzer.get_singular_values()[:n_components]

    # 第一主成分方向为最优方向
    optimal_direction = pcs[:, 0]

    # 各原始配置参数的重要性
    importance_scores = np.abs(optimal_direction) * sv[0]

    return optimal_direction, importance_scores
