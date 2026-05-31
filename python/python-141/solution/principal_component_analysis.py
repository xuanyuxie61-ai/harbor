"""
主成分分析模块
================
基于种子项目 326_eigenfaces 的核心算法改造。

在金融工程中，主成分分析(PCA)广泛应用于:
1. 波动率曲面降维：提取少数主成分解释隐含波动率的横截面变化
2. 利率期限结构建模：Nelson-Siegel框架本质上是PCA的连续版本
3. 多因子随机波动率模型：通过PCA从高维相关波动率中提取独立驱动因子

数学背景:
---------
给定数据矩阵 A ∈ R^{m×n}（m为变量数，n为观测数），中心化后:
    Ã = A - μ·1^T
样本协方差矩阵:
    C = (1/(n-1)) Ã Ã^T

特征分解:
    C · e_k = λ_k · e_k,   λ_1 ≥ λ_2 ≥ ... ≥ λ_m

主成分得分:
    PC_k(t) = e_k^T · Ã(:,t)

累计解释方差比:
    R(K) = Σ_{k=1}^K λ_k / Σ_{k=1}^m λ_k

本模块采用Turk-Pentland技巧：当 m >> n 时，先对 n×n 矩阵 Ã^T Ã 做特征分解，
再通过 Ã · v_k 得到 C 的特征向量，计算复杂度从 O(m³) 降为 O(n³)。
"""

import numpy as np


class PrincipalComponentAnalysis:
    """
    主成分分析器，支持高维低样本场景的高效计算。
    """

    def __init__(self, n_components=None):
        """
        参数:
        ------
        n_components : int, 保留的主成分数，默认保留全部
        """
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, A):
        """
        对数据矩阵A进行PCA拟合。

        参数:
        ------
        A : ndarray, 形状 (m, n)，m个变量，n个观测
        """
        A = np.asarray(A, dtype=np.float64)
        if A.ndim != 2:
            raise ValueError("A必须为二维矩阵")
        m, n = A.shape

        # 中心化
        self.mean_ = np.mean(A, axis=1, keepdims=True)
        A_centered = A - self.mean_

        # Turk-Pentland技巧: m >> n 时计算 A^T A 而非 A A^T
        if m > n:
            L = A_centered.T @ A_centered  # n×n
            eigvals, eigvecs = np.linalg.eigh(L)
            # 按特征值降序排列
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            # 转换为 A A^T 的特征向量
            # components 形状 (m, n)
            components = A_centered @ eigvecs
            # 归一化
            norms = np.linalg.norm(components, axis=0)
            norms[norms < 1e-12] = 1.0
            components = components / norms
            # 特征值缩放
            eigvals = eigvals / (n - 1) if n > 1 else eigvals
        else:
            C = (A_centered @ A_centered.T) / max(n - 1, 1)
            eigvals, components = np.linalg.eigh(C)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            components = components[:, idx]

        self.explained_variance_ = eigvals
        total_var = np.sum(eigvals)
        self.explained_variance_ratio_ = eigvals / total_var if total_var > 0 else np.zeros_like(eigvals)

        # 滤除数值零特征值对应的主成分
        num_good = np.sum(eigvals > 1e-10)
        if self.n_components is None:
            self.n_components = num_good
        else:
            self.n_components = min(self.n_components, num_good, m)

        self.components_ = components[:, :self.n_components]
        self.explained_variance_ = eigvals[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components]

    def transform(self, A):
        """
        将数据投影到主成分空间。

        参数:
        ------
        A : ndarray, 形状 (m, n)

        返回:
        ------
        ndarray, 形状 (n_components, n)
        """
        if self.components_ is None:
            raise RuntimeError("必须先调用fit()")
        A = np.asarray(A, dtype=np.float64)
        A_centered = A - self.mean_
        return self.components_.T @ A_centered

    def inverse_transform(self, scores):
        """
        从主成分空间重建数据。

        重建公式:
            A_recon = mean + components · scores
        """
        if self.components_ is None:
            raise RuntimeError("必须先调用fit()")
        scores = np.asarray(scores, dtype=np.float64)
        return self.mean_ + self.components_ @ scores

    def reconstruct_with_k_components(self, A, k):
        """
        使用前k个主成分重建数据。
        """
        if k > self.n_components:
            raise ValueError(f"k不能超过已保留的主成分数{self.n_components}")
        A = np.asarray(A, dtype=np.float64)
        A_centered = A - self.mean_
        scores = self.components_[:, :k].T @ A_centered
        return self.mean_ + self.components_[:, :k] @ scores


def volatility_surface_pca(maturities, strikes, iv_surface, n_pcs=3):
    """
    对隐含波动率曲面进行主成分分析。

    参数:
    ------
    maturities : array, 到期期限
    strikes    : array, 行权价
    iv_surface : ndarray, 形状 (len(maturities), len(strikes))
    n_pcs      : int, 保留的主成分数

    返回:
    ------
    dict, 包含主成分、解释方差比、重建曲面等
    """
    iv_surface = np.asarray(iv_surface, dtype=np.float64)
    if iv_surface.ndim != 2:
        raise ValueError("iv_surface必须为二维矩阵")

    # 将曲面展平为向量，每个观测为一个到期日的行权价向量
    m, n = iv_surface.shape
    # 这里我们视每个行权价为一个变量，每个到期日为一个观测
    # 转置后: (n_strikes, n_maturities)
    A = iv_surface.T

    pca = PrincipalComponentAnalysis(n_components=n_pcs)
    pca.fit(A)

    # 主成分载荷（行权价方向的模式）
    pc_loadings = pca.components_  # (n_strikes, n_pcs)

    # 主成分得分（到期日方向的演化）
    scores = pca.transform(A)  # (n_pcs, n_maturities)

    # 重建
    recon = pca.inverse_transform(scores).T  # (n_maturities, n_strikes)

    return {
        'maturities': maturities,
        'strikes': strikes,
        'pc_loadings': pc_loadings,
        'scores': scores,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'reconstructed_surface': recon,
        'mean_curve': pca.mean_.flatten()
    }


def correlated_volatility_factors(cov_matrix, n_factors=None):
    """
    从波动率协方差矩阵中提取不相关的主成分因子。

    在多因子随机波动率模型中，设 d 个资产的波动率对数服从联合高斯分布，
    协方差矩阵为 Σ。通过PCA，可将其表示为:
        log(σ_i) = μ_i + Σ_{k=1}^K b_{ik} · Z_k
    其中 Z_k ~ N(0,1) 独立，b_{ik} = e_{ik} · √λ_k。

    参数:
    ------
    cov_matrix : ndarray, 波动率协方差矩阵
    n_factors  : int, 保留因子数

    返回:
    ------
    dict, 包含载荷矩阵、特征值、累计解释方差
    """
    cov = np.asarray(cov_matrix, dtype=np.float64)
    m = cov.shape[0]
    if n_factors is None:
        n_factors = m

    pca = PrincipalComponentAnalysis(n_components=n_factors)
    # PCA通常对数据矩阵操作，这里直接传入中心化后的“观测”
    # 为利用特征分解，我们构造一个对称矩阵的平方根表示
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # 构造载荷: B = E · Λ^{1/2}
    loadings = eigvecs[:, :n_factors] * np.sqrt(np.maximum(eigvals[:n_factors], 0.0))

    total_var = np.sum(np.maximum(eigvals, 0.0))
    explained_ratio = np.maximum(eigvals[:n_factors], 0.0) / total_var if total_var > 0 else np.zeros(n_factors)

    return {
        'loadings': loadings,
        'eigenvalues': eigvals[:n_factors],
        'explained_variance_ratio': explained_ratio,
        'cumulative_variance_ratio': np.cumsum(explained_ratio)
    }
