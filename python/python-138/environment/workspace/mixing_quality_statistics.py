"""
微反应器混合质量的多变量统计度量 (基于多元正态距离统计)
=========================================================
通过计算微反应器横截面上示踪剂浓度分布的多变量距离统计量，
定量评估混合效率。

核心统计量：
    给定 m 个采样点的浓度向量 c ∈ ℝ^m，假设理想完全混合态为
    多元正态分布 N(μ·1, σ²·I)。

    定义混合缺陷指标：
        M_d = E[ ||c - μ·1||_2 ] / (σ √m)

    当 M_d → 0 时，混合完全；M_d 越大，混合越不均匀。

    此外定义多变量变异系数：
        CV_multi = √(det(Σ)) / μ^m

    以及 Kullback-Leibler 散度（相对熵）来衡量实际分布与理想分布的差异：
        D_KL = 0.5 [ tr(Σ_0^{-1} Σ) + (μ - μ_0)^T Σ_0^{-1} (μ - μ_0)
                     - m + ln(det(Σ_0)/det(Σ)) ]
"""

import numpy as np
from typing import Tuple, Optional


class MixingQualityAnalyzer:
    """
    微反应器混合质量多变量统计分析器。
    """

    def __init__(self, ideal_mean: float = 500.0, ideal_std: float = 50.0):
        self.ideal_mean = ideal_mean
        self.ideal_std = ideal_std

    def sample_concentration_field(
        self,
        n_samples: int = 1000,
        dim: int = 2,
        mixing_efficiency: float = 0.7,
    ) -> np.ndarray:
        """
        生成模拟的浓度采样数据。

        参数:
            mixing_efficiency: 0~1，1 表示完全混合，0 表示完全不混合。
        """
        if not (0.0 <= mixing_efficiency <= 1.0):
            raise ValueError("mixing_efficiency 必须在 [0,1] 区间内")

        # 完全混合态：N(μ, σ²)
        fully_mixed = np.random.normal(
            self.ideal_mean, self.ideal_std, size=(n_samples, dim)
        )
        # 未混合态：双峰分布
        cluster1 = np.random.normal(
            self.ideal_mean * 1.5, self.ideal_std * 0.3, size=(n_samples // 2, dim)
        )
        cluster2 = np.random.normal(
            self.ideal_mean * 0.5, self.ideal_std * 0.3, size=(n_samples // 2, dim)
        )
        unmixed = np.vstack([cluster1, cluster2])
        if len(unmixed) < n_samples:
            pad = np.random.normal(self.ideal_mean, self.ideal_std, size=(n_samples - len(unmixed), dim))
            unmixed = np.vstack([unmixed, pad])
        unmixed = unmixed[:n_samples]

        # 插值
        samples = mixing_efficiency * fully_mixed + (1.0 - mixing_efficiency) * unmixed
        return samples

    def compute_mixing_defect(self, samples: np.ndarray) -> float:
        """
        计算混合缺陷指标 M_d：
            M_d = E[ ||x - μ·1|| ] / (σ √m)
        """
        m, d = samples.shape
        mean_vec = np.mean(samples, axis=0)
        centered = samples - mean_vec
        distances = np.sqrt(np.sum(centered ** 2, axis=1))
        expected_distance = np.mean(distances)
        # 归一化因子：对理想球形高斯，E[||x-μ||] ≈ σ √d
        normalization = self.ideal_std * np.sqrt(d)
        if normalization < 1.0e-12:
            return 0.0
        M_d = expected_distance / normalization
        return M_d

    def compute_multivariate_cv(self, samples: np.ndarray) -> float:
        """
        计算多变量变异系数：
            CV_multi = (det(Σ))^{1/(2d)} / ||μ||
        """
        m, d = samples.shape
        mu = np.mean(samples, axis=0)
        Sigma = np.cov(samples, rowvar=False)
        if d == 1:
            Sigma = np.array([[Sigma]])
        try:
            det_sigma = np.linalg.det(Sigma)
        except np.linalg.LinAlgError:
            det_sigma = 0.0
        det_sigma = max(det_sigma, 1.0e-20)
        numerator = det_sigma ** (1.0 / (2.0 * d))
        denom = np.linalg.norm(mu)
        if denom < 1.0e-12:
            return 0.0
        return numerator / denom

    def compute_kl_divergence(self, samples: np.ndarray) -> float:
        """
        计算实际样本分布与理想多元正态分布之间的 KL 散度。
        """
        m, d = samples.shape
        mu = np.mean(samples, axis=0)
        Sigma = np.cov(samples, rowvar=False)
        if d == 1:
            Sigma = np.array([[Sigma]])

        mu0 = np.full(d, self.ideal_mean)
        Sigma0 = (self.ideal_std ** 2) * np.eye(d)

        try:
            inv_Sigma0 = np.linalg.inv(Sigma0)
            det_Sigma = np.linalg.det(Sigma)
            det_Sigma0 = np.linalg.det(Sigma0)
        except np.linalg.LinAlgError:
            return float("inf")

        det_Sigma = max(det_Sigma, 1.0e-20)
        det_Sigma0 = max(det_Sigma0, 1.0e-20)

        term1 = np.trace(inv_Sigma0 @ Sigma)
        diff = mu - mu0
        term2 = diff.T @ inv_Sigma0 @ diff
        term3 = d
        term4 = np.log(det_Sigma0 / det_Sigma)

        D_kl = 0.5 * (term1 + term2 - term3 + term4)
        return max(D_kl, 0.0)

    def compute_mixing_efficiency_index(self, samples: np.ndarray) -> float:
        """
        综合混合效率指数 (0~1)：
            η_mix = exp( - (M_d + CV_multi + D_KL) / 3 )
        """
        M_d = self.compute_mixing_defect(samples)
        cv_multi = self.compute_multivariate_cv(samples)
        D_kl = self.compute_kl_divergence(samples)
        eta = np.exp(-(M_d + cv_multi + D_kl) / 3.0)
        eta = max(0.0, min(1.0, eta))
        return eta

    def statistical_distance_between_zones(
        self, zone_a: np.ndarray, zone_b: np.ndarray
    ) -> Tuple[float, float]:
        """
        计算两个微混合区之间的 Mahalanobis 距离与 Bhattacharyya 距离：

            D_M = √[(μ_a - μ_b)^T Σ^{-1} (μ_a - μ_b)]

            D_B = 0.25 D_M² + 0.5 ln( det(Σ) / √(det(Σ_a) det(Σ_b)) )

        其中 Σ = (Σ_a + Σ_b)/2。
        """
        mu_a = np.mean(zone_a, axis=0)
        mu_b = np.mean(zone_b, axis=0)
        cov_a = np.cov(zone_a, rowvar=False)
        cov_b = np.cov(zone_b, rowvar=False)
        if zone_a.shape[1] == 1:
            cov_a = np.array([[cov_a]]) if np.isscalar(cov_a) else cov_a.reshape((1, 1))
            cov_b = np.array([[cov_b]]) if np.isscalar(cov_b) else cov_b.reshape((1, 1))

        Sigma = 0.5 * (cov_a + cov_b)
        try:
            inv_Sigma = np.linalg.inv(Sigma)
            det_a = max(np.linalg.det(cov_a), 1.0e-20)
            det_b = max(np.linalg.det(cov_b), 1.0e-20)
            det_Sigma = max(np.linalg.det(Sigma), 1.0e-20)
        except np.linalg.LinAlgError:
            return float("inf"), float("inf")

        diff = mu_a - mu_b
        D_M_sq = float(diff.T @ inv_Sigma @ diff)
        D_M = np.sqrt(max(D_M_sq, 0.0))

        D_B = 0.25 * D_M_sq + 0.5 * np.log(det_Sigma / np.sqrt(det_a * det_b))
        return D_M, D_B
