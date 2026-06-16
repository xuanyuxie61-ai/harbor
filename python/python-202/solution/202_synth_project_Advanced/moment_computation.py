"""
矩量计算模块 (Statistical Moment Computation)
===============================================
基于随机配置法的统计矩计算。

核心公式:
  统计矩的稀疏网格近似:
    E[u] ≈ Σ_{k=1}^{N_sg} w_k u(x, ξ_k)
    Var[u] ≈ Σ_{k=1}^{N_sg} w_k (u(x, ξ_k) - E[u])²

  高阶矩:
    Skewness: γ₁ = E[(u-μ)³] / σ³
    Kurtosis: γ₂ = E[(u-μ)⁴] / σ⁴ - 3

  其中 w_k 是稀疏网格权重, ξ_k 是配置点。

  误差估计:
    |E[u] - E_sg[u]| ≤ C × N_sg^{-2p/d} × ||u||_{C^{2p}}
    其中 p 是求积精度, d 是随机维度。

  置信区间:
    u_{95%} = E[u] ± 1.96 × √Var[u]  (假设近似正态)
"""

import numpy as np
from typing import Optional, Dict, Tuple


class MomentCalculator:
    """
    统计矩计算器: 从配置点解值计算统计矩。

    输入:
      - 配置点解值: u_k = u(x, ξ_k), k=1,...,N
      - 稀疏网格权重: w_k, k=1,...,N
      - 空间网格: x_i, i=1,...,N_x (可选, 用于空间分辨的矩)

    输出:
      - 均值: E[u](x)
      - 方差: Var[u](x)
      - 标准差: σ[u](x)
      - 偏度: Skew[u](x)
      - 峰度: Kurt[u](x)
      - 置信区间
    """

    def __init__(
        self,
        solution_values: np.ndarray,
        weights: np.ndarray,
        spatial_grid: Optional[np.ndarray] = None
    ):
        """
        参数:
            solution_values: 配置点解值
                shape (N_config, N_x) 或 shape (N_config,)
            weights: 配置权重, shape (N_config,)
            spatial_grid: 空间网格, shape (N_x,) (可选)
        """
        self.sol = np.atleast_2d(solution_values)  # (N_config, N_x) or (N_config, 1)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.spatial_grid = spatial_grid

        N_config = self.sol.shape[0]
        if self.weights.size != N_config:
            raise ValueError(
                f"Weights size ({self.weights.size}) != solution count ({N_config})"
            )

        # 权重归一化检查
        w_sum = np.sum(self.weights)
        self.weight_normalization_error = abs(w_sum - 1.0)

    def compute_mean(self) -> np.ndarray:
        """
        计算均值 (第一矩):
          E[u](x) = Σ_{k=1}^{N} w_k u_k(x)

        返回: shape (N_x,) 或标量
        """
        return self.weights @ self.sol

    def compute_variance(self, mean: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算方差 (第二中心矩):
          Var[u](x) = Σ_{k=1}^{N} w_k (u_k(x) - E[u](x))²

        返回: shape (N_x,) 或标量
        """
        if mean is None:
            mean = self.compute_mean()
        deviation = self.sol - mean[np.newaxis, :]
        return self.weights @ (deviation ** 2)

    def compute_std(self, mean: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算标准差:
          σ[u](x) = √Var[u](x)
        """
        var = self.compute_variance(mean)
        return np.sqrt(np.maximum(var, 0.0))

    def compute_skewness(self, mean: Optional[np.ndarray] = None,
                         std: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算偏度 (第三标准化矩):
          γ₁(x) = E[(u-μ)³] / σ³

        偏度 > 0: 右偏 (正尾较长)
        偏度 < 0: 左偏 (负尾较长)
        偏度 = 0: 对称分布
        """
        if mean is None:
            mean = self.compute_mean()
        if std is None:
            std = self.compute_std(mean)

        deviation = self.sol - mean[np.newaxis, :]
        third_moment = self.weights @ (deviation ** 3)

        # 避免除以零
        std3 = std ** 3
        std3 = np.where(std3 > 1e-30, std3, 1e-30)
        return third_moment / std3

    def compute_kurtosis(self, mean: Optional[np.ndarray] = None,
                         std: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算超额峰度 (第四标准化矩 - 3):
          γ₂(x) = E[(u-μ)⁴] / σ⁴ - 3

        正态分布: γ₂ = 0
        厚尾分布: γ₂ > 0
        薄尾分布: γ₂ < 0
        """
        if mean is None:
            mean = self.compute_mean()
        if std is None:
            std = self.compute_std(mean)

        deviation = self.sol - mean[np.newaxis, :]
        fourth_moment = self.weights @ (deviation ** 4)

        std4 = std ** 4
        std4 = np.where(std4 > 1e-30, std4, 1e-30)
        return fourth_moment / std4 - 3.0

    def compute_confidence_interval(
        self, confidence: float = 0.95, mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算置信区间:
          CI = E[u] ± z_{α/2} × σ[u]

        其中 z_{α/2} 是标准正态的分位数:
          90% CI: z = 1.645
          95% CI: z = 1.960
          99% CI: z = 2.576

        参数:
            confidence: 置信水平 (0,1)

        返回:
            (lower, upper) 置信区间边界
        """
        from scipy.stats import norm
        z = norm.ppf((1.0 + confidence) / 2.0)

        if mean is None:
            mean = self.compute_mean()
        if std is None:
            std = self.compute_std(mean)

        return mean - z * std, mean + z * std

    def compute_probability_exceedance(
        self, threshold: float, mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算超越概率:
          P(u > threshold) ≈ Σ_{k: u_k > threshold} w_k

        这是蒙特卡洛式的非参数估计, 不假设正态性。

        参数:
            threshold: 阈值

        返回:
            P(u(x) > threshold), shape (N_x,) 或标量
        """
        mask = self.sol > threshold
        # 对每个空间点求权重和
        exceedance = np.zeros(self.sol.shape[1])
        for k in range(self.sol.shape[0]):
            exceedance += self.weights[k] * mask[k].astype(float)
        return exceedance

    def compute_all_moments(self) -> Dict[str, np.ndarray]:
        """
        计算所有统计矩并返回字典。
        """
        mean = self.compute_mean()
        std = self.compute_std(mean)
        skew = self.compute_skewness(mean, std)
        kurt = self.compute_kurtosis(mean, std)
        var = std ** 2

        return {
            'mean': mean,
            'variance': var,
            'std': std,
            'skewness': skew,
            'kurtosis': kurt,
            'cv': std / np.maximum(np.abs(mean), 1e-30),  # 变异系数
        }

    def compute_relative_error_vs_reference(
        self, reference_mean: np.ndarray, reference_var: np.ndarray,
        mean: Optional[np.ndarray] = None, var: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        计算与参考解的相对误差:
          ε_mean = ||E[u] - E_ref|| / ||E_ref||
          ε_var = ||Var[u] - Var_ref|| / ||Var_ref||

        用于评估稀疏网格近似的精度。
        """
        if mean is None:
            mean = self.compute_mean()
        if var is None:
            var = self.compute_variance(mean)

        ref_mean_norm = np.linalg.norm(reference_mean)
        ref_var_norm = np.linalg.norm(reference_var)

        err_mean = np.linalg.norm(mean - reference_mean) / max(ref_mean_norm, 1e-30)
        err_var = np.linalg.norm(var - reference_var) / max(ref_var_norm, 1e-30)

        return {
            'mean_relative_error': float(err_mean),
            'variance_relative_error': float(err_var),
        }
