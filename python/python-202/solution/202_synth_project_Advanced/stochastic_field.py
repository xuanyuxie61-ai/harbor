"""
随机场表示模块 (Stochastic Field Representation)
==================================================
实现随机场的 Karhunen-Loève 展开和随机采样。

KL 展开公式:
  随机场 κ(x,ω) 的 KL 展开:
    κ(x,ω) = μ_κ(x) + Σ_{k=1}^{K} √λ_k φ_k(x) ξ_k(ω)

  其中:
    - μ_κ(x): 均值函数
    - λ_k, φ_k(x): 协方差核 C(x,x') 的特征值和特征函数
    - ξ_k(ω): 独立标准正态随机变量

  特征值问题 (Fredholm 积分方程):
    ∫_Ω C(x,x') φ_k(x') dx' = λ_k φ_k(x),  x ∈ Ω

  对于平方指数核 C(x,x') = σ² exp(-|x-x'|²/(2l_c²)):
    在 [0,L] 上, 特征函数近似为:
      φ_k(x) = sin(kπ(x-a)/(b-a)) / √((b-a)/2)
    特征值:
      λ_k = σ² l_c² (kπ)² / ((kπ)² l_c² / L² + 1)² × (2/L)
    更精确: 通过数值求解离散化特征值问题获得。

  对数变换 (确保正定性):
    log κ(x,ω) = log μ_κ + Σ √λ_k φ_k(x) ξ_k(ω)
    ⟹ κ(x,ω) = μ_κ × exp(Σ √λ_k φ_k(x) ξ_k(ω))

参考文献:
  Karhunen, K. (1946). Über lineare Methoden in der Wahrscheinlichkeitsrechnung.
  Loève, M. (1977). Probability Theory II.
"""

import numpy as np
from scipy import linalg as la
from typing import Tuple, Optional


class KarhunenLoeveExpansion:
    """
    Karhunen-Loève 展开: 随机场的低维参数化表示。

    给定协方差核 C(x,x'), 通过求解 Fredholm 积分方程的特征值问题,
    获得截断 KL 展开, 将无限维随机场降维到 K 个随机变量。

    属性:
        eigenvalues: 特征值 λ_k, shape (K,)
        eigenfunctions: 特征函数值 φ_k(x_i), shape (K, N_x)
        n_terms: 截断阶数 K
        spatial_grid: 空间离散网格, shape (N_x,)
        covariance_scale: 协方差幅度 σ²
        correlation_length: 相关长度 l_c
    """

    def __init__(
        self,
        spatial_grid: np.ndarray,
        covariance_scale: float = 1.0,
        correlation_length: float = 0.2,
        n_terms: int = 10,
        kernel_type: str = 'squared_exponential'
    ):
        """
        参数:
            spatial_grid: 空间离散点, shape (N_x,)
            covariance_scale: 协方差幅度 σ²
            correlation_length: 相关长度 l_c
            n_terms: KL 展开截断阶数 K
            kernel_type: 协方差核类型
                'squared_exponential': C(r) = σ² exp(-r²/(2l_c²))
                'exponential': C(r) = σ² exp(-|r|/l_c)
                'matern_32': C(r) = σ² (1 + √3|r|/l_c) exp(-√3|r|/l_c)
        """
        self.spatial_grid = np.asarray(spatial_grid, dtype=np.float64)
        self.n_spatial = self.spatial_grid.size
        self.covariance_scale = covariance_scale
        self.correlation_length = correlation_length
        self.n_terms = min(n_terms, self.n_spatial)
        self.kernel_type = kernel_type

        # 构造协方差矩阵并求解特征值问题
        self.C = self._build_covariance_matrix()
        self.eigenvalues, self.eigenfunctions = self._solve_eigenvalue_problem()

        # 能量捕获比
        total_energy = np.sum(np.abs(np.linalg.eigvalsh(self.C)))
        captured_energy = np.sum(self.eigenvalues)
        self.energy_ratio = captured_energy / max(total_energy, 1e-15)

    def _build_covariance_matrix(self) -> np.ndarray:
        """
        构造协方差矩阵 C:
          C_{ij} = C(x_i, x_j) = σ² K(x_i, x_j)

        其中 K 是归一化协方差核函数。

        平方指数核 (Squared Exponential):
          K(r) = exp(-r²/(2l_c²))
          性质: 无穷次可微, 生成的随机场是光滑的

        指数核 (Exponential):
          K(r) = exp(-|r|/l_c)
          性质: 连续但不可微, 生成的随机场粗糙

        Matérn 3/2 核:
          K(r) = (1 + √3|r|/l_c) exp(-√3|r|/l_c)
          性质: 一次可微, 介于 SE 和 Exp 之间
        """
        x = self.spatial_grid
        dx = np.abs(x[:, None] - x[None, :])
        lc = self.correlation_length
        sigma2 = self.covariance_scale

        if self.kernel_type == 'squared_exponential':
            C = sigma2 * np.exp(-dx ** 2 / (2.0 * lc ** 2))
        elif self.kernel_type == 'exponential':
            C = sigma2 * np.exp(-dx / lc)
        elif self.kernel_type == 'matern_32':
            sqrt3_dx_lc = np.sqrt(3.0) * dx / lc
            C = sigma2 * (1.0 + sqrt3_dx_lc) * np.exp(-sqrt3_dx_lc)
        elif self.kernel_type == 'matern_52':
            sqrt5_dx_lc = np.sqrt(5.0) * dx / lc
            C = sigma2 * (1.0 + sqrt5_dx_lc + 5.0 * dx ** 2 / (3.0 * lc ** 2)) * np.exp(-sqrt5_dx_lc)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        # 确保正定性 (添加微小正则化)
        C += 1e-12 * np.eye(self.n_spatial)
        return C

    def _solve_eigenvalue_problem(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解离散化 Fredholm 积分方程的特征值问题:
          C φ = λ φ

        使用对称特征值求解器 (因为 C 是对称正定的)。
        返回最大的 K 个特征值和对应特征函数。

        特征值 λ_k 的衰减速率取决于:
          - 协方差核的光滑性 (SE 最快, Exp 最慢)
          - 相关长度 l_c (l_c 越大, 衰减越快, 需要的项越少)
          - 域的大小 L (L/l_c 越大, 需要的项越多)
        """
        eigenvalues, eigenvectors = la.eigh(self.C)

        # 降序排列
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 取前 K 个
        K = self.n_terms
        eigenvalues = eigenvalues[:K]
        eigenfunctions = eigenvectors[:, :K].T  # shape (K, N_x)

        # 确保特征值非负
        eigenvalues = np.maximum(eigenvalues, 0.0)

        return eigenvalues, eigenfunctions

    def evaluate_field(
        self, xi: np.ndarray, mean_value: float = 1.0,
        log_transform: bool = True
    ) -> np.ndarray:
        """
        在给定的随机变量实现下评估随机场。

        对数变换模式 (确保正定性):
          κ(x) = μ × exp(Σ_{k=1}^{K} √λ_k φ_k(x) ξ_k)

        直接模式:
          κ(x) = μ + Σ_{k=1}^{K} √λ_k φ_k(x) ξ_k

        参数:
            xi: 随机变量实现, shape (K,) 或 shape (N_samples, K)
            mean_value: 均值 μ
            log_transform: 是否使用对数变换

        返回:
            field: 随机场值, shape (N_x,) 或 shape (N_samples, N_x)
        """
        xi = np.atleast_1d(xi).astype(np.float64)

        if xi.ndim == 1:
            # 单次实现
            expansion = np.zeros(self.n_spatial)
            for k in range(min(len(xi), self.n_terms)):
                expansion += np.sqrt(self.eigenvalues[k]) * self.eigenfunctions[k] * xi[k]

            if log_transform:
                field = mean_value * np.exp(expansion)
                # 数值安全: 限制指数范围
                field = np.clip(field, 1e-10, 1e10)
            else:
                field = mean_value + expansion
                # 确保非负
                field = np.maximum(field, 1e-10)
            return field
        else:
            # 批量实现
            N_samples = xi.shape[0]
            fields = np.zeros((N_samples, self.n_spatial))
            for s in range(N_samples):
                fields[s] = self.evaluate_field(
                    xi[s], mean_value, log_transform
                )
            return fields

    def truncated_variance(self) -> float:
        """
        KL 展开捕获的截断方差:
          Var_KL = Σ_{k=1}^{K} λ_k

        总方差: Var = ∫ C(x,x) dx ≈ trace(C) × Δx
        """
        return float(np.sum(self.eigenvalues))

    def __repr__(self) -> str:
        return (f"KL Expansion(K={self.n_terms}, σ²={self.covariance_scale}, "
                f"l_c={self.correlation_length}, energy={self.energy_ratio:.4f})")


class RandomFieldSampler:
    """
    随机场采样器: 从 KL 展开生成随机场实现。

    采样策略:
    1. 从标准正态分布 N(0,1) 采样 ξ = (ξ_1,...,ξ_K)
    2. 通过 KL 展开构造随机场实现 κ(x,ω)

    支持多种采样方法:
    - 蒙特卡洛 (MC): 独立同分布采样
    - 拉丁超立方采样 (LHS): 分层采样, 减少方差
    - 准蒙特卡洛 (QMC): Sobol/Halton 低差异序列
    """

    def __init__(self, kl_expansion: KarhunenLoeveExpansion):
        self.kl = kl_expansion
        self.K = kl_expansion.n_terms

    def monte_carlo_sample(
        self, n_samples: int, seed: Optional[int] = None
    ) -> np.ndarray:
        """
        蒙特卡洛采样:
          ξ^{(s)} ~ N(0, I_K),  s = 1,...,N

        方差收敛速率: O(N^{-1/2})
        """
        rng = np.random.RandomState(seed)
        return rng.randn(n_samples, self.K)

    def latin_hypercube_sample(
        self, n_samples: int, seed: Optional[int] = None
    ) -> np.ndarray:
        """
        拉丁超立方采样 (Latin Hypercube Sampling):
          将 [0,1] 分为 N 等份, 每份取一个随机点
          然后通过逆 CDF 变换到目标分布

        相比 MC, LHS 的方差通常更小 (尤其对单调函数):
          Var_LHS ≤ Var_MC 对于单调函数 f
        """
        rng = np.random.RandomState(seed)
        samples = np.zeros((n_samples, self.K))

        for d in range(self.K):
            # 均匀分层
            perm = rng.permutation(n_samples)
            u = (perm + rng.uniform(size=n_samples)) / n_samples
            # 正态逆 CDF
            from scipy.stats import norm
            samples[:, d] = norm.ppf(u)

        return samples

    def evaluate_samples(
        self,
        samples: np.ndarray,
        mean_value: float = 1.0,
        log_transform: bool = True
    ) -> np.ndarray:
        """
        评估多个随机场实现。

        参数:
            samples: shape (N_samples, K)
            mean_value: 均值
            log_transform: 是否使用对数变换

        返回:
            fields: shape (N_samples, N_x)
        """
        return self.kl.evaluate_field(samples, mean_value, log_transform)
