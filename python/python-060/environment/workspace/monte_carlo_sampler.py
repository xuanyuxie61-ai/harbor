"""
蒙特卡洛采样与参数扰动模块

本模块实现平流层化学模型的随机参数采样，包括：
- 多维正态/均匀分布随机数生成
- 反应速率参数的随机扰动
- 人为排放源的不确定性采样
- 基于 Cholesky 分解的相关参数采样

科学背景:
平流层化学模型中的参数不确定性来源于:
1. 实验室测定的反应速率常数误差 (~10-30%)
2. 光解截面的光谱不确定性
3. 人为排放清单的时空变异
4. 气象场驱动的传输不确定性

科学公式:
1. 多元正态分布:
   X ~ N(μ, Σ)
   其中 Σ 为协方差矩阵

2. Cholesky 采样:
   X = μ + L Z
   其中 Σ = L L^T, Z ~ N(0, I)

3. 对数正态扰动 (保证正值):
   k' = k * exp(σ * Z)
   log(k') ~ N(log(k), σ²)

4. 拉丁超立方采样 (LHS):
   将每个维度分为 n 层，每层恰好采样一次

融入原项目: 1006_random_data (多维随机数生成), 547_human_data (排放轮廓参数化)
"""

import numpy as np
from typing import Tuple, Optional, Dict


class RandomDataGenerator:
    """
    多维随机数据生成器
    融入 1006_random_data 的核心算法
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def normal_square(self, n: int, d: int) -> np.ndarray:
        """
        在 d 维空间中生成 n 个标准正态分布点
        X ~ N(0, I_d)
        """
        if n <= 0 or d <= 0:
            raise ValueError("n > 0 且 d > 0")
        return self.rng.standard_normal((n, d))

    def uniform_in_hypercube(self, n: int, d: int,
                              a: float = 0.0, b: float = 1.0) -> np.ndarray:
        """
        在 d 维超立方体 [a,b]^d 中生成均匀随机点
        """
        if n <= 0 or d <= 0:
            raise ValueError("n > 0 且 d > 0")
        return a + (b - a) * self.rng.random((n, d))

    def uniform_in_hypersphere(self, n: int, d: int,
                                radius: float = 1.0) -> np.ndarray:
        """
        在 d 维超球内生成均匀随机点
        方法: 先生成正态点，再归一化并缩放
        """
        if n <= 0 or d <= 0:
            raise ValueError("n > 0 且 d > 0")
        x = self.rng.standard_normal((n, d))
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        u = self.rng.random((n, 1)) ** (1.0 / d)
        return radius * u * x / (norms + 1e-30)

    def direction_uniform_nd(self, n: int, d: int) -> np.ndarray:
        """
        在 d 维空间中生成均匀随机方向向量
        (单位球面上的均匀分布)
        """
        x = self.rng.standard_normal((n, d))
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (norms + 1e-30)

    def normal_circular(self, n: int) -> np.ndarray:
        """
        圆内二维正态分布 (径向分布)
        """
        r = np.sqrt(-2.0 * np.log(self.rng.random(n)))
        theta = 2.0 * np.pi * self.rng.random(n)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack([x, y])

    def latin_hypercube(self, n: int, d: int) -> np.ndarray:
        """
        拉丁超立方采样
        """
        if n <= 0 or d <= 0:
            raise ValueError("n > 0 且 d > 0")

        result = np.zeros((n, d))
        for i in range(d):
            perm = self.rng.permutation(n)
            result[:, i] = (perm + self.rng.random(n)) / n

        return result


class EmissionProfileSampler:
    """
    排放源轮廓采样器
    融入 547_human_data 的参数化轮廓思想
    """

    def __init__(self):
        self.rng = np.random.default_rng(123)

    def gaussian_emission_profile(self, z_km: np.ndarray,
                                   peak_height: float = 0.0,
                                   width: float = 5.0,
                                   total_emission: float = 1.0e6) -> np.ndarray:
        """
        高斯型排放轮廓
        E(z) = E_total * exp(-(z - z_peak)² / (2 σ²)) / (√(2π) σ)
        """
        profile = np.exp(-((z_km - peak_height) / width) ** 2)
        profile = profile / (np.sum(profile) + 1e-30)
        return total_emission * profile

    def multi_source_profile(self, z_km: np.ndarray,
                              sources: list) -> np.ndarray:
        """
        多源叠加排放轮廓
        """
        profile = np.zeros_like(z_km)
        for src in sources:
            p = self.gaussian_emission_profile(
                z_km,
                peak_height=src.get('height', 0.0),
                width=src.get('width', 5.0),
                total_emission=src.get('total', 1e6)
            )
            profile += p
        return profile

    def sample_profile_parameters(self, n_samples: int) -> Dict:
        """
        随机采样排放轮廓参数
        """
        samples = {
            'n2o_peak_height_km': self.rng.normal(0.0, 2.0, n_samples),
            'n2o_width_km': self.rng.uniform(3.0, 8.0, n_samples),
            'cfc11_peak_height_km': self.rng.normal(0.0, 1.5, n_samples),
            'cfc11_width_km': self.rng.uniform(2.0, 5.0, n_samples),
            'nox_aircraft_peak_km': self.rng.normal(11.0, 1.0, n_samples),
            'nox_aircraft_width_km': self.rng.uniform(1.5, 3.0, n_samples),
        }
        return samples

    def perturbed_profile(self, base_profile: np.ndarray,
                          relative_std: float = 0.1) -> np.ndarray:
        """
        对基础排放轮廓施加随机扰动
        """
        noise = self.rng.lognormal(0.0, relative_std, len(base_profile))
        perturbed = base_profile * noise
        return np.clip(perturbed, 0.0, 1e20)


class CorrelatedParameterSampler:
    """
    基于 Cholesky 分解的相关参数采样器
    """

    def __init__(self, n_params: int = 10):
        self.n_params = n_params
        self.rng = np.random.default_rng(456)

    def build_covariance_matrix(self, sigmas: np.ndarray,
                                 correlations: Optional[np.ndarray] = None) -> np.ndarray:
        """
        构建协方差矩阵
        Σ_ii = σ_i²
        Σ_ij = ρ_ij σ_i σ_j
        """
        if len(sigmas) != self.n_params:
            raise ValueError("sigmas 长度不匹配")

        Sigma = np.diag(sigmas ** 2)

        if correlations is not None:
            if correlations.shape != (self.n_params, self.n_params):
                raise ValueError("correlations 形状不匹配")
            for i in range(self.n_params):
                for j in range(i + 1, self.n_params):
                    Sigma[i, j] = correlations[i, j] * sigmas[i] * sigmas[j]
                    Sigma[j, i] = Sigma[i, j]

        # 确保正定性
        eigvals = np.linalg.eigvalsh(Sigma)
        if np.min(eigvals) < 1e-14:
            Sigma += (1e-12 - np.min(eigvals)) * np.eye(self.n_params)

        return Sigma

    def sample(self, mu: np.ndarray, Sigma: np.ndarray,
               n_samples: int = 100) -> np.ndarray:
        """
        使用 Cholesky 分解生成相关正态样本
        X = μ + L Z, 其中 Σ = L L^T
        """
        if len(mu) != self.n_params:
            raise ValueError("mu 长度不匹配")

        try:
            L = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            # 添加正则化
            Sigma_reg = Sigma + 1e-10 * np.eye(self.n_params)
            L = np.linalg.cholesky(Sigma_reg)

        Z = self.rng.standard_normal((n_samples, self.n_params))
        X = mu + Z @ L.T
        return X

    def lognormal_sample(self, mu_log: np.ndarray, Sigma: np.ndarray,
                         n_samples: int = 100) -> np.ndarray:
        """
        对数正态分布采样 (保证参数正值)
        """
        X_normal = self.sample(mu_log, Sigma, n_samples)
        return np.exp(X_normal)


class OzoneMonteCarloExperiment:
    """
    臭氧模型的蒙特卡洛实验框架
    """

    def __init__(self, n_ensemble: int = 500):
        self.n_ensemble = n_ensemble
        self.random_gen = RandomDataGenerator(seed=789)
        self.emission_sampler = EmissionProfileSampler()
        self.param_sampler = CorrelatedParameterSampler(n_params=8)

    def run_parameter_perturbation_experiment(self) -> Dict:
        """
        运行参数扰动蒙特卡洛实验
        """
        # 定义参数均值和标准差
        param_names = [
            'k_O_O2_M', 'k_O_O3', 'k_NO_O3', 'k_Cl_O3',
            'k_OH_O3', 'J_O2', 'J_O3', 'Kzz_scale'
        ]
        mu = np.array([-33.5, -11.7, -11.7, -10.5,
                       -11.8, -10.0, -2.0, 0.0])  # log尺度
        sigma_rel = np.array([0.15, 0.10, 0.12, 0.20,
                              0.15, 0.10, 0.08, 0.25])

        # 构建相关性矩阵 (相邻反应相关)
        corr = np.eye(8)
        corr[0, 1] = corr[1, 0] = 0.3  # O相关反应
        corr[2, 3] = corr[3, 2] = 0.2  # 催化循环
        corr[5, 6] = corr[6, 5] = 0.4  # 光解相关

        Sigma = self.param_sampler.build_covariance_matrix(sigma_rel, corr)
        samples = self.param_sampler.sample(mu, Sigma, self.n_ensemble)

        # 计算臭氧柱的响应
        o3_columns = []
        for i in range(self.n_ensemble):
            params = 10.0 ** samples[i]
            # 简化响应模型
            o3 = 300.0 * (params[5] / 1e-10) ** 0.3 * \
                 (params[6] / 1e-2) ** (-0.2) * \
                 (params[7] / 1.0) ** (-0.15)
            o3_columns.append(o3)

        o3_columns = np.array(o3_columns)

        return {
            'param_names': param_names,
            'samples': samples,
            'o3_columns': o3_columns,
            'o3_mean': np.mean(o3_columns),
            'o3_std': np.std(o3_columns),
            'o3_ci_95': (np.percentile(o3_columns, 2.5),
                        np.percentile(o3_columns, 97.5)),
            'o3_min': np.min(o3_columns),
            'o3_max': np.max(o3_columns)
        }

    def run_emission_uncertainty_experiment(self, z_km: np.ndarray) -> Dict:
        """
        运行排放不确定性实验
        """
        n2o_profiles = []
        for _ in range(self.n_ensemble):
            height = self.random_gen.rng.normal(0.0, 2.0)
            width = self.random_gen.rng.uniform(3.0, 8.0)
            profile = self.emission_sampler.gaussian_emission_profile(
                z_km, peak_height=height, width=width)
            n2o_profiles.append(profile)

        n2o_profiles = np.array(n2o_profiles)

        return {
            'n2o_mean_profile': np.mean(n2o_profiles, axis=0),
            'n2o_std_profile': np.std(n2o_profiles, axis=0),
            'n2o_q05': np.percentile(n2o_profiles, 5, axis=0),
            'n2o_q95': np.percentile(n2o_profiles, 95, axis=0),
        }
