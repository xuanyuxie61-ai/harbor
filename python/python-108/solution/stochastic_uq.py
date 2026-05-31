# -*- coding: utf-8 -*-
"""
stochastic_uq.py
不确定性量化与蒙特卡洛随机分析模块

核心公式与物理背景
------------------
1. 高维超球面均匀采样
   在 m 维单位超球面 S^{m-1} 上均匀采样：
       1) 生成 g ~ N(0, I_m)
       2) 归一化 u = g / ||g||_2
   理论性质：对任意正交变换 Q，Qu 与 u 同分布。

2. 超球面角度统计
   对两个独立均匀采样点 u, v ∈ S^{m-1}，夹角 θ 满足：
       cos θ = u·v
   当 m → ∞ 时，E[|cos θ|] → 0，即随机方向趋于正交。
   该性质用于分析制造误差矢量的独立性。

3. 随机变量生成（ranlib 算法族）
   - 标准正态：Ahrens-Dieter 算法 FL（ rejection + 分区）
   - Gamma 分布：Ahrens-Dieter 算法 GD（modified rejection）
   - Beta 分布：Cheng 算法 BB/BC（logistic transformation）
   这些分布用于材料参数（热导率、折射率）的不确定性建模。

4. 蒙特卡洛不确定性传播
   对输出量 Y = f(X)，其中 X 为随机输入向量：
       E[Y] ≈ (1/N) Σ f(X_i)
       Var(Y) ≈ (1/(N-1)) Σ (f(X_i) - E[Y])²

融合来源
--------
- 563_hypersphere_angle : 高维超球面采样与角度统计
- 1012_ranlib           : 多分布随机数生成（正态、Gamma、Beta、指数等）
"""

import numpy as np
from typing import Tuple, Optional, Callable


class HypersphereSampler:
    """
    m 维单位超球面 S^{m-1} 上的均匀采样与统计量计算。
    """

    def __init__(self, dim: int, seed: Optional[int] = None):
        if dim < 2:
            raise ValueError("维度必须 ≥ 2")
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    def sample(self, n: int) -> np.ndarray:
        """
        生成 n 个 S^{m-1} 上的均匀采样点（N×m 数组）。
        """
        g = self.rng.standard_normal(size=(n, self.dim))
        norms = np.linalg.norm(g, axis=1, keepdims=True)
        norms = np.where(norms < 1e-15, 1.0, norms)
        return g / norms

    def angle_statistics(self, n_pairs: int) -> dict:
        """
        采样 n_pairs 对独立点，计算夹角统计量。

        返回
        ----
        dict 包含 mean_abs_cos, std_abs_cos, mean_angle_rad, std_angle_rad
        """
        u = self.sample(n_pairs)
        v = self.sample(n_pairs)
        cos_theta = np.sum(u * v, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        abs_cos = np.abs(cos_theta)
        theta = np.arccos(abs_cos)
        return {
            "dim": self.dim,
            "n_pairs": n_pairs,
            "mean_abs_cos": float(np.mean(abs_cos)),
            "std_abs_cos": float(np.std(abs_cos, ddof=1)),
            "mean_angle_rad": float(np.mean(theta)),
            "std_angle_rad": float(np.std(theta, ddof=1)),
            "theoretical_mean_abs_cos": self._theoretical_mean_abs_cos(),
        }

    def _theoretical_mean_abs_cos(self) -> float:
        """
        理论上 E[|cos θ|] 的闭式表达：
            E[|cos θ|] = Γ(m/2) / (√π · Γ((m+1)/2))
        """
        from scipy.special import gamma as Gamma
        m = self.dim
        return float(Gamma(m / 2.0) / (np.sqrt(np.pi) * Gamma((m + 1) / 2.0)))


class RandomVariateGenerator:
    """
    科研级随机变量生成器，基于经典 ranlib 算法思想。
    提供多种分布的高质量采样。
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    # ---------- 基本分布 ----------
    def uniform(self, a: float = 0.0, b: float = 1.0, size: Optional[Tuple] = None) -> np.ndarray:
        """均匀分布 U(a,b)"""
        return self.rng.uniform(a, b, size=size)

    def normal(self, mu: float = 0.0, sigma: float = 1.0, size: Optional[Tuple] = None) -> np.ndarray:
        """
        标准正态分布 N(μ,σ²)。
        底层使用 Ziggurat 或 Box-Muller（numpy 实现），
        等效于 ranlib 的 snorm。
        """
        return self.rng.normal(mu, sigma, size=size)

    def exponential(self, lam: float = 1.0, size: Optional[Tuple] = None) -> np.ndarray:
        """
        指数分布 Exp(λ)，密度 p(x) = λ·exp(-λx)，x≥0。
         ranlib 中 sgamma 的特例 (shape=1)。
        """
        if lam <= 0:
            raise ValueError("λ 必须 > 0")
        return self.rng.exponential(1.0 / lam, size=size)

    def gamma(self, shape: float, scale: float = 1.0, size: Optional[Tuple] = None) -> np.ndarray:
        """
        Gamma 分布 Γ(α, β)。
        ranlib 中 sgamma 使用 Ahrens-Dieter GD 算法（modified rejection）。
        这里直接调用 numpy 的等价实现，其底层也是拒绝采样。
        """
        if shape <= 0 or scale <= 0:
            raise ValueError("shape 和 scale 必须 > 0")
        return self.rng.gamma(shape, scale, size=size)

    def beta(self, a: float, b: float, size: Optional[Tuple] = None) -> np.ndarray:
        """
        Beta 分布 Beta(α, β)。
        ranlib 中 genbet 使用 Cheng 的 BB/BC 算法（logistic transformation）。
        """
        if a <= 0 or b <= 0:
            raise ValueError("a 和 b 必须 > 0")
        return self.rng.beta(a, b, size=size)

    def chi_square(self, df: int, size: Optional[Tuple] = None) -> np.ndarray:
        """卡方分布 χ²(k)，k 个独立标准正态的平方和。"""
        if df <= 0:
            raise ValueError("自由度必须 > 0")
        return self.rng.chisquare(df, size=size)

    # ---------- 组合分布 ----------
    def multivariate_normal(self, mean: np.ndarray, cov: np.ndarray, size: Optional[int] = None) -> np.ndarray:
        """多元正态分布 N(μ, Σ)。"""
        return self.rng.multivariate_normal(mean, cov, size=size)


class MonteCarloUQ:
    """
    蒙特卡洛不确定性传播分析器。
    用于评估微腔制造参数波动对谐振特性的影响。
    """

    def __init__(self, n_samples: int = 2000, seed: Optional[int] = 42):
        self.n_samples = n_samples
        self.rng = RandomVariateGenerator(seed)
        self.hyper = HypersphereSampler(dim=5, seed=seed)

    def parameter_perturbation(self,
                                base_params: dict,
                                std_params: dict) -> dict:
        """
        对物理参数施加随机扰动。
        参数向量包括：R_major, r_minor, n_ring, kappa, alpha_abs。
        """
        perturbed = {}
        for key, base in base_params.items():
            std = std_params.get(key, 0.0)
            if std > 0:
                perturbed[key] = base + self.rng.normal(0.0, std)
            else:
                perturbed[key] = base
        # 边界保护：确保物理量正
        for key in ["R_major", "r_minor", "n_ring", "kappa", "alpha_abs"]:
            if key in perturbed and perturbed[key] <= 0:
                perturbed[key] = base_params[key] * 0.5
        return perturbed

    def run_mc_propagation(self,
                           base_params: dict,
                           std_params: dict,
                           forward_model: Callable[[dict], dict]) -> dict:
        """
        执行蒙特卡洛不确定性传播。

        参数
        ----
        base_params : dict
            名义参数值
        std_params : dict
            各参数的标准差
        forward_model : Callable[[dict], dict]
            前向模型，输入参数字典，输出结果字典（须包含标量输出）

        返回
        ----
        results : dict
            包含均值、标准差、5%/95% 分位数的统计汇总
        """
        outputs = []
        for _ in range(self.n_samples):
            p = self.parameter_perturbation(base_params, std_params)
            try:
                out = forward_model(p)
                outputs.append(out)
            except Exception:
                # 跳过数值失败样本
                continue

        if not outputs:
            raise RuntimeError("所有蒙特卡洛样本均失败")

        # 收集所有标量键
        scalar_keys = set()
        for out in outputs:
            for k, v in out.items():
                if np.isscalar(v):
                    scalar_keys.add(k)

        summary = {"n_success": len(outputs), "n_requested": self.n_samples}
        for k in scalar_keys:
            vals = np.array([o[k] for o in outputs if k in o])
            summary[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "p5": float(np.percentile(vals, 5)),
                "p95": float(np.percentile(vals, 95)),
            }
        return summary

    def sample_on_hypersphere(self, radius: float, n: int) -> np.ndarray:
        """
        在 5D 参数空间的超球面上均匀采样（用于分析扰动方向的各向同性）。
        返回 n×5 数组，每行为一个扰动向量（已缩放至给定半径）。
        """
        pts = self.hyper.sample(n)
        # 缩放扰动幅度
        return radius * pts
