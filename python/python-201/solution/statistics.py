"""
statistics.py — 统计分析模块
==============================
从 PCE 系数中计算统计量: 均值、方差、高阶矩、
Sobol 全局灵敏度指数、概率密度函数、失效概率。

核心公式:
  均值:
    E[u] = û_0 (归一化基: Ψ_0 = 1)

  方差:
    Var[u] = Σ_{k=1}^{P} û_k^2 (正交归一基)

  高阶矩:
    E[u^n] = Σ_{k_1,...,k_n} û_{k_1}...û_{k_n} <Ψ_{k_1}...Ψ_{k_n}>

  偏度:
    γ_1 = E[(u-μ)^3] / σ^3

  峰度:
    γ_2 = E[(u-μ)^4] / σ^4 - 3

  Sobol 指数 (一阶):
    S_i = Var_{ξ_i}[E[u|ξ_i]] / Var[u]

  Sobol 指数 (总效应):
    S_{Ti} = E_{ξ_{~i}}[Var_{ξ_i}[u|ξ_{~i}]] / Var[u]
           = 1 - Var_{ξ_{~i}}[E[u|ξ_{~i}]] / Var[u]

  部分方差:
    Var_{ξ_i}[E[u|ξ_i]] = Σ_{k∈S_i} û_k^2
    S_i = {k : i_k > 0, i_j = 0 ∀j≠i}

映射种子项目:
  - 068_ball_integrals: 高维积分 → 矩计算
  - 1180_Yuva12345: 贝叶斯统计 → 后验统计
  - 033_asa076: 正态 CDF → 失效概率
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from config import GlobalConfig
from polynomial_basis import OrthogonalPolynomialBasis
from measure import (gauss_quadrature, tensor_product_quadrature,
                     owen_t_function, failure_probability)


class PCEStatistics:
    """
    PCE 统计分析器。

    从 PCE 系数中直接提取统计量 (无需采样)。
    """

    def __init__(self, config: GlobalConfig,
                 basis: OrthogonalPolynomialBasis):
        self.config = config
        self.basis = basis
        self.P = basis.n_basis
        self.d = basis.n_dim

    def compute_mean(self, pce_coeffs: np.ndarray) -> float:
        """
        计算 PCE 均值。

        E[u] = û_0 (对归一化基 Ψ_0 = 1)

        参数:
            pce_coeffs: shape (P,) 或 (n_field, P)

        返回:
            mean: 均值
        """
        if pce_coeffs.ndim == 1:
            return float(pce_coeffs[0])
        return float(np.mean(pce_coeffs[:, 0]))

    def compute_variance(self, pce_coeffs: np.ndarray) -> float:
        """
        计算 PCE 方差。

        Var[u] = Σ_{k=1}^{P} û_k^2

        参数:
            pce_coeffs: PCE 系数

        返回:
            variance
        """
        if pce_coeffs.ndim == 1:
            return float(np.sum(pce_coeffs[1:] ** 2))
        return float(np.sum(pce_coeffs[:, 1:] ** 2, axis=1).mean())

    def compute_std(self, pce_coeffs: np.ndarray) -> float:
        """标准差"""
        return float(np.sqrt(max(self.compute_variance(pce_coeffs), 0.0)))

    def compute_skewness(self, pce_coeffs: np.ndarray,
                         quad_nodes: np.ndarray,
                         quad_weights: np.ndarray) -> float:
        """
        计算偏度 γ_1。

        γ_1 = E[(u-μ)^3] / σ^3

        使用求积近似计算三阶矩。

        参数:
            pce_coeffs:  PCE 系数
            quad_nodes:  求积节点
            quad_weights: 求积权重

        返回:
            skewness
        """
        mu = self.compute_mean(pce_coeffs)
        sigma = self.compute_std(pce_coeffs)

        if sigma < 1e-15:
            return 0.0

        # 在求积节点上计算 u
        Psi = self.basis.evaluate(quad_nodes)
        u_values = Psi @ pce_coeffs

        # 三阶中心矩
        centered = u_values - mu
        m3 = np.sum(quad_weights * centered ** 3)

        return float(m3 / sigma ** 3)

    def compute_kurtosis(self, pce_coeffs: np.ndarray,
                         quad_nodes: np.ndarray,
                         quad_weights: np.ndarray) -> float:
        """
        计算超额峰度 γ_2。

        γ_2 = E[(u-μ)^4] / σ^4 - 3

        参数:
            pce_coeffs, quad_nodes, quad_weights

        返回:
            excess_kurtosis
        """
        mu = self.compute_mean(pce_coeffs)
        sigma = self.compute_std(pce_coeffs)

        if sigma < 1e-15:
            return 0.0

        Psi = self.basis.evaluate(quad_nodes)
        u_values = Psi @ pce_coeffs
        centered = u_values - mu
        m4 = np.sum(quad_weights * centered ** 4)

        return float(m4 / sigma ** 4 - 3.0)

    def compute_sobol_first_order(self, pce_coeffs: np.ndarray
                                  ) -> np.ndarray:
        """
        计算一阶 Sobol 灵敏度指数。

        S_i = Var_{ξ_i}[E[u|ξ_i]] / Var[u]

        Var_{ξ_i}[E[u|ξ_i]] = Σ_{k∈S_i} û_k^2

        其中 S_i = {k : multi_index[k,i] > 0 且 multi_index[k,j]=0 ∀j≠i}

        即只有第 i 个分量非零的基函数。

        参数:
            pce_coeffs: shape (P,)

        返回:
            S: shape (d,), 一阶 Sobol 指数
        """
        if pce_coeffs.ndim > 1:
            pce_coeffs = pce_coeffs[0]

        total_var = self.compute_variance(pce_coeffs)
        if total_var < 1e-30:
            return np.zeros(self.d)

        S = np.zeros(self.d)
        indices = self.basis.indices

        for dim in range(self.d):
            partial_var = 0.0
            for k in range(1, self.P):
                mi = indices[k]
                # 检查是否只有 dim 分量非零
                is_pure = True
                for j in range(self.d):
                    if j != dim and mi[j] > 0:
                        is_pure = False
                        break
                if is_pure and mi[dim] > 0:
                    partial_var += pce_coeffs[k] ** 2
            S[dim] = partial_var / total_var

        return S

    def compute_sobol_total(self, pce_coeffs: np.ndarray
                            ) -> np.ndarray:
        """
        计算总效应 Sobol 指数。

        S_{Ti} = 1 - Var_{~i} / Var[u]

        Var_{~i} = Σ_{k: mi[k,i]=0} û_k^2

        即第 i 个分量为零的所有基函数的系数平方和。

        参数:
            pce_coeffs: shape (P,)

        返回:
            ST: shape (d,), 总效应 Sobol 指数
        """
        if pce_coeffs.ndim > 1:
            pce_coeffs = pce_coeffs[0]

        total_var = self.compute_variance(pce_coeffs)
        if total_var < 1e-30:
            return np.zeros(self.d)

        ST = np.zeros(self.d)
        indices = self.basis.indices

        for dim in range(self.d):
            # 与 dim 无关的方差
            var_without = 0.0
            for k in range(1, self.P):
                if indices[k, dim] == 0:
                    var_without += pce_coeffs[k] ** 2
            ST[dim] = 1.0 - var_without / total_var

        return np.clip(ST, 0.0, 1.0)

    def compute_pdf(self, pce_coeffs: np.ndarray,
                    n_grid: int = 200,
                    n_sigma: float = 4.0
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 PCE 代理模型估计概率密度函数。

        方法: 在均值 ± n_sigma*σ 上生成网格,
        通过大量采样计算经验 PDF。

        参数:
            pce_coeffs: PCE 系数
            n_grid:     PDF 网格点数
            n_sigma:    范围 (±n_sigma 倍标准差)

        返回:
            x_grid: PDF 网格点
            pdf:    密度值
        """
        mu = self.compute_mean(pce_coeffs)
        sigma = self.compute_std(pce_coeffs)

        if sigma < 1e-15:
            x_grid = np.array([mu])
            pdf = np.array([1.0])
            return x_grid, pdf

        x_grid = np.linspace(mu - n_sigma * sigma,
                             mu + n_sigma * sigma, n_grid)

        # 使用 Monte Carlo 采样估计 PDF
        n_mc = 10000
        rng = np.random.RandomState(123)

        # 从各维测度采样
        samples = np.zeros((n_mc, self.d))
        for dim_idx, meas in enumerate(self.config.measures):
            if meas.measure_type == "uniform":
                samples[:, dim_idx] = rng.uniform(
                    meas.params["a"], meas.params["b"], n_mc)
            elif meas.measure_type == "gauss":
                samples[:, dim_idx] = rng.normal(
                    meas.params["mu"], meas.params["sigma"], n_mc)
            else:
                samples[:, dim_idx] = rng.uniform(
                    meas.support[0], meas.support[1], n_mc)

        # 计算 PCE 值
        Psi = self.basis.evaluate(samples)
        u_samples = Psi @ pce_coeffs

        # 核密度估计
        bandwidth = 1.06 * sigma * n_mc ** (-0.2)  # Silverman 规则
        pdf = np.zeros(n_grid)
        for i, x in enumerate(x_grid):
            kernel = np.exp(-0.5 * ((u_samples - x) / bandwidth) ** 2)
            pdf[i] = np.mean(kernel) / (bandwidth * np.sqrt(2 * np.pi))

        # 归一化
        dx = x_grid[1] - x_grid[0] if n_grid > 1 else 1.0
        pdf_sum = np.sum(pdf) * dx
        if pdf_sum > 1e-15:
            pdf /= pdf_sum

        return x_grid, pdf

    def compute_percentiles(self, pce_coeffs: np.ndarray,
                            percentiles: List[float] = None
                            ) -> Dict[float, float]:
        """
        计算分位数。

        使用 MC 采样从 PCE 代理模型计算。
        """
        if percentiles is None:
            percentiles = [5.0, 25.0, 50.0, 75.0, 95.0]

        n_mc = 10000
        rng = np.random.RandomState(456)

        samples = np.zeros((n_mc, self.d))
        for dim_idx, meas in enumerate(self.config.measures):
            if meas.measure_type == "gauss":
                samples[:, dim_idx] = rng.normal(
                    meas.params["mu"], meas.params["sigma"], n_mc)
            elif meas.measure_type == "uniform":
                samples[:, dim_idx] = rng.uniform(
                    meas.params["a"], meas.params["b"], n_mc)
            else:
                samples[:, dim_idx] = rng.uniform(
                    meas.support[0], meas.support[1], n_mc)

        Psi = self.basis.evaluate(samples)
        u_samples = Psi @ pce_coeffs

        result = {}
        for p in percentiles:
            result[p] = float(np.percentile(u_samples, p))

        return result

    def compute_all_statistics(self, pce_coeffs: np.ndarray
                               ) -> dict:
        """
        计算所有统计量的完整报告。

        参数:
            pce_coeffs: PCE 系数

        返回:
            stats: 统计量字典
        """
        # 准备求积规则
        n_q = min(self.P + 3, 8)
        q_nodes, q_weights = tensor_product_quadrature(
            self.config.measures, n_q)

        mean = self.compute_mean(pce_coeffs)
        var = self.compute_variance(pce_coeffs)
        std = np.sqrt(max(var, 0.0))

        stats = {
            "mean": mean,
            "variance": var,
            "std": std,
            "cv": std / abs(mean) if abs(mean) > 1e-15 else float('inf'),
            "sobol_first_order": self.compute_sobol_first_order(
                pce_coeffs),
            "sobol_total": self.compute_sobol_total(pce_coeffs),
            "n_basis": self.P,
        }

        # 偏度和峰度 (如果求积点足够)
        if len(q_nodes) > 0:
            stats["skewness"] = self.compute_skewness(
                pce_coeffs, q_nodes, q_weights)
            stats["kurtosis"] = self.compute_kurtosis(
                pce_coeffs, q_nodes, q_weights)

        # 分位数
        stats["percentiles"] = self.compute_percentiles(pce_coeffs)

        return stats

    def format_statistics(self, stats: dict,
                          var_names: List[str] = None) -> str:
        """格式化统计量为可读字符串"""
        if var_names is None:
            var_names = [f"ξ_{i}" for i in range(self.d)]

        lines = [
            "=" * 50,
            "PCE 统计分析结果",
            "=" * 50,
            f"均值 (Mean):      {stats['mean']:.6e}",
            f"方差 (Variance):  {stats['variance']:.6e}",
            f"标准差 (Std):     {stats['std']:.6e}",
            f"变异系数 (CV):    {stats.get('cv', 'N/A'):.4f}"
            if isinstance(stats.get('cv'), float) else
            f"变异系数 (CV):    {stats.get('cv', 'N/A')}",
            "",
            "Sobol 一阶灵敏度指数:",
        ]
        S = stats.get("sobol_first_order", np.zeros(self.d))
        for i, name in enumerate(var_names):
            lines.append(f"  S_{name} = {S[i]:.4f}" if i < len(S)
                         else f"  S_{name} = 0.0000")

        lines.append("")
        lines.append("Sobol 总效应灵敏度指数:")
        ST = stats.get("sobol_total", np.zeros(self.d))
        for i, name in enumerate(var_names):
            lines.append(f"  ST_{name} = {ST[i]:.4f}" if i < len(ST)
                         else f"  ST_{name} = 0.0000")

        if "skewness" in stats:
            lines.extend([
                "",
                f"偏度 (Skewness):  {stats['skewness']:.4f}",
                f"峰度 (Kurtosis):  {stats['kurtosis']:.4f}",
            ])

        if "percentiles" in stats:
            lines.extend(["", "分位数:"])
            for p, v in stats["percentiles"].items():
                lines.append(f"  P{p:.0f}: {v:.6e}")

        lines.append("=" * 50)
        return "\n".join(lines)
