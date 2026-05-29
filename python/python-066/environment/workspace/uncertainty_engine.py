"""
uncertainty_engine.py
================================================================================
地下水模型不确定性量化（UQ）蒙特卡洛引擎

基于种子项目：
  - 189_clock_solitaire_simulation：离散事件蒙特卡洛统计采样框架

科学背景：
  地下水模型充满不确定性：水力传导度 K 的空间分布、弥散度 α_L、边界条件、
  污染源强度等均无法精确测定。不确定性量化（UQ）通过概率方法评估这些不确定性
  对模型预测（如浓度超标概率）的影响。

  核心数学框架：
    1. 输入不确定性由联合概率密度函数 π(β) 描述
    2. 模型输出 Y = M(β) 为随机变量
    3. 蒙特卡洛估计：
         E[Y] ≈ (1/N) Σ_{i=1}^N M(β_i)
         Var[Y] ≈ (1/N) Σ_{i=1}^N (M(β_i) - E[Y])²
    4. 超标概率：
         P(Y > Y_c) ≈ (1/N) Σ_{i=1}^N I(M(β_i) > Y_c)

  低差异序列（QMC）替代伪随机数可将收敛速率从 O(1/√N) 提升至 O(1/N)。
================================================================================
"""

import numpy as np
from typing import Callable, List, Optional


class MonteCarloEngine:
    """
    通用蒙特卡洛不确定性量化引擎。
    """

    def __init__(self, model_func: Callable[[np.ndarray], float],
                 param_sampler: Callable[[np.ndarray], np.ndarray],
                 n_params: int):
        """
        参数
        ----------
        model_func : callable
            M(β) -> float，给定参数向量 β 返回模型输出（如井口浓度）
        param_sampler : callable
            输入随机数 u ~ U[0,1]^d，返回参数样本 β = sampler(u)
        n_params : int
            参数维度
        """
        self.model_func = model_func
        self.param_sampler = param_sampler
        self.n_params = n_params

    def run_mc(self, N: int, seed: int = 42) -> dict:
        """
        标准蒙特卡洛采样。

        返回字典包含：
            mean, variance, std, min, max, samples
        """
        if N < 1:
            raise ValueError("N 必须 ≥ 1")
        rng = np.random.default_rng(seed)
        outputs = np.zeros(N)
        params = np.zeros((N, self.n_params))

        for i in range(N):
            u = rng.random(self.n_params)
            beta = self.param_sampler(u)
            params[i] = beta
            outputs[i] = self.model_func(beta)

        return self._summarize(outputs, params)

    def run_qmc(self, N: int, seed: int = 42) -> dict:
        """
        拟蒙特卡洛采样（使用 Sobol 序列）。
        """
        if N < 1:
            raise ValueError("N 必须 ≥ 1")
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.n_params, scramble=True, seed=seed)
            u_samples = sampler.random(n=N)
        except Exception:
            # 回退到简单低差异序列
            u_samples = self._simple_latin_hypercube(N, seed)

        outputs = np.zeros(N)
        params = np.zeros((N, self.n_params))
        for i in range(N):
            beta = self.param_sampler(u_samples[i])
            params[i] = beta
            outputs[i] = self.model_func(beta)

        return self._summarize(outputs, params)

    def _simple_latin_hypercube(self, N: int, seed: int) -> np.ndarray:
        """简化的拉丁超立方采样作为回退。"""
        rng = np.random.default_rng(seed)
        samples = np.zeros((N, self.n_params))
        for d in range(self.n_params):
            bins = np.arange(N) / N
            offsets = rng.random(N) / N
            samples[:, d] = bins + offsets
        return samples

    def _summarize(self, outputs: np.ndarray, params: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(outputs)),
            "variance": float(np.var(outputs, ddof=1)),
            "std": float(np.std(outputs, ddof=1)),
            "min": float(np.min(outputs)),
            "max": float(np.max(outputs)),
            "median": float(np.median(outputs)),
            "samples_output": outputs.copy(),
            "samples_params": params.copy(),
        }

    def estimate_exceedance_probability(self, outputs: np.ndarray,
                                        threshold: float) -> dict:
        """
        估计模型输出超过阈值的概率及其置信区间。

        P_exceed = (1/N) Σ I(y_i > threshold)
        95% 置信区间（正态近似）：
            P ± 1.96 * sqrt(P(1-P)/N)
        """
        N = len(outputs)
        if N == 0:
            raise ValueError("输出样本不能为空")
        indicator = (outputs > threshold).astype(float)
        P = float(np.mean(indicator))
        se = np.sqrt(P * (1.0 - P) / max(N, 1))
        ci_lower = max(0.0, P - 1.96 * se)
        ci_upper = min(1.0, P + 1.96 * se)
        return {
            "threshold": threshold,
            "exceedance_probability": P,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "standard_error": float(se),
        }

    def global_sensitivity_sobol(self, outputs_A: np.ndarray,
                                  outputs_B: np.ndarray,
                                  outputs_AB: np.ndarray) -> dict:
        """
        Sobol 一阶敏感性指数估计（Sobol, 1993）。

        使用 Saltelli 采样方案：
          - A, B：两个独立的 N×d 样本矩阵
          - AB_i：将 B 的第 i 列替换为 A 的第 i 列

        一阶 Sobol 指数：
            S_i = Var(E[Y | β_i]) / Var(Y)
                ≈ (1/N) Σ_j f(A)_j (f(AB_i)_j - f(B)_j) / V̂(Y)
        """
        N = len(outputs_A)
        if N == 0:
            raise ValueError("样本不能为空")
        var_Y = np.var(np.concatenate([outputs_A, outputs_B]), ddof=1)
        if var_Y < 1e-15:
            return {"S1": np.array([0.0]), "total_variance": 0.0}

        S1 = np.mean(outputs_A * (outputs_AB - outputs_B)) / var_Y
        return {
            "S1": float(S1),
            "total_variance": float(var_Y),
            "N": N,
        }


def convergence_analysis(model_func: Callable[[np.ndarray], float],
                         param_sampler: Callable[[np.ndarray], np.ndarray],
                         n_params: int,
                         sample_sizes: List[int] = [50, 100, 200, 500, 1000]) -> dict:
    """
    分析蒙特卡洛估计随样本量的收敛行为，验证大数定律。

    返回每个样本量下的均值和标准误。
    """
    results = []
    for N in sample_sizes:
        engine = MonteCarloEngine(model_func, param_sampler, n_params)
        res = engine.run_mc(N, seed=42)
        results.append({
            "N": N,
            "mean": res["mean"],
            "std_err": res["std"] / np.sqrt(N),
        })
    return {"convergence": results}


if __name__ == "__main__":
    # 测试：模型 y = β0 + β1 * x，参数不确定性传播
    def model(beta):
        return beta[0] + beta[1] * 2.0

    def sampler(u):
        # β0 ~ N(1, 0.2), β1 ~ N(0.5, 0.1)
        from math import sqrt
        return np.array([1.0 + sqrt(0.2) * (u[0] - 0.5) * 2 * sqrt(3),
                         0.5 + sqrt(0.1) * (u[1] - 0.5) * 2 * sqrt(3)])

    engine = MonteCarloEngine(model, sampler, n_params=2)
    res = engine.run_mc(1000)
    assert res["mean"] > 0

    exceed = engine.estimate_exceedance_probability(res["samples_output"], 2.0)
    assert 0.0 <= exceed["exceedance_probability"] <= 1.0
    print("uncertainty_engine: 自测试通过")
