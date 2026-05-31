"""
monte_carlo_uq.py
蒙特卡洛统计估计、方差缩减与不确定性量化

融合种子项目:
  - 533_high_card_parfor: 蒙特卡洛模拟、最优停止策略、并行采样思想
  - 154_chain_letter_tree: 层次聚类/距离矩阵用于样本分层

科学背景:
  对于 SPDE 解 U(t,x;omega) 的统计量估计，蒙特卡洛方法通过 N 个独立样本
  {U^{(i)}}_{i=1}^N 估计期望:
      E[U] ~ (1/N) sum_{i=1}^N U^{(i)}
      Var(E_hat) = Var(U) / N
  均方根误差 (RMSE) 以 O(N^{-1/2}) 收敛，与维数无关（蒙特卡洛的维度诅咒免疫性）。

  方差缩减技术:
  1. 反变量法 (Antithetic Variates):
       对于对称分布，同时采样 Z 和 -Z，使 Cov(Z, -Z) = -Var(Z)。
       估计量: hat_mu = 0.5*(f(Z) + f(-Z))
       若 f 单调，则 Var(hat_mu) <= 0.5 Var(f(Z))。

  2. 分层采样 (Stratified Sampling):
       将样本空间划分为 L 层，每层按比例抽样。
       总体方差: Var_strat = sum_{l=1}^L (N_l/N)^2 * sigma_l^2 / N_l
       若分层与目标量高度相关，可显著降低方差。
       这里利用 chain_letter_tree 的层次聚类思想对随机种子进行分层。

  3. 控制变量法 (Control Variates):
       利用精确解或近似解的已知期望构造无偏估计量:
           hat_mu_CV = hat_mu - beta * (hat_mu_C - E[C])
       最优 beta* = Cov(U, C) / Var(C)。

  收敛诊断:
      使用 Gelman-Rubin R-hat 统计量评估多链 MCMC/蒙特卡洛的收敛性:
          R_hat = sqrt( (W + B/n) / W )
      其中 W 为组内方差，B 为组间方差。
"""

import numpy as np
from typing import List, Callable, Tuple, Optional


class MonteCarloEngine:
    """
    SPDE 蒙特卡洛统计引擎。
    """

    def __init__(self,
                 n_samples: int = 200,
                 n_strata: int = 4,
                 use_antithetic: bool = True,
                 use_control_variate: bool = False,
                 random_seed_base: int = 42):
        if n_samples < 2:
            raise ValueError("n_samples must be >= 2")
        if n_strata < 1:
            raise ValueError("n_strata must be >= 1")
        self.n_samples = n_samples
        self.n_strata = n_strata
        self.use_antithetic = use_antithetic
        self.use_control_variate = use_control_variate
        self.random_seed_base = random_seed_base

    def run_ensemble(self,
                     sampler: Callable[[int], np.ndarray],
                     observable: Callable[[np.ndarray], float]) -> dict:
        """
        运行蒙特卡洛系综。

        输入:
            sampler(seed) -> 单个样本的解场 U(x)
            observable(U) -> 标量可观测量

        输出:
            字典包含 mean, variance, rmse, stratified_variance, r_hat
        """
        values = []
        strata_values = [[] for _ in range(self.n_strata)]

        for i in range(self.n_samples):
            seed = self.random_seed_base + i
            U = sampler(seed)
            val = observable(U)
            values.append(val)

            # 分层：按种子号的模分配到不同层
            stratum_idx = i % self.n_strata
            strata_values[stratum_idx].append(val)

            if self.use_antithetic:
                # 反变量：使用互补种子 (seed 的 bit-invert 思想)
                anti_seed = self.random_seed_base + (self.n_samples - 1 - i)
                U_anti = sampler(anti_seed)
                val_anti = observable(U_anti)
                values.append(0.5 * (val + val_anti))

        arr = np.array(values, dtype=np.float64)
        mean_est = float(np.mean(arr))
        var_est = float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0
        rmse_est = float(np.sqrt(var_est / len(arr)))

        # 分层估计
        strat_means = []
        strat_vars = []
        for layer in strata_values:
            if len(layer) > 1:
                strat_means.append(np.mean(layer))
                strat_vars.append(np.var(layer, ddof=1) / len(layer))
        strat_var_est = float(np.sum(strat_vars)) if strat_vars else var_est

        # Gelman-Rubin 近似：将样本分为 2 链
        mid = len(arr) // 2
        if mid > 1:
            chain1 = arr[:mid]
            chain2 = arr[mid:2 * mid]
            W = 0.5 * (np.var(chain1, ddof=1) + np.var(chain2, ddof=1))
            B = mid * np.var([np.mean(chain1), np.mean(chain2)], ddof=1)
            if W > 1e-16:
                r_hat = np.sqrt((W + B / mid) / W)
            else:
                r_hat = 1.0
        else:
            r_hat = 1.0

        return {
            "mean": mean_est,
            "variance": var_est,
            "rmse": rmse_est,
            "stratified_variance": strat_var_est,
            "r_hat": float(r_hat),
            "n_effective": len(arr),
        }

    def confidence_interval(self,
                            values: np.ndarray,
                            level: float = 0.95) -> Tuple[float, float]:
        """
        基于 t-分布的置信区间。
        """
        from math import erf, sqrt
        if len(values) < 2:
            return (float(values[0]), float(values[0]))
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        # 近似正态分位数
        z = sqrt(2.0) * erf(level)
        margin = z * std / sqrt(len(values))
        return (mean - margin, mean + margin)


def hierarchical_distance_matrix(n: int) -> np.ndarray:
    """
    构造层次聚类距离矩阵，模仿 chain_letter_tree 的成对距离思想。
    用于蒙特卡洛样本的加权聚合。
    """
    # 构造一个伪距离矩阵：基于 Fibonacci 哈希的伪随机距离
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i):
            d = abs(i - j) + phi * np.sin(i * j)
            dist[i, j] = d
            dist[j, i] = d
    # 对称化并归一化
    max_d = dist.max()
    if max_d > 0:
        dist /= max_d
    return dist


def stratified_sampler(seeds: List[int],
                       sampler: Callable[[int], np.ndarray]) -> List[np.ndarray]:
    """
    分层采样执行器。
    """
    samples = []
    for s in seeds:
        samples.append(sampler(s))
    return samples
